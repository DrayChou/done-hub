package relay

import (
	"bufio"
	"context"
	"done-hub/common/logger"
	"done-hub/common/requester"
	"done-hub/relay/relay_util"
	"done-hub/types"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// Exercise the real header-wait -> stream-reader -> downstream HTTP chain.
// Millisecond delays are deliberate: the production Cloudflare crossing is a
// separate deployment gate, not a repeated two-minute unit test.
func TestSlowUpstreamHeartbeatHandoffAndFailure(t *testing.T) {
	logger.Logger = zap.NewNop()
	previous := requester.HTTPClient
	requester.HTTPClient = &http.Client{Timeout: 3 * time.Second}
	defer func() { requester.HTTPClient = previous }()
	for _, failHeaders := range []bool{false, true} {
		t.Run(map[bool]string{false: "slow-first-data-and-gap", true: "failure-after-heartbeat"}[failHeaders], func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				time.Sleep(30 * time.Millisecond)
				if failHeaders {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = io.WriteString(w, `{"error":{"message":"synthetic failure"}}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				time.Sleep(35 * time.Millisecond)
				_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"PONG\"}}]}\n\n")
				w.(http.Flusher).Flush()
				time.Sleep(35 * time.Millisecond)
				_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			}))
			defer upstream.Close()
			completed := make(chan *types.OpenAIErrorWithStatusCode, 1)
			downstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				c, _ := gin.CreateTestContext(w)
				c.Request = req
				c.Set("requestStartTime", time.Now())
				c.Set(logger.RequestIdKey, "synthetic-slow-chain")
				c.Set("stream_heartbeat_interval", 10*time.Millisecond)
				h := relay_util.NewHeartbeat(true, relay_util.HeartbeatConfig{TimeoutSeconds: 0}, c)
				h.Start()
				defer h.Stop()
				r := requester.NewHTTPRequester("", nil)
				r.HeaderWaitContext = req.Context()
				u, _ := http.NewRequest(http.MethodPost, upstream.URL, strings.NewReader("{}"))
				resp, apiErr := r.SendRequestRaw(u)
				h.Stop() // joins before any error/data/header writes
				if apiErr != nil {
					_, _ = io.WriteString(c.Writer, formatStreamError(c, apiErr, true))
					c.Writer.Flush()
					completed <- apiErr
					return
				}
				stream, _ := requester.RequestStream(r, resp, func(raw *[]byte, data chan string, errs chan error) {
					if strings.HasPrefix(string(*raw), "data:") {
						data <- strings.TrimSpace(string((*raw)[5:]))
					}
				})
				_, apiErr = responseStreamClient(c, stream, nil)
				completed <- apiErr
			}))
			defer downstream.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, downstream.URL, nil)
			res, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			reader := bufio.NewReader(res.Body)
			first, err := reader.ReadString('\n')
			if err != nil || !strings.HasPrefix(first, ":") {
				t.Fatalf("no pre-header heartbeat: %q %v", first, err)
			}
			raw, err := io.ReadAll(reader)
			_ = res.Body.Close()
			if err != nil {
				t.Fatal(err)
			}
			apiErr := <-completed
			if failHeaders {
				if apiErr == nil || !strings.Contains(string(raw), "error") || strings.Contains(string(raw), "[DONE]") {
					t.Fatalf("header failure disguised as success: %s", raw)
				}
			} else if apiErr != nil || !strings.Contains(string(raw), "PONG") || strings.Count(string(raw), ": ping") < 2 || strings.Count(string(raw), "[DONE]") != 1 {
				t.Fatalf("heartbeat handoff/gap/completion failed: %s %v", raw, apiErr)
			}
		})
	}
}
