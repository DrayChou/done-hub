package relay

import (
	"bufio"
	"context"
	"done-hub/common/config"
	"done-hub/common/logger"
	"done-hub/common/requester"
	"done-hub/types"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type syntheticStream struct {
	data chan string
	errs chan error
}

func (s *syntheticStream) Recv() (<-chan string, <-chan error) { return s.data, s.errs }
func (s *syntheticStream) Close()                              {}

func TestStreamIdleHeartbeatAndTerminalErrors(t *testing.T) {
	logger.Logger = zap.NewNop()
	for _, protocol := range []string{"chat", "responses", "claude", "gemini"} {
		for _, failed := range []bool{false, true} {
			t.Run(protocol+map[bool]string{false: "/success", true: "/failed"}[failed], func(t *testing.T) {
				path := map[string]string{"chat": "/v1/chat/completions", "responses": "/v1/responses", "claude": "/claude/v1/messages", "gemini": "/gemini/v1beta/models/test:streamGenerateContent"}[protocol]
				done := make(chan *types.OpenAIErrorWithStatusCode, 1)
				release := make(chan struct{})
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					c, _ := gin.CreateTestContext(w)
					c.Request = r
					c.Set("requestStartTime", time.Now())
					c.Set(logger.RequestIdKey, "synthetic-stream-test")
					c.Set("stream_heartbeat_interval", 10*time.Millisecond)
					s := &syntheticStream{make(chan string), make(chan error)}
					go func() {
						<-release
						if failed {
							s.errs <- &types.OpenAIError{Code: "synthetic_upstream_error", Type: "upstream_error", Message: "synthetic failure"}
						} else {
							if protocol == "chat" {
								s.data <- `{"choices":[{"delta":{"content":"PONG"},"finish_reason":"stop"}]}`
							} else {
								s.data <- "data: {\"result\":\"PONG\"}\n\n"
							}
							s.errs <- io.EOF
						}
						close(s.data)
						close(s.errs)
					}()
					_, err := relayResponseStream(c, s, nil, protocol == "chat")
					done <- err
				}))
				defer srv.Close()
				client := &http.Client{Timeout: time.Second}
				res, err := client.Get(srv.URL + path)
				if err != nil {
					close(release)
					t.Fatal(err)
				}
				reader := bufio.NewReader(res.Body)
				first, err := reader.ReadString('\n')
				if err != nil || !strings.HasPrefix(first, ":") {
					t.Fatalf("no heartbeat while waiting for first data: %q %v", first, err)
				}
				close(release)
				raw, err := io.ReadAll(reader)
				_ = res.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				finalErr := <-done
				if failed {
					if finalErr == nil || strings.Contains(string(raw), "[DONE]") || !strings.Contains(string(raw), "error") {
						t.Fatalf("failure disguised as success: %s (%v)", raw, finalErr)
					}
					for _, l := range strings.Split(string(raw), "\n") {
						if strings.HasPrefix(l, "data:") && !json.Valid([]byte(strings.TrimSpace(l[5:]))) {
							t.Fatalf("non-JSON stream error: %s", l)
						}
					}
				} else if finalErr != nil || !strings.Contains(string(raw), "PONG") || protocol == "chat" && strings.Count(string(raw), "[DONE]") != 1 {
					t.Fatalf("normal completion regressed: %s %v", raw, finalErr)
				}
			})
		}
	}
}

func TestStreamClosedChannelsAndDisconnect(t *testing.T) {
	logger.Logger = zap.NewNop()
	for _, cancelClient := range []bool{false, true} {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		ctx, cancel := context.WithCancel(context.Background())
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
		s := &syntheticStream{make(chan string), make(chan error)}
		close(s.data)
		close(s.errs)
		if cancelClient {
			cancel()
		}
		_, err := responseStreamClient(c, s, nil)
		cancel()
		if cancelClient && (err == nil || err.Code != "client_disconnected" || strings.Contains(w.Body.String(), "[DONE]")) {
			t.Fatalf("closed/cancelled stream reported success: %s %v", w.Body.String(), err)
		}
	}
}

func TestTimeoutErrorIsNotReclassifiedAsSaturation(t *testing.T) {
	for _, local := range []bool{false, true} {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		c.Set(logger.RequestIdKey, "synthetic-timeout-request")
		apiErr := &types.OpenAIErrorWithStatusCode{LocalError: local, StatusCode: http.StatusGatewayTimeout,
			OpenAIError: types.OpenAIError{Code: "upstream_timeout", Type: "one_hub_error", Message: "upstream timed out waiting for a response"}}
		if local && shouldRetry(c, apiErr, config.ChannelTypeGemini) {
			t.Fatal("locally timed out request silently retried")
		}
		if !local {
			apiErr.OpenAIError = types.OpenAIError{Code: "bad_response_status_code", Type: "upstream_error", Message: "private-backend.invalid/private?api_key=credential-never-expose"}
		}
		safe := FilterOpenAIErr(c, apiErr)
		formatted := formatStreamError(c, apiErr, true)
		if safe.StatusCode != http.StatusGatewayTimeout || !strings.Contains(formatted, "upstream_timeout") || strings.Contains(formatted, "service_unavailable") || !strings.Contains(formatted, "synthetic-timeout-request") {
			t.Fatalf("lost specific timeout: %+v %s", safe, formatted)
		}
		if !local && (strings.Contains(formatted, "private-backend") || strings.Contains(formatted, "credential-never-expose") || strings.Contains(formatted, "private?")) {
			t.Fatalf("raw upstream timeout detail leaked: %s", formatted)
		}
	}
}

func TestGeminiSignatureRepairDoesNotUseGenericChannelRetry(t *testing.T) {
	for _, repaired := range []bool{false, true} {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		c.Set("thought_signature_retried", repaired)
		apiErr := &types.OpenAIErrorWithStatusCode{StatusCode: http.StatusBadRequest,
			OpenAIError: types.OpenAIError{Message: "Invalid thought signature."}}
		if shouldRetry(c, apiErr, config.ChannelTypeGemini) {
			t.Fatal("Gemini signature failure escaped its provider's opt-in, bounded repair")
		}
	}
}

// A real chunked response can be truncated after headers. Its transport failure
// must reach the stream error path; clean EOF and closed channels are not equivalent.
func TestTruncatedUpstreamStreamProducesError(t *testing.T) {
	logger.Logger = zap.NewNop()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("no hijacker")
		}
		conn, writer, _ := h.Hijack()
		_, _ = io.WriteString(writer, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\ndata:")
		_ = writer.Flush()
		_ = conn.Close()
	}))
	defer upstream.Close()
	response, err := http.Get(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	r := requester.NewHTTPRequester("", nil)
	stream, apiErr := requester.RequestStream(r, response, func(raw *[]byte, data chan string, errs chan error) {})
	if apiErr != nil {
		t.Fatal(apiErr)
	}
	data, errs := stream.Recv()
	defer stream.Close()
	for data != nil || errs != nil {
		select {
		case _, ok := <-data:
			if !ok {
				data = nil
			}
		case e, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			if e != nil && !errors.Is(e, io.EOF) {
				return
			}
		case <-time.After(time.Second):
			t.Fatal("truncated upstream stream hung")
		}
	}
	t.Fatal("truncation was treated as successful EOF")
}
