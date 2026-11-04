package relay_util

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestHeartbeatFlushAndWriterHandoff(t *testing.T) {
	for _, stream := range []bool{true, false} {
		t.Run(map[bool]string{true: "sse", false: "json"}[stream], func(t *testing.T) {
			handed := make(chan struct{})
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				c, _ := gin.CreateTestContext(w)
				c.Request = r
				h := NewHeartbeat(stream, HeartbeatConfig{TimeoutSeconds: 0, IntervalSeconds: 5}, c)
				h.Start()
				<-release
				h.Stop()
				if !h.HasWrittenHeader() {
					t.Error("heartbeat never reached the wire")
				}
				// After Stop returns, no worker can touch the writer/context again.
				for i := 0; i < 10; i++ {
					_, _ = io.WriteString(c.Writer, "final")
				}
				h.Close()
				close(handed)
			}))
			defer server.Close()
			client := &http.Client{Timeout: 2 * time.Second}
			response, err := client.Get(server.URL)
			if err != nil {
				close(release)
				t.Fatal(err)
			}
			reader := bufio.NewReader(response.Body)
			first, err := reader.ReadString('\n')
			if err != nil {
				t.Fatal(err)
			}
			if stream && (!strings.HasPrefix(first, ":") || !strings.Contains(response.Header.Get("Content-Type"), "event-stream")) {
				t.Fatalf("not a legal flushed SSE comment: %q", first)
			}
			close(release)
			body, err := io.ReadAll(reader)
			_ = response.Body.Close()
			if err != nil || strings.Count(string(body), "final") != 10 {
				t.Fatalf("handoff corrupted output: %q %v", body, err)
			}
			<-handed
		})
	}
}

func TestHeartbeatLifecycle(t *testing.T) {
	for i := 0; i < 40; i++ {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		ctx, cancel := context.WithCancel(context.Background())
		c.Request = httptest.NewRequest(http.MethodGet, "/", nil).WithContext(ctx)
		h := NewHeartbeat(true, HeartbeatConfig{TimeoutSeconds: 30}, c)
		var wg sync.WaitGroup
		wg.Add(3)
		go func() { defer wg.Done(); h.Start(); h.Start() }()
		go func() { defer wg.Done(); h.Stop() }()
		go func() { defer wg.Done(); cancel(); h.Close() }()
		wg.Wait()
		h.Stop()
		if w.Body.Len() != 0 {
			t.Fatal("heartbeat wrote after cancellation/before configured delay")
		}
	}
}
