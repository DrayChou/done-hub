package requester

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHeaderWaitCancellationAndPostHeaderSettlement(t *testing.T) {
	original := HTTPClient
	HTTPClient = &http.Client{Timeout: 2 * time.Second}
	defer func() { HTTPClient = original }()
	for _, stage := range []string{"before", "after"} {
		t.Run(stage, func(t *testing.T) {
			seen := make(chan struct{})
			cancelled := make(chan struct{})
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				close(seen)
				if stage == "before" {
					select {
					case <-r.Context().Done():
						close(cancelled)
					case <-time.After(3 * time.Second):
					}
					return
				}
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				<-release
				_, _ = io.WriteString(w, "settled-output")
			}))
			defer server.Close()
			clientCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := NewHTTPRequester("", nil)
			r.HeaderWaitContext = clientCtx
			req, _ := http.NewRequest(http.MethodPost, server.URL, strings.NewReader("{}"))
			if stage == "before" {
				done := make(chan error, 1)
				go func() { _, err := r.doUntilHeaders(req); done <- err }()
				<-seen
				cancel()
				select {
				case err := <-done:
					apiErr := requestTransportError(err)
					if apiErr.Code != "client_disconnected" || !apiErr.LocalError {
						t.Fatalf("wrong disconnect diagnostic: %+v", apiErr)
					}
				case <-time.After(time.Second):
					t.Fatal("disconnected client left upstream header wait alive")
				}
				select {
				case <-cancelled:
				case <-time.After(time.Second):
					t.Fatal("upstream request was not cancelled")
				}
				return
			}
			resp, err := r.doUntilHeaders(req)
			if err != nil {
				t.Fatal(err)
			}
			cancel()
			close(release)
			data, err := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if err != nil || string(data) != "settled-output" {
				t.Fatalf("post-header usage/output drain regressed: %s %v", data, err)
			}
		})
	}
}

func TestRequestTimeoutUsesExistingBudgets(t *testing.T) {
	originalClient, originalRequestTimeout := HTTPClient, relayRequestTimeout
	defer func() { HTTPClient, relayRequestTimeout = originalClient, originalRequestTimeout }()
	for _, tt := range []struct {
		total, request time.Duration
		stream         bool
		want           time.Duration
	}{
		{600 * time.Second, 300 * time.Second, true, 600 * time.Second},
		{600 * time.Second, 300 * time.Second, false, 300 * time.Second},
		{5 * time.Second, 300 * time.Second, false, 5 * time.Second},
		{0, 300 * time.Second, false, 300 * time.Second},
		{0, 300 * time.Second, true, 0},
	} {
		HTTPClient, relayRequestTimeout = &http.Client{Timeout: tt.total}, tt.request
		if got := RequestTimeout(tt.stream); got != tt.want {
			t.Fatalf("total=%s request=%s stream=%t got=%s want=%s", tt.total, tt.request, tt.stream, got, tt.want)
		}
	}
}

func TestStreamFirstDataWaitIsBounded(t *testing.T) {
	original := streamIdleTimeout
	streamIdleTimeout = 30 * time.Millisecond
	defer func() { streamIdleTimeout = original }()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer server.Close()
	response, err := http.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	r := NewHTTPRequester("", nil)
	stream, _ := RequestStream(r, response, func(raw *[]byte, data chan string, errs chan error) {})
	_, errs := stream.Recv()
	defer stream.Close()
	select {
	case err := <-errs:
		if err == nil || !strings.Contains(err.Error(), "idle timeout") {
			t.Fatalf("missing first-data timeout: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("response headers without data left stream unbounded")
	}
}

func TestHeaderTimeoutKeepsPreciseError(t *testing.T) {
	original := HTTPClient
	transport := &http.Transport{ResponseHeaderTimeout: 30 * time.Millisecond}
	HTTPClient = &http.Client{Transport: transport}
	defer func() { HTTPClient = original; transport.CloseIdleConnections() }()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
		}
	}))
	defer server.Close()
	r := NewHTTPRequester("", nil)
	req, _ := http.NewRequest(http.MethodPost, server.URL, nil)
	_, apiErr := r.SendRequestRaw(req)
	if apiErr == nil || apiErr.Code != "upstream_timeout" || apiErr.StatusCode != http.StatusGatewayTimeout || !apiErr.LocalError {
		t.Fatalf("upstream timeout lost: %+v", apiErr)
	}
}
