package requester

import (
	"context"
	"done-hub/common"
	"done-hub/types"
	"errors"
	"io"
	"net"
	"net/http"
	"sync"
)

// Tie only the pre-header phase to the downstream client. The returned body's
// owner releases the derived context on Close, including outputResp/raw streams.
func (r *HTTPRequester) doUntilHeaders(req *http.Request) (*http.Response, error) {
	if r.HeaderWaitContext == nil {
		return HTTPClient.Do(req)
	}
	ctx, cancel := context.WithCancel(req.Context())
	var phase sync.Mutex
	headersReceived := false
	stop := context.AfterFunc(r.HeaderWaitContext, func() {
		phase.Lock()
		defer phase.Unlock()
		if !headersReceived {
			cancel()
		}
	})
	if r.HeaderWaitContext.Err() != nil {
		cancel()
	}
	resp, err := HTTPClient.Do(req.WithContext(ctx))
	phase.Lock()
	headersReceived = err == nil
	phase.Unlock()
	stop() // no cancellation is forwarded after the response headers arrive
	if err != nil {
		cancel()
		return nil, err
	}
	resp.Body = &cancelOnCloseBody{ReadCloser: resp.Body, cancel: cancel}
	return resp, nil
}

type cancelOnCloseBody struct {
	io.ReadCloser
	cancel context.CancelFunc
	once   sync.Once
}

func (b *cancelOnCloseBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.cancel)
	return err
}

func requestTransportError(err error) *types.OpenAIErrorWithStatusCode {
	if errors.Is(err, context.Canceled) {
		return common.StringErrorWrapperLocal("client disconnected before upstream response headers", "client_disconnected", 499)
	}
	var timeout net.Error
	if errors.Is(err, context.DeadlineExceeded) || errors.As(err, &timeout) && timeout.Timeout() {
		return common.StringErrorWrapperLocal("upstream timed out waiting for a response", "upstream_timeout", http.StatusGatewayTimeout)
	}
	return common.ErrorWrapper(err, "http_request_failed", http.StatusBadGateway)
}
