package common

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

type interruptedUpload struct{ reads int }

func (r *interruptedUpload) Read(p []byte) (int, error) {
	r.reads++
	if r.reads > 1 {
		return 0, errors.New("must not read upload again")
	}
	return copy(p, []byte(`{"model":`)), io.ErrUnexpectedEOF
}
func (r *interruptedUpload) Close() error { return nil }

func TestInterruptedUploadRetainsOriginalFailure(t *testing.T) {
	r := &interruptedUpload{}
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	c.Request.Body = r
	c.Request.ContentLength = 1000
	for i := 0; i < 2; i++ {
		_, err := ReadBodyRaw(c)
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("upload error replaced: %v", err)
		}
	}
	if r.reads != 1 || c.GetInt64("request_body_bytes") != 9 || c.GetString("request_failure_stage") != "ingress_read" {
		t.Fatalf("incorrect bounded upload diagnostics, reads=%d bytes=%d", r.reads, c.GetInt64("request_body_bytes"))
	}
}

func TestJSONBodyFailureStageAndReadCache(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":`))
	c.Request.Header.Set("Content-Type", "application/json")
	var output map[string]interface{}
	if err := UnmarshalBodyReusable(c, &output); err == nil || c.GetString("request_failure_stage") != "ingress_json" {
		t.Fatalf("malformed JSON stage unavailable: %v", err)
	}
	data, err := ReadBodyRaw(c)
	if err != nil || string(data) != `{"model":` {
		t.Fatalf("read cache corrupted: %q %v", data, err)
	}
}
