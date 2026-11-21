package openai

import (
	"bytes"
	"done-hub/common/config"
	"done-hub/common/requester"
	"done-hub/model"
	"done-hub/providers/base"
	"done-hub/types"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestDecodeEmbeddingData(t *testing.T) {
	t.Run("standard array", func(t *testing.T) {
		data, native, err := decodeEmbeddingData(json.RawMessage(`[{"object":"embedding","embedding":[0.1,0.2],"index":0}]`))
		if err != nil || native || len(data) != 1 || data[0].Object != "embedding" {
			t.Fatalf("unexpected array result: data=%+v native=%v err=%v", data, native, err)
		}
	})
	t.Run("native object", func(t *testing.T) {
		data, native, err := decodeEmbeddingData(json.RawMessage(`{"object":"embedding","embedding":[0.1],"multi_embedding":[[0.2]],"sparse_embedding":[{"index":1,"value":0.3}]}`))
		if err != nil || !native || data != nil {
			t.Fatalf("unexpected object result: data=%+v native=%v err=%v", data, native, err)
		}
	})
	t.Run("object missing dense embedding", func(t *testing.T) {
		if _, _, err := decodeEmbeddingData(json.RawMessage(`{"multi_embedding":[[0.2]]}`)); err == nil {
			t.Fatal("expected missing native embedding error")
		}
	})
}

func TestCreateEmbeddingsNativeObjectPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := []byte(`{"object":"list","model":"doubao-embedding-vision-251215","data":{"object":"embedding","embedding":[0.1,0.2],"multi_embedding":[[0.3]],"sparse_embedding":[{"index":7,"value":0.4}]},"usage":{"prompt_tokens":12,"prompt_tokens_details":{"text_tokens":12,"image_tokens":0},"total_tokens":12}}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v3/embeddings/multimodal" {
			t.Errorf("unexpected upstream path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(upstream)
	}))
	defer server.Close()
	requester.HTTPClient = server.Client()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte(`{}`)))

	channel := &model.Channel{
		Id:      1,
		Type:    config.ChannelTypeCustom,
		BaseURL: &server.URL,
		Key:     "synthetic",
	}
	provider := &OpenAIProvider{
		BaseProvider: base.BaseProvider{
			Config:    base.ProviderConfig{Embeddings: "/api/v3/embeddings/multimodal"},
			Channel:   channel,
			Requester: requester.NewHTTPRequester("", RequestErrorHandle),
		},
	}
	provider.SetUsage(&types.Usage{})
	provider.SetContext(c)

	response, err := provider.CreateEmbeddings(&types.EmbeddingRequest{
		Model: "doubao-embedding-vision-251215",
		Input: []any{map[string]any{"type": "text", "text": "synthetic"}},
	})
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %+v", err)
	}
	if response == nil || response.Usage == nil || response.Usage.PromptTokens != 12 || provider.Usage.PromptTokens != 12 {
		t.Fatalf("usage was not parsed: response=%+v providerUsage=%+v", response, provider.Usage)
	}
	raw, ok := c.Get(config.GinRawResponseBodyKey)
	if !ok {
		t.Fatal("native object response was not staged for passthrough")
	}
	rawBytes, ok := raw.([]byte)
	if !ok || !bytes.Equal(rawBytes, upstream) {
		t.Fatalf("native response bytes changed: %s", rawBytes)
	}
	if !bytes.Contains(rawBytes, []byte(`"multi_embedding"`)) || !bytes.Contains(rawBytes, []byte(`"sparse_embedding"`)) {
		t.Fatal("native vector fields were dropped")
	}
}

func TestCreateEmbeddingsStandardArrayDoesNotStageRaw(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := []byte(`{"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","embedding":[0.1],"index":0}],"usage":{"prompt_tokens":3,"total_tokens":3}}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(upstream)
	}))
	defer server.Close()
	requester.HTTPClient = server.Client()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte(`{}`)))
	channel := &model.Channel{Id: 1, Type: config.ChannelTypeCustom, BaseURL: &server.URL, Key: "synthetic"}
	provider := &OpenAIProvider{BaseProvider: base.BaseProvider{
		Config:    base.ProviderConfig{Embeddings: "/v1/embeddings"},
		Channel:   channel,
		Requester: requester.NewHTTPRequester("", RequestErrorHandle),
	}}
	provider.SetUsage(&types.Usage{})
	provider.SetContext(c)

	response, err := provider.CreateEmbeddings(&types.EmbeddingRequest{Model: "text-embedding-3-small", Input: "synthetic"})
	if err != nil {
		t.Fatalf("CreateEmbeddings returned error: %+v", err)
	}
	if response == nil || len(response.Data) != 1 || response.Data[0].Index != 0 {
		t.Fatalf("standard array response was not parsed: %+v", response)
	}
	if _, ok := c.Get(config.GinRawResponseBodyKey); ok {
		t.Fatal("standard array must keep the existing struct response path")
	}
}
