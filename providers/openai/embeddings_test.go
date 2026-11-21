package openai

import (
	"bytes"
	"done-hub/common/config"
	"done-hub/common/logger"
	"done-hub/common/requester"
	"done-hub/model"
	"done-hub/providers/base"
	"done-hub/types"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
	"gorm.io/datatypes"
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

func embeddingBatchTestProvider(t *testing.T, serverURL string, usage *types.Usage) *OpenAIProvider {
	t.Helper()
	plugin := datatypes.NewJSONType(model.PluginType{"customize": {"3": "/api/v3/embeddings/multimodal"}})
	channel := &model.Channel{
		Id:      1,
		Type:    config.ChannelTypeCustom,
		BaseURL: &serverURL,
		Key:     "synthetic",
		Plugin:  &plugin,
	}
	provider := &OpenAIProvider{BaseProvider: base.BaseProvider{
		Config:    base.ProviderConfig{Embeddings: "/api/v3/embeddings/multimodal"},
		Channel:   channel,
		Requester: requester.NewHTTPRequester("", RequestErrorHandle),
	}}
	provider.SetUsage(usage)
	return provider
}

func embeddingBatchTestContext() (*gin.Context, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte(`{}`)))
	return c, w
}

func TestCreateEmbeddingsBatchStandardInput(t *testing.T) {
	var requests int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requests, 1)
		var body struct {
			Model      string `json:"model"`
			Dimensions int    `json:"dimensions"`
			Input      []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "doubao-embedding-vision-251215" || body.Dimensions != 1024 || len(body.Input) != 1 || body.Input[0].Type != "text" {
			t.Errorf("unexpected upstream request: %+v", body)
		}
		text := body.Input[0].Text
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  body.Model,
			"data": map[string]any{
				"object":    "embedding",
				"embedding": []int{len(text)},
			},
			"usage": map[string]any{
				"prompt_tokens": len(text),
				"prompt_tokens_details": map[string]int{
					"text_tokens": len(text),
				},
				"total_tokens": len(text),
			},
		})
	}))
	defer server.Close()
	requester.HTTPClient = server.Client()
	usage := &types.Usage{}
	provider := embeddingBatchTestProvider(t, server.URL, usage)
	c, _ := embeddingBatchTestContext()
	provider.SetContext(c)

	response, err := provider.CreateEmbeddings(&types.EmbeddingRequest{
		Model:          "doubao-embedding-vision-251215",
		Input:          []any{"a", "bb", "ccc"},
		Dimensions:     1024,
		EncodingFormat: "float",
	})
	if err != nil {
		t.Fatalf("batch failed: %+v", err)
	}
	if len(response.Data) != 3 || response.Object != "list" || response.Model != "doubao-embedding-vision-251215" {
		t.Fatalf("unexpected batch response: %+v", response)
	}
	for i, expected := range []int{1, 2, 3} {
		raw, ok := response.Data[i].Embedding.(json.RawMessage)
		if !ok {
			t.Fatalf("embedding %d type %T", i, response.Data[i].Embedding)
		}
		var values []int
		if err := json.Unmarshal(raw, &values); err != nil || len(values) != 1 || values[0] != expected || response.Data[i].Index != i {
			t.Fatalf("order mismatch at %d: index=%d values=%v err=%v", i, response.Data[i].Index, values, err)
		}
	}
	if usage.PromptTokens != 6 || usage.TotalTokens != 6 || usage.PromptTokensDetails.TextTokens != 6 {
		t.Fatalf("usage was not aggregated: %+v", usage)
	}
	if atomic.LoadInt32(&requests) != 3 {
		t.Fatalf("expected 3 upstream calls, got %d", requests)
	}
	if _, raw := c.Get(config.GinRawResponseBodyKey); raw {
		t.Fatal("standard batch must not use native raw passthrough")
	}
}

func TestCreateEmbeddingsBatchRejectsOverLimit(t *testing.T) {
	var requests int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requests, 1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()
	requester.HTTPClient = server.Client()
	provider := embeddingBatchTestProvider(t, server.URL, &types.Usage{})
	c, _ := embeddingBatchTestContext()
	provider.SetContext(c)
	items := make([]any, embeddingBatchLimit+1)
	for i := range items {
		items[i] = "x"
	}
	_, err := provider.CreateEmbeddings(&types.EmbeddingRequest{Model: "doubao-embedding-vision-251215", Input: items, Dimensions: 1024})
	if err == nil || err.StatusCode != http.StatusBadRequest || atomic.LoadInt32(&requests) != 0 {
		t.Fatalf("over-limit batch not rejected before upstream: err=%+v requests=%d", err, requests)
	}
}

func TestCreateEmbeddingsBatchItemErrorStopsWholeResponse(t *testing.T) {
	logger.Logger = zap.NewNop()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []struct {
				Text string `json:"text"`
			} `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if len(body.Input) == 1 && body.Input[0].Text == "bad" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"synthetic rejection","type":"invalid_request_error","code":"bad_input"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"object":"list","data":{"object":"embedding","embedding":[0.1]},"usage":{"prompt_tokens":1,"total_tokens":1}}`))
	}))
	defer server.Close()
	requester.HTTPClient = server.Client()
	provider := embeddingBatchTestProvider(t, server.URL, &types.Usage{})
	c, _ := embeddingBatchTestContext()
	provider.SetContext(c)
	_, err := provider.CreateEmbeddings(&types.EmbeddingRequest{Model: "doubao-embedding-vision-251215", Input: []any{"ok", "bad"}, Dimensions: 1024})
	if err == nil || err.StatusCode != http.StatusBadRequest || !bytes.Contains([]byte(err.Message), []byte("batch item")) {
		t.Fatalf("single-item error did not fail batch cleanly: %+v", err)
	}
}
