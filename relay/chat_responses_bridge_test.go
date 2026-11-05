package relay

import (
	"done-hub/common/config"
	"done-hub/common/logger"
	"done-hub/common/requester"
	"done-hub/model"
	"done-hub/providers/gemini"
	"done-hub/providers/openai"
	"done-hub/types"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
	"gorm.io/datatypes"
)

func TestChatResponsesBridgeSelection(t *testing.T) {
	for _, tt := range []struct {
		models           any
		original, mapped string
		want, invalid    bool
	}{
		{"", "gpt-6.1-sol", "mapped-gpt", false, false},
		{"gpt-6.1-sol, gpt-6-astra", "gpt-6.1-sol", "mapped-gpt", true, false},
		{"mapped-gpt", "public-gpt", "mapped-gpt", true, false},
		{"*", "unknown-new-model", "upstream-alias", true, false},
		{"gpt-6.1-sol", "other-model", "other-alias", false, false},
		{[]string{"gpt-6.1-sol"}, "gpt-6.1-sol", "gpt-6.1-sol", false, true},
	} {
		plugin := datatypes.NewJSONType(model.PluginType{"chat_to_responses": {"models": tt.models}})
		enabled, err := chatResponsesBridgeEnabled(&model.Channel{Plugin: &plugin}, tt.original, tt.mapped)
		if enabled != tt.want || (err != nil) != tt.invalid {
			t.Fatalf("models=%v enabled=%t err=%v", tt.models, enabled, err)
		}
	}
}

func TestChatResponsesBridgeUsesMappedModelAndKeepsProtocol(t *testing.T) {
	previous := requester.HTTPClient
	requester.HTTPClient = http.DefaultClient
	defer func() { requester.HTTPClient = previous }()
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "json", true: "stream"}[stream], func(t *testing.T) {
			var attempts atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts.Add(1)
				body, _ := io.ReadAll(r.Body)
				if r.URL.Path != "/v1/responses" || gjson.GetBytes(body, "model").String() != "mapped-gpt" || gjson.GetBytes(body, "tools.0.function").Exists() || gjson.GetBytes(body, "tools.0.name").String() != "lookup" || gjson.GetBytes(body, "tool_choice.name").String() != "lookup" {
					t.Errorf("incorrect Responses request: path=%s body=%s", r.URL.Path, body)
				}
				if stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "data:{\"type\":\"response.created\",\"response\":{\"id\":\"resp_synthetic\",\"model\":\"mapped-gpt\",\"status\":\"in_progress\"}}\n\n")
					_, _ = io.WriteString(w, "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n")
					_, _ = io.WriteString(w, "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{}\"}\n\n")
					_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_synthetic\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n")
				} else {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"resp_synthetic","model":"mapped-gpt","object":"response","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}"}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`)
				}
			}))
			defer upstream.Close()
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			c.Request.Header.Set("Content-Type", "application/json")
			c.Set("original_model", "public-gpt")
			c.Set(config.GinRawPassThroughAllowedKey, true)
			plugin := datatypes.NewJSONType(model.PluginType{"chat_to_responses": {"models": "public-gpt"}})
			ch := &model.Channel{Type: config.ChannelTypeOpenAI, BaseURL: &upstream.URL, Key: "synthetic", Plugin: &plugin, PassThroughBody: true}
			p := openai.OpenAIProviderFactory{}.Create(ch)
			p.SetContext(c)
			p.SetUsage(&types.Usage{})
			r := NewRelayChat(c)
			r.provider = p
			r.originalModel = "public-gpt"
			r.modelName = "mapped-gpt"
			r.chatRequest = types.ChatCompletionRequest{Model: "public-gpt", Stream: stream, MaxTokens: 512, Messages: []types.ChatCompletionMessage{{Role: "user", Content: "lookup"}}, Tools: []*types.ChatCompletionTool{{Type: "function", Function: types.ChatCompletionFunction{Name: "lookup", Parameters: map[string]any{"type": "object"}}}}, ToolChoice: map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}}
			apiErr, _ := r.send()
			if apiErr != nil || attempts.Load() != 1 || c.GetBool(config.GinRawPassThroughAllowedKey) {
				t.Fatalf("bridge failed: err=%v attempts=%d", apiErr, attempts.Load())
			}
			body := w.Body.String()
			if !strings.Contains(body, "tool_calls") || strings.Contains(body, "\"object\":\"response\"") || strings.Contains(body, "function_call_arguments") {
				t.Fatalf("response is not Chat format: %s", body)
			}
			if !stream {
				var response types.ChatCompletionResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || response.Choices[0].Message.ToolCalls[0].Id != "call_1" {
					t.Fatalf("lost call ID: %s", body)
				}
			} else if strings.Count(body, "[DONE]") != 1 {
				t.Fatalf("bad stream terminator: %s", body)
			}
		})
	}
}

func TestChatResponsesBridgeRejectsMissingTerminalEvent(t *testing.T) {
	logger.Logger = zap.NewNop()
	previous := requester.HTTPClient
	requester.HTTPClient = http.DefaultClient
	defer func() { requester.HTTPClient = previous }()
	for _, marker := range []string{"", "data: [DONE]\n\n"} {
		t.Run(map[bool]string{false: "http-eof", true: "bare-done"}[marker != ""], func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_cut\",\"status\":\"in_progress\"}}\n\n")
				_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n"+marker)
			}))
			defer upstream.Close()
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			plugin := datatypes.NewJSONType(model.PluginType{"chat_to_responses": {"models": "gpt-test"}})
			ch := &model.Channel{Type: config.ChannelTypeOpenAI, BaseURL: &upstream.URL, Key: "synthetic", Plugin: &plugin}
			p := openai.OpenAIProviderFactory{}.Create(ch)
			p.SetContext(c)
			p.SetUsage(&types.Usage{})
			r := NewRelayChat(c)
			r.provider = p
			r.originalModel = "gpt-test"
			r.modelName = "gpt-test"
			r.chatRequest = types.ChatCompletionRequest{Model: "gpt-test", Stream: true, Messages: []types.ChatCompletionMessage{{Role: "user", Content: "synthetic"}}}
			apiErr, done := r.send()
			if apiErr == nil || apiErr.Code != "incomplete_upstream_stream" || !done || strings.Contains(w.Body.String(), "[DONE]") || !strings.Contains(w.Body.String(), "incomplete_upstream_stream") {
				t.Fatalf("truncated Responses stream reported success: err=%v done=%t body=%s", apiErr, done, w.Body.String())
			}
		})
	}
}

func TestChatResponsesBridgeReusesCustomEndpointAndParameters(t *testing.T) {
	previous := requester.HTTPClient
	requester.HTTPClient = http.DefaultClient
	defer func() { requester.HTTPClient = previous }()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if r.URL.Path != "/vendor/responses" || gjson.GetBytes(body, "temperature").Exists() ||
			gjson.GetBytes(body, "reasoning.effort").String() != "low" || gjson.GetBytes(body, "model").String() != "mapped-gpt" {
			t.Errorf("custom endpoint/parameter rules bypassed: %s %s", r.URL.Path, body)
		}
		_, _ = io.WriteString(w, `{"id":"resp_custom","model":"mapped-gpt","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"CUSTOM_OK"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`)
	}))
	defer upstream.Close()
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	plugin := datatypes.NewJSONType(model.PluginType{
		"chat_to_responses": {"models": "mapped-gpt"},
		"customize":         {"16": "/vendor/responses"},
	})
	custom := `{"per_model":true,"mapped-gpt":{"remove_params":["temperature"],"reasoning":{"effort":"low"}}}`
	ch := &model.Channel{Type: config.ChannelTypeCustom, BaseURL: &upstream.URL, Key: "synthetic", Plugin: &plugin, CustomParameter: &custom}
	p := openai.CreateOpenAIProvider(ch, upstream.URL)
	p.SetContext(c)
	p.SetUsage(&types.Usage{})
	r := NewRelayChat(c)
	r.provider, r.originalModel, r.modelName = p, "public-gpt", "mapped-gpt"
	temperature := 0.7
	r.chatRequest = types.ChatCompletionRequest{Model: "public-gpt", Temperature: &temperature, Messages: []types.ChatCompletionMessage{{Role: "user", Content: "hello"}}}
	apiErr, _ := r.send()
	if apiErr != nil || !strings.Contains(w.Body.String(), "CUSTOM_OK") {
		t.Fatalf("custom channel bridge failed: %v %s", apiErr, w.Body.String())
	}
}

func TestChatResponsesBridgeRejectsUnrepresentableContent(t *testing.T) {
	for _, req := range []*types.ChatCompletionRequest{
		{Messages: []types.ChatCompletionMessage{{Role: "user", Content: []types.ChatMessagePart{{Type: "input_audio"}}}}},
		{Messages: []types.ChatCompletionMessage{{Role: "user", Content: []types.ChatMessagePart{{Type: "file"}}}}},
		{Stop: "STOP"},
		{Functions: []*types.ChatCompletionFunction{{Name: "legacy"}}},
	} {
		if validateChatResponsesInput(req) == nil {
			t.Fatalf("conversion silently discarded unsupported data: %+v", req)
		}
	}
}

func TestChatResponsesBridgeRejectsUnsupportedProvider(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	plugin := datatypes.NewJSONType(model.PluginType{"chat_to_responses": {"models": "*"}})
	ch := &model.Channel{Type: config.ChannelTypeGemini, Plugin: &plugin}
	p := gemini.GeminiProviderFactory{}.Create(ch)
	p.SetContext(c)
	r := NewRelayChat(c)
	r.provider = p
	r.modelName = "gemini-test"
	r.originalModel = "gemini-test"
	err, done := r.send()
	if err == nil || err.Code != "unsupported_api" || !done {
		t.Fatalf("unsupported bridge silently fell back: %v", err)
	}
}
