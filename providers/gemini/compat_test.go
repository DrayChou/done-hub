package gemini

import (
	"bytes"
	"done-hub/common/cache"
	"done-hub/common/config"
	"done-hub/common/logger"
	"done-hub/model"
	"done-hub/types"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
	"gorm.io/datatypes"
)

var signatureCacheOnce sync.Once

func initTestSignatureCache() {
	signatureCacheOnce.Do(func() {
		config.RedisEnabled = false
		cache.InitCacheManager()
		logger.Logger = zap.NewNop()
	})
}

func TestThoughtSignatureRoundTripIsolation(t *testing.T) {
	initTestSignatureCache()
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Set("id", 1)
	channel := &model.Channel{Id: 25, Key: "synthetic-test-credential"}
	scope := thoughtSignatureScope(channel, ctx, "test-gemini")
	makeResponse := func(signature string) []GeminiChatCandidate {
		return []GeminiChatCandidate{{Content: GeminiChatContent{Role: "model", Parts: []GeminiPart{{
			FunctionCall:     &GeminiFunctionCall{Name: "lookup", Id: "upstream-reused-id", Args: map[string]interface{}{"x": "same"}},
			ThoughtSignature: json.RawMessage(strconvQuote(signature)),
		}}}}}
	}
	first, second := makeResponse("signature-session-one"), makeResponse("signature-session-two")
	preserveFunctionCallSignatures(scope, first)
	preserveFunctionCallSignatures(scope, second)
	request := &types.ChatCompletionRequest{Tools: []*types.ChatCompletionTool{{Type: "function"}}}
	one := first[0].ToOpenAIChoice(request).Message.ToolCalls[0]
	two := second[0].ToOpenAIChoice(request).Message.ToolCalls[0]
	if one.Id == two.Id || one.Id == "upstream-reused-id" {
		t.Fatal("distinct call occurrences must not share a client ID")
	}
	if first[0].Content.Parts[0].FunctionCall.Id != "upstream-reused-id" {
		t.Fatal("native upstream ID was changed")
	}
	body := func(id, args string) []byte {
		return []byte(fmt.Sprintf(`{"contents":[{"role":"model","parts":[{"functionCall":{"id":%q,"name":"lookup","args":%s}}]}]}`, id, args))
	}
	for _, tt := range []struct{ id, want string }{{one.Id, "signature-session-one"}, {two.Id, "signature-session-two"}} {
		out, hits, fallback, err := restoreThoughtSignatures(body(tt.id, `{"x":"same"}`), scope, false)
		if err != nil || hits != 1 || fallback != 0 || gjson.GetBytes(out, "contents.0.parts.0.thoughtSignature").String() != tt.want {
			t.Fatalf("incorrect signature restoration: hits=%d fallback=%d err=%v", hits, fallback, err)
		}
	}
	ctx.Set("id", 2)
	otherUser := thoughtSignatureScope(channel, ctx, "test-gemini")
	ctx.Set("id", 1)
	for _, isolated := range []string{
		otherUser, thoughtSignatureScope(&model.Channel{Id: 26, Key: channel.Key}, ctx, "test-gemini"),
		thoughtSignatureScope(channel, ctx, "other-model"),
		thoughtSignatureScope(&model.Channel{Id: channel.Id, Key: "rotated-credential"}, ctx, "test-gemini"),
	} {
		out, hits, _, err := restoreThoughtSignatures(body(one.Id, `{"x":"same"}`), isolated, false)
		if err != nil || hits != 0 || gjson.GetBytes(out, "contents.0.parts.0.thoughtSignature").Exists() {
			t.Fatal("signature leaked across a scope boundary")
		}
	}
	for _, args := range []string{`{"x":"changed"}`, `[]`, `"invalid"`} {
		_, hits, _, err := restoreThoughtSignatures(body(one.Id, args), scope, false)
		if err != nil || hits != 0 {
			t.Fatal("modified/malformed call reused a signature")
		}
	}
	key := thoughtSignatureKey(scope, one.Id)
	if strings.Contains(key, channel.Key) {
		t.Fatal("credential present in cache key")
	}
	if err := cache.DeleteCache(key); err != nil {
		t.Fatal(err)
	}
	out, hits, fallback, err := restoreThoughtSignatures(body(one.Id, `{}`), scope, false)
	if err != nil || hits+fallback != 0 || gjson.GetBytes(out, "contents.0.parts.0.thoughtSignature").Exists() {
		t.Fatal("cache miss silently enabled fallback")
	}
	out, hits, fallback, err = restoreThoughtSignatures(body(one.Id, `{}`), scope, true)
	if err != nil || hits != 0 || fallback != 1 || gjson.GetBytes(out, "contents.0.parts.0.thoughtSignature").String() != SkipThoughtSignatureValidator {
		t.Fatal("explicit fallback did not handle cache miss")
	}
}

func strconvQuote(value string) string {
	data, _ := json.Marshal(value)
	return string(data)
}

func TestGeminiFinalRequestPaths(t *testing.T) {
	initTestSignatureCache()
	const raw = `{"contents":[{"role":"model","parts":[{"functionCall":{"id":"historic","name":"lookup","args":[]}}]},{"role":"user","parts":[{"functionResponse":{"name":"lookup","response":{"output":"ok"}}}]}]}`
	for _, path := range []string{"struct", "raw", "processed-bytes", "processed-map"} {
		t.Run(path, func(t *testing.T) {
			ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
			ctx.Set("id", 1)
			ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(raw))
			ctx.Request.Header.Set("Content-Type", "application/json")
			baseURL := "https://synthetic.invalid"
			channel := &model.Channel{Id: 25, Key: "synthetic", BaseURL: &baseURL,
				Plugin: &datatypes.JSONType[model.PluginType]{}}
			channel.Plugin = func() *datatypes.JSONType[model.PluginType] {
				p := datatypes.NewJSONType(model.PluginType{ThoughtSignatureAutofillPlugin: {"enable": true}})
				return &p
			}()
			provider := GeminiProviderFactory{}.Create(channel).(*GeminiProvider)
			provider.SetContext(ctx)
			provider.SetOriginalModel("test-gemini")
			input := &GeminiChatRequest{Model: "test-gemini", Contents: []GeminiChatContent{
				{Role: "model", Parts: []GeminiPart{{FunctionCall: &GeminiFunctionCall{Name: "lookup", Id: "historic"}}}},
			}}
			switch path {
			case "raw":
				ctx.Set(config.GinRequestBodyKey, []byte(raw))
			case "processed-bytes":
				provider.SetProcessedBodyBytes([]byte(raw), false)
			case "processed-map":
				var m map[string]interface{}
				_ = json.Unmarshal([]byte(raw), &m)
				provider.SetProcessedBody(m, false)
			}
			req, apiErr := provider.getChatRequest(input, path != "struct")
			if apiErr != nil {
				t.Fatalf("request construction failed: %v", apiErr)
			}
			body, err := io.ReadAll(req.Body)
			if err != nil || gjson.GetBytes(body, "contents.0.parts.0.thoughtSignature").String() != SkipThoughtSignatureValidator {
				t.Fatalf("final body missing fallback: %s (%v)", body, err)
			}
			if int64(len(body)) != req.ContentLength {
				t.Fatal("Content-Length not updated")
			}
			again, err := req.GetBody()
			if err != nil {
				t.Fatal(err)
			}
			replay, _ := io.ReadAll(again)
			_ = again.Close()
			if !bytes.Equal(body, replay) {
				t.Fatal("redirect/retry body differs")
			}
			if cached, _, ok := provider.GetProcessedBodyBytes(); ok && bytes.Contains(cached, []byte(SkipThoughtSignatureValidator)) {
				t.Fatal("fallback polluted shared retry bytes")
			}
			if m, _, ok := provider.GetProcessedBody(); ok {
				cached, _ := json.Marshal(m)
				if bytes.Contains(cached, []byte(SkipThoughtSignatureValidator)) {
					t.Fatal("fallback polluted shared retry map")
				}
			}
		})
	}
	for _, raw := range []string{
		`{"contents":[{"role":"model","parts":[{"functionCall":{"args":null},"thoughtSignature":"keep-original"}]}]}`,
		`{"contents":[{"role":"user","parts":[{"functionCall":{"args":{}}}]}]}`,
		`{"contents":[{"role":"model","parts":[{"text":"image; untouched","inlineData":{"data":"AAAA"}}]}]}`,
		`{"contents":null}`, `null`, `invalid-json`,
	} {
		out, hits, fallback, err := restoreThoughtSignatures([]byte(raw), "", true)
		if err != nil || hits+fallback != 0 || !bytes.Equal(out, []byte(raw)) {
			t.Fatalf("untouched input changed: %s", raw)
		}
	}
}

func TestGeminiParallelStreamToolRoundTrip(t *testing.T) {
	initTestSignatureCache()
	request := &types.ChatCompletionRequest{Model: "test-gemini", Tools: []*types.ChatCompletionTool{{Type: "function"}}}
	h := &GeminiStreamHandler{Usage: &types.Usage{}, Request: request, signatureScope: "parallel-test"}
	output := make(chan string, 20)
	errors := make(chan error, 1)
	send := func(parts []GeminiPart, finish *string) {
		response := GeminiChatResponse{Candidates: []GeminiChatCandidate{{Content: GeminiChatContent{Role: "model", Parts: parts}, FinishReason: finish}}}
		data, _ := json.Marshal(response)
		line := append([]byte("data: "), data...)
		h.HandlerStream(&line, output, errors)
	}
	for i := 0; i < 2; i++ {
		send([]GeminiPart{{FunctionCall: &GeminiFunctionCall{Name: "lookup", Args: map[string]interface{}{"n": i}}, ThoughtSignature: json.RawMessage(strconvQuote(fmt.Sprintf("signature-%d", i)))}}, nil)
	}
	finish := "STOP"
	send(nil, &finish)
	close(output)
	var calls []*types.ChatCompletionToolCalls
	var finishes []string
	for data := range output {
		var event types.ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatal(err)
		}
		for _, choice := range event.Choices {
			calls = append(calls, choice.Delta.ToolCalls...)
			if finish, ok := choice.FinishReason.(string); ok && finish != "" {
				finishes = append(finishes, finish)
			}
		}
	}
	if len(calls) != 2 || calls[0].Id == calls[1].Id || calls[0].Index != 0 || calls[1].Index != 1 {
		t.Fatalf("parallel tools lost identity: %+v", calls)
	}
	if !reflect.DeepEqual(finishes, []string{types.FinishReasonToolCalls}) {
		t.Fatalf("contradictory/early finish reasons: %v", finishes)
	}
	for i, call := range calls {
		data := []byte(fmt.Sprintf(`{"contents":[{"role":"model","parts":[{"functionCall":{"id":%q,"name":"lookup","args":{"n":%d}}}]}]}`, call.Id, i))
		out, hits, _, err := restoreThoughtSignatures(data, h.signatureScope, false)
		if err != nil || hits != 1 || gjson.GetBytes(out, "contents.0.parts.0.thoughtSignature").String() != fmt.Sprintf("signature-%d", i) {
			t.Fatal("stream signature did not round-trip")
		}
	}

	legacy := &GeminiStreamHandler{Usage: &types.Usage{}, Request: &types.ChatCompletionRequest{Model: "test-gemini", Functions: []*types.ChatCompletionFunction{{Name: "lookup"}}}}
	legacyOutput := make(chan string, 4)
	legacy.convertToOpenaiStream(&GeminiChatResponse{Candidates: []GeminiChatCandidate{{Content: GeminiChatContent{Parts: []GeminiPart{{FunctionCall: &GeminiFunctionCall{Name: "lookup", Args: map[string]interface{}{}}}}}, FinishReason: &finish}}}, legacyOutput)
	close(legacyOutput)
	legacyFinish := ""
	for data := range legacyOutput {
		var event types.ChatCompletionStreamResponse
		_ = json.Unmarshal([]byte(data), &event)
		for _, c := range event.Choices {
			if s, ok := c.FinishReason.(string); ok && s != "" {
				legacyFinish = s
			}
		}
	}
	if legacyFinish != types.FinishReasonFunctionCall {
		t.Fatalf("legacy function protocol regressed: %s", legacyFinish)
	}
}

func TestGeminiSchemaPositionsPreserveConstraints(t *testing.T) {
	var schema map[string]interface{}
	if err := json.Unmarshal([]byte(`{
	  "type":"object", "$schema":"draft", "examples":["annotation"],
	  "properties":{
	    "const":{"type":"string", "const":"fixed"},
	    "$ref":{"type":"object", "properties":{"deprecated":{"type":"string"}}},
	    "metadata":{"type":"object","patternProperties":{"^.*$":{}}},
	    "restricted":{"type":"object","patternProperties":{"^a":{"type":"number"}}},
	    "closed":{"type":"object","patternProperties":{"^.*$":{}},"additionalProperties":false},
	    "nested":{"type":"array","items":{"type":"object","properties":{"examples":{"type":"string","deprecated":true}}}},
	    "reference":{"$ref":"#/$defs/item"}
	  },
	  "$defs":{"item":{"type":"string","const":"fixed"}},
	  "additionalProperties":false,
	  "default":{"examples":"user data","deprecated":"also data"}
	}`), &schema); err != nil {
		t.Fatal(err)
	}
	cleanSchemaRecursively(schema)
	data, _ := json.Marshal(schema)
	for _, path := range []string{"properties.const.const", "properties.$ref.properties.deprecated", "properties.restricted.patternProperties", "properties.closed.patternProperties", "properties.reference.$ref", "$defs.item.const", "additionalProperties", "default.examples"} {
		if !gjson.GetBytes(data, path).Exists() {
			t.Fatalf("semantic/user data deleted at %s", path)
		}
	}
	for _, path := range []string{"$schema", "examples", "properties.metadata.patternProperties", "properties.nested.items.properties.examples.deprecated"} {
		if gjson.GetBytes(data, path).Exists() {
			t.Fatalf("unsupported annotation/redundant constraint remains at %s", path)
		}
	}
}

func TestGeminiGatewayErrorsRemainVisibleAndRedacted(t *testing.T) {
	for _, tt := range []struct{ body, want string }{
		{`{"detail":"Unknown name patternProperties; synthetic-secret synthetic-secret"}`, "Unknown name patternProperties"},
		{`{"detail":[{"msg":"bad","input":"private payload"}]}`, "upstream parameter validation failed"},
		{`{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"missing signature"}}`, "missing signature"},
		{`[{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"array error"}}]`, "array error"},
	} {
		response := &http.Response{StatusCode: 400, Body: io.NopCloser(strings.NewReader(tt.body)), Header: http.Header{"Content-Type": {"application/json"}}}
		apiErr := RequestErrorHandle("synthetic-secret")(response)
		if apiErr == nil || !strings.Contains(apiErr.Message, tt.want) || strings.Contains(apiErr.Message, "synthetic-secret") || strings.Contains(apiErr.Message, "private payload") {
			t.Fatalf("diagnostic missing/leaked: %+v", apiErr)
		}
	}
}

func TestThoughtSignatureCacheConcurrentOccurrences(t *testing.T) {
	initTestSignatureCache()
	var wait sync.WaitGroup
	for i := 0; i < 16; i++ {
		wait.Add(1)
		go func(i int) {
			defer wait.Done()
			parts := []GeminiChatCandidate{{Content: GeminiChatContent{Parts: []GeminiPart{{FunctionCall: &GeminiFunctionCall{Name: "same", Args: map[string]interface{}{}}, ThoughtSignature: json.RawMessage(strconvQuote(fmt.Sprintf("sig-%d", i)))}}}}}
			preserveFunctionCallSignatures("same-scope", parts)
			call := parts[0].Content.Parts[0].FunctionCall.ToOpenAITool()
			data := []byte(fmt.Sprintf(`{"contents":[{"role":"model","parts":[{"functionCall":{"id":%q,"name":"same","args":{}}}]}]}`, call.Id))
			out, hits, _, err := restoreThoughtSignatures(data, "same-scope", false)
			if err != nil || hits != 1 || gjson.GetBytes(out, "contents.0.parts.0.thoughtSignature").String() != fmt.Sprintf("sig-%d", i) {
				t.Errorf("concurrent occurrence %d mismatched", i)
			}
		}(i)
	}
	wait.Wait()
}
