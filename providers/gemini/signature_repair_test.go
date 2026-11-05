package gemini

import (
	"context"
	"done-hub/common/config"
	"done-hub/common/requester"
	"done-hub/model"
	"done-hub/types"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"gorm.io/datatypes"
)

const repairHistory = `{"contents":[{"role":"user","parts":[{"text":"lookup 1"}]},{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"x":"1","thoughtSignature":"argument-not-a-signature"}},"thoughtSignature":"rejected-native-signature"}]},{"role":"user","parts":[{"functionResponse":{"name":"lookup","response":{"output":"found","thoughtSignature":"result-not-a-signature"}}}]}],"generationConfig":{"maxOutputTokens":128}}`

func repairTestProvider(t *testing.T, serverURL string, enabled, stream bool, clientCtx context.Context) *GeminiProvider {
	t.Helper()
	initTestSignatureCache()
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(clientCtx)
	c.Request.Header.Set("Content-Type", "application/json")
	c.Set("id", 1)
	c.Set("is_stream", stream)
	plugin := datatypes.NewJSONType(model.PluginType{ThoughtSignatureAutofillPlugin: {"enable": enabled}})
	ch := &model.Channel{Id: 3, Type: config.ChannelTypeGemini, Key: "synthetic-credential", BaseURL: &serverURL, Plugin: &plugin}
	p := GeminiProviderFactory{}.Create(ch).(*GeminiProvider)
	p.SetContext(c)
	p.SetOriginalModel("test-gemini")
	p.SetUsage(&types.Usage{})
	return p
}

func setRepairTestClient(t *testing.T, timeout time.Duration) {
	t.Helper()
	previous := requester.HTTPClient
	transport := &http.Transport{}
	requester.HTTPClient = &http.Client{Transport: transport, Timeout: timeout}
	t.Cleanup(func() {
		requester.HTTPClient = previous
		transport.CloseIdleConnections()
	})
}

func signatureError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": status, "status": "INVALID_ARGUMENT", "message": message}})
}

// Both public protocols use the same final-body repair. The wire-level checks
// verify that resending cannot change a tool call/result or switch credentials.
func TestSignatureRepairChatAndNative(t *testing.T) {
	setRepairTestClient(t, 2*time.Second)
	for _, native := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("native=%t/stream=%t", native, stream), func(t *testing.T) {
				var attempts atomic.Int32
				var firstData []byte
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					data, _ := io.ReadAll(r.Body)
					if r.Header.Get("x-goog-api-key") != "synthetic-credential" || r.ContentLength != int64(len(data)) {
						t.Error("credential or Content-Length changed during repair")
					}
					if attempts.Add(1) == 1 {
						firstData = data
						if sig := gjson.GetBytes(data, "contents.1.parts.0.thoughtSignature").String(); sig != "rejected-native-signature" {
							t.Errorf("first request did not preserve the real signature: %q", sig)
						}
						w.Header().Set("x-request-id", "rejected-attempt")
						signatureError(w, 400, "Invalid thought signature.")
						return
					}
					var before, after map[string]any
					_ = json.Unmarshal(firstData, &before)
					_ = json.Unmarshal(data, &after)
					before["contents"].([]any)[1].(map[string]any)["parts"].([]any)[0].(map[string]any)["thoughtSignature"] = SkipThoughtSignatureValidator
					if !reflect.DeepEqual(before, after) {
						t.Error("repair changed something other than the model part signature")
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"REPAIRED\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":7,\"candidatesTokenCount\":2,\"totalTokenCount\":9}}\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"REPAIRED"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2,"totalTokenCount":9}}`)
					}
				}))
				defer server.Close()
				p := repairTestProvider(t, server.URL, true, stream, context.Background())
				p.GetContext().Set(config.GinProcessedBytesKey, []byte(repairHistory))
				var apiErr *types.OpenAIErrorWithStatusCode
				var output string
				if native {
					req := &GeminiChatRequest{Model: "test-gemini", Stream: stream}
					if stream {
						s, err := p.CreateGeminiChatStream(req)
						apiErr = err
						if err == nil {
							output = consumeRepairStream(t, s)
						}
					} else {
						r, err := p.CreateGeminiChat(req)
						apiErr = err
						if err == nil {
							output = r.Candidates[0].Content.Parts[0].Text
						}
					}
				} else {
					// Exercise real cache restoration in the OpenAI conversion path.
					parts := []GeminiChatCandidate{{Content: GeminiChatContent{Role: "model", Parts: []GeminiPart{{
						FunctionCall:     &GeminiFunctionCall{Name: "lookup", Args: map[string]any{"x": "1", "thoughtSignature": "argument-not-a-signature"}},
						ThoughtSignature: json.RawMessage(`"rejected-native-signature"`),
					}}}}}
					preserveFunctionCallSignatures(thoughtSignatureScope(p.Channel, p.Context, p.OriginalModel), parts)
					call := parts[0].Content.Parts[0].FunctionCall.ToOpenAITool()
					req := &types.ChatCompletionRequest{Model: "test-gemini", MaxTokens: 128, Stream: stream, Messages: []types.ChatCompletionMessage{
						{Role: "user", Content: "lookup 1"},
						{Role: "assistant", ToolCalls: []*types.ChatCompletionToolCalls{call}},
						{Role: "tool", ToolCallID: call.Id, Content: `{"output":"found","thoughtSignature":"result-not-a-signature"}`},
					}}
					if stream {
						s, err := p.CreateChatCompletionStream(req)
						apiErr = err
						if err == nil {
							output = consumeRepairStream(t, s)
						}
					} else {
						r, err := p.CreateChatCompletion(req)
						apiErr = err
						if err == nil {
							output = r.Choices[0].Message.StringContent()
						}
					}
				}
				if apiErr != nil || !strings.Contains(output, "REPAIRED") || attempts.Load() != 2 {
					t.Fatalf("repair failed: output=%q attempts=%d err=%v", output, attempts.Load(), apiErr)
				}
				if p.GetUsage().PromptTokens != 7 || p.GetUsage().CompletionTokens != 2 {
					t.Fatalf("successful response usage was lost: %+v", p.GetUsage())
				}
				if p.GetContext().GetString(config.GinUpstreamRequestIdKey) != "" {
					t.Fatal("failed attempt's request ID leaked into the successful response")
				}
				cached, _, _ := p.GetProcessedBodyBytes()
				if string(cached) != repairHistory {
					t.Fatal("repair polluted the shared processed-body cache")
				}
			})
		}
	}
}

func consumeRepairStream(t *testing.T, s requester.StreamReaderInterface[string]) string {
	t.Helper()
	defer s.Close()
	data, errs := s.Recv()
	var out strings.Builder
	for data != nil || errs != nil {
		select {
		case value, ok := <-data:
			if !ok {
				data = nil
			} else {
				out.WriteString(value)
			}
		case err, ok := <-errs:
			if !ok {
				errs = nil
			} else if err != nil && err != io.EOF {
				t.Fatalf("unexpected stream error: %v", err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("stream did not terminate")
		}
	}
	return out.String()
}

func TestSignatureRepairScopeAndErrorVariants(t *testing.T) {
	for _, message := range []string{"Invalid thought signature.", "Provider API error: Thought signature is not valid", "Corrupted thought signature", "thought_signature is invalid"} {
		if !IsThoughtSignatureFailure(message) {
			t.Fatalf("missed signature validation diagnostic: %s", message)
		}
	}
	const raw = `{"contents":[{"role":"model","parts":[{"text":"signed text","thought_signature":"old-signature"},{"functionCall":{"name":"lookup","args":{"thoughtSignature":"argument"}},"thoughtSignature":"old-signature"}]},{"role":"user","parts":[{"functionResponse":{"response":{"thoughtSignature":"result"}},"thoughtSignature":"user-field"}]}]}`
	out, n := ReplaceThoughtSignaturesBytes([]byte(raw))
	if n != 2 || gjson.GetBytes(out, "contents.0.parts.0.thought_signature").String() != SkipThoughtSignatureValidator ||
		gjson.GetBytes(out, "contents.0.parts.1.functionCall.args.thoughtSignature").String() != "argument" ||
		gjson.GetBytes(out, "contents.1.parts.0.functionResponse.response.thoughtSignature").String() != "result" ||
		gjson.GetBytes(out, "contents.1.parts.0.thoughtSignature").String() != "user-field" {
		t.Fatalf("repair escaped the model-part signature boundary: %s", out)
	}
	if _, n := ReplaceThoughtSignaturesBytes(out); n != 0 {
		t.Fatal("already repaired history was modified again")
	}
}

func TestSignatureRepairSafetyGates(t *testing.T) {
	setRepairTestClient(t, 2*time.Second)
	for _, tt := range []struct {
		name    string
		enabled bool
		status  int
		message string
		body    string
		want    int32
	}{
		{"disabled", false, 400, "Invalid thought signature.", repairHistory, 1},
		{"unrelated-400", true, 400, "Invalid function schema", repairHistory, 1},
		{"non-400", true, 503, "Invalid thought signature.", repairHistory, 1},
		{"already-sentinel", true, 400, "Invalid thought signature.", strings.ReplaceAll(repairHistory, "rejected-native-signature", SkipThoughtSignatureValidator), 1},
		{"no-model-signature", true, 400, "Invalid thought signature.", `{"contents":[{"role":"user","parts":[{"text":"hello","thoughtSignature":"do-not-touch"}]}]}`, 1},
		{"still-rejected", true, 400, "Invalid thought signature.", repairHistory, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var attempts atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts.Add(1)
				signatureError(w, tt.status, tt.message)
			}))
			defer server.Close()
			p := repairTestProvider(t, server.URL, tt.enabled, false, context.Background())
			p.GetContext().Set(config.GinProcessedBytesKey, []byte(tt.body))
			_, err := p.CreateGeminiChat(&GeminiChatRequest{Model: "test-gemini"})
			if err == nil || err.StatusCode != tt.status || !strings.Contains(err.Message, tt.message) || attempts.Load() != tt.want {
				t.Fatalf("unsafe resend or hidden error: attempts=%d err=%v", attempts.Load(), err)
			}
		})
	}
}

func TestSignatureRepairSharesDeadline(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			setRepairTestClient(t, 220*time.Millisecond)
			var attempts atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if attempts.Add(1) == 1 {
					time.Sleep(130 * time.Millisecond)
					signatureError(w, 400, "Invalid thought signature.")
					return
				}
				select {
				case <-r.Context().Done():
				case <-time.After(time.Second):
				}
			}))
			defer server.Close()
			p := repairTestProvider(t, server.URL, true, stream, context.Background())
			p.GetContext().Set(config.GinProcessedBytesKey, []byte(repairHistory))
			start := time.Now()
			var err *types.OpenAIErrorWithStatusCode
			if stream {
				_, err = p.CreateGeminiChatStream(&GeminiChatRequest{Model: "test-gemini", Stream: true})
			} else {
				_, err = p.CreateGeminiChat(&GeminiChatRequest{Model: "test-gemini"})
			}
			if err == nil || err.Code != "upstream_timeout" || time.Since(start) > 320*time.Millisecond || attempts.Load() != 2 {
				t.Fatalf("second attempt renewed the budget: elapsed=%s attempts=%d err=%v", time.Since(start), attempts.Load(), err)
			}
		})
	}
}

func TestSignatureRepairStopsOnClientCancellation(t *testing.T) {
	setRepairTestClient(t, 2*time.Second)
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var attempts atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				cancel()
				select {
				case <-r.Context().Done():
				case <-time.After(time.Second):
				}
			}))
			defer server.Close()
			p := repairTestProvider(t, server.URL, true, stream, ctx)
			p.GetContext().Set(config.GinProcessedBytesKey, []byte(repairHistory))
			var err *types.OpenAIErrorWithStatusCode
			if stream {
				_, err = p.CreateGeminiChatStream(&GeminiChatRequest{Model: "test-gemini", Stream: true})
			} else {
				_, err = p.CreateGeminiChat(&GeminiChatRequest{Model: "test-gemini"})
			}
			if err == nil || err.Code != "client_disconnected" || attempts.Load() != 1 {
				t.Fatalf("cancelled client caused another upstream call: attempts=%d err=%v", attempts.Load(), err)
			}
		})
	}
}

func TestSignatureRepairNeverReplaysStartedStream(t *testing.T) {
	setRepairTestClient(t, 2*time.Second)
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
		w.(http.Flusher).Flush()
		time.Sleep(10 * time.Millisecond)
		_, _ = io.WriteString(w, "data: {\"error\":{\"code\":400,\"message\":\"Invalid thought signature.\",\"status\":\"INVALID_ARGUMENT\"}}\n\n")
	}))
	defer server.Close()
	p := repairTestProvider(t, server.URL, true, true, context.Background())
	p.GetContext().Set(config.GinProcessedBytesKey, []byte(repairHistory))
	s, err := p.CreateGeminiChatStream(&GeminiChatRequest{Model: "test-gemini", Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	data, errs := s.Recv()
	var output string
	for {
		select {
		case value, ok := <-data:
			if !ok {
				data = nil
			} else {
				output += value
			}
		case err, ok := <-errs:
			if !ok {
				t.Fatal("stream hid the upstream error")
			}
			if err != nil && err != io.EOF {
				if !strings.Contains(output, "partial") || !strings.Contains(err.Error(), "Invalid thought signature") || attempts.Load() != 1 {
					t.Fatalf("started stream was replayed or corrupted: %q %v", output, err)
				}
				return
			}
		case <-time.After(3 * time.Second):
			t.Fatal("stream failed to terminate")
		}
	}
}
