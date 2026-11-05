package openai

import (
	"done-hub/types"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"done-hub/common/requester"
	"done-hub/model"

	"github.com/gin-gonic/gin"
)

func TestResponsesChatParallelToolDeltasAndTerminal(t *testing.T) {
	for _, usage := range []bool{false, true} {
		t.Run(map[bool]string{false: "no-usage", true: "with-usage"}[usage], func(t *testing.T) {
			h := OpenAIResponsesStreamHandler{Usage: &types.Usage{}, Model: "gpt-test"}
			data, errs := make(chan string, 12), make(chan error, 3)
			events := []string{
				`{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`,
				`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_a","type":"function_call","call_id":"call_a","name":"lookup_a","arguments":""}}`,
				`{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_b","type":"function_call","call_id":"call_b","name":"lookup_b","arguments":""}}`,
				`{"type":"response.function_call_arguments.delta","item_id":"fc_b","delta":"{\"b\":2}"}`,
				`{"type":"response.function_call_arguments.delta","item_id":"fc_a","delta":"{\"a\":1}"}`,
			}
			terminal := `{"type":"response.completed","response":{"id":"resp_1","status":"completed"}}
`
			if usage {
				terminal = `{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14}}}`
			}
			events = append(events, terminal)
			for _, ev := range events {
				raw := []byte("data:" + ev)
				h.HandlerChatStream(&raw, data, errs)
			}
			close(data)
			close(errs)
			for err := range errs {
				t.Fatalf("unexpected error: %v", err)
			}
			calls := map[int]types.ChatCompletionToolCalls{}
			finish := ""
			for raw := range data {
				var frame types.ChatCompletionStreamResponse
				if err := json.Unmarshal([]byte(raw), &frame); err != nil {
					t.Fatal(err)
				}
				for _, choice := range frame.Choices {
					if s, ok := choice.FinishReason.(string); ok && s != "" {
						finish = s
					}
					for _, delta := range choice.Delta.ToolCalls {
						call := calls[delta.Index]
						if call.Function == nil {
							call.Function = &types.ChatCompletionToolCallsFunction{}
						}
						if delta.Id != "" {
							call.Id = delta.Id
							call.Function.Name = delta.Function.Name
						}
						call.Function.Arguments += delta.Function.Arguments
						calls[delta.Index] = call
					}
				}
			}
			if len(calls) != 2 || calls[0].Id != "call_a" || calls[1].Id != "call_b" || calls[0].Function.Arguments != `{"a":1}` || calls[1].Function.Arguments != `{"b":2}` || finish != types.FinishReasonToolCalls {
				t.Fatalf("parallel calls corrupted: calls=%+v finish=%s", calls, finish)
			}
			if usage && h.Usage.TotalTokens != 14 {
				t.Fatal("terminal usage lost")
			}
		})
	}
}

func TestResponsesChatFinalToolArguments(t *testing.T) {
	for _, tt := range []struct {
		name, delta, final string
		itemDone, fail     bool
	}{
		{"done-only", "", `{"x":1}`, false, false},
		{"partial-plus-done", `{"x":`, `{"x":1}`, false, false},
		{"full-plus-done", `{"x":1}`, `{"x":1}`, false, false},
		{"item-done-only", "", `{"x":1}`, true, false},
		{"conflict", `{"x":2}`, `{"x":1}`, false, true},
		{"truncated", "", `{"x":`, false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h := OpenAIResponsesStreamHandler{Usage: &types.Usage{}, Model: "gpt-test"}
			data, errs := make(chan string, 8), make(chan error, 2)
			events := []map[string]any{{"type": "response.output_item.added", "item": map[string]any{"id": "fc_a", "type": "function_call", "call_id": "call_a", "name": "lookup", "arguments": ""}}}
			if tt.delta != "" {
				events = append(events, map[string]any{"type": "response.function_call_arguments.delta", "item_id": "fc_a", "delta": tt.delta})
			}
			final := map[string]any{"type": "response.function_call_arguments.done", "item_id": "fc_a", "arguments": tt.final}
			if tt.itemDone {
				final = map[string]any{"type": "response.output_item.done", "item": map[string]any{"id": "fc_a", "type": "function_call", "call_id": "call_a", "name": "lookup", "arguments": tt.final}}
			}
			events = append(events, final)
			for _, event := range events {
				raw, _ := json.Marshal(event)
				line := append([]byte("data: "), raw...)
				h.HandlerChatStream(&line, data, errs)
			}
			if (len(errs) > 0) != tt.fail {
				t.Fatalf("final arguments accepted/rejected incorrectly: errors=%d", len(errs))
			}
			if tt.fail {
				return
			}
			// Both final events often carry the same full JSON. The second one
			// must not append it again or count it twice for fallback billing.
			raw, _ := json.Marshal(final)
			line := append([]byte("data: "), raw...)
			h.HandlerChatStream(&line, data, errs)
			close(data)
			var arguments string
			for raw := range data {
				var frame types.ChatCompletionStreamResponse
				if err := json.Unmarshal([]byte(raw), &frame); err != nil {
					t.Fatal(err)
				}
				for _, choice := range frame.Choices {
					for _, tool := range choice.Delta.ToolCalls {
						arguments += tool.Function.Arguments
					}
				}
			}
			if arguments != tt.final || h.Usage.TextBuilder.String() != tt.final || len(errs) != 0 {
				t.Fatalf("arguments duplicated/lost: got=%q billed=%q errors=%d", arguments, h.Usage.TextBuilder.String(), len(errs))
			}
		})
	}
}

func TestResponsesChatFailureIsNotSuccess(t *testing.T) {
	for _, raw := range []string{
		`{"type":"error","message":"synthetic fail"}`,
		`{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"type":"upstream_error","message":"synthetic fail"}}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"failed"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"in_progress"}}`,
	} {
		h := OpenAIResponsesStreamHandler{Usage: &types.Usage{}}
		data, errs := make(chan string, 2), make(chan error, 2)
		line := []byte("data: " + raw)
		h.HandlerChatStream(&line, data, errs)
		if len(errs) != 1 || len(data) != 0 {
			t.Fatalf("failure returned successful Chat finish: %s", raw)
		}
	}
}

func TestResponsesChatBodyFailureIsNotSuccess(t *testing.T) {
	previous := requester.HTTPClient
	requester.HTTPClient = &http.Client{Timeout: time.Second}
	defer func() { requester.HTTPClient = previous }()
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"id":"resp_1","status":"failed","error":{"type":"upstream_error","message":"synthetic fail"}}`)
	}))
	defer up.Close()
	// Exercise the request/response methods without a relay side success marker.
	p := CreateOpenAIProvider(&model.Channel{BaseURL: &up.URL, Key: "synthetic"}, up.URL)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	p.SetContext(c)
	p.SetUsage(&types.Usage{})
	_, err := p.CreateResponses(&types.OpenAIResponsesRequest{Model: "gpt-test", ConvertChat: true})
	if err == nil || err.StatusCode != http.StatusBadGateway || !strings.Contains(err.Message, "synthetic fail") {
		t.Fatalf("HTTP 200 Responses failure became a successful Chat response: %v", err)
	}
}
