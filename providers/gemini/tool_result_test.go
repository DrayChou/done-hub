package gemini

import (
	"done-hub/types"
	"encoding/json"
	"testing"
)

func TestConvertedToolResultsUseGeminiUserTurn(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		var messages []types.ChatCompletionMessage
		if legacy {
			name := "lookup"
			messages = []types.ChatCompletionMessage{
				{Role: "user", Content: "synthetic question"},
				{Role: "assistant", Content: "", FunctionCall: &types.ChatCompletionToolCallsFunction{Name: name, Arguments: `{}`}},
				{Role: "function", Name: &name, Content: "synthetic result"},
			}
		} else {
			messages = []types.ChatCompletionMessage{
				{Role: "user", Content: "synthetic question"},
				{Role: "assistant", Content: "", ToolCalls: []*types.ChatCompletionToolCalls{
					{Id: "call_a", Function: &types.ChatCompletionToolCallsFunction{Name: "lookup", Arguments: `{"code":"A"}`}},
					{Id: "call_b", Function: &types.ChatCompletionToolCallsFunction{Name: "lookup", Arguments: `{"code":"B"}`}},
				}},
				{Role: "tool", ToolCallID: "call_a", Content: "result A"},
				{Role: "tool", ToolCallID: "call_b", Content: "result B"},
			}
		}
		contents, _, apiErr := OpenAIToGeminiChatContent(messages)
		if apiErr != nil || len(contents) != 3 || contents[2].Role != "user" {
			t.Fatalf("invalid native result turn, legacy=%t: %#v %v", legacy, contents, apiErr)
		}
		want := 2
		if legacy {
			want = 1
		}
		if len(contents[2].Parts) != want {
			t.Fatalf("parallel results were not grouped: %d", len(contents[2].Parts))
		}
		for i, part := range contents[2].Parts {
			if part.FunctionResponse == nil || part.FunctionResponse.Name != "lookup" {
				t.Fatal("tool result dropped/renamed")
			}
			if !legacy {
				var id string
				_ = json.Unmarshal(part.FunctionResponse.ID, &id)
				if id != []string{"call_a", "call_b"}[i] {
					t.Fatal("tool result pairing changed")
				}
			}
		}
		messages = append(messages, types.ChatCompletionMessage{Role: "user", Content: "follow-up"})
		contents, _, apiErr = OpenAIToGeminiChatContent(messages)
		if apiErr != nil || len(contents) != 4 || contents[3].Role != "user" || contents[3].Parts[0].Text != "follow-up" {
			t.Fatal("follow-up text was merged into function responses")
		}
	}
}
