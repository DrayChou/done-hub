package types

import (
	"encoding/json"
	"testing"
)

func TestChatResponsesRoundTripRetainsToolsAndText(t *testing.T) {
	store := false
	effort := "low"
	instructions := "preserve the task"
	request := &ChatCompletionRequest{Model: "gpt-test", MaxTokens: 2048, Store: &store, Instructions: &instructions, ReasoningEffort: &effort,
		ToolChoice: map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}},
		Tools:      []*ChatCompletionTool{{Type: "function", Function: ChatCompletionFunction{Name: "lookup", Parameters: map[string]any{"type": "object"}}}},
		Messages: []ChatCompletionMessage{
			{Role: "system", Content: "system instruction"},
			{Role: "assistant", Content: "I will check.", ToolCalls: []*ChatCompletionToolCalls{{Id: "call_1", Function: &ChatCompletionToolCallsFunction{Name: "lookup", Arguments: `{"key":"value"}`}}}},
			{Role: "tool", ToolCallID: "call_1", Content: "result"},
		},
	}
	converted := request.ToResponsesRequest()
	if converted.Instructions != instructions || converted.Store == nil || *converted.Store || converted.Reasoning == nil || *converted.Reasoning.Effort != effort || converted.ToolChoice.(map[string]any)["name"] != "lookup" {
		t.Fatalf("request-level contract lost: %+v", converted)
	}
	inputs := converted.Input.([]InputResponses)
	if len(inputs) != 4 || inputs[1].Type != InputTypeMessage || inputs[2].CallID != "call_1" || inputs[3].CallID != "call_1" || inputs[3].Output != "result" {
		t.Fatalf("assistant text/tool linkage lost: %+v", inputs)
	}
	wire, _ := json.Marshal(converted.Tools)
	var tools []map[string]any
	if err := json.Unmarshal(wire, &tools); err != nil || tools[0]["name"] != "lookup" || tools[0]["function"] != nil {
		t.Fatalf("function tools are not flattened: %s", wire)
	}
	response := &OpenAIResponsesResponses{Status: ResponseStatusCompleted, Usage: &ResponsesUsage{}, Output: []ResponsesOutput{
		{Type: InputTypeMessage, Content: "part one "}, {Type: InputTypeMessage, Content: "part two"},
		{Type: InputTypeFunctionCall, CallID: "call_1", Name: "lookup", Arguments: ArgumentsFromString(`{"key":"value"}`)},
	}}
	chat := response.ToChat()
	if chat.Choices[0].Message.StringContent() != "part one part two" || chat.Choices[0].Message.ToolCalls[0].Id != "call_1" || chat.Choices[0].FinishReason != FinishReasonToolCalls {
		t.Fatalf("response contract lost: %+v", chat.Choices[0])
	}
}

func TestResponsesIncompleteToolCallDoesNotBecomeExecutable(t *testing.T) {
	for _, itemStatus := range []string{"in_progress", ResponseStatusCompleted} {
		response := &OpenAIResponsesResponses{Status: ResponseStatusIncomplete, Usage: &ResponsesUsage{},
			IncompleteDetail: &IncompleteDetail{Reason: "max_output_tokens"},
			Output:           []ResponsesOutput{{Type: InputTypeFunctionCall, Status: itemStatus, CallID: "call_partial", Name: "lookup", Arguments: ArgumentsFromString(`{"x":`)}}}
		chat := response.ToChat()
		if chat.Choices[0].FinishReason != FinishReasonLength {
			t.Fatalf("truncated tool was marked executable: %+v", chat.Choices[0])
		}
	}
}

func TestParallelToolStreamRetainsDistinctCallIDs(t *testing.T) {
	choice := ChatCompletionStreamChoice{Index: 0, Delta: ChatCompletionStreamChoiceDelta{
		ToolCalls: []*ChatCompletionToolCalls{
			{Id: "first-call", Function: &ChatCompletionToolCallsFunction{Name: "lookup", Arguments: `{"x":1}`}},
			{Id: "second-call", Function: &ChatCompletionToolCallsFunction{Name: "lookup", Arguments: `{"x":2}`}},
		},
	}}
	events := choice.ConvertOpenaiStream()
	ids := make(map[int]string)
	for _, event := range events {
		for _, call := range event.Delta.ToolCalls {
			if call.Id != "" {
				ids[call.Index] = call.Id
			}
		}
	}
	if ids[0] != "first-call" || ids[1] != "second-call" {
		t.Fatalf("parallel function results cannot be paired: %v", ids)
	}
}
