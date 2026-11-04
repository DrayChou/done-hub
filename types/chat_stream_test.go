package types

import "testing"

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
