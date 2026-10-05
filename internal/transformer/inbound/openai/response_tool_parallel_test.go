package openai

import (
	"context"
	"strings"
	"testing"
)

func TestResponsesParallelToolCallsStayInOneAssistantTurn(t *testing.T) {
	requestBody := `{
		"model":"kimi-k3",
		"input":[
			{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"README.md\"}"},
			{"type":"function_call","call_id":"call_2","name":"list_files","arguments":"{\"path\":\"src\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"Readme content"},
			{"type":"function_call_output","call_id":"call_2","output":["main.rs","lib.rs"]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Continue"}]}
		]
	}`
	request, err := (&ResponseInbound{}).TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if len(request.Messages) != 4 {
		t.Fatalf("expected assistant, two tool results, and user, got %#v", request.Messages)
	}
	assistant := request.Messages[0]
	if assistant.Role != "assistant" || len(assistant.ToolCalls) != 2 ||
		assistant.ToolCalls[0].ID != "call_1" || assistant.ToolCalls[1].ID != "call_2" {
		t.Fatalf("expected both parallel calls in one assistant turn, got %#v", assistant)
	}
	if request.Messages[1].Role != "tool" || *request.Messages[1].ToolCallID != "call_1" ||
		request.Messages[2].Role != "tool" || *request.Messages[2].ToolCallID != "call_2" {
		t.Fatalf("expected both tool results to stay adjacent, got %#v", request.Messages)
	}
	if request.Messages[3].Role != "user" {
		t.Fatalf("expected trailing user message, got %#v", request.Messages[3])
	}
}

func TestResponsesParallelToolMediaGroupsAfterAllToolOutputs(t *testing.T) {
	largeImage := "data:image/png;base64," + strings.Repeat("A", 8192)
	requestBody := `{
		"model":"kimi-k3",
		"input":[
			{"type":"function_call","call_id":"call_1","name":"render","arguments":"{}"},
			{"type":"function_call","call_id":"call_2","name":"render","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":{"type":"input_image","image_url":"` + largeImage + `"}},
			{"type":"reasoning","summary":[{"type":"summary_text","text":"keep outputs adjacent"}]},
			{"type":"function_call_output","call_id":"call_2","output":{"type":"image","mimeType":"image/webp","data":"SECOND_IMAGE_SENTINEL"}}
		]
	}`
	request, err := (&ResponseInbound{}).TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if len(request.Messages) != 4 {
		t.Fatalf("expected assistant, two tool results, and grouped media, got %#v", request.Messages)
	}
	assistant := request.Messages[0]
	if assistant.Role != "assistant" || len(assistant.ToolCalls) != 2 {
		t.Fatalf("expected parallel calls in one assistant turn, got %#v", assistant)
	}
	if assistant.ReasoningContent == nil || !strings.Contains(*assistant.ReasoningContent, "keep outputs adjacent") {
		t.Fatalf("expected reasoning to stay on assistant tool turn, got %#v", assistant.ReasoningContent)
	}
	for _, toolMessage := range request.Messages[1:3] {
		if toolMessage.Role != "tool" || toolMessage.Content.Content == nil ||
			strings.Contains(*toolMessage.Content.Content, largeImage) ||
			strings.Contains(*toolMessage.Content.Content, "SECOND_IMAGE_SENTINEL") {
			t.Fatalf("expected media to leave tool result, got %#v", toolMessage)
		}
	}
	mediaMessage := request.Messages[3]
	if mediaMessage.Role != "user" || len(mediaMessage.Content.MultipleContent) != 4 {
		t.Fatalf("expected two call markers and two media parts, got %#v", mediaMessage)
	}
	if mediaMessage.Content.MultipleContent[1].Type != "image_url" ||
		mediaMessage.Content.MultipleContent[1].ImageURL == nil ||
		mediaMessage.Content.MultipleContent[1].ImageURL.URL != largeImage {
		t.Fatalf("unexpected first moved image: %#v", mediaMessage.Content.MultipleContent[1])
	}
	if mediaMessage.Content.MultipleContent[3].Type != "image_url" ||
		mediaMessage.Content.MultipleContent[3].ImageURL == nil ||
		mediaMessage.Content.MultipleContent[3].ImageURL.URL != "data:image/webp;base64,SECOND_IMAGE_SENTINEL" {
		t.Fatalf("unexpected second moved image: %#v", mediaMessage.Content.MultipleContent[3])
	}
}

func TestResponsesToolMediaFlushesBeforeNextToolCallBatch(t *testing.T) {
	largeImage := "data:image/png;base64," + strings.Repeat("A", 8192)
	requestBody := `{
		"model":"kimi-k3",
		"input":[
			{"type":"function_call","call_id":"call_1","name":"render","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":{"type":"input_image","image_url":"` + largeImage + `"}},
			{"type":"function_call","call_id":"call_2","name":"render","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_2","output":"second result"}
		]
	}`
	request, err := (&ResponseInbound{}).TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if len(request.Messages) != 5 {
		t.Fatalf("expected two assistant/tool batches with media between them, got %#v", request.Messages)
	}
	expectedRoles := []string{"assistant", "tool", "user", "assistant", "tool"}
	for index, role := range expectedRoles {
		if request.Messages[index].Role != role {
			t.Fatalf("expected role %q at index %d, got %#v", role, index, request.Messages[index])
		}
	}
}

func TestResponsesCanonicalizesJSONToolPayloads(t *testing.T) {
	requestBody := `{
		"model":"kimi-k3",
		"input":[
			{"type":"function_call","call_id":"call_json","name":"lookup","arguments":"{ \"b\": 2, \"a\": 1 }"},
			{"type":"function_call_output","call_id":"call_json","output":"{ \"z\": true, \"a\": [2, 1] }"},
			{"type":"function_call","call_id":"call_plain","name":"read_file","arguments":"not json"},
			{"type":"function_call_output","call_id":"call_plain","output":"plain text result"}
		]
	}`
	request, err := (&ResponseInbound{}).TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if len(request.Messages) != 4 {
		t.Fatalf("expected two assistant/tool batches, got %#v", request.Messages)
	}
	jsonCall := request.Messages[0]
	if len(jsonCall.ToolCalls) != 1 || jsonCall.ToolCalls[0].Function.Arguments != `{"a":1,"b":2}` {
		t.Fatalf("expected canonical JSON arguments, got %#v", jsonCall)
	}
	if request.Messages[1].Content.Content == nil || *request.Messages[1].Content.Content != `{"a":[2,1],"z":true}` {
		t.Fatalf("expected canonical JSON tool output, got %#v", request.Messages[1].Content)
	}
	plainCall := request.Messages[2]
	if len(plainCall.ToolCalls) != 1 || plainCall.ToolCalls[0].Function.Arguments != "not json" {
		t.Fatalf("expected plain-text arguments to remain unchanged, got %#v", plainCall)
	}
	if request.Messages[3].Content.Content == nil || *request.Messages[3].Content.Content != "plain text result" {
		t.Fatalf("expected plain-text tool output to remain unchanged, got %#v", request.Messages[3].Content)
	}
}

func TestResponsesReasoningStaysOnFinalAnswerAfterToolCall(t *testing.T) {
	requestBody := `{
		"model":"kimi-k3",
		"input":[
			{"type":"reasoning","summary":[{"type":"summary_text","text":"need to read a file"}]},
			{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"Readme content"},
			{"type":"reasoning","summary":[{"type":"summary_text","text":"now I can answer"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"The file says hello."}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Continue"}]}
		]
	}`
	request, err := (&ResponseInbound{}).TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if len(request.Messages) != 4 {
		t.Fatalf("expected assistant tool turn, tool result, final assistant, and user, got %#v", request.Messages)
	}
	toolTurn := request.Messages[0]
	if toolTurn.Role != "assistant" || toolTurn.ReasoningContent == nil ||
		*toolTurn.ReasoningContent != "need to read a file" {
		t.Fatalf("expected first reasoning on tool turn, got %#v", toolTurn)
	}
	finalAnswer := request.Messages[2]
	if finalAnswer.Role != "assistant" || finalAnswer.Content.Content == nil ||
		*finalAnswer.Content.Content != "The file says hello." ||
		finalAnswer.ReasoningContent == nil || *finalAnswer.ReasoningContent != "now I can answer" {
		t.Fatalf("expected final answer to retain its own reasoning, got %#v", finalAnswer)
	}
	if request.Messages[3].Role != "user" || request.Messages[3].ReasoningContent != nil {
		t.Fatalf("expected user boundary without reasoning, got %#v", request.Messages[3])
	}
}
