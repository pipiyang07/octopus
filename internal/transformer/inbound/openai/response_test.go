package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestConvertToInternalRequestPreservesRawInputItems(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-4o",
		Input: ResponsesInput{Items: []ResponsesItem{
			{Type: "input_text", Text: stringPtr("hello")},
		}},
	}

	internalReq, err := convertToInternalRequest(req)
	if err != nil {
		t.Fatalf("convertToInternalRequest failed: %v", err)
	}
	if len(internalReq.RawInputItems) == 0 {
		t.Fatalf("expected raw input items to be preserved")
	}

	var items []map[string]any
	if err := json.Unmarshal(internalReq.RawInputItems, &items); err != nil {
		t.Fatalf("unmarshal raw input items failed: %v", err)
	}
	if len(items) != 1 || items[0]["type"] != "input_text" {
		t.Fatalf("expected original raw input items to be kept, got %#v", items)
	}
	if internalReq.TransformOptions.ArrayInputs == nil || !*internalReq.TransformOptions.ArrayInputs {
		t.Fatalf("expected array input flag to stay true")
	}
}

func TestConvertToInternalRequestMarksPassthroughForUnsupportedToolType(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-4o",
		Input: ResponsesInput{Text: stringPtr("hello")},
		Tools: []ResponsesTool{{
			Type: "apply_patch",
		}},
	}

	internalReq, err := convertToInternalRequest(req)
	if err != nil {
		t.Fatalf("convertToInternalRequest failed: %v", err)
	}
	if !internalReq.HasOpenAIResponsesPassthrough() {
		t.Fatalf("expected unsupported responses tool to require passthrough")
	}
	if ext := internalReq.GetOpenAIExtensions(); !ext.ResponsesPassthroughRequired || ext.ResponsesPassthroughReason != "tool:apply_patch" {
		t.Fatalf("expected OpenAI extension passthrough view, got %#v", ext)
	}
}

func TestConvertToInternalRequestCoalescesResponsesAssistantToolTurn(t *testing.T) {
	req := &ResponsesRequest{
		Model: "kimi-k3",
		Input: ResponsesInput{Items: []ResponsesItem{
			{
				Type: "reasoning",
				Summary: []ResponsesReasoningSummary{{
					Type: "summary_text",
					Text: "Need the shell result.",
				}},
			},
			{
				Type: "message",
				Role: "assistant",
				Content: &ResponsesInput{Items: []ResponsesItem{
					{Type: "output_text", Text: stringPtr("Running the command.")},
				}},
			},
			{
				Type:      "function_call",
				CallID:    "call_kimi",
				Name:      "exec_command",
				Arguments: json.RawMessage(`{"cmd":"pwd"}`),
			},
			{
				Type:      "function_call_output",
				CallID:    "call_kimi",
				RawOutput: json.RawMessage(`"/tmp"`),
			},
		}},
	}

	internalReq, err := convertToInternalRequest(req)
	if err != nil {
		t.Fatalf("convertToInternalRequest failed: %v", err)
	}
	var roles []string
	for _, message := range internalReq.Messages {
		roles = append(roles, message.Role)
	}
	if strings.Join(roles, ",") != "assistant,tool" {
		t.Fatalf("expected one assistant tool-call message followed by tool result, got roles=%v messages=%#v", roles, internalReq.Messages)
	}
	assistant := internalReq.Messages[0]
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].ID != "call_kimi" {
		t.Fatalf("expected tool call to be retained, got %#v", assistant)
	}
	if assistant.Content.Content == nil || *assistant.Content.Content != "Running the command." {
		t.Fatalf("expected commentary text to stay on the assistant message, got %#v", assistant.Content)
	}
	if assistant.ReasoningContent == nil || !strings.Contains(*assistant.ReasoningContent, "Need the shell result.") {
		t.Fatalf("expected reasoning to attach to the tool-call turn, got %#v", assistant.ReasoningContent)
	}
}

func TestConvertToInternalRequestBackfillsAssistantToolCallReasoning(t *testing.T) {
	req := &ResponsesRequest{
		Model: "kimi-k3",
		Input: ResponsesInput{Items: []ResponsesItem{
			{
				Type:      "function_call",
				CallID:    "call_no_reasoning",
				Name:      "echo",
				Arguments: json.RawMessage(`{"text":"ok"}`),
			},
			{
				Type:      "function_call_output",
				CallID:    "call_no_reasoning",
				RawOutput: json.RawMessage(`"ok"`),
			},
		}},
	}

	internalReq, err := convertToInternalRequest(req)
	if err != nil {
		t.Fatalf("convertToInternalRequest failed: %v", err)
	}
	if len(internalReq.Messages) != 2 || internalReq.Messages[0].Role != "assistant" {
		t.Fatalf("expected assistant tool call and tool output, got %#v", internalReq.Messages)
	}
	if internalReq.Messages[0].ReasoningContent == nil || strings.TrimSpace(*internalReq.Messages[0].ReasoningContent) == "" {
		t.Fatalf("expected assistant tool call to receive reasoning placeholder, got %#v", internalReq.Messages[0])
	}
}

func TestConvertToInternalRequestMarksPassthroughForUnsupportedInputItem(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-4o",
		Input: ResponsesInput{Items: []ResponsesItem{{
			Type:   "apply_patch_call_output",
			CallID: "apc_123",
		}}},
	}

	internalReq, err := convertToInternalRequest(req)
	if err != nil {
		t.Fatalf("convertToInternalRequest failed: %v", err)
	}
	if !internalReq.HasOpenAIResponsesPassthrough() {
		t.Fatalf("expected unsupported responses input item to require passthrough")
	}
	if ext := internalReq.GetOpenAIExtensions(); !ext.ResponsesPassthroughRequired || ext.ResponsesPassthroughReason != "input:apply_patch_call_output" {
		t.Fatalf("expected OpenAI extension passthrough view, got %#v", ext)
	}
}

func TestConvertToInternalRequestDoesNotMarkPassthroughForSupportedFileAndAudioInputs(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-4o",
		Input: ResponsesInput{Items: []ResponsesItem{
			{
				Type: "message",
				Role: "user",
				Content: &ResponsesInput{Items: []ResponsesItem{
					{Type: "input_file", FileID: stringPtr("file_123")},
					{Type: "input_audio", InputAudio: &ResponsesInputAudio{Format: "wav", Data: "AAA="}},
				}},
			},
		}},
	}

	internalReq, err := convertToInternalRequest(req)
	if err != nil {
		t.Fatalf("convertToInternalRequest failed: %v", err)
	}
	if internalReq.HasOpenAIResponsesPassthrough() {
		t.Fatalf("expected supported file/audio inputs to stay normalized without passthrough")
	}
	if len(internalReq.Messages) != 1 || len(internalReq.Messages[0].Content.MultipleContent) != 2 {
		t.Fatalf("expected supported file/audio inputs to normalize into message content, got %#v", internalReq.Messages)
	}
	if internalReq.Messages[0].Content.MultipleContent[0].Type != "file" {
		t.Fatalf("expected file content part, got %#v", internalReq.Messages[0].Content.MultipleContent[0])
	}
	if internalReq.Messages[0].Content.MultipleContent[1].Type != "input_audio" {
		t.Fatalf("expected input_audio content part, got %#v", internalReq.Messages[0].Content.MultipleContent[1])
	}
}

func TestConvertToInternalRequestNormalizesTopLevelInputFile(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-4o",
		Input: ResponsesInput{Items: []ResponsesItem{{
			Type:     "input_file",
			FileID:   stringPtr("file_456"),
			Filename: stringPtr("notes.txt"),
		}}},
	}

	internalReq, err := convertToInternalRequest(req)
	if err != nil {
		t.Fatalf("convertToInternalRequest failed: %v", err)
	}
	if internalReq.HasOpenAIResponsesPassthrough() {
		t.Fatalf("expected top-level input_file to stay normalized without passthrough")
	}
	if len(internalReq.Messages) != 1 {
		t.Fatalf("expected one normalized message, got %#v", internalReq.Messages)
	}
	if internalReq.Messages[0].Role != "user" {
		t.Fatalf("expected top-level input_file to default to user role, got %#v", internalReq.Messages[0].Role)
	}
	if len(internalReq.Messages[0].Content.MultipleContent) != 1 || internalReq.Messages[0].Content.MultipleContent[0].Type != "file" {
		t.Fatalf("expected top-level input_file to become file content, got %#v", internalReq.Messages[0].Content)
	}
	if internalReq.Messages[0].Content.MultipleContent[0].File == nil || internalReq.Messages[0].Content.MultipleContent[0].File.FileID != "file_456" {
		t.Fatalf("expected normalized file reference to preserve file_id, got %#v", internalReq.Messages[0].Content.MultipleContent[0].File)
	}
}

func stringPtr(value string) *string {
	return &value
}
