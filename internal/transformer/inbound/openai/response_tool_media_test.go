package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestResponsesToolOutputMovesSupportedMediaToUserMessage(t *testing.T) {
	requestBody := `{
		"model":"gpt-5.4",
		"input":[
			{"type":"function_call","call_id":"call_media","name":"render","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_media","output":{
				"content":[
					{"type":"input_text","text":"render finished"},
					{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo=","detail":"original"},
					{"type":"input_file","file_id":"file_123","filename":"report.pdf"},
					{"type":"input_audio","input_audio":{"data":"AUDIO_SENTINEL","format":"wav"}}
				]
			}}
		]
	}`
	inbound := &ResponseInbound{}
	request, err := inbound.TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if request.HasOpenAIResponsesPassthrough() {
		t.Fatalf("supported tool media should be Chat-convertible, got %q", request.OpenAIResponsesPassthroughReasonTextValue())
	}
	if len(request.Messages) != 3 {
		t.Fatalf("expected assistant, tool, and synthetic user messages, got %#v", request.Messages)
	}

	toolMessage := request.Messages[1]
	if toolMessage.Role != "tool" || toolMessage.ToolCallID == nil || *toolMessage.ToolCallID != "call_media" {
		t.Fatalf("unexpected tool message: %#v", toolMessage)
	}
	if toolMessage.Content.Content == nil || !strings.Contains(*toolMessage.Content.Content, responsesToolMediaMovedMarker) {
		t.Fatalf("expected tool result to contain media moved marker, got %#v", toolMessage.Content)
	}
	if strings.Contains(*toolMessage.Content.Content, "AUDIO_SENTINEL") {
		t.Fatalf("expected media payload to leave tool result, got %q", *toolMessage.Content.Content)
	}

	mediaMessage := request.Messages[2]
	if mediaMessage.Role != "user" || len(mediaMessage.Content.MultipleContent) != 4 {
		t.Fatalf("unexpected synthetic media message: %#v", mediaMessage)
	}
	if mediaMessage.Content.MultipleContent[0].Type != "text" ||
		!strings.Contains(*mediaMessage.Content.MultipleContent[0].Text, "media output of tool call call_media") {
		t.Fatalf("expected media call marker, got %#v", mediaMessage.Content.MultipleContent[0])
	}
	image := mediaMessage.Content.MultipleContent[1]
	if image.Type != "image_url" || image.ImageURL == nil || !strings.HasPrefix(image.ImageURL.URL, "data:image/png;base64,") {
		t.Fatalf("unexpected image part: %#v", image)
	}
	if image.ImageURL.Detail == nil || *image.ImageURL.Detail != "auto" {
		t.Fatalf("expected original detail to be downgraded to auto, got %#v", image.ImageURL.Detail)
	}
	file := mediaMessage.Content.MultipleContent[2]
	if file.Type != "file" || file.File == nil || file.File.FileID != "file_123" || file.File.Filename != "report.pdf" {
		t.Fatalf("unexpected file part: %#v", file)
	}
	audio := mediaMessage.Content.MultipleContent[3]
	if audio.Type != "input_audio" || audio.Audio == nil || audio.Audio.Data != "AUDIO_SENTINEL" || audio.Audio.Format != "wav" {
		t.Fatalf("unexpected audio part: %#v", audio)
	}
}

func TestResponsesCustomToolOutputClampsResidualLargeMedia(t *testing.T) {
	requestBody := `{
		"model":"gpt-5.4",
		"input":[
			{"type":"custom_tool_call","call_id":"call_custom","name":"render","input":"draw"},
			{
				"type":"custom_tool_call_output",
				"call_id":"call_custom",
				"status":"completed",
				"output":{
					"content":[
						{"type":"input_image","image_url":"data:image/png;base64,CUSTOM_SENTINEL"},
						{"type":"video","data":"` + strings.Repeat("A", 20000) + `"}
					]
				}
			}
		]
	}`
	inbound := &ResponseInbound{}
	request, err := inbound.TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("expected custom call, tool result, and synthetic user message, got %#v", request.Messages)
	}

	toolMessage := request.Messages[1]
	if toolMessage.Role != "tool" || toolMessage.Content.Content == nil {
		t.Fatalf("unexpected custom tool result: %#v", toolMessage)
	}
	toolContent := *toolMessage.Content.Content
	if !strings.Contains(toolContent, responsesToolMediaMovedMarker) {
		t.Fatalf("expected media moved marker, got %s", toolContent)
	}
	if !strings.Contains(toolContent, "[octopus: omitted 20000 bytes]") {
		t.Fatalf("expected residual media payload to be clamped, got %s", toolContent)
	}
	if strings.Contains(toolContent, strings.Repeat("A", 128)) {
		t.Fatalf("expected large payload bytes to be omitted, got %s", toolContent)
	}

	mediaMessage := request.Messages[2]
	if mediaMessage.Role != "user" || len(mediaMessage.Content.MultipleContent) != 2 ||
		mediaMessage.Content.MultipleContent[0].Type != "text" ||
		!strings.Contains(*mediaMessage.Content.MultipleContent[0].Text, "media output of tool call call_custom") ||
		mediaMessage.Content.MultipleContent[1].Type != "image_url" ||
		mediaMessage.Content.MultipleContent[1].ImageURL == nil ||
		mediaMessage.Content.MultipleContent[1].ImageURL.URL != "data:image/png;base64,CUSTOM_SENTINEL" {
		t.Fatalf("unexpected moved image: %#v", mediaMessage)
	}
}

func TestPlanResponsesToolOutputRejectsFalsePositiveMediaShapes(t *testing.T) {
	raw := json.RawMessage(`{"content":[{"type":"image","name":"business metadata"}]}`)
	content, mediaParts, err := planResponsesToolOutput(raw)
	if err != nil {
		t.Fatalf("planResponsesToolOutput failed: %v", err)
	}
	if len(mediaParts) != 0 {
		t.Fatalf("expected no media parts, got %#v", mediaParts)
	}
	if !strings.Contains(content, "business metadata") {
		t.Fatalf("expected metadata object to remain unchanged, got %s", content)
	}
}
