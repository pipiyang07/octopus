package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

func TestBuildChatCompletionsRequestUsesExplicitWhitelist(t *testing.T) {
	content := "hello"
	user := "legacy-user"
	safetyID := "safe-user"
	enableThinking := true

	req := &model.InternalLLMRequest{
		Model: "gpt-4o",
		Messages: []model.Message{{
			Role: "developer",
			Content: model.MessageContent{
				Content: &content,
			},
		}},
		User:                    &user,
		SafetyIdentifier:        &safetyID,
		EnableThinking:          &enableThinking,
		Metadata:                map[string]string{"trace_id": "abc123"},
		ResponsesPromptCacheKey: stringPtr("resp_cache_only"),
		Audio: &struct {
			Format string `json:"format,omitempty"`
			Voice  string `json:"voice,omitempty"`
		}{
			Format: "mp3",
			Voice:  "alloy",
		},
	}

	wire := buildChatCompletionsRequest(req)
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal chat request failed: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal chat request failed: %v", err)
	}

	if got := payload["model"]; got != "gpt-4o" {
		t.Fatalf("expected model to be preserved, got %#v", got)
	}
	if got := payload["safety_identifier"]; got != safetyID {
		t.Fatalf("expected safety_identifier to be preserved, got %#v", got)
	}
	if _, ok := payload["metadata"]; !ok {
		t.Fatalf("expected metadata to be preserved, got %#v", payload)
	}
	// O-H1: `user` is a legacy but still-accepted OpenAI field; forward it when
	// the client supplied one so downstreams keying on it keep working.
	if got := payload["user"]; got != user {
		t.Fatalf("expected legacy user to be forwarded, got %#v", got)
	}
	if _, ok := payload["enable_thinking"]; ok {
		t.Fatalf("expected provider-specific enable_thinking to be omitted, got %#v", payload["enable_thinking"])
	}
	if _, ok := payload["prompt_cache_key"]; ok {
		t.Fatalf("expected responses-only prompt_cache_key to be omitted, got %#v", payload["prompt_cache_key"])
	}

	audio, ok := payload["audio"].(map[string]any)
	if !ok || audio["format"] != "mp3" || audio["voice"] != "alloy" {
		t.Fatalf("expected audio settings to be preserved, got %#v", payload["audio"])
	}
}

func TestChatOutboundDeepSeekResponsesCompatibility(t *testing.T) {
	visibleReasoning := "private reasoning"
	toolArgs := `{"cmd":"pwd"}`
	maxOutputTokens := int64(512)
	req := &model.InternalLLMRequest{
		Model:               "deepseek-v4.1-flash",
		RawAPIFormat:        model.APIFormatOpenAIResponse,
		MaxCompletionTokens: &maxOutputTokens,
		ReasoningEffort:     "high",
		Messages: []model.Message{
			{Role: "user", Content: model.MessageContent{Content: stringPtr("run a command")}},
			{Role: "assistant", ReasoningContent: &visibleReasoning},
			{Role: "assistant", ToolCalls: []model.ToolCall{{
				ID: "call_1", Type: "function",
				Function: model.FunctionCall{Name: "exec_command", Arguments: toolArgs},
			}}},
			{Role: "tool", ToolCallID: stringPtr("call_1"), Content: model.MessageContent{Content: stringPtr("/tmp")}},
			{Role: "assistant", Content: model.MessageContent{Content: stringPtr("Done")}},
		},
	}

	httpReq, err := (&ChatOutbound{}).TransformRequest(context.Background(), req, "https://api.deepseek.com/v1", "test-key")
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	var payload struct {
		Messages            []model.Message       `json:"messages"`
		MaxTokens           *int64                `json:"max_tokens"`
		MaxCompletionTokens *int64                `json:"max_completion_tokens"`
		ReasoningEffort     string                `json:"reasoning_effort"`
		Thinking            *model.ThinkingConfig `json:"thinking"`
		Store               *bool                 `json:"store"`
		ServiceTier         *string               `json:"service_tier"`
		Metadata            map[string]string     `json:"metadata"`
		SafetyIdentifier    *string               `json:"safety_identifier"`
		PromptCacheKey      *string               `json:"prompt_cache_key"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if payload.MaxTokens == nil || *payload.MaxTokens != maxOutputTokens || payload.MaxCompletionTokens != nil {
		t.Fatalf("expected max_completion_tokens mapped to max_tokens, got %#v", payload)
	}
	if payload.ReasoningEffort != "" {
		t.Fatalf("expected OpenAI reasoning_effort omitted for DeepSeek, got %q", payload.ReasoningEffort)
	}
	if payload.Thinking == nil || payload.Thinking.Type != "enabled" {
		t.Fatalf("expected reasoning effort mapped to DeepSeek thinking mode, got %#v", payload.Thinking)
	}
	if payload.Store != nil || payload.ServiceTier != nil || payload.Metadata != nil ||
		payload.SafetyIdentifier != nil || payload.PromptCacheKey != nil {
		t.Fatalf("expected Responses/OpenAI-only fields omitted for DeepSeek Chat, got %#v", payload)
	}
	if len(payload.Messages) != 4 {
		t.Fatalf("expected reasoning-only assistant message removed and all other history retained, got %#v", payload.Messages)
	}
	if payload.Messages[1].Role != "assistant" || len(payload.Messages[1].ToolCalls) != 1 {
		t.Fatalf("expected assistant tool call to be preserved, got %#v", payload.Messages[1])
	}
	for _, message := range payload.Messages {
		if message.Role == "assistant" && message.Content.Content == nil && len(message.ToolCalls) == 0 {
			t.Fatalf("DeepSeek payload contains empty assistant message: %#v", message)
		}
	}
}

func TestDeepSeekResponsesCompatibilityDoesNotAffectNativeChatRequests(t *testing.T) {
	maxCompletionTokens := int64(128)
	request := &model.InternalLLMRequest{
		Model:               "deepseek-v4.1-flash",
		RawAPIFormat:        model.APIFormatOpenAIChatCompletion,
		MaxCompletionTokens: &maxCompletionTokens,
		ReasoningEffort:     "high",
		Messages: []model.Message{{
			Role: "assistant", ReasoningContent: stringPtr("native chat reasoning"),
		}},
	}
	wire := buildChatCompletionsRequest(request)
	if wire.MaxCompletionTokens == nil || wire.MaxTokens != nil || wire.ReasoningEffort != "high" || len(wire.Messages) != 1 {
		t.Fatalf("expected native Chat request to keep existing behavior, got %#v", wire)
	}
}

func TestChatOutboundTransformResponseAggregatesUnlabeledSSEBody(t *testing.T) {
	body := strings.Join([]string{
		`: keepalive`,
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"hel"}}]}`,
		``,
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	outbound := &ChatOutbound{}
	response, err := outbound.TransformResponse(nil, &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("TransformResponse failed: %v", err)
	}
	if len(response.Choices) != 1 || response.Choices[0].Message == nil ||
		response.Choices[0].Message.Content.Content == nil ||
		*response.Choices[0].Message.Content.Content != "hello" {
		t.Fatalf("expected aggregated hello response, got %#v", response)
	}
	if response.Usage == nil || response.Usage.PromptTokens != 3 || response.Usage.CompletionTokens != 2 {
		t.Fatalf("expected usage to be retained, got %#v", response.Usage)
	}
}

func TestChatOutboundNormalizesReasoningAliases(t *testing.T) {
	outbound := &ChatOutbound{}
	response := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{
			"id":"chatcmpl_reasoning",
			"object":"chat.completion",
			"created":1,
			"model":"reasoning-model",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"content":"answer",
					"reasoning":{"content":"need to inspect the request"}
				},
				"finish_reason":"stop"
			}]
		}`)),
	}
	internalResponse, err := outbound.TransformResponse(context.Background(), response)
	if err != nil {
		t.Fatalf("TransformResponse failed: %v", err)
	}
	if len(internalResponse.Choices) != 1 || internalResponse.Choices[0].Message == nil {
		t.Fatalf("unexpected response: %#v", internalResponse)
	}
	if got := internalResponse.Choices[0].Message.GetReasoningContent(); got != "need to inspect the request" {
		t.Fatalf("expected object reasoning alias to be normalized, got %q", got)
	}

	stream, err := outbound.TransformStream(context.Background(), []byte(`{
		"id":"chatcmpl_reasoning_stream",
		"object":"chat.completion.chunk",
		"created":1,
		"model":"reasoning-model",
		"choices":[{
			"index":0,
			"delta":{
				"role":"assistant",
				"reasoning_details":[
					{"type":"reasoning_text","text":"first step"},
					{"type":"reasoning_text","text":"second step"}
				]
			}
		}]
	}`))
	if err != nil {
		t.Fatalf("TransformStream failed: %v", err)
	}
	if len(stream.Choices) != 1 || stream.Choices[0].Delta == nil {
		t.Fatalf("unexpected stream: %#v", stream)
	}
	if got := stream.Choices[0].Delta.GetReasoningContent(); got != "first step\n\nsecond step" {
		t.Fatalf("expected reasoning_details alias to be normalized, got %q", got)
	}
}

// TestBuildChatCompletionsRequestForwardsPromptCacheKey verifies that a
// client-supplied prompt_cache_key on the Chat entrypoint reaches the
// upstream Chat Completions payload. Before O-C4, PromptCacheKey was a
// *bool that json.Unmarshal never populated from a string wire value, so
// the cache key was silently dropped — losing up to ~90% input-cost savings
// for any client that relied on prompt-cache bucketing.
func TestBuildChatCompletionsRequestForwardsPromptCacheKey(t *testing.T) {
	cacheKey := "session-abc-123"
	content := "hi"
	req := &model.InternalLLMRequest{
		Model: "gpt-4o",
		Messages: []model.Message{{
			Role:    "user",
			Content: model.MessageContent{Content: &content},
		}},
		PromptCacheKey: &cacheKey,
	}

	wire := buildChatCompletionsRequest(req)
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := payload["prompt_cache_key"]; got != cacheKey {
		t.Fatalf("expected prompt_cache_key=%q in chat payload, got %#v", cacheKey, got)
	}
}

func TestBuildChatCompletionsRequestDerivesAnthropicPromptCacheKey(t *testing.T) {
	first := anthropicCacheRequest("latest question")
	second := anthropicCacheRequest("different latest question")

	firstWire := buildChatCompletionsRequest(first)
	secondWire := buildChatCompletionsRequest(second)
	if firstWire.PromptCacheKey == nil || secondWire.PromptCacheKey == nil {
		t.Fatalf("expected derived chat prompt_cache_key, got %+v and %+v", firstWire.PromptCacheKey, secondWire.PromptCacheKey)
	}
	if *firstWire.PromptCacheKey != *secondWire.PromptCacheKey {
		t.Fatalf("expected latest user changes to keep stable chat cache key, got %q and %q", *firstWire.PromptCacheKey, *secondWire.PromptCacheKey)
	}
}

// O-H1: 2025 Chat fields (verbosity, prediction, web_search_options) must land
// on the outbound payload when the client supplied them. Before this fix the
// whitelist simply dropped them, so gpt-5 verbosity and predicted outputs were
// silently unavailable through the aggregator.
func TestBuildChatCompletionsRequestForwards2025Fields(t *testing.T) {
	content := "hello"
	verbosity := "high"
	req := &model.InternalLLMRequest{
		Model: "gpt-5",
		Messages: []model.Message{{
			Role:    "user",
			Content: model.MessageContent{Content: &content},
		}},
		Verbosity:        &verbosity,
		Prediction:       json.RawMessage(`{"type":"content","content":"code edit preview"}`),
		WebSearchOptions: json.RawMessage(`{"search_context_size":"low"}`),
	}

	wire := buildChatCompletionsRequest(req)
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got := payload["verbosity"]; got != "high" {
		t.Fatalf("expected verbosity=high, got %#v", got)
	}
	prediction, ok := payload["prediction"].(map[string]any)
	if !ok || prediction["type"] != "content" {
		t.Fatalf("expected prediction to be forwarded, got %#v", payload["prediction"])
	}
	search, ok := payload["web_search_options"].(map[string]any)
	if !ok || search["search_context_size"] != "low" {
		t.Fatalf("expected web_search_options to be forwarded, got %#v", payload["web_search_options"])
	}
}

// TestTransformRequestPreservesDeveloperRole verifies O-L5: the Chat
// outbound no longer downgrades `developer` messages to `system`.
// OpenAI 2025+ model spec treats developer as the canonical instruction
// role for reasoning models, and the API accepts it natively, so we
// forward the caller's original role to keep gpt-5-series behaviour
// correct.
func TestTransformRequestPreservesDeveloperRole(t *testing.T) {
	outbound := &ChatOutbound{}
	content := "you are terse"
	req := &model.InternalLLMRequest{
		Model: "gpt-5",
		Messages: []model.Message{
			{Role: "developer", Content: model.MessageContent{Content: &content}},
			{Role: "user", Content: model.MessageContent{Content: stringPtr("hi")}},
		},
	}
	httpReq, err := outbound.TransformRequest(context.Background(), req, "https://api.openai.com", "sk-test")
	if err != nil {
		t.Fatalf("TransformRequest: %v", err)
	}
	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var payload struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(payload.Messages) < 1 {
		t.Fatalf("expected messages, got %+v", payload)
	}
	if role := payload.Messages[0]["role"]; role != "developer" {
		t.Errorf("expected first message role=developer, got %q", role)
	}
}

// Chat outbound forwards OpenAI-Organization / OpenAI-Project when they
// are present in TransformerMetadata, and skips them cleanly when absent
// or blank.
func TestBuildChatCompletionsRequestDropsImageGenerationTools(t *testing.T) {
	content := "hello"
	req := &model.InternalLLMRequest{
		Model: "gpt-4o",
		Messages: []model.Message{{
			Role:    "user",
			Content: model.MessageContent{Content: &content},
		}},
		Tools: []model.Tool{
			{Type: "function", Function: model.Function{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}},
			{Type: "image_generation", ImageGeneration: &model.ImageGeneration{Background: "transparent"}},
		},
	}

	wire := buildChatCompletionsRequest(req)
	body, err := json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	tools, ok := payload["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("expected only function tool in chat payload, got %#v", payload["tools"])
	}
	tool, ok := tools[0].(map[string]any)
	if !ok || tool["type"] != "function" {
		t.Fatalf("expected function tool in chat payload, got %#v", tools[0])
	}
}

func TestTransformRequestAttachesOrgAndProjectHeaders(t *testing.T) {
	outbound := &ChatOutbound{}
	content := "hi"
	req := &model.InternalLLMRequest{
		Model: "gpt-5",
		Messages: []model.Message{
			{Role: "user", Content: model.MessageContent{Content: &content}},
		},
	}
	req.SetTransformerMetadataValue(model.TransformerMetadataOpenAIOrganization, "org-abc")
	req.SetTransformerMetadataValue(model.TransformerMetadataOpenAIProject, "proj-xyz")
	httpReq, err := outbound.TransformRequest(context.Background(), req, "https://api.openai.com", "sk-test")
	if err != nil {
		t.Fatalf("TransformRequest: %v", err)
	}
	if got := httpReq.Header.Get("OpenAI-Organization"); got != "org-abc" {
		t.Errorf("OpenAI-Organization: got %q, want %q", got, "org-abc")
	}
	if got := httpReq.Header.Get("OpenAI-Project"); got != "proj-xyz" {
		t.Errorf("OpenAI-Project: got %q, want %q", got, "proj-xyz")
	}

	// Whitespace-only values are skipped so the header is not sent blank.
	req.SetTransformerMetadataValue(model.TransformerMetadataOpenAIOrganization, "   ")
	httpReq, err = outbound.TransformRequest(context.Background(), req, "https://api.openai.com", "sk-test")
	if err != nil {
		t.Fatalf("TransformRequest blank: %v", err)
	}
	if got := httpReq.Header.Get("OpenAI-Organization"); got != "" {
		t.Errorf("expected OpenAI-Organization omitted for blank value, got %q", got)
	}
}
