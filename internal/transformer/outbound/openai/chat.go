package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/bestruirui/octopus/internal/transformer/model"
	"github.com/tmaxmax/go-sse"
)

type ChatOutbound struct{}

// ChatCompletionsTool is the explicit OpenAI chat/completions wire tool payload.
// Keeping this separate from the shared model prevents provider-specific fields
// from leaking into the Chat request body.
type ChatCompletionsTool struct {
	Type     string         `json:"type,omitempty"`
	Function model.Function `json:"function,omitempty"`
}

// ChatCompletionsRequest is the explicit OpenAI chat/completions wire payload.
// Keeping this as a whitelist prevents internal/provider-specific fields on the
// shared InternalLLMRequest from leaking to OpenAI-compatible upstreams.
type ChatCompletionsRequest struct {
	Messages []model.Message `json:"messages"`
	Model    string          `json:"model"`

	FrequencyPenalty    *float64              `json:"frequency_penalty,omitempty"`
	Logprobs            *bool                 `json:"logprobs,omitempty"`
	MaxCompletionTokens *int64                `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int64                `json:"max_tokens,omitempty"`
	PresencePenalty     *float64              `json:"presence_penalty,omitempty"`
	Seed                *int64                `json:"seed,omitempty"`
	Store               *bool                 `json:"store,omitempty"`
	Temperature         *float64              `json:"temperature,omitempty"`
	TopLogprobs         *int64                `json:"top_logprobs,omitempty"`
	TopP                *float64              `json:"top_p,omitempty"`
	LogitBias           map[string]int64      `json:"logit_bias,omitempty"`
	Metadata            map[string]string     `json:"metadata,omitempty"`
	Modalities          []string              `json:"modalities,omitempty"`
	Audio               *ChatCompletionsAudio `json:"audio,omitempty"`
	ReasoningEffort     string                `json:"reasoning_effort,omitempty"`
	Thinking            *model.ThinkingConfig `json:"thinking,omitempty"`
	ReasoningSplit      *bool                 `json:"reasoning_split,omitempty"`
	ServiceTier         *string               `json:"service_tier,omitempty"`
	Stop                *model.Stop           `json:"stop,omitempty"`
	Stream              *bool                 `json:"stream,omitempty"`
	StreamOptions       *model.StreamOptions  `json:"stream_options,omitempty"`
	ParallelToolCalls   *bool                 `json:"parallel_tool_calls,omitempty"`
	Tools               []ChatCompletionsTool `json:"tools,omitempty"`
	ToolChoice          *model.ToolChoice     `json:"tool_choice,omitempty"`
	ResponseFormat      *model.ResponseFormat `json:"response_format,omitempty"`
	SafetyIdentifier    *string               `json:"safety_identifier,omitempty"`
	// PromptCacheKey mirrors the top-level model field. Only forwarded when
	// the client populated the field on the Chat entrypoint — Responses
	// inbound carries its own ResponsesPromptCacheKey pass-through that
	// stays isolated from this builder.
	PromptCacheKey *string `json:"prompt_cache_key,omitempty"`
	// User is OpenAI's legacy caller-supplied end-user identifier. OpenAI now
	// prefers `safety_identifier` + `prompt_cache_key`, but the field is still
	// accepted for backward compatibility; we forward it when the client sets
	// it so downstreams that key on `user` keep working.
	User *string `json:"user,omitempty"`
	// Verbosity is the gpt-5 detail-level knob ("low" | "medium" | "high").
	Verbosity *string `json:"verbosity,omitempty"`
	// Prediction forwards the OpenAI "predicted outputs" payload verbatim.
	Prediction json.RawMessage `json:"prediction,omitempty"`
	// WebSearchOptions configures the Chat Completions built-in web search
	// tool; kept as raw JSON for schema stability.
	WebSearchOptions json.RawMessage `json:"web_search_options,omitempty"`
}

type ChatCompletionsAudio struct {
	Format string `json:"format,omitempty"`
	Voice  string `json:"voice,omitempty"`
}

func (o *ChatOutbound) TransformRequest(ctx context.Context, request *model.InternalLLMRequest, baseUrl, key string) (*http.Request, error) {
	request.ClearHelpFields()
	request.NormalizeMessages()
	request.FlattenUnsupportedBlocks(model.AlternationProviderOpenAI)

	// developer role is preserved as-is on OpenAI outbound (O-L5). OpenAI
	// 2025+ model spec treats "developer" as the canonical instruction
	// role for reasoning models; the latest Chat Completions API accepts
	// it natively and silently maps "system" to "developer" on reasoning
	// models for backward compatibility. Previously we forced
	// developer → system which worked on gpt-4 / gpt-4o (where the two
	// are interchangeable) but lost the semantic distinction on gpt-5
	// reasoning models. Keep the original role so upstreams that depend
	// on it (and downstreams that replay it) see the caller's intent.
	// Ref: https://platform.openai.com/docs/api-reference/chat

	if request.Stream != nil && *request.Stream {
		if request.StreamOptions == nil {
			request.StreamOptions = &model.StreamOptions{IncludeUsage: true}
		} else if !request.StreamOptions.IncludeUsage {
			request.StreamOptions.IncludeUsage = true
		}
	}

	body, err := json.Marshal(buildChatCompletionsRequest(request))
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	applyOpenAIOrgProjectHeaders(req, request)

	parsedUrl, err := url.Parse(strings.TrimSuffix(baseUrl, "/"))
	if err != nil {
		return nil, fmt.Errorf("failed to parse base url: %w", err)
	}
	parsedUrl.Path = parsedUrl.Path + "/chat/completions"
	req.URL = parsedUrl
	req.Method = http.MethodPost
	return req, nil
}

// applyOpenAIOrgProjectHeaders forwards the optional OpenAI-Organization and
// OpenAI-Project headers from TransformerMetadata. Both headers are scoped
// to multi-org / multi-project deployments where a single API key can hit
// several billing scopes; callers set the metadata keys upstream (in relay
// configuration or per-request overrides) and the outbound transformer
// blindly forwards the values. Empty / whitespace-only values are dropped
// so we don't emit header keys with blank values. O-M7.
// Ref: https://platform.openai.com/docs/api-reference/debugging-requests
func applyOpenAIOrgProjectHeaders(req *http.Request, request *model.InternalLLMRequest) {
	if req == nil || request == nil {
		return
	}
	if org := request.TransformerMetadataValue(model.TransformerMetadataOpenAIOrganization); org != "" {
		req.Header.Set("OpenAI-Organization", org)
	}
	if project := request.TransformerMetadataValue(model.TransformerMetadataOpenAIProject); project != "" {
		req.Header.Set("OpenAI-Project", project)
	}
}

func chatPromptCacheKey(request *model.InternalLLMRequest) *string {
	if request == nil {
		return nil
	}
	if request.PromptCacheKey != nil {
		return request.PromptCacheKey
	}
	key, _ := derivedAnthropicCacheMetadata(request)
	return key
}

func buildChatCompletionsRequest(request *model.InternalLLMRequest) *ChatCompletionsRequest {
	if request == nil {
		return &ChatCompletionsRequest{}
	}
	deepSeekResponsesCompat := request.RawAPIFormat == model.APIFormatOpenAIResponse && isDeepSeekModel(request.Model)
	maxCompletionTokens := request.MaxCompletionTokens
	maxTokens := request.MaxTokens
	reasoningEffort := request.ReasoningEffort
	store := request.Store
	serviceTier := request.ServiceTier
	metadata := request.Metadata
	safetyIdentifier := request.SafetyIdentifier
	promptCacheKey := chatPromptCacheKey(request)
	thinking := request.Thinking
	messages := request.Messages
	if deepSeekResponsesCompat {
		// DeepSeek's Chat API accepts max_tokens and requires assistant history
		// messages to contain visible content or tool calls. Responses reasoning
		// items become reasoning-only assistant messages internally, so omit them
		// from this provider-specific wire payload.
		if maxTokens == nil {
			maxTokens = maxCompletionTokens
		}
		maxCompletionTokens = nil
		reasoningEffort = ""
		store = nil
		serviceTier = nil
		metadata = nil
		safetyIdentifier = nil
		promptCacheKey = nil
		if thinking == nil && request.ReasoningEffort != "" {
			thinkingType := "enabled"
			if strings.EqualFold(strings.TrimSpace(request.ReasoningEffort), "none") {
				thinkingType = "disabled"
			}
			thinking = &model.ThinkingConfig{Type: thinkingType}
		}
		messages = deepSeekChatMessages(messages)
	}

	result := &ChatCompletionsRequest{
		Messages:            messages,
		Model:               request.Model,
		FrequencyPenalty:    request.FrequencyPenalty,
		Logprobs:            request.Logprobs,
		MaxCompletionTokens: maxCompletionTokens,
		MaxTokens:           maxTokens,
		PresencePenalty:     request.PresencePenalty,
		Seed:                request.Seed,
		Store:               store,
		Temperature:         request.Temperature,
		TopLogprobs:         request.TopLogprobs,
		TopP:                request.TopP,
		LogitBias:           request.LogitBias,
		Metadata:            metadata,
		Modalities:          request.Modalities,
		ReasoningEffort:     reasoningEffort,
		Thinking:            thinking,
		ReasoningSplit:      request.ReasoningSplit,
		ServiceTier:         serviceTier,
		Stop:                request.Stop,
		Stream:              request.Stream,
		StreamOptions:       request.StreamOptions,
		ParallelToolCalls:   request.ParallelToolCalls,
		Tools:               convertToolsToChatCompletions(request.Tools),
		ToolChoice:          request.ToolChoice,
		ResponseFormat:      request.ResponseFormat,
		SafetyIdentifier:    safetyIdentifier,
		PromptCacheKey:      promptCacheKey,
		User:                request.User,
		Verbosity:           request.Verbosity,
		Prediction:          request.Prediction,
		WebSearchOptions:    request.WebSearchOptions,
	}

	// Anthropic 的 stop_sequences 语义比 Chat Completions 的 stop 更宽松
	// （Anthropic 只在模型明确输出该序列时截断，而 Chat 在子串层面进行匹配），
	// 透传会导致 Chat 上游在第一个分段（如 "\n\n"）就提前触发 finish_reason="stop"，
	// 表现为"回一句话就中断"。仅在 inbound 是 Anthropic 时剥离。
	if request.RawAPIFormat == model.APIFormatAnthropicMessage {
		result.Stop = nil
	}

	// 推理系列模型（o1/o3/o4/gpt-5）把 max_tokens 弃用，改为 max_completion_tokens。
	// 若上层只填了 MaxTokens，将其重定向到新字段，避免新模型在还没输出可见内容前
	// 就被 max_tokens 限额打断。
	if result.MaxCompletionTokens == nil && result.MaxTokens != nil && isReasoningChatModel(result.Model) {
		result.MaxCompletionTokens = result.MaxTokens
		result.MaxTokens = nil
	}

	if request.Audio != nil {
		result.Audio = &ChatCompletionsAudio{
			Format: request.Audio.Format,
			Voice:  request.Audio.Voice,
		}
	}

	return result
}

func isDeepSeekModel(modelName string) bool {
	modelName = strings.ToLower(strings.TrimSpace(modelName))
	if slash := strings.LastIndexByte(modelName, '/'); slash >= 0 {
		modelName = modelName[slash+1:]
	}
	return strings.HasPrefix(modelName, "deepseek-")
}

func deepSeekChatMessages(messages []model.Message) []model.Message {
	filtered := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		if message.Role == "assistant" && messageHasOnlyReasoning(&message) {
			continue
		}
		filtered = append(filtered, message)
	}
	return filtered
}

func messageHasOnlyReasoning(message *model.Message) bool {
	if message == nil || message.Role != "assistant" || len(message.ToolCalls) > 0 {
		return false
	}
	if message.Content.Content != nil && strings.TrimSpace(*message.Content.Content) != "" {
		return false
	}
	for _, part := range message.Content.MultipleContent {
		if part.Type != "text" || (part.Text != nil && strings.TrimSpace(*part.Text) != "") {
			return false
		}
	}
	return (message.ReasoningContent != nil && *message.ReasoningContent != "") ||
		(message.Reasoning != nil && *message.Reasoning != "") || len(message.ReasoningBlocks) > 0 ||
		len(message.RedactedThinkingBlocks) > 0 || (message.ReasoningSignature != nil && *message.ReasoningSignature != "")
}

// isReasoningChatModel 判断 Chat Completions 端的模型是否属于推理系列
// （o1/o3/o4/gpt-5）。这些系列只接受 max_completion_tokens，旧的 max_tokens
// 会被 OpenAI 官方 API 直接拒绝或悄悄忽略。
func isReasoningChatModel(modelName string) bool {
	name := strings.ToLower(strings.TrimSpace(modelName))
	if name == "" {
		return false
	}
	if strings.HasPrefix(name, "o1") || strings.HasPrefix(name, "o3") || strings.HasPrefix(name, "o4") {
		return true
	}
	if strings.HasPrefix(name, "gpt-5") {
		return true
	}
	return false
}

func convertToolsToChatCompletions(tools []model.Tool) []ChatCompletionsTool {
	result := make([]ChatCompletionsTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "function" {
			continue
		}
		result = append(result, ChatCompletionsTool{
			Type:     "function",
			Function: tool.Function,
		})
	}
	return result
}

func (o *ChatOutbound) TransformResponse(ctx context.Context, response *http.Response) (*model.InternalLLMResponse, error) {
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if len(body) == 0 {
		return nil, fmt.Errorf("response body is empty")
	}

	normalizedBody, err := normalizeChatReasoningBody(body)
	if err != nil {
		return nil, err
	}
	var resp model.InternalLLMResponse
	if err := json.Unmarshal(normalizedBody, &resp); err != nil {
		if bodyLooksLikeSSE(body) {
			return aggregateChatSSEBody(body)
		}
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	return &resp, nil
}

func bodyLooksLikeSSE(body []byte) bool {
	trimmed := bytes.TrimSpace(body)
	return bytes.Contains(trimmed, []byte("data:")) &&
		(bytes.HasPrefix(trimmed, []byte("data:")) || bytes.Contains(trimmed, []byte("\ndata:")))
}

func aggregateChatSSEBody(body []byte) (*model.InternalLLMResponse, error) {
	aggregator := model.StreamAggregator{}
	for event, err := range sse.Read(bytes.NewReader(body), &sse.ReadConfig{MaxEventSize: 32 * 1024 * 1024}) {
		if err != nil {
			return nil, fmt.Errorf("failed to parse upstream SSE body: %w", err)
		}
		data := strings.TrimSpace(event.Data)
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk model.InternalLLMResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("failed to unmarshal upstream SSE chunk: %w", err)
		}
		aggregator.Add(&chunk)
	}
	response := aggregator.BuildAndReset()
	if response == nil || !response.IsChatResponse() {
		return nil, fmt.Errorf("upstream SSE body did not contain chat completion chunks")
	}
	return response, nil
}

func (o *ChatOutbound) TransformStream(ctx context.Context, eventData []byte) (*model.InternalLLMResponse, error) {
	if bytes.HasPrefix(eventData, []byte("[DONE]")) {
		return &model.InternalLLMResponse{
			Object: "[DONE]",
		}, nil
	}

	var errCheck struct {
		Error *model.ErrorDetail `json:"error"`
	}
	if err := json.Unmarshal(eventData, &errCheck); err == nil && errCheck.Error != nil {
		return nil, &model.ResponseError{
			Detail: *errCheck.Error,
		}
	}

	normalizedEvent, err := normalizeChatReasoningBody(eventData)
	if err != nil {
		return nil, err
	}
	var resp model.InternalLLMResponse
	if err := json.Unmarshal(normalizedEvent, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal stream chunk: %w", err)
	}
	return &resp, nil
}

func normalizeChatReasoningBody(body []byte) ([]byte, error) {
	var generic map[string]any
	if err := json.Unmarshal(body, &generic); err != nil {
		return append([]byte(nil), body...), nil
	}
	choices, _ := generic["choices"].([]any)
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			continue
		}
		for _, field := range []string{"message", "delta"} {
			message, ok := choice[field].(map[string]any)
			if !ok {
				continue
			}
			if reasoning, exists := message["reasoning"]; exists {
				if text := chatReasoningText(reasoning); text != "" {
					message["reasoning"] = text
				} else {
					delete(message, "reasoning")
				}
			}
			if details, exists := message["reasoning_details"]; exists {
				if text := chatReasoningText(details); text != "" {
					if _, hasReasoningContent := message["reasoning_content"]; !hasReasoningContent {
						message["reasoning_content"] = text
					}
				}
				delete(message, "reasoning_details")
			}
		}
	}
	return json.Marshal(generic)
}

func chatReasoningText(value any) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case []any:
		var parts []string
		for _, item := range typed {
			if text := chatReasoningText(item); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n\n")
	case map[string]any:
		for _, key := range []string{"content", "text", "summary"} {
			if text := chatReasoningText(typed[key]); text != "" {
				return text
			}
		}
		return ""
	default:
		return ""
	}
}

func (o *ChatOutbound) TransformStreamEvent(ctx context.Context, eventData []byte) ([]model.StreamEvent, error) {
	stream, err := o.TransformStream(ctx, eventData)
	if err != nil {
		return nil, err
	}
	return model.StreamEventsFromInternalResponse(stream), nil
}
