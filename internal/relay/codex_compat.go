package relay

import (
	"encoding/json"
	"net/http"
	"net/url"
	"strings"

	dbmodel "github.com/bestruirui/octopus/internal/model"
	transformerModel "github.com/bestruirui/octopus/internal/transformer/model"
)

func applyCodexCompat(channel *dbmodel.Channel, headers http.Header, request *transformerModel.InternalLLMRequest) {
	if channel == nil || request == nil || request.RawAPIFormat != transformerModel.APIFormatOpenAIResponse {
		return
	}

	applyCodexPromptCacheRouting(channel, headers, request)
	applyCodexReasoningParam(channel.CodexCompat, request)
	if moonshotHostRequiresRefSiblingAllOf(channel.GetBaseUrl()) {
		wrapRefSiblingsInChatTools(request)
	}
}

func applyCodexPromptCacheRouting(channel *dbmodel.Channel, headers http.Header, request *transformerModel.InternalLLMRequest) {
	mode := ""
	if channel.CodexCompat != nil {
		mode = strings.TrimSpace(channel.CodexCompat.PromptCacheRouting)
	}
	if mode != "enabled" && !(mode == "auto" && promptCacheKeyCompatibleUpstream(channel.GetBaseUrl())) {
		return
	}

	if request.ResponsesPromptCacheKey != nil && strings.TrimSpace(*request.ResponsesPromptCacheKey) != "" {
		key := strings.TrimSpace(*request.ResponsesPromptCacheKey)
		request.PromptCacheKey = &key
		return
	}
	if sessionID := stableCodexSessionID(headers, request); sessionID != "" {
		request.PromptCacheKey = &sessionID
	}
}

func promptCacheKeyCompatibleUpstream(rawBaseURL string) bool {
	parsed, err := url.Parse(strings.TrimSpace(rawBaseURL))
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	path := strings.TrimSuffix(parsed.Path, "/")
	if host == "api.openai.com" {
		return true
	}
	if host == "api.kimi.com" {
		return path == "/coding" || strings.HasPrefix(path, "/coding/")
	}
	return false
}

func stableCodexSessionID(headers http.Header, request *transformerModel.InternalLLMRequest) string {
	if request != nil {
		if userID := strings.TrimSpace(request.Metadata["user_id"]); userID != "" {
			if sessionID := sessionIDFromUserID(userID); sessionID != "" {
				return sessionID
			}
		}
		if sessionID := strings.TrimSpace(request.Metadata["session_id"]); sessionID != "" {
			return sessionID
		}
	}
	if headers != nil {
		for _, name := range []string{"session_id", "x-session-id"} {
			if sessionID := strings.TrimSpace(headers.Get(name)); sessionID != "" {
				return sessionID
			}
		}
	}
	return ""
}

func sessionIDFromUserID(userID string) string {
	if index := strings.LastIndex(userID, "_session_"); index >= 0 {
		return strings.TrimSpace(userID[index+len("_session_"):])
	}
	return ""
}

func applyCodexReasoningParam(config *dbmodel.CodexCompatConfig, request *transformerModel.InternalLLMRequest) {
	if config == nil || request == nil {
		return
	}
	param := strings.TrimSpace(config.ReasoningParam)
	if param == "" || param == "reasoning_effort" {
		return
	}
	if request.ReasoningEffort == "" && request.ReasoningBudget == nil {
		return
	}

	switch param {
	case "thinking":
		request.Thinking = &transformerModel.ThinkingConfig{Type: "enabled"}
	case "enable_thinking":
		enabled := true
		request.EnableThinking = &enabled
	case "reasoning_split":
		enabled := true
		request.ReasoningSplit = &enabled
	default:
		return
	}
	request.ReasoningEffort = ""
}

func moonshotHostRequiresRefSiblingAllOf(rawBaseURL string) bool {
	parsed, err := url.Parse(strings.TrimSpace(rawBaseURL))
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	for _, suffix := range []string{"moonshot.cn", "moonshot.ai", "kimi.com"} {
		if host == suffix || strings.HasSuffix(host, "."+suffix) {
			return true
		}
	}
	return false
}

func wrapRefSiblingsInChatTools(request *transformerModel.InternalLLMRequest) {
	if request == nil {
		return
	}
	for toolIndex := range request.Tools {
		parameters := request.Tools[toolIndex].Function.Parameters
		if len(parameters) == 0 {
			continue
		}
		var schema any
		if err := json.Unmarshal(parameters, &schema); err != nil {
			continue
		}
		if wrapRefSiblings(schema) == 0 {
			continue
		}
		if rewritten, err := json.Marshal(schema); err == nil {
			request.Tools[toolIndex].Function.Parameters = rewritten
		}
	}
}

func wrapRefSiblings(schema any) int {
	object, ok := schema.(map[string]any)
	if !ok {
		return 0
	}

	rewritten := 0
	if len(object) > 1 {
		if ref, exists := object["$ref"]; exists {
			if _, isString := ref.(string); isString {
				delete(object, "$ref")
				branches, _ := object["allOf"].([]any)
				object["allOf"] = append(branches, map[string]any{"$ref": ref})
				rewritten++
			}
		}
	}

	for key, value := range object {
		switch key {
		case "properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies":
			if entries, ok := value.(map[string]any); ok {
				for _, child := range entries {
					rewritten += wrapRefSiblings(child)
				}
			}
		case "allOf", "anyOf", "oneOf", "prefixItems":
			if entries, ok := value.([]any); ok {
				for _, child := range entries {
					rewritten += wrapRefSiblings(child)
				}
			}
		case "items", "additionalItems", "unevaluatedItems", "contains", "additionalProperties", "unevaluatedProperties", "propertyNames", "not", "if", "then", "else", "contentSchema":
			if entries, ok := value.([]any); ok {
				for _, child := range entries {
					rewritten += wrapRefSiblings(child)
				}
			} else {
				rewritten += wrapRefSiblings(value)
			}
		}
	}
	return rewritten
}
