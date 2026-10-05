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
	if request.ReasoningEffort == "" && request.ReasoningBudget == nil {
		return
	}

	param := strings.TrimSpace(config.ReasoningParam)
	reasoningEnabled := !codexReasoningExplicitlyDisabled(request.ReasoningEffort)
	switch param {
	case "thinking":
		request.Thinking = &transformerModel.ThinkingConfig{Type: codexReasoningState(reasoningEnabled)}
	case "enable_thinking":
		enabled := reasoningEnabled
		request.EnableThinking = &enabled
	case "reasoning_split":
		enabled := reasoningEnabled
		request.ReasoningSplit = &enabled
		request.ReasoningEffort = ""
		request.ReasoningEffortObject = nil
	case "none":
		request.ReasoningEffort = ""
		request.ReasoningEffortObject = nil
	case "", "reasoning_effort":
		applyCodexReasoningEffort(config, request)
	default:
		return
	}
	if param != "" && param != "reasoning_effort" {
		request.ReasoningEffort = ""
		request.ReasoningEffortObject = nil
	}
}

func applyCodexReasoningEffort(config *dbmodel.CodexCompatConfig, request *transformerModel.InternalLLMRequest) {
	effortParam := strings.TrimSpace(config.EffortParam)
	if effortParam == "" {
		effortParam = "reasoning_effort"
	}
	if effortParam == "none" {
		request.ReasoningEffort = ""
		request.ReasoningEffortObject = nil
		return
	}

	explicitDisabled := codexReasoningExplicitlyDisabled(request.ReasoningEffort)
	mapped := mapCodexReasoningEffort(
		request.ReasoningEffort,
		strings.TrimSpace(config.EffortValueMode),
		codexModelReasoningLevels(config, request.Model),
	)
	request.ReasoningEffort = ""
	request.ReasoningEffortObject = nil
	if explicitDisabled && effortParam == "reasoning.effort" {
		request.ReasoningEffortObject = &transformerModel.ReasoningEffortObject{Effort: "none"}
		return
	}
	if mapped == "" {
		return
	}
	if effortParam == "reasoning.effort" {
		request.ReasoningEffortObject = &transformerModel.ReasoningEffortObject{Effort: mapped}
		return
	}
	if effortParam == "reasoning_effort" {
		request.ReasoningEffort = mapped
	}
}

func codexReasoningExplicitlyDisabled(effort string) bool {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "none", "off", "disabled":
		return true
	default:
		return false
	}
}

func codexReasoningState(enabled bool) string {
	if enabled {
		return "enabled"
	}
	return "disabled"
}

func codexModelReasoningLevels(config *dbmodel.CodexCompatConfig, modelName string) []string {
	if config == nil || len(config.ModelReasoningLevels) == 0 {
		return nil
	}
	modelName = strings.ToLower(strings.TrimSpace(modelName))
	if levels, ok := config.ModelReasoningLevels[modelName]; ok {
		return levels
	}
	if slash := strings.LastIndexByte(modelName, '/'); slash >= 0 {
		if levels, ok := config.ModelReasoningLevels[modelName[slash+1:]]; ok {
			return levels
		}
	}
	return nil
}

func mapCodexReasoningEffort(effort, mode string, levels []string) string {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if codexReasoningExplicitlyDisabled(effort) {
		return ""
	}
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "passthrough":
		switch effort {
		case "minimal", "low", "medium", "high", "xhigh", "max", "ultra":
			return effort
		default:
			return ""
		}
	case "deepseek":
		switch effort {
		case "max", "xhigh", "ultra":
			return "max"
		default:
			return "high"
		}
	case "low_high":
		switch effort {
		case "minimal", "low":
			return "low"
		default:
			return "high"
		}
	case "openrouter":
		switch effort {
		case "max", "xhigh", "ultra":
			return "xhigh"
		case "high", "medium", "low", "minimal":
			return effort
		default:
			return ""
		}
	case "zen":
		return clampZenReasoningEffort(effort, levels)
	default:
		return ""
	}
}

func clampZenReasoningEffort(effort string, levels []string) string {
	requested, ok := zenReasoningEffortRank(effort)
	if !ok {
		return ""
	}
	closestAtOrAboveRank := -1
	closestAtOrAbove := ""
	highestBelowRank := -1
	highestBelow := ""
	anyValid := false
	anyValidLevel := ""
	anyValidRank := -1

	for _, level := range levels {
		rank, ok := zenReasoningEffortRank(level)
		if !ok {
			continue
		}
		normalized := strings.ToLower(strings.TrimSpace(level))
		if !anyValid || rank < anyValidRank {
			anyValid = true
			anyValidRank = rank
			anyValidLevel = normalized
		}
		if rank >= requested && (closestAtOrAboveRank < 0 || rank < closestAtOrAboveRank) {
			closestAtOrAboveRank = rank
			closestAtOrAbove = normalized
		}
		if rank < requested && rank > highestBelowRank {
			highestBelowRank = rank
			highestBelow = normalized
		}
	}
	if closestAtOrAbove != "" {
		return closestAtOrAbove
	}
	if highestBelow != "" {
		return highestBelow
	}
	if anyValid {
		return anyValidLevel
	}
	return ""
}

func zenReasoningEffortRank(effort string) (int, bool) {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal":
		return 0, true
	case "low":
		return 1, true
	case "medium":
		return 2, true
	case "high":
		return 3, true
	case "xhigh":
		return 4, true
	case "max":
		return 5, true
	case "ultra":
		return 6, true
	default:
		return 0, false
	}
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
