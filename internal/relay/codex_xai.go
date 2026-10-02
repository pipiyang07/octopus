package relay

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"strconv"
	"strings"

	dbmodel "github.com/bestruirui/octopus/internal/model"
	"github.com/bestruirui/octopus/internal/transformer/model"
	"github.com/bestruirui/octopus/internal/transformer/outbound"
)

var xAIResponsesToolTypes = map[string]struct{}{
	"function":           {},
	"web_search":         {},
	"x_search":           {},
	"image_generation":   {},
	"collections_search": {},
	"file_search":        {},
	"code_execution":     {},
	"code_interpreter":   {},
	"mcp":                {},
	"shell":              {},
}

func needsXAIResponsesCompat(channel *dbmodel.Channel) bool {
	if channel == nil || channel.Type != outbound.OutboundTypeOpenAIResponse {
		return false
	}
	parsed, err := url.Parse(strings.TrimSpace(channel.GetBaseUrl()))
	if err != nil {
		return false
	}
	return strings.EqualFold(parsed.Hostname(), "api.x.ai")
}

func prepareXAIResponsesRequestBody(raw []byte) ([]byte, map[string]model.CodexToolSpec, error) {
	var body map[string]any
	if err := decodeJSONUseNumber(raw, &body); err != nil {
		return nil, nil, fmt.Errorf("failed to decode responses request: %w", err)
	}

	restoreMap := buildXAINamespaceRestoreMap(body)
	changed := false
	changed = flattenXAIRequestNamespaces(body, restoreMap) || changed
	changed = sanitizeXAIResponsesRequest(body) || changed

	if !changed {
		return append([]byte(nil), raw...), restoreMap, nil
	}
	rewritten, err := json.Marshal(body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to encode xAI responses request: %w", err)
	}
	return rewritten, restoreMap, nil
}

func buildXAINamespaceRestoreMap(body map[string]any) map[string]model.CodexToolSpec {
	result := make(map[string]model.CodexToolSpec)
	tools, _ := body["tools"].([]any)
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok || stringValueMap(tool, "type") != "namespace" {
			continue
		}
		namespace := strings.TrimSpace(stringValueMap(tool, "name"))
		if namespace == "" {
			continue
		}
		for _, rawChild := range xAINamespaceChildren(tool) {
			child, ok := rawChild.(map[string]any)
			if !ok || stringValueMap(child, "type") != "function" {
				continue
			}
			name := strings.TrimSpace(stringValueMap(child, "name"))
			if name == "" {
				continue
			}
			flat := model.FlattenCodexNamespaceToolName(namespace, name)
			if _, exists := result[flat]; exists {
				continue
			}
			result[flat] = model.CodexToolSpec{
				Kind:      model.CodexToolKindNamespace,
				Name:      name,
				Namespace: namespace,
			}
		}
	}
	return result
}

func flattenXAIRequestNamespaces(body map[string]any, restoreMap map[string]model.CodexToolSpec) bool {
	rawTools, ok := body["tools"].([]any)
	if !ok {
		return false
	}
	hasNamespace := false
	for _, rawTool := range rawTools {
		if tool, ok := rawTool.(map[string]any); ok && stringValueMap(tool, "type") == "namespace" {
			hasNamespace = true
			break
		}
	}
	if !hasNamespace {
		return false
	}

	flattened := make([]any, 0, len(rawTools))
	seen := make(map[string]struct{})
	for _, rawTool := range rawTools {
		tool, ok := rawTool.(map[string]any)
		if !ok || stringValueMap(tool, "type") != "namespace" {
			flattened = append(flattened, rawTool)
			continue
		}
		namespace := strings.TrimSpace(stringValueMap(tool, "name"))
		for _, rawChild := range xAINamespaceChildren(tool) {
			child, ok := rawChild.(map[string]any)
			if !ok || stringValueMap(child, "type") != "function" {
				continue
			}
			name := strings.TrimSpace(stringValueMap(child, "name"))
			if name == "" {
				continue
			}
			flat := model.FlattenCodexNamespaceToolName(namespace, name)
			if _, duplicate := seen[flat]; duplicate {
				continue
			}
			seen[flat] = struct{}{}
			lifted := make(map[string]any, len(child))
			for key, value := range child {
				lifted[key] = value
			}
			lifted["name"] = flat
			flattened = append(flattened, lifted)
		}
	}
	body["tools"] = flattened

	rewriteXAINamespaceQualifiedCalls(body["input"], restoreMap)
	if choice, ok := body["tool_choice"].(map[string]any); ok {
		if stringValueMap(choice, "type") == "namespace" {
			body["tool_choice"] = "auto"
		} else {
			rewriteXAINamespaceQualifiedCall(choice, restoreMap)
		}
	}
	return true
}

func xAINamespaceChildren(tool map[string]any) []any {
	children, _ := tool["tools"].([]any)
	if children == nil {
		children, _ = tool["children"].([]any)
	}
	return children
}

func rewriteXAINamespaceQualifiedCalls(value any, restoreMap map[string]model.CodexToolSpec) {
	switch typed := value.(type) {
	case []any:
		for _, item := range typed {
			rewriteXAINamespaceQualifiedCalls(item, restoreMap)
		}
	case map[string]any:
		if stringValueMap(typed, "type") == "function_call" {
			rewriteXAINamespaceQualifiedCall(typed, restoreMap)
			return
		}
		for _, child := range typed {
			rewriteXAINamespaceQualifiedCalls(child, restoreMap)
		}
	}
}

func rewriteXAINamespaceQualifiedCall(item map[string]any, restoreMap map[string]model.CodexToolSpec) bool {
	namespace := strings.TrimSpace(stringValueMap(item, "namespace"))
	name := strings.TrimSpace(stringValueMap(item, "name"))
	if namespace == "" || name == "" {
		return false
	}
	flat := model.FlattenCodexNamespaceToolName(namespace, name)
	spec, ok := restoreMap[flat]
	if !ok || spec.Namespace != namespace || spec.Name != name {
		return false
	}
	item["name"] = flat
	delete(item, "namespace")
	return true
}

func sanitizeXAIResponsesRequest(body map[string]any) bool {
	changed := false
	for _, field := range []string{"prompt_cache_retention", "safety_identifier"} {
		if _, exists := body[field]; exists {
			delete(body, field)
			changed = true
		}
	}
	if xAIRequestTargetsGrok45(body) {
		for _, field := range []string{"presence_penalty", "presencePenalty", "frequency_penalty", "frequencyPenalty", "stop"} {
			if _, exists := body[field]; exists {
				delete(body, field)
				changed = true
			}
		}
	}
	changed = removeXAIFieldRecursive(body, "external_web_access") || changed
	changed = promoteXAIAdditionalTools(body) || changed
	changed = stripXAINullReasoningContent(body) || changed
	changed = filterXAIUnsupportedTools(body) || changed
	changed = normalizeXAIFunctionToolSchemas(body) || changed
	return changed
}

func xAIRequestTargetsGrok45(body map[string]any) bool {
	modelName := strings.TrimSpace(stringValueMap(body, "model"))
	if index := strings.LastIndex(modelName, "/"); index >= 0 {
		modelName = strings.TrimSpace(modelName[index+1:])
	}
	return strings.EqualFold(modelName, "grok-4.5")
}

func removeXAIFieldRecursive(value any, field string) bool {
	switch typed := value.(type) {
	case map[string]any:
		changed := false
		if _, exists := typed[field]; exists {
			delete(typed, field)
			changed = true
		}
		for _, child := range typed {
			changed = removeXAIFieldRecursive(child, field) || changed
		}
		return changed
	case []any:
		changed := false
		for _, child := range typed {
			changed = removeXAIFieldRecursive(child, field) || changed
		}
		return changed
	default:
		return false
	}
}

func promoteXAIAdditionalTools(body map[string]any) bool {
	input, ok := body["input"].([]any)
	if !ok {
		return false
	}
	hasCarrier := false
	for _, rawItem := range input {
		if item, ok := rawItem.(map[string]any); ok && stringValueMap(item, "type") == "additional_tools" {
			hasCarrier = true
			break
		}
	}
	if !hasCarrier {
		return false
	}

	merged, _ := body["tools"].([]any)
	seen := make(map[string]struct{})
	for _, rawTool := range merged {
		seen[xAIToolDedupKey(rawTool)] = struct{}{}
	}
	filteredInput := make([]any, 0, len(input))
	promoted := false
	for _, rawItem := range input {
		item, ok := rawItem.(map[string]any)
		if !ok || stringValueMap(item, "type") != "additional_tools" {
			filteredInput = append(filteredInput, rawItem)
			continue
		}
		carrierTools, _ := item["tools"].([]any)
		for _, rawTool := range carrierTools {
			key := xAIToolDedupKey(rawTool)
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}
			merged = append(merged, rawTool)
			promoted = true
		}
	}
	body["input"] = filteredInput
	if promoted {
		body["tools"] = merged
	}
	return true
}

func xAIToolDedupKey(rawTool any) string {
	tool, ok := rawTool.(map[string]any)
	if !ok {
		return fmt.Sprint(rawTool)
	}
	toolType := strings.TrimSpace(stringValueMap(tool, "type"))
	if toolType == "mcp" {
		return "type:mcp\x00server_label:" + strings.TrimSpace(stringValueMap(tool, "server_label"))
	}
	if toolType != "" {
		name := strings.TrimSpace(stringValueMap(tool, "name"))
		if name != "" {
			return "type:" + toolType + "\x00name:" + name
		}
	}
	raw, err := json.Marshal(tool)
	if err != nil {
		return fmt.Sprint(tool)
	}
	return "json:" + string(raw)
}

func stripXAINullReasoningContent(body map[string]any) bool {
	input, ok := body["input"].([]any)
	if !ok {
		return false
	}
	changed := false
	for _, rawItem := range input {
		item, ok := rawItem.(map[string]any)
		if !ok || stringValueMap(item, "type") != "reasoning" {
			continue
		}
		if value, exists := item["content"]; exists && value == nil {
			delete(item, "content")
			changed = true
		}
	}
	return changed
}

func filterXAIUnsupportedTools(body map[string]any) bool {
	tools, ok := body["tools"].([]any)
	if !ok {
		return false
	}
	filtered := make([]any, 0, len(tools))
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			continue
		}
		if _, supported := xAIResponsesToolTypes[strings.TrimSpace(stringValueMap(tool, "type"))]; supported {
			filtered = append(filtered, rawTool)
		}
	}
	if len(filtered) == len(tools) {
		return false
	}
	if len(filtered) == 0 {
		delete(body, "tools")
	} else {
		body["tools"] = filtered
	}
	if shouldDropXAIToolChoice(body, filtered) {
		delete(body, "tool_choice")
	}
	return true
}

func shouldDropXAIToolChoice(body map[string]any, tools []any) bool {
	choice, ok := body["tool_choice"].(map[string]any)
	if !ok {
		return false
	}
	if len(tools) == 0 {
		return true
	}
	choiceType := strings.TrimSpace(stringValueMap(choice, "type"))
	if _, supported := xAIResponsesToolTypes[choiceType]; !supported {
		return true
	}
	if choiceType != "function" {
		return false
	}
	name := strings.TrimSpace(stringValueMap(choice, "name"))
	if name == "" {
		if function, ok := choice["function"].(map[string]any); ok {
			name = strings.TrimSpace(stringValueMap(function, "name"))
		}
	}
	if name == "" {
		return false
	}
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok || stringValueMap(tool, "type") != "function" {
			continue
		}
		toolName := strings.TrimSpace(stringValueMap(tool, "name"))
		if toolName == "" {
			if function, ok := tool["function"].(map[string]any); ok {
				toolName = strings.TrimSpace(stringValueMap(function, "name"))
			}
		}
		if toolName == name {
			return false
		}
	}
	return true
}

func normalizeXAIFunctionToolSchemas(body map[string]any) bool {
	tools, ok := body["tools"].([]any)
	if !ok {
		return false
	}
	changed := false
	for _, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok || stringValueMap(tool, "type") != "function" {
			continue
		}
		params, paramsExists := tool["parameters"]
		if !paramsExists {
			if function, ok := tool["function"].(map[string]any); ok {
				params, paramsExists = function["parameters"]
				if simplified := simplifyXAIParameters(params, paramsExists); simplified != nil {
					function["parameters"] = simplified
					changed = true
				}
			}
			continue
		}
		if simplified := simplifyXAIParameters(params, paramsExists); simplified != nil {
			tool["parameters"] = simplified
			changed = true
		}
	}
	return changed
}

func simplifyXAIParameters(params any, exists bool) any {
	if !exists || params == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": true}
	}
	schema, ok := params.(map[string]any)
	if !ok || len(schema) == 0 {
		return map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": true}
	}
	for _, unionKey := range []string{"oneOf", "anyOf"} {
		branches, _ := schema[unionKey].([]any)
		hasNonObject := false
		for _, rawBranch := range branches {
			branch, ok := rawBranch.(map[string]any)
			if !ok || stringValueMap(branch, "type") != "object" {
				hasNonObject = true
				break
			}
		}
		if hasNonObject {
			return flattenXAIUnionBranches(branches)
		}
	}
	if stringValueMap(schema, "type") != "object" {
		result := make(map[string]any, len(schema))
		for key, value := range schema {
			result[key] = value
		}
		result["type"] = "object"
		if _, exists := result["properties"]; !exists {
			result["properties"] = map[string]any{}
		}
		return result
	}
	return nil
}

func flattenXAIUnionBranches(branches []any) any {
	objectBranches := make([]map[string]any, 0, len(branches))
	for _, rawBranch := range branches {
		branch, ok := rawBranch.(map[string]any)
		if ok && stringValueMap(branch, "type") == "object" {
			objectBranches = append(objectBranches, branch)
		}
	}
	if len(objectBranches) == 0 {
		return map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": true}
	}
	if len(objectBranches) == 1 {
		result := make(map[string]any, len(objectBranches[0]))
		for key, value := range objectBranches[0] {
			result[key] = value
		}
		result["type"] = "object"
		if _, exists := result["properties"]; !exists {
			result["properties"] = map[string]any{}
		}
		return result
	}
	properties := make(map[string]any)
	var required []any
	requiredInitialized := false
	for _, branch := range objectBranches {
		if branchProperties, ok := branch["properties"].(map[string]any); ok {
			for key, value := range branchProperties {
				if _, exists := properties[key]; !exists {
					properties[key] = value
				}
			}
		}
		branchRequired, _ := branch["required"].([]any)
		if !requiredInitialized {
			required = append([]any(nil), branchRequired...)
			requiredInitialized = true
			continue
		}
		next := required[:0]
		for _, item := range required {
			for _, candidate := range branchRequired {
				if fmt.Sprint(item) == fmt.Sprint(candidate) {
					next = append(next, item)
					break
				}
			}
		}
		required = next
	}
	result := map[string]any{"type": "object", "properties": properties}
	if len(required) > 0 {
		result["required"] = required
	}
	return result
}

func decodeJSONUseNumber(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	return decoder.Decode(target)
}

func restoreXAIResponsesBody(body []byte, restoreMap map[string]model.CodexToolSpec) []byte {
	var value any
	if err := decodeJSONUseNumber(body, &value); err != nil {
		return body
	}
	changed := restoreXAIResponseValue(value, restoreMap)
	if !changed {
		return body
	}
	rewritten, err := json.Marshal(value)
	if err != nil {
		return body
	}
	return rewritten
}

func restoreXAIResponseValue(value any, restoreMap map[string]model.CodexToolSpec) bool {
	switch typed := value.(type) {
	case []any:
		changed := false
		for _, item := range typed {
			changed = restoreXAIResponseValue(item, restoreMap) || changed
		}
		return changed
	case map[string]any:
		changed := false
		if stringValueMap(typed, "type") == "function_call" {
			if name := stringValueMap(typed, "name"); name != "" {
				if spec, ok := restoreMap[name]; ok {
					typed["name"] = spec.Name
					typed["namespace"] = spec.Namespace
					changed = true
				}
			}
		}
		if normalizeXAIArgumentsValue(typed) {
			changed = true
		}
		for _, child := range typed {
			changed = restoreXAIResponseValue(child, restoreMap) || changed
		}
		return changed
	default:
		return false
	}
}

func normalizeXAIArgumentsValue(object map[string]any) bool {
	eventType := stringValueMap(object, "type")
	if eventType == "response.function_call_arguments.delta" {
		return false
	}
	if eventType != "response.function_call_arguments.done" && eventType != "function_call" {
		return false
	}
	arguments, ok := object["arguments"].(string)
	if !ok || strings.TrimSpace(arguments) == "" {
		return false
	}
	var parsed any
	if err := decodeJSONUseNumber([]byte(arguments), &parsed); err != nil {
		return false
	}
	parsed, changed := rewriteXAIWholeNumberFloats(parsed)
	if !changed {
		return false
	}
	rewritten, err := json.Marshal(parsed)
	if err != nil {
		return false
	}
	object["arguments"] = string(rewritten)
	return true
}

func rewriteXAIWholeNumberFloats(value any) (any, bool) {
	switch typed := value.(type) {
	case json.Number:
		text := typed.String()
		if !strings.ContainsAny(text, ".eE") {
			return typed, false
		}
		number, err := typed.Float64()
		if err != nil || math.IsInf(number, 0) || math.IsNaN(number) || math.Trunc(number) != number {
			return typed, false
		}
		if number >= math.MinInt64 && number <= math.MaxInt64 && float64(int64(number)) == number {
			return json.Number(strconv.FormatInt(int64(number), 10)), true
		}
		return typed, false
	case []any:
		changed := false
		for index, item := range typed {
			rewritten, itemChanged := rewriteXAIWholeNumberFloats(item)
			if itemChanged {
				typed[index] = rewritten
				changed = true
			}
		}
		return typed, changed
	case map[string]any:
		changed := false
		for key, child := range typed {
			rewritten, childChanged := rewriteXAIWholeNumberFloats(child)
			if childChanged {
				typed[key] = rewritten
				changed = true
			}
		}
		return typed, changed
	default:
		return value, false
	}
}

func rewriteXAIResponsesSSEBlock(block []byte, restoreMap map[string]model.CodexToolSpec) []byte {
	eventName := ""
	var dataParts []string
	for _, line := range strings.Split(strings.TrimRight(string(block), "\r\n"), "\n") {
		line = strings.TrimRight(line, "\r")
		if value, exists := strings.CutPrefix(line, "event:"); exists {
			eventName = strings.TrimSpace(value)
		}
		if value, exists := strings.CutPrefix(line, "data:"); exists {
			dataParts = append(dataParts, value)
		}
	}
	if len(dataParts) == 0 {
		return block
	}
	data := strings.TrimSpace(strings.Join(dataParts, "\n"))
	if data == "[DONE]" {
		return block
	}
	var event any
	if err := decodeJSONUseNumber([]byte(data), &event); err != nil {
		return block
	}
	changed := restoreXAIResponseValue(event, restoreMap)
	if !changed {
		return block
	}
	rewritten, err := json.Marshal(event)
	if err != nil {
		return block
	}
	var builder strings.Builder
	if eventName != "" {
		builder.WriteString("event: ")
		builder.WriteString(eventName)
		builder.WriteByte('\n')
	}
	builder.WriteString("data: ")
	builder.Write(rewritten)
	builder.WriteString("\n\n")
	return []byte(builder.String())
}

func stringValueMap(value map[string]any, key string) string {
	if text, ok := value[key].(string); ok {
		return text
	}
	return ""
}
