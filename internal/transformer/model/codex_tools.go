package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	CodexToolKindFunction   CodexToolKind = "function"
	CodexToolKindNamespace  CodexToolKind = "namespace"
	CodexToolKindCustom     CodexToolKind = "custom"
	CodexToolKindToolSearch CodexToolKind = "tool_search"
)

const (
	codexToolSearchProxyName = "tool_search"
	codexCustomInputField    = "input"
	codexChatToolNameMaxLen  = 64
)

type CodexToolKind string

type CodexToolSpec struct {
	Kind      CodexToolKind
	Name      string
	Namespace string
}

type CodexToolContext struct {
	Tools      []Tool
	seen       map[string]struct{}
	specs      map[string]CodexToolSpec
	namespaces map[string]string
}

func NewCodexToolContext() *CodexToolContext {
	return &CodexToolContext{
		seen:       make(map[string]struct{}),
		specs:      make(map[string]CodexToolSpec),
		namespaces: make(map[string]string),
	}
}

func BuildCodexToolContext(tools []json.RawMessage, inputItems []json.RawMessage) *CodexToolContext {
	context := NewCodexToolContext()
	for _, raw := range tools {
		context.AddResponseTool(raw)
	}
	for _, raw := range inputItems {
		context.collectInputTools(raw)
	}
	return context
}

func (c *CodexToolContext) AddResponseTool(raw json.RawMessage) {
	if c == nil || len(raw) == 0 {
		return
	}

	var name string
	if err := json.Unmarshal(raw, &name); err == nil {
		if strings.TrimSpace(name) != "" {
			c.addCustomTool(map[string]any{"type": "custom", "name": name})
		}
		return
	}

	var tool map[string]any
	if err := json.Unmarshal(raw, &tool); err != nil {
		return
	}
	switch stringValue(tool["type"]) {
	case "function":
		c.addFunctionTool(tool, "")
	case "custom":
		c.addCustomTool(tool)
	case "tool_search":
		c.addToolSearchTool()
	case "namespace":
		c.addNamespaceTool(tool)
	}
}

func (c *CodexToolContext) Lookup(chatName string) (CodexToolSpec, bool) {
	if c == nil {
		return CodexToolSpec{}, false
	}
	spec, ok := c.specs[chatName]
	return spec, ok
}

func (c *CodexToolContext) ChatNameForResponseFunction(name, namespace string) string {
	if c == nil {
		return flattenCodexNamespaceToolName(namespace, name)
	}
	if namespace != "" {
		if chatName, ok := c.namespaces[namespace+"\x00"+name]; ok {
			return chatName
		}
		return flattenCodexNamespaceToolName(namespace, name)
	}
	return name
}

func (c *CodexToolContext) addChatTool(chatName string, spec CodexToolSpec, tool Tool) {
	chatName = strings.TrimSpace(chatName)
	if chatName == "" {
		return
	}
	if _, exists := c.seen[chatName]; exists {
		return
	}
	c.seen[chatName] = struct{}{}
	c.specs[chatName] = spec
	if spec.Namespace != "" {
		c.namespaces[spec.Namespace+"\x00"+spec.Name] = chatName
	}
	c.Tools = append(c.Tools, tool)
}

func (c *CodexToolContext) addFunctionTool(tool map[string]any, namespace string) {
	originalName := responseToolName(tool)
	if originalName == "" {
		return
	}
	chatName := originalName
	kind := CodexToolKindFunction
	if namespace != "" {
		chatName = flattenCodexNamespaceToolName(namespace, originalName)
		kind = CodexToolKindNamespace
	}

	function := Function{Name: chatName}
	if nested, ok := tool["function"].(map[string]any); ok {
		function.Description = stringValue(nested["description"])
		function.Parameters = normalizeCodexFunctionParameters(nested["parameters"])
		if strict, ok := tool["strict"]; ok {
			function.Strict = boolPointer(strict)
		} else if strict, ok := nested["strict"]; ok {
			function.Strict = boolPointer(strict)
		}
	} else {
		function.Description = stringValue(tool["description"])
		function.Parameters = normalizeCodexFunctionParameters(tool["parameters"])
		function.Strict = boolPointer(tool["strict"])
	}

	c.addChatTool(chatName, CodexToolSpec{
		Kind:      kind,
		Name:      originalName,
		Namespace: namespace,
	}, Tool{Type: "function", Function: function})
}

func (c *CodexToolContext) addCustomTool(tool map[string]any) {
	name := strings.TrimSpace(stringValue(tool["name"]))
	if name == "" {
		return
	}
	description := "Original tool definition:\n```json\n" + canonicalCodexJSON(tool) + "\n```"
	parameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			codexCustomInputField: map[string]any{
				"type":        "string",
				"description": "Raw string input for the original custom tool. Preserve formatting exactly and follow the original tool definition embedded in the description.",
			},
		},
		"required": []string{codexCustomInputField},
	}
	rawParameters, _ := json.Marshal(parameters)
	c.addChatTool(name, CodexToolSpec{Kind: CodexToolKindCustom, Name: name}, Tool{
		Type: "function",
		Function: Function{
			Name:        name,
			Description: description,
			Parameters:  rawParameters,
		},
	})
}

func (c *CodexToolContext) addToolSearchTool() {
	parameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "Search query for tools or connectors to load.",
			},
			"limit": map[string]any{
				"type":        "integer",
				"description": "Maximum number of tool groups to return.",
			},
		},
		"required": []string{"query"},
	}
	rawParameters, _ := json.Marshal(parameters)
	c.addChatTool(codexToolSearchProxyName, CodexToolSpec{Kind: CodexToolKindToolSearch, Name: codexToolSearchProxyName}, Tool{
		Type: "function",
		Function: Function{
			Name:        codexToolSearchProxyName,
			Description: "Search and load Codex tools, plugins, connectors, and MCP namespaces for the current task.",
			Parameters:  rawParameters,
		},
	})
}

func (c *CodexToolContext) addNamespaceTool(tool map[string]any) {
	namespace := strings.TrimSpace(stringValue(tool["name"]))
	if namespace == "" {
		return
	}
	children, ok := tool["tools"].([]any)
	if !ok {
		children, _ = tool["children"].([]any)
	}
	for _, child := range children {
		childMap, ok := child.(map[string]any)
		if !ok || stringValue(childMap["type"]) != "function" {
			continue
		}
		c.addFunctionTool(childMap, namespace)
	}
}

func (c *CodexToolContext) collectInputTools(value any) {
	switch typed := value.(type) {
	case json.RawMessage:
		if len(typed) == 0 {
			return
		}
		var decoded any
		if err := json.Unmarshal(typed, &decoded); err != nil {
			return
		}
		c.collectInputTools(decoded)
	case map[string]any:
		if stringValue(typed["type"]) == "tool_search_output" {
			if tools, ok := typed["tools"].([]any); ok {
				for _, tool := range tools {
					if raw, err := json.Marshal(tool); err == nil {
						c.AddResponseTool(raw)
					}
				}
			}
		}
		for _, child := range typed {
			c.collectInputTools(child)
		}
	case []any:
		for _, child := range typed {
			c.collectInputTools(child)
		}
	}
}

func responseToolName(tool map[string]any) string {
	if nested, ok := tool["function"].(map[string]any); ok {
		if name := strings.TrimSpace(stringValue(nested["name"])); name != "" {
			return name
		}
	}
	return strings.TrimSpace(stringValue(tool["name"]))
}

func normalizeCodexFunctionParameters(value any) json.RawMessage {
	parameters := map[string]any{"type": "object", "properties": map[string]any{}}
	if object, ok := value.(map[string]any); ok {
		parameters = object
		if stringValue(parameters["type"]) != "object" {
			parameters["type"] = "object"
		}
	}
	raw, _ := json.Marshal(parameters)
	return raw
}

func flattenCodexNamespaceToolName(namespace, name string) string {
	fullName := namespace + "__" + name
	if len(fullName) <= codexChatToolNameMaxLen {
		return fullName
	}
	digest := sha256.Sum256([]byte(fullName))
	suffix := "__" + hex.EncodeToString(digest[:8])
	prefixLength := codexChatToolNameMaxLen - len(suffix)
	if prefixLength < 0 {
		prefixLength = 0
	}
	if prefixLength > len(fullName) {
		prefixLength = len(fullName)
	}
	return fullName[:prefixLength] + suffix
}

// FlattenCodexNamespaceToolName exposes the deterministic Chat tool name used
// by both the Chat bridge and native Responses compatibility layers.
func FlattenCodexNamespaceToolName(namespace, name string) string {
	return flattenCodexNamespaceToolName(namespace, name)
}

func canonicalCodexJSON(value any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(raw)
}

func stringValue(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	return ""
}

func boolPointer(value any) *bool {
	if result, ok := value.(bool); ok {
		return &result
	}
	return nil
}
