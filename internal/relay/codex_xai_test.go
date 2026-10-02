package relay

import (
	"encoding/json"
	"strings"
	"testing"

	dbmodel "github.com/bestruirui/octopus/internal/model"
	"github.com/bestruirui/octopus/internal/transformer/model"
	"github.com/bestruirui/octopus/internal/transformer/outbound"
)

func TestNeedsXAIResponsesCompat(t *testing.T) {
	channel := &dbmodel.Channel{
		Type:     outbound.OutboundTypeOpenAIResponse,
		BaseUrls: []dbmodel.BaseUrl{{URL: "https://api.x.ai/v1"}},
	}
	if !needsXAIResponsesCompat(channel) {
		t.Fatal("expected first-party xAI Responses channel to be enabled")
	}
	channel.Type = outbound.OutboundTypeOpenAIChat
	if needsXAIResponsesCompat(channel) {
		t.Fatal("xAI native Responses compatibility must not run on Chat channels")
	}
}

func TestPrepareXAIResponsesRequestBody(t *testing.T) {
	raw := `{
		"model":"grok-4.5",
		"prompt_cache_retention":"24h",
		"safety_identifier":"user-1",
		"presence_penalty":0.1,
		"frequencyPenalty":0.2,
		"stop":["x"],
		"tools":[
			{"type":"function","name":"plain_tool","parameters":{"type":"object","properties":{"q":{"type":"string"}}},"external_web_access":true},
			{"type":"tool_search"},
			{
				"type":"namespace",
				"name":"mcp__files__",
				"tools":[{"type":"function","name":"read","description":"read a file","parameters":{"oneOf":[{"type":"object","properties":{"q":{"type":"string"}}},{"type":"null"}]}}]
			}
		],
		"input":[
			{"type":"function_call","name":"read","namespace":"mcp__files__","call_id":"c1","arguments":"{}"},
			{"type":"reasoning","content":null},
			{"type":"additional_tools","tools":[{"type":"function","name":"extra"}]}
		],
		"tool_choice":{"type":"namespace","name":"mcp__files__"}
	}`

	rewritten, restoreMap, err := prepareXAIResponsesRequestBody([]byte(raw))
	if err != nil {
		t.Fatalf("prepareXAIResponsesRequestBody failed: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(rewritten, &body); err != nil {
		t.Fatalf("rewritten request is not JSON: %v", err)
	}

	for _, field := range []string{"prompt_cache_retention", "safety_identifier", "presence_penalty", "frequencyPenalty", "stop"} {
		if _, exists := body[field]; exists {
			t.Fatalf("expected unsupported field %q to be removed", field)
		}
	}
	if strings.Contains(string(rewritten), "external_web_access") {
		t.Fatalf("expected recursive external_web_access field to be removed: %s", rewritten)
	}
	if body["tool_choice"] != "auto" {
		t.Fatalf("expected namespace tool_choice to degrade to auto, got %#v", body["tool_choice"])
	}

	tools, _ := body["tools"].([]any)
	var toolNames []string
	for _, rawTool := range tools {
		tool, _ := rawTool.(map[string]any)
		toolNames = append(toolNames, tool["name"].(string))
	}
	for _, name := range []string{"plain_tool", "mcp__files____read", "extra"} {
		if !contains(toolNames, name) {
			t.Fatalf("expected tool %q, got %v", name, toolNames)
		}
	}
	if contains(toolNames, "tool_search") {
		t.Fatalf("expected unsupported tool_search to be removed, got %v", toolNames)
	}

	input, _ := body["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("expected additional_tools carrier to be removed, got %#v", input)
	}
	call, _ := input[0].(map[string]any)
	if call["name"] != "mcp__files____read" || call["namespace"] != nil {
		t.Fatalf("expected namespace-qualified call to be flattened, got %#v", call)
	}
	reasoning, _ := input[1].(map[string]any)
	if _, exists := reasoning["content"]; exists {
		t.Fatalf("expected null reasoning content to be removed, got %#v", reasoning)
	}

	var readTool map[string]any
	for _, rawTool := range tools {
		tool, _ := rawTool.(map[string]any)
		if tool["name"] == "mcp__files____read" {
			readTool = tool
		}
	}
	parameters, _ := readTool["parameters"].(map[string]any)
	if parameters["type"] != "object" {
		t.Fatalf("expected root union schema to collapse to object, got %#v", readTool["parameters"])
	}
	if _, exists := parameters["oneOf"]; exists {
		t.Fatalf("expected root union to be removed, got %#v", parameters)
	}

	if spec, ok := restoreMap["mcp__files____read"]; !ok ||
		spec.Name != "read" || spec.Namespace != "mcp__files__" ||
		spec.Kind != "namespace" {
		t.Fatalf("unexpected restore map: %#v", restoreMap)
	}
}

func TestRestoreXAIResponsesBody(t *testing.T) {
	restoreMap := map[string]model.CodexToolSpec{
		"mcp__files____read": {Name: "read", Namespace: "mcp__files__", Kind: "namespace"},
	}
	body := []byte(`{"output":[{"type":"function_call","name":"mcp__files____read","arguments":"{\"session_id\":92116.0,\"nested\":{\"value\":1.5}}"}]}`)
	restored := restoreXAIResponsesBody(body, restoreMap)
	var payload struct {
		Output []struct {
			Type      string `json:"type"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
			Arguments string `json:"arguments"`
		} `json:"output"`
	}
	if err := json.Unmarshal(restored, &payload); err != nil {
		t.Fatalf("restored response is not JSON: %v", err)
	}
	if len(payload.Output) != 1 {
		t.Fatalf("expected one output item, got %s", restored)
	}
	item := payload.Output[0]
	if item.Name != "read" || item.Namespace != "mcp__files__" {
		t.Fatalf("expected namespace identity to be restored, got %#v", item)
	}
	if !strings.Contains(item.Arguments, `"session_id":92116`) || !strings.Contains(item.Arguments, `"value":1.5`) {
		t.Fatalf("expected whole float to become an integer while non-whole float stays, got %q", item.Arguments)
	}
}

func TestRewriteXAIResponsesSSEBlock(t *testing.T) {
	restoreMap := map[string]model.CodexToolSpec{
		"mcp__files____read": {Name: "read", Namespace: "mcp__files__", Kind: "namespace"},
	}
	done := []byte("event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"session_id\\\":92116.0}\"}\n\n")
	rewritten := string(rewriteXAIResponsesSSEBlock(done, restoreMap))
	if !strings.Contains(rewritten, "event: response.function_call_arguments.done") {
		t.Fatalf("expected event name to be preserved, got %s", rewritten)
	}
	if !strings.Contains(rewritten, `\"session_id\":92116`) {
		t.Fatalf("expected completed arguments to be normalized, got %s", rewritten)
	}

	delta := []byte("event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"session_id\\\":92116.0\"}\n\n")
	if got := string(rewriteXAIResponsesSSEBlock(delta, restoreMap)); got != string(delta) {
		t.Fatalf("expected incomplete delta bytes to remain unchanged, got %s", got)
	}
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
