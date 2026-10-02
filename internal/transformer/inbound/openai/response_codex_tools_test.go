package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/samber/lo"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

const codexToolSearchFixture = `{
  "model": "gpt-5.4",
  "tools": [{"type": "tool_search"}],
  "input": [
    {
      "type": "tool_search_call",
      "call_id": "call_tool_search_1",
      "status": "completed",
      "execution": "client",
      "arguments": {"query": "Gmail search emails", "limit": 5}
    },
    {
      "type": "tool_search_output",
      "call_id": "call_tool_search_1",
      "status": "completed",
      "tools": [{
        "type": "namespace",
        "name": "mcp__codex_apps__gmail",
        "description": "Find and reference emails from your inbox.",
        "tools": [{
          "type": "function",
          "name": "_search_emails",
          "description": "Search Gmail for emails matching a query.",
          "strict": false,
          "parameters": {
            "type": "object",
            "properties": {
              "query": {"type": "string"},
              "max_results": {"type": "integer"}
            },
            "required": ["query"]
          }
        }]
      }]
    },
    {"type": "message", "role": "user", "content": "Search unread inbox mail."}
  ]
}`

func TestResponsesInboundConvertsCodexToolSearchAndNamespace(t *testing.T) {
	inbound := &ResponseInbound{}
	request, err := inbound.TransformRequest(context.Background(), []byte(codexToolSearchFixture))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if request.HasOpenAIResponsesPassthrough() {
		t.Fatalf("expected Codex tools to be Chat-convertible, got passthrough reason %q", request.OpenAIResponsesPassthroughReasonTextValue())
	}

	var toolNames []string
	for _, tool := range request.Tools {
		toolNames = append(toolNames, tool.Function.Name)
	}
	if !contains(toolNames, "tool_search") || !contains(toolNames, "mcp__codex_apps__gmail___search_emails") {
		t.Fatalf("expected tool_search and flattened namespace tool, got %v", toolNames)
	}

	if len(request.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %#v", request.Messages)
	}
	assistant := request.Messages[0]
	if assistant.Role != "assistant" || len(assistant.ToolCalls) != 1 {
		t.Fatalf("unexpected first message: %#v", assistant)
	}
	toolCall := assistant.ToolCalls[0]
	if toolCall.ID != "call_tool_search_1" || toolCall.Function.Name != "tool_search" {
		t.Fatalf("unexpected tool_search call: %#v", toolCall)
	}
	var arguments map[string]any
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
		t.Fatalf("tool_search arguments are not JSON: %v", err)
	}
	if arguments["query"] != "Gmail search emails" {
		t.Fatalf("unexpected tool_search arguments: %#v", arguments)
	}

	toolResult := request.Messages[1]
	if toolResult.Role != "tool" || toolResult.ToolCallID == nil || *toolResult.ToolCallID != "call_tool_search_1" {
		t.Fatalf("unexpected tool result: %#v", toolResult)
	}
	if toolResult.Content.Content == nil || !strings.Contains(*toolResult.Content.Content, "mcp__codex_apps__gmail") {
		t.Fatalf("expected namespace definition in tool result, got %#v", toolResult.Content)
	}
}

func TestResponsesInboundRestoresCodexToolCallsFromChat(t *testing.T) {
	inbound := &ResponseInbound{}
	request, err := inbound.TransformRequest(context.Background(), []byte(codexToolSearchFixture))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	_ = request

	chatResponse := &model.InternalLLMResponse{
		ID:      "chatcmpl_1",
		Model:   "gpt-5.4",
		Created: 123,
		Choices: []model.Choice{{
			Message: &model.Message{
				Role: "assistant",
				ToolCalls: []model.ToolCall{
					{
						ID:   "call_search",
						Type: "function",
						Function: model.FunctionCall{
							Name:      "tool_search",
							Arguments: `{"query":"gmail","limit":5}`,
						},
					},
					{
						ID:   "call_gmail",
						Type: "function",
						Function: model.FunctionCall{
							Name:      "mcp__codex_apps__gmail___search_emails",
							Arguments: `{"query":"unread"}`,
						},
					},
				},
			},
			FinishReason: lo.ToPtr("tool_calls"),
		}},
	}

	rawResponse, err := inbound.TransformResponse(context.Background(), chatResponse)
	if err != nil {
		t.Fatalf("TransformResponse failed: %v", err)
	}
	var response struct {
		Output []ResponsesItem `json:"output"`
	}
	if err := json.Unmarshal(rawResponse, &response); err != nil {
		t.Fatalf("failed to decode responses response: %v", err)
	}
	if len(response.Output) != 2 {
		t.Fatalf("expected two restored tool calls, got %#v", response.Output)
	}

	search := response.Output[0]
	if search.Type != "tool_search_call" || search.CallID != "call_search" || search.Execution != "client" {
		t.Fatalf("unexpected tool_search call: %#v", search)
	}
	var searchArguments map[string]any
	if err := json.Unmarshal(search.Arguments, &searchArguments); err != nil {
		t.Fatalf("tool_search response arguments are not JSON: %v", err)
	}
	if searchArguments["query"] != "gmail" {
		t.Fatalf("unexpected tool_search response arguments: %#v", searchArguments)
	}

	namespaced := response.Output[1]
	if namespaced.Type != "function_call" || namespaced.Name != "_search_emails" {
		t.Fatalf("unexpected namespace function call: %#v", namespaced)
	}
	if namespaced.Namespace == nil || *namespaced.Namespace != "mcp__codex_apps__gmail" {
		t.Fatalf("expected namespace to be restored, got %#v", namespaced.Namespace)
	}
}

func TestResponsesInboundConvertsAndRestoresCustomTool(t *testing.T) {
	inbound := &ResponseInbound{}
	requestBody := `{
	  "model":"gpt-5.4",
	  "tools":[{
	    "type":"custom",
	    "name":"apply_patch",
	    "description":"Apply a patch.",
	    "format":{"type":"grammar","syntax":"lark","definition":"start: begin_patch"}
	  }],
	  "tool_choice":{"type":"custom","name":"apply_patch"},
	  "input":[{
	    "type":"custom_tool_call",
	    "call_id":"call_patch",
	    "name":"apply_patch",
	    "input":"*** Begin Patch"
	  }]
	}`
	request, err := inbound.TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if request.HasOpenAIResponsesPassthrough() {
		t.Fatalf("custom tool should not require Responses passthrough: %q", request.OpenAIResponsesPassthroughReasonTextValue())
	}
	if len(request.Tools) != 1 || request.Tools[0].Function.Name != "apply_patch" {
		t.Fatalf("unexpected custom tool conversion: %#v", request.Tools)
	}
	if !strings.Contains(request.Tools[0].Function.Description, `"type":"custom"`) {
		t.Fatalf("custom tool metadata was not preserved: %q", request.Tools[0].Function.Description)
	}
	if len(request.Messages) != 1 || len(request.Messages[0].ToolCalls) != 1 {
		t.Fatalf("unexpected custom tool call history: %#v", request.Messages)
	}
	if request.Messages[0].ToolCalls[0].Function.Arguments != `{"input":"*** Begin Patch"}` {
		t.Fatalf("unexpected custom tool arguments: %q", request.Messages[0].ToolCalls[0].Function.Arguments)
	}
	if request.ToolChoice == nil || request.ToolChoice.NamedToolChoice == nil ||
		request.ToolChoice.NamedToolChoice.Function == nil ||
		request.ToolChoice.NamedToolChoice.Function.Name != "apply_patch" {
		t.Fatalf("unexpected custom tool choice: %#v", request.ToolChoice)
	}

	chatResponse := &model.InternalLLMResponse{
		ID:    "chatcmpl_custom",
		Model: "gpt-5.4",
		Choices: []model.Choice{{
			Message: &model.Message{
				Role: "assistant",
				ToolCalls: []model.ToolCall{{
					ID:   "call_patch",
					Type: "function",
					Function: model.FunctionCall{
						Name:      "apply_patch",
						Arguments: `{"input":"*** Begin Patch\n*** End Patch"}`,
					},
				}},
			},
			FinishReason: lo.ToPtr("tool_calls"),
		}},
	}
	rawResponse, err := inbound.TransformResponse(context.Background(), chatResponse)
	if err != nil {
		t.Fatalf("TransformResponse failed: %v", err)
	}
	var response struct {
		Output []ResponsesItem `json:"output"`
	}
	if err := json.Unmarshal(rawResponse, &response); err != nil {
		t.Fatalf("failed to decode responses response: %v", err)
	}
	if len(response.Output) != 1 {
		t.Fatalf("expected custom tool call output, got %#v", response.Output)
	}
	item := response.Output[0]
	if item.Type != "custom_tool_call" || item.Name != "apply_patch" || item.CallID != "call_patch" {
		t.Fatalf("unexpected restored custom tool call: %#v", item)
	}
	var input string
	if err := json.Unmarshal(item.Input, &input); err != nil {
		t.Fatalf("custom input is not a JSON string: %v", err)
	}
	if input != "*** Begin Patch\n*** End Patch" {
		t.Fatalf("unexpected custom input: %q", input)
	}
}

func TestResponsesInboundLiftsAdditionalToolsCarrier(t *testing.T) {
	inbound := &ResponseInbound{}
	requestBody := `{
	  "model":"gpt-5.4",
	  "input":[
	    {
	      "type":"additional_tools",
	      "tools":[{
	        "type":"function",
	        "name":"lookup_email",
	        "description":"Lookup an email",
	        "parameters":{
	          "type":"object",
	          "properties":{"query":{"type":"string"}},
	          "required":["query"]
	        }
	      }]
	    },
	    {"type":"message","role":"user","content":"Search unread inbox mail."}
	  ]
	}`
	request, err := inbound.TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if request.HasOpenAIResponsesPassthrough() {
		t.Fatalf("additional_tools carrier should be Chat-convertible, got %q", request.OpenAIResponsesPassthroughReasonTextValue())
	}
	if len(request.Tools) != 1 || request.Tools[0].Function.Name != "lookup_email" {
		t.Fatalf("expected additional_tools function to be lifted, got %#v", request.Tools)
	}
	if request.Tools[0].Function.Description != "Lookup an email" {
		t.Fatalf("unexpected lifted tool description: %q", request.Tools[0].Function.Description)
	}
	if len(request.Messages) != 1 || request.Messages[0].Role != "user" {
		t.Fatalf("expected additional_tools carrier to be omitted from messages, got %#v", request.Messages)
	}
	if request.Messages[0].Content.Content == nil || *request.Messages[0].Content.Content != "Search unread inbox mail." {
		t.Fatalf("unexpected user message: %#v", request.Messages[0].Content)
	}
}

func TestResponsesStreamRestoresCodexToolCallKinds(t *testing.T) {
	requestBody := `{
	  "model":"gpt-5.4",
	  "tools":[{"type":"custom","name":"apply_patch"},{"type":"tool_search"}],
	  "input":[
	    {
	      "type":"tool_search_output",
	      "call_id":"call_search",
	      "tools":[{
	        "type":"namespace",
	        "name":"mcp__files__",
	        "tools":[{"type":"function","name":"read","parameters":{"type":"object","properties":{}}}]
	      }]
	    }
	  ]
	}`
	inbound := &ResponseInbound{}
	if _, err := inbound.TransformRequest(context.Background(), []byte(requestBody)); err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}

	chunks := []*model.InternalLLMResponse{
		chunkWithDelta("gpt-5.4", &model.Message{
			ToolCalls: []model.ToolCall{
				{
					Index: 0,
					ID:    "call_patch",
					Type:  "function",
					Function: model.FunctionCall{
						Name:      "apply_patch",
						Arguments: `{"input":"*** Begin Patch`,
					},
				},
				{
					Index: 1,
					ID:    "call_search",
					Type:  "function",
					Function: model.FunctionCall{
						Name:      "tool_search",
						Arguments: `{"query":"files"}`,
					},
				},
				{
					Index: 2,
					ID:    "call_read",
					Type:  "function",
					Function: model.FunctionCall{
						Name:      "mcp__files____read",
						Arguments: `{"path":"a.txt"}`,
					},
				},
			},
		}),
		chunkWithDelta("gpt-5.4", &model.Message{
			ToolCalls: []model.ToolCall{{
				Index: 0,
				Function: model.FunctionCall{
					Arguments: `\n*** End Patch"}`,
				},
			}},
		}),
		chunkWithFinish("gpt-5.4", "tool_calls"),
	}

	var output bytes.Buffer
	for _, chunk := range chunks {
		encoded, err := inbound.TransformStream(context.Background(), chunk)
		if err != nil {
			t.Fatalf("TransformStream failed: %v", err)
		}
		output.Write(encoded)
	}
	events := parseSSEEvents(t, output.Bytes())
	eventNames := eventTypes(events)

	added := make(map[string]*ResponsesItem)
	for index := range events {
		if events[index].Type != "response.output_item.added" || events[index].Item == nil {
			continue
		}
		added[events[index].Item.CallID] = events[index].Item
	}
	if added["call_patch"] == nil || added["call_patch"].Type != "custom_tool_call" || added["call_patch"].Name != "apply_patch" {
		t.Fatalf("expected custom_tool_call to be restored, got %#v; events=%v", added["call_patch"], eventNames)
	}
	if added["call_search"] == nil || added["call_search"].Type != "tool_search_call" || added["call_search"].Execution != "client" {
		t.Fatalf("expected tool_search_call to be restored, got %#v; events=%v", added["call_search"], eventNames)
	}
	if added["call_read"] == nil || added["call_read"].Type != "function_call" ||
		added["call_read"].Name != "read" ||
		added["call_read"].Namespace == nil || *added["call_read"].Namespace != "mcp__files__" {
		t.Fatalf("expected namespace function_call to be restored, got %#v; events=%v", added["call_read"], eventNames)
	}

	if findEvent(events, "response.function_call_arguments.delta") == nil ||
		findEvent(events, "response.function_call_arguments.done") == nil {
		t.Fatalf("expected function argument events for tool_search and namespace calls, got %v", eventNames)
	}
	customDone := findEvent(events, "response.custom_tool_call_input.done")
	if customDone == nil || customDone.Input != "*** Begin Patch\n*** End Patch" {
		t.Fatalf("expected custom tool input to be restored, got %#v; events=%v", customDone, eventNames)
	}

	var doneTypes []string
	for _, event := range events {
		if event.Type == "response.output_item.done" && event.Item != nil {
			doneTypes = append(doneTypes, event.Item.Type)
		}
	}
	if !contains(doneTypes, "custom_tool_call") || !contains(doneTypes, "tool_search_call") || !contains(doneTypes, "function_call") {
		t.Fatalf("expected all restored item types to finish, got %v; events=%v", doneTypes, eventNames)
	}
}

func TestResponsesInboundDropsHostedWebSearchForChat(t *testing.T) {
	inbound := &ResponseInbound{}
	requestBody := `{
	  "model":"glm-5.3",
	  "tools":[{"type":"web_search"}],
	  "tool_choice":"auto",
	  "parallel_tool_calls":true,
	  "input":"hello"
	}`
	request, err := inbound.TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if request.HasOpenAIResponsesPassthrough() {
		t.Fatalf("hosted web_search should be dropped for Chat conversion, got %q", request.OpenAIResponsesPassthroughReasonTextValue())
	}
	if len(request.Tools) != 0 {
		t.Fatalf("expected hosted web_search to stay out of Chat tools, got %#v", request.Tools)
	}
	if request.ToolChoice != nil {
		t.Fatalf("expected tool_choice to be dropped when no tools remain, got %#v", request.ToolChoice)
	}
	if request.ParallelToolCalls != nil {
		t.Fatalf("expected parallel_tool_calls to be dropped when no tools remain, got %#v", *request.ParallelToolCalls)
	}
}

func TestResponsesInboundKeepsConvertibleToolsAlongsideWebSearch(t *testing.T) {
	inbound := &ResponseInbound{}
	requestBody := `{
	  "model":"glm-5.3",
	  "tools":[
	    {"type":"web_search"},
	    {"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}},
	    {"type":"namespace","name":"mcp__files__","tools":[{"type":"function","name":"read","parameters":{"type":"object","properties":{}}}]}
	  ],
	  "input":"hello"
	}`
	request, err := inbound.TransformRequest(context.Background(), []byte(requestBody))
	if err != nil {
		t.Fatalf("TransformRequest failed: %v", err)
	}
	if request.HasOpenAIResponsesPassthrough() {
		t.Fatalf("web_search alongside convertible tools should not require Responses passthrough, got %q", request.OpenAIResponsesPassthroughReasonTextValue())
	}
	var names []string
	for _, tool := range request.Tools {
		names = append(names, tool.Function.Name)
	}
	if !contains(names, "get_weather") || !contains(names, "mcp__files____read") {
		t.Fatalf("expected function and flattened namespace tools, got %v", names)
	}
	if contains(names, "web_search") {
		t.Fatalf("hosted web_search should not be sent as a Chat function, got %v", names)
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
