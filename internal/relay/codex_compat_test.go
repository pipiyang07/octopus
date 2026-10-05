package relay

import (
	"encoding/json"
	"net/http"
	"testing"

	dbmodel "github.com/bestruirui/octopus/internal/model"
	transformerModel "github.com/bestruirui/octopus/internal/transformer/model"
)

func TestApplyCodexCompatPromptCacheRoutingModes(t *testing.T) {
	newRequest := func() *transformerModel.InternalLLMRequest {
		return &transformerModel.InternalLLMRequest{
			RawAPIFormat: transformerModel.APIFormatOpenAIResponse,
			Metadata: map[string]string{
				"user_id": "user_42_session_stable-session",
			},
		}
	}

	tests := []struct {
		name      string
		baseURL   string
		config    *dbmodel.CodexCompatConfig
		wantKey   string
		wantNoKey bool
	}{
		{
			name:      "nil config keeps default behavior",
			baseURL:   "https://api.openai.com/v1",
			wantNoKey: true,
		},
		{
			name:    "auto known upstream injects stable session",
			baseURL: "https://api.openai.com/v1",
			config:  &dbmodel.CodexCompatConfig{PromptCacheRouting: "auto"},
			wantKey: "stable-session",
		},
		{
			name:      "auto unknown upstream stays conservative",
			baseURL:   "https://strict-gateway.example.com/v1",
			config:    &dbmodel.CodexCompatConfig{PromptCacheRouting: "auto"},
			wantNoKey: true,
		},
		{
			name:    "enabled injects for unknown upstream",
			baseURL: "https://strict-gateway.example.com/v1",
			config:  &dbmodel.CodexCompatConfig{PromptCacheRouting: "enabled"},
			wantKey: "stable-session",
		},
		{
			name:      "disabled never injects",
			baseURL:   "https://api.openai.com/v1",
			config:    &dbmodel.CodexCompatConfig{PromptCacheRouting: "disabled"},
			wantNoKey: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			channel := &dbmodel.Channel{BaseUrls: []dbmodel.BaseUrl{{URL: tt.baseURL}}, CodexCompat: tt.config}
			request := newRequest()
			applyCodexCompat(channel, http.Header{}, request)
			if tt.wantNoKey {
				if request.PromptCacheKey != nil {
					t.Fatalf("expected no prompt cache key, got %q", *request.PromptCacheKey)
				}
				return
			}
			if request.PromptCacheKey == nil || *request.PromptCacheKey != tt.wantKey {
				t.Fatalf("expected prompt cache key %q, got %#v", tt.wantKey, request.PromptCacheKey)
			}
		})
	}
}

func TestApplyCodexCompatExplicitPromptCacheKeyWins(t *testing.T) {
	channel := &dbmodel.Channel{
		BaseUrls:    []dbmodel.BaseUrl{{URL: "https://strict-gateway.example.com/v1"}},
		CodexCompat: &dbmodel.CodexCompatConfig{PromptCacheRouting: "enabled"},
	}
	request := &transformerModel.InternalLLMRequest{
		RawAPIFormat:            transformerModel.APIFormatOpenAIResponse,
		Metadata:                map[string]string{"session_id": "session-key"},
		ResponsesPromptCacheKey: stringPointerForTest("client-key"),
	}
	applyCodexCompat(channel, http.Header{}, request)
	if request.PromptCacheKey == nil || *request.PromptCacheKey != "client-key" {
		t.Fatalf("expected explicit client key to win, got %#v", request.PromptCacheKey)
	}
}

func TestApplyCodexCompatReasoningParamMappings(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{name: "thinking", want: "thinking"},
		{name: "enable_thinking", want: "enable_thinking"},
		{name: "reasoning_split", want: "reasoning_split"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &transformerModel.InternalLLMRequest{
				RawAPIFormat:    transformerModel.APIFormatOpenAIResponse,
				ReasoningEffort: "high",
			}
			applyCodexReasoningParam(&dbmodel.CodexCompatConfig{ReasoningParam: tt.want}, request)
			if request.ReasoningEffort != "" {
				t.Fatalf("expected reasoning_effort to be cleared, got %q", request.ReasoningEffort)
			}
			switch tt.want {
			case "thinking":
				if request.Thinking == nil || request.Thinking.Type != "enabled" {
					t.Fatalf("expected thinking enabled, got %#v", request.Thinking)
				}
			case "enable_thinking":
				if request.EnableThinking == nil || !*request.EnableThinking {
					t.Fatalf("expected enable_thinking=true, got %#v", request.EnableThinking)
				}
			case "reasoning_split":
				if request.ReasoningSplit == nil || !*request.ReasoningSplit {
					t.Fatalf("expected reasoning_split=true, got %#v", request.ReasoningSplit)
				}
			}
		})
	}
}

func TestApplyCodexReasoningEffortModes(t *testing.T) {
	tests := []struct {
		name       string
		mode       string
		effort     string
		wantEffort string
	}{
		{name: "passthrough keeps known effort", mode: "passthrough", effort: "ultra", wantEffort: "ultra"},
		{name: "passthrough drops unknown effort", mode: "passthrough", effort: "bogus", wantEffort: ""},
		{name: "deepseek clamps deepest effort", mode: "deepseek", effort: "xhigh", wantEffort: "max"},
		{name: "deepseek promotes ordinary effort", mode: "deepseek", effort: "medium", wantEffort: "high"},
		{name: "low high clamps low", mode: "low_high", effort: "minimal", wantEffort: "low"},
		{name: "low high clamps high", mode: "low_high", effort: "max", wantEffort: "high"},
		{name: "openrouter clamps max", mode: "openrouter", effort: "max", wantEffort: "xhigh"},
		{name: "openrouter keeps medium", mode: "openrouter", effort: "medium", wantEffort: "medium"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &transformerModel.InternalLLMRequest{
				Model:           "test-model",
				ReasoningEffort: tt.effort,
			}
			applyCodexReasoningParam(&dbmodel.CodexCompatConfig{
				ReasoningParam:  "reasoning_effort",
				EffortValueMode: tt.mode,
			}, request)
			if request.ReasoningEffort != tt.wantEffort {
				t.Fatalf("expected effort %q, got %q", tt.wantEffort, request.ReasoningEffort)
			}
			if request.ReasoningEffortObject != nil {
				t.Fatalf("expected no reasoning object, got %#v", request.ReasoningEffortObject)
			}
		})
	}
}

func TestApplyCodexZenReasoningEffortUsesModelLevels(t *testing.T) {
	config := &dbmodel.CodexCompatConfig{
		ReasoningParam:  "reasoning_effort",
		EffortValueMode: "zen",
		ModelReasoningLevels: map[string][]string{
			"deepseek-v4-flash": {"low", "high", "max"},
			"kimi-k3":           {"max"},
		},
	}
	tests := []struct {
		model      string
		effort     string
		wantEffort string
	}{
		{model: "deepseek-v4-flash", effort: "minimal", wantEffort: "low"},
		{model: "deepseek-v4-flash", effort: "medium", wantEffort: "high"},
		{model: "deepseek-v4-flash", effort: "ultra", wantEffort: "max"},
		{model: "provider/kimi-k3", effort: "low", wantEffort: "max"},
		{model: "provider/kimi-k3", effort: "max", wantEffort: "max"},
		{model: "unknown-model", effort: "high", wantEffort: ""},
	}
	for _, tt := range tests {
		t.Run(tt.model+"/"+tt.effort, func(t *testing.T) {
			request := &transformerModel.InternalLLMRequest{
				Model:           tt.model,
				ReasoningEffort: tt.effort,
			}
			applyCodexReasoningParam(config, request)
			if request.ReasoningEffort != tt.wantEffort {
				t.Fatalf("expected effort %q, got %q", tt.wantEffort, request.ReasoningEffort)
			}
		})
	}
}

func TestApplyCodexOpenRouterReasoningEffortObject(t *testing.T) {
	request := &transformerModel.InternalLLMRequest{
		Model:           "openrouter-model",
		ReasoningEffort: "max",
	}
	applyCodexReasoningParam(&dbmodel.CodexCompatConfig{
		ReasoningParam:  "reasoning_effort",
		EffortParam:     "reasoning.effort",
		EffortValueMode: "openrouter",
	}, request)
	if request.ReasoningEffort != "" {
		t.Fatalf("expected top-level effort to be cleared, got %q", request.ReasoningEffort)
	}
	if request.ReasoningEffortObject == nil || request.ReasoningEffortObject.Effort != "xhigh" {
		t.Fatalf("expected reasoning.effort=xhigh, got %#v", request.ReasoningEffortObject)
	}

	request = &transformerModel.InternalLLMRequest{ReasoningEffort: "none"}
	applyCodexReasoningParam(&dbmodel.CodexCompatConfig{
		ReasoningParam: "reasoning_effort",
		EffortParam:    "reasoning.effort",
	}, request)
	if request.ReasoningEffortObject == nil || request.ReasoningEffortObject.Effort != "none" {
		t.Fatalf("expected explicit reasoning.effort=none, got %#v", request.ReasoningEffortObject)
	}
}

func TestApplyCodexReasoningParamRespectsExplicitDisable(t *testing.T) {
	request := &transformerModel.InternalLLMRequest{ReasoningEffort: "off"}
	applyCodexReasoningParam(&dbmodel.CodexCompatConfig{ReasoningParam: "thinking"}, request)
	if request.Thinking == nil || request.Thinking.Type != "disabled" {
		t.Fatalf("expected thinking disabled, got %#v", request.Thinking)
	}
	if request.ReasoningEffort != "" {
		t.Fatalf("expected original effort to be cleared, got %q", request.ReasoningEffort)
	}
}

func TestApplyCodexCompatMoonshotRefSiblings(t *testing.T) {
	channel := &dbmodel.Channel{
		BaseUrls: []dbmodel.BaseUrl{{URL: "https://api.moonshot.cn/v1"}},
	}
	parameters := json.RawMessage(`{
		"type":"object",
		"properties":{
			"prompt":{"$ref":"#/$defs/P","description":"prompt"}
		},
		"$defs":{"P":{"$ref":"#/$defs/S","type":"string"}}
	}`)
	request := &transformerModel.InternalLLMRequest{
		RawAPIFormat: transformerModel.APIFormatOpenAIResponse,
		Tools: []transformerModel.Tool{{
			Type: "function",
			Function: transformerModel.Function{
				Name:       "automation_update",
				Parameters: parameters,
			},
		}},
	}

	applyCodexCompat(channel, http.Header{}, request)

	var schema struct {
		Properties struct {
			Prompt struct {
				AllOf []struct {
					Ref string `json:"$ref"`
				} `json:"allOf"`
			} `json:"prompt"`
		} `json:"properties"`
		Defs map[string]struct {
			AllOf []struct {
				Ref string `json:"$ref"`
			} `json:"allOf"`
		} `json:"$defs"`
	}
	if err := json.Unmarshal(request.Tools[0].Function.Parameters, &schema); err != nil {
		t.Fatalf("unmarshal rewritten schema failed: %v", err)
	}
	if len(schema.Properties.Prompt.AllOf) != 1 || schema.Properties.Prompt.AllOf[0].Ref != "#/$defs/P" {
		t.Fatalf("expected property $ref sibling rewrite, got %s", request.Tools[0].Function.Parameters)
	}
	if len(schema.Defs["P"].AllOf) != 1 || schema.Defs["P"].AllOf[0].Ref != "#/$defs/S" {
		t.Fatalf("expected $defs $ref sibling rewrite, got %s", request.Tools[0].Function.Parameters)
	}
}

func stringPointerForTest(value string) *string {
	return &value
}
