package openai

import "testing"

func TestIsReasoningChatModel(t *testing.T) {
	positive := []string{
		"o1",
		"o3-mini",
		"o4-mini",
		"gpt-5",
		"gpt-5.4",
		"gpt-5-codex",
		"openrouter/openai/gpt-5.4",
		"grok-4.5",
		"grok-4.10",
		"grok-4.10-build",
		"grok-build-0.1",
		"provider/grok-build-0.1",
	}
	for _, modelName := range positive {
		if !isReasoningChatModel(modelName) {
			t.Errorf("expected %q to be a reasoning Chat model", modelName)
		}
	}

	negative := []string{
		"",
		"openai/gpt-4o",
		"claude-sonnet-4-6",
		"grok-4",
		"grok-4.4",
		"grok-4.build",
	}
	for _, modelName := range negative {
		if isReasoningChatModel(modelName) {
			t.Errorf("expected %q not to be a reasoning Chat model", modelName)
		}
	}
}
