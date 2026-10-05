package relay

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bestruirui/octopus/internal/model"
	"github.com/bestruirui/octopus/internal/op"
	"github.com/bestruirui/octopus/internal/transformer/outbound"
	"github.com/gin-gonic/gin"
)

func TestHandleResponsesCompactChatChannelSavesReplayState(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ctx := setupRelayTestDB(t)
	resetResponsesReplayStore()
	defer resetResponsesReplayStore()

	var capturedBodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request body failed: %v", err)
			return
		}
		capturedBodies = append(capturedBodies, body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl_compact_saved","object":"chat.completion","created":1,"model":"compact-model","choices":[{"index":0,"message":{"role":"assistant","content":"compacted history"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}`))
	}))
	defer server.Close()

	channel := &model.Channel{
		Name:     "relay-compact-chat-save-replay",
		Type:     outbound.OutboundTypeOpenAIChat,
		Enabled:  true,
		BaseUrls: []model.BaseUrl{{URL: server.URL + "/v1"}},
		Model:    "compact-model",
		Keys:     []model.ChannelKey{{Enabled: true, ChannelKey: "chat-key"}},
	}
	if err := op.ChannelCreate(channel, ctx); err != nil {
		t.Fatalf("ChannelCreate failed: %v", err)
	}
	group := &model.Group{Name: "relay-compact-chat-save-replay-group", Mode: model.GroupModeFailover}
	if err := op.GroupCreate(group, ctx); err != nil {
		t.Fatalf("GroupCreate failed: %v", err)
	}
	if err := op.GroupItemAdd(&model.GroupItem{GroupID: group.ID, ChannelID: channel.ID, ModelName: "compact-model", Priority: 1, Weight: 1}, ctx); err != nil {
		t.Fatalf("GroupItemAdd failed: %v", err)
	}

	apiKeyID := 45
	requestModel := "relay-compact-chat-save-replay-group"
	newContext := func(body string) (*gin.Context, *httptest.ResponseRecorder) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Set("api_key_id", apiKeyID)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses/compact", strings.NewReader(body))
		c.Request.Header.Set("Content-Type", "application/json")
		return c, recorder
	}

	c, recorder := newContext(`{"model":"` + requestModel + `","input":[{"type":"message","role":"user","content":"hello"}]}`)
	HandleResponsesCompact(c)
	if recorder.Code != http.StatusOK {
		t.Fatalf("expected first compact request to succeed, got status %d body %s", recorder.Code, recorder.Body.String())
	}
	var firstResponse struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &firstResponse); err != nil || firstResponse.ID == "" {
		t.Fatalf("expected first compact response id, got %s", recorder.Body.String())
	}

	secondBody := `{"model":"` + requestModel + `","previous_response_id":"` + firstResponse.ID + `","input":[{"type":"message","role":"user","content":"Continue after compact"}]}`
	c, recorder = newContext(secondBody)
	HandleResponsesCompact(c)
	if recorder.Code != http.StatusOK {
		t.Fatalf("expected second compact continuation to succeed, got status %d body %s", recorder.Code, recorder.Body.String())
	}
	if len(capturedBodies) != 2 {
		t.Fatalf("expected two upstream requests, got %d", len(capturedBodies))
	}

	var payload struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		PreviousResponseID string `json:"previous_response_id"`
	}
	if err := json.Unmarshal(capturedBodies[1], &payload); err != nil {
		t.Fatalf("unmarshal second upstream request failed: %v", err)
	}
	if payload.PreviousResponseID != "" {
		t.Fatalf("expected compact replay state to remove previous_response_id, got %q", payload.PreviousResponseID)
	}
	joined := ""
	for _, message := range payload.Messages {
		joined += message.Role + ":" + message.Content + "\n"
	}
	for _, expected := range []string{"user:hello", "assistant:compacted history", "user:Continue after compact"} {
		if !strings.Contains(joined, expected) {
			t.Fatalf("expected compact replay history to contain %q, got %s", expected, joined)
		}
	}
}
