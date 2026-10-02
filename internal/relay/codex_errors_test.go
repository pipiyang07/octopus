package relay

import (
	"net/http"
	"testing"
)

func TestChatErrorToResponsesErrorDetail(t *testing.T) {
	body := `{"error":{"message":"invalid tool schema","type":"invalid_request_error","param":"tools[0].function.parameters","code":"invalid_json_schema"}}`
	detail := chatErrorToResponsesErrorDetail(http.StatusBadRequest, body)
	if detail.Message != "invalid tool schema" || detail.Type != "invalid_request_error" ||
		detail.Param != "tools[0].function.parameters" || detail.Code != "invalid_json_schema" {
		t.Fatalf("unexpected error detail: %+v", detail)
	}

	detail = chatErrorToResponsesErrorDetail(http.StatusBadRequest, `{"base_resp":{"status_code":1004,"status_msg":"rate limited"}}`)
	if detail.Message != "rate limited" || detail.Code != "1004" {
		t.Fatalf("unexpected MiniMax-style error detail: %+v", detail)
	}
}
