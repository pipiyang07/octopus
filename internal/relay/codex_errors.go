package relay

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

func writeResponsesUpstreamError(c *gin.Context, hb *earlyHeartbeat, status int, body string) {
	detail := chatErrorToResponsesErrorDetail(status, body)
	if hb != nil && hb.HeaderWritten() {
		payload, _ := json.Marshal(map[string]any{
			"type":    "error",
			"code":    detail.Code,
			"message": detail.Message,
			"param":   detail.Param,
		})
		_, _ = c.Writer.Write([]byte("event: error\ndata: "))
		_, _ = c.Writer.Write(payload)
		_, _ = c.Writer.Write([]byte("\n\n"))
		c.Writer.Flush()
		return
	}
	c.JSON(status, map[string]any{"error": detail})
}

func chatErrorToResponsesErrorDetail(status int, body string) model.ErrorDetail {
	detail := model.ErrorDetail{
		Code:    http.StatusText(status),
		Message: "upstream error",
		Type:    "upstream_error",
	}
	if strings.TrimSpace(body) == "" {
		return detail
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		detail.Message = strings.TrimSpace(body)
		return detail
	}

	if errObject, ok := parsed["error"].(map[string]any); ok {
		detail.Message = firstString(errObject, "message", "msg")
		detail.Type = firstString(errObject, "type")
		detail.Code = firstStringOrNumber(errObject, "code")
		detail.Param = firstString(errObject, "param")
	} else {
		detail.Message = firstString(parsed, "message", "msg", "detail")
		detail.Code = firstStringOrNumber(parsed, "code")
		detail.Param = firstString(parsed, "param")
		if baseResp, ok := parsed["base_resp"].(map[string]any); ok {
			if message := firstString(baseResp, "status_msg", "message"); message != "" {
				detail.Message = message
			}
			if code := firstStringOrNumber(baseResp, "status_code"); code != "" {
				detail.Code = code
			}
		}
	}

	if detail.Message == "" {
		detail.Message = "upstream error"
	}
	if detail.Type == "" {
		detail.Type = "upstream_error"
	}
	if detail.Code == "" {
		detail.Code = http.StatusText(status)
	}
	return detail
}

func firstStringOrNumber(value map[string]any, key string) string {
	switch typed := value[key].(type) {
	case string:
		return strings.TrimSpace(typed)
	case float64:
		return strconv.FormatInt(int64(typed), 10)
	default:
		return ""
	}
}

func firstString(value map[string]any, keys ...string) string {
	for _, key := range keys {
		if text, ok := value[key].(string); ok && strings.TrimSpace(text) != "" {
			return strings.TrimSpace(text)
		}
	}
	return ""
}
