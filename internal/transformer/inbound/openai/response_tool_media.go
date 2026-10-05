package openai

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/bestruirui/octopus/internal/transformer/model"
)

const (
	responsesToolMediaMovedMarker = "[octopus: tool result media moved to the following user message]"
	responsesWholeDataURLMinBytes = 8 * 1024
	responsesBase64ishMinBytes    = 16 * 1024
	responsesMaxMediaDepth        = 32
)

func planResponsesToolOutput(raw json.RawMessage) (string, []model.MessageContentPart, error) {
	if len(raw) == 0 {
		return "", nil, nil
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return string(raw), nil, nil
	}

	if text, isString := value.(string); isString {
		trimmed := strings.TrimSpace(text)
		if wholeImageDataURL(trimmed) != "" {
			return responsesToolMediaMovedMarker, []model.MessageContentPart{{
				Type:     "image_url",
				ImageURL: &model.ImageURL{URL: trimmed},
			}}, nil
		}
		if trimmed == "" || !json.Valid([]byte(trimmed)) {
			return text, nil, nil
		}
		var nested any
		if err := json.Unmarshal([]byte(trimmed), &nested); err != nil {
			return text, nil, nil
		}
		transformed, mediaParts, changed := stripResponsesToolMedia(nested, 0)
		if !changed {
			return canonicalJSONStringIfParseable(trimmed), nil, nil
		}
		clampResponsesMediaAdjacentStrings(transformed)
		rewritten, err := json.Marshal(transformed)
		if err != nil {
			return text, nil, nil
		}
		return string(rewritten), mediaParts, nil
	}

	transformed, mediaParts, changed := stripResponsesToolMedia(value, 0)
	if !changed {
		rewritten, err := json.Marshal(value)
		if err != nil {
			return string(raw), nil, nil
		}
		return string(rewritten), nil, nil
	}
	clampResponsesMediaAdjacentStrings(transformed)
	rewritten, err := json.Marshal(transformed)
	if err != nil {
		return string(raw), mediaParts, nil
	}
	return string(rewritten), mediaParts, nil
}

func rewriteResponsesToolOutputContainer(rawItem, rawOutput json.RawMessage) (string, []model.MessageContentPart, error) {
	if len(rawItem) == 0 || len(rawOutput) == 0 {
		return string(rawItem), nil, nil
	}

	var item map[string]any
	if err := json.Unmarshal(rawItem, &item); err != nil {
		return string(rawItem), nil, nil
	}
	var output any
	if err := json.Unmarshal(rawOutput, &output); err != nil {
		return string(rawItem), nil, nil
	}

	transformed, mediaParts, changed := stripResponsesToolMedia(output, 0)
	if !changed {
		return string(rawItem), nil, nil
	}
	clampResponsesMediaAdjacentStrings(transformed)
	item["output"] = transformed
	rewritten, err := json.Marshal(item)
	if err != nil {
		return string(rawItem), mediaParts, nil
	}
	return string(rewritten), mediaParts, nil
}

func stripResponsesToolMedia(value any, depth int) (any, []model.MessageContentPart, bool) {
	if depth > responsesMaxMediaDepth {
		return value, nil, false
	}

	switch typed := value.(type) {
	case string:
		trimmed := strings.TrimSpace(typed)
		if dataURL := wholeImageDataURL(trimmed); dataURL != "" {
			return responsesToolMediaMovedMarker, []model.MessageContentPart{{
				Type:     "image_url",
				ImageURL: &model.ImageURL{URL: dataURL},
			}}, true
		}
		if trimmed == "" || !json.Valid([]byte(trimmed)) {
			return typed, nil, false
		}
		var nested any
		if err := json.Unmarshal([]byte(trimmed), &nested); err != nil {
			return typed, nil, false
		}
		transformed, mediaParts, changed := stripResponsesToolMedia(nested, depth+1)
		if !changed {
			return typed, nil, false
		}
		rewritten, err := json.Marshal(transformed)
		if err != nil {
			return typed, mediaParts, true
		}
		return string(rewritten), mediaParts, true

	case []any:
		changed := false
		mediaParts := make([]model.MessageContentPart, 0)
		for index, item := range typed {
			transformed, itemMedia, itemChanged := stripResponsesToolMedia(item, depth+1)
			if itemChanged {
				typed[index] = transformed
				mediaParts = append(mediaParts, itemMedia...)
				changed = true
			}
		}
		return typed, mediaParts, changed

	case map[string]any:
		if mediaPart, ok := responsesToolMediaPart(typed); ok {
			return map[string]any{
				"type": "text",
				"text": responsesToolMediaMovedMarker,
			}, []model.MessageContentPart{mediaPart}, true
		}
		if content, exists := typed["content"]; exists {
			transformed, mediaParts, changed := stripResponsesToolMedia(content, depth+1)
			if changed {
				typed["content"] = transformed
			}
			return typed, mediaParts, changed
		}
		return typed, nil, false

	default:
		return value, nil, false
	}
}

func responsesToolMediaPart(item map[string]any) (model.MessageContentPart, bool) {
	switch itemType, _ := item["type"].(string); itemType {
	case "input_image", "image_url":
		if part, ok := responsesImagePart(item); ok {
			return part, true
		}
	case "input_file":
		if part, ok := responsesFilePart(item); ok {
			return part, true
		}
	case "input_audio":
		if part, ok := responsesAudioPart(item); ok {
			return part, true
		}
	case "image":
		if part, ok := responsesTypedImagePart(item); ok {
			return part, true
		}
	case "":
		if _, exists := item["type"]; !exists {
			if part, ok := responsesLooseImagePart(item); ok {
				return part, true
			}
		}
	}
	return model.MessageContentPart{}, false
}

func responsesImagePart(item map[string]any) (model.MessageContentPart, bool) {
	imageURL, detail, ok := normalizedResponsesImageURL(item)
	if !ok {
		return model.MessageContentPart{}, false
	}
	return model.MessageContentPart{
		Type:     "image_url",
		ImageURL: &model.ImageURL{URL: imageURL, Detail: detail},
	}, true
}

func responsesTypedImagePart(item map[string]any) (model.MessageContentPart, bool) {
	if source, ok := item["source"].(map[string]any); ok {
		mediaType := firstResponsesString(source, "media_type", "mime_type", "mimeType")
		if mediaType != "" && !strings.HasPrefix(strings.ToLower(mediaType), "image/") {
			return model.MessageContentPart{}, false
		}
		if url := firstResponsesString(source, "url"); url != "" {
			return model.MessageContentPart{
				Type:     "image_url",
				ImageURL: &model.ImageURL{URL: url},
			}, true
		}
		if data := firstResponsesString(source, "data"); data != "" {
			if mediaType == "" {
				mediaType = "image/png"
			}
			url := data
			if !strings.HasPrefix(strings.ToLower(data), "data:image/") {
				url = "data:" + mediaType + ";base64," + data
			}
			return model.MessageContentPart{
				Type:     "image_url",
				ImageURL: &model.ImageURL{URL: url},
			}, true
		}
	}

	data := firstResponsesString(item, "data")
	mediaType := firstResponsesString(item, "mimeType", "mime_type")
	if data == "" || !strings.HasPrefix(strings.ToLower(mediaType), "image/") {
		return model.MessageContentPart{}, false
	}
	url := data
	if !strings.HasPrefix(strings.ToLower(data), "data:image/") {
		url = "data:" + mediaType + ";base64," + data
	}
	return model.MessageContentPart{
		Type:     "image_url",
		ImageURL: &model.ImageURL{URL: url},
	}, true
}

func responsesLooseImagePart(item map[string]any) (model.MessageContentPart, bool) {
	if _, exists := item["type"]; exists {
		return model.MessageContentPart{}, false
	}
	if imageURL := wholeImageDataURL(firstResponsesString(item, "image_url")); imageURL != "" {
		return model.MessageContentPart{
			Type:     "image_url",
			ImageURL: &model.ImageURL{URL: imageURL},
		}, true
	}
	return model.MessageContentPart{}, false
}

func normalizedResponsesImageURL(item map[string]any) (string, *string, bool) {
	switch typed := item["image_url"].(type) {
	case string:
		if strings.TrimSpace(typed) == "" {
			return "", nil, false
		}
		return strings.TrimSpace(typed), responsesImageDetail(item["detail"]), true
	case map[string]any:
		url := firstResponsesString(typed, "url")
		if url == "" {
			return "", nil, false
		}
		detail := firstResponsesString(typed, "detail")
		if detail == "" {
			detail = firstResponsesString(item, "detail")
		}
		if strings.EqualFold(detail, "original") {
			detail = "auto"
		}
		if detail == "" {
			return url, nil, true
		}
		return url, &detail, true
	default:
		return "", nil, false
	}
}

func responsesImageDetail(value any) *string {
	detail, _ := value.(string)
	if strings.EqualFold(detail, "original") {
		detail = "auto"
	}
	if detail == "" {
		return nil
	}
	return &detail
}

func responsesFilePart(item map[string]any) (model.MessageContentPart, bool) {
	fileID := firstResponsesString(item, "file_id")
	fileData := firstResponsesString(item, "file_data")
	if fileID == "" && fileData == "" {
		return model.MessageContentPart{}, false
	}
	return model.MessageContentPart{
		Type: "file",
		File: &model.File{
			FileID:   fileID,
			FileData: fileData,
			Filename: firstResponsesString(item, "filename"),
		},
	}, true
}

func responsesAudioPart(item map[string]any) (model.MessageContentPart, bool) {
	audio, ok := item["input_audio"].(map[string]any)
	if !ok {
		return model.MessageContentPart{}, false
	}
	data := firstResponsesString(audio, "data")
	if data == "" {
		return model.MessageContentPart{}, false
	}
	return model.MessageContentPart{
		Type:  "input_audio",
		Audio: &model.Audio{Data: data, Format: firstResponsesString(audio, "format")},
	}, true
}

func wholeImageDataURL(value string) string {
	trimmed := strings.TrimSpace(value)
	if len(trimmed) < responsesWholeDataURLMinBytes {
		return ""
	}
	if !strings.HasPrefix(strings.ToLower(trimmed), "data:image/") {
		return ""
	}
	if !strings.Contains(strings.ToLower(trimmed), ";base64,") {
		return ""
	}
	return trimmed
}

func clampResponsesMediaAdjacentStrings(value any) {
	switch typed := value.(type) {
	case string:
		trimmed := strings.TrimSpace(typed)
		shouldOmit := (len(trimmed) >= responsesWholeDataURLMinBytes && strings.HasPrefix(strings.ToLower(trimmed), "data:")) ||
			looksLikeResponsesBase64(trimmed)
		if shouldOmit {
			_ = typed
			return
		}
	case []any:
		for index, item := range typed {
			clampResponsesMediaAdjacentStrings(item)
			if text, ok := item.(string); ok {
				trimmed := strings.TrimSpace(text)
				if (len(trimmed) >= responsesWholeDataURLMinBytes && strings.HasPrefix(strings.ToLower(trimmed), "data:")) ||
					looksLikeResponsesBase64(trimmed) {
					typed[index] = fmt.Sprintf("[octopus: omitted %d bytes]", len(text))
				}
			}
		}
	case map[string]any:
		for key, item := range typed {
			clampResponsesMediaAdjacentStrings(item)
			if text, ok := item.(string); ok {
				trimmed := strings.TrimSpace(text)
				if (len(trimmed) >= responsesWholeDataURLMinBytes && strings.HasPrefix(strings.ToLower(trimmed), "data:")) ||
					looksLikeResponsesBase64(trimmed) {
					typed[key] = fmt.Sprintf("[octopus: omitted %d bytes]", len(text))
				}
			}
		}
	}
}

func looksLikeResponsesBase64(value string) bool {
	if len(value) < responsesBase64ishMinBytes {
		return false
	}
	for _, char := range value {
		if !((char >= 'a' && char <= 'z') ||
			(char >= 'A' && char <= 'Z') ||
			(char >= '0' && char <= '9') ||
			char == '+' || char == '/' || char == '=') {
			return false
		}
	}
	return true
}

func firstResponsesString(value map[string]any, keys ...string) string {
	for _, key := range keys {
		if text, ok := value[key].(string); ok && strings.TrimSpace(text) != "" {
			return strings.TrimSpace(text)
		}
	}
	return ""
}
