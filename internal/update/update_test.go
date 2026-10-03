package update

import "testing"

func TestParseLatestInfoTreatsNotFoundAsNoRelease(t *testing.T) {
	info, err := parseLatestInfo([]byte(`{"message":"Not Found"}`))
	if err != nil {
		t.Fatalf("expected Not Found to be treated as no release, got error: %v", err)
	}
	if info == nil {
		t.Fatal("expected an empty release info, got nil")
	}
	if info.TagName != "" || info.PublishedAt != "" || info.Body != "" || info.Message != "" {
		t.Fatalf("expected all release fields to stay empty, got %#v", info)
	}
}

func TestParseLatestInfoUsesRealReleaseFields(t *testing.T) {
	info, err := parseLatestInfo([]byte(`{"tag_name":"v1.2.3","published_at":"2026-10-03T00:00:00Z","body":"release notes"}`))
	if err != nil {
		t.Fatalf("expected real release metadata to parse, got error: %v", err)
	}
	if info.TagName != "v1.2.3" || info.PublishedAt != "2026-10-03T00:00:00Z" || info.Body != "release notes" {
		t.Fatalf("unexpected release info: %#v", info)
	}
}
