package conf

import (
	"runtime/debug"
	"testing"
)

func TestMetadataFromBuildInfo(t *testing.T) {
	tests := []struct {
		name       string
		settings   []debug.BuildSetting
		wantCommit string
		wantTime   string
	}{
		{
			name: "clean revision",
			settings: []debug.BuildSetting{
				{Key: "vcs.revision", Value: "1234567890abcdef"},
				{Key: "vcs.time", Value: "2026-10-06T09:21:18Z"},
			},
			wantCommit: "1234567890abcdef",
			wantTime:   "2026-10-06T09:21:18Z",
		},
		{
			name: "dirty revision",
			settings: []debug.BuildSetting{
				{Key: "vcs.revision", Value: "1234567890abcdef"},
				{Key: "vcs.modified", Value: "true"},
			},
			wantCommit: "1234567890abcdef-dirty",
		},
		{
			name: "missing metadata",
			settings: []debug.BuildSetting{
				{Key: "vcs.modified", Value: "false"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCommit, gotTime := metadataFromBuildInfo(&debug.BuildInfo{
				Settings: tt.settings,
			})
			if gotCommit != tt.wantCommit {
				t.Fatalf("commit = %q, want %q", gotCommit, tt.wantCommit)
			}
			if gotTime != tt.wantTime {
				t.Fatalf("time = %q, want %q", gotTime, tt.wantTime)
			}
		})
	}
}
