package conf

import (
	"os/exec"
	"runtime/debug"
	"strings"
)

var (
	Version   = "dev"
	Commit    = "unknown"
	BuildTime = "unknown"
	Author    = "pipiyang"
	Repo      = "https://github.com/pipiyang07/octopus"
)

func init() {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return
	}

	commit, buildTime := metadataFromBuildInfo(info)
	if commit == "" || buildTime == "" {
		gitCommit, gitBuildTime := metadataFromGit()
		if commit == "" {
			commit = gitCommit
		}
		if buildTime == "" {
			buildTime = gitBuildTime
		}
	}
	if Commit == "unknown" && commit != "" {
		Commit = commit
	}
	if BuildTime == "unknown" && buildTime != "" {
		BuildTime = buildTime
	}
}

func metadataFromBuildInfo(info *debug.BuildInfo) (commit, buildTime string) {
	if info == nil {
		return "", ""
	}

	settings := make(map[string]string, len(info.Settings))
	for _, setting := range info.Settings {
		settings[setting.Key] = setting.Value
	}

	commit = strings.TrimSpace(settings["vcs.revision"])
	buildTime = strings.TrimSpace(settings["vcs.time"])
	if commit != "" && strings.EqualFold(strings.TrimSpace(settings["vcs.modified"]), "true") {
		commit += "-dirty"
	}
	return commit, buildTime
}

func metadataFromGit() (commit, buildTime string) {
	commitOutput, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "", ""
	}
	commit = strings.TrimSpace(string(commitOutput))
	if commit == "" {
		return "", ""
	}

	timeOutput, err := exec.Command("git", "log", "-1", "--format=%cI").Output()
	if err == nil {
		buildTime = strings.TrimSpace(string(timeOutput))
	}

	statusOutput, err := exec.Command("git", "status", "--porcelain").Output()
	if err == nil && strings.TrimSpace(string(statusOutput)) != "" {
		commit += "-dirty"
	}
	return commit, buildTime
}
