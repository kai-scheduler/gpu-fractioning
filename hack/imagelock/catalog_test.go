// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func validCatalog() imageCatalog {
	return imageCatalog{
		APIVersion:   catalogAPIVersion,
		Name:         "kai-gpu-fractioning",
		ArtifactName: "gpu-fractioning",
		Repository:   "ghcr.io/kai-scheduler/kai-gpu-fractioning",
		Images: []catalogImage{
			{Name: "operator"},
			{Name: "fractiond"},
			{Name: "metricsd"},
			{Name: "mpsd"},
		},
		Profiles: []catalogProfile{
			{Name: "standard", TagTemplate: versionPlaceholder},
			{Name: "fips", TagTemplate: versionPlaceholder + "-fips"},
		},
		Platforms: []platform{
			{OS: "linux", Architecture: "amd64"},
			{OS: "linux", Architecture: "arm64"},
		},
	}
}

func TestCommittedCatalogDefinesCompleteReleaseMatrix(t *testing.T) {
	catalog, err := loadCatalog("catalog.yaml")
	if err != nil {
		t.Fatal(err)
	}
	plan, err := buildReleasePlan(catalog, "v1.2.3")
	if err != nil {
		t.Fatal(err)
	}
	if plan.combinationCount != 16 || len(plan.targets) != 4 || len(plan.images) != 8 {
		t.Fatalf("plan has %d combinations, %d targets, %d profile images; want 16, 4, 8",
			plan.combinationCount, len(plan.targets), len(plan.images))
	}

	wantTags := map[string]string{"standard": "v1.2.3", "fips": "v1.2.3-fips"}
	for _, image := range plan.images {
		if image.tag != wantTags[image.profile] {
			t.Errorf("%s/%s tag = %q, want %q", image.profile, image.name, image.tag, wantTags[image.profile])
		}
		if len(image.platforms) != 2 {
			t.Errorf("%s/%s has %d platforms, want 2", image.profile, image.name, len(image.platforms))
		}
	}
}

func TestBuildReleasePlanAppliesStructuredExclusions(t *testing.T) {
	catalog := validCatalog()
	catalog.Exclude = []catalogExclusion{
		{Image: "mpsd", Profile: wildcardSelector, Platform: "linux/amd64", Reason: "not built on amd64"},
		{Image: "fractiond", Profile: "fips", Platform: wildcardSelector, Reason: "no FIPS variant"},
		{Image: "metricsd", Profile: "fips", Platform: "linux/amd64", Reason: "unsupported combination"},
	}
	if err := validateCatalog(catalog); err != nil {
		t.Fatal(err)
	}
	plan, err := buildReleasePlan(catalog, "v1.2.3")
	if err != nil {
		t.Fatal(err)
	}
	if plan.combinationCount != 11 {
		t.Fatalf("combination count = %d, want 11", plan.combinationCount)
	}

	counts := map[lockTarget]int{}
	for _, image := range plan.images {
		for _, imagePlatform := range image.platforms {
			counts[lockTarget{profile: image.profile, platform: imagePlatform}]++
		}
	}
	want := map[lockTarget]int{
		{profile: "standard", platform: platform{OS: "linux", Architecture: "amd64"}}: 3,
		{profile: "standard", platform: platform{OS: "linux", Architecture: "arm64"}}: 4,
		{profile: "fips", platform: platform{OS: "linux", Architecture: "amd64"}}:     1,
		{profile: "fips", platform: platform{OS: "linux", Architecture: "arm64"}}:     3,
	}
	for target, wantCount := range want {
		if counts[target] != wantCount {
			t.Errorf("%s/%s has %d images, want %d", target.profile, target.platform, counts[target], wantCount)
		}
	}
}

func TestValidateExpectedImagesMatchesBuildInventory(t *testing.T) {
	catalog := validCatalog()
	tests := []struct {
		name     string
		expected []string
		wantErr  string
	}{
		{name: "same set in different order", expected: []string{"mpsd", "operator", "metricsd", "fractiond"}},
		{name: "missing build image", expected: []string{"operator", "fractiond", "metricsd"}, wantErr: "do not match build images"},
		{name: "extra build image", expected: []string{"operator", "fractiond", "metricsd", "mpsd", "whateverd"}, wantErr: "do not match build images"},
		{name: "duplicate build image", expected: []string{"operator", "fractiond", "metricsd", "mpsd", "mpsd"}, wantErr: "provided more than once"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateExpectedImages(catalog, test.expected)
			if test.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("want error containing %q, got %v", test.wantErr, err)
			}
		})
	}
}

func TestValidateChartValuesMatchesCatalog(t *testing.T) {
	valid := `
images:
  operator:
    repository: ghcr.io/kai-scheduler/kai-gpu-fractioning/operator
  fractiond:
    repository: ghcr.io/kai-scheduler/kai-gpu-fractioning/fractiond
  metricsd:
    repository: ghcr.io/kai-scheduler/kai-gpu-fractioning/metricsd
  mpsd:
    repository: ghcr.io/kai-scheduler/kai-gpu-fractioning/mpsd
`
	tests := []struct {
		name     string
		contents string
		wantErr  string
	}{
		{name: "matching values", contents: valid},
		{name: "missing image", contents: strings.Replace(valid, "  mpsd:\n    repository: ghcr.io/kai-scheduler/kai-gpu-fractioning/mpsd\n", "", 1), wantErr: "do not match build images"},
		{name: "extra image", contents: valid + "  whateverd:\n    repository: ghcr.io/kai-scheduler/kai-gpu-fractioning/whateverd\n", wantErr: "do not match build images"},
		{name: "renamed repository", contents: strings.Replace(valid, "kai-gpu-fractioning/mpsd", "kai-gpu-fractioning/renamed-mpsd", 1), wantErr: `image "mpsd" repository`},
		{name: "malformed values", contents: "images: [unterminated", wantErr: "parse chart values"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "values.yaml")
			if err := os.WriteFile(path, []byte(test.contents), 0o600); err != nil {
				t.Fatal(err)
			}
			err := validateChartValues(validCatalog(), path)
			if test.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("want error containing %q, got %v", test.wantErr, err)
			}
		})
	}
}

func TestValidateCatalogRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*imageCatalog)
		wantErr string
	}{
		{name: "wrong API version", mutate: func(c *imageCatalog) { c.APIVersion = "v1" }, wantErr: "apiVersion"},
		{name: "bad artifact name", mutate: func(c *imageCatalog) { c.ArtifactName = "GPU Fractioning" }, wantErr: "artifactName"},
		{name: "repository trailing slash", mutate: func(c *imageCatalog) { c.Repository += "/" }, wantErr: "trailing slash"},
		{name: "duplicate image", mutate: func(c *imageCatalog) { c.Images = append(c.Images, c.Images[0]) }, wantErr: "duplicate image"},
		{name: "duplicate profile", mutate: func(c *imageCatalog) { c.Profiles = append(c.Profiles, c.Profiles[0]) }, wantErr: "duplicate profile"},
		{name: "duplicate platform", mutate: func(c *imageCatalog) { c.Platforms = append(c.Platforms, c.Platforms[0]) }, wantErr: "duplicate platform"},
		{name: "missing version placeholder", mutate: func(c *imageCatalog) { c.Profiles[0].TagTemplate = "latest" }, wantErr: "exactly once"},
		{name: "repeated version placeholder", mutate: func(c *imageCatalog) { c.Profiles[0].TagTemplate = versionPlaceholder + versionPlaceholder }, wantErr: "exactly once"},
		{
			name: "unknown profile selector",
			mutate: func(c *imageCatalog) {
				c.Exclude = []catalogExclusion{{Image: "mpsd", Profile: "fibs", Platform: wildcardSelector, Reason: "typo"}}
			},
			wantErr: `unknown profile selector "fibs"`,
		},
		{
			name: "missing exclusion reason",
			mutate: func(c *imageCatalog) {
				c.Exclude = []catalogExclusion{{Image: "mpsd", Profile: "fips", Platform: wildcardSelector}}
			},
			wantErr: "must include a reason",
		},
		{
			name: "duplicate exclusion",
			mutate: func(c *imageCatalog) {
				exclusion := catalogExclusion{Image: "mpsd", Profile: "fips", Platform: wildcardSelector, Reason: "unsupported"}
				c.Exclude = []catalogExclusion{exclusion, exclusion}
			},
			wantErr: "duplicates an earlier exclusion",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			catalog := validCatalog()
			test.mutate(&catalog)
			if err := validateCatalog(catalog); err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("want error containing %q, got %v", test.wantErr, err)
			}
		})
	}
}

func TestBuildReleasePlanRejectsUnsafeExclusions(t *testing.T) {
	tests := []struct {
		name       string
		exclusions []catalogExclusion
		wantErr    string
	}{
		{
			name: "overlapping exclusions",
			exclusions: []catalogExclusion{
				{Image: "mpsd", Profile: wildcardSelector, Platform: "linux/amd64", Reason: "first"},
				{Image: wildcardSelector, Profile: "standard", Platform: "linux/amd64", Reason: "second"},
			},
			wantErr: "overlapping exclusions",
		},
		{
			name: "empty lock",
			exclusions: []catalogExclusion{
				{Image: wildcardSelector, Profile: "standard", Platform: "linux/amd64", Reason: "unsupported target"},
			},
			wantErr: "with no images",
		},
		{
			name: "image removed everywhere",
			exclusions: []catalogExclusion{
				{Image: "mpsd", Profile: wildcardSelector, Platform: wildcardSelector, Reason: "never applicable"},
			},
			wantErr: "remove image \"mpsd\" from every",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			catalog := validCatalog()
			catalog.Exclude = test.exclusions
			if err := validateCatalog(catalog); err != nil {
				t.Fatal(err)
			}
			if _, err := buildReleasePlan(catalog, "v1.2.3"); err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("want error containing %q, got %v", test.wantErr, err)
			}
		})
	}
}

func TestLoadCatalogRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.yaml")
	contents := []byte(`
apiVersion: imagelock.kai-scheduler.io/v1alpha1
name: kai-gpu-fractioning
artifactName: gpu-fractioning
repository: ghcr.io/kai-scheduler/kai-gpu-fractioning
images: [{name: operator}]
profiles: [{name: standard, tagTemplate: "{{version}}"}]
platforms: [{os: linux, architecture: amd64}]
fibs: true
`)
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadCatalog(path); err == nil || !strings.Contains(err.Error(), `unknown field "fibs"`) {
		t.Fatalf("want strict unknown-field error, got %v", err)
	}
}
