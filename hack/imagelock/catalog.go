// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/google/go-containerregistry/pkg/name"
	"sigs.k8s.io/yaml"
)

const (
	catalogAPIVersion  = "imagelock.kai-scheduler.io/v1alpha1"
	versionPlaceholder = "{{version}}"
	wildcardSelector   = "*"
)

var catalogName = regexp.MustCompile(`^[a-z0-9]+(?:[-.][a-z0-9]+)*$`)

type imageCatalog struct {
	APIVersion   string             `json:"apiVersion"`
	Name         string             `json:"name"`
	ArtifactName string             `json:"artifactName"`
	Repository   string             `json:"repository"`
	Images       []catalogImage     `json:"images"`
	Profiles     []catalogProfile   `json:"profiles"`
	Platforms    []platform         `json:"platforms"`
	Exclude      []catalogExclusion `json:"exclude"`
}

type catalogImage struct {
	Name string `json:"name"`
}

type catalogProfile struct {
	Name        string `json:"name"`
	TagTemplate string `json:"tagTemplate"`
}

type catalogExclusion struct {
	Image    string `json:"image"`
	Profile  string `json:"profile"`
	Platform string `json:"platform"`
	Reason   string `json:"reason"`
}

type imageChartValues struct {
	Images map[string]struct {
		Repository string `json:"repository"`
	} `json:"images"`
}

type lockTarget struct {
	profile  string
	platform platform
}

type plannedImage struct {
	profile string
	chartImage
	platforms []platform
}

type releasePlan struct {
	targets          []lockTarget
	images           []plannedImage
	combinationCount int
}

func loadCatalog(path string) (imageCatalog, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return imageCatalog{}, fmt.Errorf("read catalog %s: %w", path, err)
	}
	var catalog imageCatalog
	if err := yaml.UnmarshalStrict(contents, &catalog); err != nil {
		return imageCatalog{}, fmt.Errorf("parse catalog %s: %w", path, err)
	}
	if err := validateCatalog(catalog); err != nil {
		return imageCatalog{}, fmt.Errorf("validate catalog %s: %w", path, err)
	}
	return catalog, nil
}

func validateCatalog(catalog imageCatalog) error {
	if catalog.APIVersion != catalogAPIVersion {
		return fmt.Errorf("apiVersion must be %q, got %q", catalogAPIVersion, catalog.APIVersion)
	}
	if !catalogName.MatchString(catalog.Name) {
		return fmt.Errorf("name %q must contain lowercase letters, digits, dots, or hyphens", catalog.Name)
	}
	if !catalogName.MatchString(catalog.ArtifactName) {
		return fmt.Errorf("artifactName %q must contain lowercase letters, digits, dots, or hyphens", catalog.ArtifactName)
	}
	if strings.HasSuffix(catalog.Repository, "/") {
		return fmt.Errorf("repository %q must not have a trailing slash", catalog.Repository)
	}
	if _, err := name.NewRepository(catalog.Repository); err != nil {
		return fmt.Errorf("invalid repository %q: %w", catalog.Repository, err)
	}

	images, err := uniqueCatalogNames("image", imageNames(catalog.Images))
	if err != nil {
		return err
	}
	profiles, err := uniqueCatalogNames("profile", profileNames(catalog.Profiles))
	if err != nil {
		return err
	}
	platforms, err := uniquePlatforms(catalog.Platforms)
	if err != nil {
		return err
	}

	for _, profile := range catalog.Profiles {
		if strings.Count(profile.TagTemplate, versionPlaceholder) != 1 {
			return fmt.Errorf("profile %q tagTemplate must contain %q exactly once", profile.Name, versionPlaceholder)
		}
	}

	seenExclusions := map[string]struct{}{}
	for index, exclusion := range catalog.Exclude {
		if strings.TrimSpace(exclusion.Reason) == "" {
			return fmt.Errorf("exclude[%d] must include a reason", index)
		}
		if err := validateSelector("image", exclusion.Image, images); err != nil {
			return fmt.Errorf("exclude[%d]: %w", index, err)
		}
		if err := validateSelector("profile", exclusion.Profile, profiles); err != nil {
			return fmt.Errorf("exclude[%d]: %w", index, err)
		}
		if err := validateSelector("platform", exclusion.Platform, platforms); err != nil {
			return fmt.Errorf("exclude[%d]: %w", index, err)
		}
		key := exclusion.Image + "\x00" + exclusion.Profile + "\x00" + exclusion.Platform
		if _, duplicate := seenExclusions[key]; duplicate {
			return fmt.Errorf("exclude[%d] duplicates an earlier exclusion", index)
		}
		seenExclusions[key] = struct{}{}
	}
	return nil
}

func buildReleasePlan(catalog imageCatalog, version string) (releasePlan, error) {
	matchedExclusions := make([]int, len(catalog.Exclude))
	targetImageCounts := map[lockTarget]int{}
	imageCombinationCounts := map[string]int{}
	planned := map[string]*plannedImage{}
	combinationCount := 0

	for _, profile := range catalog.Profiles {
		tag := strings.Replace(profile.TagTemplate, versionPlaceholder, version, 1)
		for _, image := range catalog.Images {
			repository := catalog.Repository + "/" + image.Name
			if _, err := name.NewTag(repository + ":" + tag); err != nil {
				return releasePlan{}, fmt.Errorf("profile %q produces invalid reference for image %q: %w", profile.Name, image.Name, err)
			}
			for _, imagePlatform := range catalog.Platforms {
				matching := matchingExclusions(catalog.Exclude, image.Name, profile.Name, imagePlatform.String())
				if len(matching) > 1 {
					return releasePlan{}, fmt.Errorf("%s/%s/%s is covered by overlapping exclusions %v", image.Name, profile.Name, imagePlatform, matching)
				}
				if len(matching) == 1 {
					matchedExclusions[matching[0]]++
					continue
				}

				target := lockTarget{profile: profile.Name, platform: imagePlatform}
				targetImageCounts[target]++
				imageCombinationCounts[image.Name]++
				combinationCount++
				key := profile.Name + "\x00" + image.Name
				entry := planned[key]
				if entry == nil {
					entry = &plannedImage{
						profile: profile.Name,
						chartImage: chartImage{
							name: image.Name,
							repo: repository,
							tag:  tag,
						},
					}
					planned[key] = entry
				}
				entry.platforms = append(entry.platforms, imagePlatform)
			}
		}
	}

	for index, matches := range matchedExclusions {
		if matches == 0 {
			return releasePlan{}, fmt.Errorf("exclude[%d] does not match any image/profile/platform combination", index)
		}
	}
	for _, image := range catalog.Images {
		if imageCombinationCounts[image.Name] == 0 {
			return releasePlan{}, fmt.Errorf("exclusions remove image %q from every profile/platform combination", image.Name)
		}
	}

	plan := releasePlan{combinationCount: combinationCount}
	for _, profile := range catalog.Profiles {
		for _, imagePlatform := range catalog.Platforms {
			target := lockTarget{profile: profile.Name, platform: imagePlatform}
			if targetImageCounts[target] == 0 {
				return releasePlan{}, fmt.Errorf("exclusions leave %s/%s with no images", profile.Name, imagePlatform)
			}
			plan.targets = append(plan.targets, target)
		}
	}
	for _, image := range planned {
		plan.images = append(plan.images, *image)
	}
	sort.Slice(plan.targets, func(i, j int) bool {
		if plan.targets[i].profile != plan.targets[j].profile {
			return plan.targets[i].profile < plan.targets[j].profile
		}
		return plan.targets[i].platform.String() < plan.targets[j].platform.String()
	})
	sort.Slice(plan.images, func(i, j int) bool {
		if plan.images[i].profile != plan.images[j].profile {
			return plan.images[i].profile < plan.images[j].profile
		}
		return plan.images[i].name < plan.images[j].name
	})
	return plan, nil
}

func validateExpectedImages(catalog imageCatalog, expected []string) error {
	if len(expected) == 0 {
		return nil
	}
	want := append([]string(nil), expected...)
	got := imageNames(catalog.Images)
	sort.Strings(want)
	sort.Strings(got)
	for index := 1; index < len(want); index++ {
		if want[index] == want[index-1] {
			return fmt.Errorf("expected image %q was provided more than once", want[index])
		}
	}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		return fmt.Errorf("catalog images %v do not match build images %v", got, want)
	}
	return nil
}

func validateChartValues(catalog imageCatalog, path string) error {
	contents, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read chart values %s: %w", path, err)
	}
	var values imageChartValues
	if err := yaml.Unmarshal(contents, &values); err != nil {
		return fmt.Errorf("parse chart values %s: %w", path, err)
	}
	names := make([]string, 0, len(values.Images))
	for imageName := range values.Images {
		names = append(names, imageName)
	}
	if err := validateExpectedImages(catalog, names); err != nil {
		return fmt.Errorf("chart values %s: %w", path, err)
	}
	for _, image := range catalog.Images {
		want := catalog.Repository + "/" + image.Name
		if got := values.Images[image.Name].Repository; got != want {
			return fmt.Errorf("chart values %s: image %q repository is %q, want %q", path, image.Name, got, want)
		}
	}
	return nil
}

func imageNames(images []catalogImage) []string {
	names := make([]string, 0, len(images))
	for _, image := range images {
		names = append(names, image.Name)
	}
	return names
}

func profileNames(profiles []catalogProfile) []string {
	names := make([]string, 0, len(profiles))
	for _, profile := range profiles {
		names = append(names, profile.Name)
	}
	return names
}

func uniqueCatalogNames(kind string, names []string) (map[string]struct{}, error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("catalog must contain at least one %s", kind)
	}
	unique := make(map[string]struct{}, len(names))
	for _, value := range names {
		if !catalogName.MatchString(value) {
			return nil, fmt.Errorf("%s name %q must contain lowercase letters, digits, dots, or hyphens", kind, value)
		}
		if _, duplicate := unique[value]; duplicate {
			return nil, fmt.Errorf("duplicate %s %q", kind, value)
		}
		unique[value] = struct{}{}
	}
	return unique, nil
}

func uniquePlatforms(entries []platform) (map[string]struct{}, error) {
	if len(entries) == 0 {
		return nil, errors.New("catalog must contain at least one platform")
	}
	unique := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		if entry.OS == "" || entry.Architecture == "" || strings.Contains(entry.OS, "/") || strings.Contains(entry.Architecture, "/") {
			return nil, fmt.Errorf("invalid platform %q, want non-empty os and architecture", entry)
		}
		if _, duplicate := unique[entry.String()]; duplicate {
			return nil, fmt.Errorf("duplicate platform %q", entry)
		}
		unique[entry.String()] = struct{}{}
	}
	return unique, nil
}

func validateSelector(kind, selector string, known map[string]struct{}) error {
	if selector == wildcardSelector {
		return nil
	}
	if selector == "" {
		return fmt.Errorf("%s selector must not be empty", kind)
	}
	if _, exists := known[selector]; !exists {
		return fmt.Errorf("unknown %s selector %q", kind, selector)
	}
	return nil
}

func matchingExclusions(exclusions []catalogExclusion, image, profile, imagePlatform string) []int {
	var matches []int
	for index, exclusion := range exclusions {
		if selectorMatches(exclusion.Image, image) &&
			selectorMatches(exclusion.Profile, profile) &&
			selectorMatches(exclusion.Platform, imagePlatform) {
			matches = append(matches, index)
		}
	}
	return matches
}

func selectorMatches(selector, value string) bool {
	return selector == wildcardSelector || selector == value
}
