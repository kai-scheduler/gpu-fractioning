// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/registry"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/empty"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/google/go-containerregistry/pkg/v1/random"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"sigs.k8s.io/yaml"
)

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		wantVersion string
		wantCatalog string
		wantValues  string
		wantImages  []string
		wantErr     string
	}{
		{name: "release", args: []string{"--version", "v1.2.3"}, wantVersion: "v1.2.3", wantCatalog: "catalog.yaml"},
		{name: "prerelease", args: []string{"--version", "v0.2.2-rc.1"}, wantVersion: "v0.2.2-rc.1", wantCatalog: "catalog.yaml"},
		{name: "catalog override", args: []string{"--version", "v1.2.3", "--catalog", "/tmp/images.yaml"}, wantVersion: "v1.2.3", wantCatalog: "/tmp/images.yaml"},
		{name: "chart values", args: []string{"--version", "v1.2.3", "--chart-values", "/tmp/values.yaml"}, wantVersion: "v1.2.3", wantCatalog: "catalog.yaml", wantValues: "/tmp/values.yaml"},
		{name: "expected images", args: []string{"--version", "v1.2.3", "--expected-image", "operator", "--expected-image", "mpsd"}, wantVersion: "v1.2.3", wantCatalog: "catalog.yaml", wantImages: []string{"operator", "mpsd"}},
		{name: "verify default version", args: []string{"--verify-only"}, wantVersion: verifyVersion, wantCatalog: "catalog.yaml"},
		{name: "missing version", wantErr: "--version is required"},
		{name: "missing v", args: []string{"--version", "1.2.3"}, wantErr: "not a v-prefixed release version"},
		{name: "latest", args: []string{"--version", "latest"}, wantErr: "not a v-prefixed release version"},
		{name: "leading zero", args: []string{"--version", "v01.2.3"}, wantErr: "not a v-prefixed release version"},
		{name: "build metadata", args: []string{"--version", "v1.2.3+build.1"}, wantErr: "not a v-prefixed release version"},
		{name: "zero stability reads", args: []string{"--version", "v1.2.3", "--stability-reads", "0"}, wantErr: "must be at least 1"},
		{name: "positional argument", args: []string{"--version", "v1.2.3", "extra"}, wantErr: "unexpected positional arguments"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opts, err := parseFlags(test.args)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("want error containing %q, got %v", test.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if opts.version != test.wantVersion || opts.catalog != test.wantCatalog {
				t.Fatalf("options = %+v, want version %q and catalog %q", opts, test.wantVersion, test.wantCatalog)
			}
			if strings.Join(opts.expectedImages, ",") != strings.Join(test.wantImages, ",") {
				t.Fatalf("expected images = %v, want %v", opts.expectedImages, test.wantImages)
			}
			if opts.chartValues != test.wantValues {
				t.Fatalf("chart values = %q, want %q", opts.chartValues, test.wantValues)
			}
		})
	}
}

func TestResolveMultiArch(t *testing.T) {
	registryServer := httptest.NewServer(registry.New())
	defer registryServer.Close()
	ref := mustHost(t, registryServer.URL) + "/gpu-fractioning/operator:v1.2.3"

	amd64 := mustRandomImage(t)
	arm64 := mustRandomImage(t)
	attestation := mustRandomImage(t)
	index := mutate.AppendManifests(empty.Index,
		mutate.IndexAddendum{Add: amd64, Descriptor: v1.Descriptor{Platform: &v1.Platform{OS: "linux", Architecture: "amd64"}}},
		mutate.IndexAddendum{Add: arm64, Descriptor: v1.Descriptor{Platform: &v1.Platform{OS: "linux", Architecture: "arm64"}}},
		mutate.IndexAddendum{Add: attestation, Descriptor: v1.Descriptor{Platform: &v1.Platform{OS: "unknown", Architecture: "unknown"}}},
	)
	writeIndex(t, ref, index)

	platforms := []platform{{OS: "linux", Architecture: "amd64"}, {OS: "linux", Architecture: "arm64"}}
	indexDigest, perPlatform, err := (remoteResolver{stabilityReads: 1}).resolve(ref, platforms)
	if err != nil {
		t.Fatal(err)
	}
	wantIndex, _ := index.Digest()
	if indexDigest != wantIndex.String() {
		t.Errorf("index digest = %s, want %s", indexDigest, wantIndex)
	}
	amd64Digest, _ := amd64.Digest()
	arm64Digest, _ := arm64.Digest()
	if perPlatform[platforms[0]] != amd64Digest.String() || perPlatform[platforms[1]] != arm64Digest.String() {
		t.Errorf("platform digests = %v, want amd64=%s arm64=%s", perPlatform, amd64Digest, arm64Digest)
	}
}

func TestResolveRejectsSingleArch(t *testing.T) {
	registryServer := httptest.NewServer(registry.New())
	defer registryServer.Close()
	ref := mustHost(t, registryServer.URL) + "/gpu-fractioning/operator:v1.2.3"
	tag, err := name.NewTag(ref)
	if err != nil {
		t.Fatal(err)
	}
	if err := remote.Write(tag, mustRandomImage(t)); err != nil {
		t.Fatal(err)
	}
	_, _, err = (remoteResolver{stabilityReads: 1}).resolve(ref, []platform{{OS: "linux", Architecture: "amd64"}})
	if err == nil || !strings.Contains(err.Error(), "not a multi-platform image index") {
		t.Fatalf("want single-architecture rejection, got %v", err)
	}
}

func TestResolveRejectsMissingOrAmbiguousPlatform(t *testing.T) {
	tests := []struct {
		name       string
		platforms  []platform
		wantErrMsg string
	}{
		{
			name:       "missing arm64",
			platforms:  []platform{{OS: "linux", Architecture: "amd64"}},
			wantErrMsg: "no linux/arm64 manifest",
		},
		{
			name: "ambiguous amd64",
			platforms: []platform{
				{OS: "linux", Architecture: "amd64"},
				{OS: "linux", Architecture: "amd64"},
				{OS: "linux", Architecture: "arm64"},
			},
			wantErrMsg: "ambiguous linux/amd64",
		},
	}

	requested := []platform{{OS: "linux", Architecture: "amd64"}, {OS: "linux", Architecture: "arm64"}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			registryServer := httptest.NewServer(registry.New())
			defer registryServer.Close()
			ref := mustHost(t, registryServer.URL) + "/gpu-fractioning/operator:v1.2.3"

			var manifests []mutate.IndexAddendum
			for _, imagePlatform := range test.platforms {
				manifests = append(manifests, mutate.IndexAddendum{
					Add: mustRandomImage(t),
					Descriptor: v1.Descriptor{Platform: &v1.Platform{
						OS: imagePlatform.OS, Architecture: imagePlatform.Architecture,
					}},
				})
			}
			writeIndex(t, ref, mutate.AppendManifests(empty.Index, manifests...))
			_, _, err := (remoteResolver{stabilityReads: 1}).resolve(ref, requested)
			if err == nil || !strings.Contains(err.Error(), test.wantErrMsg) {
				t.Fatalf("want error containing %q, got %v", test.wantErrMsg, err)
			}
		})
	}
}

func TestValidateDigestsRequiresEveryPlannedPlatform(t *testing.T) {
	amd64 := platform{OS: "linux", Architecture: "amd64"}
	arm64 := platform{OS: "linux", Architecture: "arm64"}
	err := validateDigests(
		"sha256:"+strings.Repeat("a", 64),
		map[platform]string{amd64: "sha256:" + strings.Repeat("b", 64)},
		[]platform{amd64, arm64},
	)
	if err == nil || !strings.Contains(err.Error(), "missing linux/arm64 digest") {
		t.Fatalf("want missing digest error, got %v", err)
	}
}

func TestBuildLockSelectsProfileAndPlatform(t *testing.T) {
	amd64 := platform{OS: "linux", Architecture: "amd64"}
	indexDigest := "sha256:" + strings.Repeat("a", 64)
	childDigest := "sha256:" + strings.Repeat("b", 64)
	repository := "ghcr.io/kai-scheduler/kai-gpu-fractioning/operator"
	resolved := []resolvedImage{
		{
			profile:           "standard",
			chartImage:        chartImage{name: "operator", repo: repository, tag: "v1.2.3"},
			indexDigest:       indexDigest,
			digestPerPlatform: map[platform]string{amd64: childDigest},
		},
		{
			profile:           "fips",
			chartImage:        chartImage{name: "operator", repo: repository, tag: "v1.2.3-fips"},
			indexDigest:       indexDigest,
			digestPerPlatform: map[platform]string{amd64: "sha256:" + strings.Repeat("c", 64)},
		},
	}

	lock, err := buildLock("kai-gpu-fractioning", "v1.2.3", lockTarget{profile: "standard", platform: amd64}, resolved)
	if err != nil {
		t.Fatal(err)
	}
	if lock.Metadata.Name != "kai-gpu-fractioning" || lock.Metadata.Version != "v1.2.3" || lock.Spec.Profile != "standard" {
		t.Fatalf("lock metadata/profile = %+v/%q", lock.Metadata, lock.Spec.Profile)
	}
	if len(lock.Spec.Images) != 1 {
		t.Fatalf("images = %+v", lock.Spec.Images)
	}
	image := lock.Spec.Images[0]
	if image.Source != repository+":v1.2.3" || image.IndexDigest != indexDigest || image.Image != repository+"@"+childDigest {
		t.Fatalf("locked image = %+v", image)
	}
}

func TestLockFileName(t *testing.T) {
	tests := []struct {
		name    string
		profile string
		want    string
	}{
		{name: "standard profile is unlabeled", profile: "standard", want: "imagelock-gpu-fractioning-v1.2.3-linux-amd64.yaml"},
		{name: "variant profile is labeled", profile: "fips", want: "imagelock-gpu-fractioning-v1.2.3-fips-linux-amd64.yaml"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			lock := imageLock{}
			lock.Metadata.Version = "v1.2.3"
			lock.Spec.Profile = test.profile
			lock.Spec.Platform = platform{OS: "linux", Architecture: "amd64"}
			if got := lockFileName("gpu-fractioning", lock); got != test.want {
				t.Fatalf("lockFileName = %q, want %q", got, test.want)
			}
		})
	}
}

func TestRunGeneratesCatalogProfilePlatformMatrix(t *testing.T) {
	registryServer := httptest.NewServer(registry.New())
	defer registryServer.Close()
	host := mustHost(t, registryServer.URL)
	for _, tag := range []string{"v1.2.3", "v1.2.3-fips"} {
		index := mutate.AppendManifests(empty.Index,
			mutate.IndexAddendum{Add: mustRandomImage(t), Descriptor: v1.Descriptor{Platform: &v1.Platform{OS: "linux", Architecture: "amd64"}}},
			mutate.IndexAddendum{Add: mustRandomImage(t), Descriptor: v1.Descriptor{Platform: &v1.Platform{OS: "linux", Architecture: "arm64"}}},
		)
		writeIndex(t, host+"/gpu-fractioning/operator:"+tag, index)
	}

	catalog := validCatalog()
	catalog.Repository = host + "/gpu-fractioning"
	catalog.Images = []catalogImage{{Name: "operator"}}
	catalogBytes, err := yaml.Marshal(catalog)
	if err != nil {
		t.Fatal(err)
	}
	tempDir := t.TempDir()
	catalogPath := filepath.Join(tempDir, "catalog.yaml")
	if err := os.WriteFile(catalogPath, catalogBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	outDir := filepath.Join(tempDir, "out")
	if err := run([]string{
		"--catalog", catalogPath,
		"--version", "v1.2.3",
		"--out-dir", outDir,
		"--stability-reads", "1",
		"--expected-image", "operator",
	}); err != nil {
		t.Fatal(err)
	}

	matches, err := filepath.Glob(filepath.Join(outDir, "imagelock-*.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 4 {
		t.Fatalf("generated %d locks, want 4: %v", len(matches), matches)
	}
	wantFiles := map[string]bool{
		"imagelock-gpu-fractioning-v1.2.3-linux-amd64.yaml":      false,
		"imagelock-gpu-fractioning-v1.2.3-linux-arm64.yaml":      false,
		"imagelock-gpu-fractioning-v1.2.3-fips-linux-amd64.yaml": false,
		"imagelock-gpu-fractioning-v1.2.3-fips-linux-arm64.yaml": false,
	}
	for _, path := range matches {
		filename := filepath.Base(path)
		if _, expected := wantFiles[filename]; !expected {
			t.Errorf("unexpected generated filename %q", filename)
		} else {
			wantFiles[filename] = true
		}
		contents, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var lock imageLock
		if err := yaml.Unmarshal(contents, &lock); err != nil {
			t.Fatal(err)
		}
		wantTag := "v1.2.3"
		if lock.Spec.Profile == "fips" {
			wantTag += "-fips"
		}
		if len(lock.Spec.Images) != 1 || lock.Spec.Images[0].Source != host+"/gpu-fractioning/operator:"+wantTag {
			t.Errorf("%s contains unexpected images: %+v", path, lock.Spec.Images)
		}
	}
	for filename, found := range wantFiles {
		if !found {
			t.Errorf("expected generated file %q was not found", filename)
		}
	}
}

func TestWriteLocksIsDeterministic(t *testing.T) {
	catalog := validCatalog()
	amd64 := platform{OS: "linux", Architecture: "amd64"}
	arm64 := platform{OS: "linux", Architecture: "arm64"}
	targets := []lockTarget{
		{profile: "standard", platform: amd64},
		{profile: "standard", platform: arm64},
		{profile: "fips", platform: amd64},
		{profile: "fips", platform: arm64},
	}
	repository := catalog.Repository + "/operator"
	resolved := []resolvedImage{
		resolvedFixture("standard", repository, "v1.2.3", amd64, arm64),
		resolvedFixture("fips", repository, "v1.2.3-fips", amd64, arm64),
	}

	var firstOutput map[string][]byte
	for range 2 {
		outDir := t.TempDir()
		opts := options{version: "v1.2.3", outDir: outDir}
		if err := writeLocks(opts, catalog, targets, resolved); err != nil {
			t.Fatal(err)
		}
		matches, err := filepath.Glob(filepath.Join(outDir, "imagelock-*.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if len(matches) != 4 {
			t.Fatalf("generated %d locks, want 4: %v", len(matches), matches)
		}
		currentOutput := map[string][]byte{}
		for _, path := range matches {
			contents, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if info.Mode().Perm() != 0o644 {
				t.Errorf("mode for %s = %o, want 644", path, info.Mode().Perm())
			}
			var lock imageLock
			if err := yaml.Unmarshal(contents, &lock); err != nil {
				t.Fatal(err)
			}
			currentOutput[filepath.Base(path)] = contents
		}
		if len(firstOutput) != 0 {
			for filename, want := range firstOutput {
				if !bytes.Equal(currentOutput[filename], want) {
					t.Errorf("%s differs between identical generations", filename)
				}
			}
		}
		firstOutput = currentOutput
		if staging, err := filepath.Glob(filepath.Join(outDir, ".imagelock-*.tmp")); err != nil || len(staging) != 0 {
			t.Errorf("staged files after successful write = %v, err = %v", staging, err)
		}
	}
}

func TestWriteLocksCleansPartialPublication(t *testing.T) {
	catalog := validCatalog()
	amd64 := platform{OS: "linux", Architecture: "amd64"}
	arm64 := platform{OS: "linux", Architecture: "arm64"}
	targets := []lockTarget{{profile: "standard", platform: amd64}, {profile: "standard", platform: arm64}}
	resolved := []resolvedImage{resolvedFixture("standard", catalog.Repository+"/operator", "v1.2.3", amd64, arm64)}
	outDir := t.TempDir()
	opts := options{version: "v1.2.3", outDir: outDir}
	armLock, err := buildLock(catalog.Name, opts.version, targets[1], resolved)
	if err != nil {
		t.Fatal(err)
	}
	armPath := filepath.Join(outDir, lockFileName(catalog.ArtifactName, armLock))
	if err := os.Mkdir(armPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeLocks(opts, catalog, targets, resolved); err == nil {
		t.Fatal("want publication failure, got nil")
	}
	amdLock, err := buildLock(catalog.Name, opts.version, targets[0], resolved)
	if err != nil {
		t.Fatal(err)
	}
	amdPath := filepath.Join(outDir, lockFileName(catalog.ArtifactName, amdLock))
	if _, err := os.Stat(amdPath); !os.IsNotExist(err) {
		t.Fatalf("partial amd64 lock remains after failure: %v", err)
	}
	if staging, err := filepath.Glob(filepath.Join(outDir, ".imagelock-*.tmp")); err != nil || len(staging) != 0 {
		t.Errorf("staged files after failed write = %v, err = %v", staging, err)
	}
}

func resolvedFixture(profile, repository, tag string, platforms ...platform) resolvedImage {
	digests := make(map[platform]string, len(platforms))
	for index, imagePlatform := range platforms {
		digests[imagePlatform] = "sha256:" + strings.Repeat(string(rune('b'+index)), 64)
	}
	return resolvedImage{
		profile:           profile,
		chartImage:        chartImage{name: "operator", repo: repository, tag: tag},
		indexDigest:       "sha256:" + strings.Repeat("a", 64),
		digestPerPlatform: digests,
	}
}

func mustRandomImage(t *testing.T) v1.Image {
	t.Helper()
	image, err := random.Image(256, 1)
	if err != nil {
		t.Fatal(err)
	}
	return image
}

func writeIndex(t *testing.T, ref string, index v1.ImageIndex) {
	t.Helper()
	tag, err := name.NewTag(ref)
	if err != nil {
		t.Fatal(err)
	}
	if err := remote.WriteIndex(tag, index); err != nil {
		t.Fatal(err)
	}
}

func mustHost(t *testing.T, raw string) string {
	t.Helper()
	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return parsed.Host
}
