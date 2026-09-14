// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Command imagelock resolves the release matrix declared in catalog.yaml and
// writes one digest-pinned ImageLock YAML file per profile and platform.
package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"time"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"sigs.k8s.io/yaml"
)

const (
	lockAPIVersion  = "artifacts.run.ai/v1alpha1"
	lockKind        = "ImageLock"
	standardProfile = "standard"
	verifyVersion   = "v0.0.0-imagelock-verify"
)

const (
	registryAttempts = 3
	registryBackoff  = 2 * time.Second
)

// defaultStabilityReads is how many times each tag's digest is re-read and
// required to agree, guarding against a tag that moves mid-run.
const defaultStabilityReads = 10

var sha256Digest = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

var releaseVersion = regexp.MustCompile(`^v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$`)

type platform struct {
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
}

func (p platform) String() string { return p.OS + "/" + p.Architecture }

type chartImage struct {
	name string
	repo string
	tag  string
}

func (c chartImage) reference() string { return c.repo + ":" + c.tag }

type resolvedImage struct {
	profile string
	chartImage
	indexDigest       string
	digestPerPlatform map[platform]string
}

// digestResolver is an interface so tests can use an in-memory registry.
type digestResolver interface {
	resolve(ref string, platforms []platform) (indexDigest string, digestPerPlatform map[platform]string, err error)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("imagelock: ")
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	opts, err := parseFlags(args)
	if err != nil {
		return err
	}
	catalog, err := loadCatalog(opts.catalog)
	if err != nil {
		return err
	}
	if err := validateExpectedImages(catalog, opts.expectedImages); err != nil {
		return err
	}
	if opts.chartValues != "" {
		if err := validateChartValues(catalog, opts.chartValues); err != nil {
			return err
		}
	}
	plan, err := buildReleasePlan(catalog, opts.version)
	if err != nil {
		return err
	}

	if opts.verifyOnly {
		log.Printf("ok: %d image(s), %d profile(s), %d platform(s), %d exclusion(s), %d release combination(s)",
			len(catalog.Images), len(catalog.Profiles), len(catalog.Platforms), len(catalog.Exclude), plan.combinationCount)
		return nil
	}

	resolved, err := resolveImages(plan.images, remoteResolver{stabilityReads: opts.stabilityReads})
	if err != nil {
		return err
	}
	return writeLocks(opts, catalog, plan.targets, resolved)
}

type options struct {
	catalog        string
	chartValues    string
	version        string
	outDir         string
	verifyOnly     bool
	stabilityReads int
	expectedImages []string
}

func parseFlags(args []string) (options, error) {
	flags := flag.NewFlagSet("imagelock", flag.ContinueOnError)
	catalog := flags.String("catalog", "catalog.yaml", "path to the image release catalog")
	chartValues := flags.String("chart-values", "", "optional Helm values file whose image inventory must match the catalog")
	version := flags.String("version", "", "exact v-prefixed release version")
	outDir := flags.String("out-dir", "../../dist", "output directory")
	verifyOnly := flags.Bool("verify-only", false, "validate and expand the catalog; no network or files")
	stabilityReads := flags.Int("stability-reads", defaultStabilityReads, "re-read each tag's digest this many times and require they agree")
	var expectedImages []string
	flags.Func("expected-image", "production image name expected by the build, repeatable", func(value string) error {
		expectedImages = append(expectedImages, value)
		return nil
	})
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	if flags.NArg() != 0 {
		return options{}, fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}

	opts := options{
		catalog:        *catalog,
		chartValues:    *chartValues,
		version:        *version,
		outDir:         *outDir,
		verifyOnly:     *verifyOnly,
		stabilityReads: *stabilityReads,
		expectedImages: expectedImages,
	}
	if opts.stabilityReads < 1 {
		return options{}, errors.New("--stability-reads must be at least 1")
	}
	if opts.verifyOnly && opts.version == "" {
		opts.version = verifyVersion
	}
	if opts.version == "" {
		return options{}, errors.New("--version is required unless --verify-only is set")
	}
	if !releaseVersion.MatchString(opts.version) {
		return options{}, fmt.Errorf("--version %q is not a v-prefixed release version like vX.Y.Z", opts.version)
	}
	return opts, nil
}

// resolveImages touches no files, so a failure leaves no partial output.
func resolveImages(images []plannedImage, resolver digestResolver) ([]resolvedImage, error) {
	resolved := make([]resolvedImage, 0, len(images))
	for _, image := range images {
		indexDigest, digestPerPlatform, err := resolver.resolve(image.reference(), image.platforms)
		if err != nil {
			return nil, fmt.Errorf("resolve %s profile image %s: %w", image.profile, image.reference(), err)
		}
		if err := validateDigests(indexDigest, digestPerPlatform, image.platforms); err != nil {
			return nil, fmt.Errorf("resolve %s profile image %s: %w", image.profile, image.reference(), err)
		}
		resolved = append(resolved, resolvedImage{
			profile:           image.profile,
			chartImage:        image.chartImage,
			indexDigest:       indexDigest,
			digestPerPlatform: digestPerPlatform,
		})
	}
	return resolved, nil
}

func validateDigests(indexDigest string, digestPerPlatform map[platform]string, expected []platform) error {
	if !sha256Digest.MatchString(indexDigest) {
		return fmt.Errorf("bad index digest %q", indexDigest)
	}
	for _, imagePlatform := range expected {
		digest, found := digestPerPlatform[imagePlatform]
		if !found {
			return fmt.Errorf("missing %s digest", imagePlatform)
		}
		if !sha256Digest.MatchString(digest) {
			return fmt.Errorf("bad %s digest %q", imagePlatform, digest)
		}
	}
	return nil
}

type remoteResolver struct {
	stabilityReads int
}

func (r remoteResolver) resolve(ref string, platforms []platform) (string, map[platform]string, error) {
	reference, err := name.ParseReference(ref)
	if err != nil {
		return "", nil, err
	}
	descriptor, err := resolveStableDescriptor(reference, r.stabilityReads)
	if err != nil {
		return "", nil, err
	}
	if !descriptor.MediaType.IsIndex() {
		return "", nil, fmt.Errorf("source is %s, not a multi-platform image index", descriptor.MediaType)
	}
	digestPerPlatform, err := manifestDigestsPerPlatform(descriptor, platforms)
	if err != nil {
		return "", nil, err
	}
	return descriptor.Digest.String(), digestPerPlatform, nil
}

func resolveStableDescriptor(ref name.Reference, reads int) (*remote.Descriptor, error) {
	if reads < 1 {
		reads = defaultStabilityReads
	}
	first, err := fetchDescriptor(ref)
	if err != nil {
		return nil, err
	}
	for range reads - 1 {
		again, err := fetchDescriptor(ref)
		if err != nil {
			return nil, err
		}
		if again.Digest != first.Digest {
			return nil, fmt.Errorf("digest changed between reads: %s vs %s", first.Digest, again.Digest)
		}
	}
	return first, nil
}

func fetchDescriptor(ref name.Reference) (*remote.Descriptor, error) {
	auth := remote.WithAuthFromKeychain(authn.DefaultKeychain)
	var lastErr error
	for attempt := 1; attempt <= registryAttempts; attempt++ {
		descriptor, err := remote.Get(ref, auth)
		if err == nil {
			return descriptor, nil
		}
		lastErr = err
		if attempt < registryAttempts {
			time.Sleep(registryBackoff)
		}
	}
	return nil, fmt.Errorf("after %d attempts: %w", registryAttempts, lastErr)
}

func manifestDigestsPerPlatform(descriptor *remote.Descriptor, platforms []platform) (map[platform]string, error) {
	index, err := descriptor.ImageIndex()
	if err != nil {
		return nil, err
	}
	indexManifest, err := index.IndexManifest()
	if err != nil {
		return nil, err
	}

	digestPerPlatform := make(map[platform]string, len(platforms))
	for _, imagePlatform := range platforms {
		var matches []string
		for _, entry := range indexManifest.Manifests {
			if entry.Platform != nil && entry.Platform.OS == imagePlatform.OS && entry.Platform.Architecture == imagePlatform.Architecture {
				matches = append(matches, entry.Digest.String())
			}
		}
		switch len(matches) {
		case 0:
			return nil, fmt.Errorf("no %s manifest in index", imagePlatform)
		case 1:
			digestPerPlatform[imagePlatform] = matches[0]
		default:
			return nil, fmt.Errorf("ambiguous %s: %d manifests match", imagePlatform, len(matches))
		}
	}
	return digestPerPlatform, nil
}

type imageLock struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Metadata   struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	} `json:"metadata"`
	Spec struct {
		Profile  string        `json:"profile"`
		Platform platform      `json:"platform"`
		Images   []lockedImage `json:"images"`
	} `json:"spec"`
}

type lockedImage struct {
	Name        string `json:"name"`
	Image       string `json:"image"`
	Source      string `json:"source"`
	IndexDigest string `json:"indexDigest,omitempty"`
}

type lockOutput struct {
	path string
	yaml []byte
}

func writeLocks(opts options, catalog imageCatalog, targets []lockTarget, resolved []resolvedImage) error {
	outputs, err := marshalLocks(opts, catalog, targets, resolved)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(opts.outDir, 0o755); err != nil {
		return fmt.Errorf("create %s: %w", opts.outDir, err)
	}

	stagedPaths := make([]string, len(outputs))
	cleanupStaged := func() {
		for _, path := range stagedPaths {
			if path != "" {
				_ = os.Remove(path)
			}
		}
	}
	for index, output := range outputs {
		staged, err := os.CreateTemp(opts.outDir, ".imagelock-*.tmp")
		if err != nil {
			cleanupStaged()
			return fmt.Errorf("create staged lock: %w", err)
		}
		stagedPaths[index] = staged.Name()
		if err := staged.Chmod(0o644); err != nil {
			_ = staged.Close()
			cleanupStaged()
			return fmt.Errorf("set mode on %s: %w", staged.Name(), err)
		}
		if _, err := staged.Write(output.yaml); err != nil {
			_ = staged.Close()
			cleanupStaged()
			return fmt.Errorf("write %s: %w", staged.Name(), err)
		}
		if err := staged.Close(); err != nil {
			cleanupStaged()
			return fmt.Errorf("close %s: %w", staged.Name(), err)
		}
	}

	var publishedPaths []string
	for index, output := range outputs {
		if err := os.Rename(stagedPaths[index], output.path); err != nil {
			for _, path := range publishedPaths {
				_ = os.Remove(path)
			}
			cleanupStaged()
			return fmt.Errorf("publish %s: %w", output.path, err)
		}
		stagedPaths[index] = ""
		publishedPaths = append(publishedPaths, output.path)
	}
	log.Printf("wrote %d lock(s) to %s", len(outputs), opts.outDir)
	return nil
}

func marshalLocks(opts options, catalog imageCatalog, targets []lockTarget, resolved []resolvedImage) ([]lockOutput, error) {
	outputs := make([]lockOutput, 0, len(targets))
	for _, target := range targets {
		lock, err := buildLock(catalog.Name, opts.version, target, resolved)
		if err != nil {
			return nil, err
		}
		yamlBytes, err := yaml.Marshal(lock)
		if err != nil {
			return nil, fmt.Errorf("marshal lock for %s/%s: %w", target.profile, target.platform, err)
		}
		outputs = append(outputs, lockOutput{
			path: filepath.Join(opts.outDir, lockFileName(catalog.ArtifactName, lock)),
			yaml: yamlBytes,
		})
	}
	return outputs, nil
}

func buildLock(catalogName, version string, target lockTarget, resolved []resolvedImage) (imageLock, error) {
	lock := imageLock{APIVersion: lockAPIVersion, Kind: lockKind}
	lock.Metadata.Name = catalogName
	lock.Metadata.Version = version
	lock.Spec.Profile = target.profile
	lock.Spec.Platform = target.platform

	for _, image := range resolved {
		if image.profile != target.profile {
			continue
		}
		digest, included := image.digestPerPlatform[target.platform]
		if !included {
			continue
		}
		lock.Spec.Images = append(lock.Spec.Images, lockedImage{
			Name:        image.name,
			Image:       image.repo + "@" + digest,
			Source:      image.reference(),
			IndexDigest: image.indexDigest,
		})
	}
	if len(lock.Spec.Images) == 0 {
		return imageLock{}, fmt.Errorf("cannot build empty lock for %s/%s", target.profile, target.platform)
	}
	sort.Slice(lock.Spec.Images, func(i, j int) bool {
		return lock.Spec.Images[i].Name < lock.Spec.Images[j].Name
	})
	return lock, nil
}

func lockFileName(artifactName string, lock imageLock) string {
	if lock.Spec.Profile == standardProfile {
		return fmt.Sprintf("imagelock-%s-%s-%s-%s.yaml", artifactName,
			lock.Metadata.Version, lock.Spec.Platform.OS, lock.Spec.Platform.Architecture)
	}
	return fmt.Sprintf("imagelock-%s-%s-%s-%s-%s.yaml", artifactName,
		lock.Metadata.Version, lock.Spec.Profile, lock.Spec.Platform.OS, lock.Spec.Platform.Architecture)
}
