# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# kai-gpu-fractioning
# -----------------------------------------------------------
VERSION  ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
DOCKER_REPO_BASE ?= ghcr.io/kai-scheduler/kai-gpu-fractioning
# Target platform for image builds. GPU clusters are amd64; override for others.
# buildkit emulates (qemu) when the host arch differs.
PLATFORM ?= linux/amd64

# Escape hatch for extra flags on every image build, both `docker build` and
# `docker buildx build`. FIPS=1 appends its --build-arg here.
DOCKER_BUILD_ARGS ?=

# -----------------------------------------------------------
# FIPS
#
# FIPS=1 links every Go binary against the CMVP-validated Go Cryptographic
# Module (https://go.dev/doc/security/fips140) and suffixes image tags with
# "-fips", so FIPS variants sit alongside the regular images in the registry.
#
# GOFIPS140 is exported so the host `go build` targets and any sub-make inherit
# it. The docker targets pass it as a build arg instead, because compilation
# happens inside each Dockerfile's builder stage.
#
# v1.0.0 is pinned deliberately: it holds CMVP Certificate #5247, while newer
# module versions are still under review. Do not switch this to the `certified`
# alias — that would silently change what gets linked when a new validation
# lands, which is exactly what a compliance build must not do.
# -----------------------------------------------------------
FIPS ?= 0
GOFIPS140_VERSION ?= v1.0.0

ifeq ($(FIPS),1)
# Both overrides are required: make ignores a plain assignment or += to a
# variable that came from the command line, and VERSION and DOCKER_BUILD_ARGS
# are both things a caller may set that way.
override VERSION := $(VERSION)-fips
override DOCKER_BUILD_ARGS += --build-arg GOFIPS140=$(GOFIPS140_VERSION)
export GOFIPS140 := $(GOFIPS140_VERSION)
endif

LOCALBIN ?= $(CURDIR)/bin
ADDLICENSE ?= $(LOCALBIN)/addlicense
ADDLICENSE_VERSION ?= v1.2.0

# The exact copyright holder string OSRB checks for. Do not reword.
LICENSE_HOLDER := NVIDIA CORPORATION & AFFILIATES. All rights reserved.
LICENSE_YEAR ?= 2026

# addlicense does not honor .gitignore, so ignored source-like paths are listed
# here to keep validation reproducible in developer worktrees.
LICENSE_IGNORES := \
	-ignore 'bin/**' \
	-ignore 'dist/**' \
	-ignore '.context/**' \
	-ignore '.gocache/**' \
	-ignore '.gotmp/**' \
	-ignore '.local/**' \
	-ignore '.idea/**' \
	-ignore '.vscode/**' \
	-ignore '**/testdata/**'

# -----------------------------------------------------------
# Build
# -----------------------------------------------------------

.PHONY: build build-operator build-mpsd build-fractiond

build: build-operator build-mpsd build-fractiond

build-operator:
	$(MAKE) -C operator build

build-mpsd:
	CGO_ENABLED=1 go build -o bin/mpsd ./fractioning-manager/mpsd/cmd

build-fractiond:
	go build -o bin/fractiond ./fractioning-manager/fractiond/cmd

# -----------------------------------------------------------
# Test
# -----------------------------------------------------------

.PHONY: test test-operator test-fractioning-manager test-metricsd

test: test-operator test-fractioning-manager test-metricsd

test-operator:
	$(MAKE) -C operator test

test-fractioning-manager:
	go test ./fractioning-manager/... -race -count=1

# metricsd is a separate Go module (own go.mod), so `go test ./fractioning-manager/...`
# above does not descend into it. Delegate to its own Makefile, which handles the
# cgo/NVML build the metrics collector needs.
test-metricsd:
	$(MAKE) -C fractioning-manager/metricsd test

# -----------------------------------------------------------
# Helm chart
# -----------------------------------------------------------

.PHONY: chart-test

# Renders the chart against the cases in operator/charts/tests/. Requires the
# helm-unittest plugin:
#   helm plugin install https://github.com/helm-unittest/helm-unittest
#
# Deliberately not part of `test`: that runs on the Go toolchain every developer
# already has, and this needs a Helm plugin they may not. CI runs it as part of
# chart validation.
chart-test:
	helm unittest operator/charts

# -----------------------------------------------------------
# Air-gap / ImageLock
# -----------------------------------------------------------

IMAGE_LOCK_OUT_DIR ?= $(CURDIR)/dist
IMAGE_LOCK_CATALOG ?= $(CURDIR)/hack/imagelock/catalog.yaml
IMAGE_LOCK_CHART_VALUES ?= $(CURDIR)/operator/charts/values.yaml
IMAGE_LOCK_IMAGE_ARGS = $(foreach image,$(IMAGES),--expected-image $(image))

.PHONY: image-lock image-lock-verify image-lock-test

image-lock:
	@test -n "$(VERSION)" || { echo "VERSION is required (for example: make image-lock VERSION=v1.2.3)" >&2; exit 1; }
	GOWORK=off go -C hack/imagelock run . \
		--version "$(VERSION)" \
		--catalog "$(IMAGE_LOCK_CATALOG)" \
		--chart-values "$(IMAGE_LOCK_CHART_VALUES)" \
		--out-dir "$(IMAGE_LOCK_OUT_DIR)" \
		$(IMAGE_LOCK_IMAGE_ARGS)

image-lock-verify:
	GOWORK=off go -C hack/imagelock run . \
		--verify-only \
		--catalog "$(IMAGE_LOCK_CATALOG)" \
		--chart-values "$(IMAGE_LOCK_CHART_VALUES)" \
		$(IMAGE_LOCK_IMAGE_ARGS)

image-lock-test:
	GOWORK=off go -C hack/imagelock test ./...

# -----------------------------------------------------------
# E2E — see test/e2e/e2e.mk (targets: e2e, e2e-cluster-up/down,
# e2e-deploy/undeploy, test-e2e, test-e2e-metrics).
# -----------------------------------------------------------

include test/e2e/e2e.mk

# -----------------------------------------------------------
# Code quality
# -----------------------------------------------------------

.PHONY: fmt vet lint validate

fmt:
	$(MAKE) -C operator fmt
	go fmt ./fractioning-manager/...

vet:
	$(MAKE) -C operator vet
	go vet ./fractioning-manager/...

lint:
	$(MAKE) -C operator lint
	golangci-lint run ./fractioning-manager/...

validate: license-check
	$(MAKE) -C operator validate
	go fmt ./fractioning-manager/...
	go vet ./fractioning-manager/...
	golangci-lint run ./fractioning-manager/...

# -----------------------------------------------------------
# License headers
#
# Every authored file carries the two-line Apache-2.0 SPDX header. This is
# enforced repo-wide (all modules, the Helm chart, the Dockerfiles and CI
# config), not left to convention — `license-check` is part of `validate` and
# runs as its own required CI job.
# -----------------------------------------------------------

.PHONY: addlicense gen-license license-check

$(LOCALBIN):
	mkdir -p $(LOCALBIN)

addlicense: $(ADDLICENSE) ## Install addlicense locally.
$(ADDLICENSE): | $(LOCALBIN)
	test -s $(ADDLICENSE) || GOBIN=$(LOCALBIN) go install github.com/google/addlicense@$(ADDLICENSE_VERSION)

gen-license: addlicense ## Add the Apache-2.0 SPDX header to files that are missing it.
	$(ADDLICENSE) -c "$(LICENSE_HOLDER)" -y $(LICENSE_YEAR) -s=only -l apache -v \
		$(LICENSE_IGNORES) .

license-check: addlicense ## Verify every file carries the Apache-2.0 SPDX header.
	$(ADDLICENSE) -check -c "$(LICENSE_HOLDER)" -s=only -l apache \
		$(LICENSE_IGNORES) .

# -----------------------------------------------------------
# Third-party attribution
#
# THIRD-PARTY.txt is generated from the modules `go list -deps` reports for the
# four shipped binaries, and is copied into every image at /THIRD-PARTY.txt.
# -----------------------------------------------------------

.PHONY: third-party third-party-check

third-party: ## Regenerate THIRD-PARTY.txt from the linked dependency set.
	python3 hack/gen-third-party.py

third-party-check: ## Fail if THIRD-PARTY.txt does not match the linked dependency set.
	python3 hack/gen-third-party.py --check

# -----------------------------------------------------------
# Generate (CRDs, deepcopy, manifests) — delegated to operator
# -----------------------------------------------------------

.PHONY: generate manifests

generate:
	$(MAKE) -C operator generate

manifests:
	$(MAKE) -C operator manifests

# -----------------------------------------------------------
# Deploy
# -----------------------------------------------------------

.PHONY: deploy

deploy:
	$(MAKE) -C operator deploy

# -----------------------------------------------------------
# Docker
# -----------------------------------------------------------

# The shipped component images and the Dockerfile that builds each one. This is
# the single source of truth: the aggregate targets expand from IMAGES, and both
# FIPS verification scripts read these lists rather than repeating them. A
# component added here without FIPS plumbing therefore fails a check instead of
# being published as a "-fips" image built with ordinary crypto.
IMAGES := operator mpsd fractiond metricsd

DOCKERFILE_operator  := operator/Dockerfile
DOCKERFILE_mpsd      := fractioning-manager/mpsd/build/Dockerfile
DOCKERFILE_fractiond := fractioning-manager/fractiond/build/Dockerfile
DOCKERFILE_metricsd  := fractioning-manager/metricsd/Dockerfile

.PHONY: docker-build $(addprefix docker-build-,$(IMAGES))
.PHONY: docker-push $(addprefix docker-push-,$(IMAGES))

docker-build: $(addprefix docker-build-,$(IMAGES))

docker-build-operator:
	$(MAKE) -C operator docker-build IMG=$(DOCKER_REPO_BASE)/operator:$(VERSION) PLATFORM=$(PLATFORM) DOCKER_BUILD_ARGS="$(DOCKER_BUILD_ARGS)"

docker-build-mpsd:
	docker build --platform $(PLATFORM) $(DOCKER_BUILD_ARGS) -f $(DOCKERFILE_mpsd) -t $(DOCKER_REPO_BASE)/mpsd:$(VERSION) .

docker-build-fractiond:
	docker build --platform $(PLATFORM) $(DOCKER_BUILD_ARGS) -f $(DOCKERFILE_fractiond) -t $(DOCKER_REPO_BASE)/fractiond:$(VERSION) .

# metricsd links NVML (cgo) and is built from the repo root so its replace of the
# shared fractiond module resolves. The build stage runs as the target platform so
# cgo uses a native toolchain. GO_TAGS=e2e builds the fake-GPU test image.
docker-build-metricsd:
	docker build --platform $(PLATFORM) $(DOCKER_BUILD_ARGS) -f $(DOCKERFILE_metricsd) -t $(DOCKER_REPO_BASE)/metricsd:$(VERSION) .

docker-push: $(addprefix docker-push-,$(IMAGES))

docker-push-operator:
	$(MAKE) -C operator docker-push IMG=$(DOCKER_REPO_BASE)/operator:$(VERSION)

docker-push-mpsd:
	docker push $(DOCKER_REPO_BASE)/mpsd:$(VERSION)

docker-push-metricsd:
	docker push $(DOCKER_REPO_BASE)/metricsd:$(VERSION)

docker-push-fractiond:
	docker push $(DOCKER_REPO_BASE)/fractiond:$(VERSION)

# -----------------------------------------------------------
# Multi-arch image build/push (release)
# -----------------------------------------------------------
# Release builds go through `docker buildx` so a single command produces a
# multi-platform manifest. Multi-platform manifests can't be loaded into the
# local daemon, so they push directly (DOCKER_BUILDX_OUTPUT defaults to --push).
# Requires a docker-container buildx builder (CI: docker/setup-buildx-action) plus
# QEMU/binfmt (docker/setup-qemu-action) for building any non-native platform.
#
#   make docker-buildx VERSION=v0.1.0 \
#     DOCKER_BUILD_PLATFORM=linux/amd64,linux/arm64             # multi-arch, push
#   make docker-buildx-operator DOCKER_BUILDX_OUTPUT=--load \
#     DOCKER_BUILD_PLATFORM=linux/amd64                         # single-arch, local
#
# NOTE: operator and fractiond are CGO_ENABLED=0 and cross-compile natively on
# the build host. mpsd and metricsd link NVML via cgo, so their non-native arch
# stages run under qemu emulation (slower).

# Comma-separated platform list. GPU nodes may be amd64 or arm64 (e.g. GB200 is
# arm64), so releases build both — the release workflow passes the full list. The
# default here is a single arch for convenient local `--load` builds.
DOCKER_BUILD_PLATFORM ?= linux/amd64
# Passed to `docker buildx build`: --push to publish, --load for a local single-arch image.
DOCKER_BUILDX_OUTPUT  ?= --push

# Release metadata for images that bake it into the binary (currently metricsd only).
GIT_COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: docker-buildx $(addprefix docker-buildx-,$(IMAGES))

# $(1)=Dockerfile path, $(2)=image name, $(3)=optional extra build args.
# Build context is always the repo root.
define buildx-image
docker buildx build --platform $(DOCKER_BUILD_PLATFORM) $(DOCKER_BUILDX_OUTPUT) $(DOCKER_BUILD_ARGS) $(3) \
	-f $(1) -t $(DOCKER_REPO_BASE)/$(2):$(VERSION) .
endef

docker-buildx: $(addprefix docker-buildx-,$(IMAGES))

docker-buildx-operator:
	$(call buildx-image,$(DOCKERFILE_operator),operator)

docker-buildx-mpsd:
	$(call buildx-image,$(DOCKERFILE_mpsd),mpsd)

docker-buildx-fractiond:
	$(call buildx-image,$(DOCKERFILE_fractiond),fractiond)

# metricsd bakes version metadata into the binary via ldflags (ARG VERSION/COMMIT/DATE
# in its Dockerfile), so pass them through — otherwise a release image reports version=dev.
docker-buildx-metricsd:
	$(call buildx-image,$(DOCKERFILE_metricsd),metricsd,--build-arg VERSION=$(VERSION) --build-arg COMMIT=$(GIT_COMMIT) --build-arg DATE=$(BUILD_DATE))

# Promotes an already-verified image from one tag to another. This is a
# registry-side manifest copy, so it neither rebuilds nor re-pushes layers and
# the multi-arch index is preserved. The release builds under a staging tag,
# verifies that, and only then creates the tag consumers pull.
#   make docker-promote FROM_TAG=v0.1.0-unverified-fips TO_TAG=v0.1.0-fips
.PHONY: docker-promote
docker-promote:
	@test -n "$(FROM_TAG)" || { echo "FROM_TAG is required" >&2; exit 1; }
	@test -n "$(TO_TAG)"   || { echo "TO_TAG is required" >&2; exit 1; }
	$(foreach i,$(IMAGES),docker buildx imagetools create \
		--tag $(DOCKER_REPO_BASE)/$(i):$(TO_TAG) \
		$(DOCKER_REPO_BASE)/$(i):$(FROM_TAG) &&) true

# Single source of truth for CI's FIPS verification step, so the pinned module
# version can't drift between the build and the assertion that checks it.
.PHONY: print-gofips140-version
print-gofips140-version:
	@echo $(GOFIPS140_VERSION)

# Consumed by hack/verify-fips-images.sh and hack/verify-fips-build-args.sh so
# that neither keeps its own copy of the component list. Adding a component to
# IMAGES without wiring FIPS into its Dockerfile then fails a check.
.PHONY: print-image-names
print-image-names:
	@echo $(IMAGES)

.PHONY: print-image-dockerfiles
print-image-dockerfiles:
	@$(foreach i,$(IMAGES),echo "$(i) $(DOCKERFILE_$(i))";)

# -----------------------------------------------------------
# Clean
# -----------------------------------------------------------

.PHONY: clean

clean:
	rm -rf bin/ coverage.out
	$(MAKE) -C operator clean
