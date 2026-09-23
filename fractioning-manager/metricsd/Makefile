# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DOCKER_REPO_BASE ?= ghcr.io/kai-scheduler/gpu-fractioning
IMAGE ?= $(DOCKER_REPO_BASE)/metricsd
TAG ?= dev
PLATFORMS ?= linux/amd64,linux/arm64
BIN_DIR ?= bin
CGO_ENABLED ?= 1
DOCKER ?= docker
# GO_TAGS selects optional build tags for `go build` and docker image builds.
GO_TAGS ?=

# GOFIPS140 selects the Go Cryptographic Module version to link. "off" (the
# default) builds without FIPS; the root Makefile sets this to a pinned version
# when invoked with FIPS=1. Exported so the plain `build` target honors it too.
GOFIPS140 ?= off
export GOFIPS140

VERSION ?= $(TAG)
COMMIT ?= $(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)
DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

NVML_MOCK_IMAGE  ?= ghcr.io/nvidia/nvml-mock:latest
# Path where the nvml-mock DaemonSet installs libnvidia-ml.so on the host.
# Override if extracting the .so from the image into a local directory.
NVML_MOCK_SO_DIR ?= /var/lib/nvml-mock/driver/usr/lib64

.PHONY: fmt test smoke-nvml build clean docker-build docker-buildx

fmt:
	gofmt -w ./cmd ./internal

test:
	go test -v ./...

# Pulls the nvml-mock image, extracts libnvidia-ml.so into a temp directory,
# runs the production NVML collector smoke tests against it, then cleans up.
# Requires Docker and a Linux host (the .so is a Linux ELF shared library).
smoke-nvml:
	@SO_DIR=$$(mktemp -d); \
	echo ">>> Pulling $(NVML_MOCK_IMAGE)"; \
	docker pull -q $(NVML_MOCK_IMAGE); \
	echo ">>> Extracting libnvidia-ml.so"; \
	docker run --rm --entrypoint sh \
		-v $$SO_DIR:/out \
		$(NVML_MOCK_IMAGE) \
		-c 'find / -maxdepth 10 -name "libnvidia-ml.so*" 2>/dev/null | xargs -I{} cp -f {} /out/ 2>/dev/null; true'; \
	if [ -z "$$(ls $$SO_DIR 2>/dev/null)" ]; then \
		echo "ERROR: libnvidia-ml.so not found in $(NVML_MOCK_IMAGE)"; \
		rm -rf $$SO_DIR; \
		exit 1; \
	fi; \
	echo "Extracted: $$(ls $$SO_DIR)"; \
	ls $$SO_DIR/libnvidia-ml.so > /dev/null 2>&1 || \
		ln -sf $$(ls $$SO_DIR/libnvidia-ml.so.* | head -1) $$SO_DIR/libnvidia-ml.so; \
	echo ">>> Running smoke tests (LD_LIBRARY_PATH=$$SO_DIR)"; \
	LD_LIBRARY_PATH=$$SO_DIR MOCK_NVML_CONFIG=$$PWD/internal/metrics/testdata/nvml_smoke_config.yaml CGO_ENABLED=1 \
		go test -tags nvml_smoke -v -count=1 ./internal/metrics/; \
	STATUS=$$?; \
	rm -rf $$SO_DIR; \
	exit $$STATUS

build:
	mkdir -p $(BIN_DIR)
	CGO_ENABLED=$(CGO_ENABLED) go build -trimpath -tags="$(GO_TAGS)" \
		-ldflags="-s -w -X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(DATE)" \
		-o $(BIN_DIR)/metricsd ./cmd

clean:
	rm -rf bin coverage.out

docker-build:
	$(DOCKER) build \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg DATE=$(DATE) \
		--build-arg GO_TAGS=$(GO_TAGS) \
		--build-arg GOFIPS140=$(GOFIPS140) \
		-t $(IMAGE):$(TAG) .

docker-buildx:
	$(DOCKER) buildx build \
		--platform $(PLATFORMS) \
		--build-arg VERSION=$(VERSION) \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg DATE=$(DATE) \
		--build-arg GO_TAGS=$(GO_TAGS) \
		--build-arg GOFIPS140=$(GOFIPS140) \
		-t $(IMAGE):$(TAG) .
