# syntax=docker/dockerfile:1.7
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Build context MUST be the repository root (not fractioning-manager/metricsd): the
# metricsd module replaces github.com/kai-scheduler/gpu-fractioning => ../../ to use
# the shared fractiond/mapping packages, so the root module source must be present
# in the build. Build with:  docker build -f fractioning-manager/metricsd/Dockerfile .

# Build on the TARGET platform, not the build platform: metricsd links NVML via
# cgo, and cross-compiling cgo needs a target cross-toolchain (the native gcc
# rejects the cross flags, e.g. "-m64" when building amd64 on an arm64 host).
# Running the build stage as $TARGETPLATFORM uses a native toolchain (emulated by
# buildx/qemu when host≠target), so cgo compiles cleanly. Select the target with
# `docker buildx build --platform linux/amd64`.
FROM --platform=$TARGETPLATFORM golang:1.27-bookworm AS build

# TARGETOS/TARGETARCH are populated by BuildKit from --platform. Do NOT give them
# defaults: a default value shadows BuildKit's built-in, so a linux/arm64 build
# would run on an arm64 toolchain while still setting GOARCH=amd64, making cgo
# cross-link with the native gcc ("unrecognized command-line option '-m64'").
ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev
ARG COMMIT=unknown
ARG DATE=unknown
# GO_TAGS selects optional build tags. Empty (the default) = production. Set to
# "e2e" to compile the fake-GPU detection used for testing on GPU-less clusters.
# Never ship an e2e-tagged image to prod.
ARG GO_TAGS=""
# GOFIPS140 selects the Go Cryptographic Module version to link. "off" is the Go
# default and produces a regular build; the FIPS image variant is built with a
# pinned version (see the root Makefile's FIPS target). Compatible with
# CGO_ENABLED=1 — the module swap is independent of cgo linkage.
ARG GOFIPS140=off

WORKDIR /workspace

# Root module go.mod/go.sum, then the whole fractioning-manager tree (both the
# metricsd module and the shared fractiond packages it references via replace).
COPY go.mod go.sum ./
COPY pkg/ pkg/
COPY fractioning-manager/ fractioning-manager/

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd fractioning-manager/metricsd && \
    GOFIPS140=${GOFIPS140} CGO_ENABLED=1 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH:-amd64} go build \
    -trimpath -tags="${GO_TAGS}" \
    -ldflags="-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.date=${DATE}" \
    -o /out/metricsd ./cmd

# NVIDIA-approved distroless base pulled from NGC (OSRB requires an approved
# NVIDIA base container; gcr.io/distroless is not one). This variant ships
# glibc — including libdl/libpthread — which metricsd needs because it is built
# CGO_ENABLED=1 and the NVML bindings dlopen libnvidia-ml.so at runtime; the
# driver itself is injected by the NVIDIA container runtime. No OS packages are
# added on top of the base.
FROM nvcr.io/nvidia/distroless/go:v4.1.4

# Root, because reading per-process GPU state through NVML requires it. The base
# image defaults to its non-root `nvs` user, so this has to be explicit.
USER 0:0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY --from=build /out/metricsd /usr/local/bin/metricsd
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt

ENTRYPOINT ["/usr/local/bin/metricsd"]
