# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build on the native BUILDPLATFORM and cross-compile to the target arch. The
# operator is CGO_ENABLED=0, so cross-compilation is pure Go (no emulation) —
# much faster than running the whole build under qemu for a foreign --platform.
FROM --platform=$BUILDPLATFORM golang:1.27.1 AS builder
ARG TARGETOS
ARG TARGETARCH
# GOFIPS140 selects the Go Cryptographic Module version to link. "off" is the Go
# default and produces a regular build; the FIPS image variant is built with a
# pinned version (see the root Makefile's FIPS target).
ARG GOFIPS140=off

WORKDIR /workspace

# Copy the API module (lightweight, changes rarely).
COPY api/go.mod api/go.sum api/
COPY api/v1alpha1/ api/v1alpha1/

# Copy pkg module (local replace in operator/go.mod).
COPY pkg/ pkg/

# Copy operator module manifests and download deps first for layer caching.
COPY operator/go.mod operator/go.sum operator/
RUN cd operator && go mod download

# Copy operator source.
COPY operator/cmd/       operator/cmd/
COPY operator/internal/  operator/internal/

RUN cd operator && GOFIPS140=${GOFIPS140} CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH:-amd64} go build -o manager cmd/main.go

# NVIDIA-approved distroless base pulled from NGC (OSRB requires an approved
# NVIDIA base container; gcr.io/distroless is not one). Adds no OS packages of
# our own — the image is the base plus the static manager binary.
FROM nvcr.io/nvidia/distroless/go:v4.1.3
WORKDIR /
COPY --from=builder /workspace/operator/manager .
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt
# 1000 is the base image's `nvs` user; the chart requires runAsNonRoot.
USER 1000:1000
ENTRYPOINT ["/manager"]
