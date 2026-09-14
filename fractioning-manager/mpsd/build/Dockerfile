# syntax=docker/dockerfile:1.7
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build mpsd on the TARGET platform: it links go-nvml via cgo, and
# cross-compiling cgo needs a target cross-toolchain. Running this stage as
# $TARGETPLATFORM uses the native compiler for that platform (emulated by
# buildx/qemu when host != target).
FROM --platform=$TARGETPLATFORM golang:1.27.1-bookworm AS builder
ARG TARGETOS=linux
ARG TARGETARCH
# GOFIPS140 selects the Go Cryptographic Module version to link. "off" is the Go
# default and produces a regular build; the FIPS image variant is built with a
# pinned version (see the root Makefile's FIPS target). Compatible with
# CGO_ENABLED=1 — the module swap is independent of cgo linkage.
ARG GOFIPS140=off

WORKDIR /workspace

COPY go.mod go.sum ./
COPY pkg/ pkg/
RUN --mount=type=cache,target=/go/pkg/mod go mod download

COPY fractioning-manager/common/ fractioning-manager/common/
COPY fractioning-manager/mpsd/ fractioning-manager/mpsd/

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    GOFIPS140=${GOFIPS140} CGO_ENABLED=1 GOOS=$TARGETOS GOARCH=$TARGETARCH go build \
    -trimpath -ldflags="-s -w" -o /mpsd ./fractioning-manager/mpsd/cmd

# Runtime stage — nvidia/cuda base provides nvidia-cuda-mps-control which mpsd
# needs to exec. The NVIDIA container runtime only injects device access, not
# the MPS control binary itself.
#
# This is an OSRB-approved NVIDIA base container (the public NVIDIA CUDA images
# on Docker Hub are explicitly exempt from the "must be pulled from an NVIDIA
# managed repository" rule). No `apt install` layer is added on top of it, so
# this image adds no OS packages — only the mpsd binary.
FROM nvidia/cuda:13.3.1-base-ubuntu24.04

COPY --from=builder /mpsd /usr/local/bin/mpsd
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt

ENTRYPOINT ["/usr/local/bin/mpsd"]
