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

# Runtime stage — UBI9 base, pulled from NGC. Adds no OS packages; only the
# mpsd binary.
#
# mpsd needs two things from it: glibc, because it is built CGO_ENABLED=1 and
# the NVML bindings dlopen libnvidia-ml.so at runtime, and /usr/bin/test, which
# the readiness and liveness probes exec against the MPS control socket.
#
# It does NOT need the base for nvidia-cuda-mps-control. No nvidia/cuda image
# ships that binary; the NVIDIA container runtime injects it from the host
# driver because this container requests NVIDIA_DRIVER_CAPABILITIES=compute.
# The base is CUDA only because NGC publishes no plain UBI9 image.
FROM nvcr.io/nvidia/cuda:13.3.1-base-ubi9

COPY --from=builder /mpsd /usr/local/bin/mpsd
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt

ENTRYPOINT ["/usr/local/bin/mpsd"]
