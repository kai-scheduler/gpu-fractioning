# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build on the native BUILDPLATFORM and cross-compile: fractiond is CGO_ENABLED=0,
# so this is pure-Go cross-compilation (no qemu emulation) to the target arch.
FROM --platform=$BUILDPLATFORM golang:1.27 AS builder
ARG TARGETARCH
# GOFIPS140 selects the Go Cryptographic Module version to link. "off" is the Go
# default and produces a regular build; the FIPS image variant is built with a
# pinned version (see the root Makefile's FIPS target).
ARG GOFIPS140=off
WORKDIR /workspace
COPY go.mod go.sum ./
COPY pkg/ pkg/
RUN go mod download
COPY fractioning-manager/ fractioning-manager/
RUN GOFIPS140=${GOFIPS140} CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -ldflags="-s -w" -o fractiond ./fractioning-manager/fractiond/cmd

# NVIDIA-approved distroless base pulled from NGC (OSRB requires an approved
# NVIDIA base container; gcr.io/distroless is not one). fractiond is a
# statically-linked Go binary with no GPU or OS-level dependencies — it only
# communicates via the NRI Unix socket — so nothing from the base is needed
# beyond a filesystem to run in, and no OS packages are added.
# The NVML shadow library fractiond stages onto the host and bind-mounts into
# GPU containers. It forwards through hand-written x86-64 trampolines, so it
# builds on that architecture only; its Makefile emits nothing elsewhere and the
# stage below then ships an empty directory. fractiond treats a missing library
# as "feature off" and keeps running, so an arm64 image loses the capped
# nvidia-smi view but nothing else.
#
# Built natively rather than under --platform=$TARGETPLATFORM to avoid dragging
# qemu into the build for the one architecture that produces no output anyway.
# The TARGETARCH/uname pair is what prevents the opposite failure: silently
# copying an amd64 object into an arm64 image when cross-compiling.
FROM --platform=$BUILDPLATFORM golang:1.27 AS shimbuilder
ARG TARGETARCH
WORKDIR /shim
COPY fractioning-manager/nvmlshim/ ./
RUN mkdir -p /shimout && \
    if [ "$TARGETARCH" = "amd64" ] && [ "$(uname -m)" = "x86_64" ]; then \
        make && cp build/libnvidia-ml.so.1 /shimout/; \
    else \
        echo "nvmlshim: skipping build (TARGETARCH=$TARGETARCH on $(uname -m))"; \
    fi

FROM nvcr.io/nvidia/distroless/go:v4.1.3
COPY --from=builder /workspace/fractiond /fractiond
# Trailing slash on the source: copies the directory's contents, and tolerates
# it being empty on architectures where the shim is not built.
COPY --from=shimbuilder /shimout/ /usr/lib/gpu-fractioning/
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt
# Root, because the NRI socket fractiond connects to is root-owned. The base
# image defaults to its non-root `nvs` user, so this has to be explicit.
USER 0:0
ENTRYPOINT ["/fractiond"]
