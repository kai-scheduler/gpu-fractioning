# syntax=docker/dockerfile:1.7
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fake mpsd image for fake-GPU E2E clusters.
#
# It contains the REAL mpsd supervisor (same code as the production image) plus a
# FAKE nvidia-cuda-mps-control at the default path (/usr/bin/nvidia-cuda-mps-control)
# that just creates the control socket and idles. This exercises the real
# supervisor + readiness + controller status/node-condition flow without a GPU,
# and needs NO controller/CRD change (the supervisor uses the default binary path).
#
# Build from the REPO ROOT (context must include go.mod, pkg/, fractioning-manager/, hack/):
#   docker build -f hack/fake-mps/Dockerfile -t <name>:<tag> .
FROM --platform=$TARGETPLATFORM golang:1.26.4-bookworm AS mpsd-builder
ARG TARGETOS=linux
ARG TARGETARCH
WORKDIR /workspace

# Real mpsd supervisor (same code as fractioning-manager/mpsd/build/Dockerfile).
# pkg/ is copied before `go mod download` because the root go.mod has
# `replace github.com/kai-scheduler/kai-gpu-fractioning/pkg => ./pkg`.
COPY go.mod go.sum ./
COPY pkg/ pkg/
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY fractioning-manager/common/ fractioning-manager/common/
COPY fractioning-manager/mpsd/ fractioning-manager/mpsd/
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=1 GOOS=$TARGETOS GOARCH=$TARGETARCH go build \
    -trimpath -ldflags="-s -w" -o /mpsd ./fractioning-manager/mpsd/cmd

# Fake nvidia-cuda-mps-control — standalone std-lib-only module (hack/fake-mps).
FROM --platform=$BUILDPLATFORM golang:1.26.4-alpine AS fake-builder
ARG TARGETOS=linux
ARG TARGETARCH
COPY hack/fake-mps/ /fake-mps-src/
RUN cd /fake-mps-src && \
    CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH GOTOOLCHAIN=local go build \
    -trimpath -ldflags="-s -w" -o /fake-mps .

# Extract libnvidia-ml.so from the nvml-mock image so mpsd can read a fake
# driver version before it starts the fake MPS control binary.
#
# Pinned by digest, not :latest — upstream retags :latest in place, so a build
# that passed yesterday can silently pick up a different library today. This is
# the same multi-arch index digest the nvml-mock DaemonSet uses (see
# fractioning-manager/metricsd/deploy/fake-gpu-cluster/nvml-mock.yaml); bump the
# two together.
FROM ghcr.io/nvidia/nvml-mock@sha256:2718efebb1695121f0d3f1a4325a22050a2e95362d98f4f3abe9f70dd7316e9c AS nvml-mock
RUN cp /usr/local/lib/libnvidia-ml.so.1 /libnvidia-ml.so.1 && \
    chmod 755 /libnvidia-ml.so.1 && \
    test -x /libnvidia-ml.so.1

# Runtime: Debian slim provides glibc for the cgo-built mpsd and `test -S` for
# the readiness/liveness probes.
FROM debian:bookworm-slim
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/lib/nvml-mock
ENV MOCK_NVML_CONFIG=/etc/nvml-mock/config.yaml

COPY --from=mpsd-builder /mpsd /usr/local/bin/mpsd
COPY --from=fake-builder /fake-mps /usr/bin/nvidia-cuda-mps-control
COPY --from=nvml-mock /libnvidia-ml.so.1 /usr/lib/nvml-mock/libnvidia-ml.so.1
COPY hack/fake-mps/nvml-mock-config.yaml /etc/nvml-mock/config.yaml
ENTRYPOINT ["/usr/local/bin/mpsd"]
