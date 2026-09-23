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

# The manager is CGO_ENABLED=0 and fully static, so it needs no base userspace.
FROM scratch
# Nothing reads this today — the in-cluster client uses the service account CA
# bundle — so it is here to keep future outbound TLS failing on an untrusted
# cert rather than on a missing root store.
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
WORKDIR /
COPY --from=builder /workspace/operator/manager .
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt
# Must stay numeric: the chart sets runAsNonRoot and scratch has no /etc/passwd.
USER 1000:1000
ENTRYPOINT ["/manager"]
