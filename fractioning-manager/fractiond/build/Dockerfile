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

# fractiond is CGO_ENABLED=0 and fully static, with no GPU or OS-level
# dependencies — it only talks over the NRI Unix socket and serves /readyz — so
# it needs no base userspace.
FROM scratch
# Nothing reads this today: fractiond makes no outbound TLS connections. It is
# here to keep any future one failing on an untrusted cert rather than on a
# missing root store.
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /workspace/fractiond /fractiond
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt
# Root, because the NRI socket fractiond connects to is root-owned. Explicit
# rather than inherited from scratch's default.
USER 0:0
ENTRYPOINT ["/fractiond"]
