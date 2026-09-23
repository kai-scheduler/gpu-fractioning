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

# UBI9 base, pulled from NGC. Installs no new OS packages; it patches the ones
# the base already carries and drops those nothing here uses (see below).
#
# metricsd needs a glibc and dynamic loader: it is built CGO_ENABLED=1 and the
# NVML bindings dlopen libnvidia-ml.so at runtime, which the NVIDIA container
# runtime injects from the host driver.
#
# It does not need CUDA — the base is CUDA only because NGC publishes no plain
# UBI9 image. Kept on the same tag as mpsd so the layers are pulled once on a
# GPU node, where metricsd (a sidecar in the fractiond DaemonSet) and mpsd are
# always co-resident.
FROM nvcr.io/nvidia/cuda:13.3.1-base-ubi9

# Base-image hardening, byte-identical to the mpsd runtime stage — change
# one, change the other. Unused tooling is removed rather than patched, and
# omitting --nodeps lets rpm verify the runtime closure at build time. Steps
# are &&-chained so a failed step fails the build; set -e alone would not, as
# errexit is exempt for non-final commands in an && list. Keeps glibc, bash
# and coreutils (mpsd's probes exec `test`).
RUN set -e; \
    dnf -y --setopt=install_weak_deps=False update && \
    dnf -y clean all && \
    PRUNE="" && \
    for pkg in \
        gdb-gdbserver vim-minimal \
        python3 python3-libs python3-chardet python3-dateutil python3-dbus \
        python3-decorator python3-gobject-base python3-gobject-base-noarch \
        python3-gpg python3-idna python3-iniparse python3-inotify python3-pip-wheel \
        python3-pysocks python3-requests python3-rpm python3-setuptools \
        python3-setuptools-wheel python3-six python3-systemd python3-urllib3 \
        systemd systemd-libs systemd-pam systemd-rpm-macros \
        dbus dbus-broker dbus-common dbus-libs \
        dnf dnf-data yum python3-dnf python3-dnf-plugins-core libdnf python3-libdnf \
        python3-hawkey libcomps python3-libcomps librepo python3-librepo \
        libmodulemd libsolv librhsm \
        subscription-manager subscription-manager-rhsm-certificates \
        libdnf-plugin-subscription-manager python3-subscription-manager-rhsm \
        python3-cloud-what \
        glib2 gobject-introspection json-glib \
        libuser usermode passwd shadow-utils \
        virt-what dmidecode iproute libbpf libmnl psmisc \
        gnupg2 gpgme libassuan libksba npth rpm-sign-libs crypto-policies-scripts \
        expat util-linux util-linux-core libblkid libmount libfdisk libsmartcols \
        libuuid \
    ; do \
        rpm -q "$pkg" >/dev/null 2>&1 && PRUNE="$PRUNE $pkg" || true; \
    done && \
    if [ -n "$PRUNE" ]; then rpm -e $PRUNE; fi && \
    rm -rf /var/cache/dnf /var/cache/yum

# Root, because reading per-process GPU state through NVML requires it. The base
# image defaults to its non-root `nvs` user, so this has to be explicit.
USER 0:0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY --from=build /out/metricsd /usr/local/bin/metricsd
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt

ENTRYPOINT ["/usr/local/bin/metricsd"]
