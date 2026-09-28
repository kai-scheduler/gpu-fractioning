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

# Runtime stage — UBI9 base, pulled from NGC. Installs no new OS packages; it
# patches the ones the base already carries and drops those nothing here uses
# (see below).
#
# mpsd needs two things from it: glibc, because it is built CGO_ENABLED=1 and
# the NVML bindings dlopen libnvidia-ml.so at runtime, and /usr/bin/test, which
# the readiness and liveness probes exec against the MPS control socket.
#
# It does NOT need the base for nvidia-cuda-mps-control. No nvidia/cuda image
# ships that binary; the NVIDIA container runtime injects it from the host
# driver because this container requests NVIDIA_DRIVER_CAPABILITIES=compute.
# The base is CUDA only because NGC publishes no plain UBI9 image.
FROM nvcr.io/nvidia/cuda:13.4.1-base-ubi9

# Base-image hardening, byte-identical to the metricsd runtime stage — change
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

COPY --from=builder /mpsd /usr/local/bin/mpsd
# Attribution for the third-party code statically linked into the binary.
COPY THIRD-PARTY.txt /THIRD-PARTY.txt

ENTRYPOINT ["/usr/local/bin/mpsd"]
