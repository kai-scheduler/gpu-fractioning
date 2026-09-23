#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verifies that published "-fips" images really were built against the pinned Go
# Cryptographic Module, by reading the build info out of every shipped binary.
#
# GOFIPS140 is threaded through each Dockerfile individually, so a component
# built without its build arg would publish a "-fips" image containing ordinary
# crypto. The component list is therefore read from the Makefile rather than
# kept here, so a component that is built but not verified fails the drift
# check below instead of being skipped. Only the in-image binary paths live
# here, since nothing else knows them.
#
#   hack/verify-fips-images.sh --repo ghcr.io/kai-scheduler/gpu-fractioning \
#     --tag v0.1.0-fips --module v1.0.0 --platforms linux/amd64,linux/arm64

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# <service>:<path of the binary inside the image>
TARGETS=(
  "operator:/manager"
  "fractiond:/fractiond"
  "mpsd:/usr/local/bin/mpsd"
  "metricsd:/usr/local/bin/metricsd"
)

usage() {
  sed -n '5,17p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

# Guards against this check and the build disagreeing about what ships. Without
# it, a component missing from TARGETS is simply never inspected, which is the
# failure mode the header describes.
assert_no_component_drift() {
  local built checked
  built="$(cd "$repo_root" && make -s print-image-names | tr ' ' '\n' | sort)"
  checked="$(printf '%s\n' "${TARGETS[@]%%:*}" | sort)"

  [[ "$built" == "$checked" ]] && return 0

  echo "FAIL  component drift between the Makefile and $(basename "$0"):" >&2
  comm -13 <(echo "$checked") <(echo "$built") |
    sed 's/^/        built but not verified here: /' >&2
  comm -23 <(echo "$checked") <(echo "$built") |
    sed 's/^/        verified here but not built: /' >&2
  exit 1
}

repo=""
tag=""
module=""
platforms="linux/amd64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)      repo="$2"; shift 2 ;;
    --tag)       tag="$2"; shift 2 ;;
    --module)    module="$2"; shift 2 ;;
    --platforms) platforms="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "$repo" && -n "$tag" && -n "$module" ]] || {
  echo "--repo, --tag and --module are required" >&2
  usage
}

assert_no_component_drift

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

# Copies a binary out of an image without running it, so a foreign-arch image
# needs no emulation here.
extract() { # <image ref> <platform> <path in image> <destination>
  local cid
  cid="$(docker create --platform "$2" "$1")"
  docker cp "$cid:$3" "$4" >/dev/null
  docker rm -f "$cid" >/dev/null
}

status=0
bin="$workdir/bin-under-test"

for target in "${TARGETS[@]}"; do
  svc="${target%%:*}"
  path="${target#*:}"

  for platform in ${platforms//,/ }; do
    extract "${repo}/${svc}:${tag}" "$platform" "$path" "$bin"
    info="$(go version -m "$bin")"
    rm -f "$bin"

    # The recorded value carries a build suffix (GOFIPS140=v1.0.0-c2097c7c), so
    # this is deliberately a prefix match on the pinned version.
    if grep -q "GOFIPS140=${module}" <<<"$info"; then
      echo "ok    ${svc} ${platform}: linked against ${module}"
    else
      echo "FAIL  ${svc} ${platform}: pinned module ${module} is not linked"
      status=1
    fi

    # Linking the module is not the same as running in FIPS mode. A GOFIPS140
    # build also bakes in this default, and it is what makes the binary FIPS by
    # default rather than merely FIPS-capable.
    if grep -q 'DefaultGODEBUG=.*fips140=on' <<<"$info"; then
      echo "ok    ${svc} ${platform}: fips140=on is the built-in default"
    else
      echo "FAIL  ${svc} ${platform}: fips140 is not on by default"
      status=1
    fi
  done
done

exit $status
