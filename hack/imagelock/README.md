<!-- Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ImageLock generator

This tool generates the architecture- and profile-specific, digest-pinned
ImageLock files published with KAI GPU Fractioning releases. The locks let
air-gapped users mirror the exact container manifests required by a release
instead of relying on mutable tags.

The release contract is declared in `catalog.yaml`. Helm is not used to decide
which images were published: the catalog defines release intent, and the
container registry supplies the index and platform-manifest digests that were
actually published.

The tool lives in its own Go module so release-tool dependencies do not enter
the shipped binaries' dependency or license graph.

## Release catalog

The catalog has four independent sections:

- `images`: production image names below the shared repository;
- `profiles`: the tag template used by each image variant;
- `platforms`: the platform manifests required in each image index;
- `exclude`: deliberately unsupported image/profile/platform combinations.

The committed catalog currently expands to:

```text
(operator, fractiond, metricsd, mpsd)
  × (standard, fips)
  × (linux/amd64, linux/arm64)
```

The `standard` profile resolves `<version>` tags. The `fips` profile resolves
`<version>-fips` tags. A release fails if any non-excluded image tag, index, or
platform manifest cannot be resolved.

Adding a production image to `images` automatically adds it to every profile
and platform unless an explicit exclusion applies.

The Make targets pass the production `IMAGES` build list to the tool as
repeatable `--expected-image` arguments. Catalog validation fails if the build
inventory and catalog inventory differ, so adding `whateverd` to only one side
cannot silently produce an incomplete release.

The Make targets also pass `operator/charts/values.yaml`. The tool compares the
keys and repositories under `images` directly with the catalog. This preserves
deployment-inventory drift detection without running Helm or using rendered
manifests to infer release tags.

## Exclusions

Exclusions use structured selectors. Each dimension accepts a catalog value or
`"*"`, and every rule requires a reason.

```yaml
exclude:
  # No mpsd on linux/amd64, for any profile.
  - image: mpsd
    profile: "*"
    platform: linux/amd64
    reason: "Example unsupported platform"

  # No fractiond FIPS image, on any platform.
  - image: fractiond
    profile: fips
    platform: "*"
    reason: "Example unsupported variant"

  # No metricsd FIPS image on linux/amd64.
  - image: metricsd
    profile: fips
    platform: linux/amd64
    reason: "Example unsupported combination"
```

The catalog validator rejects:

- unknown image, profile, or platform selectors, including spelling mistakes;
- duplicate or overlapping exclusions;
- exclusions without a reason;
- exclusions that do not match the declared matrix;
- exclusions that leave an entire profile/platform lock empty.

An exclusion must represent a genuinely unsupported or unused combination. It
must not be used merely to hide an image that failed to build: doing so would
produce an incomplete air-gap lock.

## How generation works

1. Strictly parse and validate `catalog.yaml`.
2. Expand the image × profile × platform matrix.
3. Apply and validate the exclusions.
4. Resolve every remaining tagged image through the default Docker credential
   keychain.
5. Require a multi-platform index containing exactly one manifest for every
   required platform.
6. Record the shared index digest and each platform's child digest.
7. Stage and atomically publish one deterministic ImageLock per profile and
   platform.

## Usage

```bash
# From the repository root.
make image-lock-verify
make image-lock-test
make image-lock VERSION=vX.Y.Z
```

Generation requires registry access to all catalog tags. Authenticate with
Docker, for example with `docker login ghcr.io`, before running it locally.

Use `IMAGE_LOCK_CATALOG`, `IMAGE_LOCK_CHART_VALUES`, or `IMAGE_LOCK_OUT_DIR` to
override the catalog, chart-values, and output paths used by Make.

The committed catalog writes:

```text
dist/
  imagelock-gpu-fractioning-vX.Y.Z-linux-amd64.yaml
  imagelock-gpu-fractioning-vX.Y.Z-linux-arm64.yaml
  imagelock-gpu-fractioning-vX.Y.Z-fips-linux-amd64.yaml
  imagelock-gpu-fractioning-vX.Y.Z-fips-linux-arm64.yaml
```

In every lock, `metadata.version` is the GitHub release version. `spec.profile`
selects the variant, so FIPS `source` values use `<version>-fips` while their
lock metadata still uses `<version>`. The standard profile is the default and
is therefore omitted from filenames; non-standard profile names are included.

## Air-gap install

Download the Helm chart and the lock matching the desired profile and node
architecture from the GitHub Release. A registry-copy tool such as `skopeo`,
`crane`, or `regctl` can then mirror every digest-pinned `image` reference into
the private registry.

```bash
LOCK=imagelock-gpu-fractioning-vX.Y.Z-linux-amd64.yaml
yq -r '.spec.images[] | .image + " " + .source' "$LOCK" | while read -r digest_ref tag_ref; do
  echo "$digest_ref -> $tag_ref"
done
```

Use the first value as the immutable source to copy. Preserve or map the tagged
`source` name according to the private registry layout used by the Helm values.
