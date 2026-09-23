# FIPS 140-3

Deployments in regulated environments often may not use *any* implementation of
a cryptographic algorithm — the implementation itself has to have been submitted
to a laboratory, tested, and certified under
[FIPS 140-3](https://csrc.nist.gov/pubs/fips/140-3/final).

Go ships such an implementation, the
[Go Cryptographic Module](https://go.dev/doc/security/fips140#the-go-cryptographic-module).
This project publishes a second set of images built against it, alongside the
regular ones, and the Helm chart selects between them with a single value.

## What is and is not covered

Covered: the Go cryptographic code in the four binaries built from this
repository — `operator`, `fractiond`, `metricsd` and `mpsd`. In a FIPS image
each of those links the validated module instead of the standard library's own
crypto implementation.

Not covered: anything not compiled here. That includes the operating-system
crypto libraries inside the base images, shared libraries loaded at runtime from
the host NVIDIA driver stack, and any third-party executable that arrives as
part of a base image.

So "FIPS image" means **the Go cryptography in our binaries is FIPS-validated**.
If a requirement calls for a fully validated container filesystem — a validated
OpenSSL, a FIPS-mode host kernel — that is a materially larger piece of work and
is not what these images provide.

## The three modes

The chart value is `global.fipsMode`, and it accepts exactly three values:

| Mode | Images | Runtime enforcement |
|---|---|---|
| `off` (default) | regular | none |
| `on` | `-fips` | none needed — the build already enables FIPS mode |
| `only` | `-fips` | `GODEBUG=fips140=only,tlsmlkem=0` on every container |

Anything else fails the render rather than quietly falling back to the regular
images, since an installation that believes it is FIPS while running ordinary
crypto is worse than one that refuses to start.

`on` needs no runtime setting because building with the module also makes
`fips140=on` the binary's built-in default. Such a binary runs the module's
mandated self-tests at startup, operates in FIPS mode, and — per the Go
documentation — `crypto/tls` "will ignore and not negotiate any protocol
version, cipher suite, signature algorithm, or key exchange mechanism that is
not FIPS 140-3 approved." That enforcement is graceful: non-approved options are
simply never offered.

## Installing

FIPS images, no runtime enforcement. This is the setting to use in production:

```sh
helm install gpu-fractioning \
  oci://ghcr.io/kai-scheduler/gpu-fractioning \
  --version <VERSION> \
  --namespace gpu-fractioning --create-namespace \
  --set global.fipsMode=on
```

The `-fips` suffix is appended to the resolved image tag, so FIPS selection
composes with version pinning rather than overriding it. Pinning
`images.mpsd.tag=v1.2.3` together with `fipsMode=on` yields `v1.2.3-fips`.

The value lives under `global` so an umbrella chart can set it once for every
subchart that understands it, rather than each one being configured separately.

### `only`, and why it is not a production setting

`only` adds `GODEBUG=fips140=only` to every container, which makes a call into a
non-approved algorithm return an error or panic instead of succeeding. Straight
from the [Go documentation](https://go.dev/doc/security/fips140#the-fips140-godebug-option):

> If set to `only`, cryptographic algorithms that are not FIPS 140-3 compliant
> will return an error or panic. Note that this is a best effort mode meant for
> **testing, assessment, and debugging**. It is **not intended to be used in
> production**, it is not required by the Security Policy, it introduces crashes
> and potentially unhandled errors by design, and it may have false positives or
> false negatives.

It also reaches code we do not control, including transitive dependencies. Use
it to *assess* an installation, then run that installation with `on`.

```sh
helm upgrade gpu-fractioning ... --set global.fipsMode=only
```

`tlsmlkem=0` is set alongside it and is not optional. `crypto/tls` prefers the
`X25519MLKEM768` hybrid key exchange; that curve is FIPS-allowed, but its
implementation calls the plain X25519 primitive, which is not approved. Under
`fips140=only` every outbound TLS handshake therefore fails with:

```
crypto/ecdh: use of X25519 is not allowed in FIPS 140-only mode
```

That includes the operator's connection to the API server, which stops pods from
starting. Setting `tlsmlkem=0` drops the hybrid key exchange, so the handshake
negotiates an approved curve (P-256) and succeeds. See
[golang/go#78298](https://github.com/golang/go/issues/78298) and
[kubernetes/kubernetes#133743](https://github.com/kubernetes/kubernetes/issues/133743).

Expect a performance cost in FIPS mode. The Go documentation notes that pairwise
consistency tests on generated keys "can cause a slowdown of up to 2x for
certain key types, which is especially relevant for ephemeral keys."

## Verifying a published image

Whether a binary was built against the validated module is recorded in its Go
build information, so it can be checked after the fact without trusting the tag.
Most of these images have no shell, so copy the binary out rather than running
anything inside:

```sh
IMAGE=ghcr.io/kai-scheduler/gpu-fractioning/operator:<VERSION>-fips
CID=$(docker create --platform linux/amd64 "$IMAGE")
docker cp "$CID:/manager" ./manager && docker rm -f "$CID"

go version -m ./manager | grep -E 'GOFIPS140|DefaultGODEBUG'
```

Expected output, where the trailing hash is part of how the module version is
recorded:

```
	build	DefaultGODEBUG=fips140=on
	build	GOFIPS140=v1.0.0-c2097c7c
```

Both lines matter, and they assert different things. `GOFIPS140` says the
validated module was linked. `DefaultGODEBUG=fips140=on` says the binary
actually runs in FIPS mode rather than merely being capable of it. A regular
build emits neither line, so their absence is what distinguishes it.

The binary paths differ per image: `/manager`, `/fractiond`,
`/usr/local/bin/mpsd`, `/usr/local/bin/metricsd`.

Every release runs this check across all four images and both architectures:

```sh
hack/verify-fips-images.sh \
  --repo ghcr.io/kai-scheduler/gpu-fractioning \
  --tag <VERSION>-fips \
  --module "$(make -s print-gofips140-version)" \
  --platforms linux/amd64,linux/arm64
```

It runs before the `-fips` tag exists, not after. A multi-arch build cannot be
held locally, so the release publishes under `<VERSION>-unverified-fips`,
verifies that, and only then creates `<VERSION>-fips` as a registry-side
manifest copy. So the tag you pull exists only if the check passed, and a
compliance failure is never published under it. The staging tag remains in the
registry either way; when verification fails it is the artifact to investigate.

## Building FIPS images locally

`FIPS=1` links the pinned module and appends `-fips` to every image tag:

```sh
make docker-build FIPS=1
```

The module version is pinned to `v1.0.0`, which holds
[CMVP Certificate #5247](https://csrc.nist.gov/projects/cryptographic-module-validation-program/certificate/5247)
— an Active FIPS 140-3 Level 1 validation. Newer module versions exist but are
still under review, and for a compliance build "validated" beats "newer". The
pin is deliberately an explicit version rather than Go's `certified` alias,
which would silently change what gets linked when a new validation lands.

Because compilation happens inside each Dockerfile's builder stage, the module
version travels as a Docker build argument rather than an environment variable.
That makes the plumbing per-Dockerfile, which is why the release verifies every
published binary instead of trusting that each build passed the flag along.

The same omission is also checked without a registry, so it fails on the pull
request that introduces it rather than at the next release:

```sh
hack/verify-fips-build-args.sh
```

It reads the Dockerfiles listed by `make print-image-dockerfiles` and requires
each to declare `ARG GOFIPS140=off` and pass it to `go build`. Both scripts take
the component list from the Makefile's `IMAGES`, so adding a component there is
enough for it to be checked — and a component that is built but unverifiable
fails rather than being skipped.

## References

- [Go FIPS 140-3 documentation](https://go.dev/doc/security/fips140)
- [The `fips140` GODEBUG option](https://go.dev/doc/security/fips140#the-fips140-godebug-option)
- [CMVP Certificate #5247](https://csrc.nist.gov/projects/cryptographic-module-validation-program/certificate/5247)
