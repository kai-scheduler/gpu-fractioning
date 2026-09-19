# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- CNCF project-repository requirements, ahead of making the repository public in
  the `kai-scheduler` organization: `GOVERNANCE.md` (this repository's own
  governance — its maintainers, decision making, and how that group changes),
  `MAINTAINERS.md`, `ADOPTERS.md`, `SUPPORT.md`, `ROADMAP.md`, and
  `LICENSE-docs` (CC-BY-4.0) alongside the Apache-2.0 `LICENSE`, per CNCF
  Charter section 11. `CODE_OF_CONDUCT.md` now explicitly adopts the CNCF Code
  of Conduct and names <conduct@cncf.io> as an escalation path, and the README
  references it, carries the CNCF footer and the LF Projects
  copyright/trademark notice, and states the dual code/docs licensing.
- Open-source compliance plumbing ahead of the public release: `CLA.md`
  (Developer Certificate of Origin 1.1), `CODE_OF_CONDUCT.md`, a generated
  `THIRD-PARTY.txt` covering the 92 third-party Go modules linked into the
  shipped binaries (also shipped in every image at `/THIRD-PARTY.txt`), and the
  Apache-2.0 SPDX header on every authored file. `make license-check` and
  `make third-party-check` enforce both in CI.
- metricsd metric names are now configurable via Helm (`metricsAgent.metricNames.{gpuMemoryUsedBytes,gpuSmUtilizationPercent,gpuSmUtilizationPercentNormalized}`), and the per-pod metric labels are `namespace, pod, pod_uuid, gpu_uuid, gpu` (renamed from `pod_uid`/`gpu_index`) to integrate with external metric consumers.
- New `sm-sharing` GPU compute-sharing mode, selected per-container via `nvidia.com/container.<container-name>.gpu-compute.mode: "time-slicing" | "sm-sharing"` (defaults to `time-slicing`, today's unchanged behavior). mpsd runs a second, parameterless MPS server (`context-share` enabled, with the default socket excluded from context sharing so only containers routed to the shared server share a context — time-slicing containers keep the default socket and their memacct-enforced memory limits) alongside the default one; fractiond routes a container annotated `sm-sharing` to that shared server's socket instead, so its GPU compute is shared via MPS (concurrent SM occupancy) rather than time-sliced. Any other annotation value — including a present-but-empty one — fails container creation (or falls back to `time-slicing` under fail-open). The whole feature is gated by a new installation-time Helm value, `supportSmSharing` (defaults to `true`); disabling it reverts to pre-feature behaviour without a code rollback — mpsd reverts to its pre-feature MPS config and daemon invocation (the shared server and its required multiuser mode both go away) and fractiond rejects the `sm-sharing` annotation like any other invalid value. The toggle applies to containers created afterwards and does not migrate running ones, so stop sm-sharing workloads and confirm the shared server has no clients before disabling it.

- FIPS 140-3 support. Every release now publishes a second set of images, tagged
  `<version>-fips`, whose Go binaries link the CMVP-validated Go Cryptographic
  Module (pinned to `v1.0.0`, CMVP Certificate #5247) instead of the standard
  library's own crypto; the release verifies the linked module in every binary
  before the `-fips` tag exists — it builds under `<version>-unverified-fips`,
  checks that, then promotes — so a `-fips` tag cannot ship ordinary crypto and
  the staging tag is all that remains if the check fails. A new chart
  value, `global.fipsMode`, selects between them: `off` (the default, entirely
  unchanged behaviour), `on` (FIPS images — the compliant production setting,
  needing no runtime flag because a module-linked binary already runs in FIPS
  mode and `crypto/tls` already declines non-approved options gracefully), and
  `only`, which additionally sets `GODEBUG=fips140=only,tlsmlkem=0` on the
  operator and on every daemon it builds, making calls into non-approved
  algorithms fail loudly. Per upstream Go, `only` is a best-effort mode for
  testing and assessment that is crash-prone by design and not intended for
  production; `tlsmlkem=0` accompanies it because `crypto/tls` otherwise
  prefers a hybrid key exchange whose implementation calls an unapproved
  primitive, which fails every outbound TLS handshake. The suffix is applied to
  the resolved image tag, so FIPS selection composes with per-image version
  pinning rather than overriding it, and any value other than the three above
  fails the render instead of silently falling back to non-FIPS images. Locally,
  `make docker-build FIPS=1` produces the same variants. See
  [`docs/fips/README.md`](docs/fips/README.md) for scope — this covers the Go
  cryptography in binaries built from this repository, not base-image OS crypto
  or libraries loaded from the host driver stack.

### Changed
- All four images now build from an NVIDIA-approved base container. `operator`,
  `fractiond` and `metricsd` move from `gcr.io/distroless/*` to
  `nvcr.io/nvidia/distroless/go:v4.0.8`; `mpsd` stays on the approved public
  `nvidia/cuda` base. No OS packages are added on top of any base. The operator
  container now runs as uid **1000** (the base image's `nvs` user) instead of
  65532, and `fractiond` and `metricsd` set `USER 0:0` explicitly because the
  new base defaults to a non-root user and both need root (the NRI socket and
  NVML respectively).
- The minimum supported NVIDIA GPU Operator version is now **v26.7.1** (was v26.7.0). A cluster running v26.7.0 is reported as unsupported on the `GpuFractioningConfig` `Ready` condition and the node-level daemons are not rolled out. Note that a ClusterPolicy labelled only `26.7` normalizes to `v26.7.0` and is therefore also rejected; label it with the full patch version.

### Fixed
- GPU Operator prereleases from the minimum supported patch train are now
  accepted by the dependency check. For example, `v26.7.1-rc.1` satisfies the
  `v26.7.1` minimum, while prereleases from older patch trains remain rejected.
- fractiond now injects the GPU-memory limits under the names their consumer actually reads: `NVIDIA_GPU_MEMORY_REQUEST` and `NVIDIA_GPU_MEMORY_LIMIT`, singular where they were previously plural. The NVIDIA container toolkit's `apply-cuda-memory-limits` CDI hook looks both up by exact name and returns early when it finds neither, so under the plural spelling an injected limit was never applied and the container's GPU memory went unfenced. The values are unchanged (whole MiB). The retroactive-enforcement audit looks for the new names as well, so it does not mistake a correctly injected container for one that slipped through.
- The mpsd pod now runs with `hostPID: true`, without which MPS memory accounting could not register any client and GPU memory limits went unenforced. The MPS control daemon identifies a client by the PID in its socket's peer credentials, and the kernel only translates that PID for the daemon's own PID namespace or a descendant of it; from inside its own pod namespace mpsd therefore saw every workload container as pid 0, logged `[memacct] failed to register client pid 0`, and attributed memory-accounting events to `target=unknown`. The host PID namespace is an ancestor of every container's, so PIDs and their cgroups now resolve. Workloads require no change.
- fractiond now defaults a missing GPU-memory request or limit from the other (so `request == limit`). A container that annotates only `.request` now also gets `NVIDIA_GPU_MEMORY_LIMIT` injected (enforced at the requested size instead of being unbounded), and a container that annotates only `.limit` gets `NVIDIA_GPU_MEMORY_REQUEST` populated. The retroactive-enforcement audit applies the same defaulting.
