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

- New installation-time Helm value `skipGpuStackVersionChecks` (defaults to
  `false`), an escape hatch that bypasses the operator's NVIDIA GPU stack version
  gates, for an installation whose versions the operator reads wrongly — an
  operand pinned by digest, a vendored ClusterPolicy, a requirement that has
  since moved — so the daemons can roll out without a code rollback. It
  disables verification only and does not make an unsupported GPU stack work:
  with the gates off, a stack that cannot enforce GPU memory limits rolls out
  anyway, `Ready` goes True, and GPU memory goes unfenced with nothing reported
  on any condition. ClusterPolicy readiness reporting and the NVIDIA driver
  version check are unaffected.

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
- All four images move to new base images. `operator` and `fractiond` move from
  `nvcr.io/nvidia/distroless/go:v4.1.4` to `scratch`: both are `CGO_ENABLED=0`
  and fully static, so each image is now just the binary, its attribution and a
  CA bundle. `metricsd` (from the same distroless image) and `mpsd` (from the
  Docker Hub `nvidia/cuda:13.3.1-base-ubuntu24.04`) move to
  `nvcr.io/nvidia/cuda:13.3.1-base-ubi9`, whose glibc and OpenSSL are FIPS 140-3
  validated Red Hat modules; both are `CGO_ENABLED=1` and dlopen
  `libnvidia-ml.so`, so neither can be static. No image adds OS packages, mpsd's
  CUDA version is unchanged, and `operator` keeps `USER 1000:1000` — numeric,
  because `scratch` has no `/etc/passwd` for the chart's `runAsNonRoot`. FIPS
  behaviour is unchanged: it comes from `GOFIPS140` linking the validated Go
  Cryptographic Module into each binary, independent of the base image.
- Corrected the documented reason for mpsd using a CUDA base. The Dockerfile
  said the base supplies `nvidia-cuda-mps-control`; no `nvidia/cuda` image ships
  that binary. The NVIDIA container runtime injects it from the host driver
  because the container requests `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
  mpsd needs the base for glibc and for `/usr/bin/test`, which its readiness and
  liveness probes exec against the MPS control socket.
- All four images now build from an NVIDIA-approved base container. `operator`,
  `fractiond` and `metricsd` move from `gcr.io/distroless/*` to
  `nvcr.io/nvidia/distroless/go:v4.0.8`; `mpsd` stays on the approved public
  `nvidia/cuda` base. No OS packages are added on top of any base. The operator
  container now runs as uid **1000** (the base image's `nvs` user) instead of
  65532, and `fractiond` and `metricsd` set `USER 0:0` explicitly because the
  new base defaults to a non-root user and both need root (the NRI socket and
  NVML respectively).
- The GPU Operator dependency check now gates on the operand versions gpu-fractioning actually depends on — **container toolkit v1.20.1** and **device plugin v0.20.1**, read from the `ClusterPolicy` `spec.toolkit.version` and `spec.devicePlugin.version` — instead of inferring them from the GPU Operator version, which is no longer gated at all where a `ClusterPolicy` exists. v26.7.1 is still the first release whose defaults satisfy both operands, but the operator version was never the real requirement: each operand version is independently overridable, so the old check rejected working clusters (v26.7.0 with the operands overridden, as the README documented) and admitted broken ones (v26.7.1 or newer with the toolkit pinned below its floor, where fractiond injects GPU-memory limits that no CDI hook applies and GPU memory goes unfenced with nothing reported). An operand below its minimum is now reported on the `Ready` condition with the new reason `GPUOperandVersionUnsupported` and the node-level daemons are not rolled out. An operand left unset on the `ClusterPolicy` runs the GPU Operator's own build default, so it is resolved through the GPU Operator floor, which stays at **v26.7.1** — the first release whose defaults satisfy both operands. That inference is valid precisely because nothing pinned the operand, and so is not applied to one that is pinned. An operand pinned to an opaque tag such as a digest is neither: the GPU Operator version says nothing about it and an unreadable version is not evidence of an unsupported one, so it does not block. The same floor covers installs with no `ClusterPolicy` at all, where the version comes from the OLM `ClusterServiceVersion` and no operand versions are readable; an install that overrode its operands on top of an older operator is rejected there, since the overrides are invisible — `skipGpuStackVersionChecks` covers it. Distro-suffixed toolkit tags (`v1.20.1-ubuntu20.04`) compare on their release core, since as semver prereleases they would order *below* the floor they meet.

### Fixed
- The fractiond and mpsd pods now carry the NVIDIA GPU Operator's third-party GPU client label, `nvidia.com/gpu.deploy.client: "true"`, in their nodeSelector, so a GPU Operator upgrade evicts them before it unloads the NVIDIA driver. Previously the daemons were invisible to the upgrade: its pod-deletion step selects only pods that requested the `nvidia.com/gpu` resource — which these deliberately do not, taking GPUs through the nvidia runtime instead — and both that step and the drain fallback hardcode `IgnoreAllDaemonSets`. The existing nodeAffinity on `nvidia.com/gpu-driver-upgrade-state` was therefore the only thing that drained them, and it only fires once something sets that label, which a `helm upgrade` of the GPU Operator does not always do (the driver DaemonSet is always `OnDelete`, so an *absent* driver pod is recreated from the new template outside the upgrade state machine). mpsd then kept the kernel module pinned through `nvidia-cuda-mps-control`, the unload failed with `resource temporarily unavailable`, and `nvidia-driver-daemonset` entered CrashLoopBackOff, taking the node's GPUs down until an admin intervened. Unlike the upgrade-state label, the GPU Operator always maintains this one and waits for the labelled pods to terminate before unloading the driver, which also closes the race where mpsd was still inside its graceful MPS shutdown window. The label is added by the operator when it builds the DaemonSets rather than required in `spec.nodeSelector`, which is immutable, so existing `GpuFractioningConfig` resources need no change. The `nvidia.com/gpu-driver-upgrade-state` nodeAffinity is retained as a second, independent drain signal. Requires GPU Operator v26.7.0 or newer, which is already below the minimum supported version.
- fractiond now injects the GPU-memory limits under the names their consumer actually reads: `NVIDIA_GPU_MEMORY_REQUEST` and `NVIDIA_GPU_MEMORY_LIMIT`, singular where they were previously plural. The NVIDIA container toolkit's `apply-cuda-memory-limits` CDI hook looks both up by exact name and returns early when it finds neither, so under the plural spelling an injected limit was never applied and the container's GPU memory went unfenced. The values are unchanged (whole MiB). The retroactive-enforcement audit looks for the new names as well, so it does not mistake a correctly injected container for one that slipped through.
- The mpsd pod now runs with `hostPID: true`, without which MPS memory accounting could not register any client and GPU memory limits went unenforced. The MPS control daemon identifies a client by the PID in its socket's peer credentials, and the kernel only translates that PID for the daemon's own PID namespace or a descendant of it; from inside its own pod namespace mpsd therefore saw every workload container as pid 0, logged `[memacct] failed to register client pid 0`, and attributed memory-accounting events to `target=unknown`. The host PID namespace is an ancestor of every container's, so PIDs and their cgroups now resolve. Workloads require no change.
- fractiond now defaults a missing GPU-memory request or limit from the other (so `request == limit`). A container that annotates only `.request` now also gets `NVIDIA_GPU_MEMORY_LIMIT` injected (enforced at the requested size instead of being unbounded), and a container that annotates only `.limit` gets `NVIDIA_GPU_MEMORY_REQUEST` populated. The retroactive-enforcement audit applies the same defaulting.
