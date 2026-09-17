# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- GPU **compute** limits, not just memory limits. A container whose pod carries
  the new per-container annotation
  `nvidia.com/container.<container-name>.gpu-compute.portion` (a GPU portion in
  `(0, 1]`, written by the scheduler at bind time from the portion it charged
  the workload's quota for) now has `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` injected
  alongside its memory limits, so MPS caps its SM occupancy to match. Without
  it, `sm-sharing` let two containers occupy the SMs concurrently but put no
  ceiling on either: a container holding half a GPU's memory could still take
  all of its compute. The cap applies in both compute modes — under
  `time-slicing` it bounds the container's own MPS server, under `sm-sharing`
  its share of the shared one. Absent annotation means no cap, the pre-existing
  behaviour.
- Memory limits are now also enforced by MPS itself, via a
  `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` injected per assigned GPU. This
  deliberately duplicates the cap `NVIDIA_GPU_MEMORY_LIMIT` asks the container
  toolkit's `apply-cuda-memory-limits` CDI hook to apply, because the two have
  different prerequisites: the hook needs GPU Operator v26.7.1+, this needs only
  MPS. Clusters with one but not the other still get exactly one enforced limit;
  clusters with both set the same value twice.
- The limit-carrying env vars are now removed before being injected, so a
  container that ships its own `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` or
  `NVIDIA_GPU_MEMORY_*` cannot keep it. Previously a workload could have set
  them itself and opted out of its own caps.
- Automatic MPS client draining before a fractional container is stopped, and
  MPS recycling once a node goes idle — a workaround for the MPS wedge
  documented by NVIDIA for CUDA 13.5 and earlier. An MPS client killed with GPU
  work in flight orphans that work; every other client on the same GPU then
  fails with `cudaErrorIllegalAddress`, and if nothing consumes the resulting
  fault, every client that connects afterwards hangs forever inside CUDA init
  while the control plane reports healthy throughout. Only an MPS restart clears
  it. fractiond's NRI `StopContainer` hook now calls a new mpsd endpoint while
  the container is still alive, and mpsd terminates its MPS clients through the
  control daemon — NVIDIA's supported workaround, which drains outstanding work
  first — with no preStop hook or workload cooperation required. The hook waits
  only briefly (clamped to half the request timeout the NRI runtime negotiated,
  because overrunning it is fatal and closes the plugin) and mpsd completes the
  drain in the background. A terminate
  that does not return is taken as evidence the server is already wedged and
  escalates to a hard MPS restart that also kills any MPS server left behind.
  When a drain leaves no clients attached, mpsd recycles MPS (`mpsDaemon.recycleWhenIdle`,
  on by default): nothing is attached, so the restart is free, and a fault
  picked up during one run of a benchmark sweep cannot survive into the next.
  Drain failures never block a container stop.
- Per-daemon `resources` overrides on the CR
  (`spec.fractioningAgent.resources`, `spec.metricsAgent.resources`,
  `spec.mpsDaemon.resources`), merged over the built-in defaults so raising one
  limit does not drop the requests the pods' QoS class depends on. Previously
  the only way to change a daemon's resources was to patch the DaemonSet
  directly, which the controller then reconciled away.
- The minimum NVIDIA GPU Operator version is now configurable
  (`gpuOperator.minimumVersion`, default `v26.7.1`; `""` skips the check). What
  v26.7.1 brings is the container toolkit carrying the CDI memory-limit hook;
  with MPS-enforced limits now injected as well, a cluster on an older GPU
  Operator can run with MPS-only enforcement instead of being blocked.
- `make crd-validations-check` fails when the chart's CRD and the
  controller-gen CRD disagree on their CEL validation rules. The chart ships a
  deliberately trimmed copy of the CRD and that copy is what installs, so a rule
  added through a kubebuilder marker and not mirrored there is enforced on no
  real cluster — with both files individually valid, nothing else in the build
  noticed.
- The operator now sets the `gpu-fractioning.nvidia.com/Ready` node condition to
  `Unknown` on every targeted node as it shuts down
  (`markNodesUnknownOnShutdown`, on by default; only the replica that held
  leadership does it). Nothing else expires that condition, so scaling the
  operator to zero previously froze every node's readiness at whatever it last
  said, and a scheduler gating fractional placement on it kept placing pods on
  nodes whose daemon health nobody was watching.
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
- mpsd could not start on a node with more than about five GPUs. Its container
  memory limit was hardcoded at 256Mi, but the MPS control daemon holds a CUDA
  server context per GPU at roughly 50 MiB of host memory each, so the sixth
  context ran the pod out of memory. The failure was badly misleading: the CUDA
  allocation failed before the kernel could OOM-kill anything, so the pod exited
  1 with `CUDA_ERROR_OUT_OF_MEMORY` — reading as a GPU memory problem — while
  every GPU on the node sat idle at 1 MiB used. The default is now 1Gi (enough
  for a 16-GPU node) and is overridable via `spec.mpsDaemon.resources`.
- Helm values for the daemon stanzas now actually reach the
  `GpuFractioningConfig`. The default-CR template enumerated a handful of fields,
  so `--set fractioningAgent.nriSocketPath=...` was accepted by Helm and then
  silently dropped — on exactly the distributions (microk8s, k3s, RKE2) whose
  non-default NRI and CRI socket paths make that field necessary. All three
  stanzas (`fractioningAgent`, `metricsAgent`, `mpsDaemon`) are now passed
  through verbatim, so every CRD field is reachable from values.
- fractiond now injects the GPU-memory limits under the names their consumer actually reads: `NVIDIA_GPU_MEMORY_REQUEST` and `NVIDIA_GPU_MEMORY_LIMIT`, singular where they were previously plural. The NVIDIA container toolkit's `apply-cuda-memory-limits` CDI hook looks both up by exact name and returns early when it finds neither, so under the plural spelling an injected limit was never applied and the container's GPU memory went unfenced. The values are unchanged (whole MiB). The retroactive-enforcement audit looks for the new names as well, so it does not mistake a correctly injected container for one that slipped through.
- The mpsd pod now runs with `hostPID: true`, without which MPS memory accounting could not register any client and GPU memory limits went unenforced. The MPS control daemon identifies a client by the PID in its socket's peer credentials, and the kernel only translates that PID for the daemon's own PID namespace or a descendant of it; from inside its own pod namespace mpsd therefore saw every workload container as pid 0, logged `[memacct] failed to register client pid 0`, and attributed memory-accounting events to `target=unknown`. The host PID namespace is an ancestor of every container's, so PIDs and their cgroups now resolve. Workloads require no change.
- fractiond now defaults a missing GPU-memory request or limit from the other (so `request == limit`). A container that annotates only `.request` now also gets `NVIDIA_GPU_MEMORY_LIMIT` injected (enforced at the requested size instead of being unbounded), and a container that annotates only `.limit` gets `NVIDIA_GPU_MEMORY_REQUEST` populated. The retroactive-enforcement audit applies the same defaulting.
