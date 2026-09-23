# GPU Fractioning Operator

A Kubernetes operator that enables **multiple pods to safely share a single GPU with enforced memory boundaries**.

The operator manages the full lifecycle of GPU fractioning on a cluster: it deploys an [NRI](https://github.com/containerd/nri) plugin that injects per-container GPU memory limits at container creation time — before the container process starts — and runs [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/index.html) with GPU memory accounting enabled on every shared GPU.

It is designed to run alongside [KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler): KAI Scheduler decides *which* fraction of *which* GPU a workload gets, and gpu-fractioning enforces that memory boundary on the node and exports per-pod metrics for the resulting fractional GPUs.

## How It Works

1. A cluster admin installs the operator and a `GpuFractioningConfig` custom resource is created (the Helm chart ships a default one).
2. The **operator** (controller) reconciles the CR and rolls out the node-level components as DaemonSets to the selected GPU nodes.
3. **mpsd** runs an NVIDIA MPS control daemon on each node with per-process GPU memory accounting (`memacct`) enabled, plus a parameterless shared MPS server (`context-share`) that `sm-sharing` containers can opt into (see below) instead of the default per-node MPS socket.
4. **fractiond** registers as an NRI plugin with the container runtime. When a pod carrying GPU-memory annotations is created, fractiond injects `NVIDIA_GPU_MEMORY_REQUEST` / `NVIDIA_GPU_MEMORY_LIMIT` into the container **before it starts**, along with an MPS pipe mount — by default (`time-slicing`) to mpsd's default per-node socket, or, for a container annotated `gpu-compute.mode: sm-sharing`, to mpsd's shared MPS server instead, so its GPU **compute** (not just memory) is shared via MPS with other `sm-sharing` containers.
5. The NVIDIA driver enforces `NVIDIA_GPU_MEMORY_LIMIT` as a hard cap, so a container cannot allocate beyond its share and impact its neighbors on the same GPU. A container that exceeds its limit is terminated (out-of-memory), the same way a container exceeding its Kubernetes memory limit is.
6. **metricsd** (a sidecar alongside fractiond) exports per-pod GPU memory and utilization metrics for the shared GPUs.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Controller (Deployment)                                  │
│  Reconciles GpuFractioningConfig CR → manages the DaemonSets  │
└───────────────┬───────────────────────────┬───────────────┘
                │                           │
                ▼                           ▼
┌───────────────────────┐   ┌───────────────────────────────┐
│  mpsd (DaemonSet)     │   │  fractiond (DaemonSet)         │
│  Per-node MPS control │   │  NRI plugin — injects GPU     │
│  daemon; GPU memory   │   │  memory limits into containers│
│  accounting/enforce   │   │  at creation time             │
│                       │   │   └── metricsd (sidecar):     │
│                       │   │       per-pod GPU metrics     │
└───────────────────────┘   └───────────────────────────────┘
```

## Components

| Component | Description |
|-----------|-------------|
| **operator** | Kubernetes controller that reconciles `GpuFractioningConfig` and manages the node-level DaemonSets |
| **mpsd** | Runs and supervises the NVIDIA MPS control daemon on each GPU node |
| **fractiond** | NRI plugin that enforces per-container GPU memory limits at container creation |
| **metricsd** | Sidecar that exports per-pod GPU memory/utilization metrics for shared GPUs |

## Prerequisites

- Kubernetes 1.28+
- containerd 2.0+ with **NRI enabled**, or CRI-O with NRI support
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) **v26.7.0 or newer**, which provides the default `nvidia` [RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/) for daemon GPU/NVML access, running a **container toolkit v1.20.1 or newer** and a **device plugin v0.20.1 or newer** — see below
- **NVIDIA driver `r615` or newer (CUDA 13.4)** on the GPU nodes — see below, this is *not* the GPU Operator default
- A scheduler that assigns fractional GPUs — designed to run alongside [KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler)

### Selecting the r615 driver

`r615` is a short-lived branch, so GPU Operator v26.7.1 does **not** install it by default. You have to ask for it explicitly when installing the GPU Operator:

```sh
helm install gpu-operator nvidia/gpu-operator \
  --version v26.7.1 \
  --namespace gpu-operator --create-namespace \
  --set driver.version=615.<patch>   # any r615 release
```

The mpsd DaemonSet labels each GPU node with the node-local NVIDIA driver major version at startup (`gpu-fractioning.kai.scheduler/nvidia-driver-version.major`) and the dependency check requires major **>= 615**, so any `r615` release satisfies it.

Driver-version diagnostics intentionally use this gpu-fractioning-owned label rather than the GPU Operator label `nvidia.com/cuda.driver-version.major`. The GPU Operator label can be missing, stale, or unavailable when the NVIDIA driver is installed by another mechanism, such as managed cloud images or custom node images. Reading the actual node-local driver version through NVML keeps the reported reason tied to the driver state that the node daemons run against.

The label is written by mpsd during startup. NVIDIA GPU Operator driver upgrades drain and reschedule the mpsd pod, so the label is refreshed after those supported upgrades. If you change the driver out of band without recreating the pod, delete the mpsd pod on that node so startup reruns and refreshes the label.

Use v26.7.1 rather than v26.7.0: on v26.7.1 the bundled device-plugin and container-toolkit versions are already the ones GPU fractioning needs, and the driver is the only thing you have to override. On v26.7.0 the device-plugin and toolkit had to be overridden as well.

### GPU Operator operand versions

What GPU fractioning depends on is two GPU Operator operands, not the GPU Operator version that bundles them:

| Operand | Minimum | Why |
|---------|---------|-----|
| container toolkit | `v1.20.1` | carries the `apply-cuda-memory-limits` CDI hook that applies the GPU-memory limits fractiond injects |
| device plugin | `v0.20.1` | required for fractional GPU allocation |

Both are independently overridable, so v26.7.0 works as long as they are overridden along with the driver:

```sh
helm install gpu-operator nvidia/gpu-operator \
  --version v26.7.0 \
  --namespace gpu-operator --create-namespace \
  --set driver.version=615.<patch> \
  --set toolkit.version=v1.20.1-ubuntu20.04 \
  --set devicePlugin.version=v0.20.1
```

The operator reads both from the `ClusterPolicy` (`spec.toolkit.version` and `spec.devicePlugin.version`) and reports an operand below its minimum on the `Ready` condition as `GPUOperandVersionUnsupported`, without rolling out the node-level daemons. How an operand is recorded decides what can be checked:

| `spec.<operand>.version` | Effective version | Checked against |
|--------------------------|-------------------|-----------------|
| a version tag (`v1.20.1-ubuntu20.04`) | that tag | the operand minimum, directly |
| unset | the GPU Operator's own build default | the GPU Operator floor, **v26.7.1** |
| an opaque tag such as a digest | that image | nothing — not checked |

An unset operand is not unchecked: leaving it out means the GPU Operator supplies its build's default, so its own version is what determines the operand, and the floor is the first release whose defaults satisfy both. That inference is only valid because nothing pinned the operand — which is exactly why it is *not* applied to one that is pinned. A digest is neither: it names an image the GPU Operator version says nothing about, and an unreadable version is not evidence of an unsupported one, so it does not block.

The same floor applies where there is no `ClusterPolicy` at all and the version comes from the OLM `ClusterServiceVersion` instead — an OpenShift install, typically — for the same reason: no operand versions are readable there, so defaults are all that can be inferred. An install that overrode its operands onto supported versions on top of an older operator is rejected on that path, since the overrides are invisible. Use `skipGpuStackVersionChecks` if that is your situation.

If these checks read your install wrongly, `--set skipGpuStackVersionChecks=true` bypasses all of them. It turns verification off rather than making an unsupported stack work: with the gates off, a GPU stack that cannot enforce GPU memory limits rolls the daemons out anyway, `Ready` goes True, and GPU memory goes unfenced with nothing reported. Confirm the operand versions yourself before setting it:

```sh
kubectl get clusterpolicy -o jsonpath='{.items[0].spec.toolkit.version}{"\n"}{.items[0].spec.devicePlugin.version}{"\n"}'
```

If a GPU node ends up on an older driver, the `GpuFractioningConfig` `Ready` condition reports it (`GPUDriverVersionUnsupported`) and the node-level daemons are not rolled out there.

## Install

The operator and its Helm chart are published as OCI artifacts to GitHub Container Registry.

```sh
helm install gpu-fractioning \
  oci://ghcr.io/kai-scheduler/gpu-fractioning \
  --version <VERSION> \
  --namespace gpu-fractioning --create-namespace
```

The chart installs the CRD, the controller, and a default `GpuFractioningConfig` targeting nodes labelled `nvidia.com/gpu.present=true`. Verify the rollout:

```sh
kubectl -n gpu-fractioning get pods
kubectl get gpufractioningconfig default -o yaml   # check .status.conditions → Ready
```

Common chart values (see [`operator/charts/values.yaml`](operator/charts/values.yaml) for the full list):

| Value | Default | Purpose |
|-------|---------|---------|
| `runtimeClassName` | `nvidia` | RuntimeClass for daemon pods that need NVIDIA GPU/NVML access; set `""` to use a node default runtime with NVIDIA GPU/NVML access |
| `supportSmSharing` | `true` | installation-time toggle for the `sm-sharing` compute mode (mpsd's shared MPS server + fractiond's routing to it); disable it to revert to pre-feature behaviour without a code rollback, and the `gpu-compute.mode: sm-sharing` annotation is rejected like any other invalid value. Applies to containers created afterwards — stop sm-sharing workloads and drain the shared server before disabling |
| `skipGpuStackVersionChecks` | `false` | escape hatch that bypasses the NVIDIA GPU stack version gates (GPU Operator, container toolkit, device plugin) when the operator misreads a supported installation and refuses to roll out. It disables verification only — an unsupported GPU stack then rolls out with `Ready` True and nothing reporting the problem. The NVIDIA driver check is separate and is not bypassed |
| `metricsAgent.enabled` | `true` | run the metricsd metrics sidecar |
| `metrics.enabled` / `metrics.port` | `true` / `8080` | controller metrics endpoint (plain HTTP) |
| `prometheus.enabled` | `false` | install a `ServiceMonitor` + `PodMonitor` (also requires `metrics.enabled` and the Prometheus-Operator CRDs) |
| `nodeSelector` | `{}` | scheduling constraint for the **controller** Deployment |
| `global.fipsMode` | `off` | `on` deploys the FIPS 140-3 image variants (`<version>-fips`); `only` also enforces FIPS at runtime, for assessment rather than production. See [FIPS 140-3](docs/fips/README.md) |

> The GPU **nodeSelector** for the DaemonSets is set on the `GpuFractioningConfig` CR (`spec.nodeSelector`), not the chart-level `nodeSelector`.

## Requesting a fractional GPU

A workload opts into GPU fractioning with **per-container** pod annotations that declare its GPU-memory request and limit:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fractional-gpu
  annotations:
    # nvidia.com/container.<container-name>.gpu-memory.<request|limit>
    nvidia.com/container.trainer.gpu-memory.request: 8Gi
    # limit is optional; omit it and it defaults to the request
    nvidia.com/container.trainer.gpu-memory.limit: 16Gi
spec:
  automountServiceAccountToken: false
  containers:
    - name: trainer
      image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["sleep", "infinity"]
      resources:
        requests:
          cpu: 500m
          memory: 1Gi
          ephemeral-storage: 1Gi
        limits:
          memory: 2Gi
```

- The container name in the annotation key selects which container the limits apply to; a pod may carry annotations for several containers.
- **Both `request` and `limit` are optional**, but at least one must be present for the container to be treated as a shared-GPU container. If only one is set, the other defaults to it — so a request-only container is capped at its request rather than left unbounded, and a limit-only container gets its request populated for accounting.
- **limit** is the hard memory cap the driver enforces. **request** is the workload's declared share; the GPU fraction used to normalize SM-utilization metrics is derived from the limit, falling back to the request.
- Values are Kubernetes quantities (`8Gi`, `512Mi`, `1G`, …) and must resolve to at least 1 MiB. fractiond normalizes them to the integer MiB values consumed by NVIDIA memory env vars (`1000Mi` -> `1000`, `1000M` -> `954`). A malformed value fails container creation unless fractiond is running fail-open.
- The GPU **device assignment** (`nvidia.com/container.<name>.gpus.devices`) is set by the scheduler (KAI Scheduler); fractiond injects `NVIDIA_VISIBLE_DEVICES` from it.

### Selecting a compute-sharing mode

GPU **compute** can be shared two ways, selected per-container:

```yaml
annotations:
  # nvidia.com/container.<container-name>.gpu-compute.mode: "time-slicing" | "sm-sharing"
  nvidia.com/container.trainer.gpu-compute.mode: sm-sharing
```

- **`time-slicing`** (the default; same as omitting the annotation) — compute is shared via GPU time-slicing (the driver schedules processes in turns) against mpsd's default per-node MPS server. This is today's behavior.
- **`sm-sharing`** — compute is shared via MPS itself (concurrent SM occupancy): fractiond routes the container to mpsd's shared MPS server instead. `sm-sharing` only makes sense alongside a GPU-memory annotation (above), since it changes how compute is shared, not memory.
- Any other value fails container creation (or falls back to `time-slicing` if fractiond is running fail-open).

## `GpuFractioningConfig` reference

A single cluster-scoped CR configures the whole stack. Field docs are authoritative in [`api/v1alpha1/gpufractioningconfig_types.go`](api/v1alpha1/gpufractioningconfig_types.go).

| Field | Description |
|-------|-------------|
| `spec.nodeSelector` *(required)* | Which nodes the fractiond/mpsd DaemonSets target. **Immutable** — set once at creation. The operator adds `nvidia.com/gpu.deploy.client=true` to the daemons' own nodeSelector on top of this (see below), so it does not belong here. |
| `spec.runtimeClassName` | RuntimeClass for daemon pods that need NVIDIA GPU/NVML access. Defaults to `nvidia`; set `""` to use the node default runtime. |
| `spec.fractioningAgent` | fractiond options (annotation prefix, log level, fail-open, retroactive enforcement). |
| `spec.metricsAgent` | metricsd options (`enabled`, metric-name overrides, extra NVML volumes/mounts). |
| `spec.mpsDaemon` | mpsd supervisor options (e.g. `gracefulStopDelay`). |

Status is surfaced as conditions on the CR:
- **`FractiondReady`** / **`MpsdReady`** — per-daemon rollout health (ready vs desired nodes).
- **`Ready`** — aggregate health of the managed daemons. When not ready, it is refined with NVIDIA GPU Operator dependency failures (e.g. the GPU Operator is missing or below the required version) to explain why.
- **`DriverUpgradeInProgress`** — `True` while a targeted GPU node is undergoing an NVIDIA driver upgrade; the daemons are automatically drained from that node (so MPS shuts down cleanly before the driver unloads) and rescheduled when it completes.

### Coordinating with NVIDIA driver upgrades

mpsd holds the NVIDIA kernel module open for as long as it runs, so it has to be off a node before the driver there can be unloaded. Two independent mechanisms take care of that, and both are automatic:

- The daemons carry **`nvidia.com/gpu.deploy.client=true`** in their nodeSelector. This is the GPU Operator's declaration that a node runs third-party GPU *client* pods — daemons that hold GPU device handles through the nvidia runtime rather than by requesting the `nvidia.com/gpu` resource. The GPU Operator moves the label off `true` while it upgrades a node, which drains the daemons, and it waits for them to terminate before unloading the driver. Without the label the daemons are invisible to the upgrade: its pod-deletion step selects only pods that requested `nvidia.com/gpu`, and both that step and the drain fallback ignore DaemonSets outright.
- A required nodeAffinity on **`nvidia.com/gpu-driver-upgrade-state`** keeps the daemons off any node whose label is set to something other than the terminal `upgrade-done`.

The operator adds both to the DaemonSets it builds; there is nothing to configure. Note that the first one makes the label a **scheduling requirement** — on a cluster whose GPU Operator does not set `nvidia.com/gpu.deploy.client` the daemons will not schedule at all. GPU Operator v26.7.0 and newer set it, which every supported version satisfies.

You can see where a node stands with:

```sh
kubectl get nodes -L nvidia.com/gpu.deploy.client,nvidia.com/gpu-driver-upgrade-state
```

Per-node health is also published as a `gpu-fractioning.nvidia.com/Ready` **node condition** (refined with the CUDA driver version when a node is unhealthy).

## Observability

metricsd exports per-pod GPU metrics (Prometheus, plain HTTP). Built-in metric names (labels: `namespace`, `pod`, `pod_uuid`, `gpu_uuid`, `gpu`):

- `gpu_fractioning_gpu_memory_used_bytes`
- `gpu_fractioning_gpu_sm_utilization_percent`
- `gpu_fractioning_gpu_sm_utilization_percent_normalized` — SM utilization divided by the pod's GPU fraction, capped at 100.

The controller also exports operational metrics (`gpu_fractioning_daemon_ready_nodes`, `gpu_fractioning_daemon_desired_nodes`, `gpu_fractioning_nodes_ready`, `gpu_fractioning_nodes_degraded`) plus the standard controller-runtime `controller_runtime_reconcile_*` series.

With the Prometheus Operator installed, set `prometheus.enabled=true` to have the chart create a `ServiceMonitor` (controller) and a `PodMonitor` (metricsd, one scrape target per GPU node). Both are gated on `metrics.enabled` as well, so setting `metrics.enabled=false` drops the metricsd `PodMonitor` too, not just the controller `ServiceMonitor`. Both scrape over plain HTTP; restrict access with a NetworkPolicy if needed. Metric names are overridable via `metricsAgent.metricNames` / `spec.metricsAgent.metricNames`.

## Repository structure

```
├── api/               # GpuFractioningConfig CRD types (v1alpha1)
├── operator/          # Controller (reconciler) + Helm chart (operator/charts/)
├── fractioning-manager/   # Node-level components
│   ├── mpsd/          #   MPS control daemon supervisor
│   ├── fractiond/      #   NRI plugin (memory-limit injection)
│   ├── metricsd/      #   per-pod GPU metrics sidecar
│   └── common/        #   shared packages
├── pkg/               # Shared libraries
├── docs/              # Topic guides (e.g. FIPS 140-3)
├── hack/              # Build and dev scripts
└── test/              # Cross-component / e2e tests
```

## Development

```sh
make build      # build the operator, mpsd, and fractiond binaries
make test       # run unit tests
make validate   # format, vet, and lint
```

metricsd is a separate Go module (cgo/NVML), so it is not covered by the top-level `make build`; build it with `make -C fractioning-manager/metricsd build`. `make test` and `make docker-build` do cover all four components.

## Roadmap

Planned work lives in the [issue tracker](https://github.com/kai-scheduler/gpu-fractioning/issues);
[ROADMAP.md](ROADMAP.md) explains where to look and how to propose something.

## Community, discussion and support

- **Questions and discussion** — the `#kai-scheduler` channel on the [CNCF Slack](https://slack.cncf.io).
- **Bugs and feature requests** — [GitHub issues](https://github.com/kai-scheduler/gpu-fractioning/issues). [SUPPORT.md](SUPPORT.md) lists what to include.
- **Security vulnerabilities** — never a public issue; use the private channels in [SECURITY.md](SECURITY.md).
- **Who maintains this** — [MAINTAINERS.md](MAINTAINERS.md), governed as described in [GOVERNANCE.md](GOVERNANCE.md).
- **Using this in production?** Add yourself to [ADOPTERS.md](ADOPTERS.md).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) — every commit
must be signed off under the [Developer Certificate of Origin](CLA.md), and all
participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md),
which adopts the [CNCF Code of Conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md).

Security issues go through the private channels in [SECURITY.md](SECURITY.md),
not public issues.

## License

Code is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
Documentation is licensed under the Creative Commons Attribution 4.0 International
License — see [LICENSE-docs](LICENSE-docs).

Third-party components statically linked into the shipped binaries are listed in
[THIRD-PARTY.txt](THIRD-PARTY.txt), which is also included in every image at
`/THIRD-PARTY.txt`.

---

<div align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cncf/artwork/refs/heads/main/other/cncf/horizontal/color-whitetext/cncf-color-whitetext.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/cncf/artwork/refs/heads/main/other/cncf/horizontal/color/cncf-color.svg">
      <img width="300" alt="Cloud Native Computing Foundation logo" src="https://raw.githubusercontent.com/cncf/artwork/refs/heads/main/other/cncf/horizontal/color-whitetext/cncf-color-whitetext.svg">
    </picture>
    <p>gpu-fractioning is part of <a href="https://github.com/kai-scheduler/KAI-Scheduler">KAI Scheduler</a>, a <a href="https://cncf.io">Cloud Native Computing Foundation</a> sandbox project.</p>
</div>

Copyright Contributors to KAI Scheduler, established as KAI Scheduler a Series of LF Projects, LLC.
For website terms of use, trademark policy and other project policies please see [lfprojects.org/policies](https://lfprojects.org/policies/).
