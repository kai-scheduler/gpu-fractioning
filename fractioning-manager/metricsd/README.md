# metricsd

`metricsd` is the per-node GPU metrics exporter. It runs as a sidecar alongside
fractiond, samples NVML for per-process GPU memory and utilization, resolves each
GPU process back to the Kubernetes pod that owns it, and exports the result as
Prometheus metrics.

It is an observer only: it does not register as an NRI plugin and it does not
modify containers. Injecting the GPU memory environment variables and the MPS
pipe mount is fractiond's job — see [`../fractiond`](../fractiond). metricsd learns
which container belongs to which pod by reading the container→pod mapping that
fractiond writes to a shared directory (`--map-dir`).

## Layout

```text
metricsd/
  Dockerfile
  Makefile
  cmd/
  internal/metrics/
  go.mod
  go.sum
```

This directory is an independent Go module.

## Build And Test

From the repository root:

```sh
make -C fractioning-manager/metricsd fmt
make -C fractioning-manager/metricsd test
make -C fractioning-manager/metricsd build
```

metricsd is not covered by the top-level `make build`, because it links the
NVIDIA Go NVML bindings and therefore builds with cgo. The root `make test` and
`make docker-build` do include it.

The component build writes:

```text
fractioning-manager/metricsd/bin/metricsd
```

To write into the root `bin/` directory instead:

```sh
make -C fractioning-manager/metricsd build BIN_DIR=../bin
```

## Image

The default image is `ghcr.io/kai-scheduler/gpu-fractioning/metricsd:dev`.

Build the image from the repository root:

```sh
make -C fractioning-manager/metricsd docker-build TAG=dev
```

For multi-platform builds:

```sh
make -C fractioning-manager/metricsd docker-buildx TAG=dev
```

The image uses `metricsd/Dockerfile`. Because of the cgo/NVML dependency the
runtime image uses the Debian distroless base rather than the static one.

## Runtime Flags

```text
--map-dir           shared directory fractiond writes the container->pod mapping to,
                    default /var/run/gpu-fractioning/map
--log-level         debug, info, warn, or error; default info
--metrics-enabled   run the Prometheus exporter, default true
--metrics-address   exporter listen address, default :2112
--metrics-path      metrics HTTP path, default /metrics
--metrics-interval  NVML sampling interval, default 5s (floored at 2s)
--proc-root         /proc root used to resolve GPU process PIDs to cgroups, default /proc
--sm-util-window    sliding-window averaging for SM utilization; 0 (default) disables
--metric-name-gpu-memory-used-bytes
--metric-name-gpu-sm-utilization-percent
--metric-name-gpu-sm-utilization-percent-normalized
                    override the corresponding metric name
```

Every flag can also be set from the environment: `MAP_DIR`, `LOG_LEVEL`,
`METRICS_ENABLED`, `METRICS_ADDRESS`, `METRICS_PATH`, `METRICS_INTERVAL`,
`PROC_ROOT`, `SM_UTIL_WINDOW`, and the `METRIC_NAME_*` overrides. There is no
configuration file.

In a cluster these are driven from the `GpuFractioningConfig` CR
(`spec.metricsAgent`) rather than set by hand.

## GPU Metrics

When `--metrics-enabled` is set, metricsd starts a Prometheus exporter on
`--metrics-address` and serves `--metrics-path` over plain HTTP. Every
`--metrics-interval` it queries NVML process metrics, resolves each GPU process
PID through `<proc-root>/<pid>/cgroup`, and matches it against the container
mapping to attach pod identity.

Exported metrics (labels `namespace`, `pod`, `pod_uuid`, `gpu_uuid`, `gpu`):

- `gpu_fractioning_gpu_memory_used_bytes`
- `gpu_fractioning_gpu_sm_utilization_percent`
- `gpu_fractioning_gpu_sm_utilization_percent_normalized` — SM utilization divided by
  the pod's GPU fraction and capped at 100. The fraction is the pod's requested
  GPU memory ÷ the device's total memory (from NVML), where the requested memory
  comes from the `nvidia.com/container.<name>.gpu-memory.{limit,request}`
  annotation that fractiond records (limit, else request); a pod using as much of
  the GPU as it requested reports 100. When the request is unknown or the
  device's total memory is unavailable it falls back to a fraction of 1, so the
  value equals the raw SM utilization.

NVML returns process-level data. metricsd joins `nvmlDeviceGetProcessUtilization`
with the compute and graphics running-process queries, then aggregates matching
processes by pod and GPU. For active annotated pods attached to a GPU, the
exporter keeps publishing zero values when NVML has no process reading; deleted
pods are pruned once their last tracked container is removed.
