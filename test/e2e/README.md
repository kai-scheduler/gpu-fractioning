# End-to-end tests with GPU Operator and Mokka

This draft replaces FGO with NVIDIA GPU Operator v26.7.2 and Mokka on a disposable kind cluster. It changes test infrastructure and helpers only. Production code, chart defaults and public APIs are unchanged.

## What is installed

- kind v1.35.0 nodes with containerd NRI enabled: two GPU workers, one CPU worker and a control plane.
- Mokka on the GPU workers only, with two A100 GPUs per worker, kernel simulation and NRI integration enabled.
- The real NVIDIA driver, toolkit, device plugin, GPU Feature Discovery and DCGM exporter through GPU Operator. CUDA workload validation is disabled; the mock cannot execute CUDA. MIG is disabled, as in the previous e2e setup.
- A local image registry for Skaffold's four component images.
- The gpu-fractioning chart with the real `nvidia` RuntimeClass, the real operator and daemons, the existing fake MPS control program, and controlled NVML process data for metricsd.

The Mokka build is pinned to `fe859af25b05f1ea4e4e91e56c89491b8e6dc7b4` on the public `JRosenboimNVIDIA/mokka` fork. This is the tested PR 1109 integration at c7605a7 plus the memory-limit acknowledgement from PR 1130. The temporary fork dependency is explicit until GPU Operator support is released upstream. `hack/install-mokka-gpu-operator.sh` builds the unmodified upstream Dockerfile and loads its image into kind.

FGO, its status updater, and the old standalone NVML DaemonSet are not installed. The latter would overwrite the driver files now owned by Mokka and GPU Operator. Host PID discovery uses Mokka's privileged `node-agent` container instead.

Fake MPS remains necessary: it exercises the real supervisor and socket readiness without CUDA. The metrics image and its ConfigMap remain test-controlled so tests can supply known host PIDs, GPU memory and utilization. This is not a test of hardware memory enforcement. The real NVIDIA memory-limit hook is enabled; Mokka acknowledges its NVML call without enforcing a limit.

The fake MPS image selects its control program through `MPS_CONTROL_BINARY` at `/usr/local/libexec/gpu-fractioning-test/nvidia-cuda-mps-control`. The NVIDIA runtime can mount the real control program at `/usr/bin/nvidia-cuda-mps-control`; the separate test path prevents that mount from replacing the fake program.

Metricsd reads its test ConfigMap from `/etc/gpu-fractioning-test/config.yaml`. This keeps test process data separate from Mokka's CDI mount at `/etc/nvml-mock`.
The test image sets this path at process startup because CDI can replace image environment variables. It also clears Mokka's node override path so node activity cannot change the controlled metrics input.

## Run

Requires Docker, kind, kubectl, Helm 3, Skaffold, Python 3 and the Go toolchain.

```sh
make e2e
make e2e-cluster-down
```

For separate setup and execution:

```sh
make e2e-cluster-up
make e2e-deploy
make test-e2e-operator
make test-e2e-fractiond
make test-e2e-metrics
```

The suites cover:

| Suite | Coverage |
| --- | --- |
| operator | CR validation, DaemonSet shape, status and node conditions, configuration propagation, faults, recovery and teardown |
| fractiond | Memory env injection, MPS mounts, compute-mode routing, valid runtime GPU assignments, fail-open/fail-closed behavior |
| metrics | Per-pod memory/utilization attribution, multi-GPU separation, aggregation, cleanup and recovery using controlled NVML process data |

The runtime assignment case queries Mokka's inventory for valid GPU IDs. It checks the devices visible with `nvidia-smi`; a CDI runtime may consume `NVIDIA_VISIBLE_DEVICES` and replace it with `void`. Idle workload pods use Ubuntu 24.04 to provide glibc for the injected NVIDIA utilities.

The tests do not install KAI Scheduler. GPU assignment and metrics inputs are provided directly by the test helpers. There are no NUMA or KWOK suites in this repository, and neither is added here.

## Configuration

| Variable | Default |
| --- | --- |
| `E2E_CLUSTER_NAME` | `gpu-fractioning-e2e` |
| `E2E_KIND_IMAGE` | `kindest/node:v1.35.0` |
| `E2E_GPU_WORKER_NODES` | `2` |
| `E2E_NON_GPU_WORKER_NODES` | `1` |
| `E2E_GPUS_PER_NODE` | `2` |
| `E2E_GPU_OPERATOR_VERSION` | `v26.7.2` |
| `E2E_REGISTRY_PORT` | `5001` |
| `E2E_OPERATOR_NAMESPACE` | `gpu-fractioning` |
| `E2E_KUBECONFIG` | `~/.kube/<cluster-name>.yaml` |

Mokka's source and image can be changed through `MOKKA_REPOSITORY`, `MOKKA_REF`, `MOKKA_IMAGE_REPOSITORY` and `MOKKA_IMAGE_TAG`.

Provisioning deletes and recreates only the explicitly named disposable kind cluster and its registry. A dedicated kubeconfig is used throughout; setup does not alter the user's default context. The opt-in `e2e-kubeconfig-merge` target adds context `kind-<cluster-name>` to the default kubeconfig.

CI has separate operator, fractiond and metrics jobs, each using Ubuntu 24.04 to match the pinned driver image and the same kind setup. Full e2e validation is required before this draft becomes a supported replacement.
