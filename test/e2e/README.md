# gpu-fractioning e2e tests

Stage 1: metrics-only e2e tests for the metricsd metrics endpoint, running
against a real k3d cluster with
[fake-gpu-operator](https://github.com/run-ai/fake-gpu-operator) installed. The
gpu-fractioning stack is built and installed the way it ships — [Skaffold](https://skaffold.dev)
(`skaffold.yaml`) builds the four component images and `helm install`s the
operator Helm chart (`operator/charts`) — and the operator creates the
`fractiond` DaemonSet, which hosts the **metricsd sidecar** under test. (metricsd
is no longer a standalone DaemonSet; it runs as a container in the fractiond
pods.)

## Quick start

```sh
export E2E_FAKE_GPU_OPERATOR_VERSION=0.2.0   # see releases link below
make e2e
```

This creates a k3d cluster, installs fake-gpu-operator, then runs
`skaffold run -p e2e` to build the four images (operator, fractiond, metricsd,
mpsd), load them into the cluster, and install the operator chart pointing at
them — and finally runs the test suite. Tear the cluster down afterwards with:

```sh
make e2e-cluster-down
```

## How it's split

Cluster lifecycle, deployment, and the Go test suite are deliberately separate
concerns — the suite only connects and asserts, it deploys nothing:

| Step | Owner | What |
|---|---|---|
| Create k3d cluster (N worker nodes, NRI enabled) | `test/e2e/hack/create-cluster.py` | `k3d cluster create` with retry-on-failure |
| Install fake-gpu-operator | `test/e2e/hack/create-cluster.py` | `helm upgrade -i` with per-pool GPU count/product/memory |
| Build + load images + install operator | `make e2e-deploy` (`skaffold run -p e2e`) | Skaffold builds the 4 images, loads them into k3d, and helm-installs `operator/charts`; the operator then reconciles the CR and creates the fractiond (+ metricsd sidecar) and mpsd DaemonSets |
| Verify GPU nodes | `test/e2e/suite` (`nodes.VerifyGPUNodes`) | precondition check, no mutation |
| Run test assertions | `test/e2e/tests` | scrape the metricsd sidecar `/metrics`, assert |

> **RuntimeClass on k3d:** the e2e chart values clear `runtimeClassName` so daemon
> pods use k3d's default runtime. The e2e images provide the fake NVIDIA/NVML
> behavior needed by the tests.

## Build + deploy (`skaffold.yaml`)

Skaffold builds all four images and deploys them through the operator Helm chart
in one flow — the "4 images, use Skaffold + Helm" direction:

- **Default profile** builds prod-style images.
- **`e2e` profile** uses fractiond's `e2e` Docker target and metricsd's
  `Dockerfile.e2e`, both with nvml-mock baked in for fake-GPU clusters. It also
  passes metricsd's `TARGETARCH` (from `E2E_ARCH`, exported by the Makefile)
  because that Dockerfile hardcodes `ARG TARGETARCH=amd64`.

Each built image is mapped into the chart's `images.<name>.{repository,tag}`
values via `setValueTemplates`. Skaffold auto-loads images into a k3d/kind
cluster when the current kube-context is one, so `skaffold`/`make e2e-deploy`
must run with the e2e cluster as the current context.

```sh
skaffold run   -p e2e     # build + load + helm install (what make e2e-deploy runs)
skaffold dev   -p e2e     # same, rebuild/redeploy on changes
skaffold delete -p e2e    # uninstall (what make e2e-undeploy runs)
```

> Skaffold would otherwise log `image [fractiond|mpsd|metricsd] is not used` during
> deploy: those images are referenced indirectly (the chart passes their repo/tag
> to the operator, which then creates the pods), so Skaffold's rendered-manifest
> scan doesn't see them. They're false positives (the operator wires them in at
> reconcile time), so `make e2e-deploy` runs Skaffold with `--verbosity=error` to
> suppress them.

a standalone script (`create-cluster.py`) provisions the cluster and
cluster-level dependencies, while the Go suite only connects, verifies
preconditions, and deploys/tests the component actually under test.
`create-cluster.py` uses pydantic-settings for `E2E_*`-prefixed config, typer for
the CLI, and retry-on-create-failure, scoped down to what this project needs: a
k3d cluster with fake-gpu-operator
and nothing else, so the dependency list is just `typer`, `pydantic-settings`,
`sh` — no docker SDK (nothing here builds or pre-pulls images) and no `rich`
(plain stdout is enough for CI logs).

## Cluster configuration (`test/e2e/hack/create-cluster.py`)

All configurable via `E2E_*` environment variables:

| Variable                        | Default                    | Purpose                                                                                                     |
|---------------------------------|----------------------------|-------------------------------------------------------------------------------------------------------------|
| `E2E_CLUSTER_NAME`              | `gpu-fractioning-e2e`          | k3d cluster name                                                                                            |
| `E2E_GPU_WORKER_NODES`          | `2`                        | number of agent nodes, each labeled into the fake-GPU node pool                                             |
| `E2E_K3S_IMAGE`                 | `rancher/k3s:v1.31.5-k3s1` | k3s node image                                                                                              |
| `E2E_GPU_NODE_POOL`             | `default`                  | fake-gpu-operator node pool name                                                                            |
| `E2E_GPUS_PER_NODE`             | `2`                        | GPUs advertised per node in that pool                                                                       |
| `E2E_GPU_PRODUCT`               | `NVIDIA A100-SXM4-40GB`    | simulated GPU product                                                                                       |
| `E2E_GPU_MEMORY_MIB`            | `40960`                    | simulated GPU memory (MiB)                                                                                  |
| `E2E_FAKE_GPU_OPERATOR_VERSION` | *(required)*               | fake-gpu-operator Helm chart version — see [releases](https://github.com/run-ai/fake-gpu-operator/releases) |
| `E2E_MAX_RETRIES`               | `3`                        | cluster-creation retry attempts                                                                             |
| `E2E_KUBECONFIG`                | `~/.kube/<cluster>.yaml`   | where the cluster's kubeconfig is written (see note below)                                                  |

> **Kubeconfig handling.** The NRI `config.toml.tmpl` volume this script mounts
> (to let fractiond's NRI plugin run) breaks `k3d kubeconfig get/merge` — so the
> script does **not** touch `~/.kube/config`. Instead it extracts the kubeconfig
> from the server container via `docker`, rewrites the API URL to the published
> port and the context name to `k3d-<cluster>` (so Skaffold auto-loads images
> into it), and writes it to `E2E_KUBECONFIG`. Every downstream step
> (`e2e-deploy`, the Go suite) reads that file, so the whole flow is independent
> of whatever your current kube-context happens to be.

```sh
pip install -r test/e2e/hack/requirements.txt

E2E_GPU_WORKER_NODES=4 E2E_FAKE_GPU_OPERATOR_VERSION=0.2.0 test/e2e/hack/create-cluster.py
test/e2e/hack/create-cluster.py --delete
test/e2e/hack/create-cluster.py --skip-fake-gpu-operator   # cluster only, e.g. for iterating on the script itself
```

Requires `k3d`, `kubectl`, `docker`, `helm` (unless `--skip-fake-gpu-operator`),
and Python 3.9+ on PATH.

The script installs `nvml-mock` by default. The fake-mps image bakes in the mock
library so mpsd can read a fake driver version before starting, and the metrics
suite uses the test-controlled ConfigMap to feed metricsd deterministic GPU
process state. Pass `--skip-gpu-mock` only when iterating on cluster setup pieces
that do not deploy the gpu-fractioning stack.

## Makefile targets

| Target                        | What                                                                                                                                                                                                       |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `make e2e`                    | cluster up → skaffold build+load+deploy → run tests                                                                                                                                                        |
| `make e2e-cluster-up`         | create the k3d cluster + install fake-gpu-operator                                                                                                                                                         |
| `make e2e-cluster-down`       | delete the k3d cluster                                                                                                                                                                                     |
| `make e2e-cluster-deps`       | `pip install -r test/e2e/hack/requirements.txt`                                                                                                                                                            |
| `make e2e-deploy`             | `skaffold run -p e2e` (build 4 images + load + helm install), then wait for the fractiond DaemonSet rollout                                                                                                 |
| `make e2e-undeploy`           | `skaffold delete -p e2e` (helm uninstall)                                                                                                                                                                  |
| `make e2e-kubeconfig-merge`   | opt-in: add the cluster to `~/.kube/config` as context `k3d-<cluster>` (current-context preserved, prior file backed up), so `kubectl config use-context k3d-<cluster>` works without setting `KUBECONFIG` |
| `make e2e-kubeconfig-unmerge` | remove that context from `~/.kube/config` (run automatically by `e2e-cluster-down`)                                                                                                                        |
| `make test-e2e`               | run all Go suites (cluster deployed already)                                                                                                                                                               |
| `make test-e2e-metrics`       | run only the metrics suite (`./tests/metrics/...`)                                                                                                                                                         |
| `make test-e2e-operator`      | run only the operator/controller suite (`./tests/operator/...`)                                                                                                                                            |
| `make test-e2e-fractiond`      | run only the fractiond NRI-injection suite (`./tests/fractiond/...`)                                                                                                                                         |
| `make run-e2e`                | alias for `make test-e2e` — run against any cluster (`E2E_KUBECONFIG=...`), regardless of how it was created/deployed                                                                                      |

`E2E_CLUSTER_NAME`, `E2E_GPU_WORKER_NODES`, `E2E_OPERATOR_NAMESPACE`, `E2E_ARCH`,
`E2E_FAKE_GPU_OPERATOR_VERSION` are overridable `make` variables. `PYTHON`
overrides the Python interpreter (default `python3`). Image builds/tags are
managed by `skaffold.yaml`, not `make` variables.

## Go test suite configuration

| Variable                        | Default                       | Purpose                                                                                      |
|---------------------------------|-------------------------------|----------------------------------------------------------------------------------------------|
| `E2E_KUBECONFIG` / `KUBECONFIG` | `~/.kube/<cluster>.yaml`      | cluster to connect to (written by `create-cluster.py`; the make targets set it for you)      |
| `E2E_OPERATOR_NAMESPACE`        | `gpu-fractioning`        | namespace the operator is installed into (and where it creates the fractiond/mpsd DaemonSets) |
| `E2E_GPU_NODE_SELECTOR`         | `nvidia.com/gpu.present=true` | label selector used to verify GPU nodes                                                      |
| `E2E_GPU_NODE_COUNT`            | `0` (unchecked)               | exact GPU node count to assert, if > 0 — set to `E2E_GPU_WORKER_NODES`                       |

## What's covered

- `TestE2E_GPUFractioningPluginMetricsEndpointHealthy` — smoke test: the metricsd
  sidecar's `/metrics` endpoint (port 2112) in the operator-created fractiond
  pods is reachable and returns valid Prometheus output.

## CI

`.github/workflows/e2e.yaml` runs the pipeline on PRs touching
`fractioning-manager/**` or `test/e2e/**`: it installs `k3d` (official install
script), Helm, and Skaffold via the `e2e-setup` composite action, then runs
`make e2e-deploy` (`skaffold run -p e2e` builds+loads+helm-installs the operator,
as grove deployed its operator in-workflow) → `make test-e2e-metrics`, and tears
the cluster down in an `if: always()` step. On failure it dumps node/pod state,
the `GpuFractioningConfig` status, and fractiond/operator/status-updater logs before
tearing down.

## Package layout

Each Go package is single-purpose, so dependencies only point one way (the
`metrics` leaf has no dependency on the rest of the framework, so future
additions can depend on it without an import cycle). The k8s-touching helpers
live under `k8s/`; deployment is not a Go package — it's `helm` (see
`e2e.mk`'s `e2e-deploy`):

- `config/` — env-driven `Config` (no deps)
- `k8s/cluster/` — kubeconfig → `Client` (controller-runtime client + rest.Config); depends on `config`
- `k8s/nodes/` — GPU node count/label precondition check; depends on `cluster`
- `k8s/pods/` — list helpers for pods running in the cluster; depends on `cluster`
- `k8s/portforward/` — SPDY port-forward to a pod (like `kubectl port-forward`); depends on `cluster`
- `metrics/` — scrape + parse Prometheus text format (no deps)
- `suite/` — ties `config`+`cluster`+`nodes` together; used by `tests/*/main_test.go`
- `harness/` — shared, importable (`//go:build e2e`) test support for the operator and fractiond suites: the FX-STEADY fixture machinery, workload/exec/pod/node/daemonset helpers, diagnostics, and the cross-suite constants. Because Go can't import another package's `_test.go` files, anything shared across suites lives here; each suite's `aliases_test.go` re-exports the bits it uses under short unqualified names.
- `tests/metrics/` — the metrics suite: scrapes the metricsd sidecar `/metrics`
- `tests/operator/` — the operator/controller suite: CR/CEL validation, DaemonSet deployment shape, CR status + node conditions, fault injection/recovery, config propagation, teardown
- `tests/fractiond/` — the fractiond data-plane suite: fractiond's NRI injection (GPU-memory env + MPS pipe mount on annotated containers, fail-closed/fail-open)
- `hack/` — `create-cluster.py` + `requirements.txt`, the k3d + fake-gpu-operator provisioning script (Python, not Go — see above for why)

## Upgrade check

`make test-e2e-upgrade` installs the last published chart (`E2E_PREVIOUS_CHART_VERSION`,
default `0.1.8`), upgrades it to the checkout through `e2e-deploy`, and checks that the
`default` `GpuFractioningConfig` and the DaemonSets it owns kept their UIDs and that the
CR is now owned by the release; it then uninstalls and checks both are gone. It needs a
fresh e2e cluster with no release installed, and is run by the `upgrade` job in
`.github/workflows/e2e.yaml`.
