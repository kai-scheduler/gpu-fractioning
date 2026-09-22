#!/usr/bin/env python3
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""create-cluster.py - k3d cluster + fake-gpu-operator for gpu-fractioning e2e tests.

Uses an env-driven pydantic-settings config, a typer CLI, and
retry-on-create-failure, scoped to what this project needs: a k3d cluster with a
single fake-GPU node pool and fake-gpu-operator installed. It also stands up a
k3d-managed local image registry (see create_registry) that Skaffold pushes the
gpu-fractioning component images to, so nodes pull them (pullPolicy IfNotPresent)
and an image evicted from a node's containerd under disk pressure is re-pulled
instead of wedging at ErrImageNeverPull. No application components are deployed
here.

Dependencies: see requirements.txt (typer, pydantic-settings, sh — no docker
SDK since this script never builds or pre-pulls images, no rich since plain
stdout is enough for CI logs).

Environment variables (E2E_ prefix, see ClusterConfig):
    E2E_FAKE_GPU_OPERATOR_VERSION  (required) e.g. "0.2.0"
    E2E_CLUSTER_NAME               (default: gpu-fractioning-e2e)
    E2E_GPU_WORKER_NODES           (default: 2)   # GPU worker nodes
    E2E_NON_GPU_WORKER_NODES       (default: 1)   # plain (no-GPU) worker nodes
    E2E_K3S_IMAGE                  (default: rancher/k3s:v1.31.5-k3s1)
    E2E_GPU_NODE_POOL              (default: default)
    E2E_GPUS_PER_NODE              (default: 2)
    E2E_GPU_PRODUCT                (default: NVIDIA A100-SXM4-40GB)
    E2E_GPU_MEMORY_MIB             (default: 40960)
    E2E_MAX_RETRIES                (default: 3)
    E2E_KUBECONFIG                 (default: ~/.kube/<cluster_name>.yaml)
    E2E_REGISTRY_PORT              (default: 5001)  # k3d local image registry

The cluster's kubeconfig is written to E2E_KUBECONFIG (not ~/.kube/config) via
docker, because the NRI config.toml.tmpl volume this script mounts breaks k3d's
own `k3d kubeconfig` retrieval. Every downstream e2e step reads that file.

Usage:
    E2E_FAKE_GPU_OPERATOR_VERSION=0.2.0 ./create-cluster.py
    E2E_FAKE_GPU_OPERATOR_VERSION=0.2.0 E2E_GPU_WORKER_NODES=4 ./create-cluster.py
    ./create-cluster.py --skip-fake-gpu-operator --skip-gpu-mock
    ./create-cluster.py --delete

Requires: k3d, kubectl, docker, helm (unless --skip-fake-gpu-operator).
"""

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import sh
import typer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

app = typer.Typer(help="k3d cluster setup for gpu-fractioning e2e tests")

FAKE_GPU_OPERATOR_CHART = "oci://ghcr.io/run-ai/fake-gpu-operator/fake-gpu-operator"

# nvml-mock: NVIDIA's mock libnvidia-ml.so DaemonSet (supplements fake-gpu-operator).
# Applied as a static manifest — its setup.sh self-labels nodes
# nvidia.com/gpu.present=true (the fractiond DaemonSet's nodeSelector) and installs a
# real libnvidia-ml.so into /var/lib/nvml-mock/driver, so metricsd's NVML calls
# resolve against the mock instead of falling back to NoopCollector. No Helm
# release and no device plugin are needed for the metrics e2e suite: attribution
# keys off the fractional annotation + cgroup + NVML UUID, not a scheduled
# nvidia.com/gpu resource.
NVML_MOCK_MANIFEST = "fractioning-manager/metricsd/deploy/fake-gpu-cluster/nvml-mock.yaml"

# fractiond (the gpu-fractioning's NRI DaemonSet, created by the Helm chart)
# mounts /var/run/nri as a hostPath of type "Directory", which requires the path
# to already exist on the node. Stock k3s images ship with containerd's NRI
# plugin disabled, so that directory/socket is never created and the
# fractiond pods hang forever in ContainerCreating. Override each node's
# containerd config to enable NRI so containerd creates the socket (and its
# parent directory) on startup.
CONTAINERD_NRI_CONFIG_TEMPLATE = """{{ template "base" . }}

[plugins."io.containerd.nri.v1.nri"]
  disable = false
  socket_path = "/var/run/nri/nri.sock"
"""


def write_containerd_nri_template() -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".toml.tmpl", delete=False) as f:
        f.write(CONTAINERD_NRI_CONFIG_TEMPLATE)
        return f.name


class ClusterConfig(BaseSettings):
    """Configuration auto-loaded from E2E_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="E2E_", extra="ignore")

    cluster_name: str = "gpu-fractioning-e2e"
    # GPU worker nodes (carry the fake-gpu-operator node-pool label; get
    # nvidia.com/gpu.present=true). Non-GPU workers below are plain agents with
    # no GPU label, so the cluster mirrors a real mixed GPU/CPU topology.
    gpu_worker_nodes: int = Field(default=2, ge=1, le=50)
    non_gpu_worker_nodes: int = Field(default=1, ge=0, le=50)
    k3s_image: str = "rancher/k3s:v1.31.5-k3s1"
    gpu_node_pool: str = "default"
    gpus_per_node: int = Field(default=2, ge=1, le=16)
    gpu_product: str = "NVIDIA A100-SXM4-40GB"
    gpu_memory_mib: int = Field(default=40960, ge=1)
    fake_gpu_operator_version: str = ""
    cluster_timeout: str = "120s"
    # Readiness wait for the fake-GPU cluster components (fake-gpu-operator's
    # status-updater pod + device-plugin DaemonSet, and the nvml-mock DaemonSet).
    # A fresh cluster has an empty containerd image cache, so the very first pull
    # of these images dominates the wait; 120s/180s regularly time out on cold
    # caches (local first run and CI runners alike) even though the pods come up
    # fine given a little more time. Overridable via E2E_FAKE_GPU_OPERATOR_TIMEOUT.
    fake_gpu_operator_timeout: str = "300s"
    max_retries: int = Field(default=3, ge=1, le=10)
    # Where to write the cluster's kubeconfig (E2E_KUBECONFIG). Empty -> a
    # default derived from cluster_name (see main). This file — not ~/.kube/config
    # — is what every downstream e2e step uses.
    kubeconfig: str = ""
    # Host-published port for the k3d-managed local image registry (see
    # create_registry). Skaffold pushes to localhost:PORT and image refs use it;
    # the container itself always listens on 5000 internally (REGISTRY_INTERNAL_PORT),
    # which is what the in-cluster mirror endpoint targets. Default 5001 dodges
    # macOS's AirPlay Receiver on 5000.
    registry_port: int = Field(default=5001, ge=1, le=65535)


def log(msg: str) -> None:
    print(f"[create-cluster] {msg}", flush=True)


def require_command(cmd: str) -> None:
    if shutil.which(cmd) is None:
        log(f"required command '{cmd}' not found. Please install it first.")
        raise typer.Exit(1)


def delete_cluster(config: ClusterConfig) -> None:
    log(f"Deleting k3d cluster '{config.cluster_name}'...")
    sh.k3d("cluster", "delete", config.cluster_name, _ok_code=[0, 1])


# Port the k3d registry container listens on inside the cluster network. k3d's
# --port flag only remaps the host-published port; the container itself always
# serves on 5000, so the in-cluster mirror endpoint must target this, not the
# host port.
REGISTRY_INTERNAL_PORT = 5000


def registry_name(config: ClusterConfig) -> tuple[str, str]:
    """(base, full) k3d registry names. k3d prepends "k3d-" to everything it
    creates, so we pass `base` to `k3d registry create`/`delete` but reference
    `full` in --registry-use, the mirror endpoint and node-side pulls."""
    base = f"{config.cluster_name}-registry"
    return base, f"k3d-{base}"


def write_registries_yaml(config: ClusterConfig) -> str:
    """Write a containerd registries.yaml that mirrors localhost:<host_port> to
    the in-cluster registry container. Skaffold pushes images tagged
    localhost:<host_port>/<img> (which the host reaches via the published port),
    and this mirror teaches every node to pull that same ref from the registry
    over the k3d Docker network — where it's reachable as
    k3d-<cluster>-registry:<internal_port> (5000, NOT the host port). One image
    ref works on both sides, so no /etc/hosts entry or *.localhost DNS is needed,
    which keeps it portable across macOS and CI runners."""
    _, full = registry_name(config)
    content = (
        "mirrors:\n"
        f'  "localhost:{config.registry_port}":\n'
        "    endpoint:\n"
        f"      - http://{full}:{REGISTRY_INTERNAL_PORT}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(content)
        return f.name


def create_registry(config: ClusterConfig) -> None:
    """Create the k3d-managed local registry the e2e stack pushes to. Recreated
    from scratch (delete-then-create) so a stale registry from a prior run can't
    serve outdated image layers. --registry-use (in create_cluster) connects it
    to the cluster network and generates the node registries.yaml."""
    base, full = registry_name(config)
    log(f"Creating k3d registry '{full}' on port {config.registry_port} (delete-then-create)...")
    sh.k3d("registry", "delete", full, _ok_code=[0, 1])
    sh.k3d("registry", "create", base, "--port", str(config.registry_port))


def delete_registry(config: ClusterConfig) -> None:
    _, full = registry_name(config)
    log(f"Deleting k3d registry '{full}'...")
    sh.k3d("registry", "delete", full, _ok_code=[0, 1])


def create_cluster(config: ClusterConfig) -> bool:
    log("Cluster configuration:")
    for key, value in config.model_dump().items():
        log(f"  {key:26s}: {value}")

    nri_template_path = write_containerd_nri_template()
    registries_yaml_path = write_registries_yaml(config)
    _, registry_full = registry_name(config)

    # k3d indexes agents 0..(total-1). GPU agents come first and carry the GPU
    # node labels below; the remaining agents stay unlabeled (non-GPU workers).
    #
    #   nvidia.com/gpu.present      what everything downstream selects on: the
    #                               nvml-mock DaemonSet's nodeSelector (which must
    #                               match before it can schedule and patch the
    #                               node) and the operator's default-CR nodeSelector.
    #   nvidia.com/gpu.deploy.client  the gpu-operator's third-party GPU client
    #                               declaration, which the operator adds to the
    #                               fractiond/mpsd nodeSelector so a driver upgrade
    #                               evicts them. The fake gpu-operator does not
    #                               maintain it, so set it here — without it the
    #                               daemons have nowhere to schedule and the suite
    #                               fails in preflight with "no plugin pods found".
    total_agents = config.gpu_worker_nodes + config.non_gpu_worker_nodes
    gpu_node_filter = ";".join(f"agent:{i}" for i in range(config.gpu_worker_nodes))
    gpu_node_labels = [
        f"nvidia.com/gpu.present=true@{gpu_node_filter}",
        f"nvidia.com/gpu.deploy.client=true@{gpu_node_filter}",
    ]

    try:
        for attempt in range(1, config.max_retries + 1):
            log(f"Cluster creation attempt {attempt}/{config.max_retries}...")

            sh.k3d("cluster", "delete", config.cluster_name, _ok_code=[0, 1])

            try:
                sh.k3d(
                    "cluster", "create", config.cluster_name,
                    "--servers", "1",
                    "--agents", str(total_agents),
                    "--image", config.k3s_image,
                    *[arg for label in gpu_node_labels for arg in ("--k3s-node-label", label)],
                    "--volume", f"{nri_template_path}:/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl@server:0;agent:*",
                    # Connect the local registry to the cluster network (so its
                    # name resolves inside nodes) and add the localhost mirror so
                    # nodes can pull images Skaffold pushed there. No port on
                    # --registry-use: k3d looks the registry up by name, and the
                    # actual pull path is our mirror (see write_registries_yaml).
                    "--registry-use", registry_full,
                    "--registry-config", registries_yaml_path,
                    "--timeout", config.cluster_timeout,
                    "--wait",
                    # Don't touch the user's default kubeconfig. The NRI config.toml.tmpl
                    # volume above breaks `k3d kubeconfig get/merge` (the tools node fails
                    # to copy over the mounted file), so we extract the kubeconfig
                    # ourselves via docker in write_kubeconfig() instead.
                    "--kubeconfig-update-default=false",
                    "--kubeconfig-switch-context=false",
                )
                log(f"Cluster created successfully on attempt {attempt}.")
                return True
            except sh.ErrorReturnCode as e:
                log(f"cluster creation failed: {e}")
                if attempt < config.max_retries:
                    log("Retrying in 10s...")
                    time.sleep(10)

        log(f"Cluster creation failed after {config.max_retries} attempts.")
        return False
    finally:
        Path(nri_template_path).unlink(missing_ok=True)
        Path(registries_yaml_path).unlink(missing_ok=True)


def write_kubeconfig(config: ClusterConfig) -> str:
    """Extract the cluster's kubeconfig via docker and write it to config.kubeconfig.

    We can't use `k3d kubeconfig get`: the NRI config.toml.tmpl volume mount makes
    k3d's tools node fail to copy the kubeconfig out. Instead we read k3s's own
    kubeconfig from the server container and rewrite:
      - the API server URL to the host-published serverlb port, and
      - the "default" context/cluster/user names to `k3d-<cluster>`, so Skaffold
        recognises it as a k3d cluster and auto-loads built images into it.

    The path is also set as KUBECONFIG for the rest of this process (so the
    fake-gpu-operator install runs against it).
    """
    server = f"k3d-{config.cluster_name}-server-0"
    serverlb = f"k3d-{config.cluster_name}-serverlb"
    ctx = f"k3d-{config.cluster_name}"

    port = str(sh.docker("port", serverlb, "6443/tcp")).strip().splitlines()[0].rsplit(":", 1)[-1]
    raw = str(sh.docker("exec", server, "cat", "/etc/rancher/k3s/k3s.yaml"))

    out = []
    for line in raw.splitlines():
        line = line.replace("https://127.0.0.1:6443", f"https://127.0.0.1:{port}")
        line = line.replace("https://0.0.0.0:6443", f"https://127.0.0.1:{port}")
        stripped = line.strip()
        # k3s.yaml names the context/cluster/user all "default"; rename to k3d-<cluster>.
        if stripped in ("name: default", "cluster: default", "user: default", "- name: default"):
            line = line.replace("default", ctx)
        elif stripped == "current-context: default":
            line = f"current-context: {ctx}"
        out.append(line)

    path = Path(config.kubeconfig).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    os.environ["KUBECONFIG"] = str(path)
    log(f"Wrote kubeconfig to {path} (context {ctx})")
    return str(path)


def wait_for_nodes(config: ClusterConfig) -> None:
    log("Waiting for all nodes to be ready...")
    sh.kubectl("wait", "--for=condition=Ready", "nodes", "--all", "--timeout=120s")
    log("All nodes are ready.")


_IMAGE_LINE_RE = re.compile(r"""^\s*image:\s*["']?([^"'\s]+)["']?\s*$""")


def image_refs(manifest_text: str) -> list[str]:
    """Extract the unique container image refs from rendered k8s manifests
    (Helm template output or a static manifest)."""
    return sorted({m.group(1) for m in map(_IMAGE_LINE_RE.match, manifest_text.splitlines()) if m})


def prewarm_images(config: ClusterConfig, images: list[str]) -> None:
    """Make each image available in the k3d cluster's containerd *before* the
    workloads that need it are created, so pod startup never blocks on a pull.

    The k3d nodes pull directly from the registry, and on a throttled network
    ghcr.io pulls of even small (~60-100MB) images can take 10+ minutes and blow
    the readiness waits. Instead we pull once into the host Docker (cached across
    FRESH recreates, unlike the per-node containerd) and `k3d image import` it —
    which is a fast local transfer. A warm Docker cache makes recreates instant;
    CI (fast ghcr) is unaffected since a cache miss just falls back to a normal
    fast pull here.
    """
    if not images:
        return
    log(f"Pre-warming {len(images)} image(s) into cluster '{config.cluster_name}':")
    for img in images:
        log(f"  - {img}")
    for img in images:
        try:
            # Captured (not printed); success => already in the local Docker cache.
            sh.docker("image", "inspect", img)
            present = True
        except sh.ErrorReturnCode:
            present = False
        if not present:
            log(f"docker pull {img} (one-time; may be slow on a throttled network)...")
            sh.docker("pull", img, _fg=True)
        log(f"k3d image import {img} -> {config.cluster_name}...")
        try:
            sh.k3d("image", "import", img, "-c", config.cluster_name, _fg=True)
        except sh.ErrorReturnCode:
            # Non-fatal by design: this whole function is a latency optimization,
            # so a failed import must not fail cluster creation. A digest-pinned
            # ref has no repo tag once saved, which k3d may refuse to import; the
            # nodes then pull it themselves, which is the pre-pinning behaviour.
            log(f"k3d image import failed for {img}; nodes will pull it directly.")


def install_fake_gpu_operator(config: ClusterConfig) -> None:
    log(f"Installing fake-gpu-operator {config.fake_gpu_operator_version}...")

    values = f"""
# k3s pre-creates RuntimeClass "nvidia"; Helm can't adopt it (missing ownership
# annotations). Disable — nvml-mock doesn't need it.
runtimeClass:
  enabled: false
devicePlugin:
  enabled: true
statusUpdater:
  enabled: true
statusExporter:
  enabled: true
topologyServer:
  enabled: false
draPlugin:
  enabled: false
kwokDraPlugin:
  enabled: false
kwokGpuDevicePlugin:
  enabled: false
migFaker:
  enabled: false
gpuOperator:
  enabled: false
computeDomainController:
  enabled: false
computeDomainDraPlugin:
  enabled: false
kwokComputeDomainDraPlugin:
  enabled: false
topology:
  nodePoolLabelKey: run.ai/simulated-gpu-node-pool
  nodePools:
    {config.gpu_node_pool}:
      gpuProduct: "{config.gpu_product}"
      gpuCount: {config.gpus_per_node}
      gpuMemory: {config.gpu_memory_mib}
"""

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(values)
        values_path = f.name

    try:
        # Render the chart with these values so we know exactly which images the
        # install will schedule, then pre-warm them into the cluster before the
        # helm upgrade creates the pods (see prewarm_images).
        rendered = str(sh.helm(
            "template", "fake-gpu-operator",
            FAKE_GPU_OPERATOR_CHART,
            "--version", config.fake_gpu_operator_version,
            "--namespace", "gpu-operator",
            "-f", values_path,
        ))
        prewarm_images(config, image_refs(rendered))

        sh.helm(
            "upgrade", "-i", "fake-gpu-operator",
            FAKE_GPU_OPERATOR_CHART,
            "--version", config.fake_gpu_operator_version,
            "--namespace", "gpu-operator",
            "--create-namespace",
            "-f", values_path,
        )
    finally:
        Path(values_path).unlink(missing_ok=True)

    log(f"Waiting for status-updater to be ready (timeout {config.fake_gpu_operator_timeout})...")
    sh.kubectl("wait", "--for=condition=Ready", "pod", "-l", "app=status-updater", "-n", "gpu-operator", f"--timeout={config.fake_gpu_operator_timeout}")

    log(f"Waiting for device-plugin daemonset rollout (timeout {config.fake_gpu_operator_timeout})...")
    sh.kubectl("rollout", "status", "daemonset/device-plugin", "-n", "gpu-operator", f"--timeout={config.fake_gpu_operator_timeout}")

    wait_for_gpu_node_labels(config)


def install_nvml_mock(config: ClusterConfig) -> None:
    log("Installing nvml-mock (real mock libnvidia-ml.so + node labels)...")

    # test/e2e/hack -> repo root (hack, e2e, test, <root>).
    manifest = Path(__file__).resolve().parents[3] / NVML_MOCK_MANIFEST

    # The manifest spans gpu-operator and gpu-fractioning namespaces; create
    # both up-front so apply doesn't race. _ok_code tolerates AlreadyExists.
    sh.kubectl("create", "namespace", "gpu-operator", _ok_code=[0, 1])
    sh.kubectl("create", "namespace", "gpu-fractioning", _ok_code=[0, 1])

    # Pre-warm the nvml-mock image so the DaemonSet rollout below doesn't block on
    # a slow registry pull (see prewarm_images).
    prewarm_images(config, image_refs(manifest.read_text()))

    sh.kubectl("apply", "-f", str(manifest))

    # Guard against a silent no-op: a DaemonSet whose nodeSelector matches zero
    # nodes reports "rollout status" success with desiredNumberScheduled=0, so
    # without this check a selector mismatch looks like success and only surfaces
    # later as a confusing GPU-node-label timeout.
    desired = int(str(sh.kubectl(
        "get", "daemonset/nvml-mock", "-n", "gpu-operator",
        "-o", "jsonpath={.status.desiredNumberScheduled}",
    )).strip() or "0")
    if desired < config.gpu_worker_nodes:
        log(f"nvml-mock scheduled onto {desired} node(s), expected {config.gpu_worker_nodes} "
            f"— check the DaemonSet nodeSelector matches the GPU nodes' labels.")
        raise typer.Exit(1)

    log(f"Waiting for nvml-mock daemonset rollout (timeout {config.fake_gpu_operator_timeout})...")
    sh.kubectl("rollout", "status", "daemonset/nvml-mock", "-n", "gpu-operator", f"--timeout={config.fake_gpu_operator_timeout}")

    # Agents are pre-labeled nvidia.com/gpu.present=true at cluster creation and
    # nvml-mock's setup.sh re-applies it; confirm the label is present before the
    # suite (and the plugin's nodeSelector) relies on it.
    wait_for_gpu_node_labels(config)


def wait_for_gpu_node_labels(config: ClusterConfig, max_retries: int = 30, interval_seconds: int = 2) -> None:
    log("Waiting for status-updater to label GPU nodes (nvidia.com/gpu.present=true)...")
    for attempt in range(1, max_retries + 1):
        out = str(sh.kubectl(
            "get", "nodes", "-l", "nvidia.com/gpu.present=true",
            "--no-headers", _ok_code=[0, 1],
        )).strip()
        count = len(out.splitlines()) if out else 0
        if count >= config.gpu_worker_nodes:
            log(f"{count} node(s) labeled.")
            return
        if attempt == max_retries:
            log(f"timed out waiting for GPU node labels (found {count}/{config.gpu_worker_nodes})")
            raise typer.Exit(1)
        time.sleep(interval_seconds)


@app.command()
def main(
    delete: bool = typer.Option(False, "--delete", help="Delete the cluster and exit"),
    skip_fake_gpu_operator: bool = typer.Option(
        False, "--skip-fake-gpu-operator", help="Skip fake-gpu-operator install"
    ),
    skip_gpu_mock: bool = typer.Option(
        False, "--skip-gpu-mock", help="Skip nvml-mock install"
    ),
) -> None:
    """Create (or delete) a k3d cluster with fake-gpu-operator and nvml-mock for gpu-fractioning e2e tests."""
    config = ClusterConfig()
    if not config.kubeconfig:
        config.kubeconfig = str(Path.home() / ".kube" / f"{config.cluster_name}.yaml")

    require_command("k3d")
    require_command("kubectl")
    require_command("docker")

    if not skip_fake_gpu_operator:
        require_command("helm")
        if not config.fake_gpu_operator_version:
            log("E2E_FAKE_GPU_OPERATOR_VERSION must be set (or pass --skip-fake-gpu-operator).")
            log("See https://github.com/run-ai/fake-gpu-operator/releases for available versions.")
            raise typer.Exit(1)

    if delete:
        delete_cluster(config)
        delete_registry(config)
        Path(config.kubeconfig).expanduser().unlink(missing_ok=True)
        return

    # The registry must exist before the cluster: create_cluster's --registry-use
    # connects it to the cluster network and generates the node registries.yaml.
    create_registry(config)

    if not create_cluster(config):
        raise typer.Exit(1)

    # Extract the kubeconfig (and point this process at it) before any kubectl/helm.
    write_kubeconfig(config)

    wait_for_nodes(config)

    if not skip_fake_gpu_operator:
        install_fake_gpu_operator(config)

    if not skip_gpu_mock:
        install_nvml_mock(config)

    log(f"Cluster '{config.cluster_name}' is ready for e2e tests.")
    log("Next steps (make targets set KUBECONFIG for you):")
    log(f"  make e2e-deploy         # build + load images + install operator")
    log(f"  make test-e2e-metrics   # run the metrics suite")
    log(f"Or point your shell at it directly: export KUBECONFIG={config.kubeconfig}")


if __name__ == "__main__":
    app()
