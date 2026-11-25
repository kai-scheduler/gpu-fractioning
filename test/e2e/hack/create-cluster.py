#!/usr/bin/env python3
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disposable kind cluster with GPU Operator + Mokka; no FGO or KWOK."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]


def run(*args, env=None, input_text=None):
    subprocess.run(args, check=True, env=env, input=input_text, text=True)


def cluster_config(gpu_nodes, cpu_nodes):
    text = '''kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
  - |-
    [plugins."io.containerd.nri.v1.nri"]
      disable = false
    [plugins."io.containerd.cri.v1.images".registry]
      config_path = "/etc/containerd/certs.d"
nodes:
  - role: control-plane
'''
    text += '''  - role: worker
    labels:
      mokka.nvidia.com/e2e-gpu: "true"
''' * gpu_nodes
    text += '  - role: worker\n' * cpu_nodes
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    name = os.environ.get('E2E_CLUSTER_NAME', 'gpu-fractioning-e2e')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name):
        parser.error('E2E_CLUSTER_NAME must be a kind/Docker-compatible name')
    registry = name + '-registry'
    kubeconfig = Path(os.environ.get('E2E_KUBECONFIG', str(Path.home() / '.kube' / (name + '.yaml')))).expanduser()
    env = os.environ | {'KUBECONFIG': str(kubeconfig), 'CLUSTER_NAME': name,
                        'GPU_OPERATOR_VERSION': os.environ.get('E2E_GPU_OPERATOR_VERSION', 'v26.7.2')}
    gpu_nodes = int(os.environ.get('E2E_GPU_WORKER_NODES', '2'))
    cpu_nodes = int(os.environ.get('E2E_NON_GPU_WORKER_NODES', '1'))
    gpu_count = int(os.environ.get('E2E_GPUS_PER_NODE', '2'))
    port = int(os.environ.get('E2E_REGISTRY_PORT', '5001'))
    if gpu_nodes < 1 or cpu_nodes < 0 or not 1 <= gpu_count <= 8 or not 1 <= port <= 65535:
        parser.error('invalid node count, GPU count, or registry port')
    run('kind', 'delete', 'cluster', '--name', name)
    containers = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
    if registry in containers:
        run('docker', 'rm', '-f', registry)
    if args.delete:
        return

    kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    image = os.environ.get('E2E_KIND_IMAGE', 'kindest/node:v1.35.0')
    with tempfile.TemporaryDirectory(prefix='gpu-fractioning-kind-') as tmp:
        path = Path(tmp) / 'kind.yaml'
        path.write_text(cluster_config(gpu_nodes, cpu_nodes))
        run('kind', 'create', 'cluster', '--name', name, '--image', image,
            '--config', str(path), '--kubeconfig', str(kubeconfig), '--wait', '180s')

    run('docker', 'run', '-d', '--restart=always', '--name', registry,
        '--network', 'kind', '-p', f'127.0.0.1:{port}:5000', 'registry:2')
    hosts = f'server = "http://localhost:{port}"\n[host."http://{registry}:5000"]\n  capabilities = ["pull", "resolve"]\n'
    nodes = subprocess.check_output(['kind', 'get', 'nodes', '--name', name], text=True).splitlines()
    for node in nodes:
        run('docker', 'exec', '-i', node, 'sh', '-c',
            'mkdir -p "$1"; cat > "$1/hosts.toml"', '--',
            f'/etc/containerd/certs.d/localhost:{port}', input_text=hosts)

    run('bash', str(ROOT / 'hack/install-mokka-gpu-operator.sh'),
        env=env | {'MOKKA_GPU_COUNT': str(gpu_count)})
    run('kubectl', 'create', 'namespace', 'gpu-fractioning', env=env)
    # Only metricsd gets the controlled process data. The old NVML DaemonSet
    # would overwrite the driver files now owned by Mokka and GPU Operator.
    run('kubectl', 'apply', '-f', str(ROOT / 'test/e2e/nvmlmock/configmap.yaml'), env=env)
    print(json.dumps({'context': 'kind-' + name, 'kubeconfig': str(kubeconfig),
                      'gpu_nodes': gpu_nodes, 'gpus_per_node': gpu_count}))


if __name__ == '__main__':
    main()
