# AMD GPUs

fractiond enforces GPU-memory fractions on AMD GPUs as well as NVIDIA GPUs. The
scheduler writes the same per-container annotations; with `--gpu-vendor=amd`,
fractiond turns them into limits of the Linux **dmem cgroup controller**, and the
kernel refuses allocations past them.

This guide covers what is and is not covered on AMD GPUs, the requirements, how
to run fractiond on AMD GPU nodes, and how to check that a limit is in force.

## How it works

AMD GPUs have no driver-side memory limit and no MPS. Since Linux 6.14, though,
the `amdgpu` driver registers each GPU's VRAM with the dmem cgroup controller as a
region named `drm/<PCI address>/vram`:

```console
$ cat /sys/fs/cgroup/dmem.capacity
drm/0000:03:00.0/vram 48301604864
```

A cgroup's `dmem.max` caps how much of a region its processes can allocate, and
`dmem.min` protects an amount from eviction by other cgroups. For a container
with GPU-memory annotations, fractiond has the container runtime write both files
on the **container's** cgroup when the container is created, through NRI
(`linux.resources.unified`):

| Annotation | Written as | Effect |
| --- | --- | --- |
| `<prefix><name>.gpu-memory.limit` | `dmem.max` | allocations past it fail (`hipErrorOutOfMemory`) |
| `<prefix><name>.gpu-memory.request` | `dmem.min` | that much is protected from eviction |
| `<prefix><name>.gpus.devices` | which regions | the GPUs' PCI addresses; the request and limit apply to each |

As on NVIDIA GPUs, a missing request or limit defaults to the other. For a
container named `trainer` on the GPU at `0000:03:00.0`:

```yaml
annotations:
  nvidia.com/container.trainer.gpu-memory.request: 8Gi
  nvidia.com/container.trainer.gpu-memory.limit: 16Gi
  nvidia.com/container.trainer.gpus.devices: "0000:03:00.0"
```

becomes, on the container's cgroup:

```text
dmem.min  drm/0000:03:00.0/vram 8589934592
dmem.max  drm/0000:03:00.0/vram 17179869184
```

fractiond itself never touches the host's cgroup tree: the runtime writes the
files, so fractiond needs only the NRI socket.

## What is and is not covered

| | NVIDIA GPUs | AMD GPUs |
| --- | --- | --- |
| Memory limit | `NVIDIA_GPU_MEMORY_LIMIT`, enforced by the driver | `dmem.max`, enforced by the kernel |
| Memory request | for accounting | `dmem.min`: protected from eviction |
| Limit seen by the process | the GPU reports the limit | the GPU reports its whole memory (`hipMemGetInfo`); see below |
| Compute sharing | time-slicing or `sm-sharing` (MPS) | time-slicing only; `sm-sharing` is rejected |
| Device assignment | `gpus.devices` → `NVIDIA_VISIBLE_DEVICES` (and CDI) | `gpus.devices` names the GPUs to limit; the GPU reaches the container through AMD's device plugin or the scheduler's own device injection |
| Retroactive enforcement | yes | no (turned off) |
| Per-pod metrics (metricsd) | NVML | not yet; `dmem.current` on the container's cgroup holds its usage |
| Deployed by the operator | yes | not yet: run fractiond yourself, as below |

ROCm still reports the whole GPU's memory to the process inside a limit, so
frameworks that size themselves from it should be told their share: vLLM's
`--gpu-memory-utilization` (for example `0.45` for half a GPU) or PyTorch's
`torch.cuda.set_per_process_memory_fraction`. A process that ignores its share is
stopped at the limit rather than taking its neighbors' memory.

## Requirements

- Linux **6.14 or newer** on the GPU nodes, with the `amdgpu` driver bound to the
  GPUs: each GPU's region must appear in `/sys/fs/cgroup/dmem.capacity`.
- **cgroup v2**, with the `dmem` controller enabled down to the containers. The
  kubelet enables every controller the kernel offers on the pod hierarchy; check
  with `cat /sys/fs/cgroup/kubepods.slice/cgroup.subtree_control`.
- containerd 2.0+ with **NRI enabled**, or CRI-O with NRI support, and a runtime
  that writes `linux.resources.unified` (runc does).
- A scheduler that writes the annotations with the GPUs' **PCI addresses** in
  `gpus.devices`.

On a node whose kernel has no dmem region for the GPU, the runtime cannot write
the limit and container creation fails: fractiond does not start a container it
cannot limit (unless it runs fail-open).

## Running fractiond on AMD GPU nodes

The operator does not deploy fractiond on AMD GPU nodes yet. Until it does, run it
as a DaemonSet of your own on the AMD GPU nodes. It needs no GPU, no MPS daemon and
no NVIDIA runtime class: only the NRI socket, as root, because the socket is
root-owned.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fractiond-amd
  namespace: gpu-fractioning
spec:
  selector:
    matchLabels: {app: fractiond-amd}
  template:
    metadata:
      labels: {app: fractiond-amd}
    spec:
      nodeSelector:
        amd.com/gpu.present: "true"      # however your AMD GPU nodes are labelled
      priorityClassName: system-node-critical
      automountServiceAccountToken: false
      containers:
        - name: fractiond
          image: ghcr.io/kai-scheduler/gpu-fractioning/fractiond:<VERSION>
          args: [--gpu-vendor=amd, --map-dir=/run/fractiond/map]
          securityContext:
            runAsUser: 0
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
          readinessProbe:
            httpGet: {path: /readyz, port: 8093}
          volumeMounts:
            - {name: nri, mountPath: /var/run/nri}
            - {name: run, mountPath: /run/fractiond}
      volumes:
        - {name: nri, hostPath: {path: /var/run/nri, type: Directory}}
        - {name: run, emptyDir: {}}
```

`--annotation-prefix`, `--fail-open`, `--log-level` and the NRI connection flags
work as on NVIDIA GPU nodes. Do not run it on nodes with NVIDIA GPUs: one
fractiond enforces one vendor.

## Checking a limit

fractiond logs each container it limits:

```json
{"level":"INFO","msg":"adjusting container with AMD GPU memory limits","container":"trainer","pod":"train-0","requestMiB":"8192","limitMiB":"16384","devices":["0000:03:00.0"]}
```

On the node, the container's cgroup carries the limits, while the pod's cgroup
does not:

```console
$ cat /sys/fs/cgroup/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<uid>.slice/cri-containerd-<id>.scope/dmem.max
drm/0000:03:00.0/vram 17179869184
$ cat /sys/fs/cgroup/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<uid>.slice/dmem.max
drm/0000:03:00.0/vram max
```

`dmem.current` next to them shows what the container holds.

Tested on a Radeon PRO W7900 (48 GB) with Linux 7.0, Kubernetes 1.35 and containerd with NRI: a container
limited to 11,516 MiB that asked for 30 GiB was stopped at 11,264 MiB, while a
neighbor on the same GPU allocated its 20 GiB.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Container creation fails: `no GPU assigned` | The pod has GPU-memory annotations but no `gpus.devices` for that container. The scheduler must name the GPU. |
| Container creation fails: `not a PCI address` | `gpus.devices` holds something else (an NVIDIA UUID, an index, `all`). Use the PCI address, for example `0000:03:00.0` (`lspci -D`). |
| Container creation fails writing `dmem.max` | The GPU's VRAM is not a dmem region on that node (kernel older than 6.14, or `amdgpu` not bound), or `dmem` is not enabled down to the container (`cgroup.subtree_control`). |
| A process fails with out-of-memory below what the GPU reports | Expected: ROCm reports the whole GPU. Size the framework from its share. |
| `sm-sharing` is rejected | AMD GPUs have no MPS. Remove `gpu-compute.mode` or set it to `time-slicing`. |

## References

- [The dmem controller, Control Group v2 (Linux kernel documentation)](https://docs.kernel.org/admin-guide/cgroup-v2.html#dmem)
- [NRI, Node Resource Interface](https://github.com/containerd/nri)
- [ROCm/rocm-systems#14860: `hipMemGetInfo` and the dmem limit](https://github.com/ROCm/rocm-systems/issues/14860)
