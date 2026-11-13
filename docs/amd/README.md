# AMD GPUs

fractiond enforces GPU-memory fractions on AMD GPUs as well as NVIDIA GPUs. The
scheduler writes the same per-container annotations; with `--gpu-vendor=amd`,
fractiond turns them into a limit of the Linux **dmem cgroup controller**, and the
kernel refuses VRAM allocations past it.

This guide covers what is and is not covered on AMD GPUs, the requirements, how
to run fractiond on AMD GPU nodes, and how to check that a limit is in force.

## How it works

AMD GPUs have no driver-side memory limit and no MPS. Since Linux **6.15**, though,
the `amdgpu` driver registers each GPU's VRAM with the dmem cgroup controller as a
region named `drm/<PCI address>/vram` (Linux 6.14 added the controller itself, but
amdgpu registered no region until 6.15):

```console
$ cat /sys/fs/cgroup/dmem.capacity
drm/0000:03:00.0/vram 48301604864
```

A cgroup's `dmem.max` caps how much of a region its processes can allocate. For a
container with GPU-memory annotations, fractiond has the container runtime write
`dmem.max` on the **container's** cgroup when the container is created, through
NRI (`linux.resources.unified`):

| Annotation | Written as | Effect |
| --- | --- | --- |
| `<prefix><name>.gpu-memory.limit` | `dmem.max` | allocations past it fail (`hipErrorOutOfMemory`) |
| `<prefix><name>.gpu-memory.request` | nothing | used for accounting only, as on NVIDIA GPUs |
| `<prefix><name>.gpus.devices` | which regions | the GPUs' PCI addresses; the limit applies to each |

As on NVIDIA GPUs, a missing limit defaults to the request. For a container named
`trainer` on the GPU at `0000:03:00.0`:

```yaml
annotations:
  nvidia.com/container.trainer.gpu-memory.request: 8Gi
  nvidia.com/container.trainer.gpu-memory.limit: 16Gi
  nvidia.com/container.trainer.gpus.devices: "0000:03:00.0"
```

becomes, on the container's cgroup:

```text
dmem.max  drm/0000:03:00.0/vram 17179869184
```

The request is not written as `dmem.min`. `dmem.min` is consulted only when the
kernel decides whose buffers it may evict, never to refuse an allocation, and its
effective value is capped by the ancestors' (`kubepods.slice` and the pod's cgroup
set none, so it would be 0). amdgpu gained dmem reclaim in Linux 7.3, which is
where a protection could start to mean something.

fractiond never writes the host's cgroup tree: the runtime writes the container's
files. It reads the host's `dmem.capacity` (read-only) to check that each GPU's
VRAM is a dmem region before it asks for a limit.

## What is and is not covered

| | NVIDIA GPUs | AMD GPUs |
| --- | --- | --- |
| Memory limit | `NVIDIA_GPU_MEMORY_LIMIT`, enforced by the driver | `dmem.max` on VRAM, enforced by the kernel |
| Memory request | for accounting | for accounting |
| Limit seen by the process | the GPU reports the limit | the GPU reports its whole memory (`hipMemGetInfo`); see below |
| Compute sharing | time-slicing or `sm-sharing` (MPS) | time-slicing only; `sm-sharing` is rejected |
| Device assignment | `gpus.devices` → `NVIDIA_VISIBLE_DEVICES` (and CDI) | `gpus.devices` names the GPUs to limit; the GPU reaches the container through AMD's device plugin or the scheduler's own device injection |
| Retroactive enforcement | stops a container created while fractiond was down, so it is recreated | sets the missing `dmem.max` on it in place, without a restart |
| Per-pod metrics (metricsd) | NVML | not yet; `dmem.current` on the container's cgroup holds its usage |
| Deployed by the operator | yes | not yet: run fractiond yourself, as below |

### Limits of the cap

- **Only VRAM is capped.** amdgpu registers a dmem region for VRAM, not for GTT
  (system memory the GPU maps). An allocation that falls back to, or asks for, host
  memory is not limited; this has been seen with `hipMallocAsync`
  ([Project-HAMi/amd-hami-core#6](https://github.com/Project-HAMi/amd-hami-core/issues/6)).
- **ROCm reports the whole GPU.** `hipMemGetInfo` and the HSA memory pool size ignore
  the cap ([ROCm/rocm-systems#14860](https://github.com/ROCm/rocm-systems/issues/14860)),
  so frameworks that size themselves from it should be told their share: vLLM's
  `--gpu-memory-utilization` (for example `0.45` for half a GPU) or PyTorch's
  `torch.cuda.set_per_process_memory_fraction`. A process that ignores its share is
  stopped at the limit rather than taking its neighbors' memory.
- **Instinct partitions are untested.** amdgpu registers one region per card, keyed by
  the card's PCI address, while compute partitions (CPX, and the like on MI300 series
  GPUs) appear as separate DRM devices. A cap per partition cannot be expressed: one
  cap applies across the whole card. All lab data so far is from a Radeon PRO W7900.
- **Several GPUs per container are untested on hardware.** The value has one line per
  GPU, and the kernel takes one region per write, so it depends on the runtime
  writing it line by line (runc 1.2.0+, crun). It is covered by unit tests only.

## Requirements

1. Linux **6.15 or newer** with `CONFIG_CGROUP_DMEM` and the `amdgpu` driver bound to
   the GPUs: each GPU's region must appear in `/sys/fs/cgroup/dmem.capacity`.
2. **cgroup v2**, with `dmem` in `cgroup.subtree_control` down to the containers. The
   kubelet enables every controller the kernel offers on the pod hierarchy; check with
   `cat /sys/fs/cgroup/kubepods.slice/cgroup.subtree_control`.
3. containerd 2.0+ or CRI-O with **NRI enabled**. Both pass `linux.resources.unified`
   from NRI to the runtime.
4. **runc 1.2.0 or newer, or crun**, for containers on more than one GPU: they write a
   multi-line unified value line by line. runc 1.1.x writes it in one call, which the
   kernel refuses, so every multi-GPU container would fail.
5. A scheduler that writes the annotations with the GPUs' **PCI addresses** in
   `gpus.devices` (for KAI Scheduler, see
   [kai-scheduler/KAI-Scheduler#2556](https://github.com/kai-scheduler/KAI-Scheduler/issues/2556)),
   and an admission policy that lets only the scheduler set them.
6. AMD's device plugin, or the scheduler, to give the container the GPU: fractiond
   injects no devices on AMD GPUs.
7. The host's `/sys/fs/cgroup` mounted read-only into fractiond, for `dmem.capacity`
   (see below).

The operator has no AMD support yet (no vendor field in `GpuFractioningConfig`), so
fractiond runs as a DaemonSet of your own on AMD GPU nodes.

## Running fractiond on AMD GPU nodes

Run it as a DaemonSet on the AMD GPU nodes. It needs no GPU, no MPS daemon and no
NVIDIA runtime class: only the NRI socket, as root, because the socket is
root-owned, and the host's cgroup filesystem, read-only.

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
            - {name: cgroup, mountPath: /host/sys/fs/cgroup, readOnly: true}
      volumes:
        - {name: nri, hostPath: {path: /var/run/nri, type: Directory}}
        - {name: run, emptyDir: {}}
        - {name: cgroup, hostPath: {path: /sys/fs/cgroup, type: Directory}}
```

The host's `/sys/fs/cgroup` is needed because `dmem.capacity` exists only on the root
cgroup, which a container's own cgroup namespace does not show. fractiond reads
`/host/sys/fs/cgroup/dmem.capacity` by default; `--dmem-capacity-file`
(`DMEM_CAPACITY_FILE`) changes the path.

fractiond becomes ready only once `dmem.capacity` lists at least one GPU VRAM
region. On a node without one (an older kernel, `amdgpu` not bound, or the mount
missing) it stays not ready and logs why, so a DaemonSet rollout stops there
instead of every GPU pod failing.

`--retroactive-enforcement` (on by default) re-applies the limits when fractiond
connects to the runtime: a fractional container created while fractiond was not
running (a restart, an upgrade, a node booting) has no `dmem.max`, and fractiond
has the runtime set it in place. No CRI socket is needed for this on AMD GPU nodes.
`--annotation-prefix`, `--fail-open`, `--log-level` and the NRI connection flags
work as on NVIDIA GPU nodes. Do not run it on nodes with NVIDIA GPUs: one fractiond
enforces one vendor.

## Checking a limit

fractiond logs each container it limits:

```json
{"level":"INFO","msg":"adjusting container with AMD GPU memory limits","container":"trainer","pod":"train-0","requestMiB":"8192","limitMiB":"16384","devices":["0000:03:00.0"]}
```

On the node, the container's cgroup carries the limit, while the pod's cgroup
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
| fractiond not ready: `cannot check the node's dmem regions` | `dmem.capacity` cannot be read: mount the host's `/sys/fs/cgroup` read-only at `/host/sys/fs/cgroup`, or point `--dmem-capacity-file` at it. |
| fractiond not ready: `no GPU VRAM is a dmem region on this node` | The kernel is older than 6.15, `CONFIG_CGROUP_DMEM` is off, or `amdgpu` is not bound to the GPUs. |
| Container creation fails: `GPU <address> has no dmem VRAM region` | `gpus.devices` names a GPU this node has no region for: a wrong address, or a GPU without `amdgpu`. With `--fail-open` the container starts uncapped instead. |
| Container creation fails: `no GPU assigned` | The pod has GPU-memory annotations but no `gpus.devices` for that container. The scheduler must name the GPU. |
| Container creation fails: `not a PCI address` | `gpus.devices` holds something else (an NVIDIA UUID, an index, `all`). Use the PCI address, for example `0000:03:00.0` (`lspci -D`). |
| Container creation fails: `unable to set unified resource "dmem.max"` | `dmem` is not enabled down to the container (`cgroup.subtree_control`), or the container is on several GPUs and the runtime is runc older than 1.2.0. |
| A process fails with out-of-memory below what the GPU reports | Expected: ROCm reports the whole GPU. Size the framework from its share. |
| `sm-sharing` is rejected | AMD GPUs have no MPS. Remove `gpu-compute.mode` or set it to `time-slicing`. |

## References

- [The dmem controller, Control Group v2 (Linux kernel documentation)](https://docs.kernel.org/admin-guide/cgroup-v2.html#dmem)
- [NRI, Node Resource Interface](https://github.com/containerd/nri)
- [ROCm/rocm-systems#14860: `hipMemGetInfo` and the dmem limit](https://github.com/ROCm/rocm-systems/issues/14860)
- [Project-HAMi/amd-device-plugin#99](https://github.com/Project-HAMi/amd-device-plugin/issues/99): the same dmem cap in HAMi's AMD device plugin
