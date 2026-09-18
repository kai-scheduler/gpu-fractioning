<!--
Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvmlshim

A shadow `libnvidia-ml.so.1` that gives a container a capped, container-scoped
view of GPU memory, and forwards every other NVML call to the real driver
untouched.

## The problem

We share one physical GPU between pods with MPS. Each container gets
`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` and `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, and
MPS enforces both at the CUDA layer. NVML does not go through MPS, so
`nvidia-smi`, `pynvml` and DCGM all read the raw physical device. Measured in a
pod with a 48431 MiB limit on a 97887 MiB card, sharing the GPU with one other
pod holding 40 GiB:

```
memory.total = 97887 MiB   <- the card, not the 48431 MiB limit
memory.used  = 82662 MiB   <- both tenants; the neighbour's footprint leaks across the pod boundary
memory.free  = 14679 MiB   <- device-wide
```

Frameworks size their allocators off `memory.total`, and `memory.used` leaking a
neighbour's footprint is an isolation problem in its own right. Both numbers are
produced inside NVML, so that is where they have to be corrected.

## Why LD_PRELOAD does not work

`nvidia-smi` does not link `libnvidia-ml.so` at all. `readelf -d` on it lists
only `libpthread`, `libm`, `libdl`, `libc` and `librt`. It `dlopen`s NVML at
runtime and resolves each entry point with `dlsym(handle, ...)`.

`dlsym` on an explicit handle searches that object and its dependencies. It does
not consult the preload list, so an `LD_PRELOAD`ed definition of
`nvmlDeviceGetMemoryInfo` is never reached — verified empirically, the probe's
hooks were never called. Interposition only works for symbols bound through the
global lookup scope, which is not how any NVML consumer resolves NVML.

## How the shadow mechanism works

Instead of interposing, we *are* the library the loader finds:

1. The shim is built as `libnvidia-ml.so.1` with `SONAME=libnvidia-ml.so.1`.
2. Its directory goes on `LD_LIBRARY_PATH` ahead of the driver's. `dlopen("libnvidia-ml.so.1")`
   then finds ours.
3. On load, the constructor `dlopen`s the *real* driver by absolute path and
   fills a table of function pointers from it.
4. 429 of the 431 NVML entry points are forwarding trampolines through that
   table. Two are implemented here.

### Finding the real driver without finding ourselves

The driver's SONAME is also `libnvidia-ml.so.1`, so a `DT_NEEDED` dependency
would resolve straight back to us. The shim `dlopen`s an absolute path instead,
resolved at runtime in this order:

1. `GPU_FRACTIONING_REAL_NVML`, if set — a colon-separated list of full paths.
2. `libnvidia-ml.so.*[0-9]` under `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`,
   `/usr/lib64/nvidia`, `/usr/lib`, `/usr/lib/nvidia`, `/usr/local/nvidia/lib64`
   and `/usr/local/nvidia/lib`.

`/usr/lib/x86_64-linux-gnu` is the one that matters on the Debian- and
Ubuntu-based CUDA images this ships into, and it is notably absent from
`nvidia-smi`'s own hardcoded search list. `/usr/local/nvidia/{lib,lib64}` are
included because the NVIDIA container runtime puts them on `LD_LIBRARY_PATH`
inside GPU containers.

Note that `libnvidia-ml.so.*[0-9]` **does** match the bare `libnvidia-ml.so.1` —
the `*` can match nothing — which is the name the shim itself is installed
under. Three defences, in order:

- Candidates are tried in two passes, version-suffixed names
  (`libnvidia-ml.so.615.71.09`, which can only be the driver) before the bare
  SONAME. The alias is still tried last, since an image that ships only the
  SONAME is possible.
- Each candidate is `stat`ed and compared with this library's own device and
  inode, obtained via `dladdr` on one of our static functions. `stat` follows
  symlinks, so the driver's `libnvidia-ml.so.1 -> libnvidia-ml.so.615.71.09`
  compares as the versioned file it points at.
- After a successful `dlopen`, `dladdr` on a symbol from the new handle must not
  land in our own mapping. This catches the case where `stat` failed and the
  loader handed back our own already-loaded object.

### Forwarding 431 symbols

We have signatures for none of the 431 entry points, and a C wrapper must name
its argument and return types — getting either wrong corrupts the call. So the
forwarders are generated x86-64 assembly instead:

```asm
nvmlDeviceGetAccountingMode:
	endbr64
	movq	nvmlshim_slots+104(%rip), %r11
	testq	%r11, %r11
	je	.Lnvmlshim_unresolved
	jmp	*%r11
```

The tail jump leaves every argument register, the stack and the return value
exactly as the caller left them, which is correct for every signature precisely
because it never claims to know what they are. `%r11` is the one register that
is both caller-saved and unused by the SysV AMD64 convention for arguments, the
static chain (`%r10`) and the varargs vector count (`%al`).

`gen-trampolines.sh` emits these from the committed `symbols.txt`, minus the
entries in `overrides.txt`. The symbol list is committed rather than probed so
the library builds in a container stage with no driver installed; when a driver
*is* present the generator diffs against it and fails the build on drift rather
than silently changing.

A slot that could not be resolved — an entry point this driver version does not
have — returns `NVML_ERROR_FUNCTION_NOT_FOUND` (13), which is what NVML itself
returns in that situation, rather than jumping through a NULL.

## What is capped

| Entry point | `total` | `used` | `free` | `reserved` |
| --- | --- | --- | --- | --- |
| `nvmlDeviceGetMemoryInfo` | the limit | this container's processes | `total - used` | n/a |
| `nvmlDeviceGetMemoryInfo_v2` | the limit | this container's processes | `total - used` | passed through |

Both versions are intercepted deliberately. `nvidia-smi` on driver 615.71.09
calls `_v2`, but `pynvml` and older DCGM builds call the unversioned entry point,
and one uncapped version leaks the real device size to everything that uses it.

`used` is the sum of `usedGpuMemory` over
`nvmlDeviceGetComputeRunningProcesses_v3` (falling back to `_v2`, then v1).
Inside the container that list is already scoped to the container's own
processes — verified in a pod, where it returned exactly one entry, our own
process, with a namespaced PID. The PID namespace does the scoping, so no access
to the MPS control socket is needed. Entries reporting `NVML_VALUE_NOT_AVAILABLE`
are skipped rather than summed.

`used` is clamped to the limit. A container cannot really own more than MPS
lets it allocate, so a larger sum means the list was not container-scoped after
all (running on the host, or a pod sharing the host PID namespace). Clamping
keeps the triple self-consistent; without it every consumer computing
`total - used` underflows into an enormous `free`.

## What is NOT capped

- **Utilization.** `nvmlDeviceGetUtilizationRates` is left forwarding. Device-wide
  utilization is real information, and a fabricated per-container number could
  mislead an autoscaler considerably worse than an honest device-wide one. A
  container with a 50% thread percentage will still see whole-device SM
  utilization, including its neighbours' work.
- **Anything that opens the driver by absolute path.** The shadow mechanism works
  by winning a search of `LD_LIBRARY_PATH`. A process that `dlopen`s
  `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` directly, or that is linked with
  an `RPATH`/`RUNPATH` naming the driver's directory, bypasses the shim entirely
  and sees the raw device. Statically linked NVML consumers likewise.
- **Everything other than memory.** Every other one of the 431 entry points
  reports the physical device: BAR1 memory, ECC counters, clocks, power,
  processes, MIG topology. Only the two memory calls above are rewritten.
- **Graphics and MPS-server process lists.** `nvmlDeviceGetGraphicsRunningProcesses`
  and `nvmlDeviceGetMPSComputeRunningProcesses` forward unchanged; only the
  compute list feeds the capped `used`.
- **Non-NVML paths.** The CUDA driver API's own memory queries do not go through
  NVML and are unaffected. MPS remains the thing that actually enforces the
  limit; this component only makes the reporting agree with it.

## The fail-open contract

This library sits in front of every GPU process in the container. Every failure
degrades to reporting the real device, never to an error and never to a crash:

| Situation | Behaviour |
| --- | --- |
| `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` unset or empty | real device reported |
| limit malformed, zero, or overflowing | whole spec rejected, real device reported |
| no limit listed for this device index | real device reported |
| device index unresolvable, several limits configured | real device reported |
| device index unresolvable, exactly one limit configured | that limit applied |
| compute-process query fails | limit still applied; the driver's own `used` reported, clamped |
| an entry point missing from this driver | that call returns `NVML_ERROR_FUNCTION_NOT_FOUND` |
| `GPU_FRACTIONING_NVML_SHIM=0` | pure pass-through |

A zero limit is treated as malformed rather than honoured: reporting a GPU with
zero total memory reads to a scheduler or autoscaler as a broken device, which
is a louder and more damaging lie than the uncapped number it falls back to.

One caveat on the unresolved-slot path: it returns 13 in the integer return
register, which is right for the `nvmlReturn_t` that all but a handful of NVML
entry points return, but would be a bogus non-NULL pointer for one that returns
a pointer (`nvmlErrorString`). That only arises if the driver does not export
the symbol at all, in which case the caller had no working function either way;
returning 0 instead would be worse, since every integer-returning call would
then report `NVML_SUCCESS` over an untouched output buffer.

The one case that is not transparent is the driver being absent altogether — if
no real NVML can be found there is nothing to forward to, and every call returns
`NVML_ERROR_FUNCTION_NOT_FOUND`. In that situation NVML was already unusable.

## Environment

| Variable | Meaning |
| --- | --- |
| `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` | the limit, as MPS itself takes it: `0=48431M`, `0=1024M,1=2G`. `K`/`M`/`G` suffixes and a bare byte count are accepted. |
| `GPU_FRACTIONING_NVML_SHIM=0` | disable capping; forward everything. For debugging. |
| `GPU_FRACTIONING_NVML_SHIM_DEBUG=1` | diagnostics to stderr. Off by default: this is in the path of every process. |
| `GPU_FRACTIONING_REAL_NVML` | colon-separated absolute paths to try before the search globs. |

### Known gaps in limit matching

- MPS keys its limits by the device index **as the control daemon sees it**,
  which need not be how NVML numbers devices inside the container. The shim
  matches on `nvmlDeviceGetIndex`. Where the two disagree and more than one
  limit is configured, no limit is applied and the real device is reported.
- MPS also accepts a **device UUID** as a key. The shim does not resolve UUIDs to
  handles, so a UUID-keyed limit is only applied when the device index cannot be
  resolved at all *and* it is the only entry. On a one-GPU container with a
  resolvable index, a UUID-keyed limit currently falls open.
- v1's `used` is documented as reserved plus allocated, whereas the capped value
  is the sum over this container's processes and so excludes the driver's own
  reserve (a few hundred MiB). Attributing the driver's bookkeeping to one tenant
  of a shared GPU would be the less accurate choice.

## Architecture support

| Architecture | Behaviour |
| --- | --- |
| x86-64 | fully supported: capping and forwarding |
| everything else (incl. arm64 / GH200 / GB200) | **not built** |

The forwarders are hand-written x86-64 assembly. aarch64 needs a different
sequence (`br` through a register) and a different calling convention, and an
untested trampoline in the path of every GPU process is worse than no shim at
all — so rather than guess, the build emits nothing on other architectures and
exits successfully. `make` prints why.

That keeps the multi-arch image build green while leaving the arm64 image with
no shadow library, so `dlopen("libnvidia-ml.so.1")` finds the driver's own and
the container behaves exactly as it does today: **uncapped reporting, never a
broken workload**. `nvmlshim.h` additionally fails the compile with `#error` if
the C is ever built for another target, so no path can quietly produce a library
that returns wrong numbers.

## Building and testing

```sh
make          # build/libnvidia-ml.so.1
make test     # build, then run the self-test against the real driver
make clean
```

Nothing in the build needs a driver or the CUDA toolkit: the entry-point list
comes from `symbols.txt` and the NVML structs are restated in `nvmlshim.h`, with
`_Static_assert`s on their sizes. Building on a GPU node additionally verifies
`symbols.txt` against the installed driver.

`make verify-symbols` checks the committed list against the local driver on its
own; `make regen-symbols` re-captures it after a qualified driver upgrade.

The self-test covers the capped total through `nvidia-smi`, both memory entry
points in bytes, the `free + used == total` invariant, all four fail-open cases,
the opt-out, and forwarding (identical `name`/`driver_version`/`uuid`, and a
clean exit from the full `nvidia-smi` table and `nvidia-smi -q`, which between
them walk a wide swath of the 429 forwarded entry points).

The paths a healthy GPU host cannot be made to produce — a driver missing most
of its entry points, a failing process query, a container over its own limit, an
unresolvable device index — run against `test/fakenvml.c`, a synthetic driver
reached through `GPU_FRACTIONING_REAL_NVML`. That also pins the capping
arithmetic to exact figures instead of to whatever is resident on the machine.

## Deployment

The shim only works if its directory precedes the driver's on the library search
path, so `LD_LIBRARY_PATH` must name it in the container. Note that GPU
containers already ship with
`LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64`; the shim's
directory must be **prepended** to that, not replace it.

## Layout

| Path | |
| --- | --- |
| `nvmlshim.c` | resolution, limit parsing, container-scoped usage, the two overrides |
| `nvmlshim.h` | NVML ABI restatement, `_Static_assert`s, generated-table interface |
| `gen-trampolines.sh` | emits the trampolines; diffs `symbols.txt` against a driver when present |
| `symbols.txt` | the 431 entry points, captured from driver 615.71.09 |
| `overrides.txt` | the entry points implemented natively |
| `test/run-tests.sh` | the self-test |
| `test/memquery.c` | reads both memory entry points in bytes |
| `test/fakenvml.c` | synthetic driver for the degraded paths |
| `test/shimcheck.c` | drives the shim against the synthetic driver |
