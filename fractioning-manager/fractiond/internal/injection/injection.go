// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package injection holds the "output" contract of the fractiond NRI plugin: the
// environment variables and mounts the CreateContainer hook injects into
// GPU-fractioning containers. It is the counterpart to the annotations package
// (the "input" read from the pod) and exists so the injection side
// (internal/plugin.go buildAdjustment) and the verification side
// (internal/audit) reference one source of truth instead of duplicating the
// string literals or the mount-path derivation logic.
package injection

import (
	"path/filepath"
	"strconv"
	"strings"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
)

// Injected env-var keys the create hook sets on a GPU-fractioning container.
//
// The two GPU-memory keys are a contract with the NVIDIA container toolkit's
// apply-cuda-memory-limits CDI hook: it reads them off the OCI spec (values in
// whole MiB) and applies them to the container's cgroup through NVML. The hook
// returns early when neither is present, so these exact spellings — singular,
// not plural — are what makes an injected limit take effect.
const (
	EnvGPUMemoryRequest = "NVIDIA_GPU_MEMORY_REQUEST"
	EnvGPUMemoryLimit   = "NVIDIA_GPU_MEMORY_LIMIT"
	EnvMPSPipeDirectory = "CUDA_MPS_PIPE_DIRECTORY"

	// EnvVisibleDevices selects the physical GPU(s) the NVIDIA container runtime
	// exposes to the container. The create hook sets it from the scheduler's
	// per-container device-assignment annotation (see
	// annotations.ParseVisibleDevices) so a
	// fractional GPU container — which does not request the nvidia.com/gpu
	// resource and is therefore skipped by the NVIDIA device plugin — still sees
	// exactly the GPU the scheduler picked. Unlike the keys above, it is injected
	// only when that annotation is present, so it is not part of the unconditional
	// injection contract.
	EnvVisibleDevices = "NVIDIA_VISIBLE_DEVICES"

	// EnvMPSActiveThreadPercentage caps the share of a device's SMs the
	// container's MPS clients may occupy. It is read by the CUDA runtime inside
	// the container when it connects to an MPS server, so it needs nothing from
	// the container toolkit or the device plugin — only MPS, which every
	// fractional container already goes through. It is what makes a "half a
	// GPU" request mean half the compute and not just half the memory. Injected
	// only when the scheduler recorded a compute portion (see
	// annotations.ParseComputePortion).
	EnvMPSActiveThreadPercentage = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

	// EnvMPSPinnedDeviceMemLimit caps device memory per GPU, enforced by the MPS
	// server itself.
	//
	// It deliberately duplicates the cap EnvGPUMemoryLimit asks the container
	// toolkit's apply-cuda-memory-limits CDI hook to apply, because the two have
	// different prerequisites: the hook needs a container toolkit new enough to
	// ship it (GPU Operator v26.7.1+), while this needs only MPS. On a cluster
	// that has one but not the other, the container still gets exactly one
	// enforced limit; on a cluster with both, they are set to the same value, so
	// whichever binds first binds at the right number.
	EnvMPSPinnedDeviceMemLimit = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"

	// EnvLDLibraryPath is the dynamic loader's search path, prepended with
	// ContainerNVMLShimDir so the shadow libnvidia-ml.so.1 staged on the host is
	// found ahead of the driver's own copy.
	//
	// The caps above are enforced at the CUDA layer by MPS, which NVML does not
	// go through: nvidia-smi inside a capped container still reports the whole
	// physical device and the memory every co-tenant is using. The shim answers
	// those queries with a container-scoped, capped view. It has to arrive via
	// the search path rather than LD_PRELOAD because nvidia-smi dlopens NVML and
	// resolves through dlsym on that handle, which interposition does not reach.
	//
	// Unlike every other key here this one is merged, not set — see
	// PrependLibraryPath and the EnforcementEnvKeys doc below.
	EnvLDLibraryPath = "LD_LIBRARY_PATH"

	// ContainerNVMLShimDir is the fixed in-container directory the host shim
	// directory (daemonpaths.NVMLShimDir) is bind-mounted at, read-only. Fixed
	// rather than an identity mount so the container's loader path never has to
	// name a host layout, and namespaced under /opt so it cannot collide with a
	// distribution library directory the image already populates.
	ContainerNVMLShimDir = "/opt/gpu-fractioning/lib"
)

// AllEnvKeys lists every env-var key the create hook may inject. Consumers that
// need to recognize injected env (e.g. the audit's presentEnv) should range over
// this single list so a newly added key is never missed in one place while being
// added in another.
var AllEnvKeys = []string{
	EnvGPUMemoryRequest,
	EnvGPUMemoryLimit,
	EnvMPSPipeDirectory,
	EnvVisibleDevices,
	EnvMPSActiveThreadPercentage,
	EnvMPSPinnedDeviceMemLimit,
	EnvLDLibraryPath,
}

// EnforcementEnvKeys lists the injected env vars that carry a limit rather than
// a location: the keys the create hook sets with remove-then-add semantics *on
// the containers where it injects them*, so a container cannot pre-set one in
// its own pod spec and keep that value. "I set
// CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 myself" must not be a way to opt out of
// the compute cap.
//
// The scoping is deliberate and the list is not a blanket removal set. A key
// the hook has no value for on a given container — CUDA_MPS_PINNED_DEVICE_MEM_LIMIT
// with no device assignment to enumerate, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
// with no compute portion annotated — is left exactly as the container declared
// it. Stripping a self-imposed limit and putting nothing back would turn a
// workload that voluntarily capped itself into an unbounded one, which is the
// opposite of what this package is for.
//
// LD_LIBRARY_PATH is deliberately absent, and the reason is stronger than "it
// is not a limit". A GPU container under the NVIDIA container runtime arrives
// already carrying LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64,
// set by the runtime itself (observed on a live pod). Remove-then-replace
// semantics would delete the runtime's own library search paths from every GPU
// container on the node, breaking library resolution for the workload — a far
// worse failure than the capped-reporting bug the shim fixes. The key is
// merged instead: see PrependLibraryPath.
var EnforcementEnvKeys = []string{
	EnvGPUMemoryRequest,
	EnvGPUMemoryLimit,
	EnvMPSActiveThreadPercentage,
	EnvMPSPinnedDeviceMemLimit,
}

// PinnedDeviceMemLimit renders the CUDA_MPS_PINNED_DEVICE_MEM_LIMIT value that
// caps each of the container's deviceCount GPUs at limitMiB MiB, in the
// "<device>=<size>" list form MPS parses:
//
//	PinnedDeviceMemLimit("40960", 2) → "0=40960M,1=40960M"
//
// Devices are named by the ordinals the container sees, not by UUID: fractiond
// also injects NVIDIA_VISIBLE_DEVICES for the same container, so its CUDA
// runtime enumerates exactly the scheduler's devices as 0..deviceCount-1,
// whatever their UUIDs or host indices are. The per-device limit is the
// per-GPU cap (a two-GPU fractional container gets limitMiB on each), matching
// how the scheduler derives the memory value from a per-GPU portion.
//
// Returns "" when there is nothing to enforce (no limit, or no known device
// count), which the caller treats as "do not inject".
func PinnedDeviceMemLimit(limitMiB string, deviceCount int) string {
	if limitMiB == "" || deviceCount <= 0 {
		return ""
	}

	var b strings.Builder
	for device := range deviceCount {
		if device > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(device))
		b.WriteByte('=')
		b.WriteString(limitMiB)
		b.WriteByte('M')
	}
	return b.String()
}

// DeviceCount returns how many GPUs the scheduler's device-assignment
// annotation names. The value is an opaque comma-separated list (UUIDs, CDI
// device names or indices — see annotations.ParseVisibleDevices), so only the
// count is interpreted here, never the entries themselves.
func DeviceCount(visibleDevices string) int {
	visibleDevices = strings.TrimSpace(visibleDevices)
	if visibleDevices == "" {
		return 0
	}

	count := 0
	for _, device := range strings.Split(visibleDevices, ",") {
		if strings.TrimSpace(device) != "" {
			count++
		}
	}
	return count
}

// PrependLibraryPath merges dir into a colon-separated LD_LIBRARY_PATH,
// returning the new value and whether it differs from existing.
//
//	PrependLibraryPath("/usr/local/nvidia/lib:/usr/local/nvidia/lib64", "/opt/gpu-fractioning/lib")
//	  → "/opt/gpu-fractioning/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64", true
//
// Merge rather than replace, because the variable is a list and the entries
// already in it are load-bearing: the NVIDIA container runtime puts its own
// driver library directories there, so overwriting would cost the workload the
// paths its CUDA stack resolves through. Prepend rather than append, because
// position is the entire mechanism — the shim only shadows the real NVML if the
// loader reaches it first.
//
// changed=false when dir is already the leading entry, so a container that has
// been through this once (or whose image happens to lead with the same path)
// produces no adjustment entry at all rather than a duplicated prefix that
// grows on every pass.
func PrependLibraryPath(existing, dir string) (value string, changed bool) {
	if dir == "" {
		return existing, false
	}
	if existing == "" {
		return dir, true
	}

	entries := strings.Split(existing, ":")
	if entries[0] == dir {
		return existing, false
	}

	// A later occurrence is dropped rather than left in place: keeping it would
	// leave the same directory listed twice, and the loader would search it
	// twice on every unresolved symbol.
	kept := make([]string, 0, len(entries)+1)
	kept = append(kept, dir)
	for _, entry := range entries {
		if entry == dir {
			continue
		}
		kept = append(kept, entry)
	}
	return strings.Join(kept, ":"), true
}

// MPSPipeMount returns the MPS pipe bind-mount source (host path) and
// destination (in-container path) for the given host pipe directory and
// compute mode. Both the fractiond create hook (internal/plugin.go
// buildAdjustment) and its audit detector (internal/audit) call this so the
// expected mount can never drift between the two.
func MPSPipeMount(mpsPipeDir string, mode annotations.ComputeMode) (source, destination string) {
	if mode == annotations.ComputeModeSMSharing {
		// sm-sharing routes the container to the shared MPS server's socket
		// on the host, mounted at a fixed in-container path that is
		// decoupled from that host-side server/namespace naming.
		return filepath.Join(mpsPipeDir, configuration.SharedMPSSocketPath), configuration.ContainerMPSPipeDirectory
	}
	// time-slicing (the default): identity mount, unchanged from today.
	return mpsPipeDir, mpsPipeDir
}
