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
