// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package harness

import "fmt"

// Shared identifiers the e2e suites assert against. These mirror values in the
// operator/daemon source (operator/internal/common/daemonmgr,
// fractioning-manager/common/configuration, fractioning-manager/fractiond/internal/
// injection). Those packages are internal to their own modules and can't be
// imported from test/e2e, so the constants are redeclared here and kept in sync
// with the operator/fractioning-manager source.
//
// Only genuinely cross-suite identifiers live here; suite-specific ones (e.g.
// the operator's per-daemon status reasons and DaemonSet container names) stay
// in that suite's package.
const (
	// Node condition the operator sets on each targeted node (daemonmgr.NodeConditionType).
	NodeConditionType = "gpu-fractioning.nvidia.com/Ready"

	// CR status condition types (daemonmgr: "<daemon>Ready" + ConditionReady).
	CondFractiondReady = "FractiondReady"
	CondMpsdReady      = "MpsdReady"
	CondReady          = "Ready"

	// fractiond annotation prefix (configuration.DefaultAnnotationPrefix).
	AnnotationPrefix = "nvidia.com/container."

	// Env vars fractiond injects into a GPU-fractioning container (injection pkg).
	EnvGPUMemRequest = "NVIDIA_GPU_MEMORY_REQUEST"
	EnvGPUMemLimit   = "NVIDIA_GPU_MEMORY_LIMIT"
	EnvMPSPipeDir    = "CUDA_MPS_PIPE_DIRECTORY"
	// EnvVisibleDevices is set from the scheduler's per-container device
	// assignment (…gpus.devices) so a fractional pod sees exactly the GPU(s) the
	// scheduler picked (injection.EnvVisibleDevices).
	EnvVisibleDevices = "NVIDIA_VISIBLE_DEVICES"

	// MPS pipe dir (configuration.DefaultMPSPipeDirectory).
	MPSPipeDir = "/run/nvidia-mps"

	// Where an sm-sharing container's MPS pipe dir is mounted, and the host
	// directory it comes from — the shared MPS server's "default" namespace
	// (configuration.ContainerMPSPipeDirectory, SharedMPSSocketPath). A
	// time-slicing container instead gets MPSPipeDir mounted at itself.
	ContainerMPSPipeDir = "/tmp/nvidia-mps"
	SharedMPSHostDir    = MPSPipeDir + "/shared/default"

	// Values of the per-container gpu-compute.mode annotation
	// (annotations.ComputeMode*). Absent means TimeSlicing.
	ComputeModeTimeSlicing = "time-slicing"
	ComputeModeSMSharing   = "sm-sharing"

	// Component label values (app.kubernetes.io/component) on each managed DS.
	ComponentFractiond = "fractiond"
	ComponentMpsd      = "mpsd"

	// Management labels the operator stamps on the DaemonSets it owns
	// (daemonmgr LabelManagedBy/ManagedByValue/LabelComponent).
	LabelManagedBy = "app.kubernetes.io/managed-by"
	LabelComponent = "app.kubernetes.io/component"
	ManagedByValue = "gpu-fractioning"

	// Namespace for the suites' own workload pods (kept off any scheduler
	// project namespace this cluster may not have).
	WorkloadNamespace = "operator-e2e"

	// The GPU-fractioning container name used by the annotated test workload.
	WorkloadContainer = "cuda"

	// GPU-memory request/limit the fractiond-injection cases stamp onto the
	// annotated workload (MiB, no unit suffix). These are arbitrary test
	// inputs, not a cluster property: fractiond parses them as k8s Quantities
	// and re-emits the integer MiB value NVIDIA consumes (2048Mi -> 2048,
	// 4096Mi -> 4096).
	MemRequestMiB = "2048"
	MemLimitMiB   = "4096"

	// dsNamePrefix + "-" + component is the DaemonSet name the operator builds.
	dsNamePrefix = "gpu-fractioning"
)

// DSName returns the DaemonSet name for a component, e.g. "gpu-fractioning-mpsd".
func DSName(component string) string { return fmt.Sprintf("%s-%s", dsNamePrefix, component) }

// ComponentSelector returns the label selector matching a component's pods.
func ComponentSelector(component string) string {
	return fmt.Sprintf("%s=%s,%s=%s", LabelManagedBy, ManagedByValue, LabelComponent, component)
}
