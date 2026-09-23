// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package audit inspects the containers reports on an NRI
// (re)connect (the Synchronize snapshot) and finds containers that are running
// without the fractioning setup the CreateContainer hook is supposed to inject.
// Such containers slipped through while the fractioning agent was down: they
// hold a GPU share but no memory limit, so they can harm neighbours on the same
// GPU. The remediation path stops them, so kubelet recreates them through a
// healthy CreateContainer hook.
package audit

import (
	"log/slog"
	"strings"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/injection"
)

// violator describes a running container that belongs to a GPU-fractioning pod but
// is missing the fractioning setup the CreateContainer hook would have injected. It
// is produced by detector and consumed by the remediation path (same package).
type violator struct {
	containerID string
	container   string
	pod         string
	namespace   string
	missing     []string // missing lists the expected-but-absent pieces
}

// detector finds GPU-fractioning containers that are running without the expected
// injection.
// A container qualifies as a fractioning container iff annotations.ParseGPUMemoryAnnotations
// yields a non-empty config for it — the same check the create hook uses. The
// detector then reports any expected env/mount the live container lacks.
type detector struct {
	// annotationPrefix is the leading GPU-fractioning annotation prefix, e.g.
	// "nvidia.com/container." — must match the fractiond plugin's configured prefix.
	annotationPrefix string
	// mpsPipeDirectory is the base host-side MPS pipe directory — must match
	// the fractiond plugin's configured value.
	mpsPipeDirectory string
	// smSharingEnabled is the cluster's installation-time sm-sharing chicken
	// bit — must match the fractiond plugin's configured value, so a disabled
	// feature can't produce a false violation for a container the plugin
	// itself would have rejected/downgraded.
	smSharingEnabled bool
	// failOpen is the create hook's fail-open policy — must match the fractiond
	// plugin's configured value. It decides what an unusable compute-mode
	// annotation means here, exactly as it does at creation: under fail-open the
	// container would have been created with the time-slicing default, so it is
	// still owed its memory injection and must be inspected.
	failOpen bool
	// log is used for skip diagnostics; defaults to slog.Default().
	log *slog.Logger
}

// violators scans an NRI Synchronize snapshot and returns one entry per
// GPU-fractioning container that is running without the expected injection. Each
// container is matched to its pod sandbox by PodSandboxId; containers with no
// matching pod, non-enforceable states, or no fractioning annotations are skipped.
func (d detector) violators(pods []*api.PodSandbox, containers []*api.Container) []violator {
	podsByID := podSandboxLookup(pods)

	var out []violator
	for _, container := range containers {
		if v, ok := d.check(podsByID[container.GetPodSandboxId()], container); ok {
			out = append(out, v)
		}
	}
	return out
}

// check evaluates a single container against its pod. It returns ok=false for
// anything that must not be stopped: nil/pod-less containers, non-running
// states, non-fractioning containers, containers with unparseable annotations, or
// containers that already carry the full expected injection.
func (d detector) check(pod *api.PodSandbox, container *api.Container) (violator, bool) {
	if container == nil || container.GetId() == "" || pod == nil {
		return violator{}, false
	}

	// Only running/created containers can be meaningfully remediated. A stopped
	// or unknown container has nothing to enforce.
	switch container.GetState() {
	case api.ContainerState_CONTAINER_CREATED, api.ContainerState_CONTAINER_RUNNING:
	default:
		return violator{}, false
	}

	cfg, err := annotations.ParseGPUMemoryAnnotations(pod.GetAnnotations(), container.GetName(), d.annotationPrefix)
	if err != nil {
		// Malformed annotations: with fail-closed creation this container would
		// not exist, and recreating it would only hit the same parse error — skip
		// rather than risk a container we cannot reason about. Log the raw
		// annotation values so the offending input is visible, not just the error.
		requestKey := annotations.RequestAnnotationKey(d.annotationPrefix, container.GetName())
		limitKey := annotations.LimitAnnotationKey(d.annotationPrefix, container.GetName())
		ann := pod.GetAnnotations()
		d.logger().Warn("audit: skipping container with unparseable GPU memory annotations",
			"container", container.GetName(),
			"pod", pod.GetName(),
			"requestAnnotation", ann[requestKey],
			"limitAnnotation", ann[limitKey],
			"error", err,
		)
		return violator{}, false
	}
	if cfg.IsEmpty() {
		// Not a GPU-fractioning container — never touch it.
		return violator{}, false
	}
	// Mirror buildAdjustment: a request-only or limit-only container has the
	// missing value defaulted from the other, so both env vars are expected.
	cfg = cfg.ApplyDefaults()

	// Mirror buildAdjustment, whose handling of an unusable compute-mode
	// annotation depends on the same fail-open policy:
	//
	// fail-closed — creation was blocked, so a container running with this
	// annotation was not injected by us and recreating it would hit the same
	// parse error. Skip it.
	//
	// fail-open — creation continued with the time-slicing default, so the
	// container is owed the full memory injection. Skipping here would let a
	// mode typo suppress retroactive enforcement of valid memory annotations,
	// which is the very case this audit exists for. Fall back to the same
	// default and keep inspecting.
	computeMode, err := annotations.ParseComputeMode(pod.GetAnnotations(), container.GetName(), d.annotationPrefix, d.smSharingEnabled)
	if err != nil {
		if !d.failOpen {
			d.logger().Warn("audit: skipping container with unparseable GPU compute mode annotation",
				"container", container.GetName(),
				"pod", pod.GetName(),
				"error", err,
			)
			return violator{}, false
		}
		d.logger().Warn("audit: unparseable GPU compute mode annotation, auditing against the time-slicing default (fail-open)",
			"container", container.GetName(),
			"pod", pod.GetName(),
			"error", err,
		)
		computeMode = annotations.ComputeModeTimeSlicing
	}

	visibleDevices := annotations.ParseVisibleDevices(pod.GetAnnotations(), container.GetName(), d.annotationPrefix)
	missing := d.missingInjection(cfg, visibleDevices, computeMode, container)
	if len(missing) == 0 {
		return violator{}, false
	}

	return violator{
		containerID: container.GetId(),
		container:   container.GetName(),
		pod:         pod.GetName(),
		namespace:   pod.GetNamespace(),
		missing:     missing,
	}, true
}

// missingInjection compares the expected injection (derived from cfg, the
// pod's device assignment, and its compute mode exactly as buildAdjustment
// would) against the live container's env and mounts, returning the expected
// pieces that are absent. Presence-based, not value-equality: a mis-set value
// is out of scope and would risk false positives from formatting differences.
// visibleDevices is the pod's GPU device assignment ("" when unassigned);
// NVIDIA_VISIBLE_DEVICES is expected only when it is set, matching
// buildAdjustment's conditional injection.
func (d detector) missingInjection(cfg annotations.GPUMemoryConfig, visibleDevices string, computeMode annotations.ComputeMode, container *api.Container) []string {
	env := presentEnv(container.GetEnv())
	expectedMountSource, expectedMountDestination := injection.MPSPipeMount(d.mpsPipeDirectory, computeMode)

	var missing []string
	// CUDA_MPS_PIPE_DIRECTORY is always injected for a fractioning container.
	if !env[injection.EnvMPSPipeDirectory] {
		missing = append(missing, "env:"+injection.EnvMPSPipeDirectory)
	}
	if cfg.Limit != "" && !env[injection.EnvGPUMemoryLimit] {
		missing = append(missing, "env:"+injection.EnvGPUMemoryLimit)
	}
	if cfg.Request != "" && !env[injection.EnvGPUMemoryRequest] {
		missing = append(missing, "env:"+injection.EnvGPUMemoryRequest)
	}
	if visibleDevices != "" && !env[injection.EnvVisibleDevices] {
		missing = append(missing, "env:"+injection.EnvVisibleDevices)
	}
	if !hasMount(container.GetMounts(), expectedMountSource, expectedMountDestination) {
		missing = append(missing, "mount:"+expectedMountSource+":"+expectedMountDestination)
	}
	return missing
}

func (d detector) logger() *slog.Logger {
	if d.log != nil {
		return d.log
	}
	return slog.Default()
}

// presentEnv scans the container's "KEY=VALUE" env list once and reports which
// of the injected keys (and only those) are present. A token without '=' is
// treated as a bare key.
func presentEnv(env []string) map[string]bool {
	injected := make(map[string]struct{}, len(injection.AllEnvKeys))
	for _, k := range injection.AllEnvKeys {
		injected[k] = struct{}{}
	}

	present := make(map[string]bool, len(injection.AllEnvKeys))
	for _, kv := range env {
		k := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			k = kv[:i]
		}
		if _, ok := injected[k]; ok {
			present[k] = true
		}
	}
	return present
}

// hasMount reports whether any mount has exactly the given source and
// destination — mirroring buildAdjustment's mount exactly, rather than just
// its in-container destination.
func hasMount(mounts []*api.Mount, source, destination string) bool {
	for _, m := range mounts {
		if m.GetSource() == source && m.GetDestination() == destination {
			return true
		}
	}
	return false
}

// podSandboxLookup indexes pod sandboxes by their NRI id for O(1) matching
// against a container's PodSandboxId.
func podSandboxLookup(pods []*api.PodSandbox) map[string]*api.PodSandbox {
	out := make(map[string]*api.PodSandbox, len(pods))
	for _, pod := range pods {
		if pod.GetId() == "" {
			continue
		}
		out[pod.GetId()] = pod
	}
	return out
}
