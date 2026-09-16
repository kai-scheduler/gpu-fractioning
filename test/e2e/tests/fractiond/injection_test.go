// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package fractiond

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"

	"github.com/kai-scheduler/kai-gpu-fractioning/test/e2e/harness"
	"github.com/kai-scheduler/kai-gpu-fractioning/test/e2e/k8s/daemonset"
	"github.com/kai-scheduler/kai-gpu-fractioning/test/e2e/waiter"
	"github.com/kai-scheduler/kai-gpu-fractioning/test/e2e/workload"
)

// containerMpsd is the mpsd DaemonSet's container name, needed by the sm-sharing
// case to reach the node's MPS pipe directory. The operator suite declares its
// own copy; per harness/consts.go, container names stay suite-local.
const containerMpsd = "mpsd"

// InjectsMemoryEnv — fractiond injects the GPU-memory env vars and MPS
// pipe dir into an annotated container (request/limit normalized to MiB).
func caseInjectsMemoryEnv(ctx context.Context, t *testing.T) {
	pod := h.ApplyRunningWorkload(ctx, t, "d1-inject", h.DefaultWorkloadAnnotations())
	env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)

	wantReq := harness.MemRequestMiB
	wantLim := harness.MemLimitMiB
	if env[harness.EnvGPUMemRequest] != wantReq {
		t.Errorf("%s = %q, want %q", harness.EnvGPUMemRequest, env[harness.EnvGPUMemRequest], wantReq)
	}
	if env[harness.EnvGPUMemLimit] != wantLim {
		t.Errorf("%s = %q, want %q", harness.EnvGPUMemLimit, env[harness.EnvGPUMemLimit], wantLim)
	}
	if env[harness.EnvMPSPipeDir] != harness.MPSPipeDir {
		t.Errorf("%s = %q, want %q", harness.EnvMPSPipeDir, env[harness.EnvMPSPipeDir], harness.MPSPipeDir)
	}
}

// MountsMPSPipeDir — the MPS pipe directory is bind-mounted into an
// annotated container.
func caseMountsMPSPipeDir(ctx context.Context, t *testing.T) {
	pod := h.ApplyRunningWorkload(ctx, t, "d2-mount", h.DefaultWorkloadAnnotations())
	out, err := h.Exec(ctx, pod, harness.WorkloadContainer, "sh", "-c", fmt.Sprintf("test -d %s && echo yes || echo no", harness.MPSPipeDir))
	if err != nil {
		t.Fatalf("exec dir check: %v", err)
	}
	if out != "yes" {
		t.Errorf("MPS pipe dir %s not mounted into container", harness.MPSPipeDir)
	}
}

// SMSharingRoutesToSharedServer — a container annotated
// gpu-compute.mode: sm-sharing gets the *shared* MPS server's socket directory
// instead of the node's default one: CUDA_MPS_PIPE_DIRECTORY moves to
// /tmp/nvidia-mps, and what is mounted there is the host's
// /run/nvidia-mps/shared/default.
//
// The routing claim is proved by identity, not by the path string: a sentinel
// file created from the mpsd pod (whose /run/nvidia-mps is a hostPath mount of
// the node's) must appear inside the workload container. Asserting only that
// /tmp/nvidia-mps exists and holds a "control" socket would pass just as well if
// fractiond had mounted the wrong server's directory.
func caseSMSharingRoutesToSharedServer(ctx context.Context, t *testing.T) {
	pod := h.ApplyRunningWorkload(ctx, t, "d7-sm-sharing",
		h.ComputeModeAnnotations(harness.ComputeModeSMSharing))

	env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
	if env[harness.EnvMPSPipeDir] != harness.ContainerMPSPipeDir {
		t.Errorf("%s = %q, want %q", harness.EnvMPSPipeDir,
			env[harness.EnvMPSPipeDir], harness.ContainerMPSPipeDir)
	}

	// mpsd on the *same* node: the workload is pinned to a GPU node, but the
	// cluster has several, and only the co-located pod shares the directory.
	mpsdPod, ok := h.PodOnNode(ctx, t, harness.ComponentMpsd, pod.Spec.NodeName)
	if !ok {
		t.Fatalf("no mpsd pod on node %s (cannot verify the mount source)", pod.Spec.NodeName)
	}

	sentinel := fmt.Sprintf("e2e-sm-sharing-%d", time.Now().UnixNano())
	hostPath := harness.SharedMPSHostDir + "/" + sentinel
	if _, err := h.Exec(ctx, mpsdPod, containerMpsd, "sh", "-c", "touch "+hostPath); err != nil {
		t.Fatalf("create sentinel %s from mpsd: %v", hostPath, err)
	}
	t.Cleanup(func() {
		_, _ = h.Exec(context.Background(), mpsdPod, containerMpsd, "sh", "-c", "rm -f "+hostPath)
	})

	inContainer := harness.ContainerMPSPipeDir + "/" + sentinel
	out, err := h.Exec(ctx, pod, harness.WorkloadContainer, "sh", "-c",
		fmt.Sprintf("test -e %s && echo yes || echo no", inContainer))
	if err != nil {
		t.Fatalf("exec sentinel check: %v", err)
	}
	if out != "yes" {
		t.Errorf("%s is not the host's %s: sentinel %s created there is not visible in the container",
			harness.ContainerMPSPipeDir, harness.SharedMPSHostDir, sentinel)
	}
}

// TimeSlicingKeepsDefaultMount — an explicit gpu-compute.mode: time-slicing
// container keeps today's identity mount of the node's default pipe directory.
// SkipsUnannotatedContainer and InjectsMemoryEnv cover the *absent* annotation;
// this pins the explicit value, so a future change to the default can't silently
// reroute containers that asked for time-slicing by name.
func caseTimeSlicingKeepsDefaultMount(ctx context.Context, t *testing.T) {
	pod := h.ApplyRunningWorkload(ctx, t, "d8-time-slicing",
		h.ComputeModeAnnotations(harness.ComputeModeTimeSlicing))

	env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
	if env[harness.EnvMPSPipeDir] != harness.MPSPipeDir {
		t.Errorf("%s = %q, want %q", harness.EnvMPSPipeDir, env[harness.EnvMPSPipeDir], harness.MPSPipeDir)
	}
	// The shared server's directory must not be what got mounted: its socket
	// would be visible at the default path if the two were confused.
	out, err := h.Exec(ctx, pod, harness.WorkloadContainer, "sh", "-c",
		fmt.Sprintf("test -d %s && echo yes || echo no", harness.ContainerMPSPipeDir))
	if err != nil {
		t.Fatalf("exec dir check: %v", err)
	}
	if out == "yes" {
		t.Errorf("time-slicing container has %s mounted; that path is sm-sharing only",
			harness.ContainerMPSPipeDir)
	}
}

// InvalidComputeModeBlocks — an unrecognized gpu-compute.mode value is a hard
// error, not a silent fallback to time-slicing: fail-closed blocks the
// container. A workload that asked for a mode it did not get would otherwise run
// with different isolation than requested, with nothing to indicate it.
func caseInvalidComputeModeBlocks(ctx context.Context, t *testing.T) {
	name := "d9-bad-mode"
	annotations := h.ComputeModeAnnotations("not-a-mode")
	createBlockedWorkload(ctx, t, name, annotations)
	if reason := waitContainerBlocked(ctx, t, name); reason == "" {
		t.Errorf("pod %s: expected a container create error for an invalid compute mode, none observed", name)
	} else {
		t.Logf("pod %s blocked as expected: %s", name, reason)
	}
}

// InjectsVisibleDevices — fractiond promotes the scheduler's per-container device
// assignment (…gpus.devices) to NVIDIA_VISIBLE_DEVICES. A fractional container
// does not request the nvidia.com/gpu resource, so the NVIDIA device plugin never
// sets that env var; fractiond injects it so the container sees exactly the GPU(s)
// the scheduler picked. The value is a scheduler/runtime contract passed through
// verbatim, so we assert both a single-GPU assignment and a multi-GPU
// comma-separated list survive intact (the multi-GPU assignment case). The
// device handles are opaque to fractiond; nothing on the fake cluster interprets
// them (the workload uses the default runtime), so any token round-trips.
func caseInjectsVisibleDevices(ctx context.Context, t *testing.T) {
	assertVisible := func(t *testing.T, name, assignment string) {
		// The gpus.devices assignment is only promoted when the container is a
		// GPU-fractioning container, so keep the standard gpu-memory annotations too.
		annotations := h.DefaultWorkloadAnnotations()
		annotations[h.DevicesAnnotationKey(harness.WorkloadContainer)] = assignment
		pod := h.ApplyRunningWorkload(ctx, t, name, annotations)
		env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
		if env[harness.EnvVisibleDevices] != assignment {
			t.Errorf("%s = %q, want %q", harness.EnvVisibleDevices, env[harness.EnvVisibleDevices], assignment)
		}
	}
	t.Run("single-gpu", func(t *testing.T) {
		assertVisible(t, "inject-visible-single", "GPU-e2e-00000000")
	})
	t.Run("multi-gpu", func(t *testing.T) {
		assertVisible(t, "inject-visible-multi", "GPU-e2e-00000000,GPU-e2e-11111111")
	})
}

// MultiContainerInjection — a pod with two annotated containers gets each
// container's own GPU-memory config injected, keyed by container name, with no
// cross-container leakage (fractiond looks up annotations per container, so the
// two containers must end up with different injected values).
func caseMultiContainerInjection(ctx context.Context, t *testing.T) {
	const (
		ctrA = "cuda-a"
		ctrB = "cuda-b"
		memA = "1024" // MiB; request==limit so both injected vars carry this
		memB = "3072"
	)
	annotations := map[string]string{}
	for k, v := range workload.FractionalAnnotations(ctrA, memA, memA) {
		annotations[k] = v
	}
	for k, v := range workload.FractionalAnnotations(ctrB, memB, memB) {
		annotations[k] = v
	}
	pod := applyRunningMultiContainer(ctx, t, "inject-multi-container", []string{ctrA, ctrB}, annotations)

	wantA := memA
	wantB := memB

	envA := h.GetPodEnv(ctx, t, pod, ctrA)
	if envA[harness.EnvGPUMemLimit] != wantA || envA[harness.EnvGPUMemRequest] != wantA {
		t.Errorf("container %s: req=%q lim=%q, want both %q", ctrA,
			envA[harness.EnvGPUMemRequest], envA[harness.EnvGPUMemLimit], wantA)
	}
	// Isolation: container A must not receive container B's config.
	if envA[harness.EnvGPUMemLimit] == wantB {
		t.Errorf("container %s leaked container %s's limit %q", ctrA, ctrB, wantB)
	}

	envB := h.GetPodEnv(ctx, t, pod, ctrB)
	if envB[harness.EnvGPUMemLimit] != wantB || envB[harness.EnvGPUMemRequest] != wantB {
		t.Errorf("container %s: req=%q lim=%q, want both %q", ctrB,
			envB[harness.EnvGPUMemRequest], envB[harness.EnvGPUMemLimit], wantB)
	}
}

// SkipsUnannotatedContainer — a container with no gpu-fractioning
// annotation is left untouched (no injection).
func caseSkipsUnannotatedContainer(ctx context.Context, t *testing.T) {
	pod := h.ApplyRunningWorkload(ctx, t, "d3-plain", map[string]string{}) // empty ⇒ no annotations
	env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
	for _, k := range []string{harness.EnvGPUMemRequest, harness.EnvGPUMemLimit, harness.EnvMPSPipeDir} {
		if _, ok := env[k]; ok {
			t.Errorf("unannotated container has injected env %s=%q", k, env[k])
		}
	}
}

// RequestOrLimitOnly — request-only and limit-only annotations both end
// up with request==limit (ApplyDefaults), so the cap is enforced and the request
// is populated.
func caseRequestOrLimitOnly(ctx context.Context, t *testing.T) {
	t.Run("request-only", func(t *testing.T) {
		want := harness.MemRequestMiB
		pod := h.ApplyRunningWorkload(ctx, t, "d4-req", map[string]string{
			h.AnnKey(harness.WorkloadContainer, "request"): harness.MemRequestMiB + "Mi",
		})
		env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
		if env[harness.EnvGPUMemRequest] != want || env[harness.EnvGPUMemLimit] != want {
			t.Errorf("request-only: req=%q lim=%q, want both %q", env[harness.EnvGPUMemRequest], env[harness.EnvGPUMemLimit], want)
		}
	})
	t.Run("limit-only", func(t *testing.T) {
		want := harness.MemLimitMiB
		pod := h.ApplyRunningWorkload(ctx, t, "d4-lim", map[string]string{
			h.AnnKey(harness.WorkloadContainer, "limit"): harness.MemLimitMiB + "Mi",
		})
		env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
		if env[harness.EnvGPUMemRequest] != want || env[harness.EnvGPUMemLimit] != want {
			t.Errorf("limit-only: req=%q lim=%q, want both %q", env[harness.EnvGPUMemRequest], env[harness.EnvGPUMemLimit], want)
		}
	})
}

// FailClosedBlocks — fail-closed (default): a malformed annotation
// makes fractiond block container creation, so the pod never reaches Running.
func caseFailClosedBlocks(ctx context.Context, t *testing.T) {
	name := "d5-failclosed"
	createBlockedWorkload(ctx, t, name, map[string]string{
		h.AnnKey(harness.WorkloadContainer, "request"): "not-a-quantity",
	})
	if reason := waitContainerBlocked(ctx, t, name); reason == "" {
		t.Errorf("pod %s: expected a container create error (fail-closed), none observed", name)
	} else {
		t.Logf("pod %s blocked as expected: %s", name, reason)
	}
}

// FailOpenSkips — fail-open: with fractioningAgent.failOpen=true a malformed
// annotation is skipped, so the container runs but carries no injected env.
func caseFailOpenSkips(ctx context.Context, t *testing.T) {
	dsName := harness.DSName(harness.ComponentFractiond)
	// Capture the fractiond DaemonSet generation before the patch. WaitRolledOut
	// alone can match the already-converged pre-change DaemonSet in the window
	// before the operator reconciles the failOpen change, so we'd apply the
	// workload against the old (fail-closed) fractiond and flake. WaitRolledOutAfter
	// waits for the *new* generation to roll out.
	beforeGen := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation

	patch := []byte(`{"spec":{"fractioningAgent":{"failOpen":true}}}`)
	revert := []byte(`{"spec":{"fractioningAgent":{"failOpen":null}}}`)
	h.WithCRConfig(ctx, t, patch, revert, func() {
		// The failOpen change re-rolls the fractiond DaemonSet; wait for the new
		// generation to finish rolling out.
		if _, err := daemonset.WaitRolledOutAfter(ctx, h.Client(), h.NS(), dsName, beforeGen, h.RolloutTimeout(), h.PollInterval()); err != nil {
			t.Fatalf("fractiond rollout after failOpen: %v", err)
		}
		pod := h.ApplyRunningWorkload(ctx, t, "d6-failopen", map[string]string{
			h.AnnKey(harness.WorkloadContainer, "request"): "not-a-quantity",
		})
		env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
		for _, k := range []string{harness.EnvGPUMemRequest, harness.EnvGPUMemLimit, harness.EnvMPSPipeDir} {
			if _, ok := env[k]; ok {
				t.Errorf("fail-open with malformed annotation still injected %s=%q", k, env[k])
			}
		}
	})
}

// applyRunningMultiContainer creates a pod with several busybox containers on a
// GPU node, carrying the given annotations verbatim, waits for it to reach
// Running, and registers cleanup. The shared workload.Apply builds only
// single-container pods, so the multi-container case builds its own.
func applyRunningMultiContainer(ctx context.Context, t *testing.T, name string, containerNames []string, annotations map[string]string) *corev1.Pod {
	t.Helper()
	if err := workload.EnsureNamespace(ctx, h.Client(), harness.WorkloadNamespace); err != nil {
		t.Fatalf("ensure namespace: %v", err)
	}
	if err := workload.Delete(ctx, h.Client(), harness.WorkloadNamespace, name); err != nil {
		t.Fatalf("pre-clean %s: %v", name, err)
	}
	pod := &corev1.Pod{}
	pod.Name = name
	pod.Namespace = harness.WorkloadNamespace
	pod.Annotations = annotations
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	pod.Spec.NodeSelector = h.GPUSelectorMap(t)
	for _, cn := range containerNames {
		pod.Spec.Containers = append(pod.Spec.Containers, corev1.Container{
			Name:    cn,
			Image:   workload.DefaultImage,
			Command: []string{"sh", "-c", "sleep 86400 & wait"},
		})
	}
	if err := h.Client().Ctrl.Create(ctx, pod); err != nil {
		t.Fatalf("create multi-container workload %s: %v", name, err)
	}
	t.Cleanup(func() { _ = workload.Delete(context.Background(), h.Client(), harness.WorkloadNamespace, name) })
	p, err := workload.WaitRunning(ctx, h.Client(), harness.WorkloadNamespace, name,
		h.Client().Config.PodReadyTimeout, h.Client().Config.PollInterval)
	if err != nil {
		t.Fatalf("multi-container workload %s did not reach Running: %v", name, err)
	}
	return p
}

// ── FailClosedBlocks helpers ─────────────────────────────────────────────────

// createBlockedWorkload creates an annotated pod WITHOUT waiting for Running (it
// is expected to be blocked at container creation). Registers cleanup.
func createBlockedWorkload(ctx context.Context, t *testing.T, name string, annotations map[string]string) {
	t.Helper()
	if err := workload.EnsureNamespace(ctx, h.Client(), harness.WorkloadNamespace); err != nil {
		t.Fatalf("ensure namespace: %v", err)
	}
	if err := workload.Delete(ctx, h.Client(), harness.WorkloadNamespace, name); err != nil {
		t.Fatalf("pre-clean %s: %v", name, err)
	}
	pod := &corev1.Pod{}
	pod.Name = name
	pod.Namespace = harness.WorkloadNamespace
	pod.Annotations = annotations
	pod.Spec.RestartPolicy = corev1.RestartPolicyNever
	pod.Spec.NodeSelector = h.GPUSelectorMap(t)
	pod.Spec.Containers = []corev1.Container{{
		Name:    harness.WorkloadContainer,
		Image:   workload.DefaultImage,
		Command: []string{"sh", "-c", "sleep 86400 & wait"},
	}}
	if err := h.Client().Ctrl.Create(ctx, pod); err != nil {
		t.Fatalf("create blocked workload %s: %v", name, err)
	}
	t.Cleanup(func() { _ = workload.Delete(context.Background(), h.Client(), harness.WorkloadNamespace, name) })
}

// waitContainerBlocked polls until the pod shows a container Waiting state with a
// create/start error reason (fail-closed), returning that reason. It fails the
// test if the pod instead reaches Running.
func waitContainerBlocked(ctx context.Context, t *testing.T, name string) string {
	t.Helper()
	var reason string
	err := waiter.PollUntil(ctx, h.CondTimeout(), h.PollInterval(), fmt.Sprintf("pod %s to be blocked at container create", name),
		func(ctx context.Context) (bool, error) {
			p := h.GetPod(ctx, t, harness.WorkloadNamespace, name)
			if p.Status.Phase == corev1.PodRunning {
				return false, fmt.Errorf("pod %s unexpectedly Running (fail-closed did not block)", name)
			}
			for _, cs := range p.Status.ContainerStatuses {
				if cs.State.Waiting != nil && cs.State.Waiting.Reason != "" &&
					cs.State.Waiting.Reason != "ContainerCreating" && cs.State.Waiting.Reason != "PodInitializing" {
					reason = cs.State.Waiting.Reason
					return true, nil
				}
			}
			return false, nil
		})
	if err != nil {
		t.Fatalf("waiting for fail-closed block: %v", err)
	}
	return reason
}
