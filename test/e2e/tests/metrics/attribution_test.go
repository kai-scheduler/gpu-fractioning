// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/nvmlmock"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// TestE2E_SingleFractionalPodAttribution verifies that a pod carrying a fractional GPU
// annotation is attributed under the namespace/pod/pod_uid it actually has, on
// the GPU whose UUID we pinned in nvml-mock.
//
// Flow: create the annotated pod, resolve its container's host PID, tell
// nvml-mock that PID is a GPU process on a known device UUID, then assert the
// plugin attributes the pod's memory and SM-utilization series to it. See the
// NOTE at the assertion below on why the exact memory value isn't asserted.
func TestE2E_SingleFractionalPodAttribution(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}
	// Two nvml-mock + plugin DaemonSet rollouts (inside SetProcesses) plus a
	// collection cycle — give it well over the single-rollout budget.
	ctx, cancel := context.WithTimeout(context.Background(), attributionTestTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)

	spec := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc1-single-fractional-pod",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", "2048", "2048"),
		NodeSelector:  map[string]string{"kubernetes.io/hostname": targetNode},
	}

	_ = workload.Delete(ctx, c, spec.Namespace, spec.Name)
	pod, err := workload.Apply(ctx, c, spec)
	if err != nil {
		t.Fatalf("create fractional pod: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, spec.Namespace, spec.Name); err != nil {
			t.Errorf("cleanup pod %s/%s: %v", spec.Namespace, spec.Name, err)
		}
	})

	// Resolve the container's host-namespace PID and make NVML report it as a
	// GPU process using wantBytes of memory on gpuUUID.
	marker := workload.DefaultMarker(spec.Namespace, spec.Name)
	pid, err := nvmlmock.HostPID(ctx, c, targetNode, marker)
	if err != nil {
		t.Fatalf("resolve host PID for %s/%s: %v", spec.Namespace, spec.Name, err)
	}
	t.Logf("resolved host PID for %s/%s on node %s: pid=%d", spec.Namespace, spec.Name, targetNode, pid)

	const (
		gpuUUID       = nvmlmock.Device0UUID
		wantMemoryMiB = uint64(8 * 1024) // 8 GiB in MiB
		wantBytes     = wantMemoryMiB * 1024 * 1024
	)
	procs := []nvmlmock.Proc{{UUID: gpuUUID, PID: pid, UsedMemoryMiB: wantMemoryMiB}}
	t.Logf("configuring nvml-mock: pid=%d gpu_uuid=%s used_memory_mib=%d", pid, gpuUUID, wantMemoryMiB)
	if err := setProcesses(ctx, c, nvmlmock.A100, procs); err != nil {
		t.Fatalf("configure nvml-mock processes: %v", err)
	}
	resetNVMLMockOnCleanup(t, c)

	matchLabels := map[string]string{
		"namespace": spec.Namespace,
		"pod":       spec.Name,
		"pod_uuid":  string(pod.UID),
		"gpu_uuid":  gpuUUID,
	}
	// Assert the pod is attributed on the GPU whose UUID we pinned in nvml-mock,
	// and that the reported memory matches what we configured (used_memory_mib →
	// GetComputeRunningProcesses returns MiB*1024*1024 bytes).
	m, err := waitForSeries(ctx, c, memMetricName, matchLabels)
	if err != nil {
		t.Fatalf("%s: %v", memMetricName, err)
	}
	if got := uint64(m.GetGauge().GetValue()); got != wantBytes {
		t.Errorf("%s: want %d bytes, got %d", memMetricName, wantBytes, got)
	}

	// The same attributed process also produces the SM-utilization series for
	// this pod/GPU. Assert co-attribution (presence); the per-process util value
	// isn't controlled here, so we don't assert a specific number.
	if _, err := waitForSeries(ctx, c, smMetricName, matchLabels); err != nil {
		t.Errorf("%s: %v", smMetricName, err)
	}
}

// TestE2E_UnmatchedNVMLProcessIsDropped verifies that when NVML reports a GPU
// process whose host PID cannot be attributed to any GPU-annotated container,
// that process is silently dropped: it does not appear as a series of its own
// and does not contaminate the series of any other attributed pod.
//
// Setup: one fractional pod (matched) + PID 1 (node init, never in fsstore).
// NVML is given both PIDs. The matched pod's memory must equal only its own
// PID's contribution; if the unmatched PID leaked in, the value would be higher.
func TestE2E_UnmatchedNVMLProcessIsDropped(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}
	ctx, cancel := context.WithTimeout(context.Background(), attributionTestTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)
	nodeSel := map[string]string{"kubernetes.io/hostname": targetNode}

	// A fractional pod so the engine runs collect() — it is skipped when
	// activeContainers == 0 and we would never exercise the unmatched path.
	spec := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc-unmatched-pod",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", "1024", "1024"),
		NodeSelector:  nodeSel,
	}
	pod, err := workload.Apply(ctx, c, spec)
	if err != nil {
		t.Fatalf("create pod: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, spec.Namespace, spec.Name); err != nil {
			t.Errorf("cleanup pod: %v", err)
		}
	})

	marker := workload.DefaultMarker(spec.Namespace, spec.Name)
	pid, err := nvmlmock.HostPID(ctx, c, targetNode, marker)
	if err != nil {
		t.Fatalf("resolve host PID: %v", err)
	}

	const gpuUUID = nvmlmock.Device0UUID
	const (
		matchedMemMiB   = uint64(2048) // attributed to the fractional pod
		unmatchedMemMiB = uint64(1024) // PID 1 — no fsstore entry, must be dropped
	)

	// PID 1 is the node's init process. It has no GPU-memory annotation and no
	// fsstore entry, so ResolveProcess returns false and enrich() drops it.
	// If it leaked into the matched pod's series, memory would be
	// matchedMemMiB+unmatchedMemMiB = 3072 MiB; correctly it is matchedMemMiB only.
	procs := []nvmlmock.Proc{
		{UUID: gpuUUID, PID: pid, UsedMemoryMiB: matchedMemMiB, SMUtil: 20},
		{UUID: gpuUUID, PID: 1, UsedMemoryMiB: unmatchedMemMiB, SMUtil: 50},
	}
	if err := setProcesses(ctx, c, nvmlmock.A100, procs); err != nil {
		t.Fatalf("configure nvml-mock: %v", err)
	}
	resetNVMLMockOnCleanup(t, c)

	matchLabels := map[string]string{
		"namespace": spec.Namespace,
		"pod":       spec.Name,
		"pod_uuid":  string(pod.UID),
		"gpu_uuid":  gpuUUID,
	}

	// The fractional pod must be attributed.
	m, err := waitForSeries(ctx, c, memMetricName, matchLabels)
	if err != nil {
		t.Fatalf("%s: series never appeared: %v", memMetricName, err)
	}

	// Memory must equal only the matched PID's contribution. If the unmatched
	// PID 1 leaked into this pod's series the value would be 3072 MiB.
	wantBytes := matchedMemMiB * 1024 * 1024
	if got := uint64(m.GetGauge().GetValue()); got != wantBytes {
		t.Errorf("%s: want %d bytes (unmatched PID 1 must not add its %d MiB); got %d",
			memMetricName, wantBytes, unmatchedMemMiB, got)
	}
}
