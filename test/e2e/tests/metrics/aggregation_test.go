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

// TestE2E_MultiProcessPerPodAggregation verifies that when several GPU
// processes from different containers in the same pod are reported by NVML,
// metricsd sums them into a single per-pod metric series on the shared device.
//
// Pod A has 2 containers → 2 attributed processes on Device0UUID.
// Pod B has 3 containers → 3 attributed processes on Device0UUID.
//
// Each container is tracked independently by fractiond (separate cgroup, separate
// fsstore entry) but all containers in a pod share the same pod_uid, so the
// engine groups them under the same (pod_uid, gpu_uuid) key and sums:
//   - gpu_fractioning_gpu_memory_used_bytes = Σ(per-container used_memory_mib) * 1MiB
//   - gpu_fractioning_gpu_sm_utilization_percent = Σ(per-container sm_util), clamped at 100
func TestE2E_MultiProcessPerPodAggregation(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}

	ctx, cancel := context.WithTimeout(context.Background(), aggregationTestTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)
	nodeSel := map[string]string{"kubernetes.io/hostname": targetNode}

	// Pod A: 2 containers, each attributed as a separate GPU process.
	specA := multiContainerPodSpec{
		Namespace:    attributionTestNamespace,
		Name:         "tc-agg-pod-a",
		Containers:   []string{"c1", "c2"},
		MemoryMiB:    "1024",
		NodeSelector: nodeSel,
	}
	podA, err := applyMultiContainerFractionalPod(ctx, c, specA)
	if err != nil {
		t.Fatalf("create pod-a: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, specA.Namespace, specA.Name); err != nil {
			t.Errorf("cleanup pod-a: %v", err)
		}
	})

	// Pod B: 3 containers, each attributed as a separate GPU process.
	specB := multiContainerPodSpec{
		Namespace:    attributionTestNamespace,
		Name:         "tc-agg-pod-b",
		Containers:   []string{"c1", "c2", "c3"},
		MemoryMiB:    "512",
		NodeSelector: nodeSel,
	}
	podB, err := applyMultiContainerFractionalPod(ctx, c, specB)
	if err != nil {
		t.Fatalf("create pod-b: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, specB.Namespace, specB.Name); err != nil {
			t.Errorf("cleanup pod-b: %v", err)
		}
	})

	// Resolve each container's host-namespace PID. Each container runs a shell
	// with a unique marker in argv so HostPID resolves exactly one PID per call.
	pidA1, err := nvmlmock.HostPID(ctx, c, targetNode, multiContainerMarker(attributionTestNamespace, "tc-agg-pod-a", "c1"))
	if err != nil {
		t.Fatalf("resolve pid a/c1: %v", err)
	}
	pidA2, err := nvmlmock.HostPID(ctx, c, targetNode, multiContainerMarker(attributionTestNamespace, "tc-agg-pod-a", "c2"))
	if err != nil {
		t.Fatalf("resolve pid a/c2: %v", err)
	}
	pidB1, err := nvmlmock.HostPID(ctx, c, targetNode, multiContainerMarker(attributionTestNamespace, "tc-agg-pod-b", "c1"))
	if err != nil {
		t.Fatalf("resolve pid b/c1: %v", err)
	}
	pidB2, err := nvmlmock.HostPID(ctx, c, targetNode, multiContainerMarker(attributionTestNamespace, "tc-agg-pod-b", "c2"))
	if err != nil {
		t.Fatalf("resolve pid b/c2: %v", err)
	}
	pidB3, err := nvmlmock.HostPID(ctx, c, targetNode, multiContainerMarker(attributionTestNamespace, "tc-agg-pod-b", "c3"))
	if err != nil {
		t.Fatalf("resolve pid b/c3: %v", err)
	}

	// Configure nvml-mock: 2 processes for pod-a, 3 for pod-b, all on Device0UUID.
	// Memory values are in MiB; the mock returns MiB*1024*1024 bytes via
	// GetComputeRunningProcesses. SM util is per-process percent (0–100).
	const gpuUUID = nvmlmock.Device0UUID
	const (
		memA1MiB = uint64(2 * 1024) // 2 GiB
		memA2MiB = uint64(3 * 1024) // 3 GiB
		memB1MiB = uint64(1 * 1024) // 1 GiB
		memB2MiB = uint64(1 * 1024) // 1 GiB
		memB3MiB = uint64(2 * 1024) // 2 GiB
		smA1     = uint32(20)
		smA2     = uint32(30)
		smB1     = uint32(10)
		smB2     = uint32(15)
		smB3     = uint32(20)
	)

	procs := []nvmlmock.Proc{
		{UUID: gpuUUID, PID: pidA1, UsedMemoryMiB: memA1MiB, SMUtil: smA1},
		{UUID: gpuUUID, PID: pidA2, UsedMemoryMiB: memA2MiB, SMUtil: smA2},
		{UUID: gpuUUID, PID: pidB1, UsedMemoryMiB: memB1MiB, SMUtil: smB1},
		{UUID: gpuUUID, PID: pidB2, UsedMemoryMiB: memB2MiB, SMUtil: smB2},
		{UUID: gpuUUID, PID: pidB3, UsedMemoryMiB: memB3MiB, SMUtil: smB3},
	}
	if err := setProcesses(ctx, c, nvmlmock.A100, procs); err != nil {
		t.Fatalf("configure nvml-mock: %v", err)
	}
	resetNVMLMockOnCleanup(t, c)
	t.Logf("target node: %s, pids: a/c1=%d a/c2=%d b/c1=%d b/c2=%d b/c3=%d", targetNode, pidA1, pidA2, pidB1, pidB2, pidB3)

	matchA := map[string]string{
		"namespace": attributionTestNamespace,
		"pod":       "tc-agg-pod-a",
		"pod_uuid":  string(podA.UID),
		"gpu_uuid":  gpuUUID,
	}
	matchB := map[string]string{
		"namespace": attributionTestNamespace,
		"pod":       "tc-agg-pod-b",
		"pod_uuid":  string(podB.UID),
		"gpu_uuid":  gpuUUID,
	}

	// Expected aggregated SM values.
	wantSMA := float64(smA1 + smA2)        // 50 (no clamping)
	wantSMB := float64(smB1 + smB2 + smB3) // 45 (no clamping)

	if _, err := waitForSeries(ctx, c, memMetricName, matchA); err != nil {
		t.Errorf("%s pod-a: series never appeared: %v", memMetricName, err)
	}
	if _, err := waitForSeries(ctx, c, memMetricName, matchB); err != nil {
		t.Errorf("%s pod-b: series never appeared: %v", memMetricName, err)
	}

	// SM utilization: engine sums sm_util across all processes per pod×GPU, capped at 100.
	sA, err := waitForSeries(ctx, c, smMetricName, matchA)
	if err != nil {
		t.Errorf("%s pod-a: %v", smMetricName, err)
	} else if got := sA.GetGauge().GetValue(); got != wantSMA {
		t.Errorf("%s pod-a: want %.0f (sum of 2 processes), got %.2f", smMetricName, wantSMA, got)
	}

	sB, err := waitForSeries(ctx, c, smMetricName, matchB)
	if err != nil {
		t.Errorf("%s pod-b: %v", smMetricName, err)
	} else if got := sB.GetGauge().GetValue(); got != wantSMB {
		t.Errorf("%s pod-b: want %.0f (sum of 3 processes), got %.2f", smMetricName, wantSMB, got)
	}
}
