// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/metrics"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/nvmlmock"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// TestE2E_SMUtilizationClampedAtOneHundred verifies that when two processes from
// the same pod both report SM utilization on the same GPU device, the engine
// sums their contributions (70 + 60 = 130) and clamps the result to 100 —
// testing engine.go lines 236-238 (the maxSMUtilPercent guard).
//
// Two containers in one pod each carry a distinct marker so nvmlmock.HostPID
// can resolve each container's host-namespace PID independently. Both PIDs are
// injected onto Device0UUID; their SMUtil values (70 and 60) intentionally sum
// above 100. The test asserts the exported metric reads exactly 100.
//
// Requires: E2E_NVML_MOCK=1 (set for nvml-mock clusters where per-process SM
// utilization injection is available).
func TestE2E_SMUtilizationClampedAtOneHundred(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	c := s.Client
	const podName = "tc5-clamp-pod"

	targetNode := firstGPUNode(t, ctx, c)
	spec := multiContainerPodSpec{
		Namespace:    attributionTestNamespace,
		Name:         podName,
		Containers:   []string{"proc1", "proc2"},
		MemoryMiB:    "2048",
		NodeSelector: map[string]string{"kubernetes.io/hostname": targetNode},
	}
	pod, err := applyMultiContainerFractionalPod(ctx, c, spec)
	if err != nil {
		t.Fatalf("create multi-container pod: %v", err)
	}
	t.Cleanup(func() {
		_ = workload.Delete(context.Background(), c, attributionTestNamespace, podName)
	})

	pid1, err := nvmlmock.HostPID(ctx, c, targetNode, multiContainerMarker(attributionTestNamespace, podName, "proc1"))
	if err != nil {
		t.Fatalf("resolve host PID for proc1: %v", err)
	}
	pid2, err := nvmlmock.HostPID(ctx, c, targetNode, multiContainerMarker(attributionTestNamespace, podName, "proc2"))
	if err != nil {
		t.Fatalf("resolve host PID for proc2: %v", err)
	}

	// 70 + 60 = 130 > 100 — engine must clamp to 100.
	procs := []nvmlmock.Proc{
		{UUID: nvmlmock.Device0UUID, PID: pid1, SMUtil: 70},
		{UUID: nvmlmock.Device0UUID, PID: pid2, SMUtil: 60},
	}
	if err := setProcesses(ctx, c, nvmlmock.A100, procs); err != nil {
		t.Fatalf("configure nvml-mock processes: %v", err)
	}
	resetNVMLMockOnCleanup(t, c)

	match := map[string]string{
		"namespace": attributionTestNamespace,
		"pod":       podName,
		"pod_uuid":  string(pod.UID),
		"gpu_uuid":  nvmlmock.Device0UUID,
	}
	series, err := waitForSeries(ctx, c, smMetricName, match)
	if err != nil {
		t.Fatalf("%s series never appeared: %v", smMetricName, err)
	}
	got := metrics.GaugeValue(series)
	t.Logf("%s (pid1=%d sm_util=70 + pid2=%d sm_util=60 = 130): clamped value=%.2f", smMetricName, pid1, pid2, got)
	if got != 100 {
		t.Errorf("%s: want 100 (clamped from 130), got %.2f", smMetricName, got)
	}
}
