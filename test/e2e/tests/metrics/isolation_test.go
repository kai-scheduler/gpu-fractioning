// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/metrics"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/nvmlmock"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// TestE2E_MultipleGPUsOnOneNodeAreIsolated verifies that two fractional pods
// placed on different physical GPUs on the same node each carry their own
// gpu_uuid / gpu_index label — no series for pod A must ever appear under
// pod B's device, and vice versa.
//
// This is the inverse of TestE2E_TwoFractionalPodsShareOneGPU (which tests
// co-location on the same device): this test proves the plugin does not
// conflate different physical devices.
//
// The test requires GPUCountPerNode >= 2 (set E2E_GPU_COUNT_PER_NODE=2 for
// nvml-mock clusters, which always expose Device0 and Device1). It skips
// rather than fabricates topology when only one GPU is available.
//
// In the nvml-mock environment there is no device plugin, so GPU assignment
// is explicit: pidA is pinned to Device0UUID and pidB to Device1UUID via
// nvmlmock.SetProcesses. The suite then asserts:
//
//  1. Positive: pod-a's series carries gpu_uuid=Device0UUID; pod-b's carries
//     gpu_uuid=Device1UUID (with the expected SMUtil values).
//  2. Negative: pod-a's pod_uid never appears under Device1UUID; pod-b's
//     pod_uid never appears under Device0UUID.
func TestE2E_MultipleGPUsOnOneNodeAreIsolated(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}
	if s.Client.Config.GPUCountPerNode < 2 {
		t.Skipf("requires GPUCountPerNode >= 2, have %d (set E2E_GPU_COUNT_PER_NODE=2 for nvml-mock clusters)",
			s.Client.Config.GPUCountPerNode)
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)

	halfMemMiB := strconv.Itoa(c.Config.GPUMemoryMiB / 2)
	nodeSel := map[string]string{"kubernetes.io/hostname": targetNode}

	specA := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc3-pod-a",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", halfMemMiB, halfMemMiB),
		NodeSelector:  nodeSel,
	}
	specB := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc3-pod-b",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", halfMemMiB, halfMemMiB),
		NodeSelector:  nodeSel,
	}

	podA, err := workload.Apply(ctx, c, specA)
	if err != nil {
		t.Fatalf("create pod-a: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, specA.Namespace, specA.Name); err != nil {
			t.Errorf("cleanup pod-a: %v", err)
		}
	})

	podB, err := workload.Apply(ctx, c, specB)
	if err != nil {
		t.Fatalf("create pod-b: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, specB.Namespace, specB.Name); err != nil {
			t.Errorf("cleanup pod-b: %v", err)
		}
	})

	markerA := workload.DefaultMarker(specA.Namespace, specA.Name)
	pidA, err := nvmlmock.HostPID(ctx, c, targetNode, markerA)
	if err != nil {
		t.Fatalf("resolve host PID for pod-a: %v", err)
	}

	markerB := workload.DefaultMarker(specB.Namespace, specB.Name)
	pidB, err := nvmlmock.HostPID(ctx, c, targetNode, markerB)
	if err != nil {
		t.Fatalf("resolve host PID for pod-b: %v", err)
	}

	// Pin each pod to a distinct device: pod-a on Device0, pod-b on Device1.
	// This is the deterministic multi-device assignment in nvml-mock — in a
	// real cluster the device plugin would assign different physical GPUs.
	procs := []nvmlmock.Proc{
		{UUID: nvmlmock.Device0UUID, PID: pidA, UsedMemoryMiB: uint64(c.Config.GPUMemoryMiB / 2), SMUtil: 70},
		{UUID: nvmlmock.Device1UUID, PID: pidB, UsedMemoryMiB: uint64(c.Config.GPUMemoryMiB / 2), SMUtil: 30},
	}
	if err := setProcesses(ctx, c, nvmlmock.A100, procs); err != nil {
		t.Fatalf("configure nvml-mock processes: %v", err)
	}
	resetNVMLMockOnCleanup(t, c)

	// matchA / matchB include gpu_uuid and gpu_index so waitForSeries only
	// succeeds when the series is on the right device with the correct index label.
	matchA := map[string]string{
		"namespace": specA.Namespace,
		"pod":       specA.Name,
		"pod_uuid":  string(podA.UID),
		"gpu_uuid":  nvmlmock.Device0UUID,
		"gpu":       "0",
	}
	matchB := map[string]string{
		"namespace": specB.Namespace,
		"pod":       specB.Name,
		"pod_uuid":  string(podB.UID),
		"gpu_uuid":  nvmlmock.Device1UUID,
		"gpu":       "1",
	}

	// 1. Positive assertion: each pod's series appears on its assigned device.
	for _, metricName := range allMetricNames {
		seriesA, err := waitForSeries(ctx, c, metricName, matchA)
		if err != nil {
			t.Errorf("%s: pod-a series on Device0 never appeared: %v", metricName, err)
			continue
		}
		gotA := metrics.GaugeValue(seriesA)
		t.Logf("%s pod-a (pid=%d, device=0): value=%.2f", metricName, pidA, gotA)

		seriesB, err := waitForSeries(ctx, c, metricName, matchB)
		if err != nil {
			t.Errorf("%s: pod-b series on Device1 never appeared: %v", metricName, err)
			continue
		}
		gotB := metrics.GaugeValue(seriesB)
		t.Logf("%s pod-b (pid=%d, device=1): value=%.2f", metricName, pidB, gotB)

		if metricName == smMetricName {
			if gotA != 70 {
				t.Errorf("%s pod-a: want 70, got %.2f", metricName, gotA)
			}
			if gotB != 30 {
				t.Errorf("%s pod-b: want 30, got %.2f", metricName, gotB)
			}
		}
	}

	// 2. Negative assertion: no cross-device contamination.
	// pod-a's pod_uid must never appear under Device1UUID, and vice versa.
	// assertNeverAppears (sustained window) is used instead of waitForAbsence:
	// these series never existed, so waitForAbsence passes on the first poll
	// before a contamination bug even has time to produce a series.
	crossA := map[string]string{
		"pod_uuid": string(podA.UID),
		"gpu_uuid": nvmlmock.Device1UUID,
	}
	crossB := map[string]string{
		"pod_uuid": string(podB.UID),
		"gpu_uuid": nvmlmock.Device0UUID,
	}
	for _, metricName := range allMetricNames {
		assertNeverAppears(ctx, t, c, metricName, crossA, 15*time.Second, c.Config.PollInterval)
		assertNeverAppears(ctx, t, c, metricName, crossB, 15*time.Second, c.Config.PollInterval)
	}
}
