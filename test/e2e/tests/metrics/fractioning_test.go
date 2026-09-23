// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"strconv"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/metrics"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/nvmlmock"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// TestE2E_TwoFractionalPodsShareOneGPU verifies that two fractional pods
// co-located on the same GPU node each produce a distinct metric series
// attributed to their own pod identity, proving per-pod isolation of
// accounting on a shared device.
//
// GPU memory is read from c.Config.GPUMemoryMiB (E2E_GPU_MEMORY_MIB env,
// default 40960 for the nvml-mock A100 profile). Each pod requests half that
// memory so both fractions fit on the device.
//
// In the nvml-mock environment there is no device plugin, so GPU assignment
// is not determined by Kubernetes scheduling. Instead, both container PIDs are
// explicitly placed on nvmlmock.Device0UUID — the nvml-mock equivalent of the
// N==1 case in the test plan (single GPU, co-location guaranteed). The N>1
// skip path described in the test plan applies to real-GPU environments where
// the device plugin may assign different physical GPUs to the two pods; it is
// not exercised here because we control the assignment directly.
//
// Assert: both gpu_fractioning_gpu_memory_used_bytes and
// gpu_fractioning_gpu_sm_utilization_percent appear with the same gpu_uuid /
// gpu_index label but distinct pod / pod_uid labels — one series per pod.
// No split of utilization between the two pods is asserted: fraction only
// partitions memory in this system, and a single active pod may legitimately
// report up to 100% SM utilisation while the idle co-tenant reports 0.
func TestE2E_TwoFractionalPodsShareOneGPU(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}
	// Two DaemonSet rollouts (inside SetProcesses) plus collection cycles for
	// two pods — match the attribution/resilience test budget.
	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)

	// Each pod requests half the total GPU memory so both fit on one device.
	halfMemMiB := strconv.Itoa(c.Config.GPUMemoryMiB / 2)

	// Pin both pods to the same node via kubernetes.io/hostname.
	nodeSel := map[string]string{"kubernetes.io/hostname": targetNode}

	specA := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc2-pod-a",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", halfMemMiB, halfMemMiB),
		NodeSelector:  nodeSel,
	}
	specB := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc2-pod-b",
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

	// Place both PIDs on the same device (Device0UUID) so they share one GPU.
	// This is the deterministic co-location step: in the nvml-mock env we
	// choose the assignment directly rather than relying on a device plugin.
	const gpuUUID = nvmlmock.Device0UUID
	procs := []nvmlmock.Proc{
		{UUID: gpuUUID, PID: pidA, UsedMemoryMiB: uint64(c.Config.GPUMemoryMiB / 2), SMUtil: 60},
		{UUID: gpuUUID, PID: pidB, UsedMemoryMiB: uint64(c.Config.GPUMemoryMiB / 2), SMUtil: 40},
	}
	if err := setProcesses(ctx, c, nvmlmock.A100, procs); err != nil {
		t.Fatalf("configure nvml-mock processes: %v", err)
	}
	resetNVMLMockOnCleanup(t, c)

	matchA := map[string]string{
		"namespace": specA.Namespace,
		"pod":       specA.Name,
		"pod_uuid":  string(podA.UID),
		"gpu_uuid":  gpuUUID,
	}
	matchB := map[string]string{
		"namespace": specB.Namespace,
		"pod":       specB.Name,
		"pod_uuid":  string(podB.UID),
		"gpu_uuid":  gpuUUID,
	}

	// Both pods must appear on the shared device — one series each.
	// For SM utilization we assert the exact configured values (60/40) because
	// nvml-mock honours sm_util per-process and we control the injection.
	// For memory bytes we assert presence only: nvml-mock's
	// GetComputeRunningProcesses_v3 always returns usedGpuMemory=0 regardless
	// of the configured used_gpu_memory — value assertion is deferred until
	// the mock gains per-process memory fidelity (see attribution_test.go NOTE).
	for _, metricName := range allMetricNames {
		seriesA, err := waitForSeries(ctx, c, metricName, matchA)
		if err != nil {
			t.Errorf("%s: pod-a series never appeared: %v", metricName, err)
			continue
		}
		gotA := metrics.GaugeValue(seriesA)
		t.Logf("%s pod-a (pid=%d): value=%.2f", metricName, pidA, gotA)

		seriesB, err := waitForSeries(ctx, c, metricName, matchB)
		if err != nil {
			t.Errorf("%s: pod-b series never appeared: %v", metricName, err)
			continue
		}
		gotB := metrics.GaugeValue(seriesB)
		t.Logf("%s pod-b (pid=%d): value=%.2f", metricName, pidB, gotB)

		if metricName == smMetricName {
			if gotA != 60 {
				t.Errorf("%s pod-a: want 60, got %.2f", metricName, gotA)
			}
			if gotB != 40 {
				t.Errorf("%s pod-b: want 40, got %.2f", metricName, gotB)
			}
		}
	}
}
