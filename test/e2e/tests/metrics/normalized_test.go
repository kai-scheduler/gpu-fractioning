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

// TestE2E_NormalizedSMUtilIsCapped verifies that
// gpu_fractioning_gpu_sm_utilization_percent_normalized is capped at 100 when the
// raw SM utilization divided by the pod's declared GPU fraction exceeds 100.
//
// A pod holding 50% of the device's memory with SMUtil=80 would produce a raw
// normalized value of ≈160 — this test asserts the exporter clamps it to 100.
//
// Requires: E2E_NVML_MOCK=1.
func TestE2E_NormalizedSMUtilIsCapped(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)

	// 50% memory fraction; SMUtil=80 → uncapped normalized ≈ 160 → clamped to 100.
	halfMemMiB := strconv.Itoa(c.Config.GPUMemoryMiB / 2)
	const (
		gpuUUID    = nvmlmock.Device0UUID
		wantSMUtil = 80
	)

	spec := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "norm-capped-pod",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", halfMemMiB, halfMemMiB),
		NodeSelector:  map[string]string{"kubernetes.io/hostname": targetNode},
	}

	_ = workload.Delete(ctx, c, spec.Namespace, spec.Name)
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

	procs := []nvmlmock.Proc{{UUID: gpuUUID, PID: pid, SMUtil: wantSMUtil}}
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

	series, err := waitForSeries(ctx, c, normMetricName, matchLabels)
	if err != nil {
		t.Fatalf("%s: series never appeared: %v", normMetricName, err)
	}
	got := metrics.GaugeValue(series)
	t.Logf("%s: got %.2f (fraction≈0.5, smUtil=%d, uncapped would be ≈%.0f)",
		normMetricName, got, wantSMUtil, float64(wantSMUtil)/0.5)
	if got != 100 {
		t.Errorf("%s: want 100 (capped), got %.2f", normMetricName, got)
	}
}

// TestE2E_NormalizedSMUtilIsProportional verifies that when the raw SM
// utilization is below the proportional ceiling,
// gpu_fractioning_gpu_sm_utilization_percent_normalized is amplified above the raw
// value — proving the normalization formula (smUtil ÷ fraction) is applied and
// the cap is not engaged.
//
// A pod holding 50% of the device's memory with SMUtil=40 should export a
// normalized value greater than 40 (approximately 80) and strictly less than 100.
//
// Requires: E2E_NVML_MOCK=1.
func TestE2E_NormalizedSMUtilIsProportional(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)

	// 50% memory fraction; SMUtil=40 → normalized ≈ 80 (below 100 cap).
	halfMemMiB := strconv.Itoa(c.Config.GPUMemoryMiB / 2)
	const (
		gpuUUID    = nvmlmock.Device0UUID
		wantSMUtil = 40
	)

	spec := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "norm-proportional-pod",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", halfMemMiB, halfMemMiB),
		NodeSelector:  map[string]string{"kubernetes.io/hostname": targetNode},
	}

	_ = workload.Delete(ctx, c, spec.Namespace, spec.Name)
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

	procs := []nvmlmock.Proc{{UUID: gpuUUID, PID: pid, SMUtil: wantSMUtil}}
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

	series, err := waitForSeries(ctx, c, normMetricName, matchLabels)
	if err != nil {
		t.Fatalf("%s: series never appeared: %v", normMetricName, err)
	}
	got := metrics.GaugeValue(series)
	t.Logf("%s: got %.2f (fraction≈0.5, smUtil=%d)", normMetricName, got, wantSMUtil)

	// Normalized = smUtil / fraction ≈ 40 / 0.5 = 80. Assert a tight band
	// [75, 90] — wide enough for integer MiB rounding in gpuFraction,
	// tight enough to catch formula errors (e.g. not normalizing, or clamping
	// when it shouldn't). The 100-cap is not triggered at this injection level.
	const wantNorm = 80.0
	const normTolerance = 5.0
	if got < wantNorm-normTolerance || got > wantNorm+normTolerance {
		t.Errorf("%s: want %.0f±%.0f (smUtil=%d / fraction≈0.5), got %.2f",
			normMetricName, wantNorm, normTolerance, wantSMUtil, got)
	}
	if got > 100 {
		t.Errorf("%s: want ≤ 100 (cap must not apply at 80), got %.2f", normMetricName, got)
	}
}
