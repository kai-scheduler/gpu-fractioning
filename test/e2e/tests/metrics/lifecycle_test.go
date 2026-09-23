// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"fmt"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/metrics"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/nvmlmock"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// TestE2E_IdleThenActiveThenGoneLifecycle exercises the three lifecycle states
// (idle → active → gone) exercise independent code paths in the exporter that
// can each regress independently:
//
//   - idle: a tracked pod with a GPU process reporting SMUtil=0 produces a
//     zero-value series (engine.go idlePodGPUMetric / exporter.go setZero).
//   - active: bumping SMUtil updates the series value in place — same label
//     set, no duplicate (exporter.go deleteIdleLabelsForObservedGPU removes
//     any placeholder with an empty gpu_uuid, then the active value is written).
//   - gone: deleting the pod causes both series to disappear on the next
//     scrape (exporter.go pruneDeletedPods / deleteSeries).
//
// A crash or leak in any one state won't be caught by the attribution or resilience tests alone —
// only a test that runs all three in sequence can catch a regression in the
// transition logic itself (e.g. a stale series surviving pod deletion, or an
// in-place update creating a phantom duplicate).
//
// Requires: E2E_NVML_MOCK=1 (per-process SMUtil injection available).
func TestE2E_IdleThenActiveThenGoneLifecycle(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	c := s.Client

	before, err := pluginPodRestartCounts(ctx, c)
	if err != nil {
		t.Fatalf("get baseline plugin restart counts: %v", err)
	}

	spec := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc6-lifecycle-pod",
		ContainerName: "trainer",
		Annotations:   workload.FractionalAnnotations("trainer", "2048", "2048"),
	}

	_ = workload.Delete(ctx, c, spec.Namespace, spec.Name)
	pod, err := workload.Apply(ctx, c, spec)
	if err != nil {
		t.Fatalf("create pod: %v", err)
	}
	t.Cleanup(func() {
		_ = workload.Delete(context.Background(), c, spec.Namespace, spec.Name)
	})

	marker := workload.DefaultMarker(spec.Namespace, spec.Name)
	pid, err := nvmlmock.HostPID(ctx, c, pod.Spec.NodeName, marker)
	if err != nil {
		t.Fatalf("resolve host PID: %v", err)
	}

	const (
		activeSMUtil = 55
		gpuUUID      = nvmlmock.Device0UUID
	)

	// matchActive includes gpu_uuid so the active series is addressed precisely.
	// matchPod omits it to catch duplicates regardless of what gpu_uuid the
	// idle placeholder carries (empty string vs real UUID).
	matchActive := map[string]string{
		"namespace": spec.Namespace,
		"pod":       spec.Name,
		"pod_uuid":  string(pod.UID),
		"gpu_uuid":  gpuUUID,
	}
	matchPod := map[string]string{
		"namespace": spec.Namespace,
		"pod":       spec.Name,
		"pod_uuid":  string(pod.UID),
	}

	// ── Phase 1: idle ─────────────────────────────────────────────────────────
	// PID in nvml-mock with SMUtil=0 — series must appear with value 0.
	// SMUtil defaults to zero on a Proc with no SMUtil set.
	idleProcs := []nvmlmock.Proc{{UUID: gpuUUID, PID: pid}}
	if err := setProcesses(ctx, c, nvmlmock.A100, idleProcs); err != nil {
		t.Fatalf("configure nvml-mock (idle): %v", err)
	}
	resetNVMLMockOnCleanup(t, c)

	// Wait for any series for this pod to appear (engine may carry an empty
	// gpu_uuid until it learns the device UUID from NVML, so we match on pod
	// identity only for the initial detection).
	idleSeries, err := waitForSeries(ctx, c, smMetricName, matchPod)
	if err != nil {
		t.Fatalf("idle phase: %s series never appeared: %v", smMetricName, err)
	}
	gotIdle := metrics.GaugeValue(idleSeries)
	t.Logf("idle phase: %s = %.2f (want 0)", smMetricName, gotIdle)
	if gotIdle != 0 {
		t.Errorf("idle phase: %s want 0, got %.2f", smMetricName, gotIdle)
	}

	idleMemSeries, err := waitForSeries(ctx, c, memMetricName, matchPod)
	if err != nil {
		t.Fatalf("idle phase: %s series never appeared: %v", memMetricName, err)
	}
	gotIdleMem := metrics.GaugeValue(idleMemSeries)
	t.Logf("idle phase: %s = %.2f (want 0)", memMetricName, gotIdleMem)
	if gotIdleMem != 0 {
		t.Errorf("idle phase: %s want 0, got %.2f", memMetricName, gotIdleMem)
	}

	// ── Phase 2: active ───────────────────────────────────────────────────────
	// Bump SMUtil — the series must update in place (same label set, new value)
	// and there must be exactly one series per metric family for this pod at any
	// moment (no duplicate from a stale idle placeholder).
	activeProcs := []nvmlmock.Proc{{UUID: gpuUUID, PID: pid, SMUtil: activeSMUtil}}
	if err := setProcesses(ctx, c, nvmlmock.A100, activeProcs); err != nil {
		t.Fatalf("configure nvml-mock (active): %v", err)
	}

	if err := waiter.PollUntil(ctx, c.Config.PodReadyTimeout, c.Config.PollInterval,
		fmt.Sprintf("%s matching %v to reach value %d with no duplicates", smMetricName, matchActive, activeSMUtil),
		func(ctx context.Context) (bool, error) {
			all, err := findSeriesAcrossPluginPods(ctx, c, smMetricName, matchActive)
			if err != nil {
				return false, err
			}
			if len(all) > 1 {
				return false, fmt.Errorf("duplicate series: want 1, got %d", len(all))
			}
			return len(all) == 1 && metrics.GaugeValue(all[0]) == activeSMUtil, nil
		}); err != nil {
		t.Errorf("active phase: %s never reached %d: %v", smMetricName, activeSMUtil, err)
	}

	// Also verify no leftover idle series with a different gpu_uuid label.
	allActive, err := findSeriesAcrossPluginPods(ctx, c, smMetricName, matchPod)
	if err != nil {
		t.Fatalf("scrape plugin pods after active injection: %v", err)
	}
	if len(allActive) != 1 {
		t.Errorf("active phase: want exactly 1 %s series for pod (idle placeholder not pruned?), got %d", smMetricName, len(allActive))
	}

	// ── Phase 3: gone ─────────────────────────────────────────────────────────
	// Delete the pod — both metric series must be pruned on the next scrape.
	if err := workload.Delete(ctx, c, spec.Namespace, spec.Name); err != nil {
		t.Fatalf("delete pod: %v", err)
	}

	for _, metricName := range allMetricNames {
		if err := waitForAbsence(ctx, c, metricName, matchPod); err != nil {
			t.Errorf("gone phase: %s series for deleted pod never pruned: %v", metricName, err)
		}
	}

	after, err := pluginPodRestartCounts(ctx, c)
	if err != nil {
		t.Fatalf("get plugin restart counts after lifecycle: %v", err)
	}
	for podName, count := range after {
		if count != before[podName] {
			t.Errorf("plugin pod %s restarted %d → %d during lifecycle test — exporter likely crashed", podName, before[podName], count)
		}
	}
}
