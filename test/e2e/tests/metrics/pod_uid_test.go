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

// TestE2E_StaleSeriesPrunedAfterPodRestart verifies that when a pod is deleted
// and a new pod with the same name (but a different UID) is created, the old
// pod's metric series is pruned and does not resurface. This proves
// pruneDeletedPods keys on PodUID, not pod name — a regression using name
// instead would leave the old UID's series alive when the name is reused.
//
// Flow: create pod (gen1) → inject PID → confirm series → delete pod →
// confirm series pruned → recreate same name (gen2, new UID) → inject PID →
// confirm new series → assert old UID series remains absent.
func TestE2E_StaleSeriesPrunedAfterPodRestart(t *testing.T) {
	if !s.Client.Config.NVMLMock {
		t.Skip(skipNoNVMLMock)
	}
	ctx, cancel := context.WithTimeout(context.Background(), podUIDTestTimeout)
	defer cancel()

	c := s.Client

	targetNode := firstGPUNode(t, ctx, c)
	nodeSel := map[string]string{"kubernetes.io/hostname": targetNode}

	const (
		gpuUUID = nvmlmock.Device0UUID
		podName = "uid-change-pod"
	)

	// createAndWait creates the named pod, pins its PID in nvml-mock, and waits
	// for the memory series to appear. Returns the pod UID.
	createAndWait := func(genLabel string) string {
		t.Helper()
		spec := workload.FractionalPod{
			Namespace:     attributionTestNamespace,
			Name:          podName,
			ContainerName: "trainer",
			Annotations:   workload.FractionalAnnotations("trainer", "2048", "2048"),
			NodeSelector:  nodeSel,
		}
		_ = workload.Delete(ctx, c, spec.Namespace, spec.Name)
		pod, err := workload.Apply(ctx, c, spec)
		if err != nil {
			t.Fatalf("[%s] create pod: %v", genLabel, err)
		}

		marker := workload.DefaultMarker(spec.Namespace, spec.Name)
		pid, err := nvmlmock.HostPID(ctx, c, targetNode, marker)
		if err != nil {
			t.Fatalf("[%s] resolve host PID: %v", genLabel, err)
		}
		procs := []nvmlmock.Proc{{UUID: gpuUUID, PID: pid}}
		if err := setProcesses(ctx, c, nvmlmock.A100, procs); err != nil {
			t.Fatalf("[%s] configure nvml-mock: %v", genLabel, err)
		}

		matchLabels := map[string]string{
			"namespace": spec.Namespace,
			"pod":       spec.Name,
			"pod_uuid":  string(pod.UID),
			"gpu_uuid":  gpuUUID,
		}
		if _, err := waitForSeries(ctx, c, memMetricName, matchLabels); err != nil {
			t.Fatalf("[%s] %s series never appeared: %v", genLabel, memMetricName, err)
		}
		t.Logf("[%s] series confirmed for pod_uid=%s", genLabel, pod.UID)
		return string(pod.UID)
	}

	// Always clean up the pod and nvml-mock state at end of test.
	t.Cleanup(func() {
		_ = workload.Delete(context.Background(), c, attributionTestNamespace, podName)
	})
	resetNVMLMockOnCleanup(t, c)

	// Generation 1: establish a series for oldUID.
	oldUID := createAndWait("gen1")

	// Delete gen1 and reset the mock — the process no longer exists.
	if err := workload.Delete(ctx, c, attributionTestNamespace, podName); err != nil {
		t.Fatalf("delete gen1 pod: %v", err)
	}
	if err := setProcesses(ctx, c, nvmlmock.A100, nil); err != nil {
		t.Fatalf("reset nvml-mock after gen1: %v", err)
	}

	oldMatch := map[string]string{
		"namespace": attributionTestNamespace,
		"pod_uuid":  oldUID,
	}
	for _, metricName := range allMetricNames {
		if err := waitForAbsence(ctx, c, metricName, oldMatch); err != nil {
			t.Errorf("[gen1 prune] %s: series for old pod_uid=%s never pruned: %v",
				metricName, oldUID, err)
		}
	}
	t.Logf("gen1 series (pod_uid=%s) confirmed pruned", oldUID)

	// Generation 2: recreate the same pod name — Kubernetes assigns a new UID.
	newUID := createAndWait("gen2")

	if newUID == oldUID {
		t.Fatalf("gen2 pod has same UID as gen1 (%s) — Kubernetes must assign distinct UIDs", oldUID)
	}
	t.Logf("gen2 pod_uid=%s (distinct from gen1 %s)", newUID, oldUID)

	// The old UID series must not have reappeared under the new pod's name.
	for _, metricName := range allMetricNames {
		series, err := findSeriesAcrossPluginPods(ctx, c, metricName, oldMatch)
		if err != nil {
			t.Errorf("[gen2 check] scrape error: %v", err)
			continue
		}
		if len(series) > 0 {
			t.Errorf("[gen2 check] %s: old pod_uid=%s series reappeared after name reuse (pruneDeletedPods keying on name instead of UID?)",
				metricName, oldUID)
		}
	}
}
