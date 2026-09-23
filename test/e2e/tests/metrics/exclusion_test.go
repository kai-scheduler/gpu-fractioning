// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"testing"
	"time"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// TestE2E_FullGPUPodIsExcludedFromMetrics verifies that a pod requesting
// nvidia.com/gpu without a fractional gpu-memory annotation is treated as
// full-GPU and intentionally excluded from tracking
// (fractioning-manager/metricsd/internal/plugin/adapter.go:38-46). This is a
// deliberate design decision — worth a negative test since silent exclusion
// is otherwise unobservable from outside the code.
func TestE2E_FullGPUPodIsExcludedFromMetrics(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), exclusionTestTimeout)
	defer cancel()

	c := s.Client
	spec := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc7-full-gpu-pod",
		ContainerName: "trainer",
		Annotations:   map[string]string{}, // no gpu-memory annotation at all
	}

	pod, err := workload.Apply(ctx, c, spec)
	if err != nil {
		t.Fatalf("create full-GPU pod: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, spec.Namespace, spec.Name); err != nil {
			t.Errorf("cleanup pod %s/%s: %v", spec.Namespace, spec.Name, err)
		}
	})

	match := map[string]string{
		"namespace": spec.Namespace,
		"pod":       spec.Name,
		"pod_uuid":  string(pod.UID),
	}
	for _, metricName := range allMetricNames {
		assertNeverAppears(ctx, t, c, metricName, match, 15*time.Second, c.Config.PollInterval)
	}
}

// TestE2E_MalformedAnnotationIsIgnoredNotFatal verifies that a near-miss
// annotation key (here, a typo'd ".gpu-mem.limit" suffix instead of
// ".gpu-memory.limit") degrades to "not tracked" — same as a full-GPU pod —
// and does not crash the plugin. Protects against a future regex/parsing change
// turning a bad annotation into a panic or a wrongly-attributed series.
func TestE2E_MalformedAnnotationIsIgnoredNotFatal(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), exclusionTestTimeout)
	defer cancel()

	c := s.Client

	before, err := pluginPodRestartCounts(ctx, c)
	if err != nil {
		t.Fatalf("get baseline plugin restart counts: %v", err)
	}

	spec := workload.FractionalPod{
		Namespace:     attributionTestNamespace,
		Name:          "tc8-malformed-annotation-pod",
		ContainerName: "trainer",
		Annotations: map[string]string{
			// Typo'd suffix ("gpu-mem" not "gpu-memory") — doesn't match
			// annotationGPUMemoryRE, so the container must not be tracked.
			"nvidia.com/container.trainer.gpu-mem.limit": "2048",
		},
	}

	pod, err := workload.Apply(ctx, c, spec)
	if err != nil {
		t.Fatalf("create pod with malformed annotation: %v", err)
	}
	t.Cleanup(func() {
		if err := workload.Delete(context.Background(), c, spec.Namespace, spec.Name); err != nil {
			t.Errorf("cleanup pod %s/%s: %v", spec.Namespace, spec.Name, err)
		}
	})

	match := map[string]string{
		"namespace": spec.Namespace,
		"pod":       spec.Name,
		"pod_uuid":  string(pod.UID),
	}
	for _, metricName := range allMetricNames {
		assertNeverAppears(ctx, t, c, metricName, match, 15*time.Second, c.Config.PollInterval)
	}

	after, err := pluginPodRestartCounts(ctx, c)
	if err != nil {
		t.Fatalf("get plugin restart counts after malformed annotation: %v", err)
	}
	for podName, count := range after {
		if count != before[podName] {
			t.Errorf("plugin pod %s restart count changed %d -> %d — malformed annotation likely crashed it",
				podName, before[podName], count)
		}
	}
}
