// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package harness

import (
	"context"
	"sort"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/daemonset"
)

// DumpDiagOnFailure registers a cleanup that, only if the (sub)test failed,
// dumps operator/DS/CR/node state to help debug a failure. Mirrors the CI
// "Collect diagnostics" step (minus container logs, which CI already tails).
func (h *Harness) DumpDiagOnFailure(ctx context.Context, t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		if !t.Failed() {
			return
		}
		t.Log("── diagnostics (test failed) ─────────────────────────────")
		dep := h.OperatorDeployment(ctx, t)
		t.Logf("operator Deployment %s: available=%d/%d", dep.Name, dep.Status.AvailableReplicas, dep.Status.Replicas)
		for _, comp := range []string{ComponentFractiond, ComponentMpsd} {
			if ds, err := daemonset.Get(ctx, h.Client(), h.NS(), DSName(comp)); err == nil {
				t.Logf("ds %s: desired=%d ready=%d updated=%d unavailable=%d gen=%d observed=%d",
					ds.Name, ds.Status.DesiredNumberScheduled, ds.Status.NumberReady,
					ds.Status.UpdatedNumberScheduled, ds.Status.NumberUnavailable,
					ds.Generation, ds.Status.ObservedGeneration)
			} else {
				t.Logf("ds %s: %v", DSName(comp), err)
			}
			for _, p := range h.ListComponentPods(ctx, t, comp) {
				t.Logf("  pod %s node=%s phase=%s ready=%v", p.Name, p.Spec.NodeName, p.Status.Phase, PodReady(&p))
			}
		}
		var evs corev1.EventList
		if err := h.Client().Ctrl.List(ctx, &evs, ctrlclient.InNamespace(h.NS())); err == nil {
			sort.Slice(evs.Items, func(i, j int) bool {
				return evs.Items[i].LastTimestamp.Before(&evs.Items[j].LastTimestamp)
			})
			from := 0
			if len(evs.Items) > 20 {
				from = len(evs.Items) - 20
			}
			for _, e := range evs.Items[from:] {
				t.Logf("  event %s %s: %s", e.Type, e.Reason, strings.TrimSpace(e.Message))
			}
		}
	})
}
