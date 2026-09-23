// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package harness

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/daemonset"
	gsc "github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/gpufractioningconfig"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/nodes"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
)

// SteadyState reports whether the cluster is in FX-STEADY: operator Available,
// default CR Ready (all per-daemon conditions True, observedGeneration current),
// both DaemonSets rolled out over every matching node, and each matching node's
// gpu-fractioning Ready condition True. It is the single predicate behind both the
// preflight and the between-phase AssertSteady.
func (h *Harness) SteadyState(ctx context.Context) (bool, error) {
	// Operator Deployment Available.
	if ok, err := h.operatorDeploymentAvailable(ctx); !ok || err != nil {
		return false, err
	}

	// CR conditions.
	obj, err := gsc.Get(ctx, h.Client())
	if err != nil {
		return false, err
	}
	if obj.Status.ObservedGeneration != obj.Generation {
		return false, nil
	}
	for _, ct := range []string{CondFractiondReady, CondMpsdReady, CondReady} {
		cond, ok := gsc.Condition(obj, ct)
		if !ok || cond.Status != metav1.ConditionTrue {
			return false, nil
		}
	}

	// Matching nodes.
	gpuNodes, err := nodes.ListGPUNodes(ctx, h.Client())
	if err != nil {
		return false, err
	}
	want := int32(len(gpuNodes))

	// Both DaemonSets rolled out over exactly the matching nodes.
	for _, comp := range []string{ComponentFractiond, ComponentMpsd} {
		ds, err := daemonset.Get(ctx, h.Client(), h.NS(), DSName(comp))
		if err != nil {
			return false, err
		}
		if !daemonset.RolledOut(ds) || ds.Status.DesiredNumberScheduled != want || ds.Status.NumberReady != want {
			return false, nil
		}
	}

	// Node condition True on every matching node.
	for i := range gpuNodes {
		cond, ok := nodes.Condition(&gpuNodes[i], NodeConditionType)
		if !ok || cond.Status != corev1.ConditionTrue {
			return false, nil
		}
	}
	return true, nil
}

// AssertSteady blocks until FX-STEADY holds (bounded by RolloutTimeout) and
// fails the test otherwise. On first success it snapshots the shipped CR spec.
// Called at suite start and between phases so a case that fails to restore is
// blamed on that case, not the next one.
func (h *Harness) AssertSteady(ctx context.Context, t *testing.T) {
	t.Helper()
	if err := waiter.PollUntil(ctx, h.RolloutTimeout(), h.PollInterval(), "FX-STEADY (operator Ready with default CR)", h.SteadyState); err != nil {
		t.Fatalf("cluster not in FX-STEADY: %v", err)
	}
	if h.originalSpec == nil {
		obj, err := gsc.Get(ctx, h.Client())
		if err != nil {
			t.Fatalf("snapshot default CR: %v", err)
		}
		h.originalSpec = obj.Spec.DeepCopy()
	}
}

// RestoreSteady re-creates the shipped default CR from the snapshot and blocks
// until the cluster is back in FX-STEADY. Used by lifecycle cases to restore
// state after cleaning the CR.
func (h *Harness) RestoreSteady(ctx context.Context, t *testing.T) {
	t.Helper()
	if h.originalSpec == nil {
		t.Fatal("RestoreSteady called before the default CR was snapshotted")
	}
	obj := &v1alpha1.GpuFractioningConfig{
		ObjectMeta: metav1.ObjectMeta{Name: gsc.DefaultName},
		Spec:       *h.originalSpec.DeepCopy(),
	}
	if err := gsc.Create(ctx, h.Client(), obj); err != nil {
		t.Fatalf("re-create default CR: %v", err)
	}
	h.AssertSteady(ctx, t)
}

// ApplyCRWithSelector re-creates the default CR from the snapshot but with a
// different nodeSelector (nodeSelector is immutable on a live CR, so callers
// delete first). The caller waits for whatever state it expects.
func (h *Harness) ApplyCRWithSelector(ctx context.Context, t *testing.T, selector map[string]string) {
	t.Helper()
	if h.originalSpec == nil {
		t.Fatal("ApplyCRWithSelector called before the default CR was snapshotted")
	}
	spec := h.originalSpec.DeepCopy()
	spec.NodeSelector = selector
	obj := &v1alpha1.GpuFractioningConfig{
		ObjectMeta: metav1.ObjectMeta{Name: gsc.DefaultName},
		Spec:       *spec,
	}
	if err := gsc.Create(ctx, h.Client(), obj); err != nil {
		t.Fatalf("create CR with selector %v: %v", selector, err)
	}
}

// CleanCR deletes the default CR and blocks until the cluster is in FX-CLEAN:
// the CR is gone and both managed DaemonSets have been garbage-collected.
func (h *Harness) CleanCR(ctx context.Context, t *testing.T) {
	t.Helper()
	if err := gsc.Delete(ctx, h.Client(), gsc.DefaultName); err != nil {
		t.Fatalf("delete default CR: %v", err)
	}
	if err := gsc.WaitGone(ctx, h.Client(), gsc.DefaultName, h.RolloutTimeout(), h.PollInterval()); err != nil {
		t.Fatalf("wait CR deleted: %v", err)
	}
	for _, comp := range []string{ComponentFractiond, ComponentMpsd} {
		if err := daemonset.WaitGone(ctx, h.Client(), h.NS(), DSName(comp), h.RolloutTimeout(), h.PollInterval()); err != nil {
			t.Fatalf("wait DaemonSet %s GC'd: %v", DSName(comp), err)
		}
	}
}

// WithCRConfig applies a JSON merge patch to the default CR, runs fn (which does
// its own rollout waits + assertions), then always reverts with the revert patch
// and re-asserts FX-STEADY — so a config case can never leak an override into
// the next case.
func (h *Harness) WithCRConfig(ctx context.Context, t *testing.T, patch, revert []byte, fn func()) {
	t.Helper()
	if err := gsc.Patch(ctx, h.Client(), gsc.DefaultName, patch); err != nil {
		t.Fatalf("patch CR %s: %v", string(patch), err)
	}
	defer func() {
		if err := gsc.Patch(ctx, h.Client(), gsc.DefaultName, revert); err != nil {
			t.Errorf("revert CR patch %s: %v", string(revert), err)
		}
		h.AssertSteady(ctx, t)
	}()
	fn()
}

// operatorDeploymentAvailable reports whether the operator Deployment has >=1
// available replica.
func (h *Harness) operatorDeploymentAvailable(ctx context.Context) (bool, error) {
	dep, err := h.operatorDeployment(ctx)
	if err != nil {
		return false, err
	}
	return dep.Status.AvailableReplicas >= 1, nil
}
