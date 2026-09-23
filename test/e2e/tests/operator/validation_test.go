// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package operator

import (
	"context"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/harness"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/daemonset"
	gsc "github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/gpufractioningconfig"
)

// RejectsNonDefaultCRName — only a CR named "default" is accepted (CRD
// CEL rule). Dry-run create of a differently-named, otherwise-valid CR must be
// rejected; the live default is untouched (dry-run persists nothing).
func verifyRejectsNonDefaultCRName(ctx context.Context, t *testing.T) {
	obj := &v1alpha1.GpuFractioningConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "not-default"},
		Spec:       v1alpha1.GpuFractioningConfigSpec{NodeSelector: map[string]string{"nvidia.com/gpu.present": "true"}},
	}
	err := gsc.CreateDryRun(ctx, h.Client(), obj)
	if err == nil {
		t.Fatal("expected API server to reject a CR named other than 'default', got no error")
	}
	if !strings.Contains(err.Error(), "must be named 'default'") {
		t.Fatalf("rejection did not mention the name rule: %v", err)
	}
}

// RejectsMissingNodeSelector — nodeSelector is required. Dry-run create
// of a CR whose spec omits nodeSelector must be rejected with a message naming
// the field. A unique name avoids an AlreadyExists short-circuit; dry-run
// persists nothing.
func verifyRejectsMissingNodeSelector(ctx context.Context, t *testing.T) {
	err := gsc.CreateUnstructuredDryRun(ctx, h.Client(), "no-node-selector", map[string]any{})
	if err == nil {
		t.Fatal("expected API server to reject a CR with no nodeSelector, got no error")
	}
	if !strings.Contains(err.Error(), "nodeSelector") {
		t.Fatalf("rejection did not name the required nodeSelector field: %v", err)
	}
}

// NodeSelectorImmutable — spec.nodeSelector is immutable (CRD CEL rule
// "self == oldSelf"): the only way to change targeting is delete + recreate
// (which is why ZeroMatchSelectorIsReady owns the CR rather than patching it). A
// dry-run patch that changes nodeSelector on the live default must be rejected
// with an "immutable" message; dry-run persists nothing, so FX-STEADY is never at
// risk even if the rule regressed.
func verifyNodeSelectorImmutable(ctx context.Context, t *testing.T) {
	patch := []byte(`{"spec":{"nodeSelector":{"e2e.immutable/probe":"true"}}}`)
	err := gsc.PatchDryRun(ctx, h.Client(), gsc.DefaultName, patch)
	if err == nil {
		t.Fatal("expected API server to reject a nodeSelector change, got no error")
	}
	if !strings.Contains(err.Error(), "immutable") {
		t.Fatalf("rejection did not mention nodeSelector immutability: %v", err)
	}
}

// ZeroMatchSelectorIsReady — a selector matching zero nodes is a
// satisfied (Ready) state with reason NoTargetNodes and desired=0 on both
// DaemonSets. This case owns the CR: it deletes the default, applies a zero-match
// selector, asserts, then restores.
func caseZeroMatchSelectorIsReady(ctx context.Context, t *testing.T) {
	h.CleanCR(ctx, t)
	h.ApplyCRWithSelector(ctx, t, map[string]string{"e2e.nonexistent/label": "true"})

	// Restore the shipped default no matter how the assertions go.
	defer func() {
		h.CleanCR(ctx, t)
		h.RestoreSteady(ctx, t)
	}()

	// Wait for the controller to reconcile the zero-match selector to a
	// satisfied state.
	obj, err := gsc.WaitCondition(ctx, h.Client(), gsc.DefaultName, harness.CondReady, metav1.ConditionTrue, h.RolloutTimeout(), h.PollInterval())
	if err != nil {
		t.Fatalf("CR did not reach Ready=True with a zero-match selector: %v", err)
	}

	// Wait for the DaemonSets to to be ready
	for _, ct := range []string{harness.CondFractiondReady, harness.CondMpsdReady} {
		cond, ok := gsc.Condition(obj, ct)
		if !ok || cond.Status != metav1.ConditionTrue || cond.Reason != reasonNoTargetNodes {
			t.Errorf("%s = %v/%q, want True/%s", ct, harness.CondStatus(ok, cond), harness.CondReasonOf(ok, cond), reasonNoTargetNodes)
		}
	}
	// Wait for the CR to be ready
	if cond, ok := gsc.Condition(obj, harness.CondReady); !ok || cond.Reason != reasonAllComponentsReady {
		t.Errorf("Ready reason = %q, want %s", harness.CondReasonOf(ok, cond), reasonAllComponentsReady)
	}

	// Both DaemonSets converge to desired == 0 (may take a reconcile after the
	// selector change).
	for _, comp := range []string{harness.ComponentFractiond, harness.ComponentMpsd} {
		if _, err := daemonset.WaitDesired(ctx, h.Client(), h.NS(), harness.DSName(comp), 0, h.CondTimeout(), h.PollInterval()); err != nil {
			t.Errorf("%s did not converge to desired=0: %v", harness.DSName(comp), err)
		}
	}
}
