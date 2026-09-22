// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDriverUpgradeActive(t *testing.T) {
	cases := map[string]bool{
		"":                      false,
		"upgrade-done":          false,
		"upgrade-required":      true,
		"cordon-required":       true,
		"pod-deletion-required": true,
		"upgrade-failed":        true,
	}
	for value, want := range cases {
		if got := DriverUpgradeActive(value); got != want {
			t.Errorf("DriverUpgradeActive(%q) = %v, want %v", value, got, want)
		}
	}
}

func TestDaemonNodeSelector(t *testing.T) {
	cr := map[string]string{"nvidia.com/gpu.present": "true"}

	got := DaemonNodeSelector(cr)

	if got["nvidia.com/gpu.present"] != "true" {
		t.Errorf("CR selector not preserved: %v", got)
	}
	if got[GPUDeployClientLabel] != GPUDeployClientValue {
		t.Errorf("got[%s] = %q, want %q", GPUDeployClientLabel, got[GPUDeployClientLabel], GPUDeployClientValue)
	}
	// The CR's map is shared with the reconciler's node-listing paths, which must
	// keep selecting every targeted node — including ones the gpu-operator has
	// paused mid-upgrade.
	if _, added := cr[GPUDeployClientLabel]; added {
		t.Errorf("DaemonNodeSelector mutated the CR selector: %v", cr)
	}
}

// A CR pinning the label to another value would silently opt out of
// driver-upgrade coordination, so the daemon's own value has to win.
func TestDaemonNodeSelectorOverridesConflictingValue(t *testing.T) {
	got := DaemonNodeSelector(map[string]string{GPUDeployClientLabel: "false"})

	if got[GPUDeployClientLabel] != GPUDeployClientValue {
		t.Errorf("got[%s] = %q, want %q", GPUDeployClientLabel, got[GPUDeployClientLabel], GPUDeployClientValue)
	}
}

func TestDaemonNodeSelectorNilCRSelector(t *testing.T) {
	got := DaemonNodeSelector(nil)

	if len(got) != 1 || got[GPUDeployClientLabel] != GPUDeployClientValue {
		t.Errorf("DaemonNodeSelector(nil) = %v, want just %s=%s", got, GPUDeployClientLabel, GPUDeployClientValue)
	}
}

// The managed daemon must schedule only where the upgrade label is absent or
// upgrade-done, so the DaemonSet controller drains it from upgrading nodes.
func TestBaseDaemonSetDriverUpgradeAffinity(t *testing.T) {
	ds := BaseDaemonSet("fractiond", "ns")

	aff := ds.Spec.Template.Spec.Affinity
	if aff == nil || aff.NodeAffinity == nil || aff.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		t.Fatal("expected a required node affinity on the daemon pod template")
	}
	terms := aff.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
	if len(terms) != 2 {
		t.Fatalf("expected 2 ORed node selector terms, got %d", len(terms))
	}
	if got := terms[0].MatchExpressions[0]; got.Key != DriverUpgradeStateLabel || got.Operator != corev1.NodeSelectorOpDoesNotExist {
		t.Errorf("term[0] = %+v, want %s DoesNotExist", got, DriverUpgradeStateLabel)
	}
	if got := terms[1].MatchExpressions[0]; got.Operator != corev1.NodeSelectorOpIn ||
		len(got.Values) != 1 || got.Values[0] != DriverUpgradeStateDone {
		t.Errorf("term[1] = %+v, want In[%s]", got, DriverUpgradeStateDone)
	}
}

func TestDriverUpgradeCondition(t *testing.T) {
	active := DriverUpgradeCondition(true, 7)
	if active.Type != ConditionDriverUpgradeInProgress || active.Status != metav1.ConditionTrue || active.ObservedGeneration != 7 {
		t.Errorf("active condition = %+v", active)
	}
	if inactive := DriverUpgradeCondition(false, 7); inactive.Status != metav1.ConditionFalse {
		t.Errorf("inactive status = %v, want False", inactive.Status)
	}
}

// A DriverUpgradeInProgress condition (which is False in steady state) must not
// be mistaken for a not-ready daemon component when aggregating Ready.
func TestAggregateReadyIgnoresDriverUpgrade(t *testing.T) {
	conds := []metav1.Condition{
		{Type: "fractiond", Status: metav1.ConditionTrue},
		{Type: "mpsd", Status: metav1.ConditionTrue},
		{Type: ConditionDriverUpgradeInProgress, Status: metav1.ConditionFalse},
	}
	if got := AggregateReadyCondition(conds, 1); got.Status != metav1.ConditionTrue {
		t.Errorf("Ready = %v, want True (DriverUpgradeInProgress must be excluded)", got.Status)
	}
}
