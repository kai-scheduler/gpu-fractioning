// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
)

func TestDaemonHealthToCondition(t *testing.T) {
	tests := []struct {
		name           string
		daemonName     string
		health         DaemonHealth
		reconcileErr   error
		generation     int64
		expectedType   string
		expectedStatus metav1.ConditionStatus
		expectedReason string
		expectedMsg    string
	}{
		{
			name:           "all pods ready",
			daemonName:     "fractiond",
			health:         DaemonHealth{DesiredNodes: 3, ReadyNodes: 3},
			generation:     1,
			expectedType:   "FractiondReady",
			expectedStatus: metav1.ConditionTrue,
			expectedReason: ReasonAllPodsReady,
			expectedMsg:    "3 of 3 pods ready",
		},
		{
			name:           "no target nodes",
			daemonName:     "fractiond",
			health:         DaemonHealth{DesiredNodes: 0, ReadyNodes: 0},
			generation:     1,
			expectedType:   "FractiondReady",
			expectedStatus: metav1.ConditionTrue,
			expectedReason: ReasonNoTargetNodes,
			expectedMsg:    MessageNoTargetNodes,
		},
		{
			name:           "partially available",
			daemonName:     "fractiond",
			health:         DaemonHealth{DesiredNodes: 5, ReadyNodes: 3},
			generation:     2,
			expectedType:   "FractiondReady",
			expectedStatus: metav1.ConditionFalse,
			expectedReason: ReasonPartiallyAvailable,
			expectedMsg:    "3 of 5 pods ready",
		},
		{
			name:           "rollout in progress (zero ready)",
			daemonName:     "fractiond",
			health:         DaemonHealth{DesiredNodes: 5, ReadyNodes: 0},
			generation:     1,
			expectedType:   "FractiondReady",
			expectedStatus: metav1.ConditionFalse,
			expectedReason: ReasonRolloutInProgress,
			expectedMsg:    "0 of 5 pods ready",
		},
		{
			name:           "condition type uses capitalized daemon name",
			daemonName:     "mpsd",
			health:         DaemonHealth{DesiredNodes: 2, ReadyNodes: 2},
			generation:     1,
			expectedType:   "MpsdReady",
			expectedStatus: metav1.ConditionTrue,
			expectedReason: ReasonAllPodsReady,
			expectedMsg:    "2 of 2 pods ready",
		},
		{
			name:           "reconcile error",
			daemonName:     "fractiond",
			health:         DaemonHealth{},
			reconcileErr:   fmt.Errorf("create-or-update failed"),
			generation:     1,
			expectedType:   "FractiondReady",
			expectedStatus: metav1.ConditionFalse,
			expectedReason: ReasonReconcileError,
			expectedMsg:    "create-or-update failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cond := DaemonHealthToCondition(tt.daemonName, &tt.health, tt.generation, tt.reconcileErr)

			if cond.Type != tt.expectedType {
				t.Errorf("Type = %q, expected %q", cond.Type, tt.expectedType)
			}
			if cond.Status != tt.expectedStatus {
				t.Errorf("Status = %v, expected %v", cond.Status, tt.expectedStatus)
			}
			if cond.Reason != tt.expectedReason {
				t.Errorf("Reason = %q, expected %q", cond.Reason, tt.expectedReason)
			}
			if cond.Message != tt.expectedMsg {
				t.Errorf("Message = %q, expected %q", cond.Message, tt.expectedMsg)
			}
			if cond.ObservedGeneration != tt.generation {
				t.Errorf("ObservedGeneration = %d, expected %d", cond.ObservedGeneration, tt.generation)
			}
		})
	}
}

func TestAggregateReadyCondition(t *testing.T) {
	tests := []struct {
		name           string
		conditions     []metav1.Condition
		expectedStatus metav1.ConditionStatus
		expectedReason string
		expectedMsg    string
	}{
		{
			name:           "no conditions",
			conditions:     []metav1.Condition{},
			expectedStatus: metav1.ConditionUnknown,
			expectedReason: ReasonNoComponents,
			expectedMsg:    MessageNoComponents,
		},
		{
			name: "all components true",
			conditions: []metav1.Condition{
				{Type: "FractiondReady", Status: metav1.ConditionTrue},
			},
			expectedStatus: metav1.ConditionTrue,
			expectedReason: ReasonAllComponentsReady,
			expectedMsg:    MessageAllComponentsReady,
		},
		{
			name: "one false takes priority",
			conditions: []metav1.Condition{
				{Type: "FractiondReady", Status: metav1.ConditionTrue},
				{Type: "MpsdReady", Status: metav1.ConditionFalse},
			},
			expectedStatus: metav1.ConditionFalse,
			expectedReason: ReasonComponentNotReady,
			expectedMsg:    "not ready: MpsdReady",
		},
		{
			name: "multiple false lists all",
			conditions: []metav1.Condition{
				{Type: "FractiondReady", Status: metav1.ConditionFalse},
				{Type: "MpsdReady", Status: metav1.ConditionFalse},
			},
			expectedStatus: metav1.ConditionFalse,
			expectedReason: ReasonComponentNotReady,
			expectedMsg:    "not ready: FractiondReady, MpsdReady",
		},
		{
			name: "unknown when no false present",
			conditions: []metav1.Condition{
				{Type: "FractiondReady", Status: metav1.ConditionTrue},
				{Type: "MpsdReady", Status: metav1.ConditionUnknown},
			},
			expectedStatus: metav1.ConditionUnknown,
			expectedReason: ReasonComponentUnknown,
			expectedMsg:    "unknown: MpsdReady",
		},
		{
			name: "false wins over unknown - unknown components are masked",
			conditions: []metav1.Condition{
				{Type: "FractiondReady", Status: metav1.ConditionUnknown},
				{Type: "MpsdReady", Status: metav1.ConditionFalse},
			},
			expectedStatus: metav1.ConditionFalse,
			expectedReason: ReasonComponentNotReady,
			expectedMsg:    "not ready: MpsdReady",
		},
		{
			name: "multiple unknown lists all",
			conditions: []metav1.Condition{
				{Type: "FractiondReady", Status: metav1.ConditionUnknown},
				{Type: "MpsdReady", Status: metav1.ConditionUnknown},
			},
			expectedStatus: metav1.ConditionUnknown,
			expectedReason: ReasonComponentUnknown,
			expectedMsg:    "unknown: FractiondReady, MpsdReady",
		},
		{
			name: "multiple false and unknown - only false reported",
			conditions: []metav1.Condition{
				{Type: "FractiondReady", Status: metav1.ConditionFalse},
				{Type: "MpsdReady", Status: metav1.ConditionUnknown},
				{Type: "MetricsdReady", Status: metav1.ConditionFalse},
			},
			expectedStatus: metav1.ConditionFalse,
			expectedReason: ReasonComponentNotReady,
			expectedMsg:    "not ready: FractiondReady, MetricsdReady",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cond := AggregateReadyCondition(tt.conditions, 1)

			if cond.Status != tt.expectedStatus {
				t.Errorf("Status = %v, expected %v", cond.Status, tt.expectedStatus)
			}
			if cond.Reason != tt.expectedReason {
				t.Errorf("Reason = %q, expected %q", cond.Reason, tt.expectedReason)
			}
			if cond.Message != tt.expectedMsg {
				t.Errorf("Message = %q, expected %q", cond.Message, tt.expectedMsg)
			}
		})
	}
}

func TestSetCondition_Upsert(t *testing.T) {
	status := &v1alpha1.GpuFractioningConfigStatus{}

	cond1 := metav1.Condition{
		Type:               "FractiondReady",
		Status:             metav1.ConditionFalse,
		Reason:             ReasonRolloutInProgress,
		LastTransitionTime: metav1.Now(),
	}
	SetCondition(status, cond1)

	if len(status.Conditions) != 1 {
		t.Fatalf("expected 1 condition, got %d", len(status.Conditions))
	}

	// Update same type, same status — lastTransitionTime should be preserved.
	originalTime := status.Conditions[0].LastTransitionTime
	cond2 := metav1.Condition{
		Type:               "FractiondReady",
		Status:             metav1.ConditionFalse,
		Reason:             ReasonPartiallyAvailable,
		LastTransitionTime: metav1.Now(),
	}
	SetCondition(status, cond2)

	if len(status.Conditions) != 1 {
		t.Fatalf("expected 1 condition after upsert, got %d", len(status.Conditions))
	}
	if status.Conditions[0].Reason != ReasonPartiallyAvailable {
		t.Errorf("Reason not updated: %q", status.Conditions[0].Reason)
	}
	if !status.Conditions[0].LastTransitionTime.Equal(&originalTime) {
		t.Error("LastTransitionTime should be preserved when status doesn't change")
	}

	// Change status — lastTransitionTime should update.
	cond3 := metav1.Condition{
		Type:               "FractiondReady",
		Status:             metav1.ConditionTrue,
		Reason:             ReasonAllPodsReady,
		LastTransitionTime: metav1.Now(),
	}
	SetCondition(status, cond3)

	if status.Conditions[0].LastTransitionTime.Equal(&originalTime) {
		t.Error("LastTransitionTime should change when status transitions")
	}
}

func TestSetCondition_AddsNew(t *testing.T) {
	status := &v1alpha1.GpuFractioningConfigStatus{
		Conditions: []metav1.Condition{
			{Type: "FractiondReady", Status: metav1.ConditionTrue},
		},
	}

	SetCondition(status, metav1.Condition{
		Type:   "MpsdReady",
		Status: metav1.ConditionTrue,
	})

	if len(status.Conditions) != 2 {
		t.Fatalf("expected 2 conditions, got %d", len(status.Conditions))
	}
}

func TestCapitalize(t *testing.T) {
	tests := []struct {
		input, expected string
	}{
		{"fractiond", "Fractiond"},
		{"mpsd", "Mpsd"},
		{"Already", "Already"},
		{"", ""},
	}
	for _, tt := range tests {
		if got := Capitalize(tt.input); got != tt.expected {
			t.Errorf("Capitalize(%q) = %q, expected %q", tt.input, got, tt.expected)
		}
	}
}
