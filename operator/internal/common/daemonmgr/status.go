// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
)

const (
	// NodeConditionType is the condition type set on GPU nodes to indicate
	// whether all gpu-fractioning daemons are healthy on that node.
	NodeConditionType = "gpu-fractioning.nvidia.com/Ready"

	ConditionReady = "Ready"

	// Condition reasons.
	ReasonReconcileError                = "ReconcileError"
	ReasonNoTargetNodes                 = "NoTargetNodes"
	ReasonAllPodsReady                  = "AllPodsReady"
	ReasonPartiallyAvailable            = "PartiallyAvailable"
	ReasonRolloutInProgress             = "RolloutInProgress"
	ReasonComponentNotReady             = "ComponentNotReady"
	ReasonComponentUnknown              = "ComponentUnknown"
	ReasonAllComponentsReady            = "AllComponentsReady"
	ReasonNoComponents                  = "NoComponents"
	ReasonAllDaemonsReady               = "AllDaemonsReady"
	ReasonGPUOperatorNotReady           = "GPUOperatorNotReady"
	ReasonGPUOperatorVersionUnsupported = "GPUOperatorVersionUnsupported"
	ReasonGPUDriverVersionMissing       = "GPUDriverVersionMissing"
	ReasonGPUDriverVersionInvalid       = "GPUDriverVersionInvalid"
	ReasonGPUDriverVersionUnsupported   = "GPUDriverVersionUnsupported"

	// Condition messages.
	MessageNoTargetNodes      = "no target nodes found"
	MessageComponentUnknown   = "one or more components have unknown status"
	MessageAllComponentsReady = "all components are healthy"
	MessageNoComponents       = "no components have been evaluated"
	MessageAllDaemonsReady    = "all gpu-fractioning daemons are running"
)

// DaemonHealthToCondition maps a DaemonHealth to a metav1.Condition for the CR.
// If reconcileErr is non-nil, the condition is set to False with reason ReconcileError.
func DaemonHealthToCondition(daemonName string, health *DaemonHealth, generation int64, reconcileErr error) metav1.Condition {
	condType := fmt.Sprintf("%sReady", Capitalize(daemonName))
	now := metav1.NewTime(time.Now())

	if reconcileErr != nil {
		return metav1.Condition{
			Type:               condType,
			Status:             metav1.ConditionFalse,
			ObservedGeneration: generation,
			LastTransitionTime: now,
			Reason:             ReasonReconcileError,
			Message:            reconcileErr.Error(),
		}
	}

	if health.DesiredNodes == 0 {
		return metav1.Condition{
			Type:               condType,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: generation,
			LastTransitionTime: now,
			Reason:             ReasonNoTargetNodes,
			Message:            MessageNoTargetNodes,
		}
	}

	if health.ReadyNodes == health.DesiredNodes {
		return metav1.Condition{
			Type:               condType,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: generation,
			LastTransitionTime: now,
			Reason:             ReasonAllPodsReady,
			Message:            fmt.Sprintf("%d of %d pods ready", health.ReadyNodes, health.DesiredNodes),
		}
	}

	reason := ReasonPartiallyAvailable
	if health.ReadyNodes == 0 {
		reason = ReasonRolloutInProgress
	}

	return metav1.Condition{
		Type:               condType,
		Status:             metav1.ConditionFalse,
		ObservedGeneration: generation,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            fmt.Sprintf("%d of %d pods ready", health.ReadyNodes, health.DesiredNodes),
	}
}

// AggregateReadyCondition computes the aggregate Ready condition from all
// per-daemon conditions. Ready is True only when all are True.
func AggregateReadyCondition(conditions []metav1.Condition, generation int64) metav1.Condition {
	now := metav1.NewTime(time.Now())

	if len(conditions) == 0 {
		return metav1.Condition{
			Type:               ConditionReady,
			Status:             metav1.ConditionUnknown,
			ObservedGeneration: generation,
			LastTransitionTime: now,
			Reason:             ReasonNoComponents,
			Message:            MessageNoComponents,
		}
	}

	var falseComponents, unknownComponents []string
	for _, c := range conditions {
		// Skip the aggregate itself and orthogonal conditions (e.g.
		// DriverUpgradeInProgress) so only per-daemon components drive Ready.
		if c.Type == ConditionReady || c.Type == ConditionDriverUpgradeInProgress {
			continue
		}
		switch c.Status {
		case metav1.ConditionFalse:
			falseComponents = append(falseComponents, string(c.Type))
		case metav1.ConditionUnknown:
			unknownComponents = append(unknownComponents, string(c.Type))
		}
	}

	// False takes precedence over Unknown: if any component is definitively
	// not ready we report that. Unknown components are only surfaced when
	// nothing is False.
	if len(falseComponents) > 0 {
		return metav1.Condition{
			Type:               ConditionReady,
			Status:             metav1.ConditionFalse,
			ObservedGeneration: generation,
			LastTransitionTime: now,
			Reason:             ReasonComponentNotReady,
			Message:            fmt.Sprintf("not ready: %s", strings.Join(falseComponents, ", ")),
		}
	}

	if len(unknownComponents) > 0 {
		return metav1.Condition{
			Type:               ConditionReady,
			Status:             metav1.ConditionUnknown,
			ObservedGeneration: generation,
			LastTransitionTime: now,
			Reason:             ReasonComponentUnknown,
			Message:            fmt.Sprintf("unknown: %s", strings.Join(unknownComponents, ", ")),
		}
	}

	return metav1.Condition{
		Type:               ConditionReady,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: generation,
		LastTransitionTime: now,
		Reason:             ReasonAllComponentsReady,
		Message:            MessageAllComponentsReady,
	}
}

// SetCondition upserts a condition into the status. If a condition with the same
// type already exists with the same status, the lastTransitionTime is preserved.
func SetCondition(status *v1alpha1.GpuFractioningConfigStatus, cond metav1.Condition) {
	for i, existing := range status.Conditions {
		if existing.Type == cond.Type {
			// Per Kubernetes API conventions, LastTransitionTime is only updated
			// when the status actually changes (e.g. True → False). This lets
			// consumers know how long a condition has been in its current state.
			if existing.Status == cond.Status {
				cond.LastTransitionTime = existing.LastTransitionTime
			}
			status.Conditions[i] = cond
			return
		}
	}
	status.Conditions = append(status.Conditions, cond)
}

// Capitalize returns s with the first letter uppercased.
func Capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
