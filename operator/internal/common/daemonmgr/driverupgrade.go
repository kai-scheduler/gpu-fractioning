// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"time"
)

const (
	// DriverUpgradeStateLabel is the NVIDIA gpu-operator node label that tracks
	// GPU driver upgrade progress. It is a FROZEN external contract owned by the
	// gpu-operator and stays under nvidia.com.
	DriverUpgradeStateLabel = "nvidia.com/gpu-driver-upgrade-state"

	// DriverUpgradeStateDone is the gpu-operator's terminal (idle) value for
	// DriverUpgradeStateLabel. The label is absent on nodes the gpu-operator has
	// never upgraded and set to "upgrade-done" once an upgrade completes; any
	// other non-empty value (cordon-required, pod-deletion-required, …) means an
	// upgrade is actively in progress on that node.
	DriverUpgradeStateDone = "upgrade-done"

	// ConditionDriverUpgradeInProgress is the CR status condition set to True
	// while any GPU node targeted by the CR is mid driver-upgrade (its managed
	// daemons are drained). It is orthogonal to Ready and must be excluded from
	// the Ready aggregation.
	ConditionDriverUpgradeInProgress = "DriverUpgradeInProgress"

	// GPUDeployClientLabel is the NVIDIA gpu-operator node label that declares a
	// node as running third-party GPU *client* pods: daemons that hold GPU device
	// handles through the nvidia runtime (NVIDIA_VISIBLE_DEVICES) instead of
	// requesting the nvidia.com/gpu resource. It is a FROZEN external contract
	// owned by the gpu-operator and stays under nvidia.com.
	//
	// Carrying it in the daemon's nodeSelector is the gpu-operator's supported way
	// for an out-of-tree GPU client to take part in driver upgrades. Without it we
	// are invisible to the upgrade: its pod-deletion step selects only pods that
	// requested nvidia.com/gpu, and both that step and the drain fallback hardcode
	// IgnoreAllDaemonSets. mpsd would keep the driver module pinned, the unload
	// would fail with EBUSY, and the driver pod would CrashLoopBackOff.
	//
	// The gpu-operator moves the label off "true" (to paused-for-driver-upgrade)
	// while it upgrades a node, which drains us the same way
	// driverUpgradeNodeAffinity does — but on a signal it always emits, and it
	// waits for the pods to actually terminate before unloading the driver.
	GPUDeployClientLabel = "nvidia.com/gpu.deploy.client"

	// GPUDeployClientValue is the only GPUDeployClientLabel value that means
	// "GPU clients may run here". Every other value, including the gpu-operator's
	// paused-for-driver-upgrade, must keep the managed daemons off the node.
	GPUDeployClientValue = "true"
)

// DaemonNodeSelector returns the nodeSelector for a managed daemon pod: the CR's
// nodeSelector plus GPUDeployClientLabel.
//
// The label is added here rather than in the CR's spec.nodeSelector because that
// field is immutable (CEL-enforced); requiring it there would force every
// existing user to delete and recreate their GpuFractioningConfig to get the fix.
//
// crSelector is never mutated: it is the CR's own map, shared by the node-listing
// paths in the reconciler.
func DaemonNodeSelector(crSelector map[string]string) map[string]string {
	selector := make(map[string]string, len(crSelector)+1)
	for key, value := range crSelector {
		selector[key] = value
	}
	// Set last, deliberately: a CR that pins this key to another value would
	// otherwise opt itself out of driver-upgrade coordination.
	selector[GPUDeployClientLabel] = GPUDeployClientValue
	return selector
}

// DriverUpgradeActive reports whether a DriverUpgradeStateLabel value indicates
// an in-progress upgrade: any value set other than the terminal "upgrade-done".
func DriverUpgradeActive(labelValue string) bool {
	return labelValue != "" && labelValue != DriverUpgradeStateDone
}

// driverUpgradeNodeAffinity returns a required nodeAffinity that schedules the
// managed daemon only onto nodes that are NOT mid driver-upgrade: those with no
// DriverUpgradeStateLabel (the common case) or with the terminal "upgrade-done".
//
// This is how driver-upgrade coordination works with no new control API. For a
// DaemonSet, the DaemonSet controller actively deletes pods from nodes that no
// longer satisfy the pod's node affinity (unlike a Deployment, where
// IgnoredDuringExecution would leave the pod running). So when the gpu-operator
// sets the upgrade label on a node, the node stops matching, the DaemonSet
// controller drains fractiond/mpsd from it, mpsd exits and MPS graceful-quits
// before the driver is unloaded. When the label clears (or reaches upgrade-done)
// the node matches again and the daemons are rescheduled.
func driverUpgradeNodeAffinity() *corev1.Affinity {
	return &corev1.Affinity{
		NodeAffinity: &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				// The two terms are ORed: schedule where the label is absent OR
				// equal to the terminal upgrade-done value.
				NodeSelectorTerms: []corev1.NodeSelectorTerm{
					{MatchExpressions: []corev1.NodeSelectorRequirement{{
						Key:      DriverUpgradeStateLabel,
						Operator: corev1.NodeSelectorOpDoesNotExist,
					}}},
					{MatchExpressions: []corev1.NodeSelectorRequirement{{
						Key:      DriverUpgradeStateLabel,
						Operator: corev1.NodeSelectorOpIn,
						Values:   []string{DriverUpgradeStateDone},
					}}},
				},
			},
		},
	}
}

// DriverUpgradeCondition builds the CR's DriverUpgradeInProgress condition for
// the given aggregate state.
func DriverUpgradeCondition(active bool, generation int64) metav1.Condition {
	status := metav1.ConditionFalse
	reason := "NoDriverUpgrade"
	// Steady-state (nothing draining): the reason already says it, so leave the
	// message empty rather than restating it.
	message := ""
	if active {
		status = metav1.ConditionTrue
		reason = "DriverUpgrading"
		message = "a targeted GPU node is undergoing a driver upgrade; its managed daemons are drained"
	}
	return metav1.Condition{
		Type:               ConditionDriverUpgradeInProgress,
		Status:             status,
		ObservedGeneration: generation,
		LastTransitionTime: metav1.NewTime(time.Now()),
		Reason:             reason,
		Message:            message,
	}
}
