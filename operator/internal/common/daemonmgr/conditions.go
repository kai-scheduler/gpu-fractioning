// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

// This file gathers everything that reads or writes the
// gpu-fractioning.nvidia.com/Ready node condition. Helpers for the CR's own
// status conditions live in status.go.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
)

// nodeListPageSize bounds each page of the node list during condition
// cleanup, keeping peak memory flat on large clusters.
const nodeListPageSize = 500

// FindNodeCondition returns the node's gpu-fractioning.nvidia.com/Ready condition
// and whether the node carries it.
func FindNodeCondition(node *corev1.Node) (corev1.NodeCondition, bool) {
	for _, cond := range node.Status.Conditions {
		if string(cond.Type) == NodeConditionType {
			return cond, true
		}
	}
	return corev1.NodeCondition{}, false
}

// PatchNodeCondition sets or updates the gpu-fractioning.nvidia.com/Ready condition
// on a node using a strategic merge patch.
//
// reader serves the current-node read and is deliberately the uncached API
// reader: node conditions are only touched in the rare unhealthy/recovery path,
// so caching every Node cluster-wide (a cluster-scoped informer) would be pure
// overhead. writer performs the status patch, which always goes to the API
// server regardless of caching.
func PatchNodeCondition(ctx context.Context, reader client.Reader, writer client.Client, nodeName string, condition corev1.NodeCondition) error {
	log := logf.FromContext(ctx).WithValues("node", nodeName)

	now := metav1.NewTime(time.Now())
	transitionTime := now
	if string(condition.Type) != NodeConditionType {
		return fmt.Errorf("unexpected node condition type %q, expected %q", condition.Type, NodeConditionType)
	}

	// Read the current node to preserve LastTransitionTime when the condition
	// status has not changed
	// Skip the patch entirely when status, reason, and message are all identical (no-op).
	node := &corev1.Node{}
	if err := reader.Get(ctx, types.NamespacedName{Name: nodeName}, node); err != nil {
		if !apierrors.IsNotFound(err) {
			log.Error(err, "failed to read node for condition check; LastTransitionTime will reset")
		}
	} else {
		if existing, found := FindNodeCondition(node); found && existing.Status == condition.Status {
			if existing.Reason == condition.Reason && existing.Message == condition.Message {
				return nil // nothing to update
			}
			// same status, therefore LastTransitionTime should be preserved.
			transitionTime = existing.LastTransitionTime
		}
	}

	condition.LastHeartbeatTime = now
	condition.LastTransitionTime = transitionTime

	patch := map[string]any{
		"status": map[string]any{
			"conditions": []corev1.NodeCondition{condition},
		},
	}

	patchBytes, err := json.Marshal(patch)
	if err != nil {
		return fmt.Errorf("marshaling node condition patch: %w", err)
	}

	patchNode := &corev1.Node{}
	patchNode.Name = nodeName

	if err := writer.Status().Patch(ctx, patchNode, client.RawPatch(
		types.StrategicMergePatchType, patchBytes,
	)); err != nil {
		// A node that no longer exists has nothing left to assert, so this is
		// success rather than failure — matching RemoveNodeCondition. It is a
		// normal race during a cluster upgrade, when nodes are being replaced
		// at the same time as the sweep that would mark them.
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("patching node %s condition: %w", nodeName, err)
	}

	log.V(1).Info("patched node condition", "condition", NodeConditionType, "status", condition.Status, "reason", condition.Reason)
	return nil
}

// removeNodeConditionPatch deletes the gpu-fractioning.nvidia.com/Ready entry from
// the merge-keyed conditions list. Node conditions use patchMergeKey "type",
// so a plain strategic merge patch can only upsert entries; deletion needs the
// $patch:delete directive, which typed NodeConditions cannot express. The
// payload depends only on the constant condition type, so it is built once.
var removeNodeConditionPatch = []byte(`{"status":{"conditions":[{"type":"` + NodeConditionType + `","$patch":"delete"}]}}`)

// RemoveNodeCondition deletes the gpu-fractioning.nvidia.com/Ready condition from
// a node's status.
//
// The call is idempotent: deleting an absent condition is a server-side no-op,
// and a missing node is treated as success.
func RemoveNodeCondition(ctx context.Context, writer client.Client, nodeName string) error {
	log := logf.FromContext(ctx).WithValues("node", nodeName)

	patchNode := &corev1.Node{}
	patchNode.Name = nodeName

	if err := writer.Status().Patch(ctx, patchNode, client.RawPatch(
		types.StrategicMergePatchType, removeNodeConditionPatch,
	)); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("removing condition from node %s: %w", nodeName, err)
	}

	log.V(1).Info("removed node condition", "condition", NodeConditionType)
	return nil
}

// MarkNodeConditionsUnknown flips the gpu-fractioning.nvidia.com/Ready condition
// to Unknown on every node matching nodeSelector that currently carries it.
//
// It is called when the controller is shutting down. The condition is a
// controller-maintained liveness statement about a node's daemons, but nothing
// expires it: scale the operator to zero (or have it crash-loop) and every node
// keeps advertising whatever it last said, indefinitely. A scheduler that gates
// fractional placement on the condition — KAI does — then keeps placing pods on
// nodes nobody is watching. Marking Unknown on the way out makes the gap
// visible and fails in the safe direction; a restarting controller patches the
// nodes back to True on its first reconcile.
//
// Nodes that never carried the condition are left untouched, so this can never
// introduce a condition on an unrelated node. Per-node failures are joined
// rather than aborting the sweep — this runs in a shutdown window, so doing as
// much as possible before the deadline matters more than stopping at the first
// error.
func MarkNodeConditionsUnknown(ctx context.Context, reader client.Reader, writer client.Client, nodeSelector map[string]string) error {
	log := logf.FromContext(ctx)

	condition := corev1.NodeCondition{
		Type:    corev1.NodeConditionType(NodeConditionType),
		Status:  corev1.ConditionUnknown,
		Reason:  ReasonControllerUnavailable,
		Message: MessageControllerUnavailable,
	}

	var errs []error

	listOpts := []client.ListOption{
		client.MatchingLabels(nodeSelector),
		client.Limit(nodeListPageSize),
	}

	var nodeList corev1.NodeList
	for {
		if err := reader.List(ctx, &nodeList, listOpts...); err != nil {
			return fmt.Errorf("listing nodes: %w", err)
		}

		for i := range nodeList.Items {
			node := &nodeList.Items[i]
			if _, found := FindNodeCondition(node); !found {
				continue
			}
			if err := PatchNodeCondition(ctx, reader, writer, node.Name, condition); err != nil {
				log.Error(err, "failed to mark node condition unknown", "node", node.Name)
				errs = append(errs, err)
			}
		}

		if nodeList.Continue == "" {
			break
		}
		listOpts = []client.ListOption{
			client.MatchingLabels(nodeSelector),
			client.Limit(nodeListPageSize),
			client.Continue(nodeList.Continue),
		}
	}

	return errors.Join(errs...)
}

// RemoveNodeConditions removes the gpu-fractioning.nvidia.com/Ready condition from
// all nodes matching nodeSelector (the selector is reliable here because
// editing it on a live CR is unsupported). Nodes are listed via the uncached
// reader with pagination, for the same reason PatchNodeCondition avoids a
// Node informer: cleanup runs once per CR lifetime, so a cluster-scoped cache
// would be pure overhead. Per-node failures are joined rather than aborting
// the sweep, so one bad node does not prevent cleaning the rest.
func RemoveNodeConditions(ctx context.Context, reader client.Reader, writer client.Client, nodeSelector map[string]string) error {
	log := logf.FromContext(ctx)

	var errs []error

	listOpts := []client.ListOption{
		client.MatchingLabels(nodeSelector),
		client.Limit(nodeListPageSize),
	}

	var nodeList corev1.NodeList
	for {
		if err := reader.List(ctx, &nodeList, listOpts...); err != nil {
			return fmt.Errorf("listing nodes: %w", err)
		}

		for i := range nodeList.Items {
			node := &nodeList.Items[i]
			if _, found := FindNodeCondition(node); !found {
				continue
			}
			if err := RemoveNodeCondition(ctx, writer, node.Name); err != nil {
				// Name the failing node explicitly: a persistent failure here
				// keeps the finalizer in place and wedges CR deletion, so it
				// must be diagnosable from the logs.
				log.Error(err, "failed to remove node condition", "node", node.Name)
				errs = append(errs, err)
			}
		}

		if nodeList.Continue == "" {
			break
		}
		listOpts = []client.ListOption{
			client.MatchingLabels(nodeSelector),
			client.Limit(nodeListPageSize),
			client.Continue(nodeList.Continue),
		}
	}

	return errors.Join(errs...)
}
