// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nodes verifies GPU node preconditions (count, labels). It never
// labels, cordons, or otherwise mutates nodes.
package nodes

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
)

// ListGPUNodes returns all nodes matching c.Config.GPUNodeSelector.
func ListGPUNodes(ctx context.Context, c *cluster.Client) ([]corev1.Node, error) {
	sel, err := labels.Parse(c.Config.GPUNodeSelector)
	if err != nil {
		return nil, fmt.Errorf("parse GPU node selector %q: %w", c.Config.GPUNodeSelector, err)
	}
	var list corev1.NodeList
	if err := c.Ctrl.List(ctx, &list, ctrlclient.MatchingLabelsSelector{Selector: sel}); err != nil {
		return nil, fmt.Errorf("list nodes matching %q: %w", c.Config.GPUNodeSelector, err)
	}
	return list.Items, nil
}

// VerifyGPUNodes asserts that exactly c.Config.GPUNodeCount nodes match
// c.Config.GPUNodeSelector. With GPUNodeCount == 0 there is nothing to
// assert, so it returns immediately without parsing the selector or listing nodes.
func VerifyGPUNodes(ctx context.Context, c *cluster.Client) error {
	if c.Config.GPUNodeCount == 0 {
		return nil
	}

	sel, err := labels.Parse(c.Config.GPUNodeSelector)
	if err != nil {
		return fmt.Errorf("parse GPU node selector %q: %w", c.Config.GPUNodeSelector, err)
	}

	var nodeList corev1.NodeList
	if err := c.Ctrl.List(ctx, &nodeList, ctrlclient.MatchingLabelsSelector{Selector: sel}); err != nil {
		return fmt.Errorf("list nodes matching %q: %w", c.Config.GPUNodeSelector, err)
	}

	if len(nodeList.Items) == 0 {
		return fmt.Errorf("no nodes matched GPU node selector %q — is the cluster up and are its GPU nodes labeled", c.Config.GPUNodeSelector)
	}

	if len(nodeList.Items) != c.Config.GPUNodeCount {
		return fmt.Errorf("expected %d GPU node(s) matching %q, found %d", c.Config.GPUNodeCount, c.Config.GPUNodeSelector, len(nodeList.Items))
	}

	return nil
}

// ListNonGPUNodes returns nodes that do NOT match c.Config.GPUNodeSelector —
// the set the operator's node-selector must never touch (used by the
// non-GPU-nodes-untouched case, and to decide whether that topology precondition
// is satisfiable at all).
func ListNonGPUNodes(ctx context.Context, c *cluster.Client) ([]corev1.Node, error) {
	var all corev1.NodeList
	if err := c.Ctrl.List(ctx, &all); err != nil {
		return nil, fmt.Errorf("list all nodes: %w", err)
	}
	gpu, err := ListGPUNodes(ctx, c)
	if err != nil {
		return nil, err
	}
	isGPU := make(map[string]bool, len(gpu))
	for i := range gpu {
		isGPU[gpu[i].Name] = true
	}
	var out []corev1.Node
	for i := range all.Items {
		if !isGPU[all.Items[i].Name] {
			out = append(out, all.Items[i])
		}
	}
	return out, nil
}

// Get returns the named node.
func Get(ctx context.Context, c *cluster.Client, name string) (*corev1.Node, error) {
	var n corev1.Node
	if err := c.Ctrl.Get(ctx, ctrlclient.ObjectKey{Name: name}, &n); err != nil {
		return nil, err
	}
	return &n, nil
}

// Condition returns the named status condition on a node and whether it is
// present. The type is a string so callers can pass the operator's custom
// condition type (gpu-fractioning.nvidia.com/Ready).
func Condition(node *corev1.Node, condType string) (corev1.NodeCondition, bool) {
	for _, cond := range node.Status.Conditions {
		if string(cond.Type) == condType {
			return cond, true
		}
	}
	return corev1.NodeCondition{}, false
}

// WaitCondition polls the named node until its condType condition reaches
// status, returning the condition observed at that point.
func WaitCondition(ctx context.Context, c *cluster.Client, nodeName, condType string, status corev1.ConditionStatus, timeout, interval time.Duration) (corev1.NodeCondition, error) {
	var last corev1.NodeCondition
	err := waiter.PollUntil(ctx, timeout, interval,
		fmt.Sprintf("node %s condition %s=%s", nodeName, condType, status),
		func(ctx context.Context) (bool, error) {
			n, err := Get(ctx, c, nodeName)
			if err != nil {
				return false, err
			}
			cond, ok := Condition(n, condType)
			if !ok {
				return false, nil
			}
			last = cond
			return cond.Status == status, nil
		})
	return last, err
}

// WaitConditionAbsent polls the named node until its condType condition is no
// longer present (used to assert the cleanup finalizer removed it).
func WaitConditionAbsent(ctx context.Context, c *cluster.Client, nodeName, condType string, timeout, interval time.Duration) error {
	return waiter.PollUntil(ctx, timeout, interval,
		fmt.Sprintf("node %s condition %s to be removed", nodeName, condType),
		func(ctx context.Context) (bool, error) {
			n, err := Get(ctx, c, nodeName)
			if err != nil {
				return false, err
			}
			_, ok := Condition(n, condType)
			return !ok, nil
		})
}
