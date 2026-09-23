// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package daemonset provides generic, read-only inspection and rollout-wait
// helpers for Kubernetes DaemonSets. It is used by the e2e suites to assert
// deployment shape, args/mount propagation, and rollout convergence without
// shelling out to kubectl. It has no knowledge of any particular DaemonSet —
// callers supply the namespace/name (e.g. the operator suite derives those from
// its own managed-DaemonSet naming).
package daemonset

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
)

// Get returns the named DaemonSet in namespace.
func Get(ctx context.Context, c *cluster.Client, namespace, name string) (*appsv1.DaemonSet, error) {
	var ds appsv1.DaemonSet
	if err := c.Ctrl.Get(ctx, ctrlclient.ObjectKey{Namespace: namespace, Name: name}, &ds); err != nil {
		return nil, err
	}
	return &ds, nil
}

// Exists reports whether the named DaemonSet is present.
func Exists(ctx context.Context, c *cluster.Client, namespace, name string) (bool, error) {
	_, err := Get(ctx, c, namespace, name)
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// RolledOut reports whether a DaemonSet has fully converged to its current
// generation: the controller has observed the latest generation, every desired
// pod is updated and ready, and none are unavailable. A desired count of 0 (a
// selector matching no nodes) is a converged state too.
func RolledOut(ds *appsv1.DaemonSet) bool {
	s := ds.Status
	return s.ObservedGeneration == ds.Generation &&
		s.UpdatedNumberScheduled == s.DesiredNumberScheduled &&
		s.NumberReady == s.DesiredNumberScheduled &&
		s.NumberUnavailable == 0
}

// WaitRolledOut polls until the named DaemonSet is fully rolled out, returning
// the DaemonSet observed at that point.
func WaitRolledOut(ctx context.Context, c *cluster.Client, namespace, name string, timeout, interval time.Duration) (*appsv1.DaemonSet, error) {
	var latest *appsv1.DaemonSet
	err := waiter.PollUntil(ctx, timeout, interval, fmt.Sprintf("DaemonSet %s/%s to be rolled out", namespace, name),
		func(ctx context.Context) (bool, error) {
			ds, err := Get(ctx, c, namespace, name)
			if err != nil {
				return false, err
			}
			latest = ds
			return RolledOut(ds), nil
		})
	return latest, err
}

// WaitRolledOutAfter polls until the named DaemonSet has advanced past afterGen
// (metadata.generation) AND is fully rolled out at that newer generation.
//
// Use this after a CR change: WaitRolledOut alone can return the pre-change
// DaemonSet, because in the window between the CR patch and the operator's
// reconcile the old DaemonSet is still fully rolled out at its old generation
// (RolledOut is already true). A caller asserting on the new pod template (args,
// mounts) or on the fact that a rollout happened must also wait for the
// generation to advance. Capture the generation before applying the patch and
// pass it here.
func WaitRolledOutAfter(ctx context.Context, c *cluster.Client, namespace, name string, afterGen int64, timeout, interval time.Duration) (*appsv1.DaemonSet, error) {
	var latest *appsv1.DaemonSet
	err := waiter.PollUntil(ctx, timeout, interval,
		fmt.Sprintf("DaemonSet %s/%s to roll out past generation %d", namespace, name, afterGen),
		func(ctx context.Context) (bool, error) {
			ds, err := Get(ctx, c, namespace, name)
			if err != nil {
				return false, err
			}
			latest = ds
			return ds.Generation > afterGen && RolledOut(ds), nil
		})
	return latest, err
}

// WaitDesired polls until the named DaemonSet's desiredNumberScheduled equals
// want, returning the DaemonSet observed at that point. Useful when a selector
// change only converges after a reconcile — e.g. a zero-match selector settles
// to want=0.
func WaitDesired(ctx context.Context, c *cluster.Client, namespace, name string, want int32, timeout, interval time.Duration) (*appsv1.DaemonSet, error) {
	var latest *appsv1.DaemonSet
	err := waiter.PollUntil(ctx, timeout, interval,
		fmt.Sprintf("DaemonSet %s/%s desiredNumberScheduled=%d", namespace, name, want),
		func(ctx context.Context) (bool, error) {
			ds, err := Get(ctx, c, namespace, name)
			if err != nil {
				return false, err
			}
			latest = ds
			return ds.Status.DesiredNumberScheduled == want, nil
		})
	return latest, err
}

// WaitGone polls until the named DaemonSet no longer exists (owner-ref GC after
// the CR is deleted).
func WaitGone(ctx context.Context, c *cluster.Client, namespace, name string, timeout, interval time.Duration) error {
	return waiter.PollUntil(ctx, timeout, interval, fmt.Sprintf("DaemonSet %s/%s to be deleted", namespace, name),
		func(ctx context.Context) (bool, error) {
			ok, err := Exists(ctx, c, namespace, name)
			return !ok, err
		})
}

// Container returns the named container from the DaemonSet's pod template.
func Container(ds *appsv1.DaemonSet, containerName string) (corev1.Container, bool) {
	for _, cont := range ds.Spec.Template.Spec.Containers {
		if cont.Name == containerName {
			return cont, true
		}
	}
	return corev1.Container{}, false
}

// ContainerNames returns the pod-template container names in order.
func ContainerNames(ds *appsv1.DaemonSet) []string {
	names := make([]string, 0, len(ds.Spec.Template.Spec.Containers))
	for _, cont := range ds.Spec.Template.Spec.Containers {
		names = append(names, cont.Name)
	}
	return names
}

// HostPathMounts returns the set of host paths mounted into the named container
// (mount path → true), for asserting expected hostPath volumes are wired in.
func HostPathMounts(ds *appsv1.DaemonSet, containerName string) map[string]bool {
	out := map[string]bool{}
	cont, ok := Container(ds, containerName)
	if !ok {
		return out
	}
	// Map volume name → hostPath.path for volumes that are hostPaths.
	hostPaths := map[string]string{}
	for _, v := range ds.Spec.Template.Spec.Volumes {
		if v.HostPath != nil {
			hostPaths[v.Name] = v.HostPath.Path
		}
	}
	for _, m := range cont.VolumeMounts {
		if p, isHostPath := hostPaths[m.Name]; isHostPath {
			out[p] = true
		}
	}
	return out
}
