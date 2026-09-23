// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package workload creates and tears down single-purpose test pods used to
// exercise gpu-fractioning attribution. Separate from the plugin package,
// which manages the DaemonSet under test itself, not workloads that exercise
// it.
package workload

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
)

// DefaultImage is a minimal image with no GPU/CUDA dependency — attribution
// tests only need a live process holding a GPU allocation, not real compute.
const DefaultImage = "busybox:1.37"

// FractionalPod describes a single-container pod carrying the
// nvidia.com/container.<name>.gpu-memory.{limit,request} annotation that
// makes the fractiond NRI plugin track it — see
// fractioning-manager/fractiond/internal/adapter.go. Values/format mirror
// fractioning-manager/metricsd/test/workloads/test-fractional-gpu-pods.yaml.
type FractionalPod struct {
	Namespace     string
	Name          string
	ContainerName string

	// Annotations are written verbatim onto the pod. Nil or empty means the pod
	// carries no annotations at all (e.g. the full-GPU exclusion test). Build the
	// standard fractional-GPU request/limit pair with FractionalAnnotations, or
	// pass a map with a deliberately malformed key to exercise the plugin's
	// ignore / fail-closed paths.
	Annotations map[string]string

	// NodeSelector defaults to the cluster's configured GPU node selector
	// (c.Config.GPUNodeSelector) when nil — pass an explicit selector (or a
	// specific node via kubernetes.io/hostname) to control co-location.
	NodeSelector map[string]string

	// Marker is a unique token embedded in the container's command line so the
	// nvmlmock helper can resolve this container's host PID by scanning
	// /proc/<pid>/cmdline inside the nvml-mock pod (which runs hostPID: true).
	// Defaults to DefaultMarker(Namespace, Name) when empty.
	Marker string
}

// AnnotationKey builds a per-container gpu-memory annotation key using
// fractiond's default annotation prefix
// (configuration.DefaultAnnotationPrefix = "nvidia.com/container."):
//
//	nvidia.com/container.<container>.gpu-memory.<suffix>
func AnnotationKey(container, suffix string) string {
	return fmt.Sprintf("nvidia.com/container.%s.gpu-memory.%s", container, suffix)
}

// FractionalAnnotations builds the standard request+limit fractional-GPU
// annotation pair for a container. Values are MiB and get the "Mi" suffix:
// fractiond parses them as k8s resource.Quantity, so a bare "2048" would be read
// as 2048 bytes and rejected as below the 1 MiB minimum.
func FractionalAnnotations(container, requestMiB, limitMiB string) map[string]string {
	return map[string]string{
		AnnotationKey(container, "request"): requestMiB + "Mi",
		AnnotationKey(container, "limit"):   limitMiB + "Mi",
	}
}

// DefaultMarker is the process-name token Apply embeds in a pod's command line
// when FractionalPod.Marker is empty. The nvmlmock helper greps /proc for it to
// resolve the container's host-namespace PID, which it then pins as a mock NVML
// GPU process so per-pod memory/utilization becomes deterministic.
func DefaultMarker(namespace, name string) string {
	return fmt.Sprintf("gpumock-%s-%s", namespace, name)
}

// Apply creates the namespace (if missing) and the pod, and waits for the
// pod to reach Running. Any existing pod with the same name is deleted first so
// stale pods from a previous (interrupted) test run never cause "already exists"
// errors. The caller is responsible for calling Delete at the end of the test.
func Apply(ctx context.Context, c *cluster.Client, spec FractionalPod) (*corev1.Pod, error) {
	if err := EnsureNamespace(ctx, c, spec.Namespace); err != nil {
		return nil, fmt.Errorf("ensure namespace %s: %w", spec.Namespace, err)
	}
	if err := Delete(ctx, c, spec.Namespace, spec.Name); err != nil {
		return nil, fmt.Errorf("pre-create cleanup of stale pod %s/%s: %w", spec.Namespace, spec.Name, err)
	}

	nodeSelector := spec.NodeSelector
	if nodeSelector == nil {
		sel, err := labels.ConvertSelectorToLabelsMap(c.Config.GPUNodeSelector)
		if err != nil {
			return nil, fmt.Errorf("parse GPU node selector %q as a node selector: %w", c.Config.GPUNodeSelector, err)
		}
		nodeSelector = sel
	}

	marker := spec.Marker
	if marker == "" {
		marker = DefaultMarker(spec.Namespace, spec.Name)
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        spec.Name,
			Namespace:   spec.Namespace,
			Annotations: spec.Annotations,
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			NodeSelector:  nodeSelector,
			Containers: []corev1.Container{
				{
					Name:  spec.ContainerName,
					Image: DefaultImage,
					// The marker rides in the shell's command line (visible in
					// /proc/<pid>/cmdline) so nvmlmock.HostPID can find it. No
					// nvidia.com/gpu resource request: with nvml-mock there is no
					// device plugin, and attribution keys off the fractional
					// annotation + cgroup + NVML UUID, not a scheduled GPU.
					//
					// "sleep & wait" (not a bare "sleep # marker") is deliberate:
					// busybox ash exec-optimizes a single simple command, replacing
					// the shell with `sleep 86400` and dropping the `#` comment — so
					// no process would carry the marker. Backgrounding sleep and
					// waiting keeps the shell alive with the marker in its argv. The
					// matched PID (the shell) is in the pod's cgroup, which is all
					// attribution needs.
					Command: []string{"sh", "-c", fmt.Sprintf("sleep 86400 & wait # %s", marker)},
				},
			},
		},
	}

	if err := c.Ctrl.Create(ctx, pod); err != nil {
		return nil, fmt.Errorf("create pod %s/%s: %w", spec.Namespace, spec.Name, err)
	}

	return WaitRunning(ctx, c, spec.Namespace, spec.Name, c.Config.PodReadyTimeout, c.Config.PollInterval)
}

// WaitRunning polls until the pod's phase is Running, returning the latest
// observed pod object.
func WaitRunning(ctx context.Context, c *cluster.Client, namespace, name string, timeout, interval time.Duration) (*corev1.Pod, error) {
	var pod corev1.Pod
	err := waiter.PollUntil(ctx, timeout, interval, fmt.Sprintf("pod %s/%s to be Running", namespace, name),
		func(ctx context.Context) (bool, error) {
			if err := c.Ctrl.Get(ctx, ctrlclient.ObjectKey{Namespace: namespace, Name: name}, &pod); err != nil {
				return false, err
			}
			return pod.Status.Phase == corev1.PodRunning, nil
		})
	if err != nil {
		return nil, err
	}
	return &pod, nil
}

// Delete force-removes the pod (grace period 0 — test workloads' entrypoints
// don't handle SIGTERM, so a graceful delete would sit "Terminating" for the
// full default grace period) and waits for it to actually disappear, so a
// subsequent Apply with the same name doesn't collide with a
// still-terminating pod. Safe to call on an already-deleted pod.
//
// After the Pod object is gone from the API server, a short extra pause lets
// the container runtime finish killing the process. Without it, HostPID can
// find two processes with the same marker when a pod is recreated immediately:
// the old process (still visible in /proc on the host) and the new one.
func Delete(ctx context.Context, c *cluster.Client, namespace, name string) error {
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name}}
	err := c.Ctrl.Delete(ctx, pod, ctrlclient.GracePeriodSeconds(0))
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("delete pod %s/%s: %w", namespace, name, err)
	}

	if err := waiter.PollUntil(ctx, 30*time.Second, time.Second,
		fmt.Sprintf("pod %s/%s to be gone", namespace, name),
		func(ctx context.Context) (bool, error) {
			err := c.Ctrl.Get(ctx, ctrlclient.ObjectKey{Namespace: namespace, Name: name}, &corev1.Pod{})
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}); err != nil {
		return err
	}

	// The Pod object is gone but the container process may still be briefly
	// visible in /proc on the host node. HostPID greps /proc to find process
	// markers; without this pause it can match both the dying old process and
	// the new one when a pod is re-created with the same marker immediately.
	time.Sleep(2 * time.Second)
	return nil
}

// EnsureNamespace creates the namespace if it does not already exist.
func EnsureNamespace(ctx context.Context, c *cluster.Client, name string) error {
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if err := c.Ctrl.Create(ctx, ns); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}
