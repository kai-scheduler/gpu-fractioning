// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package harness

import (
	"context"
	"fmt"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/pods"
)

func (h *Harness) ListComponentPods(ctx context.Context, t *testing.T, component string) []corev1.Pod {
	t.Helper()
	list, err := pods.ListByLabel(ctx, h.Client(), h.NS(), ComponentSelector(component))
	if err != nil {
		t.Fatalf("list %s pods: %v", component, err)
	}
	return list
}

// FirstComponentPod returns any one pod of a component (fatal if none). It only
// needs a single pod, so it caps the server-side list to one item rather than
// fetching every pod of the component.
func (h *Harness) FirstComponentPod(ctx context.Context, t *testing.T, component string) *corev1.Pod {
	t.Helper()
	sel := ComponentSelector(component)
	pod, err := pods.FirstByLabel(ctx, h.Client(), h.NS(), sel)
	if err != nil {
		t.Fatalf("list %s pods: %v", component, err)
	}
	if pod == nil {
		t.Fatalf("no %s pods found (selector %q)", component, sel)
	}
	return pod
}

// PodOnNode returns the component pod scheduled on nodeName, if any.
func (h *Harness) PodOnNode(ctx context.Context, t *testing.T, component, nodeName string) (*corev1.Pod, bool) {
	t.Helper()
	list := h.ListComponentPods(ctx, t, component)
	for i := range list {
		if list[i].Spec.NodeName == nodeName {
			return &list[i], true
		}
	}
	return nil, false
}

// GetPod re-reads a pod by key.
func (h *Harness) GetPod(ctx context.Context, t *testing.T, namespace, name string) *corev1.Pod {
	t.Helper()
	var p corev1.Pod
	if err := h.Client().Ctrl.Get(ctx, ctrlclient.ObjectKey{Namespace: namespace, Name: name}, &p); err != nil {
		t.Fatalf("get pod %s/%s: %v", namespace, name, err)
	}
	return &p
}

func ContainerStatus(pod *corev1.Pod, name string) (corev1.ContainerStatus, bool) {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == name {
			return cs, true
		}
	}
	return corev1.ContainerStatus{}, false
}

func ContainerReady(pod *corev1.Pod, name string) bool {
	cs, ok := ContainerStatus(pod, name)
	return ok && cs.Ready
}

func ContainerRestarts(pod *corev1.Pod, name string) int32 {
	cs, ok := ContainerStatus(pod, name)
	if !ok {
		return 0
	}
	return cs.RestartCount
}

func PodReady(pod *corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady {
			return cond.Status == corev1.ConditionTrue
		}
	}
	return false
}

// Exec runs a command in a pod container and returns trimmed stdout.
func (h *Harness) Exec(ctx context.Context, pod *corev1.Pod, container string, command ...string) (string, error) {
	out, err := pods.Exec(ctx, h.Client(), pod.Namespace, pod.Name, container, command)
	return strings.TrimSpace(out), err
}

// GetPodEnv dumps the container's environment (via the `env` command) into a
// map. Used to
// assert presence/absence/values of fractiond-injected vars without depending on
// a command's exit code.
func (h *Harness) GetPodEnv(ctx context.Context, t *testing.T, pod *corev1.Pod, container string) map[string]string {
	t.Helper()
	out, err := h.Exec(ctx, pod, container, "env")
	if err != nil {
		t.Fatalf("exec env in %s/%s [%s]: %v", pod.Namespace, pod.Name, container, err)
	}
	m := map[string]string{}
	for _, line := range strings.Split(out, "\n") {
		if k, v, ok := strings.Cut(strings.TrimSpace(line), "="); ok {
			m[k] = v
		}
	}
	return m
}

// SocketExists reports whether path is a socket inside the container. Uses an
// explicit yes/no echo so a missing socket is unambiguous (not conflated with an
// exec transport error).
func (h *Harness) SocketExists(ctx context.Context, t *testing.T, pod *corev1.Pod, container, path string) bool {
	t.Helper()
	out, err := h.Exec(ctx, pod, container, "sh", "-c", fmt.Sprintf("test -S %s && echo yes || echo no", path))
	if err != nil {
		t.Fatalf("exec socket check in %s/%s: %v", pod.Namespace, pod.Name, err)
	}
	return out == "yes"
}
