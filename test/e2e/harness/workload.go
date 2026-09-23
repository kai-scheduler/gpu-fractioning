// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package harness

import (
	"context"
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// GPUSelectorMap converts the configured GPU node selector into a nodeSelector
// map for pinning workload pods to matching nodes.
func (h *Harness) GPUSelectorMap(t *testing.T) map[string]string {
	t.Helper()
	m, err := labels.ConvertSelectorToLabelsMap(h.Client().Config.GPUNodeSelector)
	if err != nil {
		t.Fatalf("parse GPU node selector %q: %v", h.Client().Config.GPUNodeSelector, err)
	}
	return m
}

// AnnKey builds a per-container gpu-memory annotation key with the default prefix.
func (h *Harness) AnnKey(container, suffix string) string {
	return fmt.Sprintf("%s%s.gpu-memory.%s", AnnotationPrefix, container, suffix)
}

// DevicesAnnotationKey builds the per-container GPU device-assignment annotation
// key (…gpus.devices) that the scheduler emits and fractiond promotes to
// NVIDIA_VISIBLE_DEVICES.
func (h *Harness) DevicesAnnotationKey(container string) string {
	return fmt.Sprintf("%s%s.gpus.devices", AnnotationPrefix, container)
}

// ComputeModeAnnotationKey builds the per-container compute-mode annotation key
// (…gpu-compute.mode) that selects which MPS socket fractiond mounts:
// time-slicing (the default when absent) or sm-sharing.
func (h *Harness) ComputeModeAnnotationKey(container string) string {
	return fmt.Sprintf("%s%s.gpu-compute.mode", AnnotationPrefix, container)
}

// ComputeModeAnnotations returns the standard fractional-GPU annotations plus an
// explicit compute mode. sm-sharing is only honored on a container that is
// already GPU-fractioning (fractiond evaluates compute mode only once the
// container has GPU-memory config), so the memory pair always comes along.
func (h *Harness) ComputeModeAnnotations(mode string) map[string]string {
	annotations := h.DefaultWorkloadAnnotations()
	annotations[h.ComputeModeAnnotationKey(WorkloadContainer)] = mode
	return annotations
}

// DefaultWorkloadAnnotations returns the standard request+limit fractional-GPU
// annotations for the shared workload container and memory values — the pair a
// well-formed fractional pod carries.
func (h *Harness) DefaultWorkloadAnnotations() map[string]string {
	return workload.FractionalAnnotations(WorkloadContainer, MemRequestMiB, MemLimitMiB)
}

// ApplyRunningWorkload creates a busybox pod on a GPU node with the given
// annotations (written verbatim; nil/empty ⇒ none) and waits for it to reach
// Running, registering cleanup. Use DefaultWorkloadAnnotations for the standard
// well-formed fractional pod.
func (h *Harness) ApplyRunningWorkload(ctx context.Context, t *testing.T, name string, annotations map[string]string) *corev1.Pod {
	t.Helper()
	spec := workload.FractionalPod{
		Namespace:     WorkloadNamespace,
		Name:          name,
		ContainerName: WorkloadContainer,
		Annotations:   annotations,
	}
	pod, err := workload.Apply(ctx, h.Client(), spec)
	if err != nil {
		t.Fatalf("apply workload %s: %v", name, err)
	}
	t.Cleanup(func() { _ = workload.Delete(context.Background(), h.Client(), WorkloadNamespace, name) })
	return pod
}
