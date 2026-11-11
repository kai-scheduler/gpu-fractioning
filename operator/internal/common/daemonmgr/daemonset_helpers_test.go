// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestMergeResources(t *testing.T) {
	base := DaemonResources("50m", "64Mi", "1Gi")

	t.Run("nil override keeps the defaults", func(t *testing.T) {
		got := MergeResources(base, nil)
		if got.Requests.Cpu().String() != "50m" || got.Requests.Memory().String() != "64Mi" ||
			got.Limits.Memory().String() != "1Gi" {
			t.Fatalf("unexpected resources: %v", got)
		}
	})

	t.Run("a raised memory limit keeps the default requests", func(t *testing.T) {
		got := MergeResources(base, &corev1.ResourceRequirements{
			Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("2Gi")},
		})
		if got.Limits.Memory().String() != "2Gi" {
			t.Errorf("memory limit = %q, want 2Gi", got.Limits.Memory().String())
		}
		if got.Requests.Cpu().String() != "50m" || got.Requests.Memory().String() != "64Mi" ||
			got.Requests.StorageEphemeral().String() != "100Mi" {
			t.Errorf("requests changed: %v", got.Requests)
		}
		if base.Limits.Memory().String() != "1Gi" {
			t.Errorf("base was mutated: %v", base.Limits)
		}
	})

	t.Run("a CPU limit can be added", func(t *testing.T) {
		got := MergeResources(base, &corev1.ResourceRequirements{
			Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
			Limits:   corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
		})
		if got.Requests.Cpu().String() != "200m" || got.Limits.Cpu().String() != "1" {
			t.Errorf("CPU not applied: %v", got)
		}
		if got.Limits.Memory().String() != "1Gi" {
			t.Errorf("memory limit changed: %v", got.Limits)
		}
	})
}
