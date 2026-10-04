// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"maps"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// The key and value are a contract with the NVIDIA Container Toolkit's NRI
// plugin, which matches them literally. Pin them so a rename cannot silently
// stop GPU injection: the annotation would simply be ignored and the daemon
// would come up without a GPU.
func TestManagementCDIDeviceAnnotation(t *testing.T) {
	if got := managementCDIDeviceAnnotation("main"); got != "nvidia.cdi.k8s.io/container.main" {
		t.Fatalf(`managementCDIDeviceAnnotation("main") = %q, expected nvidia.cdi.k8s.io/container.main`, got)
	}

	if ManagementCDIDeviceAll != "management.nvidia.com/gpu=all" {
		t.Fatalf("ManagementCDIDeviceAll = %q, expected management.nvidia.com/gpu=all", ManagementCDIDeviceAll)
	}
}

func TestSetManagementCDIDevice(t *testing.T) {
	tests := []struct {
		name             string
		annotations      map[string]string
		nriPluginEnabled bool
		expected         map[string]string
	}{
		{
			name:             "adds the annotation to a pod with none",
			nriPluginEnabled: true,
			expected:         map[string]string{"nvidia.cdi.k8s.io/container.main": ManagementCDIDeviceAll},
		},
		{
			// A daemon's Prometheus scrape config lives in the same map;
			// assigning rather than merging would silently stop Prometheus
			// collecting GPU metrics.
			name: "merges alongside existing annotations",
			annotations: map[string]string{
				"prometheus.io/scrape": "true",
				"prometheus.io/port":   "2112",
				"prometheus.io/path":   "/metrics",
			},
			nriPluginEnabled: true,
			expected: map[string]string{
				"prometheus.io/scrape":             "true",
				"prometheus.io/port":               "2112",
				"prometheus.io/path":               "/metrics",
				"nvidia.cdi.k8s.io/container.main": ManagementCDIDeviceAll,
			},
		},
		{
			// Not in NRI mode: the pod template must come out untouched, not
			// carrying an empty annotation map.
			name:     "runtime class mode leaves annotations unset",
			expected: nil,
		},
		{
			name:        "runtime class mode leaves existing annotations alone",
			annotations: map[string]string{"prometheus.io/scrape": "true"},
			expected:    map[string]string{"prometheus.io/scrape": "true"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podMeta := &metav1.ObjectMeta{Annotations: tt.annotations}
			SetManagementCDIDevice(podMeta, "main", tt.nriPluginEnabled)
			if !maps.Equal(podMeta.Annotations, tt.expected) {
				t.Fatalf("annotations = %v, expected %v", podMeta.Annotations, tt.expected)
			}
		})
	}
}
