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
	tests := []struct {
		containerName string
		expected      string
	}{
		{containerName: "mpsd", expected: "nvidia.cdi.k8s.io/container.mpsd"},
		{containerName: "metricsd", expected: "nvidia.cdi.k8s.io/container.metricsd"},
	}

	for _, tt := range tests {
		t.Run(tt.containerName, func(t *testing.T) {
			if got := managementCDIDeviceAnnotation(tt.containerName); got != tt.expected {
				t.Fatalf("managementCDIDeviceAnnotation(%q) = %q, expected %q", tt.containerName, got, tt.expected)
			}
		})
	}

	if ManagementCDIDeviceAll != "management.nvidia.com/gpu=all" {
		t.Fatalf("ManagementCDIDeviceAll = %q, expected management.nvidia.com/gpu=all", ManagementCDIDeviceAll)
	}
}

func TestSetManagementCDIDevice(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		device      string
		expected    map[string]string
	}{
		{
			name:     "adds the annotation to a pod with none",
			device:   ManagementCDIDeviceAll,
			expected: map[string]string{"nvidia.cdi.k8s.io/container.mpsd": ManagementCDIDeviceAll},
		},
		{
			// fractiond's scrape config lives in the same map; assigning rather
			// than merging would silently stop Prometheus collecting GPU metrics.
			name: "merges alongside existing annotations",
			annotations: map[string]string{
				"prometheus.io/scrape": "true",
				"prometheus.io/port":   "2112",
				"prometheus.io/path":   "/metrics",
			},
			device: ManagementCDIDeviceAll,
			expected: map[string]string{
				"prometheus.io/scrape":             "true",
				"prometheus.io/port":               "2112",
				"prometheus.io/path":               "/metrics",
				"nvidia.cdi.k8s.io/container.mpsd": ManagementCDIDeviceAll,
			},
		},
		{
			// Not in NRI mode: the pod template must come out untouched, not
			// carrying an empty annotation map.
			name:     "no device leaves annotations unset",
			device:   "",
			expected: nil,
		},
		{
			name:        "no device leaves existing annotations alone",
			annotations: map[string]string{"prometheus.io/scrape": "true"},
			device:      "",
			expected:    map[string]string{"prometheus.io/scrape": "true"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podMeta := &metav1.ObjectMeta{Annotations: tt.annotations}
			SetManagementCDIDevice(podMeta, "mpsd", tt.device)
			if !maps.Equal(podMeta.Annotations, tt.expected) {
				t.Fatalf("annotations = %v, expected %v", podMeta.Annotations, tt.expected)
			}
		})
	}
}
