// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"slices"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// clusterPolicyWithSpec wraps a bare ClusterPolicy spec in a ClusterPolicy,
// the shape the detection helpers consume.
func clusterPolicyWithSpec(spec map[string]any) ClusterPolicy {
	obj := &unstructured.Unstructured{Object: map[string]any{"spec": spec}}
	obj.SetGroupVersionKind(clusterPolicyGVK)
	obj.SetName("cluster-policy")
	return ClusterPolicy{Policy: obj}
}

// toolkitEnvSpec builds a ClusterPolicy spec whose toolkit carries the given
// env vars, matching ClusterPolicy spec.toolkit.env ([]{name, value}).
func toolkitEnvSpec(env ...map[string]any) map[string]any {
	entries := make([]any, 0, len(env))
	for _, e := range env {
		entries = append(entries, e)
	}
	return map[string]any{"toolkit": map[string]any{"env": entries}}
}

func TestClusterPolicyNRIPluginEnabled(t *testing.T) {
	tests := []struct {
		name     string
		policy   ClusterPolicy
		expected bool
	}{
		{
			name:     "enabled",
			policy:   clusterPolicyWithSpec(map[string]any{"cdi": map[string]any{"nriPluginEnabled": true}}),
			expected: true,
		},
		{
			name:     "explicitly disabled",
			policy:   clusterPolicyWithSpec(map[string]any{"cdi": map[string]any{"nriPluginEnabled": false}}),
			expected: false,
		},
		{
			// Upstream defaults an unset nriPluginEnabled to false.
			name:     "field absent from cdi",
			policy:   clusterPolicyWithSpec(map[string]any{"cdi": map[string]any{"enabled": true}}),
			expected: false,
		},
		{
			name:     "cdi block absent",
			policy:   clusterPolicyWithSpec(map[string]any{"toolkit": map[string]any{"version": "v1.20.1"}}),
			expected: false,
		},
		{
			// A ClusterPolicy we cannot read must not flip the daemons off the
			// RuntimeClass path; the read error is reported on Ready instead.
			name:     "no cluster policy",
			policy:   ClusterPolicy{},
			expected: false,
		},
		{
			name:     "wrong type for the field",
			policy:   clusterPolicyWithSpec(map[string]any{"cdi": map[string]any{"nriPluginEnabled": "true"}}),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.policy.NRIPluginEnabled(); got != tt.expected {
				t.Fatalf("NRIPluginEnabled() = %t, expected %t", got, tt.expected)
			}
		})
	}
}

func TestClusterPolicyManagementCDIDeviceNamespaces(t *testing.T) {
	tests := []struct {
		name     string
		policy   ClusterPolicy
		expected []string
	}{
		{
			name: "single namespace",
			policy: clusterPolicyWithSpec(toolkitEnvSpec(
				map[string]any{"name": nriManagementCDIDeviceNamespacesEnv, "value": "gpu-fractioning"},
			)),
			expected: []string{"gpu-fractioning"},
		},
		{
			name: "comma separated",
			policy: clusterPolicyWithSpec(toolkitEnvSpec(
				map[string]any{"name": nriManagementCDIDeviceNamespacesEnv, "value": "gpu-operator,gpu-fractioning,other"},
			)),
			expected: []string{"gpu-operator", "gpu-fractioning", "other"},
		},
		{
			// A padded or trailing-comma list must not yield entries that can
			// never match a real namespace.
			name: "whitespace and empty entries",
			policy: clusterPolicyWithSpec(toolkitEnvSpec(
				map[string]any{"name": nriManagementCDIDeviceNamespacesEnv, "value": " gpu-operator , gpu-fractioning ,,"},
			)),
			expected: []string{"gpu-operator", "gpu-fractioning"},
		},
		{
			name: "found among other env vars",
			policy: clusterPolicyWithSpec(toolkitEnvSpec(
				map[string]any{"name": "CONTAINERD_CONFIG", "value": "/etc/containerd/config.toml"},
				map[string]any{"name": nriManagementCDIDeviceNamespacesEnv, "value": "gpu-fractioning"},
				map[string]any{"name": "RUNTIME_CLASS", "value": "nvidia"},
			)),
			expected: []string{"gpu-fractioning"},
		},
		{
			name: "env var absent",
			policy: clusterPolicyWithSpec(toolkitEnvSpec(
				map[string]any{"name": "CONTAINERD_CONFIG", "value": "/etc/containerd/config.toml"},
			)),
			expected: nil,
		},
		{
			name:     "toolkit env absent",
			policy:   clusterPolicyWithSpec(map[string]any{"toolkit": map[string]any{"version": "v1.20.1"}}),
			expected: nil,
		},
		{
			name:     "toolkit block absent",
			policy:   clusterPolicyWithSpec(map[string]any{"cdi": map[string]any{"nriPluginEnabled": true}}),
			expected: nil,
		},
		{
			name:     "no cluster policy",
			policy:   ClusterPolicy{},
			expected: nil,
		},
		{
			name: "set but empty",
			policy: clusterPolicyWithSpec(toolkitEnvSpec(
				map[string]any{"name": nriManagementCDIDeviceNamespacesEnv, "value": ""},
			)),
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.policy.ManagementCDIDeviceNamespaces()
			if !slices.Equal(got, tt.expected) {
				t.Fatalf("ManagementCDIDeviceNamespaces() = %v, expected %v", got, tt.expected)
			}
		})
	}
}
