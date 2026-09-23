// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package harness

import (
	"context"
	"sort"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/nodes"
)

// GPUNodeNames returns the sorted names of nodes matching the configured GPU
// node selector.
func (h *Harness) GPUNodeNames(ctx context.Context, t *testing.T) []string {
	t.Helper()
	list, err := nodes.ListGPUNodes(ctx, h.Client())
	if err != nil {
		t.Fatalf("list GPU nodes: %v", err)
	}
	names := make([]string, 0, len(list))
	for i := range list {
		names = append(names, list[i].Name)
	}
	sort.Strings(names)
	return names
}

// NonGPUNodeNames returns the sorted names of nodes NOT matching the configured
// GPU node selector.
func (h *Harness) NonGPUNodeNames(ctx context.Context, t *testing.T) []string {
	t.Helper()
	list, err := nodes.ListNonGPUNodes(ctx, h.Client())
	if err != nil {
		t.Fatalf("list non-GPU nodes: %v", err)
	}
	names := make([]string, 0, len(list))
	for i := range list {
		names = append(names, list[i].Name)
	}
	sort.Strings(names)
	return names
}
