// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package harness

import (
	"context"
	"strings"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/pods"
)

// RuntimeGPUUUIDs uses the same simulated inventory as the real device plugin,
// so the toolkit can resolve the assignments exercised by the injection tests.
func (h *Harness) RuntimeGPUUUIDs(ctx context.Context, t *testing.T) []string {
	t.Helper()
	items, err := pods.ListByLabel(ctx, h.Client(), "mokka", "app.kubernetes.io/instance=nvml-mock")
	if err != nil || len(items) == 0 {
		t.Fatalf("find Mokka inventory pod: pods=%d err=%v", len(items), err)
	}
	out, err := pods.Exec(ctx, h.Client(), "mokka", items[0].Name, "node-agent",
		[]string{"nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"})
	if err != nil {
		t.Fatalf("read Mokka GPU UUIDs: %v", err)
	}
	ids := strings.Fields(out)
	if len(ids) == 0 {
		t.Fatal("Mokka inventory contains no GPUs")
	}
	return ids
}
