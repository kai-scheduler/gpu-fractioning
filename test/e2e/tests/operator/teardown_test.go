// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package operator

import (
	"context"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/harness"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/daemonset"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/nodes"
)

// TeardownAndCleanup — deleting the CR garbage-collects both DaemonSets
// AND the node-condition cleanup finalizer removes the gpu-fractioning condition from
// every previously-targeted node. Re-applying the default restores FX-STEADY.
func caseTeardownAndCleanup(ctx context.Context, t *testing.T) {
	targeted := h.GPUNodeNames(ctx, t)

	h.CleanCR(ctx, t) // deletes CR + waits both DaemonSets GC'd (FX-CLEAN)

	// The finalizer must strip the node condition from every targeted node.
	for _, name := range targeted {
		if err := nodes.WaitConditionAbsent(ctx, h.Client(), name, harness.NodeConditionType, h.RolloutTimeout(), h.PollInterval()); err != nil {
			t.Errorf("node %s still carries the gpu-fractioning condition after CR deletion: %v", name, err)
		}
	}
	// Sanity: DaemonSets are gone (cleanCR already waited, but assert explicitly).
	for _, comp := range []string{harness.ComponentFractiond, harness.ComponentMpsd} {
		if ok, err := daemonset.Exists(ctx, h.Client(), h.NS(), harness.DSName(comp)); err != nil {
			t.Errorf("check %s existence: %v", harness.DSName(comp), err)
		} else if ok {
			t.Errorf("%s still exists after CR deletion", harness.DSName(comp))
		}
	}

	// Restore FX-STEADY for the between-phase assertion and any later runs.
	h.RestoreSteady(ctx, t)
}
