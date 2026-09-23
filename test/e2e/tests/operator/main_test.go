// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

// Package operator holds the gpu-fractioning operator/controller e2e suite. It
// covers the controller surface EXCEPT metrics (owned by tests/metrics) and the
// fractiond NRI injection (owned by tests/fractiond): DaemonSet deployment shape,
// CR status + node conditions, fault injection/recovery, config propagation, and
// teardown.
//
// Run just this suite:
//
//	make test-e2e-operator
//
// or, against any already-deployed cluster (real or fake GPU — the suite is
// cluster-agnostic and provisions nothing):
//
//	cd test/e2e && go test -tags e2e ./tests/operator/... -v -timeout 40m
//
// The stack is deployed out-of-band by the operator Helm chart (make
// e2e-deploy); the suite connects, runs an ordered set of cases against
// FX-STEADY (the shipped default CR), and asserts.
package operator

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/harness"
)

// h is the shared harness (one cluster connection + FX-STEADY snapshot), set in
// TestMain and used by every case via the shims in aliases_test.go.
var h *harness.Harness

func TestMain(m *testing.M) {
	ctx := context.Background()

	created, err := harness.New(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "operator e2e setup failed: connect: %v\n", err)
		os.Exit(1)
	}
	h = created

	// Fast-fail precondition: GPU nodes present and the fractiond DaemonSet
	// deployed. Full FX-STEADY (CR Ready, mpsd Ready, node conditions) is
	// asserted at the top of TestOperator via assertSteady.
	if err := h.Preflight(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "operator e2e preflight failed: %v\n", err)
		os.Exit(1)
	}

	os.Exit(m.Run())
}
