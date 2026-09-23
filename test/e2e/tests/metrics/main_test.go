// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

// Package metrics holds the gpu-fractioning metrics e2e suite. Run just this
// suite with:
//
//	make test-e2e-metrics
//
// or, against a cluster already created and with the plugin image already
// loaded:
//
//	cd test/e2e && go test -tags e2e ./tests/metrics/... -v -timeout 20m
//
// Each e2e suite lives in its own tests/<suite> package with its own TestMain,
// so a CI job can target exactly one suite (e.g. the Metrics e2e job runs only
// ./tests/metrics/...). See test/e2e/README.md for the full workflow. This
// suite does not create clusters or deploy the stack — the operator Helm chart
// (make e2e-deploy) does that; the suite runs an ordered setup
// (connect → preflight) against an already-deployed cluster and asserts.
package metrics

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/suite"
)

var s *suite.Suite

func TestMain(m *testing.M) {
	ctx := context.Background()

	created, err := suite.New(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "e2e suite setup failed: connect: %v\n", err)
		os.Exit(1)
	}
	s = created

	// Preflight is an explicit, fast-fail precondition check, so a misconfigured
	// cluster fails immediately with a clear message instead of an opaque
	// assertion failure. The stack itself is deployed out-of-band by the
	// operator Helm chart (make e2e-deploy) before the suite runs.
	if err := s.Preflight(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "e2e preflight failed: %v\n", err)
		os.Exit(1)
	}

	os.Exit(m.Run())
}
