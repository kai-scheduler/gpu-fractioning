// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

// Package harness holds the shared, importable e2e test support used by the
// operator and fractiond suites: the cluster connection, FX-STEADY fixture
// machinery, workload/exec/pod/node/daemonset helpers, diagnostics, and the
// cross-suite constants.
//
// Unlike a suite's own _test.go files (which Go can't import from another
// package), this is a regular package so multiple suites can reuse it. The
// files carry //go:build e2e, so they are excluded from a normal `go build`
// and only compiled under `go test -tags e2e`.
//
// A Harness wraps one suite.Suite (one cluster connection) plus the FX-STEADY
// snapshot; its methods take *testing.T per call so subtests get their own t.
package harness

import (
	"context"
	"time"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	// Import for its init(): registers the GpuFractioningConfig types into the
	// scheme the shared cluster client uses, so typed CR access works.
	_ "github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/gpufractioningconfig"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/suite"
)

// Harness is the shared state for a run of an e2e suite. It only connects to an
// existing cluster (whatever E2E_KUBECONFIG points at); it never provisions one.
type Harness struct {
	S *suite.Suite

	// originalSpec is the shipped default CR's spec, snapshotted the first time
	// the suite observes FX-STEADY (see AssertSteady). Lifecycle cases re-create
	// the CR from this snapshot so they restore the exact shipped steady state
	// (rather than a hand-built approximation that might omit chart defaults).
	originalSpec *v1alpha1.GpuFractioningConfigSpec
}

// New connects to the cluster. It does not verify preconditions — callers drive
// that with Preflight so the ordering is explicit in each suite's TestMain.
func New(ctx context.Context) (*Harness, error) {
	s, err := suite.New(ctx)
	if err != nil {
		return nil, err
	}
	return &Harness{S: s}, nil
}

// Preflight runs the suite's fast, side-effect-free precondition check.
func (h *Harness) Preflight(ctx context.Context) error { return h.S.Preflight(ctx) }

// Client returns the shared cluster client.
func (h *Harness) Client() *cluster.Client { return h.S.Client }

// NS returns the operator namespace the stack is installed into.
func (h *Harness) NS() string { return h.S.Client.Config.OperatorNamespace }

func (h *Harness) RolloutTimeout() time.Duration { return h.S.Client.Config.RolloutTimeout }
func (h *Harness) CondTimeout() time.Duration    { return h.S.Client.Config.CondTimeout }
func (h *Harness) PollInterval() time.Duration   { return h.S.Client.Config.PollInterval }
