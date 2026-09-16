// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package operator

import (
	"context"
	"testing"

	"github.com/kai-scheduler/kai-gpu-fractioning/test/e2e/harness"
	"github.com/kai-scheduler/kai-gpu-fractioning/test/e2e/k8s/daemonset"
)

// MpsdConfigToArgs — mpsDaemon config propagates to the mpsd container
// args, re-rolling the DaemonSet.
func caseMpsdConfigToArgs(ctx context.Context, t *testing.T) {
	// The shipped default leaves mpsDaemon unset, so setting logLevel is a real
	// change; wait for the DaemonSet to roll out PAST its current generation so
	// we read the reconciled pod template, not the pre-patch one.
	beforeGen := h.GetDaemonSet(ctx, t, harness.ComponentMpsd).Generation
	patch := []byte(`{"spec":{"mpsDaemon":{"logLevel":"debug"}}}`)
	revert := []byte(`{"spec":{"mpsDaemon":{"logLevel":null}}}`)
	h.WithCRConfig(ctx, t, patch, revert, func() {
		ds, err := daemonset.WaitRolledOutAfter(ctx, h.Client(), h.NS(), harness.DSName(harness.ComponentMpsd), beforeGen, h.RolloutTimeout(), h.PollInterval())
		if err != nil {
			t.Fatalf("mpsd rollout after logLevel change: %v", err)
		}
		cont, _ := daemonset.Container(ds, containerMpsd)
		if !argsHasFlagValue(cont.Args, "--log-level", "debug") {
			t.Errorf("mpsd args = %v, want --log-level debug", cont.Args)
		}
	})
}

// FractiondConfigToArgs — fractioningAgent config propagates to the fractiond
// container args. Uses "warn" (the shipped default is "debug") so the change
// actually re-rolls the DaemonSet and the assertion isn't trivially satisfied.
func caseFractiondConfigToArgs(ctx context.Context, t *testing.T) {
	beforeGen := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation
	patch := []byte(`{"spec":{"fractioningAgent":{"logLevel":"warn"}}}`)
	revert := []byte(`{"spec":{"fractioningAgent":{"logLevel":null}}}`)
	h.WithCRConfig(ctx, t, patch, revert, func() {
		ds, err := daemonset.WaitRolledOutAfter(ctx, h.Client(), h.NS(), harness.DSName(harness.ComponentFractiond), beforeGen, h.RolloutTimeout(), h.PollInterval())
		if err != nil {
			t.Fatalf("fractiond rollout after logLevel change: %v", err)
		}
		cont, _ := daemonset.Container(ds, containerFractiond)
		if !argsHasFlagValue(cont.Args, "--log-level", "warn") {
			t.Errorf("fractiond args = %v, want --log-level warn", cont.Args)
		}
	})
}

// CustomAnnotationPrefix — a custom annotationPrefix propagates to
// fractiond's args AND changes runtime behavior: a container annotated with the
// custom prefix is injected, while the default prefix is ignored.
func caseCustomAnnotationPrefix(ctx context.Context, t *testing.T) {
	const customPrefix = "custom.example.com/c."
	beforeGen := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation
	patch := []byte(`{"spec":{"fractioningAgent":{"annotationPrefix":"` + customPrefix + `"}}}`)
	revert := []byte(`{"spec":{"fractioningAgent":{"annotationPrefix":null}}}`)
	h.WithCRConfig(ctx, t, patch, revert, func() {
		ds, err := daemonset.WaitRolledOutAfter(ctx, h.Client(), h.NS(), harness.DSName(harness.ComponentFractiond), beforeGen, h.RolloutTimeout(), h.PollInterval())
		if err != nil {
			t.Fatalf("fractiond rollout after annotationPrefix change: %v", err)
		}
		cont, _ := daemonset.Container(ds, containerFractiond)
		if !argsHasFlagValue(cont.Args, "--annotation-prefix", customPrefix) {
			t.Errorf("fractiond args = %v, want --annotation-prefix %s", cont.Args, customPrefix)
		}

		// Functional: a workload using the custom prefix gets injected.
		mib := harness.MemRequestMiB
		pod := h.ApplyRunningWorkload(ctx, t, "f4-customprefix", map[string]string{
			customPrefix + harness.WorkloadContainer + ".gpu-memory.request": mib + "Mi",
			customPrefix + harness.WorkloadContainer + ".gpu-memory.limit":   mib + "Mi",
			// A default-prefix annotation must now be ignored.
			h.AnnKey(harness.WorkloadContainer, "request"): "9999Mi",
		})
		env := h.GetPodEnv(ctx, t, pod, harness.WorkloadContainer)
		want := mib
		if env[harness.EnvGPUMemRequest] != want {
			t.Errorf("custom-prefix injection: %s = %q, want %q (default-prefix value must be ignored)", harness.EnvGPUMemRequest, env[harness.EnvGPUMemRequest], want)
		}
	})
}

// RollingUpdateConverges — managed DaemonSets use a RollingUpdate
// strategy, and a spec change rolls them out to convergence (rather than
// replacing all pods at once).
func caseRollingUpdateConverges(ctx context.Context, t *testing.T) {
	for _, comp := range []string{harness.ComponentFractiond, harness.ComponentMpsd} {
		ds := h.GetDaemonSet(ctx, t, comp)
		if ds.Spec.UpdateStrategy.Type != "RollingUpdate" {
			t.Errorf("%s updateStrategy = %q, want RollingUpdate", ds.Name, ds.Spec.UpdateStrategy.Type)
		}
	}

	// A config change rolls fractiond out and converges. Wait for the generation
	// to advance past `before` (WaitRolledOut alone would return the pre-patch
	// DaemonSet, still rolled out at `before`, before the operator reconciles).
	before := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation
	patch := []byte(`{"spec":{"fractioningAgent":{"logLevel":"warn"}}}`)
	revert := []byte(`{"spec":{"fractioningAgent":{"logLevel":null}}}`)
	h.WithCRConfig(ctx, t, patch, revert, func() {
		ds, err := daemonset.WaitRolledOutAfter(ctx, h.Client(), h.NS(), harness.DSName(harness.ComponentFractiond), before, h.RolloutTimeout(), h.PollInterval())
		if err != nil {
			t.Fatalf("fractiond rollout on config change: %v", err)
		}
		if ds.Generation <= before {
			t.Errorf("fractiond generation did not advance on config change: %d → %d", before, ds.Generation)
		}
	})
}

// argsHasFlagValue reports whether args contains flag immediately followed by value.
func argsHasFlagValue(args []string, flag, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}
