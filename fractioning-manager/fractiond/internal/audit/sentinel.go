// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package audit

import (
	"context"
	"log/slog"
	"sync"

	"github.com/containerd/nri/pkg/api"
)

// Sentinel ties detection to remediation for retroactive enforcement on NRI
// reconnect. It is the package's public entry point; the detector and
// remediator it drives stay unexported so all policy lives here.
//
// Detection is synchronous and pure, so it is safe on the NRI callback
// goroutine. Remediation runs on a background goroutine because stopping a
// container blocks on the runtime's grace period, which must never stall the
// callback. Only the small []violator crosses into the goroutine — never the
// runtime snapshot slices.
type Sentinel struct {
	detector   detector
	remediator remediator
	wg         sync.WaitGroup
}

// Config is the subset of the fractiond plugin's configuration that decides
// what the create hook would have injected. Detection compares a running
// container against exactly that, so every field must carry the same value the
// hook was built with — a mismatch shows up as a false violation or a missed
// one.
type Config struct {
	// AnnotationPrefix is the leading GPU-fractioning annotation prefix, e.g.
	// "nvidia.com/container.".
	AnnotationPrefix string
	// MPSPipeDirectory is the base host-side MPS pipe directory.
	MPSPipeDirectory string
	// SMSharingEnabled is the cluster's installation-time sm-sharing chicken
	// bit, so a disabled feature can't produce a false violation for a
	// container the hook itself would have rejected or downgraded.
	SMSharingEnabled bool
	// FailOpen is the create hook's fail-open policy, which decides what an
	// unusable compute-mode annotation means: skip the container, or audit it
	// against the time-slicing default the hook fell back to.
	FailOpen bool
}

// NewSentinel builds a Sentinel from the create hook's configuration. stopper
// must be non-nil; log defaults to slog.Default() when nil.
func NewSentinel(cfg Config, stopper ContainerStopper, log *slog.Logger) *Sentinel {
	if log == nil {
		log = slog.Default()
	}
	return &Sentinel{
		detector: detector{
			annotationPrefix: cfg.AnnotationPrefix,
			mpsPipeDirectory: cfg.MPSPipeDirectory,
			smSharingEnabled: cfg.SMSharingEnabled,
			failOpen:         cfg.FailOpen,
			log:              log,
		},
		remediator: remediator{
			stopper: stopper,
			log:     log,
		},
	}
}

// Audit inspects the NRI Synchronize snapshot synchronously — cheap and safe on
// the callback goroutine — and, if it finds GPU-fractioning containers missing
// injection, stops them on a background goroutine. It returns immediately.
//
// The stop context is detached from ctx (WithoutCancel) so remediation outlives
// the NRI callback that triggered it; a Synchronize handler returning must not
// cancel in-flight stops. Call Wait to block until remediation completes.
func (s *Sentinel) Audit(ctx context.Context, pods []*api.PodSandbox, containers []*api.Container) {
	violators := s.detector.violators(pods, containers)
	if len(violators) == 0 {
		return
	}

	stopCtx := context.WithoutCancel(ctx)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.remediator.remediate(stopCtx, violators)
	}()
}

// Wait blocks until all in-flight remediation goroutines have finished. Used by
// graceful shutdown and by tests to observe remediation deterministically.
//
// It has no budget of its own. Each individual stop is bounded by the
// ContainerStopper's own timeout, but their sum is not, so a snapshot with many
// violators can hold shutdown for a long time. Both callers are exit paths
// where kubelet's termination grace period is the real ceiling, which is why
// this is left as it is; an NRI callback that needed a hard bound would want a
// WaitFor variant instead.
func (s *Sentinel) Wait() {
	s.wg.Wait()
}
