// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/containerd/nri/pkg/stub"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/audit"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/readiness"
)

func main() {
	flags := parseFlags()

	logger := configuration.NewLogger(flags.logLevel)

	// Readiness flips to ready only once the NRI plugin is registered and
	// synchronized (Plugin.Synchronize) and back off when the connection drops,
	// so the kubelet readiness probe reflects actual NRI registration.
	readyState := readiness.NewState()

	// Retroactive enforcement stops offending containers out of band through the
	// CRI runtime socket (NRI has no stop call). Build the stopper only when the
	// feature is on, and close it on exit.
	var stopper audit.ContainerStopper
	if flags.retroactiveEnforcement {
		cri := audit.NewCRIStopper(flags.criSocket, flags.stopTimeout, logger)
		defer func() { _ = cri.Close() }()
		stopper = cri
		logger.Info("retroactive enforcement enabled", "criSocket", flags.criSocket, "stopTimeout", flags.stopTimeout)
	}

	plugin, err := internal.NewPlugin(internal.Config{
		AnnotationPrefix:       flags.annotationPrefix,
		MPSPipeDirectory:       flags.mpsPipeDir,
		FailOpen:               flags.failOpen,
		SupportSMSharing:       flags.supportSMSharing,
		RetroactiveEnforcement: flags.retroactiveEnforcement,
		MapDir:                 flags.mapDir,
		LogPodEvents:           flags.logPodEvents,
		Log:                    logger,
		Readiness:              readyState,
	}, stopper)
	if err != nil {
		logger.Error("failed to create plugin", "error", err)
		os.Exit(1)
	}

	logger.Info("container→pod mapping handoff directory", "mapDir", flags.mapDir)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	if flags.readinessPort > 0 {
		go func() {
			// A listen failure leaves /readyz unreachable, which the kubelet
			// treats as not-ready — the safe direction — so log and keep running.
			if err := readiness.Serve(ctx, flags.readinessPort, readyState, logger); err != nil {
				logger.Error("readiness server failed", "error", err)
			}
		}()
	}

	err = runWithRetry(ctx, logger, plugin, readyState, flags)

	// Drain any queued mapping writes before exiting so the last events reach the
	// shared directory.
	plugin.Flush()

	if err != nil {
		logger.Error("fractiond exiting with error", "error", err)
		os.Exit(1)
	}

	logger.Info("fractiond shutdown complete")
}

// runWithRetry connects to the NRI runtime and runs the plugin. If the
// connection drops (e.g. containerd restarts or the NRI socket is not yet
// available at boot), it retries with exponential backoff (capped at 60s).
//
// Two intervals control behaviour:
//   - retryInterval: how long to wait before retrying (doubles each failure, capped)
//   - stableThreshold: how long a connection must stay up to be considered stable;
//     once stable, the attempt counter and backoff reset so a process that ran for
//     years doesn't exhaust its retry budget from earlier boot failures.
func runWithRetry(ctx context.Context, logger *slog.Logger, plugin *internal.Plugin, readyState *readiness.State, flags cliFlags) error {
	const maxBackoff = 60 * time.Second

	attempt := 0
	backoff := flags.retryInterval

	for {
		attempt++
		logger.Info("connecting to NRI runtime", "attempt", attempt, "plugin", flags.pluginName, "idx", flags.pluginIdx, "socket", flags.socketPath)

		s, err := stub.New(plugin,
			stub.WithPluginName(flags.pluginName),
			stub.WithPluginIdx(flags.pluginIdx),
			stub.WithSocketPath(flags.socketPath),
		)
		if err != nil {
			return fmt.Errorf("creating NRI stub: %w", err)
		}

		startTime := time.Now()
		err = s.Run(ctx)

		// s.Run only returns when the NRI connection is gone (or never came up),
		// so the plugin is no longer registered regardless of the cause. The
		// runtime's Shutdown callback also clears readiness, but it is not
		// guaranteed to fire on abrupt connection loss.
		readyState.SetReady(false)

		// ctx was created via signal.NotifyContext with no deadline, so the
		// only reason ctx.Err() can be non-nil is context.Canceled — meaning
		// SIGTERM or SIGINT was received. Log and exit cleanly.
		if ctx.Err() != nil {
			logger.Info("received shutdown signal, exiting cleanly")
			return nil
		}

		uptime := time.Since(startTime)

		// If the connection was stable (lasted longer than stableThreshold),
		// reset the retry budget and backoff.
		if uptime >= flags.stableThreshold {
			logger.Info("connection was stable, resetting retry budget", "uptime", uptime)
			attempt = 0
			backoff = flags.retryInterval
		}

		if flags.maxRetries > 0 && attempt >= flags.maxRetries {
			return fmt.Errorf("NRI connection failed after %d attempts: %w", attempt, err)
		}

		connState := "connection lost"
		if uptime < flags.retryInterval {
			connState = "connection failed"
		}

		logger.Warn(connState+", retrying",
			"error", err,
			"attempt", attempt,
			"uptime", uptime,
			"retryIn", backoff,
		)

		select {
		case <-ctx.Done():
			return nil
		case <-time.After(backoff):
		}

		// Exponential backoff: double interval, cap at maxBackoff.
		backoff = min(backoff*2, maxBackoff)
	}
}
