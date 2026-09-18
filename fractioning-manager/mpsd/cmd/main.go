// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/drain"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/driverlabel"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/mpsctl"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/mpsns"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/procfs"
)

func main() {
	flags := parseFlags()

	logger := configuration.NewLogger(flags.logLevel)
	logger.Info("starting mpsd",
		"pipeDir", flags.pipeDir,
		"logDir", flags.logDir,
		"mpsBinary", flags.mpsBinary,
		"controlPort", flags.controlPort,
		"configPath", flags.configPath,
		"memacctAuditLog", flags.memacctAuditLog,
		"supportSMSharing", flags.supportSMSharing,
		"drainSocket", flags.drainSocket,
		"recycleWhenIdle", flags.recycleWhenIdle,
		"namespaceIsolation", flags.namespaceIsolation,
	)

	// Render the MPS control-daemon config. memacct is always on; only the
	// audit log is configurable (via Helm value -> MPS_MEMACCT_AUDIT_LOG env).
	// context-share adds a parameterless shared MPS server (with its default
	// per-container socket disabled) that fractiond routes sm-sharing
	// containers to instead of the default per-node socket — gated by the
	// sm-sharing installation-time chicken bit (Helm value ->
	// MPS_SUPPORT_SM_SHARING env) so the feature can be killed without a code
	// rollback; when disabled, the config is rendered exactly as it was before
	// the sm-sharing feature existed (as is the daemon invocation — see
	// Multiuser below). The supervisor writes this to configPath at setup.
	mpsConfig := internal.MPSConfig{
		MemacctEnabled:  internal.DefaultMemacctEnabled,
		MemacctAuditLog: flags.memacctAuditLog,
	}
	if flags.supportSMSharing {
		mpsConfig.ContextShareEnabled = internal.DefaultContextShareEnabled
		mpsConfig.ContextShareDefaultSocket = internal.DefaultContextShareDefaultSocket
		mpsConfig.SharedServerName = internal.DefaultSharedServerName
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	// Label the node before starting MPS so operator dependency diagnostics use
	// the driver version mpsd sees locally through NVML.
	if err := driverlabel.LabelCurrentNode(ctx, logger); err != nil {
		logger.Error("label node with NVIDIA driver major version", "error", err)
		os.Exit(1)
	}

	// The control-daemon client every namespace and drain command goes through.
	// One instance: they all talk to the same daemon over the same socket, and
	// a second one configured differently would talk to a different daemon
	// without saying so.
	control := mpsctl.New(mpsctl.Options{
		Binary:      flags.mpsBinary,
		ControlPort: flags.controlPort,
		PipeDir:     flags.pipeDir,
		Run:         mpsctl.NewExecRunner(flags.pipeDir),
		Log:         logger,
	})

	// Per-container MPS namespaces are what make a compute cap enforceable
	// rather than advisory. They live on the shared server, so they are gated
	// by the same sm-sharing chicken bit that creates it: with sm-sharing off
	// there is no server to put them on.
	namespaces := buildNamespaceManager(flags, control, logger)

	supervisor := internal.NewSupervisor(internal.SupervisorConfig{
		MPSBinary:     flags.mpsBinary,
		ControlPort:   flags.controlPort,
		ConfigPath:    flags.configPath,
		ConfigContent: mpsConfig.TOML(),
		// Multiuser mode is the security boundary the per-container compute cap
		// rests on, not just a convenience for mixed UIDs — see buildMPSArgs.
		// It follows the same toggle as the config above: killing sm-sharing
		// must leave neither half behind.
		Multiuser:         flags.supportSMSharing,
		PipeDir:           flags.pipeDir,
		LogDir:            flags.logDir,
		Backoff:           flags.backoff,
		MaxRetries:        flags.maxRetries,
		StableThreshold:   flags.stableThreshold,
		GracefulStopDelay: flags.gracefulStopDelay,
		// Namespaces do not survive a restart of the control daemon, and mpsd
		// restarts it (to clear a wedge, and after the last client leaves when
		// recycling is on) while the containers using them keep running.
		// Reconciling on every start is what recreates them at the same names,
		// and therefore the same pipe directories those containers are already
		// bind-mounted to.
		PostStart: namespaceReconciler(namespaces, logger),
	}, logger)

	// The drain endpoint is what turns "a pod was deleted" into "its MPS clients
	// were drained first", so it runs alongside the supervised daemon rather
	// than as a separate component: it needs the same control binary and the
	// same supervisor handle it restarts through.
	if flags.drainSocket != "" {
		drainServer := drain.New(drain.Options{
			SocketPath:         flags.drainSocket,
			Control:            control,
			Restarter:          supervisor,
			Namespaces:         namespaceServer(namespaces),
			ClientDrainTimeout: flags.clientDrainTimeout,
			RecycleWhenIdle:    flags.recycleWhenIdle,
			Log:                logger,
		})
		go func() {
			// A drain endpoint that cannot start costs the node its
			// drain-before-kill protection, but mpsd's job is to keep MPS
			// running — so log it and carry on rather than taking MPS down.
			if err := drainServer.Serve(ctx); err != nil {
				logger.Error("mpsd endpoints failed", "error", err)
			}
		}()
	} else {
		logger.Warn("mpsd endpoints disabled; a container killed with GPU work in flight can wedge MPS for every other tenant of its GPU, and sm-sharing containers cannot be given an enforceable compute cap")
	}

	// Retry the namespace deletions a still-attached client refused, and
	// reclaim the namespaces of containers that went away without a release.
	if namespaces != nil {
		go namespaces.SweepEvery(ctx, flags.namespaceSweepInterval)
	}

	if err := supervisor.Run(ctx); err != nil {
		logger.Error("mpsd exiting with error", "error", err)
		os.Exit(1)
	}

	logger.Info("mpsd shutdown complete")
}

// buildNamespaceManager builds the per-container MPS namespace manager, or
// returns nil when the node must not use one.
//
// nil is a working configuration, not a failure: fractiond then refuses to
// create sm-sharing containers rather than creating uncapped ones, so the node
// loses the sm-sharing feature and never loses the cap.
func buildNamespaceManager(flags cliFlags, control *mpsctl.Control, logger *slog.Logger) *mpsns.Manager {
	if !flags.namespaceIsolation {
		logger.Warn("MPS namespace isolation is disabled; sm-sharing containers will not be given an enforceable compute cap")
		return nil
	}
	if !flags.supportSMSharing {
		// Not a warning: this is what the sm-sharing chicken bit being off is
		// supposed to look like. The namespaces live on the shared server, and
		// there is no shared server.
		logger.Info("sm-sharing is disabled, so no per-container MPS namespaces will be provisioned")
		return nil
	}

	manager, err := mpsns.New(mpsns.Options{
		Control:             control,
		Server:              internal.DefaultSharedServerName,
		StatePath:           flags.namespaceStatePath,
		DefaultNamespaceATP: flags.defaultNamespaceATP,
		// mpsd runs in the host PID namespace, so the same /proc that tells it
		// which PIDs belong to a container tells it whether the container is
		// still there.
		Live:        func(containerID string) (bool, error) { return containerIsLive(containerID) },
		OrphanGrace: flags.namespaceOrphanGrace,
		Log:         logger,
	})
	if err != nil {
		// Misconfiguration, not a runtime failure. Carrying on without a
		// manager keeps MPS up for the time-slicing workloads on the node while
		// sm-sharing containers are refused.
		logger.Error("could not build the MPS namespace manager; sm-sharing containers will be refused", "error", err)
		return nil
	}
	return manager
}

// containerIsLive reports whether a container still has processes on the node.
func containerIsLive(containerID string) (bool, error) {
	pids, err := procfs.PIDsInContainer(procfs.DefaultRoot, containerID)
	if err != nil {
		return false, err
	}
	return len(pids) > 0, nil
}

// namespaceServer adapts a possibly-nil manager to the endpoint's interface.
//
// A typed nil pointer stored in a non-nil interface would pass the endpoint's
// "is namespace isolation configured" check and then panic on the first call,
// which is the sort of bug that only shows up on a node with the feature
// turned off.
func namespaceServer(manager *mpsns.Manager) drain.Namespaces {
	if manager == nil {
		return nil
	}
	return manager
}

// namespaceReconciler returns the supervisor's post-start hook, or nil when
// there is nothing to reconcile.
func namespaceReconciler(manager *mpsns.Manager, logger *slog.Logger) func(context.Context) {
	if manager == nil {
		return nil
	}
	return func(ctx context.Context) {
		if err := manager.Reconcile(ctx); err != nil {
			// Partial reconciliation is normal on a node that is mid-churn, and
			// every step is independent, so this is reported rather than
			// retried here — the sweep and the next daemon start both try
			// again.
			logger.Warn("MPS namespace reconciliation did not complete", "error", err)
			return
		}
		logger.Info("reconciled MPS namespaces", "leases", len(manager.Leases()))
	}
}
