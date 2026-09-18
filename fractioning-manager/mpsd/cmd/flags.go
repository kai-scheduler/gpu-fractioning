// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"time"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/drain"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/mpsns"
	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/env"
)

type cliFlags struct {
	// mpsBinary is the path to the nvidia-cuda-mps-control binary.
	mpsBinary string
	// controlPort is the -p value for nvidia-cuda-mps-control.
	controlPort string
	// configPath is the -a MPS config file path (generated at startup).
	configPath string
	// memacctAuditLog enables features.memacct.audit_log in the generated config.
	memacctAuditLog bool
	// supportSMSharing is the sm-sharing installation-time chicken bit; when
	// false, the generated config omits context-share/the shared server.
	supportSMSharing bool
	// pipeDir is CUDA_MPS_PIPE_DIRECTORY, shared with containers.
	pipeDir string
	// logDir is CUDA_MPS_LOG_DIRECTORY, the daemon log output.
	logDir string
	// logLevel is the slog level: debug, info, warn, error.
	logLevel string
	// backoff is the initial delay before restarting after an unexpected exit.
	backoff time.Duration
	// maxRetries is the max restart attempts (0 = unlimited).
	maxRetries int
	// stableThreshold is how long a run must last to reset the retry budget.
	stableThreshold time.Duration
	// gracefulStopDelay is the time to wait for SIGTERM before SIGKILL.
	gracefulStopDelay time.Duration
	// drainSocket is the unix socket the MPS client-drain endpoint is served
	// on; empty disables the endpoint.
	drainSocket string
	// clientDrainTimeout bounds one MPS client's terminate-and-drain.
	clientDrainTimeout time.Duration
	// recycleWhenIdle restarts MPS once a drain leaves no clients attached.
	recycleWhenIdle bool
	// namespaceIsolation provisions a capped MPS namespace per sm-sharing
	// container instead of pointing them all at one shared, uncapped socket.
	namespaceIsolation bool
	// namespaceStatePath is where the per-container namespace leases are
	// persisted so they survive an mpsd restart.
	namespaceStatePath string
	// defaultNamespaceATP is the ceiling put on the shared server's `default`
	// namespace as defence in depth.
	defaultNamespaceATP int
	// namespaceSweepInterval is how often refused namespace deletions are
	// retried and dead containers' namespaces reclaimed.
	namespaceSweepInterval time.Duration
	// namespaceOrphanGrace protects a freshly provisioned namespace from being
	// reclaimed before its container has started a process.
	namespaceOrphanGrace time.Duration
}

func parseFlags() cliFlags {
	var f cliFlags
	flag.StringVar(&f.mpsBinary, "mps-binary",
		env.String("MPS_CONTROL_BINARY", internal.DefaultMPSBinary),
		"path to nvidia-cuda-mps-control binary")
	flag.StringVar(&f.controlPort, "mps-control-port",
		env.String("MPS_CONTROL_PORT", internal.DefaultMPSControlPort),
		"value passed to nvidia-cuda-mps-control -p (empty to omit)")
	flag.StringVar(&f.configPath, "mps-config-path",
		env.String("MPS_CONFIG_PATH", internal.DefaultMPSConfigPath),
		"MPS control-daemon config file passed via -a (empty to omit)")
	flag.BoolVar(&f.memacctAuditLog, "memacct-audit-log",
		env.Bool("MPS_MEMACCT_AUDIT_LOG", internal.DefaultMemacctAuditLog),
		"enable features.memacct.audit_log in the generated MPS config")
	flag.BoolVar(&f.supportSMSharing, "support-sm-sharing",
		env.Bool("MPS_SUPPORT_SM_SHARING", internal.DefaultSupportSMSharing),
		"enable the shared MPS server (context-share) that fractiond routes sm-sharing containers to")
	flag.StringVar(&f.pipeDir, "pipe-dir",
		env.String("CUDA_MPS_PIPE_DIRECTORY", configuration.DefaultMPSPipeDirectory),
		"CUDA MPS pipe directory")
	flag.StringVar(&f.logDir, "log-dir",
		env.String("CUDA_MPS_LOG_DIRECTORY", internal.DefaultLogDir),
		"CUDA MPS log directory")
	flag.StringVar(&f.logLevel, "log-level",
		env.String("LOG_LEVEL", "info"),
		"log level (debug, info, warn, error)")
	flag.DurationVar(&f.backoff, "backoff",
		env.Duration("MPS_RESTART_BACKOFF", internal.DefaultBackoff),
		"initial delay before restarting MPS daemon after crash")
	flag.IntVar(&f.maxRetries, "max-retries",
		env.Int("MPS_MAX_RETRIES", internal.DefaultMaxRetries),
		"max restart attempts before giving up (0 = unlimited)")
	flag.DurationVar(&f.stableThreshold, "stable-threshold",
		env.Duration("MPS_STABLE_THRESHOLD", internal.DefaultStableThreshold),
		"how long the daemon must run to be considered stable (resets retry budget)")
	flag.DurationVar(&f.gracefulStopDelay, "graceful-stop-delay",
		env.Duration("MPS_GRACEFUL_STOP_DELAY", internal.DefaultGracefulStopDelay),
		"time to wait for SIGTERM before SIGKILL on shutdown")
	flag.StringVar(&f.drainSocket, "drain-socket",
		env.String("MPS_DRAIN_SOCKET", drain.DefaultSocketPath),
		"unix socket serving the MPS client-drain endpoint fractiond calls before a container stops (empty disables)")
	flag.DurationVar(&f.clientDrainTimeout, "client-drain-timeout",
		env.Duration("MPS_CLIENT_DRAIN_TIMEOUT", drain.DefaultClientDrainTimeout),
		"how long to wait for one MPS client to drain; exceeding it is treated as a wedged MPS server")
	flag.BoolVar(&f.recycleWhenIdle, "recycle-when-idle",
		env.Bool("MPS_RECYCLE_WHEN_IDLE", true),
		"restart the MPS daemon once a drain leaves no MPS clients attached, so a fault cannot outlive the workload that caused it")
	flag.BoolVar(&f.namespaceIsolation, "namespace-isolation",
		env.Bool("MPS_NAMESPACE_ISOLATION", internal.DefaultNamespaceIsolation),
		"give each sm-sharing container its own MPS namespace capped at its compute portion, which MPS enforces and the container cannot raise (disabling it reverts to the advisory CUDA_MPS_ACTIVE_THREAD_PERCENTAGE env var alone)")
	flag.StringVar(&f.namespaceStatePath, "namespace-state-path",
		env.String("MPS_NAMESPACE_STATE_PATH", internal.DefaultNamespaceStatePath),
		"file the per-container MPS namespace leases are persisted to; it must be on a host path so an mpsd restart can reconcile them (empty disables persistence)")
	flag.IntVar(&f.defaultNamespaceATP, "default-namespace-active-thread-percentage",
		env.Int("MPS_DEFAULT_NAMESPACE_ATP", mpsns.DefaultDefaultNamespaceATP),
		"ceiling put on the shared server's uncapped `default` namespace as defence in depth, so a container that reaches it by mistake cannot take the whole GPU (0 or less leaves it uncapped)")
	flag.DurationVar(&f.namespaceSweepInterval, "namespace-sweep-interval",
		env.Duration("MPS_NAMESPACE_SWEEP_INTERVAL", mpsns.DefaultSweepInterval),
		"how often to retry MPS namespace deletions that were refused because a client was still attached, and to reclaim namespaces whose container is gone")
	flag.DurationVar(&f.namespaceOrphanGrace, "namespace-orphan-grace",
		env.Duration("MPS_NAMESPACE_ORPHAN_GRACE", mpsns.DefaultOrphanGrace),
		"how long a freshly provisioned MPS namespace is protected from being reclaimed, covering the window between a container being created and starting its first process")
	flag.Parse()
	return f
}
