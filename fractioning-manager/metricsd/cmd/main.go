// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Command metricsd is the GPU metrics sidecar. It reads the container->pod
// mapping the fractiond NRI plugin writes to the shared map directory and exports
// per-pod GPU metrics on a Prometheus endpoint. It runs no NRI plugin of its own;
// the mapping is produced by the fractiond container it is co-scheduled with.
//
// Configuration is entirely flags/env with defaults (no config file), so it can
// be overridden just-in-time by editing the DaemonSet. See flags.go.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/fsstore"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/metricsd/internal/metrics"
)

var (
	version = "dev"
	commit  = "unknown"
	date    = "unknown"
)

func main() {
	flags := parseFlags()

	logger := configuration.NewLogger(flags.logLevel)

	logger.Info("starting gpu-fractioning-metrics",
		"version", version,
		"commit", commit,
		"date", date,
	)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if !flags.metricsEnabled {
		logger.Info("GPU metrics exporter disabled; nothing to do")
		return
	}

	logger.Info("container->pod mapping handoff directory", "mapDir", flags.mapDir)

	// fractiond writes the container->pod mapping to mapDir (a shared volume); this
	// sidecar reads it back to attribute NVML-reported GPU processes to pods.
	reader := fsstore.NewReader(flags.mapDir, logger)

	runtime, err := metrics.New(ctx, metrics.Config{
		Enabled:             true,
		Address:             flags.address,
		Path:                flags.path,
		ProcRoot:            flags.procRoot,
		Interval:            flags.interval,
		SMUtilizationWindow: flags.smUtilWindow,
		Names:               flags.metricNames.WithDefaults(),
	}, reader, logger)
	if err != nil {
		logger.Error("failed to build metrics exporter", "error", err)
		os.Exit(1)
	}
	runtime.Start(ctx)
	defer runtime.Stop(context.Background(), logger)

	<-ctx.Done()
	logger.Info("stopped gpu-fractioning-metrics")
}
