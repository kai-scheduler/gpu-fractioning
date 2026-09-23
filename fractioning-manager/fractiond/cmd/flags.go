// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"time"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/fsstore"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal"
	"github.com/kai-scheduler/gpu-fractioning/pkg/env"
)

const (
	defaultNRISocketPath = "/var/run/nri/nri.sock"
	defaultCRISocketPath = "/run/containerd/containerd.sock"

	// defaultReadinessPort serves /readyz when --readiness-port is not passed.
	// The operator only passes the flag when spec.readinessPort is set;
	// otherwise its probe targets this same default (see
	// operator/internal/fractioningmanager/components/fractiond), so the two must match.
	defaultReadinessPort = 8093
)

type cliFlags struct {
	// pluginName is the NRI plugin registration name.
	pluginName string
	// pluginIdx is the NRI plugin index; controls hook invocation order.
	pluginIdx string
	// socketPath is the NRI runtime socket path.
	socketPath string
	// annotationPrefix is the annotation prefix for GPU memory config.
	annotationPrefix string
	// mpsPipeDir is the MPS pipe directory path.
	mpsPipeDir string
	// failOpen skips a container on parse error instead of blocking it.
	failOpen bool
	// supportSMSharing is the sm-sharing installation-time chicken bit; when
	// false, the gpu-compute.mode: sm-sharing annotation is rejected like any
	// other invalid value.
	supportSMSharing bool
	// mapDir is the shared dir for the container->pod mapping handoff.
	mapDir string
	// logPodEvents logs each recorded/removed mapping event.
	logPodEvents bool
	// logLevel is the log level (debug, info, warn, error).
	logLevel string
	// retryInterval is the initial wait between NRI connection retries.
	retryInterval time.Duration
	// stableThreshold is how long a connection must last to be considered stable (resets retry budget).
	stableThreshold time.Duration
	// maxRetries is the max NRI connection retries (0 = unlimited).
	maxRetries int
	// readinessPort is the port for the /readyz readiness endpoint (0 = disabled).
	readinessPort int
	// retroactiveEnforcement stops GPU-fractioning containers missing injection on NRI reconnect.
	retroactiveEnforcement bool
	// criSocket is the CRI runtime socket used to stop containers during enforcement.
	criSocket string
	// stopTimeout is the grace period handed to the runtime per container stop.
	stopTimeout time.Duration
}

func parseFlags() cliFlags {
	var f cliFlags
	flag.StringVar(&f.pluginName, "plugin-name", internal.DefaultPluginName, "NRI plugin registration name")
	flag.StringVar(&f.pluginIdx, "plugin-idx", internal.DefaultPluginIdx, "NRI plugin index; controls hook invocation order")
	flag.StringVar(&f.socketPath, "socket-path", env.String("NRI_SOCKET_PATH", defaultNRISocketPath), "path to the NRI runtime socket")
	flag.StringVar(&f.annotationPrefix, "annotation-prefix", configuration.DefaultAnnotationPrefix, "annotation prefix for GPU memory config")
	flag.StringVar(&f.mpsPipeDir, "pipe-dir", configuration.DefaultMPSPipeDirectory, "MPS pipe directory path")
	flag.BoolVar(&f.failOpen, "fail-open", false, "if true, annotation parse errors skip the container instead of blocking it")
	flag.BoolVar(&f.supportSMSharing, "support-sm-sharing", env.Bool("SUPPORT_SM_SHARING", true), "if false, the gpu-compute.mode: sm-sharing annotation is rejected like any other invalid value")
	flag.StringVar(&f.mapDir, "map-dir", env.String("MAP_DIR", fsstore.DefaultMapDir), "shared directory for the container->pod mapping handoff read by the metrics sidecar")
	flag.BoolVar(&f.logPodEvents, "log-pod-events", env.Bool("LOG_POD_EVENTS", false), "log each recorded/removed container->pod mapping event")
	flag.StringVar(&f.logLevel, "log-level", "info", "log level (debug, info, warn, error)")
	flag.DurationVar(&f.retryInterval, "retry-interval", 5*time.Second, "initial wait between NRI connection retries")
	flag.DurationVar(&f.stableThreshold, "stable-threshold", 5*time.Minute, "connection duration considered stable (resets retry budget)")
	flag.IntVar(&f.maxRetries, "max-retries", 0, "max NRI connection retries (0 = unlimited)")
	flag.IntVar(&f.readinessPort, "readiness-port", env.Int("READINESS_PORT", defaultReadinessPort), "port for the /readyz readiness endpoint (0 disables)")
	flag.BoolVar(&f.retroactiveEnforcement, "retroactive-enforcement", env.Bool("RETROACTIVE_ENFORCEMENT", true), "on NRI (re)connect, stop GPU-fractioning containers missing injection so kubelet recreates them correctly")
	flag.StringVar(&f.criSocket, "cri-socket", env.String("CRI_SOCKET", defaultCRISocketPath), "CRI runtime socket used to stop containers during retroactive enforcement")
	flag.DurationVar(&f.stopTimeout, "stop-timeout", 30*time.Second, "grace period handed to the runtime for each container stop during enforcement")
	flag.Parse()
	return f
}
