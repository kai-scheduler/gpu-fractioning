// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package config holds all CLI-configurable settings for the operator.
// We use CLI flags rather than a ConfigMap because these values are
// deployment-time constants (addresses, TLS paths, leader election) that
// are set once at pod creation and never change at runtime.
package config

import (
	"flag"

	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/controller"
	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/env"
)

// Config holds the operator's runtime settings parsed from CLI flags.
type Config struct {
	MetricsAddr       string // address the metrics endpoint binds to ("0" disables)
	ProbeAddr         string // address the health/readiness probe binds to
	EnableLeaderElect bool   // enable leader election for HA deployments
	SecureMetrics     bool   // serve metrics over HTTPS
	EnableHTTP2       bool   // allow HTTP/2 (disabled by default for Rapid Reset CVE)
	MetricsCertPath   string // directory containing the metrics TLS certificate
	MetricsCertName   string // filename of the metrics TLS certificate
	MetricsCertKey    string // filename of the metrics TLS private key
	Development       bool   // enable development-mode logging (debug, human-readable)

	// MinGPUOperatorVersion is the lowest NVIDIA GPU Operator version the
	// dependency check accepts. Empty disables the version gate.
	MinGPUOperatorVersion string

	// MarkNodesUnknownOnShutdown makes the operator flip every targeted node's
	// gpu-fractioning Ready condition to Unknown as it exits, so a scaled-down
	// or evicted operator does not leave nodes advertising a readiness nothing
	// is maintaining.
	MarkNodesUnknownOnShutdown bool
}

// ParseFlags registers CLI flags, parses them, and returns the populated Config.
func ParseFlags() Config {
	var cfg Config
	flag.StringVar(&cfg.MetricsAddr, "metrics-bind-address", "0",
		"The address the metrics endpoint binds to. Use :8443 for HTTPS or :8080 for HTTP, or leave as 0 to disable.")
	flag.StringVar(&cfg.ProbeAddr, "health-probe-bind-address", ":8081",
		"The address the probe endpoint binds to.")
	flag.BoolVar(&cfg.EnableLeaderElect, "leader-elect", false,
		"Enable leader election for controller manager.")
	flag.BoolVar(&cfg.SecureMetrics, "metrics-secure", true,
		"If set, the metrics endpoint is served securely via HTTPS.")
	flag.StringVar(&cfg.MetricsCertPath, "metrics-cert-path", "", "The directory that contains the metrics server certificate.")
	flag.StringVar(&cfg.MetricsCertName, "metrics-cert-name", "tls.crt", "The name of the metrics server certificate file.")
	flag.StringVar(&cfg.MetricsCertKey, "metrics-cert-key", "tls.key", "The name of the metrics server key file.")
	flag.BoolVar(&cfg.EnableHTTP2, "enable-http2", false, "If set, HTTP/2 will be enabled for the metrics server.")
	flag.BoolVar(&cfg.Development, "development", false, "Enable development-mode logging (debug level, human-readable).")
	flag.StringVar(&cfg.MinGPUOperatorVersion, "min-gpu-operator-version",
		env.String("MIN_GPU_OPERATOR_VERSION", controller.DefaultMinimumGPUOperatorVersion),
		"Minimum NVIDIA GPU Operator version accepted by the dependency check. Empty disables the version gate.")
	flag.BoolVar(&cfg.MarkNodesUnknownOnShutdown, "mark-nodes-unknown-on-shutdown",
		env.Bool("MARK_NODES_UNKNOWN_ON_SHUTDOWN", true),
		"On shutdown, set the gpu-fractioning Ready condition to Unknown on every targeted node.")
	flag.Parse()
	return cfg
}
