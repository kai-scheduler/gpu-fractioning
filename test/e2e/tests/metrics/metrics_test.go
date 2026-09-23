// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/pods"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/portforward"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/metrics"
)

// metricsd runs as a sidecar in the operator-managed fractiond DaemonSet pods,
// which the operator labels with component=fractiond (metricsd is a container in
// that pod, not its own workload). Its /metrics endpoint listens on 2112.
const (
	fractiondLabelSelector = "app.kubernetes.io/managed-by=gpu-fractioning,app.kubernetes.io/component=fractiond"
	metricsPort            = 2112
)

// TestE2E_GPUFractioningPluginMetricsEndpointHealthy is a fast smoke test: the
// metricsd sidecar in the fractiond DaemonSet pods serves a valid Prometheus
// /metrics endpoint, independent of any test workload.
func TestE2E_GPUFractioningPluginMetricsEndpointHealthy(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), metricsPresenceTimeout)
	defer cancel()

	client := s.Client
	fractiondPods, err := pods.ListByLabel(ctx, client, client.Config.OperatorNamespace, fractiondLabelSelector)
	if err != nil {
		t.Fatalf("list fractiond pods: %v", err)
	}
	if len(fractiondPods) == 0 {
		t.Fatalf("no fractiond pods found in namespace %s (is the operator deployed?)", client.Config.OperatorNamespace)
	}

	pod := fractiondPods[0]
	fw, err := portforward.ToPod(client, pod.Namespace, pod.Name, metricsPort)
	if err != nil {
		t.Fatalf("port-forward to %s/%s: %v", pod.Namespace, pod.Name, err)
	}
	defer fw.Close()

	// Scrape itself already validates a 200 status, a text/plain Content-Type,
	// and a parseable body. It's the health signal this smoke test cares
	// about — the GPU gauges only emit samples once a GPU-attributed pod has
	// been observed, so an empty family set here (no GPU workload running)
	// doesn't mean the endpoint is unhealthy.
	if _, err := metrics.Scrape(fw.LocalPort, "/metrics"); err != nil {
		t.Fatalf("scrape metrics from %s/%s: %v", pod.Namespace, pod.Name, err)
	}
}
