// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package plugin

import (
	"context"
	"fmt"

	dto "github.com/prometheus/client_model/go"
	corev1 "k8s.io/api/core/v1"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/pods"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/portforward"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/metrics"
)

// LabelSelector matches every fractiond DaemonSet pod — the metricsd metrics
// exporter runs as a sidecar container in those pods (not its own DaemonSet),
// so the /metrics endpoint under test is served from here.
const LabelSelector = "app.kubernetes.io/managed-by=gpu-fractioning,app.kubernetes.io/component=fractiond"

// metricsPort is the Prometheus port the metricsd sidecar serves on.
const metricsPort = 2112

// ScrapeAll lists fractiond pods and scrapes /metrics from each. Use this only
// when a fresh pod list is needed (e.g. restart-count checks). For polling
// loops, prefer ScrapeFrom with pods cached at suite setup — listing pods on
// every poll burns the client rate limiter under CI load.
func ScrapeAll(ctx context.Context, c *cluster.Client) (map[string]map[string]*dto.MetricFamily, error) {
	fractiondPods, err := pods.ListByLabel(ctx, c, c.Config.OperatorNamespace, LabelSelector)
	if err != nil {
		return nil, fmt.Errorf("list fractiond pods: %w", err)
	}
	return scrapeFromPods(ctx, c, fractiondPods), nil
}

// ScrapeFrom scrapes /metrics from the provided fractiond pods without listing
// from the API server. Pass the pod list cached at suite setup to avoid
// repeated pod-list API calls in tight poll loops.
//
// If a pod's port-forward or scrape fails it is silently skipped. Callers that
// need stale-pod healing should use the scrapeFreshFamilies helper in the test
// package, which re-lists once when all cached pods are unreachable and updates
// the suite's pod cache for subsequent polls.
func ScrapeFrom(ctx context.Context, c *cluster.Client, fractiondPods []corev1.Pod) (map[string]map[string]*dto.MetricFamily, error) {
	if len(fractiondPods) == 0 {
		return nil, fmt.Errorf("no fractiond pods provided")
	}
	return scrapeFromPods(ctx, c, fractiondPods), nil
}

// scrapeFromPods port-forwards to each pod and scrapes /metrics. Pods that
// fail port-forwarding or scraping are silently skipped.
func scrapeFromPods(ctx context.Context, c *cluster.Client, fractiondPods []corev1.Pod) map[string]map[string]*dto.MetricFamily {
	result := make(map[string]map[string]*dto.MetricFamily, len(fractiondPods))
	for _, pod := range fractiondPods {
		fw, err := portforward.ToPod(c, pod.Namespace, pod.Name, metricsPort)
		if err != nil {
			// Pod may be restarting mid-rollout; skip and check remaining pods.
			continue
		}
		families, err := metrics.Scrape(fw.LocalPort, "/metrics")
		fw.Close()
		if err != nil {
			continue
		}
		result[pod.Name] = families
	}
	return result
}
