// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	dto "github.com/prometheus/client_model/go"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

type staticProvider struct{ snap Snapshot }

func (s *staticProvider) Snapshot(_ context.Context) (Snapshot, error) { return s.snap, nil }

func TestExporterServesHealthEndpoints(t *testing.T) {
	rt, err := New(context.Background(), Config{
		Enabled: true,
		Address: ":0",
		Names:   MetricNames{}.WithDefaults(),
	}, store.FakeStore{}, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	srv := httptest.NewServer(rt.server.Handler)
	defer srv.Close()

	for _, path := range []string{HealthzPath, ReadyzPath} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s: expected 200, got %d", path, resp.StatusCode)
		}
		if strings.TrimSpace(string(body)) != "ok" {
			t.Fatalf("GET %s: expected body %q, got %q", path, "ok", string(body))
		}
	}
}

func TestMetricNamesValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*MetricNames)
		wantErr bool
	}{
		{name: "defaults are valid"},
		{
			name:    "invalid character",
			mutate:  func(n *MetricNames) { n.GPUMemoryUsedBytes = "gpu-fractioning.bad name" },
			wantErr: true,
		},
		{
			name:    "leading digit",
			mutate:  func(n *MetricNames) { n.GPUSMUtilizationPercent = "1_bad" },
			wantErr: true,
		},
		{
			name:   "single character is allowed",
			mutate: func(n *MetricNames) { n.GPUMemoryUsedBytes = "x" },
		},
		{
			name:    "duplicate names are rejected",
			mutate:  func(n *MetricNames) { n.GPUSMUtilizationPercent = n.GPUMemoryUsedBytes },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			names := MetricNames{}.WithDefaults()
			if tt.mutate != nil {
				tt.mutate(&names)
			}
			err := names.validate()
			if tt.wantErr && err == nil {
				t.Fatalf("expected validation error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("expected no validation error, got %v", err)
			}
		})
	}
}

func TestMetricsExporterKeepsZeroSeriesForMissingReadings(t *testing.T) {
	exporter := newRuntime(nil, MetricNames{}.WithDefaults())
	metric := PodGPUMetric{
		Namespace:            "default",
		Pod:                  "pod",
		PodUID:               "pod-uid",
		GPUUUID:              "GPU-1",
		GPUIndex:             0,
		MemoryBytes:          1024,
		SMUtilizationPercent: 42,
	}
	labels := map[string]string{
		"namespace": "default",
		"pod":       "pod",
		"pod_uuid":  "pod-uid",
		"gpu_uuid":  "GPU-1",
		"gpu":       "0",
	}

	exporter.observeSnapshot(Snapshot{Metrics: []PodGPUMetric{metric}, ActivePodUIDs: map[string]struct{}{
		"pod-uid": {},
	}})
	exporter.observeSnapshot(Snapshot{ActivePodUIDs: map[string]struct{}{
		"pod-uid": {},
	}})

	for _, name := range []string{
		"gpu_fractioning_gpu_memory_used_bytes",
		"gpu_fractioning_gpu_sm_utilization_percent",
		"gpu_fractioning_gpu_sm_utilization_percent_normalized",
	} {
		value, ok := gatheredGaugeValue(t, exporter, name, labels)
		if !ok {
			t.Fatalf("expected %s series to stay exported with zero value", name)
		}
		if value != 0 {
			t.Fatalf("expected %s to be zero after missing collect reading, got %v", name, value)
		}
	}
}

func TestMetricsExporterRemovesSeriesForDeletedContainers(t *testing.T) {
	exporter := newRuntime(nil, MetricNames{}.WithDefaults())
	metric := PodGPUMetric{
		Namespace:   "default",
		Pod:         "pod",
		PodUID:      "pod-uid",
		GPUUUID:     "GPU-1",
		GPUIndex:    0,
		MemoryBytes: 1024,
	}
	labels := map[string]string{
		"namespace": "default",
		"pod":       "pod",
		"pod_uuid":  "pod-uid",
		"gpu_uuid":  "GPU-1",
		"gpu":       "0",
	}

	exporter.observeSnapshot(Snapshot{Metrics: []PodGPUMetric{metric}, ActivePodUIDs: map[string]struct{}{
		"pod-uid": {},
	}})
	exporter.observeSnapshot(Snapshot{ActivePodUIDs: map[string]struct{}{}})

	if _, ok := gatheredGaugeValue(t, exporter, "gpu_fractioning_gpu_memory_used_bytes", labels); ok {
		t.Fatalf("expected deleted container series to be removed")
	}
}

func TestMetricsExporterDropsIdleSeriesWhenRealMetricArrives(t *testing.T) {
	exporter := newRuntime(nil, MetricNames{}.WithDefaults())
	idleMetric := PodGPUMetric{
		Namespace: "default",
		Pod:       "pod",
		PodUID:    "pod-uid",
		GPUUUID:   "",
		GPUIndex:  0,
	}
	realMetric := idleMetric
	realMetric.GPUUUID = "GPU-1"
	realMetric.GPUIndex = 0
	realMetric.MemoryBytes = 1024
	idleLabels := map[string]string{
		"namespace": "default",
		"pod":       "pod",
		"pod_uuid":  "pod-uid",
		"gpu_uuid":  "",
		"gpu":       "0",
	}

	exporter.observeSnapshot(Snapshot{Metrics: []PodGPUMetric{idleMetric}, ActivePodUIDs: map[string]struct{}{
		"pod-uid": {},
	}})
	exporter.observeSnapshot(Snapshot{Metrics: []PodGPUMetric{realMetric}, ActivePodUIDs: map[string]struct{}{
		"pod-uid": {},
	}})

	if _, ok := gatheredGaugeValue(t, exporter, "gpu_fractioning_gpu_memory_used_bytes", idleLabels); ok {
		t.Fatalf("expected idle zero series to be removed after real GPU metric arrives")
	}
}

func TestMetricsExporterServesPrometheusFormat(t *testing.T) {
	snap := Snapshot{
		Metrics: []PodGPUMetric{{
			Namespace:            "default",
			Pod:                  "pod",
			PodUID:               "pod-uid",
			GPUUUID:              "GPU-0",
			GPUIndex:             0,
			MemoryBytes:          2048,
			SMUtilizationPercent: 42,
		}},
		ActivePodUIDs: map[string]struct{}{"pod-uid": {}},
	}
	exporter := newRuntime(&staticProvider{snap}, MetricNames{}.WithDefaults())

	mux := http.NewServeMux()
	mux.Handle("/metrics", exporter.handler())
	srv := httptest.NewServer(mux)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/plain") {
		t.Fatalf("expected text/plain Content-Type, got %q", ct)
	}

	body, _ := io.ReadAll(resp.Body)
	bodyStr := string(body)
	for _, want := range []string{
		"gpu_fractioning_gpu_memory_used_bytes",
		"gpu_fractioning_gpu_sm_utilization_percent",
		"gpu_fractioning_gpu_sm_utilization_percent_normalized",
		`namespace="default"`,
		`pod="pod"`,
		`pod_uuid="pod-uid"`,
		`gpu_uuid="GPU-0"`,
	} {
		if !strings.Contains(bodyStr, want) {
			t.Fatalf("expected %q in /metrics response, got:\n%s", want, bodyStr)
		}
	}
}

func gatheredGaugeValue(t *testing.T, exporter *Runtime, name string, labels map[string]string) (float64, bool) {
	t.Helper()
	families, err := exporter.registry.Gather()
	if err != nil {
		t.Fatalf("Gather returned error: %v", err)
	}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			if metricLabelsMatch(metric.GetLabel(), labels) {
				return metric.GetGauge().GetValue(), true
			}
		}
	}
	return 0, false
}

func metricLabelsMatch(pairs []*dto.LabelPair, labels map[string]string) bool {
	if len(pairs) != len(labels) {
		return false
	}
	for _, pair := range pairs {
		want, ok := labels[pair.GetName()]
		if !ok || pair.GetValue() != want {
			return false
		}
	}
	return true
}
