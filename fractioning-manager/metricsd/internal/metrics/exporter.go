// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

const (
	// HealthzPath and ReadyzPath are the fixed liveness/readiness HTTP paths the
	// exporter serves alongside the (configurable) metrics path. The metricsd
	// container's probes target these.
	HealthzPath = "/healthz"
	ReadyzPath  = "/readyz"
)

// healthHandler serves the liveness/readiness probes. It reports OK whenever the
// HTTP server is answering; that is a sufficient signal because a crashed or
// wedged exporter stops responding, flipping the probe (and thus pod readiness).
func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

type Config struct {
	Enabled  bool
	Address  string
	Path     string
	ProcRoot string
	Interval time.Duration
	// SMUtilizationWindow, when greater than Interval, enables sliding-window
	// averaging of per-pod GPU SM utilization. Each collect's sample is stored;
	// samples older than the window are evicted; the reported value is the
	// average of the remaining samples. Zero (default) disables windowing and
	// reports the latest sample only.
	SMUtilizationWindow time.Duration
	Names               MetricNames
}

// MetricNames holds the exported Prometheus metric names. Each is independently
// configurable; empty fields fall back to the defaults via WithDefaults.
type MetricNames struct {
	GPUMemoryUsedBytes                string `json:"gpuMemoryUsedBytes"`
	GPUSMUtilizationPercent           string `json:"gpuSmUtilizationPercent"`
	GPUSMUtilizationPercentNormalized string `json:"gpuSmUtilizationPercentNormalized"`
}

// defaultMetricNames returns the built-in metric names used when none are
// configured. Pod identity is carried in labels (namespace, pod, pod_uuid, gpu),
// so the names describe the measurement and unit only, per Prometheus convention.
// It is unexported on purpose: callers obtain defaults through WithDefaults so a
// MetricNames value is always the entry point (MetricNames{}.WithDefaults()).
func defaultMetricNames() MetricNames {
	return MetricNames{
		GPUMemoryUsedBytes:                "gpu_fractioning_gpu_memory_used_bytes",
		GPUSMUtilizationPercent:           "gpu_fractioning_gpu_sm_utilization_percent",
		GPUSMUtilizationPercentNormalized: "gpu_fractioning_gpu_sm_utilization_percent_normalized",
	}
}

// WithDefaults fills any unset (empty) metric name with its default, so a config
// that overrides only some names keeps the defaults for the rest.
func (n MetricNames) WithDefaults() MetricNames {
	d := defaultMetricNames()
	if strings.TrimSpace(n.GPUMemoryUsedBytes) == "" {
		n.GPUMemoryUsedBytes = d.GPUMemoryUsedBytes
	}
	if strings.TrimSpace(n.GPUSMUtilizationPercent) == "" {
		n.GPUSMUtilizationPercent = d.GPUSMUtilizationPercent
	}
	if strings.TrimSpace(n.GPUSMUtilizationPercentNormalized) == "" {
		n.GPUSMUtilizationPercentNormalized = d.GPUSMUtilizationPercentNormalized
	}
	return n
}

// metricNamePattern mirrors Prometheus's own metric-name rule (the same
// expression as prometheus/common/model.MetricNameRE): a required first
// character from [a-zA-Z_:] followed by any number of [a-zA-Z0-9_:]. This
// implies a minimum of one character and imposes no maximum length, exactly
// like Prometheus.
var metricNamePattern = regexp.MustCompile(`^[a-zA-Z_:][a-zA-Z0-9_:]*$`)

// validate rejects names Prometheus would refuse and names that collide, so a
// typo or duplicate fails startup with a clear error instead of panicking inside
// MustRegister. Every exported metric is configurable, so requiring the three to
// be mutually distinct also guarantees no clash with a non-configurable metric —
// there are none. Fields are checked in a fixed order for deterministic errors.
func (n MetricNames) validate() error {
	fields := []struct {
		key, name string
	}{
		{"gpuMemoryUsedBytes", n.GPUMemoryUsedBytes},
		{"gpuSmUtilizationPercent", n.GPUSMUtilizationPercent},
		{"gpuSmUtilizationPercentNormalized", n.GPUSMUtilizationPercentNormalized},
	}
	seen := make(map[string]string, len(fields))
	for _, f := range fields {
		if !metricNamePattern.MatchString(f.name) {
			return fmt.Errorf("metric name %s=%q is not a valid Prometheus metric name", f.key, f.name)
		}
		if other, ok := seen[f.name]; ok {
			return fmt.Errorf("metric name %s=%q duplicates %s; each exported metric must have a unique name", f.key, f.name, other)
		}
		seen[f.name] = f.key
	}
	return nil
}

type Runtime struct {
	cancel                  context.CancelFunc
	server                  *http.Server
	coll                    collector
	engine                  *metricsEngine
	wg                      sync.WaitGroup
	provider                SnapshotProvider
	log                     *slog.Logger
	mu                      sync.Mutex
	registry                *prometheus.Registry
	memoryBytes             *prometheus.GaugeVec
	smUtilization           *prometheus.GaugeVec
	smUtilizationNormalized *prometheus.GaugeVec
	knownLabels             map[podGPUKey]prometheus.Labels
}

// New builds a Runtime from cfg — validates config, wires the collector, pod
// source, controller, and HTTP server, but does not start any goroutines.
func New(ctx context.Context, cfg Config, reader store.Reader, logger *slog.Logger) (*Runtime, error) {
	if logger == nil {
		logger = slog.Default()
	}

	if err := cfg.Names.validate(); err != nil {
		return nil, fmt.Errorf("configure metric names: %w", err)
	}

	if reader == nil {
		return nil, fmt.Errorf("metrics requires an NRI container store")
	}

	gpuCollector := newCollector(ctx, cfg.collectInterval(), logger)
	controller := newMetricsController(gpuCollector, NewProcCgroupResolver(cfg.ProcRoot), reader, cfg.collectInterval(), cfg.SMUtilizationWindow, logger)
	runtime := newRuntime(controller, cfg.Names)
	runtime.log = logger
	runtime.coll = gpuCollector
	runtime.engine = controller

	mux := http.NewServeMux()
	metricsPath := cfg.metricsPath()
	mux.Handle(metricsPath, runtime.handler())
	// Liveness/readiness endpoints so the sidecar's container probes can detect a
	// wedged or dead exporter — otherwise a broken metricsd is invisible to pod
	// readiness. Skip a health path only in the (pathological) case where it has
	// been configured as the metrics path, to avoid a duplicate-registration panic.
	if metricsPath != HealthzPath {
		mux.HandleFunc(HealthzPath, healthHandler)
	}
	if metricsPath != ReadyzPath {
		mux.HandleFunc(ReadyzPath, healthHandler)
	}
	runtime.server = &http.Server{
		Addr:              cfg.Address,
		Handler:           mux,
		ReadHeaderTimeout: metricsReadHeaderTimeout,
	}

	return runtime, nil
}

// Start launches the HTTP server, the collector loop, and the metrics controller.
// It must be called once on a Runtime returned by New.
func (r *Runtime) Start(ctx context.Context) {
	metricsCtx, cancel := context.WithCancel(ctx)
	r.cancel = cancel

	go func() {
		r.log.InfoContext(metricsCtx, "starting GPU metrics exporter", "address", r.server.Addr)
		if err := r.server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			r.log.ErrorContext(metricsCtx, "GPU metrics exporter stopped unexpectedly", "error", err)
		}
	}()

	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		r.coll.Run(metricsCtx)
	}()
	go func() {
		if err := r.engine.Run(metricsCtx); err != nil {
			r.log.ErrorContext(metricsCtx, "metrics controller stopped", "error", err)
		}
	}()
}

func newRuntime(provider SnapshotProvider, names MetricNames) *Runtime {
	labels := []string{"namespace", "pod", "pod_uuid", "gpu_uuid", "gpu"}
	runtime := &Runtime{
		provider: provider,
		registry: prometheus.NewRegistry(),
		memoryBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: names.GPUMemoryUsedBytes,
			Help: "GPU memory used by processes attributed to a Kubernetes pod.",
		}, labels),
		smUtilization: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: names.GPUSMUtilizationPercent,
			Help: "GPU SM utilization attributed to a Kubernetes pod: the sum of per-process NVML SM-active time fractions, clamped to [0,100]. Approximate under MPS where a pod's processes execute concurrently.",
		}, labels),
		smUtilizationNormalized: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: names.GPUSMUtilizationPercentNormalized,
			Help: "GPU SM utilization normalized by the pod's requested GPU fraction (SM utilization ÷ requested fraction), clamped to [0,100]. A pod using as much of the GPU as it requested reports 100. Falls back to the raw SM utilization when the requested fraction is unknown.",
		}, labels),
		knownLabels: map[podGPUKey]prometheus.Labels{},
	}
	runtime.registry.MustRegister(
		runtime.memoryBytes,
		runtime.smUtilization,
		runtime.smUtilizationNormalized,
	)
	return runtime
}

func (r *Runtime) handler() http.Handler {
	handler := promhttp.HandlerFor(r.registry, promhttp.HandlerOpts{})
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		r.refresh(req.Context())
		handler.ServeHTTP(w, req)
	})
}

// refresh pulls the latest snapshot from the provider on each scrape. Because the
// provider may be remote, a fetch failure is non-fatal: we log it and serve the
// last observed values rather than dropping the scrape.
func (r *Runtime) refresh(ctx context.Context) {
	snapshot, err := r.provider.Snapshot(ctx)
	if err != nil {
		r.log.WarnContext(ctx, "failed to fetch metrics snapshot from provider; serving last known values", "error", err)
		return
	}
	r.observeSnapshot(snapshot)
}

func (r *Runtime) observeSnapshot(snapshot Snapshot) {
	r.observe(snapshot.Metrics, snapshot.ActivePodUIDs)
}

func (r *Runtime) observe(metrics []PodGPUMetric, activePodUIDs map[string]struct{}) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Remove series for gone pods, then zero-out remaining known series
	// (containers still active but currently idle).
	r.pruneDeletedPods(activePodUIDs)
	for _, labels := range r.knownLabels {
		r.setZero(labels)
	}

	// For each active metric: remove any idle placeholder for the same pod+GPU,
	// then write the real value. setZero may have touched the idle label first,
	// but deleteIdleLabelsForObservedGPU removes it from the gauge immediately.
	for _, metric := range metrics {
		key, labels := metricLabels(metric)
		r.deleteIdleLabelsForObservedGPU(key)
		r.knownLabels[key] = labels
		r.memoryBytes.With(labels).Set(float64(metric.MemoryBytes))
		r.smUtilization.With(labels).Set(metric.SMUtilizationPercent)
		r.smUtilizationNormalized.With(labels).Set(metric.SMUtilizationPercentNormalized)
	}
}

func (r *Runtime) pruneDeletedPods(activePodUIDs map[string]struct{}) {
	if activePodUIDs == nil {
		return
	}
	for key := range r.knownLabels {
		if _, ok := activePodUIDs[key.PodUID]; !ok {
			r.deleteSeries(key)
		}
	}
}

func (r *Runtime) deleteSeries(key podGPUKey) {
	labels := r.knownLabels[key]
	if labels != nil {
		r.memoryBytes.Delete(labels)
		r.smUtilization.Delete(labels)
		r.smUtilizationNormalized.Delete(labels)
	}
	delete(r.knownLabels, key)
}

func (r *Runtime) deleteIdleLabelsForObservedGPU(observed podGPUKey) {
	if observed.GPUUUID == "" {
		return
	}
	for key := range r.knownLabels {
		if key.Namespace == observed.Namespace && key.PodUID == observed.PodUID && key.GPUUUID == "" && key.GPUIndex == observed.GPUIndex {
			r.deleteSeries(key)
		}
	}
}

func (r *Runtime) setZero(labels prometheus.Labels) {
	r.memoryBytes.With(labels).Set(0)
	r.smUtilization.With(labels).Set(0)
	r.smUtilizationNormalized.With(labels).Set(0)
}

func metricLabels(metric PodGPUMetric) (podGPUKey, prometheus.Labels) {
	key := podGPUKey{
		Namespace: metric.Namespace,
		Pod:       metric.Pod,
		PodUID:    metric.PodUID,
		GPUUUID:   metric.GPUUUID,
		GPUIndex:  metric.GPUIndex,
	}
	return key, prometheus.Labels{
		"namespace": metric.Namespace,
		"pod":       metric.Pod,
		"pod_uuid":  metric.PodUID,
		"gpu_uuid":  metric.GPUUUID,
		"gpu":       strconv.Itoa(metric.GPUIndex),
	}
}

func (r *Runtime) Stop(ctx context.Context, logger *slog.Logger) {
	if r == nil {
		return
	}
	if logger == nil {
		logger = slog.Default()
	}

	if r.cancel != nil {
		r.cancel()
	}
	// Wait for gpuCollector.Run to exit before calling nvml.Shutdown so there is
	// no race between in-flight NVML calls and the shutdown sequence.
	r.wg.Wait()
	if r.server != nil {
		shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := r.server.Shutdown(shutdownCtx); err != nil {
			logger.WarnContext(ctx, "failed to shutdown GPU metrics exporter", "error", err)
		}
	}
	if r.coll != nil {
		if err := r.coll.Close(); err != nil {
			logger.WarnContext(ctx, "failed to close NVML collector", "error", err)
		}
	}
}

func (c Config) metricsPath() string {
	path := strings.TrimSpace(c.Path)
	if path == "" {
		return DefaultPath
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return path
}

const (
	// MinCollectInterval is the floor on the sampling cadence. NVML
	// process-utilization queries are coarse and frequent polling adds cost
	// without resolution, so a configured interval below this is clamped up.
	MinCollectInterval = 2 * time.Second
	// DefaultCollectInterval is used when no valid interval is configured.
	DefaultCollectInterval = 5 * time.Second
	// metricsReadHeaderTimeout is the maximum time the HTTP server waits to
	// read request headers after a TCP connection is accepted. Prevents stalled
	// clients or port-scanners from holding connections open indefinitely.
	metricsReadHeaderTimeout = 5 * time.Second
)

func (c Config) collectInterval() time.Duration {
	if c.Interval <= 0 {
		return DefaultCollectInterval
	}
	if c.Interval < MinCollectInterval {
		return MinCollectInterval
	}
	return c.Interval
}
