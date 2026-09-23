// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"maps"
	"strconv"
	"sync"
	"time"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

const DefaultPath = "/metrics"

const (
	// maxSMUtilPercent is the upper bound for a well-formed SM utilization
	// percentage; sums and normalized values are clamped to it.
	maxSMUtilPercent = 100
	// fullGPUFraction is the fallback used when a pod's requested fraction is
	// unknown, treating it as if it requested the whole GPU.
	fullGPUFraction = 1
	// bytesPerMiB converts the memory values recorded by the fractiond mapper
	// back to bytes so they can be divided by NVML's byte-valued device total
	// memory. The stored value is an integer MiB count.
	bytesPerMiB = 1024 * 1024
)

type metricsEngine struct {
	// mu protects snapshot.
	mu sync.RWMutex
	// collector is the NVML or noop source of per-process GPU metrics.
	collector GPUProcessCollector
	// pods resolves GPU processes to pod/container identity.
	pods podSource
	// interval is how often collect() fires.
	interval time.Duration
	// smUtilWindow is the smoothing window duration for SM utilisation.
	smUtilWindow time.Duration
	// smUtilWindowSize is the window in number of samples (derived: smUtilWindow/interval).
	smUtilWindowSize int
	// smUtilBuf is the rolling sample buffer per pod x GPU; nil when windowSize == 1.
	smUtilBuf map[podGPUKey][]float64
	// log is the controller's logger.
	log *slog.Logger
	// deviceUUIDs maps NVML device index → UUID learned from NVML; fills UUID for NRI-sourced devices.
	deviceUUIDs map[int]string
	// minorToNVMLIndex maps Linux device minor number → NVML device index, populated
	// from each NVML collect. On fake-GPU nodes the noop collector leaves this empty,
	// and idlePodGPUKey falls back to treating GPUDevice.Index as the NVML index.
	minorToNVMLIndex map[int]int
	// deviceTotalMemoryBytes maps NVML device index → total memory (bytes) learned from NVML; the divisor for the GPU fraction.
	deviceTotalMemoryBytes map[int]uint64
	// snapshot is the latest published snapshot, read by Snapshot().
	snapshot Snapshot
}

func newMetricsController(collector GPUProcessCollector, resolver PIDCgroupResolver, reader store.Reader, interval, smUtilWindow time.Duration, logger *slog.Logger) *metricsEngine {
	return newMetricsControllerWithPodSource(collector, newCgroupPodSource(resolver, reader, logger), interval, smUtilWindow, logger)
}

func newMetricsControllerWithPodSource(collector GPUProcessCollector, pods podSource, interval, smUtilWindow time.Duration, logger *slog.Logger) *metricsEngine {
	if logger == nil {
		logger = slog.Default()
	}
	// Window size in number of samples: max(1, window/interval). Windowing is
	// active only when windowSize > 1, i.e. when smUtilWindow > interval.
	windowSize := 1
	if interval > 0 && smUtilWindow > interval {
		windowSize = int(smUtilWindow / interval)
	}
	var smUtilBuf map[podGPUKey][]float64
	if windowSize > 1 {
		smUtilBuf = map[podGPUKey][]float64{}
	}
	return &metricsEngine{
		collector:              collector,
		pods:                   pods,
		interval:               interval,
		smUtilWindow:           smUtilWindow,
		smUtilWindowSize:       windowSize,
		smUtilBuf:              smUtilBuf,
		log:                    logger,
		deviceUUIDs:            map[int]string{},
		minorToNVMLIndex:       map[int]int{},
		deviceTotalMemoryBytes: map[int]uint64{},
	}
}

func (s *metricsEngine) Run(ctx context.Context) error {
	if s.collector == nil || s.pods == nil {
		return errors.New("metrics controller: collector or pod source is nil")
	}

	// Release backend resources (e.g. the PodResources gRPC connection) when the
	// control loop exits.
	defer func() {
		if closer, ok := s.pods.(io.Closer); ok {
			_ = closer.Close()
		}
	}()

	s.collect(ctx)
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.collect(ctx)
		case <-ctx.Done():
			return nil
		}
	}
}

// Snapshot satisfies SnapshotProvider. The in-process controller returns the latest
// locally cached snapshot and never errors; the context and error are part of
// the contract so a future out-of-process provider can honor them.
func (s *metricsEngine) Snapshot(_ context.Context) (Snapshot, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneSnapshot(s.snapshot), nil
}

func (s *metricsEngine) collect(ctx context.Context) {
	// Snapshot fetches the container mapping once; all queries in this cycle
	// hit the resulting in-memory store rather than the filesystem.
	pods := s.pods.Snapshot()
	activePodUIDs := pods.ActivePodUIDs()
	activeContainers := pods.ActiveContainers()
	if len(activeContainers) == 0 {
		s.setSnapshot(nil, activePodUIDs)
		s.log.DebugContext(ctx, "skipped GPU metrics collection because no pods are using GPUs")
		return
	}

	snapshot := s.collector.Snapshot()
	if snapshot.DeviceErrors != nil {
		// Partial result: some devices failed but others succeeded. Surface the
		// failure but still publish what we have so healthy GPUs keep reporting.
		s.log.WarnContext(ctx, "publishing partial GPU metrics; some devices failed", "error", snapshot.DeviceErrors)
	}
	s.rememberDeviceUUIDs(snapshot.DeviceUUIDs)
	s.rememberMinorToNVMLIndex(snapshot.DeviceMinorToNVMLIndex)
	s.rememberDeviceTotalMemory(snapshot.DeviceTotalMemoryBytes)

	metrics, unmatched := s.enrich(ctx, snapshot.Processes, pods)
	if s.smUtilWindowSize > 1 {
		metrics = s.windowedSMUtil(metrics)
	}
	s.normalizeSMUtil(metrics)
	s.setSnapshot(metrics, activePodUIDs)
	s.log.DebugContext(ctx, "completed GPU metrics collect", "podMetrics", len(metrics), "unmatchedGPUProcesses", unmatched)
	for _, m := range metrics {
		s.log.DebugContext(ctx, "pod GPU metric",
			"pod", m.Namespace+"/"+m.Pod,
			"gpu_uuid", m.GPUUUID,
			"gpu_index", m.GPUIndex,
			"memory_bytes", m.MemoryBytes,
			"sm_utilization_percent", m.SMUtilizationPercent,
			"sm_utilization_percent_normalized", m.SMUtilizationPercentNormalized,
			"requested_gpu_fraction", m.RequestedGPUFraction,
		)
	}
}

func (s *metricsEngine) setSnapshot(metrics []PodGPUMetric, activePodUIDs map[string]struct{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.Metrics = append([]PodGPUMetric(nil), metrics...)
	s.snapshot.ActivePodUIDs = maps.Clone(activePodUIDs)
}

func (s *metricsEngine) enrich(ctx context.Context, processes []GPUProcessMetric, pods podSource) ([]PodGPUMetric, int) {
	byPodGPU := map[podGPUKey]*PodGPUMetric{}
	observedPodDevices := map[string]struct{}{}
	// memByKey sums each pod×GPU's requested GPU memory (MiB) across its
	// containers, deduped by container ID so multiple GPU processes of one
	// container are not counted more than once. The per-GPU sum is divided by the
	// device's total memory to derive the requested fraction.
	memByKey := map[podGPUKey]map[string]int64{}
	unmatched := 0

	for _, process := range processes {
		if process.PID == 0 {
			s.log.DebugContext(ctx, "skipping GPU process with empty PID", "gpuUUID", process.GPUUUID)
			continue
		}

		container, ok := pods.ResolveProcess(process)
		if !ok {
			unmatched++
			s.log.DebugContext(ctx, "GPU process could not be attributed to a pod", "pid", process.PID, "gpuUUID", process.GPUUUID)
			continue
		}
		key := podGPUKey{
			Namespace: container.Namespace,
			Pod:       container.Pod,
			PodUID:    container.PodUID,
			GPUUUID:   process.GPUUUID,
			GPUIndex:  process.GPUIndex,
		}
		observePodDevice(observedPodDevices, key)
		recordRequestedMemory(memByKey, key, container.ContainerID, container.RequestedMemoryMiB)
		metric := byPodGPU[key]
		if metric == nil {
			metric = &PodGPUMetric{
				Namespace: key.Namespace,
				Pod:       key.Pod,
				PodUID:    key.PodUID,
				GPUUUID:   key.GPUUUID,
				GPUIndex:  key.GPUIndex,
			}
			byPodGPU[key] = metric
		}

		metric.MemoryBytes += process.UsedGPUMemoryBytes
		metric.SMUtilizationPercent += float64(process.SMUtilizationPercent)
	}

	for _, container := range pods.ActiveContainers() {
		for _, device := range container.GPUDevices {
			key, ok := s.idlePodGPUKey(container, device)
			if !ok {
				continue
			}
			recordRequestedMemory(memByKey, key, container.ContainerID, container.RequestedMemoryMiB)
			if _, ok := byPodGPU[key]; ok {
				continue
			}
			if gpuObservedForPodDevice(observedPodDevices, key) {
				continue
			}
			byPodGPU[key] = idlePodGPUMetric(key)
		}
	}

	out := make([]PodGPUMetric, 0, len(byPodGPU))
	for key, metric := range byPodGPU {
		// SM utilization is summed from per-process NVML samples, each a 0-100
		// time fraction. Concurrent processes (the common MPS case) can push the
		// sum above 100, so clamp it to keep the exported percentage well-formed.
		if metric.SMUtilizationPercent > maxSMUtilPercent {
			metric.SMUtilizationPercent = maxSMUtilPercent
		}
		metric.RequestedGPUFraction = s.gpuFraction(key.GPUIndex, sumRequestedMemory(memByKey[key]))
		out = append(out, *metric)
	}
	return out, unmatched
}

// windowedSMUtil replaces each metric's SMUtilizationPercent with the average
// of the last smUtilWindowSize samples. Called only when smUtilWindowSize > 1.
// Buffer entries for series no longer present are pruned immediately.
func (s *metricsEngine) windowedSMUtil(metrics []PodGPUMetric) []PodGPUMetric {
	current := make(map[podGPUKey]struct{}, len(metrics))
	for i, m := range metrics {
		key := podGPUKeyForMetric(m)
		current[key] = struct{}{}
		metrics[i].SMUtilizationPercent = s.pushSample(key, m.SMUtilizationPercent)
	}

	// Prune entries for series that disappeared (pod deleted, process exited).
	for key := range s.smUtilBuf {
		if _, ok := current[key]; !ok {
			delete(s.smUtilBuf, key)
		}
	}
	return metrics
}

// pushSample appends sample to the ring buffer for key, trims it to the window
// size, and returns the average of the retained samples.
func (s *metricsEngine) pushSample(key podGPUKey, sample float64) float64 {
	buf := append(s.smUtilBuf[key], sample)
	if len(buf) > s.smUtilWindowSize {
		buf = buf[len(buf)-s.smUtilWindowSize:]
	}
	s.smUtilBuf[key] = buf
	var sum float64
	for _, v := range buf {
		sum += v
	}
	return sum / float64(len(buf))
}

// normalizeSMUtil sets each metric's SMUtilizationPercentNormalized to its SM
// utilization divided by the requested GPU fraction, capped at 100.
func (s *metricsEngine) normalizeSMUtil(metrics []PodGPUMetric) {
	for i := range metrics {
		metrics[i].SMUtilizationPercentNormalized = normalizedSMUtil(
			metrics[i].SMUtilizationPercent,
			metrics[i].RequestedGPUFraction,
		)
	}
}

// normalizedSMUtil computes smUtil ÷ fraction, capped at 100. When the requested
// fraction is unknown (0) it falls back to a fraction of 1 — i.e. the pod is
// treated as if it requested the whole GPU, so the normalized value equals the
// raw SM utilization rather than a misleading 0. Negative fractions are already
// filtered to 0 at ingest (adapter.requestedGPUFraction), so they cannot reach here.
func normalizedSMUtil(smUtil, fraction float64) float64 {
	if fraction == 0 {
		fraction = fullGPUFraction
	}
	normalized := smUtil / fraction
	if normalized > maxSMUtilPercent {
		return maxSMUtilPercent
	}
	return normalized
}

// recordRequestedMemory notes containerID's requested GPU memory (MiB)
// under key, deduped by container ID so repeated observations of the same
// container (one per GPU process) do not inflate the pod's requested total.
func recordRequestedMemory(memByKey map[podGPUKey]map[string]int64, key podGPUKey, containerID string, memMiB int64) {
	if memMiB <= 0 || containerID == "" {
		return
	}
	byContainer := memByKey[key]
	if byContainer == nil {
		byContainer = map[string]int64{}
		memByKey[key] = byContainer
	}
	byContainer[containerID] = memMiB
}

// sumRequestedMemory totals the per-container requested GPU memory (MiB)
// recorded for a pod×GPU key.
func sumRequestedMemory(byContainer map[string]int64) int64 {
	var sum int64
	for _, memMiB := range byContainer {
		sum += memMiB
	}
	return sum
}

// gpuFraction derives the requested GPU fraction for a device as requested
// memory ÷ device total memory. It returns 0 when the request is unknown or the
// device's total memory has not been learned from NVML yet — in which case
// normalizeSMUtil falls back to treating the pod as holding the whole GPU (so the
// normalized value equals the raw SM utilization rather than a misleading spike).
func (s *metricsEngine) gpuFraction(gpuIndex int, requestedMiB int64) float64 {
	if requestedMiB <= 0 {
		return 0
	}
	totalBytes := s.deviceTotalMemoryBytes[gpuIndex]
	if totalBytes == 0 {
		return 0
	}
	return (float64(requestedMiB) * bytesPerMiB) / float64(totalBytes)
}

// rememberDeviceTotalMemory records each device's total memory learned from an
// NVML collect so it persists across cycles where a transient error omits it.
func (s *metricsEngine) rememberDeviceTotalMemory(deviceTotalMemory map[int]uint64) {
	for index, total := range deviceTotalMemory {
		if total == 0 {
			continue
		}
		s.deviceTotalMemoryBytes[index] = total
	}
}

func (s *metricsEngine) rememberDeviceUUIDs(deviceUUIDs map[int]string) {
	for index, uuid := range deviceUUIDs {
		if uuid == "" {
			continue
		}
		s.deviceUUIDs[index] = uuid
	}
}

func (s *metricsEngine) rememberMinorToNVMLIndex(minorToNVML map[int]int) {
	for minor, nvmlIdx := range minorToNVML {
		s.minorToNVMLIndex[minor] = nvmlIdx
	}
}

func (s *metricsEngine) idlePodGPUKey(container store.ContainerInfo, device store.GPUDevice) (podGPUKey, bool) {
	uuid := device.UUID
	index := device.Index

	if uuid != "" {
		// PodResources path: UUID known; recover the NVML index.
		if resolved, ok := s.indexForUUID(uuid); ok {
			index = resolved
		}
	} else if len(s.minorToNVMLIndex) > 0 {
		// NRI path (real GPU node): resolve Linux device minor number → NVML index → UUID.
		// minorToNVMLIndex is populated whenever the NVML collector runs, so it is
		// non-empty on any node with a GPU driver. On fake-GPU nodes the noop
		// collector leaves it empty and we fall through to the legacy branch below.
		//
		// Two record formats exist depending on when fractiond was deployed:
		//   New (MinorNumber set): {MinorNumber: N, Index: 0} — N is the Linux minor.
		//   Old (Index set):       {MinorNumber: 0, Index: N} — N is also the Linux
		//     minor, because the old realgpu path stored minor in the Index field.
		minor := device.MinorNumber
		if minor == 0 && device.Index != 0 {
			minor = device.Index
		}
		nvmlIdx, ok := s.minorToNVMLIndex[minor]
		if !ok {
			// Minor not yet in the map — NVML couldn't enumerate this device yet.
			// Return false so the caller skips this device entirely rather than
			// writing an empty-label idle metric to byPodGPU.
			s.log.Debug("GPU device minor not in minor→NVML map; skipping idle metric",
				"minor", minor, "pod", container.Pod, "namespace", container.Namespace)
			return podGPUKey{}, false
		}
		index = nvmlIdx
		uuid = s.deviceUUIDs[nvmlIdx]
	} else {
		// Fake-GPU / noop-collector path: no minor→NVML mapping available.
		// Treat GPUDevice.Index as the NVML index directly (the fake detector sets it).
		uuid = s.deviceUUIDs[device.Index]
	}

	return podGPUKey{
		Namespace: container.Namespace,
		Pod:       container.Pod,
		PodUID:    container.PodUID,
		GPUUUID:   uuid,
		GPUIndex:  index,
	}, true
}

func (s *metricsEngine) indexForUUID(uuid string) (int, bool) {
	for index, deviceUUID := range s.deviceUUIDs {
		if deviceUUID == uuid {
			return index, true
		}
	}
	return 0, false
}

func podGPUKeyForMetric(m PodGPUMetric) podGPUKey {
	return podGPUKey{Namespace: m.Namespace, Pod: m.Pod, PodUID: m.PodUID, GPUUUID: m.GPUUUID, GPUIndex: m.GPUIndex}
}

func idlePodGPUMetric(key podGPUKey) *PodGPUMetric {
	return &PodGPUMetric{
		Namespace: key.Namespace,
		Pod:       key.Pod,
		PodUID:    key.PodUID,
		GPUUUID:   key.GPUUUID,
		GPUIndex:  key.GPUIndex,
	}
}

func observePodDevice(observed map[string]struct{}, key podGPUKey) {
	if key.GPUUUID != "" {
		observed[podDeviceUUIDKey(key)] = struct{}{}
	}
	observed[podDeviceIndexKey(key)] = struct{}{}
}

func gpuObservedForPodDevice(observed map[string]struct{}, key podGPUKey) bool {
	if key.GPUUUID != "" {
		if _, ok := observed[podDeviceUUIDKey(key)]; ok {
			return true
		}
	}
	_, ok := observed[podDeviceIndexKey(key)]
	return ok
}

func podDeviceUUIDKey(key podGPUKey) string {
	return key.Namespace + "/" + key.PodUID + "/uuid/" + key.GPUUUID
}

func podDeviceIndexKey(key podGPUKey) string {
	return key.Namespace + "/" + key.PodUID + "/index/" + strconv.Itoa(key.GPUIndex)
}

func cloneSnapshot(in Snapshot) Snapshot {
	return Snapshot{
		Metrics:       append([]PodGPUMetric(nil), in.Metrics...),
		ActivePodUIDs: maps.Clone(in.ActivePodUIDs),
	}
}
