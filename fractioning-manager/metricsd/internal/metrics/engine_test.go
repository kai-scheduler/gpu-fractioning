// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"context"
	"log/slog"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

// testDeviceMemMiB is the total memory (MiB) of the simulated GPU used by the
// fraction tests: a container requesting N MiB on it holds N/testDeviceMemMiB.
const testDeviceMemMiB = 10000

type fakeCgroupResolver map[uint32][]string

func (r fakeCgroupResolver) CgroupPaths(pid uint32) ([]string, error) {
	return r[pid], nil
}

func TestMetricsControllerSkipsCollectionWhenStoreIsEmpty(t *testing.T) {
	collector := &fakeCollector{}
	controller := newMetricsController(collector, fakeCgroupResolver{}, store.FakeStore{}, 0, 0, slog.Default())

	controller.collect(context.Background())

	if collector.snapshots != 0 {
		t.Fatalf("expected collector not to be sampled for empty store, got %d snapshots", collector.snapshots)
	}
}

func TestMetricsControllerCollectsAfterFirstContainerIsStored(t *testing.T) {
	collector := &fakeCollector{}
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
		ContainerID: "container-id",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "pod-uid",
		GPUDevices:  []store.GPUDevice{{Index: 0}},
	}}}
	controller := newMetricsController(collector, fakeCgroupResolver{}, containerStore, 0, 0, slog.Default())

	controller.collect(context.Background())

	if collector.snapshots != 1 {
		t.Fatalf("expected collector to be sampled once, got %d snapshots", collector.snapshots)
	}
}

func TestMetricsControllerEnrichesGPUProcessesWithPodMetadata(t *testing.T) {
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
		ContainerID: "container-id",
		Container:   "container",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "pod-uid",
		CgroupPath:  "/kubepods.slice/pod.slice/container.scope",
		GPUDevices:  []store.GPUDevice{{Index: 0}},
	}}}

	controller := newMetricsController(nil, fakeCgroupResolver{
		1234: []string{"/kubepods.slice/pod.slice/container.scope/deeper"},
	}, containerStore, 0, 0, slog.Default())

	metrics, unmatched := controller.enrich(context.Background(), []GPUProcessMetric{
		{
			PID:                  1234,
			GPUUUID:              "GPU-1",
			GPUIndex:             0,
			UsedGPUMemoryBytes:   1024,
			SMUtilizationPercent: 25,
		},
	}, controller.pods.Snapshot())
	if unmatched != 0 {
		t.Fatalf("expected no unmatched processes, got %d", unmatched)
	}
	if len(metrics) != 1 {
		t.Fatalf("expected one pod metric, got %d", len(metrics))
	}
	metric := metrics[0]
	if metric.Namespace != "default" || metric.Pod != "pod" || metric.PodUID != "pod-uid" {
		t.Fatalf("unexpected pod metadata: %#v", metric)
	}
	if metric.MemoryBytes != 1024 || metric.SMUtilizationPercent != 25 {
		t.Fatalf("unexpected metric values: %#v", metric)
	}
}

func TestMetricsControllerAggregatesContainersByPodAndGPU(t *testing.T) {
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{
		{
			ContainerID: "container-a",
			Container:   "container-a",
			Pod:         "pod",
			Namespace:   "default",
			PodUID:      "pod-uid",
			CgroupPath:  "/kubepods.slice/pod.slice/container-a.scope",
			GPUDevices:  []store.GPUDevice{{Index: 0}},
		},
		{
			ContainerID: "container-b",
			Container:   "container-b",
			Pod:         "pod",
			Namespace:   "default",
			PodUID:      "pod-uid",
			CgroupPath:  "/kubepods.slice/pod.slice/container-b.scope",
			GPUDevices:  []store.GPUDevice{{Index: 0}},
		},
	}}

	controller := newMetricsController(nil, fakeCgroupResolver{
		1001: []string{"/kubepods.slice/pod.slice/container-a.scope/deeper"},
		1002: []string{"/kubepods.slice/pod.slice/container-b.scope/deeper"},
	}, containerStore, 0, 0, slog.Default())

	metrics, unmatched := controller.enrich(context.Background(), []GPUProcessMetric{
		{PID: 1001, GPUUUID: "GPU-1", GPUIndex: 0, UsedGPUMemoryBytes: 1024, SMUtilizationPercent: 10},
		{PID: 1002, GPUUUID: "GPU-1", GPUIndex: 0, UsedGPUMemoryBytes: 2048, SMUtilizationPercent: 20},
	}, controller.pods.Snapshot())
	if unmatched != 0 {
		t.Fatalf("expected no unmatched processes, got %d", unmatched)
	}
	if len(metrics) != 1 {
		t.Fatalf("expected one pod/device metric, got %d", len(metrics))
	}
	metric := metrics[0]
	if metric.MemoryBytes != 3072 || metric.SMUtilizationPercent != 30 {
		t.Fatalf("expected accumulated pod/device metric, got %#v", metric)
	}
}

func TestMetricsControllerNormalizesSMUtilByRequestedFraction(t *testing.T) {
	tests := []struct {
		name           string
		fraction       float64
		smUtil         uint32
		wantNormalized float64
	}{
		{name: "half request fully used", fraction: 0.5, smUtil: 30, wantNormalized: 60},
		{name: "exactly at cap", fraction: 0.5, smUtil: 50, wantNormalized: 100},
		{name: "over-utilized is capped at 100", fraction: 0.5, smUtil: 80, wantNormalized: 100},
		{name: "quarter request", fraction: 0.25, smUtil: 10, wantNormalized: 40},
		{name: "unknown request falls back to raw utilization", fraction: 0, smUtil: 40, wantNormalized: 40},
		{name: "unknown request still capped at 100", fraction: 0, smUtil: 150, wantNormalized: 100},
	}
	// The requested fraction is derived as requested memory ÷ device total memory.
	// With a testDeviceMemMiB-sized device, a container requesting
	// fraction*device MiB yields exactly that fraction.
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
				ContainerID:        "container-id",
				Container:          "container",
				Pod:                "pod",
				Namespace:          "default",
				PodUID:             "pod-uid",
				CgroupPath:         "/kubepods.slice/pod.slice/container.scope",
				GPUDevices:         []store.GPUDevice{{Index: 0}},
				RequestedMemoryMiB: int64(tt.fraction * testDeviceMemMiB),
			}}}

			controller := newMetricsController(nil, fakeCgroupResolver{
				1234: []string{"/kubepods.slice/pod.slice/container.scope/deeper"},
			}, containerStore, 0, 0, slog.Default())
			controller.rememberDeviceTotalMemory(map[int]uint64{0: testDeviceMemMiB * bytesPerMiB})

			metrics, unmatched := controller.enrich(context.Background(), []GPUProcessMetric{
				{PID: 1234, GPUUUID: "GPU-1", GPUIndex: 0, SMUtilizationPercent: tt.smUtil},
			}, controller.pods.Snapshot())
			controller.normalizeSMUtil(metrics)

			if unmatched != 0 {
				t.Fatalf("expected no unmatched processes, got %d", unmatched)
			}
			if len(metrics) != 1 {
				t.Fatalf("expected one pod metric, got %d", len(metrics))
			}
			m := metrics[0]
			if m.RequestedGPUFraction != tt.fraction {
				t.Fatalf("expected RequestedGPUFraction = %g, got %g", tt.fraction, m.RequestedGPUFraction)
			}
			if m.SMUtilizationPercentNormalized != tt.wantNormalized {
				t.Fatalf("expected normalized SM util = %g, got %g", tt.wantNormalized, m.SMUtilizationPercentNormalized)
			}
		})
	}
}

// Directly exercises normalizedSMUtil so its own cap and fallback are pinned
// independently of the enrich-level clamp (which caps raw SM util before this
// runs). The unknown-fraction case uses smUtil > 100 to prove the cap fires here.
func TestNormalizedSMUtil(t *testing.T) {
	tests := []struct {
		name     string
		smUtil   float64
		fraction float64
		want     float64
	}{
		{name: "half request scales up", smUtil: 30, fraction: 0.5, want: 60},
		{name: "quarter request scales up", smUtil: 10, fraction: 0.25, want: 40},
		{name: "over-utilized is capped at 100", smUtil: 80, fraction: 0.5, want: 100},
		{name: "unknown fraction falls back to raw util", smUtil: 40, fraction: 0, want: 40},
		{name: "unknown fraction still capped at 100", smUtil: 150, fraction: 0, want: 100},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizedSMUtil(tt.smUtil, tt.fraction); got != tt.want {
				t.Fatalf("normalizedSMUtil(%g, %g) = %g, want %g", tt.smUtil, tt.fraction, got, tt.want)
			}
		})
	}
}

// Two containers of the same pod share one GPU; their requested memory sums into
// one fraction, and each container is counted once even when it has multiple GPU
// processes.
func TestMetricsControllerSumsRequestedMemoryAcrossContainers(t *testing.T) {
	// Each container requests a quarter of the device (testDeviceMemMiB/4 MiB), so
	// the pod's fraction on the GPU is 0.5.
	quarterMiB := int64(testDeviceMemMiB / 4)
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{
		{
			ContainerID:        "container-a",
			Pod:                "pod",
			Namespace:          "default",
			PodUID:             "pod-uid",
			CgroupPath:         "/kubepods.slice/pod.slice/container-a.scope",
			GPUDevices:         []store.GPUDevice{{Index: 0}},
			RequestedMemoryMiB: quarterMiB,
		},
		{
			ContainerID:        "container-b",
			Pod:                "pod",
			Namespace:          "default",
			PodUID:             "pod-uid",
			CgroupPath:         "/kubepods.slice/pod.slice/container-b.scope",
			GPUDevices:         []store.GPUDevice{{Index: 0}},
			RequestedMemoryMiB: quarterMiB,
		},
	}}

	controller := newMetricsController(nil, fakeCgroupResolver{
		1001: []string{"/kubepods.slice/pod.slice/container-a.scope/deeper"},
		1002: []string{"/kubepods.slice/pod.slice/container-a.scope/deeper"}, // second process, same container
		1003: []string{"/kubepods.slice/pod.slice/container-b.scope/deeper"},
	}, containerStore, 0, 0, slog.Default())
	controller.rememberDeviceTotalMemory(map[int]uint64{0: testDeviceMemMiB * bytesPerMiB})

	metrics, _ := controller.enrich(context.Background(), []GPUProcessMetric{
		{PID: 1001, GPUUUID: "GPU-1", GPUIndex: 0, SMUtilizationPercent: 10},
		{PID: 1002, GPUUUID: "GPU-1", GPUIndex: 0, SMUtilizationPercent: 10},
		{PID: 1003, GPUUUID: "GPU-1", GPUIndex: 0, SMUtilizationPercent: 10},
	}, controller.pods.Snapshot())

	if len(metrics) != 1 {
		t.Fatalf("expected one pod/device metric, got %d", len(metrics))
	}
	// container-a counted once (quarter) + container-b (quarter) = half the device.
	if metrics[0].RequestedGPUFraction != 0.5 {
		t.Fatalf("expected summed fraction 0.5, got %g", metrics[0].RequestedGPUFraction)
	}
}

func TestMetricsControllerPublishesIdleMetricForActiveGPUContainer(t *testing.T) {
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
		ContainerID: "container-id",
		Container:   "container",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "pod-uid",
		GPUDevices:  []store.GPUDevice{{Index: 0}},
	}}}

	controller := newMetricsController(nil, fakeCgroupResolver{}, containerStore, 0, 0, slog.Default())
	controller.rememberDeviceUUIDs(map[int]string{0: "GPU-0"})
	metrics, unmatched := controller.enrich(context.Background(), nil, controller.pods.Snapshot())
	if unmatched != 0 {
		t.Fatalf("expected no unmatched processes, got %d", unmatched)
	}
	if len(metrics) != 1 {
		t.Fatalf("expected one idle pod metric, got %d", len(metrics))
	}
	metric := metrics[0]
	if metric.PodUID != "pod-uid" || metric.GPUUUID != "GPU-0" || metric.GPUIndex != 0 {
		t.Fatalf("unexpected idle metric identity: %#v", metric)
	}
	if metric.MemoryBytes != 0 || metric.SMUtilizationPercent != 0 {
		t.Fatalf("expected idle metric values to be zero, got %#v", metric)
	}
}

func TestMetricsControllerSkipsIdleMetricForAnnotatedContainerWithoutGPUDevices(t *testing.T) {
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
		ContainerID: "container-id",
		Container:   "container",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "pod-uid",
	}}}

	controller := newMetricsController(nil, fakeCgroupResolver{}, containerStore, 0, 0, slog.Default())
	metrics, unmatched := controller.enrich(context.Background(), nil, controller.pods.Snapshot())
	if unmatched != 0 {
		t.Fatalf("expected no unmatched processes, got %d", unmatched)
	}
	if len(metrics) != 0 {
		t.Fatalf("expected no idle metric without GPU devices, got %d", len(metrics))
	}
}

func TestMetricsControllerPublishesIdleMetricsForAnnotatedGPUIndexes(t *testing.T) {
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
		ContainerID: "container-id",
		Container:   "container",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "pod-uid",
		GPUDevices: []store.GPUDevice{
			{Index: 0},
			{Index: 1},
		},
	}}}

	controller := newMetricsController(nil, fakeCgroupResolver{}, containerStore, 0, 0, slog.Default())
	controller.rememberDeviceUUIDs(map[int]string{0: "GPU-0", 1: "GPU-1"})
	metrics, unmatched := controller.enrich(context.Background(), nil, controller.pods.Snapshot())
	if unmatched != 0 {
		t.Fatalf("expected no unmatched processes, got %d", unmatched)
	}
	if len(metrics) != 2 {
		t.Fatalf("expected two idle pod metrics, got %d", len(metrics))
	}
	seen := map[int]string{}
	for _, metric := range metrics {
		seen[metric.GPUIndex] = metric.GPUUUID
		if metric.MemoryBytes != 0 || metric.SMUtilizationPercent != 0 {
			t.Fatalf("expected idle metric values to be zero, got %#v", metric)
		}
	}
	if seen[0] != "GPU-0" || seen[1] != "GPU-1" {
		t.Fatalf("expected idle metrics for both GPU UUIDs, got %#v", seen)
	}
}

func TestMetricsControllerSkipsZeroPIDProcesses(t *testing.T) {
	containerStore := store.FakeStore{}
	resolver := fakeCgroupResolver{}
	controller := newMetricsController(nil, resolver, containerStore, 0, 0, slog.Default())

	metrics, unmatched := controller.enrich(context.Background(), []GPUProcessMetric{
		{
			PID:     0,
			GPUUUID: "GPU-1",
		},
	}, controller.pods.Snapshot())
	if unmatched != 0 {
		t.Fatalf("expected zero PID process to be skipped without counting as unmatched, got %d", unmatched)
	}
	if len(metrics) != 0 {
		t.Fatalf("expected no metrics for zero PID process, got %d", len(metrics))
	}
}

func TestWindowedSMUtilAveragesWithinWindow(t *testing.T) {
	type collect struct{ in, want float64 }
	tests := []struct {
		name       string
		windowSize int
		collects   []collect
	}{
		{
			name:       "single sample returns itself",
			windowSize: 4,
			collects:   []collect{{80, 80}},
		},
		{
			name:       "two samples averaged",
			windowSize: 4,
			collects:   []collect{{80, 80}, {20, 50}},
		},
		{
			name:       "oldest evicted when window full",
			windowSize: 4,
			collects: []collect{
				{80, 80},
				{20, 50},
				{60, (80 + 20 + 60) / 3.0},
				{40, 50},
				{100, (20 + 60 + 40 + 100) / 4.0}, // 80 evicted
			},
		},
		{
			name:       "window size 2 evicts after two samples",
			windowSize: 2,
			collects:   []collect{{80, 80}, {20, 50}, {60, 40}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &metricsEngine{
				smUtilWindowSize: tt.windowSize,
				smUtilBuf:        map[podGPUKey][]float64{},
				deviceUUIDs:      map[int]string{},
			}
			for i, sw := range tt.collects {
				got := s.windowedSMUtil([]PodGPUMetric{{PodUID: "p", GPUUUID: "G", SMUtilizationPercent: sw.in}})
				if got[0].SMUtilizationPercent != sw.want {
					t.Errorf("collect %d (in=%.1f): want %.4f, got %.4f", i+1, sw.in, sw.want, got[0].SMUtilizationPercent)
				}
			}
		})
	}
}

func TestWindowedSMUtilPrunesDeletedPodSeriesImmediately(t *testing.T) {
	controller := &metricsEngine{
		smUtilWindowSize: 3,
		smUtilBuf:        map[podGPUKey][]float64{},
		deviceUUIDs:      map[int]string{},
	}

	pod := []PodGPUMetric{{PodUID: "pod-uid", GPUUUID: "GPU-0", GPUIndex: 0, SMUtilizationPercent: 50}}
	controller.windowedSMUtil(pod)

	if len(controller.smUtilBuf) != 1 {
		t.Fatalf("expected 1 buffered series after first collect, got %d", len(controller.smUtilBuf))
	}

	// Pod disappears — next collect carries no metrics for it. Buffer entry must
	// be pruned immediately (no time-based expiry needed).
	controller.windowedSMUtil(nil)

	if len(controller.smUtilBuf) != 0 {
		t.Fatalf("expected stale series to be pruned immediately, got %d entries", len(controller.smUtilBuf))
	}
}

// TestGPUFraction pins the memory→fraction derivation and both fallbacks: an
// unknown request and unknown device memory each yield 0, so normalizeSMUtil
// treats the pod as holding the whole GPU (raw SM util) rather than dividing by
// zero or reporting a misleading spike.
func TestGPUFraction(t *testing.T) {
	const totalBytes = uint64(testDeviceMemMiB) * bytesPerMiB // a testDeviceMemMiB-sized device

	tests := []struct {
		name         string
		requestedMiB int64
		deviceTotal  map[int]uint64
		want         float64
	}{
		{name: "half the device", requestedMiB: testDeviceMemMiB / 2, deviceTotal: map[int]uint64{0: totalBytes}, want: 0.5},
		{name: "quarter the device", requestedMiB: testDeviceMemMiB / 4, deviceTotal: map[int]uint64{0: totalBytes}, want: 0.25},
		{name: "no request yields zero", requestedMiB: 0, deviceTotal: map[int]uint64{0: totalBytes}, want: 0},
		{name: "unknown device memory yields zero", requestedMiB: testDeviceMemMiB / 2, deviceTotal: map[int]uint64{}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &metricsEngine{deviceTotalMemoryBytes: tt.deviceTotal}
			if got := s.gpuFraction(0, tt.requestedMiB); got != tt.want {
				t.Fatalf("gpuFraction(0, %d) = %g, want %g", tt.requestedMiB, got, tt.want)
			}
		})
	}
}

// TestRememberDeviceTotalMemoryPersistsAcrossPartialUpdates verifies a device's
// learned total survives a later collect that omits it or reports 0 (a transient
// NVML error), so the derived fraction does not flap to the whole-GPU fallback.
func TestRememberDeviceTotalMemoryPersistsAcrossPartialUpdates(t *testing.T) {
	s := &metricsEngine{deviceTotalMemoryBytes: map[int]uint64{}}

	s.rememberDeviceTotalMemory(map[int]uint64{0: 100, 1: 200})
	// Next collect: device 0 still reports; device 1 reports 0 (transient error).
	s.rememberDeviceTotalMemory(map[int]uint64{0: 100, 1: 0})

	if s.deviceTotalMemoryBytes[0] != 100 {
		t.Fatalf("expected device 0 total 100, got %d", s.deviceTotalMemoryBytes[0])
	}
	if s.deviceTotalMemoryBytes[1] != 200 {
		t.Fatalf("expected device 1 total to persist at 200 (0 must not overwrite), got %d", s.deviceTotalMemoryBytes[1])
	}
}

// TestIdlePodGPUKeyResolvesMinorToNVMLIndex verifies that on a real GPU node
// (minorToNVMLIndex populated) the idle key is built from GPUDevice.MinorNumber
// via the minor→NVML mapping, not from GPUDevice.Index. This is the fix for the
// phantom metric bug: NVML index 0 → minor 1 → GPU-AAA, NVML index 1 → minor 0 → GPU-BBB.
// A container exposed device minor 0 must resolve to GPU-BBB / NVML index 1.
func TestIdlePodGPUKeyResolvesMinorToNVMLIndex(t *testing.T) {
	// NVML index 0 = minor 1 = GPU-AAA; NVML index 1 = minor 0 = GPU-BBB.
	s := &metricsEngine{
		deviceUUIDs:            map[int]string{0: "GPU-AAA", 1: "GPU-BBB"},
		minorToNVMLIndex:       map[int]int{1: 0, 0: 1}, // minor 1→NVML 0, minor 0→NVML 1
		deviceTotalMemoryBytes: map[int]uint64{},
	}
	container := store.ContainerInfo{
		Namespace: "default", Pod: "pod", PodUID: "pod-uid",
	}

	// Device with minor 0 → should resolve to GPU-BBB (NVML index 1).
	key, ok := s.idlePodGPUKey(container, store.GPUDevice{MinorNumber: 0})
	if !ok {
		t.Fatalf("minor 0: expected ok=true, got false")
	}
	if key.GPUUUID != "GPU-BBB" {
		t.Fatalf("minor 0: expected UUID GPU-BBB, got %s", key.GPUUUID)
	}
	if key.GPUIndex != 1 {
		t.Fatalf("minor 0: expected NVML index 1, got %d", key.GPUIndex)
	}

	// Device with minor 1 → should resolve to GPU-AAA (NVML index 0).
	key, ok = s.idlePodGPUKey(container, store.GPUDevice{MinorNumber: 1})
	if !ok {
		t.Fatalf("minor 1: expected ok=true, got false")
	}
	if key.GPUUUID != "GPU-AAA" {
		t.Fatalf("minor 1: expected UUID GPU-AAA, got %s", key.GPUUUID)
	}
	if key.GPUIndex != 0 {
		t.Fatalf("minor 1: expected NVML index 0, got %d", key.GPUIndex)
	}
}

// TestPhantomMetricSuppressedWhenMinorDiffersFromNVMLIndex is the end-to-end
// regression test for the original bug: a pod on GPU-BBB (minor 0, NVML index 1)
// must not produce a phantom idle metric for GPU-AAA (minor 1, NVML index 0).
func TestPhantomMetricSuppressedWhenMinorDiffersFromNVMLIndex(t *testing.T) {
	// NVML index 0 → minor 1 → GPU-AAA; NVML index 1 → minor 0 → GPU-BBB.
	// The pod is assigned GPU-BBB, so its container exposes device minor 0.
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
		ContainerID: "ctr",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "pod-uid",
		CgroupPath:  "/kubepods/pod.scope/ctr.scope",
		GPUDevices:  []store.GPUDevice{{MinorNumber: 0}},
	}}}

	controller := newMetricsController(nil, fakeCgroupResolver{
		1001: []string{"/kubepods/pod.scope/ctr.scope/deeper"},
	}, containerStore, 0, 0, slog.Default())

	// Teach the engine the minor→NVML mapping and UUID table.
	controller.rememberDeviceUUIDs(map[int]string{0: "GPU-AAA", 1: "GPU-BBB"})
	controller.rememberMinorToNVMLIndex(map[int]int{1: 0, 0: 1})

	// Process: PID 1001 is running on GPU-BBB (NVML index 1).
	metrics, unmatched := controller.enrich(context.Background(), []GPUProcessMetric{
		{PID: 1001, GPUUUID: "GPU-BBB", GPUIndex: 1, UsedGPUMemoryBytes: 512, SMUtilizationPercent: 80},
	}, controller.pods.Snapshot())

	if unmatched != 0 {
		t.Fatalf("expected no unmatched processes, got %d", unmatched)
	}
	// Must be exactly one metric series — the real one for GPU-BBB.
	// Before the fix a phantom idle metric for GPU-AAA would also appear.
	if len(metrics) != 1 {
		t.Fatalf("expected 1 metric (GPU-BBB active), got %d: %#v", len(metrics), metrics)
	}
	m := metrics[0]
	if m.GPUUUID != "GPU-BBB" || m.GPUIndex != 1 {
		t.Fatalf("expected metric for GPU-BBB / NVML index 1, got %+v", m)
	}
	if m.SMUtilizationPercent != 80 {
		t.Fatalf("expected SM util 80, got %g", m.SMUtilizationPercent)
	}
}

// TestIdleMetricCorrectForIdlePodWithMismatchedMinor verifies that a truly idle
// pod (no running processes) on a GPU with minor ≠ NVML index gets the correct
// UUID and NVML index in its idle metric, not the GPU at NVML index 0.
func TestIdleMetricCorrectForIdlePodWithMismatchedMinor(t *testing.T) {
	// NVML index 0 → minor 1 → GPU-AAA; NVML index 1 → minor 0 → GPU-BBB.
	// Pod is assigned GPU-BBB (minor 0, NVML index 1) but is idle.
	containerStore := store.FakeStore{Containers: []store.ContainerInfo{{
		ContainerID: "ctr",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "pod-uid",
		GPUDevices:  []store.GPUDevice{{MinorNumber: 0}},
	}}}

	controller := newMetricsController(nil, fakeCgroupResolver{}, containerStore, 0, 0, slog.Default())
	controller.rememberDeviceUUIDs(map[int]string{0: "GPU-AAA", 1: "GPU-BBB"})
	controller.rememberMinorToNVMLIndex(map[int]int{1: 0, 0: 1})

	metrics, unmatched := controller.enrich(context.Background(), nil, controller.pods.Snapshot())

	if unmatched != 0 {
		t.Fatalf("expected no unmatched processes, got %d", unmatched)
	}
	if len(metrics) != 1 {
		t.Fatalf("expected 1 idle metric, got %d: %#v", len(metrics), metrics)
	}
	m := metrics[0]
	if m.GPUUUID != "GPU-BBB" {
		t.Fatalf("idle metric UUID: expected GPU-BBB, got %s", m.GPUUUID)
	}
	if m.GPUIndex != 1 {
		t.Fatalf("idle metric GPUIndex: expected 1 (NVML index), got %d", m.GPUIndex)
	}
	if m.SMUtilizationPercent != 0 || m.MemoryBytes != 0 {
		t.Fatalf("expected idle zeros, got %+v", m)
	}
}

// TestRememberMinorToNVMLIndexPersistsAcrossPartialUpdates mirrors the device
// total memory persistence test: a mapping learned on one cycle must survive a
// later cycle that omits it (transient NVML error on that device).
func TestRememberMinorToNVMLIndexPersistsAcrossPartialUpdates(t *testing.T) {
	s := &metricsEngine{minorToNVMLIndex: map[int]int{}}

	s.rememberMinorToNVMLIndex(map[int]int{0: 1, 1: 0})
	// Next cycle: only minor 0 reported; minor 1 absent (transient error).
	s.rememberMinorToNVMLIndex(map[int]int{0: 1})

	if s.minorToNVMLIndex[0] != 1 {
		t.Fatalf("expected minor 0 → NVML 1, got %d", s.minorToNVMLIndex[0])
	}
	if s.minorToNVMLIndex[1] != 0 {
		t.Fatalf("expected minor 1 → NVML 0 to persist, got %d", s.minorToNVMLIndex[1])
	}
}

// TestIdlePodGPUKeyFallsBackToIndexWhenNoMinorMapping covers the fake-GPU /
// noop-collector path where minorToNVMLIndex is empty. In that case
// GPUDevice.Index is used directly as the NVML index (the fake detector sets it).
func TestIdlePodGPUKeyFallsBackToIndexWhenNoMinorMapping(t *testing.T) {
	s := &metricsEngine{
		deviceUUIDs:            map[int]string{0: "GPU-0", 1: "GPU-1"},
		minorToNVMLIndex:       map[int]int{}, // empty = fake-GPU node
		deviceTotalMemoryBytes: map[int]uint64{},
	}
	container := store.ContainerInfo{
		Namespace: "default", Pod: "pod", PodUID: "pod-uid",
	}

	key, ok := s.idlePodGPUKey(container, store.GPUDevice{Index: 1})
	if !ok {
		t.Fatalf("expected ok=true for fake-GPU fallback, got false")
	}
	if key.GPUUUID != "GPU-1" {
		t.Fatalf("expected GPU-1 via Index fallback, got %s", key.GPUUUID)
	}
	if key.GPUIndex != 1 {
		t.Fatalf("expected GPUIndex 1, got %d", key.GPUIndex)
	}
}

// TestRecordAndSumRequestedMemory verifies per-container dedup (a container with
// multiple GPU processes counts once) and that empty/zero entries are ignored.
func TestRecordAndSumRequestedMemory(t *testing.T) {
	memByKey := map[podGPUKey]map[string]int64{}
	key := podGPUKey{PodUID: "pod-uid", GPUIndex: 0}

	recordRequestedMemory(memByKey, key, "ctr-a", 2500)
	recordRequestedMemory(memByKey, key, "ctr-a", 2500) // same container, second process → not double counted
	recordRequestedMemory(memByKey, key, "ctr-b", 1000)
	recordRequestedMemory(memByKey, key, "", 999)    // no container ID → ignored
	recordRequestedMemory(memByKey, key, "ctr-c", 0) // zero request → ignored

	if got := sumRequestedMemory(memByKey[key]); got != 3500 {
		t.Fatalf("expected summed memory 3500 (2500 once + 1000), got %d", got)
	}
	if got := sumRequestedMemory(nil); got != 0 {
		t.Fatalf("expected 0 for empty map, got %d", got)
	}
}
