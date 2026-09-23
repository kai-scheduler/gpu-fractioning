// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

// Package-internal e2e tests (not package metrics_test) so they can access
// unexported controller state and snapshot internals. Compiled only with
// -tags e2e; require a fake-GPU cluster.
package metrics

// End-to-end metrics pipeline tests: fake GPU collector -> controller -> Prometheus
// exporter -> HTTP scrape. No real GPU, NVML, or /proc access required.
//
// Scenario: two pods (pod-a "trainer", pod-b "worker") each request 0.5 of the
// same physical GPU (GPU fractioning). Both have live processes that allocate
// GPU memory. The tests validate that the exporter reports correct per-pod
// memory and SM-utilization metrics, and that neither pod's numbers bleed into
// the other's series.

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

// Fixed identities for the two-pod fractional-GPU scenario.
const (
	e2eNamespace = "default"
	e2eGPUUUID   = "GPU-abc123"
	e2eGPUIndex  = 0

	e2ePIDPodA uint32 = 1001
	e2ePIDPodB uint32 = 1002

	// GPU memory each pod's process allocates — validated by the memory-bytes metric.
	e2eMemBytesPodA uint64 = 512 * 1024 * 1024 // 512 MiB
	e2eMemBytesPodB uint64 = 256 * 1024 * 1024 // 256 MiB

	// SM utilization reported by NVML for each process.
	e2eSMUtilPodA uint32 = 30
	e2eSMUtilPodB uint32 = 50

	// The simulated device's total memory (MiB). Each pod requests half of
	// it, so its derived GPU fraction is 0.5.
	e2eDeviceTotalMemMiB = 10000
	// GPU memory each pod requests (MiB) — half the device.
	e2eRequestedMemoryMiB int64 = e2eDeviceTotalMemMiB / 2

	// Each pod requested half of the shared GPU, so normalized SM utilization is
	// SMUtil ÷ 0.5: pod-a -> 60, pod-b -> 100 (capped).
	e2eRequestedFraction = 0.5
)

// e2ePodSource is a PodSource that resolves GPU processes to pods by PID lookup
// and reports a fixed set of active containers. It replaces the NRI cgroup
// resolver so tests require no kernel or /proc access.
type e2ePodSource struct {
	byPID      map[uint32]store.ContainerInfo
	containers []store.ContainerInfo
}

// Snapshot returns the source itself: the fixture's container set is fixed for
// the duration of a test, so no point-in-time copy is needed.
func (s *e2ePodSource) Snapshot() podSource { return s }

func (s *e2ePodSource) ResolveProcess(p GPUProcessMetric) (store.ContainerInfo, bool) {
	info, ok := s.byPID[p.PID]
	return info, ok
}

func (s *e2ePodSource) ActivePodUIDs() map[string]struct{} {
	out := make(map[string]struct{}, len(s.containers))
	for _, c := range s.containers {
		if c.PodUID != "" {
			out[c.PodUID] = struct{}{}
		}
	}
	return out
}

func (s *e2ePodSource) ActiveContainers() []store.ContainerInfo {
	return s.containers
}

// twoFractionalPodsFixture builds and wires the full pipeline for the
// fractional-GPU scenario then runs one collect so the exporter registry is
// populated. The returned *Runtime is ready to serve /metrics or to be
// inspected via gatheredGaugeValue.
func twoFractionalPodsFixture(t *testing.T) *Runtime {
	t.Helper()

	// Both pods reference GPU index 0 — they share one physical device (0.5 each).
	podA := store.ContainerInfo{
		ContainerID:       "ctr-pod-a",
		Container:         "trainer",
		Pod:               "pod-a",
		Namespace:         e2eNamespace,
		PodUID:            "uid-pod-a",
		GPUDevices:        []store.GPUDevice{{Index: e2eGPUIndex}},
		RequestedMemoryMiB: e2eRequestedMemoryMiB,
	}
	podB := store.ContainerInfo{
		ContainerID:       "ctr-pod-b",
		Container:         "worker",
		Pod:               "pod-b",
		Namespace:         e2eNamespace,
		PodUID:            "uid-pod-b",
		GPUDevices:        []store.GPUDevice{{Index: e2eGPUIndex}},
		RequestedMemoryMiB: e2eRequestedMemoryMiB,
	}

	pods := &e2ePodSource{
		byPID:      map[uint32]store.ContainerInfo{e2ePIDPodA: podA, e2ePIDPodB: podB},
		containers: []store.ContainerInfo{podA, podB},
	}

	// Two GPU processes on the same device — one per pod.
	collector := &fakeCollector{
		snapshot: GPUProcessSnapshot{
			Processes: []GPUProcessMetric{
				{
					PID:                  e2ePIDPodA,
					GPUUUID:              e2eGPUUUID,
					GPUIndex:             e2eGPUIndex,
					UsedGPUMemoryBytes:   e2eMemBytesPodA,
					SMUtilizationPercent: e2eSMUtilPodA,
				},
				{
					PID:                  e2ePIDPodB,
					GPUUUID:              e2eGPUUUID,
					GPUIndex:             e2eGPUIndex,
					UsedGPUMemoryBytes:   e2eMemBytesPodB,
					SMUtilizationPercent: e2eSMUtilPodB,
				},
			},
			DeviceUUIDs:            map[int]string{e2eGPUIndex: e2eGPUUUID},
			DeviceTotalMemoryBytes: map[int]uint64{e2eGPUIndex: e2eDeviceTotalMemMiB * bytesPerMiB},
		},
	}

	controller := newMetricsControllerWithPodSource(collector, pods, 0, 0, slog.Default())
	controller.collect(context.Background())

	runtime := newRuntime(controller, MetricNames{}.WithDefaults())
	// refresh() calls controller.Snapshot() and writes values into the Prometheus registry.
	runtime.refresh(context.Background())
	return runtime
}

// podLabels returns the full Prometheus label set for a pod + GPU combination.
func podLabels(pod, podUID string) map[string]string {
	return map[string]string{
		"namespace": e2eNamespace,
		"pod":       pod,
		"pod_uuid":  podUID,
		"gpu_uuid":  e2eGPUUUID,
		"gpu":       "0",
	}
}

// TestE2EFractionalGPUMemoryBytes validates that each pod's
// gpu_fractioning_gpu_memory_used_bytes metric matches the GPU memory its process
// allocated, with no cross-pod contamination.
func TestE2EFractionalGPUMemoryBytes(t *testing.T) {
	exporter := twoFractionalPodsFixture(t)

	tests := []struct {
		pod       string
		podUID    string
		wantBytes uint64
	}{
		{"pod-a", "uid-pod-a", e2eMemBytesPodA},
		{"pod-b", "uid-pod-b", e2eMemBytesPodB},
	}
	for _, tt := range tests {
		t.Run(tt.pod, func(t *testing.T) {
			got, ok := gatheredGaugeValue(t, exporter, "gpu_fractioning_gpu_memory_used_bytes", podLabels(tt.pod, tt.podUID))
			if !ok {
				t.Fatalf("%s: expected gpu_fractioning_gpu_memory_used_bytes series to be present", tt.pod)
			}
			if got != float64(tt.wantBytes) {
				t.Fatalf("%s: gpu_fractioning_gpu_memory_used_bytes = %g bytes, want %d (%d MiB)",
					tt.pod, got, tt.wantBytes, tt.wantBytes>>20)
			}
		})
	}

	// Sanity: the two pods' memory values must be distinct and must not equal
	// the combined total — they are tracked independently, not summed.
	gotA, _ := gatheredGaugeValue(t, exporter, "gpu_fractioning_gpu_memory_used_bytes", podLabels("pod-a", "uid-pod-a"))
	gotB, _ := gatheredGaugeValue(t, exporter, "gpu_fractioning_gpu_memory_used_bytes", podLabels("pod-b", "uid-pod-b"))
	if gotA == gotB {
		t.Fatalf("pod-a and pod-b reported identical memory bytes (%g); they should be distinct", gotA)
	}
	combined := float64(e2eMemBytesPodA + e2eMemBytesPodB)
	if gotA == combined || gotB == combined {
		t.Fatalf("one pod's memory bytes equals the combined total (%g); memory is being summed across pods", combined)
	}
}

// TestE2EFractionalGPUSMUtilization validates that SM utilization is
// reported per-pod and is not averaged or merged across the two pods sharing
// the same physical GPU.
func TestE2EFractionalGPUSMUtilization(t *testing.T) {
	exporter := twoFractionalPodsFixture(t)

	tests := []struct {
		pod      string
		podUID   string
		wantUtil float64
	}{
		{"pod-a", "uid-pod-a", float64(e2eSMUtilPodA)},
		{"pod-b", "uid-pod-b", float64(e2eSMUtilPodB)},
	}
	for _, tt := range tests {
		t.Run(tt.pod, func(t *testing.T) {
			got, ok := gatheredGaugeValue(t, exporter, "gpu_fractioning_gpu_sm_utilization_percent", podLabels(tt.pod, tt.podUID))
			if !ok {
				t.Fatalf("%s: expected gpu_fractioning_gpu_sm_utilization_percent series to be present", tt.pod)
			}
			if got != tt.wantUtil {
				t.Fatalf("%s: gpu_fractioning_gpu_sm_utilization_percent = %g, want %g", tt.pod, got, tt.wantUtil)
			}
		})
	}
}

// TestE2EFractionalGPUSMUtilizationNormalized validates that SM
// utilization is normalized per-pod by the requested GPU fraction (0.5 each) and
// capped at 100: pod-a (30% used) -> 60, pod-b (50% used) -> 100.
func TestE2EFractionalGPUSMUtilizationNormalized(t *testing.T) {
	exporter := twoFractionalPodsFixture(t)

	tests := []struct {
		pod      string
		podUID   string
		wantNorm float64
	}{
		{"pod-a", "uid-pod-a", float64(e2eSMUtilPodA) / e2eRequestedFraction}, // 60
		{"pod-b", "uid-pod-b", 100}, // 50/0.5 = 100 (at cap)
	}
	for _, tt := range tests {
		t.Run(tt.pod, func(t *testing.T) {
			got, ok := gatheredGaugeValue(t, exporter, "gpu_fractioning_gpu_sm_utilization_percent_normalized", podLabels(tt.pod, tt.podUID))
			if !ok {
				t.Fatalf("%s: expected gpu_fractioning_gpu_sm_utilization_percent_normalized series to be present", tt.pod)
			}
			if got != tt.wantNorm {
				t.Fatalf("%s: normalized SM util = %g, want %g", tt.pod, got, tt.wantNorm)
			}
		})
	}
}

// TestE2EFractionalGPUBothPodsHaveAllMetrics validates that both pods
// have all expected metric families present — neither pod is silently absent.
func TestE2EFractionalGPUBothPodsHaveAllMetrics(t *testing.T) {
	exporter := twoFractionalPodsFixture(t)

	podCases := []struct{ pod, uid string }{
		{"pod-a", "uid-pod-a"},
		{"pod-b", "uid-pod-b"},
	}
	metricNames := []string{
		"gpu_fractioning_gpu_memory_used_bytes",
		"gpu_fractioning_gpu_sm_utilization_percent",
		"gpu_fractioning_gpu_sm_utilization_percent_normalized",
	}

	for _, pc := range podCases {
		for _, name := range metricNames {
			if _, ok := gatheredGaugeValue(t, exporter, name, podLabels(pc.pod, pc.uid)); !ok {
				t.Errorf("%s/%s: missing metric %s", pc.pod, pc.uid, name)
			}
		}
	}
}

// TestE2EFractionalGPUPrometheusEndpoint validates the HTTP /metrics
// scrape endpoint for the two-pod scenario. It verifies:
//   - 200 OK with Prometheus text Content-Type
//   - Both pods' labels appear in the response body
//   - Both metric family names are present
//   - The GPU UUID label is present for both pods
func TestE2EFractionalGPUPrometheusEndpoint(t *testing.T) {
	exporter := twoFractionalPodsFixture(t)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()
	exporter.handler().ServeHTTP(w, req)
	resp := w.Result()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/plain") {
		t.Fatalf("expected text/plain Content-Type, got %q", ct)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	bodyStr := string(body)

	required := []string{
		// Metric family names
		"gpu_fractioning_gpu_memory_used_bytes",
		"gpu_fractioning_gpu_sm_utilization_percent",
		"gpu_fractioning_gpu_sm_utilization_percent_normalized",
		// Both pods present with correct labels
		`pod="pod-a"`, `pod_uuid="uid-pod-a"`,
		`pod="pod-b"`, `pod_uuid="uid-pod-b"`,
		// Shared GPU UUID appears (at least once per pod per metric)
		`gpu_uuid="` + e2eGPUUUID + `"`,
		// Namespace label
		`namespace="` + e2eNamespace + `"`,
	}
	for _, want := range required {
		if !strings.Contains(bodyStr, want) {
			t.Errorf("expected %q in /metrics body\n--- body ---\n%s", want, bodyStr)
		}
	}
}

// TestE2EFractionalGPUPodDisappears validates that when a pod's GPU
// processes stop (simulated by an empty snapshot on the second collect), its
// series transitions to zero rather than being immediately removed — it is
// cleaned up only when the pod leaves the active-pod set.
func TestE2EFractionalGPUPodDisappears(t *testing.T) {
	podA := store.ContainerInfo{
		ContainerID: "ctr-pod-a",
		Container:   "trainer",
		Pod:         "pod-a",
		Namespace:   e2eNamespace,
		PodUID:      "uid-pod-a",
		GPUDevices:  []store.GPUDevice{{Index: e2eGPUIndex}},
	}

	// First collect: pod-a has an active process.
	collector := &fakeCollector{
		snapshot: GPUProcessSnapshot{
			Processes: []GPUProcessMetric{{
				PID:                  e2ePIDPodA,
				GPUUUID:              e2eGPUUUID,
				GPUIndex:             e2eGPUIndex,
				UsedGPUMemoryBytes:   e2eMemBytesPodA,
				SMUtilizationPercent: e2eSMUtilPodA,
			}},
			DeviceUUIDs: map[int]string{e2eGPUIndex: e2eGPUUUID},
		},
	}
	pods := &e2ePodSource{
		byPID:      map[uint32]store.ContainerInfo{e2ePIDPodA: podA},
		containers: []store.ContainerInfo{podA},
	}
	controller := newMetricsControllerWithPodSource(collector, pods, 0, 0, slog.Default())
	controller.collect(context.Background())

	runtime := newRuntime(controller, MetricNames{}.WithDefaults())
	runtime.refresh(context.Background())

	mem, ok := gatheredGaugeValue(t, runtime, "gpu_fractioning_gpu_memory_used_bytes", podLabels("pod-a", "uid-pod-a"))
	if !ok {
		t.Fatalf("expected pod-a memory series after first collect")
	}
	if mem != float64(e2eMemBytesPodA) {
		t.Fatalf("expected %d bytes after first collect, got %g", e2eMemBytesPodA, mem)
	}

	// Second collect: GPU process stopped — snapshot is empty but pod is still active.
	// Series must remain, reset to zero (idle).
	collector.snapshot = GPUProcessSnapshot{
		Processes:   nil,
		DeviceUUIDs: map[int]string{e2eGPUIndex: e2eGPUUUID},
	}
	controller.collect(context.Background())
	runtime.refresh(context.Background())

	mem, ok = gatheredGaugeValue(t, runtime, "gpu_fractioning_gpu_memory_used_bytes", podLabels("pod-a", "uid-pod-a"))
	if !ok {
		t.Fatalf("expected pod-a series to stay present (zero) when pod is still active but GPU process stopped")
	}
	if mem != 0 {
		t.Fatalf("expected zero memory after process stopped, got %g", mem)
	}

	// Third collect: pod leaves (removed from active containers). Series must be deleted.
	pods.containers = nil
	pods.byPID = nil
	controller.collect(context.Background())
	runtime.refresh(context.Background())

	if _, ok := gatheredGaugeValue(t, runtime, "gpu_fractioning_gpu_memory_used_bytes", podLabels("pod-a", "uid-pod-a")); ok {
		t.Fatalf("expected pod-a series to be deleted after pod left active containers")
	}
}
