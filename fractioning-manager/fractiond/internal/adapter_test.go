// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"testing"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
)

// newAdapter returns an adapter configured with the default GPU-memory prefix,
// matching what the mutation path uses.
func newAdapter() adapter {
	return adapter{annotationPrefix: testMemPrefix}
}

// gpuMemPod builds a pod sandbox granting the named container GPU memory. Pass
// empty strings to omit either annotation. Values are Kubernetes quantities.
func gpuMemPod(containerName, limit, request string) *api.PodSandbox {
	ann := map[string]string{}
	if limit != "" {
		ann[annotations.LimitAnnotationKey(testMemPrefix, containerName)] = limit
	}
	if request != "" {
		ann[annotations.RequestAnnotationKey(testMemPrefix, containerName)] = request
	}
	return &api.PodSandbox{
		Id:          "pod-id",
		Name:        "pod",
		Namespace:   "default",
		Uid:         "pod-uid",
		Annotations: ann,
	}
}

// These tests cover the adapter's own responsibilities — assembling identity and
// gating on the GPU-memory annotation — independent of which GPU detector the
// build selects.

func TestContainerAssemblesPodIdentityAndCgroupPath(t *testing.T) {
	c := gpuContainer("container-id", "trainer", "pod-id", 0)
	c.Linux.CgroupsPath = "/container/cgroup"

	info, ok := newAdapter().container(gpuMemPod("trainer", "4Gi", ""), c)
	if !ok {
		t.Fatalf("expected container info")
	}
	if info.CgroupPath != "/container/cgroup" {
		t.Fatalf("expected cgroup path, got %q", info.CgroupPath)
	}
	if info.ContainerID != "container-id" || info.Container != "trainer" {
		t.Fatalf("expected container identity to be copied, got %#v", info)
	}
	if info.Pod != "pod" || info.Namespace != "default" || info.PodUID != "pod-uid" {
		t.Fatalf("expected pod identity to be copied, got %#v", info)
	}
	if len(info.GPUDevices) != 1 {
		t.Fatalf("expected one GPU device, got %#v", info.GPUDevices)
	}
}

func TestContainerSkippedWhenNoGPUMemoryAnnotation(t *testing.T) {
	// A container with device nodes but no GPU-memory annotation must be dropped —
	// only GPU-fractioning containers are tracked.
	_, ok := newAdapter().container(
		&api.PodSandbox{Id: "pod-id"},
		gpuContainer("gpu-container", "trainer", "pod-id", 0),
	)
	if ok {
		t.Fatalf("expected container without GPU-memory annotation to be skipped")
	}
}

func TestContainerSkippedWhenMissingID(t *testing.T) {
	if _, ok := newAdapter().container(&api.PodSandbox{}, &api.Container{}); ok {
		t.Fatalf("expected container with empty ID to be skipped")
	}
	if _, ok := newAdapter().container(&api.PodSandbox{}, nil); ok {
		t.Fatalf("expected nil container to be skipped")
	}
}

func TestContainerLimitAnnotationRecordsRequestedMemory(t *testing.T) {
	// Only the limit annotation is present; the container qualifies and the limit
	// becomes RequestedMemoryMiB (4Gi -> 4096 MiB).
	info, ok := newAdapter().container(
		gpuMemPod("trainer", "4Gi", ""),
		&api.Container{Id: "mps-client", Name: "trainer", PodSandboxId: "pod-id"},
	)
	if !ok {
		t.Fatalf("expected GPU-fractioning container to be included")
	}
	if info.RequestedMemoryMiB != 4096 {
		t.Fatalf("expected RequestedMemoryMiB 4096, got %d", info.RequestedMemoryMiB)
	}
}

func TestContainerRequestAnnotationUsedWhenNoLimit(t *testing.T) {
	// Only the request annotation is present — it qualifies and provides the
	// requested memory (2Gi -> 2048 MiB).
	info, ok := newAdapter().container(
		gpuMemPod("trainer", "", "2Gi"),
		&api.Container{Id: "mps-client", Name: "trainer", PodSandboxId: "pod-id"},
	)
	if !ok {
		t.Fatalf("expected GPU-fractioning container to be included")
	}
	if info.RequestedMemoryMiB != 2048 {
		t.Fatalf("expected RequestedMemoryMiB 2048 from request, got %d", info.RequestedMemoryMiB)
	}
}

func TestContainerPrefersLimitOverRequestForMemory(t *testing.T) {
	info, ok := newAdapter().container(
		gpuMemPod("trainer", "4Gi", "2Gi"),
		&api.Container{Id: "mps-client", Name: "trainer", PodSandboxId: "pod-id"},
	)
	if !ok {
		t.Fatalf("expected GPU-fractioning container to be included")
	}
	if info.RequestedMemoryMiB != 4096 {
		t.Fatalf("expected RequestedMemoryMiB to prefer the limit (4096), got %d", info.RequestedMemoryMiB)
	}
}

func TestContainerMalformedAnnotationSkipped(t *testing.T) {
	// A malformed quantity makes ParseGPUMemoryAnnotations error; the mutation
	// path fail-closes, so the mapping must be skipped too.
	_, ok := newAdapter().container(
		gpuMemPod("trainer", "not-a-quantity", ""),
		&api.Container{Id: "c", Name: "trainer", PodSandboxId: "pod-id"},
	)
	if ok {
		t.Fatalf("expected container with malformed annotation to be skipped")
	}
}

func TestContainerWithDeviceNodePopulatesDevices(t *testing.T) {
	c := gpuContainer("c", "trainer", "pod-id", 0)
	info, ok := newAdapter().container(gpuMemPod("trainer", "4Gi", ""), c)
	if !ok {
		t.Fatalf("expected container to be included")
	}
	if len(info.GPUDevices) == 0 {
		t.Fatalf("expected device nodes to be present")
	}
}

func TestContainerSiblingWithoutAnnotationDropped(t *testing.T) {
	// A sidecar on a GPU-fractioning pod has no annotation for its own name and must
	// be dropped even though the pod carries an annotation for the "trainer"
	// container.
	pod := gpuMemPod("trainer", "4Gi", "")
	_, ok := newAdapter().container(pod, &api.Container{Id: "sidecar", Name: "sidecar", PodSandboxId: "pod-id"})
	if ok {
		t.Fatalf("expected sidecar container without its own annotation to be skipped")
	}
}

func TestContainersDropsNonGPUAndSiblingContainers(t *testing.T) {
	// Only the container whose name is referenced in the pod annotation is kept.
	// The full-GPU pod's container and the sidecar on the fractioning pod are both
	// dropped.
	infos := newAdapter().containers(
		[]*api.PodSandbox{
			{Id: "frac-pod-id", Name: "frac-pod", Namespace: "default", Uid: "frac-uid",
				Annotations: map[string]string{annotations.LimitAnnotationKey(testMemPrefix, "trainer"): "4Gi"}},
			{Id: "full-pod-id", Name: "full-pod", Namespace: "default", Uid: "full-uid"},
		},
		[]*api.Container{
			gpuContainer("frac-container", "trainer", "frac-pod-id", 0),   // annotation matches → kept
			{Id: "sidecar", Name: "sidecar", PodSandboxId: "frac-pod-id"}, // no annotation for "sidecar" → dropped
			gpuContainer("full-container", "gpu", "full-pod-id", 0),       // pod has no annotation → dropped
		},
	)
	if len(infos) != 1 {
		t.Fatalf("expected one GPU mapping, got %d: %#v", len(infos), infos)
	}
	if infos[0].ContainerID != "frac-container" || infos[0].Pod != "frac-pod" {
		t.Fatalf("unexpected mapping: %#v", infos[0])
	}
}
