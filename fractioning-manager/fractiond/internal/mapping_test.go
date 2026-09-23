// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"context"
	"io"
	"log/slog"
	"testing"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/fsstore"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
)

// testMemPrefix is the GPU-memory annotation prefix used by the mapping tests. A
// container is recorded for metrics iff it carries a well-formed annotation under
// this prefix — the same signal the mutation path enforces on.
const testMemPrefix = "nvidia.com/container."

// memAnnotations builds the pod annotations granting the named container a GPU
// memory limit (a Kubernetes quantity such as "4Gi").
func memAnnotations(container, limit string) map[string]string {
	return map[string]string{annotations.LimitAnnotationKey(testMemPrefix, container): limit}
}

// testMappingPlugin builds a plugin whose mapping handoff goes through a
// throwaway directory, plus a reader over the same directory so a test can
// observe what the metrics sidecar would read back.
func testMappingPlugin(t *testing.T) (*Plugin, *fsstore.Reader) {
	t.Helper()
	dir := t.TempDir()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	p, err := NewPlugin(Config{
		AnnotationPrefix: testMemPrefix,
		MPSPipeDirectory: "/run/nvidia-mps",
		MapDir:           dir,
		Log:              log,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p, fsstore.NewReader(dir, log)
}

// activeContainers drains queued events and returns the containers the metrics
// sidecar would attribute processes to.
func activeContainers(t *testing.T, p *Plugin, r *fsstore.Reader) []store.ContainerInfo {
	t.Helper()
	p.Flush()
	return r.ActiveContainers()
}

// gpuContainer builds a container the GPU detector recognizes: it carries NVIDIA
// device nodes (major 195) for the given minors — the only signal gpudevices
// reads to derive GPUDevice{MinorNumber: n}.
func gpuContainer(id, name, podSandboxID string, minors ...int64) *api.Container {
	devices := make([]*api.LinuxDevice, 0, len(minors))
	for _, minor := range minors {
		devices = append(devices, &api.LinuxDevice{Major: 195, Minor: minor})
	}
	return &api.Container{
		Id:           id,
		Name:         name,
		PodSandboxId: podSandboxID,
		Linux:        &api.LinuxContainer{Devices: devices},
	}
}

func TestCreateContainerRecordsGPUMapping(t *testing.T) {
	plugin, reader := testMappingPlugin(t)

	if _, _, err := plugin.CreateContainer(context.Background(),
		&api.PodSandbox{
			Name: "pod", Namespace: "default", Uid: "pod-uid",
			Annotations: memAnnotations("container", "4Gi"),
		},
		gpuContainer("container-id", "container", "", 0, 1),
	); err != nil {
		t.Fatalf("CreateContainer returned error: %v", err)
	}

	got := activeContainers(t, plugin, reader)
	if len(got) != 1 {
		t.Fatalf("expected one recorded mapping, got %d", len(got))
	}
	info := got[0]
	if info.ContainerID != "container-id" || info.Pod != "pod" || info.Namespace != "default" || info.PodUID != "pod-uid" {
		t.Fatalf("unexpected mapping identity: %#v", info)
	}
	if len(info.GPUDevices) != 2 {
		t.Fatalf("expected two GPU devices recorded, got %#v", info.GPUDevices)
	}
	if info.RequestedMemoryMiB != 4096 {
		t.Fatalf("expected RequestedMemoryMiB 4096, got %d", info.RequestedMemoryMiB)
	}
}

func TestCreateContainerIgnoresNonGPUContainer(t *testing.T) {
	plugin, reader := testMappingPlugin(t)

	if _, _, err := plugin.CreateContainer(context.Background(),
		&api.PodSandbox{Name: "pod"},
		&api.Container{Id: "container-id", Name: "container"},
	); err != nil {
		t.Fatalf("CreateContainer returned error: %v", err)
	}

	if n := len(activeContainers(t, plugin, reader)); n != 0 {
		t.Fatalf("expected non-GPU container to be ignored, got %d mappings", n)
	}
}

// A fail-closed annotation parse error must block the container AND skip the
// mapping: the container will not exist, so recording it would be stale.
func TestCreateContainerFailClosedDoesNotRecordMapping(t *testing.T) {
	plugin, reader := testMappingPlugin(t)

	_, _, err := plugin.CreateContainer(context.Background(),
		&api.PodSandbox{
			Name: "pod", Uid: "pod-uid",
			Annotations: memAnnotations("container", "not-a-quantity"),
		},
		gpuContainer("container-id", "container", "", 0),
	)
	if err == nil {
		t.Fatal("expected fail-closed error for malformed annotation")
	}
	if n := len(activeContainers(t, plugin, reader)); n != 0 {
		t.Fatalf("expected no mapping recorded on fail-closed, got %d", n)
	}
}

func TestRemoveContainerDeletesMapping(t *testing.T) {
	plugin, reader := testMappingPlugin(t)

	if _, _, err := plugin.CreateContainer(context.Background(),
		&api.PodSandbox{
			Name: "pod", Uid: "pod-uid",
			Annotations: memAnnotations("container", "4Gi"),
		},
		gpuContainer("container-id", "container", "", 0),
	); err != nil {
		t.Fatalf("CreateContainer returned error: %v", err)
	}
	if n := len(activeContainers(t, plugin, reader)); n != 1 {
		t.Fatalf("expected mapping to be recorded before delete, got %d", n)
	}

	if err := plugin.RemoveContainer(context.Background(), &api.PodSandbox{}, &api.Container{Id: "container-id"}); err != nil {
		t.Fatalf("RemoveContainer returned error: %v", err)
	}

	if n := len(activeContainers(t, plugin, reader)); n != 0 {
		t.Fatalf("expected mapping to be deleted, got %d", n)
	}
}

func TestSynchronizeReplacesMappings(t *testing.T) {
	plugin, reader := testMappingPlugin(t)

	updates, err := plugin.Synchronize(context.Background(),
		[]*api.PodSandbox{{
			Id: "pod-id", Name: "pod", Namespace: "default", Uid: "pod-uid",
			Annotations: memAnnotations("gpu", "4Gi"),
		}},
		[]*api.Container{
			gpuContainer("gpu-container", "gpu", "pod-id", 0),
			{Id: "plain-container", Name: "plain", PodSandboxId: "pod-id"}, // no annotation → ignored
		},
	)
	if err != nil {
		t.Fatalf("Synchronize returned error: %v", err)
	}
	if len(updates) != 0 {
		t.Fatalf("expected no container updates, got %d", len(updates))
	}

	got := activeContainers(t, plugin, reader)
	if len(got) != 1 {
		t.Fatalf("expected one GPU mapping after sync, got %d", len(got))
	}
	if got[0].ContainerID != "gpu-container" || got[0].Pod != "pod" {
		t.Fatalf("unexpected synchronized mapping: %#v", got[0])
	}
}
