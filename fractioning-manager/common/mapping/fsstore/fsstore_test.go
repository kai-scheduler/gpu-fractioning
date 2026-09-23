// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fsstore

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

func newTestWriterReader(t *testing.T) (*Writer, *Reader, string) {
	t.Helper()
	dir := t.TempDir()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewWriter(dir, logger), NewReader(dir, logger), dir
}

func reload(t *testing.T, r *Reader) []store.ContainerInfo {
	t.Helper()
	return r.ActiveContainers()
}

func TestWriterUpsertRoundTripsThroughReader(t *testing.T) {
	w, r, _ := newTestWriterReader(t)

	w.Upsert(store.ContainerInfo{
		ContainerID: "0123456789abcdef",
		Container:   "trainer",
		Pod:         "trainer-7d9f",
		Namespace:   "team-ml",
		PodUID:      "3f2a-uid",
		CgroupPath:  "/should/not/be/persisted",
		GPUDevices:  []store.GPUDevice{{Index: 0}, {Index: 3}},
	})

	got := reload(t, r)
	if len(got) != 1 {
		t.Fatalf("expected one container, got %d", len(got))
	}
	info := got[0]
	if info.ContainerID != "0123456789abcdef" {
		t.Fatalf("unexpected container ID: %q", info.ContainerID)
	}
	if info.Pod != "trainer-7d9f" || info.Namespace != "team-ml" || info.PodUID != "3f2a-uid" {
		t.Fatalf("unexpected pod identity: %#v", info)
	}
	if len(info.GPUDevices) != 2 || info.GPUDevices[0].Index != 0 || info.GPUDevices[1].Index != 3 {
		t.Fatalf("unexpected GPU devices: %#v", info.GPUDevices)
	}
}

func TestRecordOmitsCgroupPathAndContainerName(t *testing.T) {
	w, _, dir := newTestWriterReader(t)

	w.Upsert(store.ContainerInfo{
		ContainerID: "abc",
		Container:   "trainer",
		Pod:         "pod",
		Namespace:   "ns",
		PodUID:      "uid",
		CgroupPath:  "/some/cgroup",
		GPUDevices:  []store.GPUDevice{{Index: 1, UUID: "GPU-should-not-persist"}},
	})

	data, err := os.ReadFile(filepath.Join(dir, "abc.json"))
	if err != nil {
		t.Fatalf("read mapping file: %v", err)
	}
	// The on-disk schema is pod identity + gpu indices only
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := raw["cgroupPath"]; ok {
		t.Fatalf("cgroup path must not be persisted: %s", data)
	}
	if _, ok := raw["containerID"]; ok {
		t.Fatalf("container ID is the file name, not a field: %s", data)
	}
	if v, _ := raw["schemaVersion"].(float64); int(v) != schemaVersion {
		t.Fatalf("expected schemaVersion %d, got %v", schemaVersion, raw["schemaVersion"])
	}
	// UUID is persisted so the controller can populate gpu_uuid without NVML
	// (e.g. fake-GPU clusters where MOCK_NVIDIA_VISIBLE_DEVICES carries the UUID).
	if !strings.Contains(string(data), "GPU-should-not-persist") {
		t.Fatalf("UUID must be persisted for fake-GPU resolution: %s", data)
	}
}

func TestWriterUpsertIsAtomicAndLeavesNoTempFiles(t *testing.T) {
	w, r, dir := newTestWriterReader(t)

	w.Upsert(store.ContainerInfo{ContainerID: "abc", Pod: "p", PodUID: "u", GPUDevices: []store.GPUDevice{{Index: 0}}})

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	if len(entries) != 1 || entries[0].Name() != "abc.json" {
		t.Fatalf("expected exactly abc.json, got %v", entries)
	}
	if n := len(reload(t, r)); n != 1 {
		t.Fatalf("expected one container after atomic write, got %d", n)
	}
}

func TestWriterDeleteRemovesMappingFile(t *testing.T) {
	w, r, _ := newTestWriterReader(t)

	w.Upsert(store.ContainerInfo{ContainerID: "abc", PodUID: "u", GPUDevices: []store.GPUDevice{{Index: 0}}})
	if n := len(reload(t, r)); n != 1 {
		t.Fatalf("expected one container before delete, got %d", n)
	}

	w.Delete("abc")
	if n := len(reload(t, r)); n != 0 {
		t.Fatalf("expected no containers after delete, got %d", n)
	}
}

func TestWriterDeleteMissingFileIsNoError(t *testing.T) {
	w, r, _ := newTestWriterReader(t)
	w.Delete("never-written") // must not panic or leave state
	if n := len(reload(t, r)); n != 0 {
		t.Fatalf("expected empty mapping, got %d", n)
	}
}

func TestWriterReplaceReconcilesDirectory(t *testing.T) {
	w, r, _ := newTestWriterReader(t)

	// Seed two containers, then Replace with a set that drops one and adds one.
	w.Upsert(store.ContainerInfo{ContainerID: "stale", PodUID: "u1", GPUDevices: []store.GPUDevice{{Index: 0}}})
	w.Upsert(store.ContainerInfo{ContainerID: "keep", PodUID: "u2", GPUDevices: []store.GPUDevice{{Index: 1}}})

	w.Replace([]store.ContainerInfo{
		{ContainerID: "keep", PodUID: "u2", GPUDevices: []store.GPUDevice{{Index: 1}}},
		{ContainerID: "fresh", PodUID: "u3", GPUDevices: []store.GPUDevice{{Index: 2}}},
	})

	got := reload(t, r)
	ids := map[string]struct{}{}
	for _, info := range got {
		ids[info.ContainerID] = struct{}{}
	}
	if len(ids) != 2 {
		t.Fatalf("expected two containers after reconcile, got %v", ids)
	}
	if _, ok := ids["stale"]; ok {
		t.Fatalf("expected stale container to be removed, got %v", ids)
	}
	if _, ok := ids["keep"]; !ok {
		t.Fatalf("expected kept container to remain, got %v", ids)
	}
	if _, ok := ids["fresh"]; !ok {
		t.Fatalf("expected fresh container to be added, got %v", ids)
	}
}

func TestReaderMatchesProcessByContainerIDInCgroupPath(t *testing.T) {
	w, r, _ := newTestWriterReader(t)
	w.Upsert(store.ContainerInfo{
		ContainerID: "0123456789abcdef",
		Pod:         "pod",
		Namespace:   "default",
		PodUID:      "uid",
		GPUDevices:  []store.GPUDevice{{Index: 0}},
	})

	info, ok := r.MatchCgroupPaths([]string{
		"/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod123.slice/cri-containerd-0123456789abcdef.scope",
	})
	if !ok {
		t.Fatalf("expected process cgroup path to match the recorded container ID")
	}
	if info.Pod != "pod" || info.PodUID != "uid" {
		t.Fatalf("unexpected matched container: %#v", info)
	}
}

func TestReaderMissingDirectoryIsEmptyMapping(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	r := NewReader(filepath.Join(t.TempDir(), "does-not-exist"), logger)
	if n := len(r.ActiveContainers()); n != 0 {
		t.Fatalf("expected empty mapping for missing dir, got %d", n)
	}
}

func TestReaderSkipsUnrelatedAndUnparseableFiles(t *testing.T) {
	w, r, dir := newTestWriterReader(t)
	w.Upsert(store.ContainerInfo{ContainerID: "good", PodUID: "u", GPUDevices: []store.GPUDevice{{Index: 0}}})

	// A non-json file, a dot-prefixed temp leftover and a corrupt json file must
	// all be ignored without dropping the good mapping.
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("ignore me"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".gpu-fractioning-123.tmp"), []byte("partial"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "corrupt.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := reload(t, r)
	if len(got) != 1 || got[0].ContainerID != "good" {
		t.Fatalf("expected only the good mapping, got %#v", got)
	}
}

func TestReaderIgnoresUnsupportedSchemaVersion(t *testing.T) {
	_, r, dir := newTestWriterReader(t)
	future := record{SchemaVersion: schemaVersion + 1, Pod: "p", PodUID: "u"}
	data, err := json.Marshal(future)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "future.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	if n := len(reload(t, r)); n != 0 {
		t.Fatalf("expected file with unsupported schema version to be ignored, got %d", n)
	}
}
