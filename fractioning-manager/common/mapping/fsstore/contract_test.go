// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fsstore

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

// goldenRecordV1 is the frozen on-disk wire contract for the container→pod
// mapping handoff (schemaVersion 1). fractiond writes these records and the
// metricsd sidecar (a separate Go module) reads them from the shared MapDir, so
// the JSON shape is a cross-container contract. This test freezes it: any change
// to the record's fields or JSON tags must be a deliberate schemaVersion bump,
// which forces this golden to be updated in lockstep.
const goldenRecordV1 = `{"schemaVersion":1,"pod":"trainer-7d9f","namespace":"team-ml","podUID":"3f2a-uid","gpuDevices":[{"index":0,"minorNumber":0},{"index":3,"minorNumber":0}],"requestedMemoryMiB":2048}`

// TestWriterProducesFrozenSchema asserts the Writer emits exactly the frozen v1
// wire format for a representative container mapping.
func TestWriterProducesFrozenSchema(t *testing.T) {
	dir := t.TempDir()
	w := NewWriter(dir, nil)

	w.Upsert(store.ContainerInfo{
		ContainerID:        "abc",
		Container:          "trainer", // intentionally not persisted
		CgroupPath:         "/not/persisted",
		Pod:                "trainer-7d9f",
		Namespace:          "team-ml",
		PodUID:             "3f2a-uid",
		GPUDevices:         []store.GPUDevice{{Index: 0}, {Index: 3}},
		RequestedMemoryMiB: 2048,
	})

	data, err := os.ReadFile(filepath.Join(dir, "abc.json"))
	if err != nil {
		t.Fatalf("read mapping file: %v", err)
	}
	if got := string(data); got != goldenRecordV1 {
		t.Fatalf("writer schema drifted from the frozen v1 contract\n got: %s\nwant: %s", got, goldenRecordV1)
	}
}
