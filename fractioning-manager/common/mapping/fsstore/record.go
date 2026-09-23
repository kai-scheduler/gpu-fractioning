// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package fsstore is the filesystem handoff for the container->pod mapping
// The NRI mapper component writes one <containerID>.json file per container under a shared directory; the metrics
// component reads those files read-only. The two sides live in separate
// containers (and may be separate processes), so the mapping crosses a process
// boundary and is durable rather than in-memory.
//
// Writer and Reader both satisfy the store.Writer / store.Reader interfaces, so
// they drop into the existing event pipeline and metrics controller in place of the
// in-memory store.ContainerStore.
package fsstore

import (
	"strings"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

// DefaultMapDir is the shared-volume directory for the container->pod mapping
// handoff: the fractiond NRI plugin writes one <containerID>.json file per
// container here and the metricsd sidecar reads them. It is the single source of
// truth for the path both sides must agree on.
const DefaultMapDir = "/var/run/gpu-fractioning/map"

// schemaVersion is the on-disk record version. Bump it when the
// record shape changes incompatibly; readers ignore files whose version they do
// not recognize rather than misinterpreting them. The project is pre-release, so
// the schema is still at its initial version.
const schemaVersion = 1

// fileExtension is the suffix of every mapping file: <containerID>.json.
const fileExtension = ".json"

// record is the on-disk container->pod mapping: pod identity plus assigned GPU
// devices, nothing else. The container ID is the file name, not a field.
type record struct {
	SchemaVersion int               `json:"schemaVersion"`
	Pod           string            `json:"pod"`
	Namespace     string            `json:"namespace"`
	PodUID        string            `json:"podUID"`
	GPUDevices    []store.GPUDevice `json:"gpuDevices"`
	// RequestedMemoryMiB is the container's allocated GPU memory in MiB, used by
	// the metrics sidecar to derive the GPU fraction for SM-util normalization.
	RequestedMemoryMiB int64 `json:"requestedMemoryMiB,omitempty"`
}

// newRecord projects a store.ContainerInfo onto the minimal on-disk schema. The
// container ID, container name and cgroup path are intentionally not persisted:
// the ID is the file name, and the metrics component matches by container ID
// extracted from /proc/<pid>/cgroup.
func newRecord(info store.ContainerInfo) record {
	return record{
		SchemaVersion:      schemaVersion,
		Pod:                info.Pod,
		Namespace:          info.Namespace,
		PodUID:             info.PodUID,
		GPUDevices:         info.GPUDevices,
		RequestedMemoryMiB: info.RequestedMemoryMiB,
	}
}

// toContainer rebuilds a store.ContainerInfo from a decoded record and the
// container ID recovered from the file name. The cgroup path is left empty: the
// reader matches processes by container-ID inclusion in the cgroup path, which
// does not need the mapper's cgroup string.
func (r record) toContainer(containerID string) store.ContainerInfo {
	return store.ContainerInfo{
		ContainerID:        containerID,
		Pod:                r.Pod,
		Namespace:          r.Namespace,
		PodUID:             r.PodUID,
		GPUDevices:         r.GPUDevices,
		RequestedMemoryMiB: r.RequestedMemoryMiB,
	}
}

// fileName maps a container ID to its mapping file name. The ID is the file's
// stem; sanitizeContainerID keeps it a single safe path component.
func fileName(containerID string) string {
	return sanitizeContainerID(containerID) + fileExtension
}

// containerIDFromFile recovers the container ID from a mapping file name. It is
// the inverse of fileName for the sanitized form; runtime container IDs are bare
// hex, so sanitization is a no-op in practice and the round trip is exact.
func containerIDFromFile(name string) string {
	return strings.TrimSuffix(name, fileExtension)
}

// sanitizeContainerID reduces a container ID to a single safe file-name
// component: it strips any "scheme://" prefix and replaces path separators. NRI
// reports bare IDs, so this is defensive normalization rather than a transform
// that runs in practice.
func sanitizeContainerID(containerID string) string {
	containerID = strings.TrimSpace(containerID)
	if idx := strings.LastIndex(containerID, "://"); idx >= 0 {
		containerID = containerID[idx+3:]
	}
	containerID = strings.ReplaceAll(containerID, "/", "_")
	containerID = strings.ReplaceAll(containerID, "\\", "_")
	return containerID
}
