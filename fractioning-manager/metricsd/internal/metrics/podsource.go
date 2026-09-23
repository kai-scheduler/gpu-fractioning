// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"log/slog"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

// podSource resolves GPU processes to the Kubernetes pod that owns them and
// enumerates the pods currently using GPUs on this node.
type podSource interface {
	// Snapshot fetches the current container mapping once and returns a
	// podSource backed by that in-memory snapshot. The controller calls this
	// once per collect cycle so all subsequent queries hit memory, not the
	// filesystem.
	Snapshot() podSource
	// ResolveProcess maps a single GPU process to its owning pod/container.
	ResolveProcess(process GPUProcessMetric) (store.ContainerInfo, bool)
	// ActivePodUIDs returns the identity of every pod currently known to use a
	// GPU, used to prune series for pods that have gone away.
	ActivePodUIDs() map[string]struct{}
	// ActiveContainers returns every GPU-using container/pod, used both to skip
	// collection when nothing uses a GPU and to publish idle (zero) series.
	ActiveContainers() []store.ContainerInfo
}

// cgroupPodSource resolves a GPU process to its pod by reading the PID's cgroup
// membership and matching it against the NRI container store.
type cgroupPodSource struct {
	resolver PIDCgroupResolver
	reader   store.Reader
	log      *slog.Logger
}

func newCgroupPodSource(resolver PIDCgroupResolver, reader store.Reader, logger *slog.Logger) *cgroupPodSource {
	if logger == nil {
		logger = slog.Default()
	}
	return &cgroupPodSource{resolver: resolver, reader: reader, log: logger}
}

// Snapshot fetches all containers from the backing reader once and returns a
// podSourceSnapshot backed by an in-memory ContainerStore. All query methods on
// the snapshot hit memory only — no further filesystem access occurs during the
// collect cycle.
func (c *cgroupPodSource) Snapshot() podSource {
	cs := store.NewContainerStore()
	cs.Replace(c.reader.ActiveContainers())
	return &podSourceSnapshot{resolver: c.resolver, store: cs, log: c.log}
}

func (c *cgroupPodSource) ResolveProcess(process GPUProcessMetric) (store.ContainerInfo, bool) {
	paths, err := c.resolver.CgroupPaths(process.PID)
	if err != nil {
		c.log.Debug("failed to resolve cgroup paths for GPU process", "pid", process.PID, "error", err)
		return store.ContainerInfo{}, false
	}
	return c.reader.MatchCgroupPaths(paths)
}

func (c *cgroupPodSource) ActivePodUIDs() map[string]struct{} {
	return c.reader.ActivePodUIDs()
}

func (c *cgroupPodSource) ActiveContainers() []store.ContainerInfo {
	return c.reader.ActiveContainers()
}

// podSourceSnapshot is the point-in-time view returned by cgroupPodSource.Snapshot.
// Its backing ContainerStore is populated once and never mutated, so all methods
// are read-only and require no further I/O.
type podSourceSnapshot struct {
	resolver PIDCgroupResolver
	store    *store.ContainerStore
	log      *slog.Logger
}

func (s *podSourceSnapshot) Snapshot() podSource { return s }

func (s *podSourceSnapshot) ResolveProcess(process GPUProcessMetric) (store.ContainerInfo, bool) {
	paths, err := s.resolver.CgroupPaths(process.PID)
	if err != nil {
		s.log.Debug("failed to resolve cgroup paths for GPU process", "pid", process.PID, "error", err)
		return store.ContainerInfo{}, false
	}
	return s.store.MatchCgroupPaths(paths)
}

func (s *podSourceSnapshot) ActivePodUIDs() map[string]struct{} {
	return s.store.ActivePodUIDs()
}

func (s *podSourceSnapshot) ActiveContainers() []store.ContainerInfo {
	return s.store.ActiveContainers()
}
