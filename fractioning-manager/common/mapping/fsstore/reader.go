// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fsstore

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

// Reader is the read side of the filesystem handoff: it loads the
// <containerID>.json files the Writer persists and answers the metrics
// component's container->pod queries. It satisfies store.Reader.
//
// Each query method scans the mapping directory on demand. Because a container
// removal also terminates its GPU processes, the directory is always consistent
// with the active process set — no explicit reload step is needed.
type Reader struct {
	dir string
	log *slog.Logger
}

var (
	_ store.Reader = (*Reader)(nil)
)

// NewReader returns a Reader over the mapping directory dir. The directory is
// not required to exist yet; a missing directory is treated as an empty mapping.
func NewReader(dir string, logger *slog.Logger) *Reader {
	if logger == nil {
		logger = slog.Default()
	}
	return &Reader{dir: dir, log: logger}
}

// MatchCgroupPaths resolves a GPU process's cgroup paths to a container.
func (r *Reader) MatchCgroupPaths(paths []string) (store.ContainerInfo, bool) {
	return r.scan(context.Background()).MatchCgroupPaths(paths)
}

// ActivePodUIDs returns every pod UID currently present in the mapping.
func (r *Reader) ActivePodUIDs() map[string]struct{} {
	return r.scan(context.Background()).ActivePodUIDs()
}

// ActiveContainers returns every container currently present in the mapping.
func (r *Reader) ActiveContainers() []store.ContainerInfo {
	return r.scan(context.Background()).ActiveContainers()
}

// scan reads the mapping directory and returns a fresh ContainerStore populated
// with the current set of container records. A missing directory is treated as
// an empty mapping; a single unreadable or unparseable file is skipped with a
// warning so one bad file never blanks the whole mapping.
func (r *Reader) scan(ctx context.Context) *store.ContainerStore {
	entries, err := os.ReadDir(r.dir)
	if err != nil {
		if !os.IsNotExist(err) {
			r.log.WarnContext(ctx, "failed to read mapping directory", "dir", r.dir, "error", err)
		}
		return store.NewContainerStore()
	}

	infos := make([]store.ContainerInfo, 0, len(entries))
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || strings.HasPrefix(name, ".") || !strings.HasSuffix(name, fileExtension) {
			continue
		}
		if info, ok := r.readFile(ctx, name); ok {
			infos = append(infos, info)
		}
	}
	s := store.NewContainerStore()
	s.Replace(infos)
	return s
}

func (r *Reader) readFile(ctx context.Context, name string) (store.ContainerInfo, bool) {
	data, err := os.ReadFile(filepath.Join(r.dir, name))
	if err != nil {
		if !os.IsNotExist(err) {
			r.log.WarnContext(ctx, "failed to read container mapping file", "file", name, "error", err)
		}
		return store.ContainerInfo{}, false
	}

	var rec record
	if err := json.Unmarshal(data, &rec); err != nil {
		r.log.WarnContext(ctx, "failed to parse container mapping file; skipping", "file", name, "error", err)
		return store.ContainerInfo{}, false
	}
	if rec.SchemaVersion != schemaVersion {
		r.log.WarnContext(ctx, "ignoring container mapping file with unsupported schema version",
			"file", name, "schemaVersion", rec.SchemaVersion, "supported", schemaVersion)
		return store.ContainerInfo{}, false
	}
	info := rec.toContainer(containerIDFromFile(name))
	r.log.DebugContext(ctx, "loaded container mapping",
		"containerID", info.ContainerID,
		"pod", info.Namespace+"/"+info.Pod,
		"podUID", info.PodUID,
		"gpuDevices", info.GPUDevices,
		"file", name,
	)
	return info, true
}
