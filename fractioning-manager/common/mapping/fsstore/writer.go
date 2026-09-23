// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fsstore

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

const (
	dirPerm  os.FileMode = 0o755 // rwxr-xr-x
	filePerm os.FileMode = 0o644 // rw-r--r-- world-readable so the metrics container can read mapper-written files

	// tempFilePattern is the os.CreateTemp pattern for the write-then-rename
	// staging file. Dot-prefixed and no .json suffix so in-flight writes are
	// never picked up by the reader's directory scan.
	tempFilePattern = ".gpu-fractioning-*.tmp"
)

// Writer persists the container->pod mapping as one <containerID>.json file per
// container under a shared directory. It is the write side
// of the NRI mapper component; the metrics component's Reader consumes the same
// files read-only. It satisfies store.Writer, so it plugs directly into the
// events processor.
//
// Each write is atomic — a temp file in the same directory followed by a rename —
// so a concurrent Reader never observes a partial or truncated file. The events
// processor drives Upsert/Delete/Replace on a single goroutine, so writes are
// already serialized; the type is nonetheless safe for concurrent use.
type Writer struct {
	dir string
	log *slog.Logger
}

var _ store.Writer = (*Writer)(nil)

// NewWriter returns a Writer that persists mapping files under dir. The directory
// is created lazily on the first write, so a not-yet-mounted volume is not an
// error at construction time.
func NewWriter(dir string, logger *slog.Logger) *Writer {
	if logger == nil {
		logger = slog.Default()
	}
	return &Writer{dir: dir, log: logger}
}

// Upsert writes (or overwrites) the mapping file for a single container. Failures
// are logged rather than returned: the events processor is fire-and-forget, and
// the file is rewritten on the next NRI Synchronize.
func (w *Writer) Upsert(info store.ContainerInfo) {
	if info.ContainerID == "" {
		return
	}
	if err := w.writeRecord(info); err != nil {
		w.log.Warn("failed to write container mapping file", "containerID", info.ContainerID, "error", err)
	}
}

// Delete unlinks the mapping file for a single container. A missing file is not
// an error: a Replace reconcile may have already removed it.
func (w *Writer) Delete(containerID string) {
	if containerID == "" {
		return
	}
	path := filepath.Join(w.dir, fileName(containerID))
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		w.log.Warn("failed to remove container mapping file", "containerID", containerID, "error", err)
	}
}

// Replace reconciles the directory against the full container set: it
// (re)writes every current file and unlinks any stale file for a container no
// longer present. Callers are responsible for not invoking Replace with an
// empty slice when the intent is to preserve existing files (e.g. on an NRI
// reconnect before containerd has replayed containers); see events.Processor.
func (w *Writer) Replace(containers []store.ContainerInfo) {
	if err := os.MkdirAll(w.dir, dirPerm); err != nil {
		w.log.Warn("failed to create mapping directory", "dir", w.dir, "error", err)
		return
	}

	w.log.Debug("NRI Synchronize: reconciling fsstore", "containers", len(containers))

	desired := make(map[string]struct{}, len(containers))
	for _, info := range containers {
		if info.ContainerID == "" {
			continue
		}
		desired[fileName(info.ContainerID)] = struct{}{}
		if err := w.writeRecord(info); err != nil {
			w.log.Warn("failed to write container mapping file during sync", "containerID", info.ContainerID, "error", err)
		}
	}

	entries, err := os.ReadDir(w.dir)
	if err != nil {
		w.log.Warn("failed to list mapping directory during sync", "dir", w.dir, "error", err)
		return
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), fileExtension) {
			continue
		}
		if _, keep := desired[entry.Name()]; keep {
			continue
		}
		if err := os.Remove(filepath.Join(w.dir, entry.Name())); err != nil && !os.IsNotExist(err) {
			w.log.Warn("failed to remove stale mapping file during sync", "file", entry.Name(), "error", err)
		}
	}
}

// writeRecord serializes one container record and writes it atomically: a temp
// file in the same directory, fsync, then rename onto the final name. The rename
// is what readers rely on — they see either the old file or the new one, never a
// half-written file.
func (w *Writer) writeRecord(info store.ContainerInfo) (err error) {
	if mkErr := os.MkdirAll(w.dir, dirPerm); mkErr != nil {
		return fmt.Errorf("create mapping dir %q: %w", w.dir, mkErr)
	}

	data, err := json.Marshal(newRecord(info))
	if err != nil {
		return fmt.Errorf("marshal record: %w", err)
	}

	tmp, err := os.CreateTemp(w.dir, tempFilePattern)
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tmpName := tmp.Name()
	// Clean up the staging file unless the rename below has consumed it.
	defer func() {
		if err != nil {
			_ = os.Remove(tmpName)
		}
	}()

	if _, err = tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write temp file: %w", err)
	}
	if err = tmp.Chmod(filePerm); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("chmod temp file: %w", err)
	}
	if err = tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("sync temp file: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("close temp file: %w", err)
	}

	if err = os.Rename(tmpName, filepath.Join(w.dir, fileName(info.ContainerID))); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}
	return nil
}
