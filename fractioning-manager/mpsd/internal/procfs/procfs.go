// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package procfs resolves host PIDs from /proc. mpsd runs in the host PID
// namespace (it has to: the MPS control daemon identifies clients by their host
// PID), so the same /proc that gives it those PIDs can also tell it which
// container each one belongs to.
package procfs

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DefaultRoot is the procfs mount point.
const DefaultRoot = "/proc"

// PIDsInContainer returns the PIDs whose cgroup membership names containerID.
//
// The match is a substring test against /proc/<pid>/cgroup rather than a parse
// of it, because the container id appears in a different shape under every
// runtime and cgroup driver — "cri-containerd-<id>.scope" under systemd,
// ".../<id>" under cgroupfs, "crio-<id>.scope" under CRI-O. A container id is a
// 64-character hex digest, so a substring hit is not something that happens by
// accident.
//
// Callers must pass a full container id. A short or empty id would match too
// much, so an id shorter than minContainerIDLength returns no PIDs rather than
// a broad match.
func PIDsInContainer(root, containerID string) ([]int, error) {
	if len(containerID) < minContainerIDLength {
		return nil, nil
	}
	if root == "" {
		root = DefaultRoot
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}

	var pids []int
	for _, entry := range entries {
		pid, ok := pidFromDirEntry(entry.Name())
		if !ok {
			continue
		}
		// A process that exits between the readdir and this read is normal and
		// not an error — it simply is not a client to drain.
		content, err := os.ReadFile(filepath.Join(root, entry.Name(), "cgroup"))
		if err != nil {
			continue
		}
		if strings.Contains(string(content), containerID) {
			pids = append(pids, pid)
		}
	}

	return pids, nil
}

// PIDsByExecutable returns the PIDs whose executable file is named name (a bare
// file name, compared against the base of the path).
//
// It deliberately does NOT use /proc/<pid>/comm, which is the obvious choice and
// the wrong one here. The kernel stores comm in a 16-byte field, so it is
// truncated to 15 characters: "nvidia-cuda-mps-server" reads back as
// "nvidia-cuda-mps", which matches nothing when queried by its real name — and
// worse, "nvidia-cuda-mps-control" truncates to those same 15 characters, so a
// comm query cannot tell the servers apart from the control daemon that spawned
// them. Killing MPS servers is a recovery step for a wedged GPU; silently
// killing none of them, or killing the control daemon by accident, both defeat
// it.
//
// The exe symlink is authoritative and is tried first. cmdline is the fallback
// for a process whose exe cannot be read; it is argv[0], which a process can
// rewrite, but the alternative is missing the process entirely.
func PIDsByExecutable(root, name string) ([]int, error) {
	if name == "" {
		return nil, nil
	}
	if root == "" {
		root = DefaultRoot
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}

	var pids []int
	for _, entry := range entries {
		pid, ok := pidFromDirEntry(entry.Name())
		if !ok {
			continue
		}
		if executableName(filepath.Join(root, entry.Name())) == name {
			pids = append(pids, pid)
		}
	}

	return pids, nil
}

// executableName returns the base name of the process's executable, or "" when
// neither the exe symlink nor cmdline yields one.
func executableName(procDir string) string {
	if target, err := os.Readlink(filepath.Join(procDir, "exe")); err == nil {
		// A binary replaced or removed since exec reads back with a " (deleted)"
		// suffix the kernel appends to the link target, not part of the path.
		return filepath.Base(strings.TrimSuffix(target, deletedSuffix))
	}

	content, err := os.ReadFile(filepath.Join(procDir, "cmdline"))
	if err != nil {
		return ""
	}
	// cmdline is NUL-separated; argv[0] is everything up to the first NUL.
	argv0, _, _ := strings.Cut(string(content), "\x00")
	if argv0 == "" {
		return ""
	}
	return filepath.Base(argv0)
}

// deletedSuffix is what the kernel appends to /proc/<pid>/exe when the
// executable has been unlinked since the process started.
const deletedSuffix = " (deleted)"

// minContainerIDLength is the shortest id PIDsInContainer will match on. Full
// container ids are 64 hex characters; this leaves room for a runtime that
// abbreviates while still being far too specific to collide.
const minContainerIDLength = 12

func pidFromDirEntry(name string) (int, bool) {
	pid, err := strconv.Atoi(name)
	if err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}
