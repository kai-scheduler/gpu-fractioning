// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nvmlshim stages the shadow NVML library that fractiond bind-mounts
// into GPU-fractioning containers.
//
// The library itself ships inside the fractiond image; the containers that need
// it are created by the runtime on the host and cannot read anything out of
// another container's filesystem. So the file has to be copied out onto a host
// directory first (daemonpaths.NVMLShimDir, hostPath-mounted into the fractiond
// pod by the operator), and that is all this package does.
//
// Nothing here understands what the library does — see the injection package
// for the env and mount contract that actually puts it in front of the real
// NVML inside a container.
package nvmlshim

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

const (
	// LibraryName is the file name the shim must be staged under. It is the
	// driver's own SONAME: the loader searches LD_LIBRARY_PATH for exactly this
	// string, so a copy staged under any other name is never consulted, however
	// correct its contents.
	LibraryName = "libnvidia-ml.so.1"

	// DefaultSourcePath is where the fractiond image build places the library.
	DefaultSourcePath = "/usr/lib/gpu-fractioning/" + LibraryName
)

// Stage copies the shim from sourcePath into destDir and returns the staged
// path. It is safe to call on every process start: an already-staged file with
// identical content is left untouched.
//
// Callers treat an error as "run without the shim", never as a startup failure
// — the GPU caps are enforced by MPS regardless, and only the in-container
// reporting view degrades. Hence the error is returned rather than logged
// here: the decision of what a failure means belongs to the caller.
func Stage(ctx context.Context, sourcePath, destDir string, log *slog.Logger) (string, error) {
	if log == nil {
		log = slog.Default()
	}
	if sourcePath == "" {
		return "", errors.New("no NVML shim source path configured")
	}
	if destDir == "" {
		return "", errors.New("no NVML shim destination directory configured")
	}

	source, err := os.ReadFile(sourcePath)
	if err != nil {
		return "", fmt.Errorf("reading NVML shim from the image: %w", err)
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", fmt.Errorf("creating NVML shim directory %q: %w", destDir, err)
	}

	destPath := filepath.Join(destDir, LibraryName)
	if sameContent(destPath, source) {
		log.InfoContext(ctx, "NVML shim already staged and up to date", "path", destPath)
		return destPath, nil
	}

	if err := writeAtomically(destPath, source); err != nil {
		return "", err
	}

	log.InfoContext(ctx, "staged NVML shim onto the host",
		"source", sourcePath, "path", destPath, "bytes", len(source))
	return destPath, nil
}

// writeAtomically writes content to destPath through a temp file in the same
// directory followed by a rename.
//
// The rename is what makes this safe to do while the node is running. Every
// GPU-fractioning container bind-mounts the directory this file lives in, and a
// container created mid-copy would map whatever bytes happened to be on disk at
// that instant — a truncated .so that the loader either rejects or, worse,
// accepts partially. Writing in place has no way to avoid that window; a rename
// within the same filesystem is atomic, so a container sees either the previous
// library or the new one and never a half of either.
//
// Same directory, not the system temp dir, because rename is only atomic within
// one filesystem and destDir is a hostPath mount that is generally not on the
// same one as /tmp.
func writeAtomically(destPath string, content []byte) error {
	dir := filepath.Dir(destPath)

	tmp, err := os.CreateTemp(dir, "."+filepath.Base(destPath)+".*")
	if err != nil {
		return fmt.Errorf("creating temp file for NVML shim in %q: %w", dir, err)
	}
	tmpPath := tmp.Name()
	// Best-effort cleanup of the temp file on every failure path below; after a
	// successful rename the path no longer exists and the remove is a no-op.
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()

	if _, err := tmp.Write(content); err != nil {
		return fmt.Errorf("writing NVML shim to %q: %w", tmpPath, err)
	}
	// The library is executed by the loader in containers that mount it, so it
	// needs the execute bit; CreateTemp makes the file 0600.
	if err := tmp.Chmod(0o755); err != nil {
		return fmt.Errorf("setting NVML shim permissions on %q: %w", tmpPath, err)
	}
	// Flush before the rename: the rename only guarantees that the name points
	// at the new inode, not that the new inode's data reached the disk, so a
	// node that loses power right after would expose a zero-length library.
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("syncing NVML shim to %q: %w", tmpPath, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("closing NVML shim temp file %q: %w", tmpPath, err)
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("staging NVML shim at %q: %w", destPath, err)
	}
	return nil
}

// sameContent reports whether destPath already holds exactly content, so a
// restart that changes nothing does not replace a library containers may be
// mapping right now. Compared by digest rather than by size or mtime: an image
// rebuild can easily produce a different library of the same length, and the
// copy's mtime says nothing about which build it came from.
func sameContent(destPath string, content []byte) bool {
	existing, err := os.Open(destPath)
	if err != nil {
		return false
	}
	defer func() { _ = existing.Close() }()

	digest := sha256.New()
	if _, err := io.Copy(digest, existing); err != nil {
		return false
	}
	want := sha256.Sum256(content)
	return string(digest.Sum(nil)) == string(want[:])
}
