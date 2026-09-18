// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nvmlshim

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

// writeSource puts a stand-in library in a temp dir and returns its path. The
// contents are opaque to this package, so any bytes will do.
func writeSource(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "image-"+LibraryName)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing test source: %v", err)
	}
	return path
}

func TestStage(t *testing.T) {
	tests := []struct {
		name string
		// source is the library content to stage; "" means "no source file at
		// all", which is how a fractiond image built without the library looks.
		source string
		// existing, when non-nil, is pre-staged at the destination.
		existing []byte
		wantErr  bool
		want     string
	}{
		{
			name:   "stages onto an empty directory",
			source: "shim-v1",
			want:   "shim-v1",
		},
		{
			name:     "identical content is left alone",
			source:   "shim-v1",
			existing: []byte("shim-v1"),
			want:     "shim-v1",
		},
		{
			// An image upgrade is the whole reason staging re-runs, so a stale
			// copy has to lose. Same length as the new one, since a size
			// comparison is the tempting shortcut that would miss this.
			name:     "different content of the same length is replaced",
			source:   "shim-v2",
			existing: []byte("shim-v1"),
			want:     "shim-v2",
		},
		{
			name:     "shorter new content fully replaces a longer stale copy",
			source:   "v2",
			existing: []byte("a much longer previous library"),
			want:     "v2",
		},
		{
			name:    "a missing source is an error, not an empty library",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			destDir := filepath.Join(t.TempDir(), "lib")
			if tt.existing != nil {
				if err := os.MkdirAll(destDir, 0o755); err != nil {
					t.Fatalf("creating dest dir: %v", err)
				}
				if err := os.WriteFile(filepath.Join(destDir, LibraryName), tt.existing, 0o755); err != nil {
					t.Fatalf("pre-staging: %v", err)
				}
			}

			sourcePath := filepath.Join(t.TempDir(), "absent")
			if tt.source != "" {
				sourcePath = writeSource(t, tt.source)
			}

			staged, err := Stage(context.Background(), sourcePath, destDir, testLogger())
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got a staged path %q", staged)
				}
				return
			}
			if err != nil {
				t.Fatalf("Stage: %v", err)
			}

			wantPath := filepath.Join(destDir, LibraryName)
			if staged != wantPath {
				t.Errorf("staged path = %q, want %q", staged, wantPath)
			}
			got, err := os.ReadFile(wantPath)
			if err != nil {
				t.Fatalf("reading staged library: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("staged content = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestStageLeavesNoPartialFiles is the reason the copy goes through a rename.
// Every GPU-fractioning container bind-mounts the staging directory, so a
// container created while a copy is in flight maps whatever is in there at that
// instant. Only the final name may ever be visible, and no temp file may be
// left behind for a container to pick up either.
func TestStageLeavesNoPartialFiles(t *testing.T) {
	destDir := filepath.Join(t.TempDir(), "lib")

	for _, content := range []string{"shim-v1", "shim-v2", "shim-v2"} {
		if _, err := Stage(context.Background(), writeSource(t, content), destDir, testLogger()); err != nil {
			t.Fatalf("Stage(%q): %v", content, err)
		}

		entries, err := os.ReadDir(destDir)
		if err != nil {
			t.Fatalf("reading dest dir: %v", err)
		}
		if len(entries) != 1 || entries[0].Name() != LibraryName {
			names := make([]string, 0, len(entries))
			for _, e := range entries {
				names = append(names, e.Name())
			}
			t.Fatalf("staging directory holds %v, want only %q", names, LibraryName)
		}
	}
}

// TestStageIsIdempotentAcrossRestarts covers the restart path specifically: an
// unchanged image must not touch the library at all, because replacing it is a
// rename that swaps the inode under containers currently mapping it.
func TestStageIsIdempotentAcrossRestarts(t *testing.T) {
	destDir := filepath.Join(t.TempDir(), "lib")
	sourcePath := writeSource(t, "shim-v1")

	if _, err := Stage(context.Background(), sourcePath, destDir, testLogger()); err != nil {
		t.Fatalf("first Stage: %v", err)
	}
	staged := filepath.Join(destDir, LibraryName)
	first, err := os.Stat(staged)
	if err != nil {
		t.Fatalf("stat after first Stage: %v", err)
	}

	if _, err := Stage(context.Background(), sourcePath, destDir, testLogger()); err != nil {
		t.Fatalf("second Stage: %v", err)
	}
	second, err := os.Stat(staged)
	if err != nil {
		t.Fatalf("stat after second Stage: %v", err)
	}

	if !first.ModTime().Equal(second.ModTime()) {
		t.Errorf("an unchanged library was rewritten (mtime %v -> %v); the rename would swap the inode under running containers",
			first.ModTime(), second.ModTime())
	}
}

// TestStageMakesTheLibraryExecutable guards the mode, not the bytes.
// os.CreateTemp makes a 0600 file, and a library the loader cannot read in the
// container is as good as no library at all.
func TestStageMakesTheLibraryExecutable(t *testing.T) {
	destDir := filepath.Join(t.TempDir(), "lib")
	if _, err := Stage(context.Background(), writeSource(t, "shim-v1"), destDir, testLogger()); err != nil {
		t.Fatalf("Stage: %v", err)
	}

	info, err := os.Stat(filepath.Join(destDir, LibraryName))
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if got := info.Mode().Perm(); got != 0o755 {
		t.Errorf("staged library mode = %04o, want %04o", got, 0o755)
	}
}

// TestStageRejectsUnusableConfiguration pins the failures the caller turns into
// "run without the shim". Each has to be an error rather than a silent success,
// because a silent success would leave fractiond mounting a directory with no
// library in it into every GPU container.
func TestStageRejectsUnusableConfiguration(t *testing.T) {
	tests := []struct {
		name       string
		sourcePath string
		destDir    string
	}{
		{name: "no source configured", sourcePath: "", destDir: t.TempDir()},
		{name: "no destination configured", sourcePath: writeSource(t, "shim"), destDir: ""},
		{name: "source is a directory", sourcePath: t.TempDir(), destDir: t.TempDir()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := Stage(context.Background(), tt.sourcePath, tt.destDir, testLogger()); err == nil {
				t.Error("expected an error, got nil")
			}
		})
	}
}

// TestStageFailsWhenTheDestinationIsNotWritable is the production failure this
// is most likely to hit: the operator did not mount the host directory, or
// mounted it read-only. It must surface as an error so the caller disables the
// feature rather than advertising a library that is not there.
func TestStageFailsWhenTheDestinationIsNotWritable(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root, which bypasses the directory permissions this checks")
	}

	parent := t.TempDir()
	if err := os.Chmod(parent, 0o500); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(parent, 0o700) })

	destDir := filepath.Join(parent, "lib")
	if _, err := Stage(context.Background(), writeSource(t, "shim"), destDir, testLogger()); err == nil {
		t.Error("expected staging into an unwritable directory to fail, got nil")
	}
}

// TestDefaultSourcePathUsesTheLoaderName keeps the in-image path and the staged
// name from drifting apart. The loader only ever looks for the SONAME, so a
// default source pointing at some other file name would stage a library nothing
// opens.
func TestDefaultSourcePathUsesTheLoaderName(t *testing.T) {
	if LibraryName != "libnvidia-ml.so.1" {
		t.Errorf("LibraryName = %q, want %q: the loader searches for the driver's SONAME and nothing else", LibraryName, "libnvidia-ml.so.1")
	}
	if got := filepath.Base(DefaultSourcePath); got != LibraryName {
		t.Errorf("DefaultSourcePath base = %q, want %q", got, LibraryName)
	}
}
