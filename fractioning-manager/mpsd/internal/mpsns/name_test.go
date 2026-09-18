// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsns

import (
	"fmt"
	"strings"
	"testing"
)

// TestNameIsAlwaysAcceptableToTheDaemon: the control daemon accepts only
// [a-z0-9_]+ as a namespace name, and rejects anything else with a message that
// says nothing about which input produced it. Every shape a container id can
// arrive in therefore has to come out of Name acceptable, including the ones
// Kubernetes produces by default — hyphens are everywhere in pod and container
// names, and a runtime is free to hand over an id in any casing.
func TestNameIsAlwaysAcceptableToTheDaemon(t *testing.T) {
	ids := []string{
		"a",
		"0",
		"abc123",
		"ABCDEF0123456789",
		"a1b2c3d4e5f60718293a4b5c6d7e8f901a2b3c4d5e6f708192a3b4c5d6e7f809",
		"cri-containerd-9f8e7d6c5b4a.scope",
		"my-pod-name-with-hyphens",
		"UPPER-and-lower_Mixed.123",
		"contains spaces and\ttabs",
		"unicode-ünïcödé-id",
		"/slashes/and:colons",
		strings.Repeat("x", 300),
		strings.Repeat("-", 64),
	}

	for _, id := range ids {
		name, err := Name(id)
		if err != nil {
			t.Fatalf("Name(%q) error = %v", id, err)
		}
		for _, r := range name {
			switch {
			case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_':
			default:
				t.Fatalf("Name(%q) = %q, which contains %q — the daemon accepts only [a-z0-9_]", id, name, r)
			}
		}
		if !IsManagedName(name) {
			t.Errorf("Name(%q) = %q, which IsManagedName does not recognise; reconciliation would ignore it", id, name)
		}
	}
}

// TestNameIsCollisionFree is the property that keeps two containers from
// sharing one compute cap.
//
// Sanitization and truncation are both lossy, so ids that differ only in the
// characters they lose — a hyphen where another has an underscore, a difference
// past the truncation point, a difference in case — map to the same readable
// half. Uniqueness has to come from the digest of the whole id, and this checks
// exactly the collisions the lossy half would produce. Two containers on one
// name would mean the second provision silently adopting the first's namespace,
// and the first release deleting the namespace the second is still using.
func TestNameIsCollisionFree(t *testing.T) {
	ids := []string{
		"abc-def",
		"abc_def",
		"ABC-DEF",
		"abc.def",
		"abc def",
		// Differ only past the readable half's truncation point.
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaa1",
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaa2",
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaa3",
		// Real container ids differing in the last nibble only.
		"a1b2c3d4e5f60718293a4b5c6d7e8f901a2b3c4d5e6f708192a3b4c5d6e7f800",
		"a1b2c3d4e5f60718293a4b5c6d7e8f901a2b3c4d5e6f708192a3b4c5d6e7f801",
		// One id being a prefix of another must not collide either.
		"abc",
		"abcd",
	}

	seen := map[string]string{}
	for _, id := range ids {
		name, err := Name(id)
		if err != nil {
			t.Fatalf("Name(%q) error = %v", id, err)
		}
		if other, clash := seen[name]; clash {
			t.Fatalf("container ids %q and %q both map to namespace %q: they would share one compute cap", other, id, name)
		}
		seen[name] = id
	}
}

// TestNameIsStable: the name is how a container is found again — by a retried
// NRI hook, by a release, and by the reconciliation that recreates namespaces
// after MPS restarts. A name that varied between calls would make every one of
// those create a second namespace and abandon the first.
func TestNameIsStable(t *testing.T) {
	const id = "a1b2c3d4e5f60718293a4b5c6d7e8f901a2b3c4d5e6f708192a3b4c5d6e7f809"

	first, err := Name(id)
	if err != nil {
		t.Fatalf("Name() error = %v", err)
	}
	for i := 0; i < 100; i++ {
		again, err := Name(id)
		if err != nil {
			t.Fatalf("Name() error = %v", err)
		}
		if again != first {
			t.Fatalf("Name(%q) = %q then %q; the name must not vary", id, first, again)
		}
	}
}

// TestNameStaysShortEnoughForASocketPath: the name becomes a path component of
// a unix socket path, and overrunning sun_path makes the control daemon fail to
// initialise with no log at all. A 64-character container id must not produce a
// 64-character name.
func TestNameStaysShortEnoughForASocketPath(t *testing.T) {
	const maxName = 48

	for _, id := range []string{
		"a1b2c3d4e5f60718293a4b5c6d7e8f901a2b3c4d5e6f708192a3b4c5d6e7f809",
		strings.Repeat("x", 1000),
	} {
		name, err := Name(id)
		if err != nil {
			t.Fatalf("Name() error = %v", err)
		}
		if len(name) > maxName {
			t.Errorf("Name(%d-character id) = %q (%d characters), want at most %d", len(id), name, len(name), maxName)
		}
	}
}

// TestNameRejectsAnEmptyID: an empty id has no identity to preserve, so a name
// derived from it would be the same name for every such call — the collision
// case, arrived at from the other direction.
func TestNameRejectsAnEmptyID(t *testing.T) {
	for _, id := range []string{"", "   ", "\t\n"} {
		if name, err := Name(id); err == nil {
			t.Errorf("Name(%q) = %q, nil; want an error", id, name)
		}
	}
}

// TestIsManagedName decides what reconciliation is allowed to delete. It has to
// be exactly the names this package generates: too wide and a sweep deletes a
// namespace somebody else created, too narrow and orphans accumulate forever.
func TestIsManagedName(t *testing.T) {
	managed, err := Name("container-id")
	if err != nil {
		t.Fatal(err)
	}

	if !IsManagedName(managed) {
		t.Errorf("IsManagedName(%q) = false, want true", managed)
	}
	for _, name := range []string{
		"default",       // the daemon's own namespace
		"shared",        // a server name that might appear in list output
		"",              // a parse artefact
		NamePrefix,      // the prefix alone names nothing
		"kaiabc",        // prefix-like but not prefixed
		"other_kai_abc", // the prefix must lead
		"KAI_abc",       // not something Name can produce
		"kai_abc-def",   // ditto
		"name",          // a header column
		"active_thread_percentage",
	} {
		if IsManagedName(name) {
			t.Errorf("IsManagedName(%q) = true; reconciliation would delete a namespace it does not own", name)
		}
	}
}

// TestCheckPipeDirLength is the guard for the failure with no diagnostic: a
// pipe directory over the sun_path budget makes the control daemon print
// "Failed to initialize MPS control daemon" and write no log at all. The error
// has to name the path and the limit, because nothing else will.
func TestCheckPipeDirLength(t *testing.T) {
	ok := "/run/nvidia-mps/shared/" + strings.Repeat("a", MaxPipeDirLength-len("/run/nvidia-mps/shared/"))
	if len(ok) != MaxPipeDirLength {
		t.Fatalf("test data error: built a %d-character path, want %d", len(ok), MaxPipeDirLength)
	}
	if err := CheckPipeDirLength(ok); err != nil {
		t.Errorf("CheckPipeDirLength(%d characters) = %v, want nil at the limit", len(ok), err)
	}

	tooLong := ok + "a"
	err := CheckPipeDirLength(tooLong)
	if err == nil {
		t.Fatalf("CheckPipeDirLength(%d characters) = nil, want an error", len(tooLong))
	}
	if !strings.Contains(err.Error(), tooLong) {
		t.Errorf("error %v does not name the offending path", err)
	}
	if !strings.Contains(err.Error(), fmt.Sprint(MaxPipeDirLength)) {
		t.Errorf("error %v does not name the limit", err)
	}

	// The measured failure: a 117-character path took the daemon down with no
	// log at all.
	if err := CheckPipeDirLength(strings.Repeat("/x", 58) + "y"); err == nil {
		t.Error("a 117-character pipe directory was accepted; it is known to break the control daemon silently")
	}
}

// TestMaxPipeDirLengthLeavesRoomForASocket: the limit is on the DIRECTORY, but
// what has to fit in sun_path is the socket inside it. A limit that used the
// whole 107 bytes would pass every directory and still fail on every socket.
func TestMaxPipeDirLengthLeavesRoomForASocket(t *testing.T) {
	if MaxPipeDirLength >= unixPathMax {
		t.Fatalf("MaxPipeDirLength = %d, which leaves no room for the socket name inside the directory (sun_path allows %d)",
			MaxPipeDirLength, unixPathMax)
	}
	if room := unixPathMax - MaxPipeDirLength; room < len("/control") {
		t.Errorf("only %d characters are reserved for the socket basename, and /control alone needs %d", room, len("/control"))
	}
}
