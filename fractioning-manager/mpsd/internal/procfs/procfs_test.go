// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package procfs

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const (
	// Real container ids, in the shape every runtime actually prints.
	containerA = "3f5a1c9e7b2d4086af1c2e3d4b5a69780f1e2d3c4b5a69788f7e6d5c4b3a2910"
	containerB = "aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888"
)

// procEntry describes one process in a fake /proc.
type procEntry struct {
	dir     string // directory name, so a test can use a non-numeric one
	cgroup  string // "" writes no cgroup file, simulating a process that exited
	comm    string // "" writes no comm file
	exe     string // "" writes no exe symlink, simulating an unreadable one
	cmdline string // argv, NUL-separated by the helper; "" writes no cmdline file
	// cgroupIsDir makes the cgroup path a directory, which is the
	// permission-independent way to make a read fail (tests run as root in CI,
	// so chmod 000 would still be readable).
	cgroupIsDir bool
}

func fakeProc(t *testing.T, entries ...procEntry) string {
	t.Helper()
	root := t.TempDir()

	for _, entry := range entries {
		dir := filepath.Join(root, entry.dir)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if entry.cgroupIsDir {
			if err := os.MkdirAll(filepath.Join(dir, "cgroup"), 0o755); err != nil {
				t.Fatal(err)
			}
		} else if entry.cgroup != "" {
			if err := os.WriteFile(filepath.Join(dir, "cgroup"), []byte(entry.cgroup), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if entry.comm != "" {
			if err := os.WriteFile(filepath.Join(dir, "comm"), []byte(entry.comm), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		// The target is never created: the kernel's exe link points at a real
		// binary, but nothing here reads through it, and a dangling link is how
		// a deleted executable behaves anyway.
		if entry.exe != "" {
			if err := os.Symlink(entry.exe, filepath.Join(dir, "exe")); err != nil {
				t.Fatal(err)
			}
		}
		if entry.cmdline != "" {
			// Real cmdline is NUL-separated and NUL-terminated.
			argv := strings.ReplaceAll(entry.cmdline, " ", "\x00") + "\x00"
			if err := os.WriteFile(filepath.Join(dir, "cmdline"), []byte(argv), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}

	return root
}

func TestPIDsInContainerCgroupShapes(t *testing.T) {
	tests := []struct {
		name   string
		cgroup string
	}{
		{
			name:   "containerd with the systemd cgroup driver",
			cgroup: "0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod1234_5678.slice/cri-containerd-" + containerA + ".scope\n",
		},
		{
			name:   "containerd with the cgroupfs driver",
			cgroup: "11:devices:/kubepods/besteffort/pod1234-5678/" + containerA + "\n",
		},
		{
			name:   "CRI-O",
			cgroup: "0::/kubepods.slice/kubepods-besteffort.slice/crio-" + containerA + ".scope\n",
		},
		{
			name:   "docker",
			cgroup: "9:cpuset:/docker/" + containerA + "\n",
		},
		{
			// A v1 host prints one line per controller; the id has to be found
			// wherever it lands, not only on the first line.
			name: "cgroup v1 multi-line, id only on a later line",
			cgroup: "12:pids:/\n" +
				"11:memory:/kubepods/pod1234/" + containerA + "\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := fakeProc(t, procEntry{dir: "4242", cgroup: tt.cgroup})

			pids, err := PIDsInContainer(root, containerA)
			if err != nil {
				t.Fatalf("PIDsInContainer() error = %v", err)
			}
			if !slices.Equal(pids, []int{4242}) {
				t.Errorf("PIDsInContainer() = %v, want [4242]; an unrecognised cgroup shape means the container is never drained", pids)
			}
		})
	}
}

func TestPIDsInContainerShortIDsMatchNothing(t *testing.T) {
	// The safety property of the whole feature: these PIDs are handed straight
	// to `mps-control client terminate`. A short id matched as a substring would
	// hit other tenants' cgroup paths and terminate their GPU work.
	root := fakeProc(t,
		procEntry{dir: "10", cgroup: "0::/kubepods.slice/cri-containerd-" + containerA + ".scope\n"},
		procEntry{dir: "11", cgroup: "0::/kubepods.slice/cri-containerd-" + containerB + ".scope\n"},
	)

	tests := []struct {
		name        string
		containerID string
		want        []int
	}{
		{name: "empty id", containerID: ""},
		{name: "one character", containerID: "3"},
		{name: "a digit every path contains", containerID: "1"},
		{name: "a slash", containerID: "/"},
		{name: "the kubepods prefix itself", containerID: "kubepods"},
		{name: "eleven characters, one short of the minimum", containerID: containerA[:minContainerIDLength-1]},
		{name: "exactly the minimum is allowed", containerID: containerA[:minContainerIDLength], want: []int{10}},
		{name: "the full id", containerID: containerA, want: []int{10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pids, err := PIDsInContainer(root, tt.containerID)
			if err != nil {
				t.Fatalf("PIDsInContainer() error = %v", err)
			}
			if !slices.Equal(pids, tt.want) {
				t.Errorf("PIDsInContainer(%q) = %v, want %v", tt.containerID, pids, tt.want)
			}
		})
	}
}

func TestPIDsInContainerDoesNotMatchAnotherContainer(t *testing.T) {
	// Terminating a neighbour's MPS client is the worst outcome this package
	// can produce: it drains and kills GPU work belonging to a pod nobody asked
	// to stop.
	root := fakeProc(t,
		procEntry{dir: "10", cgroup: "0::/kubepods.slice/cri-containerd-" + containerA + ".scope\n"},
		procEntry{dir: "20", cgroup: "0::/kubepods.slice/cri-containerd-" + containerB + ".scope\n"},
		procEntry{dir: "30", cgroup: "0::/system.slice/kubelet.service\n"},
	)

	pids, err := PIDsInContainer(root, containerB)
	if err != nil {
		t.Fatalf("PIDsInContainer() error = %v", err)
	}
	if !slices.Equal(pids, []int{20}) {
		t.Errorf("PIDsInContainer() = %v, want only [20]", pids)
	}
}

func TestPIDsInContainerSkipsNonProcessEntries(t *testing.T) {
	// A real /proc is full of non-PID entries, and several of them contain the
	// word "self" or a number. Treating one as a PID would feed a bogus target
	// to the terminate loop.
	root := fakeProc(t,
		procEntry{dir: "self", cgroup: "cri-containerd-" + containerA + ".scope"},
		procEntry{dir: "thread-self", cgroup: "cri-containerd-" + containerA + ".scope"},
		procEntry{dir: "net", cgroup: "cri-containerd-" + containerA + ".scope"},
		procEntry{dir: "1abc", cgroup: "cri-containerd-" + containerA + ".scope"},
		procEntry{dir: "0", cgroup: "cri-containerd-" + containerA + ".scope"},
		procEntry{dir: "sys", cgroup: "cri-containerd-" + containerA + ".scope"},
		procEntry{dir: "7", cgroup: "cri-containerd-" + containerA + ".scope"},
	)

	pids, err := PIDsInContainer(root, containerA)
	if err != nil {
		t.Fatalf("PIDsInContainer() error = %v", err)
	}
	if !slices.Equal(pids, []int{7}) {
		t.Errorf("PIDsInContainer() = %v, want only [7]", pids)
	}
}

func TestPIDsInContainerToleratesDisappearingProcesses(t *testing.T) {
	// A process exiting between the readdir and the read is completely normal
	// during a container stop — it is the common case, in fact. Returning an
	// error there would abandon the drain for every other client.
	root := fakeProc(t,
		procEntry{dir: "10", cgroup: "cri-containerd-" + containerA + ".scope"},
		procEntry{dir: "11"}, // exited: directory exists, cgroup file does not
		procEntry{dir: "12", cgroupIsDir: true},
		procEntry{dir: "13", cgroup: "cri-containerd-" + containerA + ".scope"},
	)

	pids, err := PIDsInContainer(root, containerA)
	if err != nil {
		t.Fatalf("PIDsInContainer() error = %v", err)
	}
	if !slices.Equal(pids, []int{10, 13}) {
		t.Errorf("PIDsInContainer() = %v, want [10 13]", pids)
	}
}

func TestPIDsInContainerMissingRootIsAnError(t *testing.T) {
	// A missing /proc means mpsd is not in the host PID namespace at all. That
	// has to surface, not read as "no clients to drain".
	root := filepath.Join(t.TempDir(), "does-not-exist")

	if _, err := PIDsInContainer(root, containerA); err == nil {
		t.Error("PIDsInContainer() error = nil, want a failure for a missing procfs")
	}
}

func TestPIDsInContainerShortIDShortCircuitsBeforeTouchingTheRoot(t *testing.T) {
	// The length check has to come first: a short id must be rejected outright,
	// not turn into an unrelated procfs error that a caller might retry past.
	pids, err := PIDsInContainer(filepath.Join(t.TempDir(), "nope"), "abc")
	if err != nil || pids != nil {
		t.Errorf("PIDsInContainer() = %v, %v; want nil, nil", pids, err)
	}
}

func TestPIDsInContainerEmptyRootUsesDefault(t *testing.T) {
	// Production passes "" when ProcRoot is unset; defaulting to /proc is what
	// makes that safe.
	if _, err := PIDsInContainer("", strings.Repeat("f", 64)); err != nil {
		t.Errorf("PIDsInContainer(\"\", ...) error = %v, want the default /proc to be readable", err)
	}
}

func TestPIDsByExecutable(t *testing.T) {
	root := fakeProc(t,
		procEntry{dir: "10", exe: "/usr/bin/nvidia-cuda-mps-server"},
		procEntry{dir: "11", exe: "/usr/bin/nvidia-cuda-mps-control"},
		procEntry{dir: "12", exe: "/usr/bin/bash"},
		// No exe link (unreadable): argv[0] is the fallback.
		procEntry{dir: "13", cmdline: "/usr/bin/nvidia-cuda-mps-server --foo"},
		// argv[0] as a bare name rather than a path.
		procEntry{dir: "14", cmdline: "nvidia-cuda-mps-server"},
		// Replaced on disk since exec; the kernel appends the suffix to the link.
		procEntry{dir: "15", exe: "/usr/bin/nvidia-cuda-mps-server (deleted)"},
		// exe wins over a spoofed argv[0].
		procEntry{dir: "16", exe: "/usr/bin/bash", cmdline: "nvidia-cuda-mps-server"},
		// Exited before either could be read.
		procEntry{dir: "17"},
		procEntry{dir: "notapid", exe: "/usr/bin/nvidia-cuda-mps-server"},
	)

	tests := []struct {
		name       string
		executable string
		want       []int
	}{
		{
			// The whole point of not using comm: the real 22-character name is
			// visible here, and the control daemon is not swept up with it.
			name:       "full executable name, servers only",
			executable: "nvidia-cuda-mps-server",
			want:       []int{10, 13, 14, 15},
		},
		{
			name:       "the control daemon is a different executable",
			executable: "nvidia-cuda-mps-control",
			want:       []int{11},
		},
		{
			// These PIDs are SIGKILLed, so a prefix match would kill the
			// control daemon and anything else sharing a prefix.
			name:       "the match is exact, not a prefix",
			executable: "nvidia-cuda-mps",
		},
		{
			// An empty name would otherwise match every process whose
			// executable cannot be determined, and SIGKILL them.
			name:       "empty name matches nothing",
			executable: "",
		},
		{
			name:       "unrelated process",
			executable: "bash",
			want:       []int{12, 16},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pids, err := PIDsByExecutable(root, tt.executable)
			if err != nil {
				t.Fatalf("PIDsByExecutable() error = %v", err)
			}
			if !slices.Equal(pids, tt.want) {
				t.Errorf("PIDsByExecutable(%q) = %v, want %v", tt.executable, pids, tt.want)
			}
		})
	}
}

// The regression this whole function exists for. The kernel stores comm in a
// 16-byte field, so /proc/<pid>/comm holds at most 15 characters:
// "nvidia-cuda-mps-server" reads back as "nvidia-cuda-mps", and so does
// "nvidia-cuda-mps-control". A comm-based lookup therefore finds no servers at
// all when queried by their real name, and cannot tell them from the control
// daemon when queried by the truncated one — so a hard restart, the recovery
// step for a wedged GPU, silently leaves the servers holding that state alive.
func TestPIDsByExecutableSeesNamesLongerThanTASK_COMM_LEN(t *testing.T) {
	const kernelTruncated = "nvidia-cuda-mps" // all comm could ever have offered

	root := fakeProc(t,
		procEntry{dir: "10", exe: "/usr/bin/nvidia-cuda-mps-server", comm: kernelTruncated + "\n"},
		procEntry{dir: "11", exe: "/usr/bin/nvidia-cuda-mps-control", comm: kernelTruncated + "\n"},
	)

	servers, err := PIDsByExecutable(root, "nvidia-cuda-mps-server")
	if err != nil {
		t.Fatalf("PIDsByExecutable() error = %v", err)
	}
	if !slices.Equal(servers, []int{10}) {
		t.Errorf("PIDsByExecutable(%q) = %v, want [10]: the 22-character name must be visible, and the control daemon must not be swept up with it",
			"nvidia-cuda-mps-server", servers)
	}

	// The name comm would have forced the query down to must match neither, or
	// the ambiguity is back.
	if truncated, err := PIDsByExecutable(root, kernelTruncated); err != nil || len(truncated) != 0 {
		t.Errorf("PIDsByExecutable(%q) = %v (err %v), want no matches", kernelTruncated, truncated, err)
	}
}

func TestPIDsByExecutableMissingRootIsAnError(t *testing.T) {
	if _, err := PIDsByExecutable(filepath.Join(t.TempDir(), "gone"), "anything"); err == nil {
		t.Error("PIDsByExecutable() error = nil, want a failure for a missing procfs")
	}
}

func TestPIDsByExecutableEmptyRootUsesDefault(t *testing.T) {
	if _, err := PIDsByExecutable("", "a-process-that-does-not-exist"); err != nil {
		t.Errorf("PIDsByExecutable(\"\", ...) error = %v, want the default /proc to be readable", err)
	}
}

func TestPIDFromDirEntry(t *testing.T) {
	tests := []struct {
		name    string
		entry   string
		wantPID int
		wantOK  bool
	}{
		{name: "ordinary pid", entry: "1234", wantPID: 1234, wantOK: true},
		{name: "pid 1", entry: "1", wantPID: 1, wantOK: true},
		{name: "zero is not a process", entry: "0"},
		{name: "negative", entry: "-1"},
		{name: "self", entry: "self"},
		{name: "empty", entry: ""},
		{name: "trailing space", entry: "12 "},
		{name: "leading space", entry: " 12"},
		// Atoi accepts a sign, so "+12" parses as 12. /proc never contains such
		// an entry, but any other root passed in could.
		{name: "explicit plus sign is accepted by Atoi", entry: "+12", wantPID: 12, wantOK: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pid, ok := pidFromDirEntry(tt.entry)
			if pid != tt.wantPID || ok != tt.wantOK {
				t.Errorf("pidFromDirEntry(%q) = %d, %t; want %d, %t", tt.entry, pid, ok, tt.wantPID, tt.wantOK)
			}
		})
	}
}
