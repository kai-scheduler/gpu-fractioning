// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package injection

import (
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
)

// The injected names are a contract with consumers outside this repo — most
// importantly the container toolkit's apply-cuda-memory-limits CDI hook, which
// looks the GPU-memory keys up by exact string and does nothing at all when it
// finds neither. Pinning the literals here is what turns a rename into a failing
// test instead of into limits that quietly stop being enforced.
func TestInjectedEnvNames(t *testing.T) {
	want := map[string]string{
		"EnvGPUMemoryRequest": "NVIDIA_GPU_MEMORY_REQUEST",
		"EnvGPUMemoryLimit":   "NVIDIA_GPU_MEMORY_LIMIT",
		"EnvMPSPipeDirectory": "CUDA_MPS_PIPE_DIRECTORY",
		"EnvVisibleDevices":   "NVIDIA_VISIBLE_DEVICES",
		// The MPS-enforced caps. These spellings are MPS's, not ours: the CUDA
		// runtime inside the container reads them when it attaches to an MPS
		// server, so a typo silently means "no limit".
		"EnvMPSActiveThreadPercentage": "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
		"EnvMPSPinnedDeviceMemLimit":   "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
	}
	got := map[string]string{
		"EnvGPUMemoryRequest":          EnvGPUMemoryRequest,
		"EnvGPUMemoryLimit":            EnvGPUMemoryLimit,
		"EnvMPSPipeDirectory":          EnvMPSPipeDirectory,
		"EnvVisibleDevices":            EnvVisibleDevices,
		"EnvMPSActiveThreadPercentage": EnvMPSActiveThreadPercentage,
		"EnvMPSPinnedDeviceMemLimit":   EnvMPSPinnedDeviceMemLimit,
	}
	for name, wantValue := range want {
		if got[name] != wantValue {
			t.Errorf("%s = %q, want %q", name, got[name], wantValue)
		}
	}

	// AllEnvKeys is what the audit ranges over to decide whether a running
	// container was injected, so a key missing from it is a key the audit is
	// blind to: a container could be running without that limit and never be
	// flagged. Compare as an exact set rather than by length plus membership —
	// the latter passes for a list that duplicates one key and drops another,
	// which is exactly the shape a careless edit produces.
	wantAll := make([]string, 0, len(want))
	for _, value := range want {
		wantAll = append(wantAll, value)
	}
	assertSameKeySet(t, "AllEnvKeys", AllEnvKeys, wantAll)

	// EnforcementEnvKeys is pinned exactly, not just as a subset of AllEnvKeys,
	// because membership is what decides whether a container's own value for a
	// key is allowed to survive. A limit key missing from this list is a
	// bypass; a location key wrongly added to it gets stripped from containers
	// the hook has no replacement value for.
	assertSameKeySet(t, "EnforcementEnvKeys", EnforcementEnvKeys, []string{
		EnvGPUMemoryRequest,
		EnvGPUMemoryLimit,
		EnvMPSActiveThreadPercentage,
		EnvMPSPinnedDeviceMemLimit,
	})

	// Every enforcement key must also be an injected key: the create hook
	// removes a container's own value for these before setting its own, and a
	// key listed here but not injected would be removed and never replaced.
	for _, key := range EnforcementEnvKeys {
		if !slices.Contains(AllEnvKeys, key) {
			t.Errorf("EnforcementEnvKeys contains %q, which is not in AllEnvKeys", key)
		}
	}

	// CUDA_MPS_PIPE_DIRECTORY tells the container where the MPS socket is; it
	// carries no limit, and it is unconditionally injected, so it has no
	// business in the enforcement list. NVIDIA_VISIBLE_DEVICES is conditional
	// on the scheduler's device assignment and is handled on its own terms.
	for _, key := range []string{EnvMPSPipeDirectory, EnvVisibleDevices} {
		if slices.Contains(EnforcementEnvKeys, key) {
			t.Errorf("EnforcementEnvKeys contains %q, which is a location, not a limit", key)
		}
	}
}

// assertSameKeySet compares an env-key list against its expected contents as a
// set, and separately rejects duplicates. A duplicated key is not harmless: the
// create hook emits one removal entry per listed key, so a repeat produces two
// removal markers for the same variable in a single adjustment.
func assertSameKeySet(t *testing.T, name string, got, want []string) {
	t.Helper()

	seen := map[string]bool{}
	for _, key := range got {
		if seen[key] {
			t.Errorf("%s lists %q more than once", name, key)
		}
		seen[key] = true
	}

	g := slices.Clone(got)
	w := slices.Clone(want)
	slices.Sort(g)
	slices.Sort(w)
	if !slices.Equal(g, w) {
		t.Errorf("%s = %v, want %v", name, g, w)
	}
}

func TestPinnedDeviceMemLimit(t *testing.T) {
	tests := []struct {
		name        string
		limitMiB    string
		deviceCount int
		want        string
	}{
		{name: "single device", limitMiB: "40960", deviceCount: 1, want: "0=40960M"},
		{name: "multiple devices are capped individually", limitMiB: "4096", deviceCount: 3, want: "0=4096M,1=4096M,2=4096M"},
		{name: "no limit means nothing to enforce", limitMiB: "", deviceCount: 2, want: ""},
		{name: "no devices means nothing to name", limitMiB: "4096", deviceCount: 0, want: ""},
		{name: "negative device count is not a limit", limitMiB: "4096", deviceCount: -1, want: ""},
		{
			// The topology this enforcement work targets. Every ordinal has to
			// appear: MPS applies the cap only to the devices named in the
			// list, so a container assigned eight GPUs and capped on seven is
			// uncapped on the eighth.
			name:     "every ordinal of an eight-GPU node is named",
			limitMiB: "10240", deviceCount: 8,
			want: "0=10240M,1=10240M,2=10240M,3=10240M,4=10240M,5=10240M,6=10240M,7=10240M",
		},
		{
			// The smallest accepted memory annotation is 1 MiB, so this is the
			// lower end of what the parser can hand us; it must still render as
			// a limit rather than as an empty "nothing to enforce" string.
			name: "one MiB still renders a limit", limitMiB: "1", deviceCount: 1, want: "0=1M",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := PinnedDeviceMemLimit(tt.limitMiB, tt.deviceCount); got != tt.want {
				t.Errorf("PinnedDeviceMemLimit(%q, %d) = %q, want %q", tt.limitMiB, tt.deviceCount, got, tt.want)
			}
		})
	}
}

func TestDeviceCount(t *testing.T) {
	tests := []struct {
		name           string
		visibleDevices string
		want           int
	}{
		{name: "empty", visibleDevices: "", want: 0},
		{name: "blank", visibleDevices: "   ", want: 0},
		{name: "single uuid", visibleDevices: "GPU-abc123", want: 1},
		{name: "two uuids", visibleDevices: "GPU-abc123,GPU-def456", want: 2},
		{name: "cdi device names", visibleDevices: "k8s.device-plugin.nvidia.com/gpu=GPU-abc,k8s.device-plugin.nvidia.com/gpu=GPU-def", want: 2},
		{name: "trailing separator is not a device", visibleDevices: "GPU-abc123,", want: 1},
		{name: "spaces around entries", visibleDevices: " GPU-abc123 , GPU-def456 ", want: 2},
		// A count that runs ahead of reality is the dangerous direction: it
		// would name device ordinals the container does not have, and MPS
		// ignores limits for devices that are not there — so the real devices
		// stay capped but the list carries noise. A count that runs behind
		// leaves a real device uncapped. Both are pinned here.
		{name: "lone separator names no device", visibleDevices: ",", want: 0},
		{name: "only separators name no device", visibleDevices: ",,,", want: 0},
		{name: "leading separator is not a device", visibleDevices: ",GPU-abc123", want: 1},
		{name: "empty entry between devices is not a device", visibleDevices: "GPU-abc123,,GPU-def456", want: 2},
		{name: "blank entry between devices is not a device", visibleDevices: "GPU-abc123, ,GPU-def456", want: 2},
		{name: "tabs and newlines are whitespace", visibleDevices: "\tGPU-abc123\n,\nGPU-def456\t", want: 2},
		{name: "device indices count like uuids", visibleDevices: "0,1,2,3", want: 4},
		{name: "index zero alone is one device", visibleDevices: "0", want: 1},
		{
			// "all" and "void" are NVIDIA container-runtime keywords, not device
			// lists. DeviceCount does not interpret entries, so each counts as
			// one — pinned so nobody assumes this function understands them.
			name: "runtime keywords are counted, not interpreted", visibleDevices: "all", want: 1,
		},
		{name: "eight-GPU assignment", visibleDevices: "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7", want: 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DeviceCount(tt.visibleDevices); got != tt.want {
				t.Errorf("DeviceCount(%q) = %d, want %d", tt.visibleDevices, got, tt.want)
			}
		})
	}
}

// The want values below are spelled out as literals rather than built from the
// configuration constants: these exact paths are the contract with mpsd (which
// renders the shared server at <pipeDir>/shared/default) and with the container
// image (which reads CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps). Deriving them
// from the same constants the code uses would make the test agree with any
// change to those constants, including a breaking one.
func TestMPSPipeMount(t *testing.T) {
	tests := []struct {
		name            string
		mpsPipeDir      string
		mode            annotations.ComputeMode
		wantSource      string
		wantDestination string
	}{
		{
			name:            "sm-sharing routes to the shared server socket",
			mpsPipeDir:      "/run/nvidia-mps",
			mode:            annotations.ComputeModeSMSharing,
			wantSource:      "/run/nvidia-mps/shared/default",
			wantDestination: "/tmp/nvidia-mps",
		},
		{
			name:            "time-slicing mounts the pipe directory as-is",
			mpsPipeDir:      "/run/nvidia-mps",
			mode:            annotations.ComputeModeTimeSlicing,
			wantSource:      "/run/nvidia-mps",
			wantDestination: "/run/nvidia-mps",
		},
		{
			// The plugin treats a missing compute-mode annotation as
			// time-slicing, but guard the zero value directly: an unset mode
			// must never select the shared socket.
			name:            "empty mode behaves as time-slicing",
			mpsPipeDir:      "/run/nvidia-mps",
			mode:            "",
			wantSource:      "/run/nvidia-mps",
			wantDestination: "/run/nvidia-mps",
		},
		{
			// The pipe directory is CRD-configurable, so only the shared
			// subpath is fixed — the host prefix has to follow the config.
			name:            "sm-sharing honors a non-default pipe directory",
			mpsPipeDir:      "/var/run/mps",
			mode:            annotations.ComputeModeSMSharing,
			wantSource:      "/var/run/mps/shared/default",
			wantDestination: "/tmp/nvidia-mps",
		},
		{
			// Only the exact "sm-sharing" string may select the shared server.
			// ParseComputeMode rejects anything else, but this function takes a
			// ComputeMode from callers (including the audit) rather than
			// re-validating, so the fallback must be the conservative one:
			// a near-miss lands on the per-container default socket, not on the
			// socket every other tenant shares.
			name:            "an unrecognized mode falls back to time-slicing",
			mpsPipeDir:      "/run/nvidia-mps",
			mode:            annotations.ComputeMode("sm_sharing"),
			wantSource:      "/run/nvidia-mps",
			wantDestination: "/run/nvidia-mps",
		},
		{
			name:            "case does not select the shared socket",
			mpsPipeDir:      "/run/nvidia-mps",
			mode:            annotations.ComputeMode("SM-Sharing"),
			wantSource:      "/run/nvidia-mps",
			wantDestination: "/run/nvidia-mps",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source, destination := MPSPipeMount(tt.mpsPipeDir, tt.mode)
			if source != tt.wantSource {
				t.Errorf("source = %q, want %q", source, tt.wantSource)
			}
			if destination != tt.wantDestination {
				t.Errorf("destination = %q, want %q", destination, tt.wantDestination)
			}
		})
	}
}

// TestPinnedDeviceMemLimitFormat guards the wire format itself rather than one
// example of it. CUDA_MPS_PINNED_DEVICE_MEM_LIMIT is parsed by MPS, not by us:
// a stray space after the comma, a missing unit suffix or a trailing separator
// makes MPS reject or misread the whole list, and a rejected list is an
// unenforced one — the container runs with no per-device memory cap at all.
func TestPinnedDeviceMemLimitFormat(t *testing.T) {
	got := PinnedDeviceMemLimit("4096", 4)

	if strings.ContainsAny(got, " \t") {
		t.Errorf("PinnedDeviceMemLimit produced whitespace in %q; MPS parses this verbatim", got)
	}
	if strings.HasPrefix(got, ",") || strings.HasSuffix(got, ",") {
		t.Errorf("PinnedDeviceMemLimit produced a dangling separator in %q", got)
	}

	entries := strings.Split(got, ",")
	if len(entries) != 4 {
		t.Fatalf("PinnedDeviceMemLimit(_, 4) produced %d entries in %q, want 4", len(entries), got)
	}
	for ordinal, entry := range entries {
		want := strconv.Itoa(ordinal) + "=4096M"
		if entry != want {
			t.Errorf("entry %d = %q, want %q", ordinal, entry, want)
		}
	}
}

// TestPinnedDeviceMemLimitMatchesDeviceCount ties the two halves together the
// way the create hook uses them: whatever the scheduler wrote into the device
// annotation is counted by DeviceCount and then enumerated by
// PinnedDeviceMemLimit. If those two disagree, some assigned GPU ends up
// without a cap while the container still has access to it.
func TestPinnedDeviceMemLimitMatchesDeviceCount(t *testing.T) {
	for _, visibleDevices := range []string{
		"GPU-abc123",
		"GPU-abc123,GPU-def456",
		"GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f,GPU-g,GPU-h",
		" GPU-abc123 , GPU-def456 ",
		"GPU-abc123,",
	} {
		t.Run(visibleDevices, func(t *testing.T) {
			count := DeviceCount(visibleDevices)
			limit := PinnedDeviceMemLimit("4096", count)
			if limit == "" {
				t.Fatalf("DeviceCount(%q) = %d produced no limit", visibleDevices, count)
			}
			if entries := strings.Split(limit, ","); len(entries) != count {
				t.Errorf("limit %q names %d devices, but DeviceCount(%q) = %d",
					limit, len(entries), visibleDevices, count)
			}
		})
	}

	// And the empty case: no assignment means no ordinals to name, so there is
	// nothing to inject rather than a "0=..." entry for a device the container
	// may not even have.
	if got := PinnedDeviceMemLimit("4096", DeviceCount("")); got != "" {
		t.Errorf("unassigned container got a pinned memory limit %q, want none", got)
	}
}
