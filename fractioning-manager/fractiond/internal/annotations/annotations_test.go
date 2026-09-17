// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package annotations

import (
	"testing"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/configuration"
)

func TestParseToMemoryMiB(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		expected    string
		expectedErr bool
	}{
		{name: "4Gi to NVIDIA MiB", input: "4Gi", expected: "4096"},
		{name: "2048Mi to NVIDIA MiB", input: "2048Mi", expected: "2048"},
		{name: "7680Mi to NVIDIA MiB", input: "7680Mi", expected: "7680"},
		{name: "1Gi to NVIDIA MiB", input: "1Gi", expected: "1024"},
		{name: "512Mi to NVIDIA MiB", input: "512Mi", expected: "512"},
		{name: "100Mi remains 100 MiB", input: "100Mi", expected: "100"},
		{name: "1000Mi remains 1000 MiB", input: "1000Mi", expected: "1000"},
		{name: "100M rounds up to MiB", input: "100M", expected: "96"},
		{name: "1024Ki minimum valid", input: "1024Ki", expected: "1"},
		{name: "4096M SI megabytes rounds up to MiB", input: "4096M", expected: "3907"},
		{name: "1024M SI megabytes rounds up to MiB", input: "1024M", expected: "977"},
		{name: "1000M SI megabytes rounds up to MiB", input: "1000M", expected: "954"},
		{name: "500M SI megabytes rounds up to MiB", input: "500M", expected: "477"},
		{name: "1G SI gigabyte rounds up to MiB", input: "1G", expected: "954"},
		{name: "5G SI gigabytes rounds up to MiB", input: "5G", expected: "4769"},
		{name: "1M below 1 MiB is error", input: "1M", expectedErr: true},
		{name: "plain integer 4Gi in bytes", input: "4294967296", expected: "4096"},
		{name: "plain byte value below 1 MiB is error", input: "4096", expectedErr: true},
		{name: "zero is error", input: "0", expectedErr: true},
		{name: "500Ki below 1 MiB", input: "500Ki", expectedErr: true},
		{name: "one byte below 1 MiB is error", input: "1048575", expectedErr: true},
		{name: "empty string", input: "", expectedErr: true},
		{name: "whitespace is invalid", input: "  2048Mi  ", expectedErr: true},
		{name: "garbage", input: "notanumber", expectedErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseToMemoryMiB(tt.input)
			if tt.expectedErr {
				if err == nil {
					t.Errorf("parseToMemoryMiB(%q) = %q, expected error", tt.input, got)
				}
				return
			}
			if err != nil {
				t.Errorf("parseToMemoryMiB(%q) error = %v", tt.input, err)
				return
			}
			if got != tt.expected {
				t.Errorf("parseToMemoryMiB(%q) = %q, expected %q", tt.input, got, tt.expected)
			}
		})
	}
}

const visibleDevicesTestKey = "nvidia.com/container.trainer.gpus.devices"

func TestParseVisibleDevices(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		expected    string
	}{
		{
			name:        "single UUID",
			annotations: map[string]string{visibleDevicesTestKey: "GPU-abc123"},
			expected:    "GPU-abc123",
		},
		{
			name:        "comma-separated UUIDs passed through verbatim",
			annotations: map[string]string{visibleDevicesTestKey: "GPU-abc123,GPU-def456"},
			expected:    "GPU-abc123,GPU-def456",
		},
		{
			name:        "surrounding whitespace trimmed",
			annotations: map[string]string{visibleDevicesTestKey: "  GPU-abc123  "},
			expected:    "GPU-abc123",
		},
		{
			name:        "index value passed through",
			annotations: map[string]string{visibleDevicesTestKey: "0"},
			expected:    "0",
		},
		{
			name:        "absent annotation yields empty",
			annotations: map[string]string{"other": "value"},
			expected:    "",
		},
		{
			name:        "blank annotation yields empty",
			annotations: map[string]string{visibleDevicesTestKey: "   "},
			expected:    "",
		},
		{
			name:        "nil annotations yields empty",
			annotations: nil,
			expected:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ParseVisibleDevices(tt.annotations, "trainer", "nvidia.com/container."); got != tt.expected {
				t.Errorf("ParseVisibleDevices() = %q, expected %q", got, tt.expected)
			}
		})
	}
}

func TestParseGPUMemoryAnnotations(t *testing.T) {
	tests := []struct {
		name            string
		annotations     map[string]string
		containerName   string
		prefix          string
		expectedRequest string
		expectedLimit   string
		expectedEmpty   bool
		expectedErr     bool
	}{
		{
			name: "both request and limit",
			annotations: map[string]string{
				"nvidia.com/container.trainer.gpu-memory.request": "2048Mi",
				"nvidia.com/container.trainer.gpu-memory.limit":   "4Gi",
			},
			containerName:   "trainer",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "2048",
			expectedLimit:   "4096",
			expectedEmpty:   false,
		},
		{
			name: "only limit",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "1024M",
			},
			containerName:   "main",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "",
			expectedLimit:   "977",
			expectedEmpty:   false,
		},
		{
			name: "only request",
			annotations: map[string]string{
				"nvidia.com/container.worker.gpu-memory.request": "512Mi",
			},
			containerName:   "worker",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "512",
			expectedLimit:   "",
			expectedEmpty:   false,
		},
		{
			name: "no matching annotations",
			annotations: map[string]string{
				"nvidia.com/container.other.gpu-memory.limit": "4096M",
			},
			containerName:   "main",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "",
			expectedLimit:   "",
			expectedEmpty:   true,
		},
		{
			name:            "nil annotations",
			annotations:     nil,
			containerName:   "main",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "",
			expectedLimit:   "",
			expectedEmpty:   true,
		},
		{
			name:            "empty annotations",
			annotations:     map[string]string{},
			containerName:   "main",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "",
			expectedLimit:   "",
			expectedEmpty:   true,
		},
		{
			name: "malformed limit value",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "not-a-number",
			},
			containerName: "main",
			prefix:        configuration.DefaultAnnotationPrefix,
			expectedErr:   true,
		},
		{
			name: "value below 1 MiB",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "500Ki",
			},
			containerName: "main",
			prefix:        configuration.DefaultAnnotationPrefix,
			expectedErr:   true,
		},
		{
			name: "multiple containers only target extracted",
			annotations: map[string]string{
				"nvidia.com/container.sidecar.gpu-memory.limit": "1024M",
				"nvidia.com/container.main.gpu-memory.limit":    "4096M",
				"nvidia.com/container.init.gpu-memory.limit":    "512M",
			},
			containerName:   "main",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "",
			expectedLimit:   "3907",
			expectedEmpty:   false,
		},
		{
			name: "container name with dashes",
			annotations: map[string]string{
				"nvidia.com/container.my-training-job.gpu-memory.limit": "2048M",
			},
			containerName:   "my-training-job",
			prefix:          configuration.DefaultAnnotationPrefix,
			expectedRequest: "",
			expectedLimit:   "1954",
			expectedEmpty:   false,
		},
		{
			name: "custom prefix",
			annotations: map[string]string{
				"gpu-fractioning.kai.scheduler/container.main.gpu-memory.limit": "4Gi",
			},
			containerName:   "main",
			prefix:          "gpu-fractioning.kai.scheduler/container.",
			expectedRequest: "",
			expectedLimit:   "4096",
			expectedEmpty:   false,
		},
		{
			name: "empty prefix still matches bare container keys",
			annotations: map[string]string{
				"main.gpu-memory.limit": "4Gi",
			},
			containerName:   "main",
			prefix:          "",
			expectedRequest: "",
			expectedLimit:   "4096",
			expectedEmpty:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := ParseGPUMemoryAnnotations(tt.annotations, tt.containerName, tt.prefix)
			if tt.expectedErr {
				if err == nil {
					t.Errorf("ParseGPUMemoryAnnotations() = %+v, expected error", cfg)
				}
				return
			}
			if err != nil {
				t.Errorf("ParseGPUMemoryAnnotations() error = %v", err)
				return
			}
			if cfg.Request != tt.expectedRequest {
				t.Errorf("Request = %q, expected %q", cfg.Request, tt.expectedRequest)
			}
			if cfg.Limit != tt.expectedLimit {
				t.Errorf("Limit = %q, expected %q", cfg.Limit, tt.expectedLimit)
			}
			if cfg.IsEmpty() != tt.expectedEmpty {
				t.Errorf("IsEmpty() = %v, expected %v", cfg.IsEmpty(), tt.expectedEmpty)
			}
		})
	}
}

const computeModeTestKey = "nvidia.com/container.trainer.gpu-compute.mode"

func TestParseComputeMode(t *testing.T) {
	tests := []struct {
		name             string
		annotations      map[string]string
		smSharingEnabled bool
		expected         ComputeMode
		expectedErr      bool
	}{
		{
			name:             "absent annotation defaults to time-slicing",
			annotations:      map[string]string{"other": "value"},
			smSharingEnabled: true,
			expected:         ComputeModeTimeSlicing,
		},
		{
			name:             "nil annotations defaults to time-slicing",
			annotations:      nil,
			smSharingEnabled: true,
			expected:         ComputeModeTimeSlicing,
		},
		{
			// Absence means "no preference"; an empty value means the mode was
			// set to nothing, which is not one of the two documented values.
			name:             "empty annotation fails",
			annotations:      map[string]string{computeModeTestKey: ""},
			smSharingEnabled: true,
			expectedErr:      true,
		},
		{
			name:             "whitespace-only annotation fails",
			annotations:      map[string]string{computeModeTestKey: "   "},
			smSharingEnabled: true,
			expectedErr:      true,
		},
		{
			name:             "explicit time-slicing",
			annotations:      map[string]string{computeModeTestKey: "time-slicing"},
			smSharingEnabled: true,
			expected:         ComputeModeTimeSlicing,
		},
		{
			name:             "explicit time-slicing, sm-sharing disabled cluster-wide",
			annotations:      map[string]string{computeModeTestKey: "time-slicing"},
			smSharingEnabled: false,
			expected:         ComputeModeTimeSlicing,
		},
		{
			name:             "sm-sharing",
			annotations:      map[string]string{computeModeTestKey: "sm-sharing"},
			smSharingEnabled: true,
			expected:         ComputeModeSMSharing,
		},
		{
			name:             "surrounding whitespace trimmed",
			annotations:      map[string]string{computeModeTestKey: "  sm-sharing  "},
			smSharingEnabled: true,
			expected:         ComputeModeSMSharing,
		},
		{
			name:             "sm-sharing rejected when disabled cluster-wide",
			annotations:      map[string]string{computeModeTestKey: "sm-sharing"},
			smSharingEnabled: false,
			expectedErr:      true,
		},
		{
			name:             "invalid value fails",
			annotations:      map[string]string{computeModeTestKey: "mig"},
			smSharingEnabled: true,
			expectedErr:      true,
		},
		{
			name:             "case-sensitive: capitalized value fails",
			annotations:      map[string]string{computeModeTestKey: "SM-Sharing"},
			smSharingEnabled: true,
			expectedErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseComputeMode(tt.annotations, "trainer", configuration.DefaultAnnotationPrefix, tt.smSharingEnabled)
			if tt.expectedErr {
				if err == nil {
					t.Errorf("ParseComputeMode() = %q, expected error", got)
				}
				return
			}
			if err != nil {
				t.Errorf("ParseComputeMode() error = %v", err)
				return
			}
			if got != tt.expected {
				t.Errorf("ParseComputeMode() = %q, expected %q", got, tt.expected)
			}
		})
	}
}

func TestApplyDefaults(t *testing.T) {
	tests := []struct {
		name        string
		in          GPUMemoryConfig
		wantRequest string
		wantLimit   string
	}{
		{"both set unchanged", GPUMemoryConfig{Request: "3221", Limit: "6442"}, "3221", "6442"},
		{"request only defaults limit", GPUMemoryConfig{Request: "4096"}, "4096", "4096"},
		{"limit only defaults request", GPUMemoryConfig{Limit: "6442"}, "6442", "6442"},
		{"empty stays empty", GPUMemoryConfig{}, "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.in.ApplyDefaults()
			if got.Request != tt.wantRequest || got.Limit != tt.wantLimit {
				t.Errorf("ApplyDefaults(%+v) = {Request:%q Limit:%q}, want {Request:%q Limit:%q}",
					tt.in, got.Request, got.Limit, tt.wantRequest, tt.wantLimit)
			}
		})
	}
}

const computePortionTestKey = "nvidia.com/container.trainer.gpu-compute.portion"

// TestParseComputePortion is the first half of the SM-occupancy cap: whatever
// this returns becomes CUDA_MPS_ACTIVE_THREAD_PERCENTAGE inside a container
// that shares a GPU with other tenants. Two failure shapes matter more than the
// happy path:
//
//   - a value that is silently accepted when it should not be (a percentage
//     pasted where a portion belongs, a negative, a zero) hands the container a
//     cap that is meaningless or, worse, that MPS reads as "no limit";
//   - a value that rounds to 0% — MPS treats 0 as unlimited, so the smallest
//     requests would become the least restricted ones. Hence the floor at 1%.
func TestParseComputePortion(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		wantPercent int
		wantFound   bool
		wantErr     bool
	}{
		{
			// Absence is the pre-existing world: memory was capped, compute was
			// not. It must stay a clean "nothing to enforce", never an error
			// that would block every pod the scheduler has not yet annotated.
			name:        "absent annotation is not found and not an error",
			annotations: map[string]string{"other": "value"},
		},
		{
			name:        "nil annotations is not found and not an error",
			annotations: nil,
		},
		{
			name:        "half a GPU",
			annotations: map[string]string{computePortionTestKey: "0.5"},
			wantPercent: 50,
			wantFound:   true,
		},
		{
			name:        "whole GPU",
			annotations: map[string]string{computePortionTestKey: "1"},
			wantPercent: 100,
			wantFound:   true,
		},
		{
			name:        "whole GPU written as 1.0",
			annotations: map[string]string{computePortionTestKey: "1.0"},
			wantPercent: 100,
			wantFound:   true,
		},
		{
			// MPS only accepts whole percents, so a third of a GPU has to round
			// somewhere. Rounding down (33, not 34) keeps the sum of three such
			// containers at or under 100% of the card.
			name:        "one third rounds to nearest whole percent",
			annotations: map[string]string{computePortionTestKey: "0.333"},
			wantPercent: 33,
			wantFound:   true,
		},
		{
			name:        "rounds up to nearest whole percent",
			annotations: map[string]string{computePortionTestKey: "0.336"},
			wantPercent: 34,
			wantFound:   true,
		},
		{
			// 0.5% and 0.49% both land on or below the rounding boundary; either
			// way the answer must be 1, not 0. A 0 here would be injected as
			// CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=0, which MPS reads as unlimited
			// — a tiny request would become an uncapped one.
			name:        "half a percent floors at 1%, never 0%",
			annotations: map[string]string{computePortionTestKey: "0.005"},
			wantPercent: 1,
			wantFound:   true,
		},
		{
			name:        "just under half a percent floors at 1%, never 0%",
			annotations: map[string]string{computePortionTestKey: "0.0049"},
			wantPercent: 1,
			wantFound:   true,
		},
		{
			name:        "vanishingly small portion still floors at 1%",
			annotations: map[string]string{computePortionTestKey: "0.0000001"},
			wantPercent: 1,
			wantFound:   true,
		},
		{
			// Scientific notation parses as a number, so it must be handled by
			// the range/floor rules rather than rejected as a typo.
			name:        "scientific notation floors at 1%",
			annotations: map[string]string{computePortionTestKey: "1e-3"},
			wantPercent: 1,
			wantFound:   true,
		},
		{
			name:        "scientific notation in range",
			annotations: map[string]string{computePortionTestKey: "5e-1"},
			wantPercent: 50,
			wantFound:   true,
		},
		{
			// THE bypass to guard: someone writes the percentage instead of the
			// portion. Accepting it would mean percent = 5000, i.e. a container
			// asking for half a GPU is handed a cap 50x the card. It must be an
			// error, not a silent pass-through.
			name:        "a percentage where a portion belongs is rejected",
			annotations: map[string]string{computePortionTestKey: "50"},
			wantErr:     true,
		},
		{
			name:        "100 (percent) is rejected, not read as 100 GPUs",
			annotations: map[string]string{computePortionTestKey: "100"},
			wantErr:     true,
		},
		{
			name:        "anything above one whole GPU is rejected",
			annotations: map[string]string{computePortionTestKey: "1.5"},
			wantErr:     true,
		},
		{
			// Guards the boundary itself: the range is (0, 1], so the smallest
			// representable step past 1 must already fail.
			name:        "barely above one whole GPU is rejected",
			annotations: map[string]string{computePortionTestKey: "1.0000001"},
			wantErr:     true,
		},
		{
			// Zero would be injected as 0%, which MPS reads as "no limit" — the
			// exact opposite of what "zero compute" asks for.
			name:        "zero is rejected",
			annotations: map[string]string{computePortionTestKey: "0"},
			wantErr:     true,
		},
		{
			name:        "zero written as 0.0 is rejected",
			annotations: map[string]string{computePortionTestKey: "0.0"},
			wantErr:     true,
		},
		{
			name:        "negative portion is rejected",
			annotations: map[string]string{computePortionTestKey: "-0.5"},
			wantErr:     true,
		},
		{
			name:        "negative zero is rejected",
			annotations: map[string]string{computePortionTestKey: "-0"},
			wantErr:     true,
		},
		{
			// Present but blank is a value, not an absence: the scheduler wrote
			// the key, so falling back to "uncapped" would hide a scheduler bug
			// behind a running, unlimited container.
			name:        "present but blank is an error, not an absence",
			annotations: map[string]string{computePortionTestKey: ""},
			wantErr:     true,
		},
		{
			name:        "whitespace-only is an error, not an absence",
			annotations: map[string]string{computePortionTestKey: "   "},
			wantErr:     true,
		},
		{
			name:        "surrounding whitespace is trimmed",
			annotations: map[string]string{computePortionTestKey: "  0.5\t"},
			wantPercent: 50,
			wantFound:   true,
		},
		{
			name:        "non-numeric is rejected",
			annotations: map[string]string{computePortionTestKey: "abc"},
			wantErr:     true,
		},
		{
			name:        "a portion with a percent sign is rejected",
			annotations: map[string]string{computePortionTestKey: "50%"},
			wantErr:     true,
		},
		{
			name:        "a k8s-style quantity is rejected",
			annotations: map[string]string{computePortionTestKey: "500m"},
			wantErr:     true,
		},
		{
			// strconv.ParseFloat accepts "NaN" and "Inf". NaN fails every
			// comparison, so a range check written as "portion <= 0" instead of
			// "!(portion > 0)" would let it through and produce a garbage
			// percentage from math.Round(NaN).
			name:        "NaN is rejected",
			annotations: map[string]string{computePortionTestKey: "NaN"},
			wantErr:     true,
		},
		{
			name:        "lowercase nan is rejected",
			annotations: map[string]string{computePortionTestKey: "nan"},
			wantErr:     true,
		},
		{
			name:        "positive infinity is rejected",
			annotations: map[string]string{computePortionTestKey: "Inf"},
			wantErr:     true,
		},
		{
			name:        "spelled-out infinity is rejected",
			annotations: map[string]string{computePortionTestKey: "+Infinity"},
			wantErr:     true,
		},
		{
			name:        "negative infinity is rejected",
			annotations: map[string]string{computePortionTestKey: "-Inf"},
			wantErr:     true,
		},
		{
			// Another container's cap must never be applied to this one: the
			// annotation key is per-container, and a prefix/name mix-up would
			// hand a sidecar the trainer's SM budget.
			name: "a sibling container's portion does not leak",
			annotations: map[string]string{
				"nvidia.com/container.sidecar.gpu-compute.portion": "0.5",
			},
		},
		{
			// A container name that is a prefix of another's must not match by
			// accident (train vs trainer).
			name: "a longer container name is not a prefix match",
			annotations: map[string]string{
				"nvidia.com/container.trainer-2.gpu-compute.portion": "0.5",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			percent, found, err := ParseComputePortion(tt.annotations, "trainer", configuration.DefaultAnnotationPrefix)

			if tt.wantErr {
				if err == nil {
					t.Fatalf("ParseComputePortion() = (%d, %v, nil), expected an error", percent, found)
				}
				// An error must never come back as "found": a caller that logs
				// the error and carries on (fail-open) reads found to decide
				// whether to inject a cap, and a true here would inject
				// whatever garbage percent came with it.
				if found {
					t.Errorf("ParseComputePortion() returned found=true alongside an error: %v", err)
				}
				if percent != 0 {
					t.Errorf("ParseComputePortion() returned percent=%d alongside an error, want 0", percent)
				}
				return
			}

			if err != nil {
				t.Fatalf("ParseComputePortion() unexpected error: %v", err)
			}
			if found != tt.wantFound {
				t.Errorf("ParseComputePortion() found = %v, want %v", found, tt.wantFound)
			}
			if percent != tt.wantPercent {
				t.Errorf("ParseComputePortion() percent = %d, want %d", percent, tt.wantPercent)
			}
			// A cap of 0 is "unlimited" to MPS, and anything above 100 is not a
			// thing. Belt and braces around every accepted value in the table.
			if found && (percent < 1 || percent > 100) {
				t.Errorf("ParseComputePortion() percent = %d, outside the usable MPS range [1, 100]", percent)
			}
		})
	}
}

// TestParseComputePortionHonorsPrefix guards the configurable annotation
// prefix: fractiond and the scheduler agree on it through configuration, and a
// portion written under a different prefix must read as absent rather than as
// an uncapped container that looks annotated.
func TestParseComputePortionHonorsPrefix(t *testing.T) {
	ann := map[string]string{"example.com/container.trainer.gpu-compute.portion": "0.25"}

	if percent, found, err := ParseComputePortion(ann, "trainer", "example.com/container."); err != nil || !found || percent != 25 {
		t.Errorf("with matching prefix = (%d, %v, %v), want (25, true, nil)", percent, found, err)
	}
	if percent, found, err := ParseComputePortion(ann, "trainer", configuration.DefaultAnnotationPrefix); err != nil || found || percent != 0 {
		t.Errorf("with the default prefix = (%d, %v, %v), want (0, false, nil)", percent, found, err)
	}
}

// TestComputePortionAnnotationKey pins the key fractiond reads against the one
// the scheduler writes. They are built in different repos, so the literal is
// the contract: a rename on this side turns every annotated pod into an
// unannotated one, i.e. silently uncapped compute.
func TestComputePortionAnnotationKey(t *testing.T) {
	if got := ComputePortionAnnotationKey(configuration.DefaultAnnotationPrefix, "trainer"); got != computePortionTestKey {
		t.Errorf("ComputePortionAnnotationKey() = %q, want %q", got, computePortionTestKey)
	}
}
