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
			// somewhere. Rounding DOWN (33, not 34) keeps the sum of three such
			// containers at or under 100% of the card.
			name:        "one third rounds down to a whole percent",
			annotations: map[string]string{computePortionTestKey: "0.333"},
			wantPercent: 33,
			wantFound:   true,
		},
		{
			// The case that makes flooring mandatory rather than tidy. Rounding
			// to nearest turns 0.336 into 34, and three tenants of one GPU into
			// 102% of it. A namespace active-thread percentage is a hard SM
			// partition, so over-subscribing it is not optimistic accounting —
			// it asks the driver for SMs that do not exist.
			name:        "a portion that would round up is floored instead",
			annotations: map[string]string{computePortionTestKey: "0.336"},
			wantPercent: 33,
			wantFound:   true,
		},
		{
			// The eight-way split that was the original defect: 0.125 rounded
			// to nearest is 13, and 13 × 8 = 104.
			name:        "an eighth of a GPU floors to 12, not 13",
			annotations: map[string]string{computePortionTestKey: "0.125"},
			wantPercent: 12,
			wantFound:   true,
		},
		{
			// Binary floating point makes 0.07*100 come out as 7.000000000000001
			// and 0.29*100 as 28.999999999999996. Flooring the latter would give
			// 28 if it were done naively on the decimal the user wrote; what
			// matters is that the result is never ABOVE the portion, so 28 is
			// the correct, conservative answer and 29 would not be.
			name:        "a portion whose percent is not exact in binary still floors",
			annotations: map[string]string{computePortionTestKey: "0.29"},
			wantPercent: 28,
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
			// The table describes the case where the cap applies: an
			// sm-sharing container on a cluster with sm-sharing enabled. The
			// two ways it does not apply have tests of their own below.
			percent, found, err := ParseComputePortion(tt.annotations, "trainer", configuration.DefaultAnnotationPrefix, ComputeModeSMSharing, true)

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

	if percent, found, err := ParseComputePortion(ann, "trainer", "example.com/container.", ComputeModeSMSharing, true); err != nil || !found || percent != 25 {
		t.Errorf("with matching prefix = (%d, %v, %v), want (25, true, nil)", percent, found, err)
	}
	if percent, found, err := ParseComputePortion(ann, "trainer", configuration.DefaultAnnotationPrefix, ComputeModeSMSharing, true); err != nil || found || percent != 0 {
		t.Errorf("with the default prefix = (%d, %v, %v), want (0, false, nil)", percent, found, err)
	}
}

// TestParseComputePortionAppliesOnlyToSMSharing is the time-slicing exemption.
//
// The cap is a per-namespace active-thread percentage, which is a hard
// partition of the SMs: a container capped at 50% cannot touch the other half
// even when the card is idle. That is what sm-sharing means and the opposite of
// what time-slicing promises — under time-slicing, unused time goes to whoever
// wants it. Applying the portion in both modes (which is what the code used to
// do) silently halves a GPU for a lone time-slicing tenant, and does it in a way
// no annotation, log line or metric distinguishes from the workload simply being
// slow.
func TestParseComputePortionAppliesOnlyToSMSharing(t *testing.T) {
	tests := []struct {
		name             string
		mode             ComputeMode
		smSharingEnabled bool
		wantPercent      int
		wantFound        bool
	}{
		{
			name:             "sm-sharing on an enabled cluster is capped",
			mode:             ComputeModeSMSharing,
			smSharingEnabled: true,
			wantPercent:      50,
			wantFound:        true,
		},
		{
			name:             "time-slicing is not capped",
			mode:             ComputeModeTimeSlicing,
			smSharingEnabled: true,
		},
		{
			// The kill switch. --support-sm-sharing=false turns off the shared
			// MPS server the per-container namespaces live on, so there is
			// nowhere to enforce a cap; leaving containers hard-capped by a
			// feature that has been switched off would make the switch a way to
			// break workloads rather than a way to back the feature out.
			name:             "a disabled cluster does not cap, even in sm-sharing mode",
			mode:             ComputeModeSMSharing,
			smSharingEnabled: false,
		},
		{
			name:             "a disabled cluster does not cap in time-slicing mode either",
			mode:             ComputeModeTimeSlicing,
			smSharingEnabled: false,
		},
		{
			// An empty mode is what a zero value looks like. It is not
			// sm-sharing, so it must not be capped: defaulting an unknown mode
			// into the capped branch is exactly the mistake that would put a
			// hard partition on a time-slicing container.
			name:             "an unset mode is not capped",
			mode:             "",
			smSharingEnabled: true,
		},
	}

	ann := map[string]string{computePortionTestKey: "0.5"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			percent, found, err := ParseComputePortion(ann, "trainer", configuration.DefaultAnnotationPrefix, tt.mode, tt.smSharingEnabled)
			if err != nil {
				t.Fatalf("ParseComputePortion() unexpected error: %v", err)
			}
			if found != tt.wantFound || percent != tt.wantPercent {
				t.Errorf("ParseComputePortion() = (%d, %v), want (%d, %v)", percent, found, tt.wantPercent, tt.wantFound)
			}
		})
	}
}

// TestParseComputePortionReportsMalformedValuesInEveryMode: the exemptions
// above decide whether a cap is applied, not whether the annotation is checked.
// The value is scheduler-written, so one that cannot be parsed means something
// upstream is broken; swallowing it because this particular container would not
// have been capped anyway would hide the breakage until a container that WOULD
// be capped hit it.
func TestParseComputePortionReportsMalformedValuesInEveryMode(t *testing.T) {
	ann := map[string]string{computePortionTestKey: "not-a-number"}

	for _, mode := range []ComputeMode{ComputeModeTimeSlicing, ComputeModeSMSharing} {
		for _, enabled := range []bool{false, true} {
			if _, _, err := ParseComputePortion(ann, "trainer", configuration.DefaultAnnotationPrefix, mode, enabled); err == nil {
				t.Errorf("mode %q, smSharingEnabled %v: error = nil, want a parse failure", mode, enabled)
			}
		}
	}
}

// TestComputePercentNeverOversubscribesTheCard is the property the floor exists
// for: whatever set of portions the scheduler hands out, if they fit on one GPU
// then so do the percentages derived from them.
//
// It is a property test rather than a table because the failure it guards
// against is arithmetic, not a specific input: the old rounding was correct for
// halves and quarters and wrong for eighths, tenths and thirds, which is
// exactly the shape of bug a hand-written table finds last.
//
// The only allowance is the 1% floor, which can over-subscribe by at most 1 per
// container that asked for less than that. It is bounded, deliberate (0 means
// "unlimited" to MPS) and accounted for explicitly here rather than smuggled
// into the tolerance.
func TestComputePercentNeverOversubscribesTheCard(t *testing.T) {
	splits := [][]float64{
		{1},
		{0.5, 0.5},
		{0.25, 0.25, 0.25, 0.25},
		{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125},
		{1.0 / 3, 1.0 / 3, 1.0 / 3},
		{1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7, 1.0 / 7},
		{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		{0.7, 0.2, 0.1},
		{0.29, 0.31, 0.4},
		{0.9, 0.05, 0.05},
		{0.99, 0.01},
		// Portions that sum to less than a whole GPU must not be inflated to
		// more than they asked for either.
		{0.3, 0.3},
		{0.001, 0.002, 0.003},
	}

	for _, split := range splits {
		var portionSum float64
		total, floored := 0, 0
		for _, portion := range split {
			portionSum += portion
			percent := ComputePercent(portion)
			if percent < 1 || percent > 100 {
				t.Fatalf("ComputePercent(%v) = %d, outside the usable MPS range [1, 100]", portion, percent)
			}
			if float64(percent) > portion*100 {
				// Only the floor may exceed the portion, and only up to 1%.
				if percent != 1 {
					t.Errorf("ComputePercent(%v) = %d, above the portion without being the 1%% floor", portion, percent)
				}
				floored++
			}
			total += percent
		}

		if portionSum > 1.0000001 {
			t.Fatalf("test data error: %v sums to %v, which is more than one GPU", split, portionSum)
		}
		if total > 100+floored {
			t.Errorf("portions %v sum to %v of a GPU but their percentages sum to %d%% (allowing %d for the 1%% floor)",
				split, portionSum, total, floored)
		}
	}
}

// TestComputePercentIsMonotonic: a bigger portion must never yield a smaller
// percentage. A rounding rule that breaks this would let a container that was
// given more of a GPU be capped lower than one given less, which nothing
// downstream would ever flag.
func TestComputePercentIsMonotonic(t *testing.T) {
	previous := 0
	for step := 1; step <= 1000; step++ {
		portion := float64(step) / 1000
		percent := ComputePercent(portion)
		if percent < previous {
			t.Fatalf("ComputePercent(%v) = %d, below the previous %d", portion, percent, previous)
		}
		previous = percent
	}
	if previous != 100 {
		t.Errorf("ComputePercent(1) = %d, want 100", previous)
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
