// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package annotations

import (
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
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
