// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package driverinfo

import "testing"

func TestParseDriverVersionMajor(t *testing.T) {
	tests := []struct {
		name      string
		version   string
		want      int
		wantError bool
	}{
		{name: "full driver version", version: "615.43.02", want: 615},
		{name: "major only", version: "615", want: 615},
		{name: "trim whitespace", version: " 620.1 ", want: 620},
		{name: "empty", version: "", wantError: true},
		{name: "missing major", version: ".615", wantError: true},
		{name: "non-numeric major", version: "r615.43.02", wantError: true},
		{name: "zero", version: "0.0", wantError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseDriverVersionMajor(tt.version)
			if tt.wantError {
				if err == nil {
					t.Fatalf("ParseDriverVersionMajor(%q) error = nil, expected error", tt.version)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseDriverVersionMajor(%q) error = %v", tt.version, err)
			}
			if got != tt.want {
				t.Fatalf("ParseDriverVersionMajor(%q) = %d, expected %d", tt.version, got, tt.want)
			}
		})
	}
}

func TestParseDriverMajorLabel(t *testing.T) {
	tests := []struct {
		name      string
		value     string
		want      int
		wantError bool
	}{
		{name: "major", value: "615", want: 615},
		{name: "trim whitespace", value: " 620 ", want: 620},
		{name: "empty", value: "", wantError: true},
		{name: "full driver version is invalid as label value", value: "615.43.02", wantError: true},
		{name: "zero", value: "0", wantError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseDriverMajorLabel(tt.value)
			if tt.wantError {
				if err == nil {
					t.Fatalf("ParseDriverMajorLabel(%q) error = nil, expected error", tt.value)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseDriverMajorLabel(%q) error = %v", tt.value, err)
			}
			if got != tt.want {
				t.Fatalf("ParseDriverMajorLabel(%q) = %d, expected %d", tt.value, got, tt.want)
			}
		})
	}
}

func TestCheckSupportedMajor(t *testing.T) {
	for _, tc := range []struct {
		major   int
		wantErr bool
	}{
		{major: 580, wantErr: true},
		{major: MinimumSupportedMajor - 1, wantErr: true},
		{major: MinimumSupportedMajor, wantErr: false},
		{major: 620, wantErr: false},
	} {
		err := CheckSupportedMajor(tc.major)
		if (err != nil) != tc.wantErr {
			t.Errorf("CheckSupportedMajor(%d) error = %v, wantErr %t", tc.major, err, tc.wantErr)
		}
	}
}
