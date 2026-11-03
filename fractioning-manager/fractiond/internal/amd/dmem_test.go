// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package amd

import (
	"strings"
	"testing"
)

func TestDevices(t *testing.T) {
	for _, tc := range []struct {
		name, value string
		want        []string
		err         string
	}{
		{name: "one GPU", value: "0000:03:00.0", want: []string{"0000:03:00.0"}},
		{name: "two GPUs, spaces and upper case", value: " 0000:C8:00.0 , 0000:03:00.0", want: []string{"0000:c8:00.0", "0000:03:00.0"}},
		{name: "empty", value: " ", err: "no GPU assigned"},
		{name: "short address", value: "03:00.0", err: "not a PCI address"},
		{name: "NVIDIA UUID", value: "GPU-1234", err: "not a PCI address"},
		{name: "all", value: "all", err: "not a PCI address"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Devices(tc.value)
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("err = %v, want %q", err, tc.err)
				}
				return
			}
			if err != nil || strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("got %v, %v; want %v", got, err, tc.want)
			}
		})
	}
}

func TestLimits(t *testing.T) {
	devices := []string{"0000:03:00.0", "0000:83:00.0"}
	minValue, maxValue, err := Limits("2048", "11516", devices)
	if err != nil {
		t.Fatal(err)
	}
	if want := "drm/0000:03:00.0/vram 2147483648\ndrm/0000:83:00.0/vram 2147483648"; minValue != want {
		t.Errorf("dmem.min = %q, want %q", minValue, want)
	}
	if want := "drm/0000:03:00.0/vram 12075401216\ndrm/0000:83:00.0/vram 12075401216"; maxValue != want {
		t.Errorf("dmem.max = %q, want %q", maxValue, want)
	}
	for _, tc := range []struct{ request, limit, err string }{
		{"x", "1", "invalid GPU memory request"},
		{"1", "0", "invalid GPU memory limit"},
		{"8", "4", "above the limit"},
	} {
		if _, _, err := Limits(tc.request, tc.limit, devices[:1]); err == nil || !strings.Contains(err.Error(), tc.err) {
			t.Errorf("Limits(%s, %s): err = %v, want %q", tc.request, tc.limit, err, tc.err)
		}
	}
}
