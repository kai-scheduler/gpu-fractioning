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

func TestMaxValue(t *testing.T) {
	devices := []string{"0000:03:00.0", "0000:83:00.0"}
	value, err := MaxValue("11516", devices)
	if err != nil {
		t.Fatal(err)
	}
	if want := "drm/0000:03:00.0/vram 12075401216\ndrm/0000:83:00.0/vram 12075401216"; value != want {
		t.Errorf("dmem.max = %q, want %q", value, want)
	}
	// The kernel takes exactly one region per write (dmemcg_limit_write); the
	// runtime splits the value by line. Each line must therefore be a whole
	// "<region> <bytes>" pair on its own.
	for i, line := range strings.Split(value, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 || fields[0] != Region(devices[i]) || strings.ContainsAny(line, "\t\r") {
			t.Errorf("line %d = %q, want %q followed by one byte count", i, line, Region(devices[i]))
		}
	}
	for _, tc := range []struct{ limit, err string }{
		{"x", "invalid GPU memory limit"},
		{"0", "invalid GPU memory limit"},
		{"-1", "invalid GPU memory limit"},
		{"8796093022208", "too large"}, // 2^43 MiB: the byte count would overflow int64
	} {
		if _, err := MaxValue(tc.limit, devices[:1]); err == nil || !strings.Contains(err.Error(), tc.err) {
			t.Errorf("MaxValue(%s): err = %v, want %q", tc.limit, err, tc.err)
		}
	}
	if _, err := MaxValue("8796093022207", devices[:1]); err != nil {
		t.Errorf("largest limit: %v", err)
	}
}

func TestParseCapacity(t *testing.T) {
	regions, err := ParseCapacity(strings.NewReader("drm/0000:83:00.0/vram 51522830336\ndrm/0000:03:00.0/vram 51522830336\ndrm/0000:00:02.0/vram0 4294967296\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if regions["drm/0000:03:00.0/vram"] != 51522830336 || len(regions) != 3 {
		t.Fatalf("regions = %v", regions)
	}
	if got := strings.Join(VRAMRegions(regions), ","); got != "drm/0000:03:00.0/vram,drm/0000:83:00.0/vram" {
		t.Errorf("VRAMRegions = %s", got)
	}
	for _, bad := range []string{"drm/0000:03:00.0/vram", "drm/0000:03:00.0/vram x"} {
		if _, err := ParseCapacity(strings.NewReader(bad)); err == nil {
			t.Errorf("ParseCapacity(%q): want an error", bad)
		}
	}
}
