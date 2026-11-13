// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package amd turns a container's GPU-memory annotations into a limit for the
// Linux dmem cgroup controller, which caps the VRAM of AMD GPUs (Linux 6.15+:
// amdgpu registers its VRAM as a dmem region from 6.15 on). amdgpu names each
// GPU's region "drm/<PCI address>/vram"; the runtime writes dmem.max on the
// container's cgroup, and the kernel refuses allocations past it.
//
// The annotations are the same as for NVIDIA GPUs:
//
//	<prefix><containerName>.gpu-memory.limit
//	<prefix><containerName>.gpus.devices = "0000:03:00.0[,0000:83:00.0...]"
//
// For AMD GPUs the device list holds the GPUs' PCI addresses; the limit applies
// to each listed GPU. The request annotation is not written to the cgroup:
// dmem.min only orders eviction, and is capped by the (unset) ancestors' values,
// so it would guarantee nothing.
package amd

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	// UnifiedMax is the cgroup v2 file holding the per-region VRAM limit.
	UnifiedMax = "dmem.max"

	bytesPerMiB = 1024 * 1024
	// maxLimitMiB is the largest limit whose byte count fits in an int64.
	maxLimitMiB = math.MaxInt64 / bytesPerMiB
)

var pciAddress = regexp.MustCompile(`^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$`)

// Devices parses the device-assignment annotation value: a comma-separated
// list of PCI addresses in domain:bus:device.function form. An empty value is
// an error: the dmem controller limits one GPU's VRAM region at a time, so the
// GPU must be named.
func Devices(value string) ([]string, error) {
	if strings.TrimSpace(value) == "" {
		return nil, fmt.Errorf("no GPU assigned: the AMD backend needs the PCI address of each GPU in the gpus.devices annotation")
	}
	var devices []string
	for _, d := range strings.Split(value, ",") {
		d = strings.ToLower(strings.TrimSpace(d))
		if !pciAddress.MatchString(d) {
			return nil, fmt.Errorf("gpus.devices: %q is not a PCI address (domain:bus:device.function, e.g. 0000:03:00.0)", d)
		}
		devices = append(devices, d)
	}
	return devices, nil
}

// Region is the dmem region amdgpu registers for the VRAM of the GPU at the
// given PCI address.
func Region(device string) string {
	return "drm/" + device + "/vram"
}

// MaxValue returns the dmem.max value for a limit in MiB (as parsed from the
// annotations): one "<region> <bytes>" line per GPU. The kernel takes one
// region per write; the runtime writes a multi-line value line by line (runc
// 1.2.0+, crun), which is what lets one container be capped on several GPUs.
func MaxValue(limitMiB string, devices []string) (string, error) {
	limit, err := strconv.ParseInt(limitMiB, 10, 64)
	if err != nil || limit <= 0 {
		return "", fmt.Errorf("invalid GPU memory limit %q MiB", limitMiB)
	}
	if limit > maxLimitMiB {
		return "", fmt.Errorf("GPU memory limit %d MiB is too large (at most %d MiB)", limit, int64(maxLimitMiB))
	}
	out := make([]string, len(devices))
	for i, d := range devices {
		out[i] = fmt.Sprintf("%s %d", Region(d), limit*bytesPerMiB)
	}
	return strings.Join(out, "\n"), nil
}

// ParseCapacity reads a dmem.capacity file (the root cgroup's list of regions
// and their sizes, one "<region> <bytes>" per line) into a map from region to
// size in bytes.
func ParseCapacity(r io.Reader) (map[string]int64, error) {
	regions := map[string]int64{}
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 2 {
			return nil, fmt.Errorf("dmem.capacity: unexpected line %q", scanner.Text())
		}
		size, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("dmem.capacity: unexpected size in %q", scanner.Text())
		}
		regions[fields[0]] = size
	}
	return regions, scanner.Err()
}

// VRAMRegions returns the GPU VRAM regions ("drm/<PCI address>/vram") among
// the given dmem regions, sorted.
func VRAMRegions(regions map[string]int64) []string {
	var out []string
	for r := range regions {
		if strings.HasPrefix(r, "drm/") && strings.HasSuffix(r, "/vram") {
			out = append(out, r)
		}
	}
	slices.Sort(out)
	return out
}
