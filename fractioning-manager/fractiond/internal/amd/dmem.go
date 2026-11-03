// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package amd turns a container's GPU-memory annotations into limits for the
// Linux dmem cgroup controller, which caps the VRAM of AMD GPUs (Linux 6.14+,
// amdgpu). amdgpu registers each GPU's VRAM as the region "drm/<PCI address>/vram";
// the runtime writes dmem.max (the limit) and dmem.min (the guaranteed request)
// on the container's cgroup, and the kernel refuses allocations past the limit.
//
// The annotations are the same as for NVIDIA GPUs:
//
//	<prefix><containerName>.gpu-memory.request
//	<prefix><containerName>.gpu-memory.limit
//	<prefix><containerName>.gpus.devices = "0000:03:00.0[,0000:83:00.0...]"
//
// For AMD GPUs the device list holds the GPUs' PCI addresses; the request and
// limit apply to each listed GPU.
package amd

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

const (
	// UnifiedMax is the cgroup v2 file holding the per-region VRAM limit.
	UnifiedMax = "dmem.max"
	// UnifiedMin is the cgroup v2 file holding the per-region VRAM guarantee.
	UnifiedMin = "dmem.min"

	bytesPerMiB = 1024 * 1024
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

// Limits returns the dmem.min and dmem.max values for the given request and
// limit in MiB (as parsed from the annotations), one line per GPU.
func Limits(requestMiB, limitMiB string, devices []string) (minValue, maxValue string, err error) {
	request, err := strconv.ParseInt(requestMiB, 10, 64)
	if err != nil || request < 0 {
		return "", "", fmt.Errorf("invalid GPU memory request %q MiB", requestMiB)
	}
	limit, err := strconv.ParseInt(limitMiB, 10, 64)
	if err != nil || limit <= 0 {
		return "", "", fmt.Errorf("invalid GPU memory limit %q MiB", limitMiB)
	}
	if request > limit {
		return "", "", fmt.Errorf("GPU memory request %d MiB is above the limit %d MiB", request, limit)
	}
	return lines(devices, request*bytesPerMiB), lines(devices, limit*bytesPerMiB), nil
}

func lines(devices []string, bytes int64) string {
	out := make([]string, len(devices))
	for i, d := range devices {
		out[i] = fmt.Sprintf("drm/%s/vram %d", d, bytes)
	}
	return strings.Join(out, "\n")
}
