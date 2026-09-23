// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package gpudevices identifies the GPUs assigned to a container from its Linux
// device nodes (NVIDIA char major 195), which are authoritative for device
// identity. It is the single, environment-independent detector used everywhere:
// making a container look like it has GPUs is the job of the cluster/runtime that
// injects the device nodes, not of this package.
package gpudevices

import (
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"

	"github.com/containerd/nri/pkg/api"
)

const (
	// nvidiaDeviceMajor is the Linux character device major number assigned to
	// NVIDIA GPUs (https://www.kernel.org/doc/Documentation/admin-guide/devices.txt).
	nvidiaDeviceMajor = 195

	// maxNVIDIAGPUMinor is the highest minor device number the driver assigns to
	// a physical GPU. Minor 0–N map to GPU indices; minors above this threshold
	// are MIG instances or control devices (e.g. nvidiactl = 255) and are skipped.
	maxNVIDIAGPUMinor = 32
)

// FromContainer returns the GPU devices assigned to the container via Linux
// device nodes. Returns nil when no NVIDIA device nodes are present.
func FromContainer(container *api.Container) []store.GPUDevice {
	linux := container.GetLinux()
	if linux == nil {
		return nil
	}

	devices := []store.GPUDevice{}
	seen := map[int]struct{}{}
	for _, device := range linux.GetDevices() {
		index, ok := gpuIndexFromLinuxDevice(device)
		if !ok {
			continue
		}
		if _, exists := seen[index]; exists {
			continue
		}
		seen[index] = struct{}{}
		devices = append(devices, store.GPUDevice{MinorNumber: index})
	}
	return devices
}

func gpuIndexFromLinuxDevice(device *api.LinuxDevice) (int, bool) {
	if device.GetMajor() != nvidiaDeviceMajor {
		return 0, false
	}
	minor := device.GetMinor()
	if minor < 0 || minor > maxNVIDIAGPUMinor {
		return 0, false
	}
	return int(minor), true
}
