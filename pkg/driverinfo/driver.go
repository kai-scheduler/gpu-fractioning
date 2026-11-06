// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package driverinfo defines gpu-fractioning-owned NVIDIA driver metadata.
package driverinfo

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	// NVIDIADriverMajorAnnotation is written by mpsd on its own pod after reading
	// the node-local driver version via NVML during startup. The operator reads
	// it from the mpsd pod of a node to explain a daemon that cannot start.
	NVIDIADriverMajorAnnotation = "gpu-fractioning.kai.scheduler/nvidia-driver-version.major"

	// NVIDIADriverMajorLabel is the same key as a node label, which releases up
	// to 0.1.8 had mpsd write with a cluster-wide nodes/patch grant. It is no
	// longer written; the operator still removes it on GpuFractioningConfig
	// deletion so upgraded clusters end up clean.
	NVIDIADriverMajorLabel = NVIDIADriverMajorAnnotation
)

// ParseDriverVersionMajor extracts the major branch from an NVML driver version
// string such as "615.43.02". NVIDIA driver versions look like dotted versions,
// but they are not a semver contract, so we only parse the major branch we need.
func ParseDriverVersionMajor(version string) (int, error) {
	version = strings.TrimSpace(version)
	if version == "" {
		return 0, fmt.Errorf("driver version is empty")
	}

	majorText, _, _ := strings.Cut(version, ".")
	return parsePositiveInt(majorText)
}

// ParseDriverMajorLabel parses the node label value, which must contain only the
// decimal driver major version.
func ParseDriverMajorLabel(value string) (int, error) {
	return parsePositiveInt(strings.TrimSpace(value))
}

func parsePositiveInt(value string) (int, error) {
	if value == "" {
		return 0, fmt.Errorf("value is empty")
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, err
	}
	if parsed <= 0 {
		return 0, fmt.Errorf("value must be positive")
	}
	return parsed, nil
}
