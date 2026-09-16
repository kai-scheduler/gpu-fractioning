// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package annotations

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	// Per-container GPU-fractioning annotation keys have the shape
	// <prefix><containerName>.<...>, where <prefix> is the configurable leading
	// prefix (configuration.DefaultAnnotationPrefix = "nvidia.com/container."). The
	// scheduler emits, for a container named "trainer":
	//
	//	nvidia.com/container.trainer.gpu-memory.request
	//	nvidia.com/container.trainer.gpu-memory.limit
	//	nvidia.com/container.trainer.gpus.devices
	//
	// memoryInfix sits between the container name and the request/limit suffix;
	// devicesSuffix is the trailing segment of the device-assignment key.
	memoryInfix   = "gpu-memory"
	devicesSuffix = "gpus.devices"

	// Annotation key suffixes appended after the memory infix.
	annotationSuffixRequest = "request"
	annotationSuffixLimit   = "limit"

	// computeModeSuffix builds the per-container compute-mode annotation key,
	// e.g. "nvidia.com/container.trainer.gpu-compute.mode". Unlike memoryInfix
	// (reused across the request/limit suffixes), compute mode has only one
	// key shape, so infix and suffix collapse into a single constant here —
	// matching the devicesSuffix pattern above.
	computeModeSuffix = "gpu-compute.mode"

	// bytesPerMiB is the number of bytes in one MiB.
	bytesPerMiB = 1024 * 1024

	// Minimum accepted annotation value, matching the documented minimum of 1 MiB.
	minMemoryBytes = bytesPerMiB
)

// ComputeMode selects how a fractional GPU container shares the node's GPU
// compute among neighbors, via the per-container
// "<prefix><containerName>.gpu-compute.mode" annotation.
type ComputeMode string

const (
	// ComputeModeTimeSlicing shares compute via GPU time-slicing (the default
	// driver scheduling across processes) — today's only behavior, and the
	// value ParseComputeMode returns when the annotation is absent.
	ComputeModeTimeSlicing ComputeMode = "time-slicing"
	// ComputeModeSMSharing shares compute via NVIDIA MPS context-sharing
	// (concurrent SM occupancy), routing the container to the shared MPS
	// server instead of the default one. Only meaningful for a container that
	// also carries a GPU-memory annotation.
	ComputeModeSMSharing ComputeMode = "sm-sharing"
)

// GPUMemoryConfig holds the parsed GPU memory request and limit for a container.
// Values are integer MiB strings suitable for NVIDIA MPS environment variables.
// Empty string means the annotation was not present.
//
// This intentionally does NOT reuse resource.Quantity or a k8s resource struct:
// the source data comes from pod annotations, not from a k8s resource spec, and
// the values are already normalized to the integer MiB value NVIDIA consumes.
type GPUMemoryConfig struct {
	Request string
	Limit   string
}

// IsEmpty returns true if neither request nor limit was specified.
func (c GPUMemoryConfig) IsEmpty() bool {
	return c.Request == "" && c.Limit == ""
}

// ApplyDefaults fills a missing request or limit from the other so a container
// that specified only one of the two ends up with both, and with request ==
// limit. This makes enforcement symmetric with the whole-GPU path:
//
//   - request only: the limit defaults to the request, so the memory cap
//     (NVIDIA_GPU_MEMORY_LIMIT) is enforced at the requested size instead of
//     being unbounded.
//   - limit only: the request defaults to the limit, so the requested size
//     (NVIDIA_GPU_MEMORY_REQUEST, used for metrics/fraction accounting) is
//     populated.
//
// A config with neither set (IsEmpty) is returned unchanged.
func (c GPUMemoryConfig) ApplyDefaults() GPUMemoryConfig {
	if c.Request == "" {
		c.Request = c.Limit
	}
	if c.Limit == "" {
		c.Limit = c.Request
	}
	return c
}

// EffectiveMemoryMiB returns the container's allocated GPU memory in MiB: the
// limit if set, otherwise the request, otherwise 0. Request and Limit are already
// MiB strings (see parseToMemoryMiB), so this just parses one back to an
// integer. It is the numerator the metrics sidecar uses to derive the GPU
// fraction (requested memory ÷ device total memory) for SM-util normalization.
func (c GPUMemoryConfig) EffectiveMemoryMiB() int64 {
	value := c.Limit
	if value == "" {
		value = c.Request
	}
	if value == "" {
		return 0
	}
	mib, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0
	}
	return mib
}

// ParseGPUMemoryAnnotations extracts GPU memory configuration for a specific
// container from the pod's annotation map.
//
// Annotation format:
//
//	<prefix><containerName>.gpu-memory.request = "<quantity>"
//	<prefix><containerName>.gpu-memory.limit   = "<quantity>"
//
// With the default prefix "nvidia.com/container.", a full key is:
//
//	nvidia.com/container.trainer.gpu-memory.request
//	nvidia.com/container.trainer.gpu-memory.limit
//
// Values must be valid Kubernetes Quantities (e.g. "4Gi", "2048Mi", "4096M").
// Returns an error if an annotation exists but cannot be parsed, or if the
// resolved value is less than 1 MiB.
func ParseGPUMemoryAnnotations(annotations map[string]string, containerName, prefix string) (GPUMemoryConfig, error) {
	config := GPUMemoryConfig{}

	requestKey := containerMemoryAnnotationKey(prefix, containerName, annotationSuffixRequest)
	limitKey := containerMemoryAnnotationKey(prefix, containerName, annotationSuffixLimit)

	if val, ok := annotations[requestKey]; ok {
		mib, err := parseToMemoryMiB(val)
		if err != nil {
			return config, fmt.Errorf("parsing annotation %q: %w", requestKey, err)
		}
		config.Request = mib
	}

	if val, ok := annotations[limitKey]; ok {
		mib, err := parseToMemoryMiB(val)
		if err != nil {
			return config, fmt.Errorf("parsing annotation %q: %w", limitKey, err)
		}
		config.Limit = mib
	}

	return config, nil
}

// ParseVisibleDevices returns the GPU device assignment the scheduler recorded
// for a specific container — the per-container key
// "<prefix><containerName>.gpus.devices" (e.g.
// "nvidia.com/container.trainer.gpus.devices") — trimmed of surrounding
// whitespace. The value is a comma-separated list of NVIDIA GPU UUIDs (or
// indices), e.g. "GPU-abc123,GPU-def456". A fractional container does not request
// the nvidia.com/gpu resource, so the NVIDIA device plugin never injects
// NVIDIA_VISIBLE_DEVICES; fractiond reads this annotation and injects it (as
// injection.EnvVisibleDevices) so the container sees exactly the GPU the
// scheduler picked. It returns "" when the annotation is absent or blank, which
// the caller treats as "no assignment to inject" (so a container that already
// carries NVIDIA_VISIBLE_DEVICES is left untouched). The value is passed through
// verbatim otherwise — its format (UUID list, index list, or "all") is the
// NVIDIA container runtime's contract, not this operator's.
func ParseVisibleDevices(annotations map[string]string, containerName, prefix string) string {
	return strings.TrimSpace(annotations[containerDevicesAnnotationKey(prefix, containerName)])
}

// ParseComputeMode extracts the compute-mode selection for a specific
// container from the pod's annotation map.
//
// Annotation format:
//
//	<prefix><containerName>.gpu-compute.mode = "time-slicing" | "sm-sharing"
//
// With the default prefix "nvidia.com/container.", a full key is:
//
//	nvidia.com/container.trainer.gpu-compute.mode
//
// Returns ComputeModeTimeSlicing when the annotation is absent (the default,
// matching today's only behavior), the parsed mode when the value is exactly
// "time-slicing" or, when smSharingEnabled is true, "sm-sharing"; an error for
// any other value, including a present-but-blank one.
//
// smSharingEnabled is the cluster's installation-time sm-sharing chicken bit
// (Helm value -> operator -> fractiond --support-sm-sharing flag). When false,
// "sm-sharing" is rejected exactly like any other invalid value: the shared
// MPS server backing it is disabled cluster-wide by the same toggle, so
// honoring the annotation would route the container to a socket that does not
// exist on the node.
func ParseComputeMode(annotations map[string]string, containerName, prefix string, smSharingEnabled bool) (ComputeMode, error) {
	key := containerComputeModeAnnotationKey(prefix, containerName)
	raw, ok := annotations[key]
	if !ok {
		return ComputeModeTimeSlicing, nil
	}

	// Present but blank is a value, not an absence: the caller asked for a mode
	// and named none, so it falls through to the same error as any other
	// unsupported value rather than quietly meaning the default.
	switch val := strings.TrimSpace(raw); ComputeMode(val) {
	case ComputeModeTimeSlicing:
		return ComputeModeTimeSlicing, nil
	case ComputeModeSMSharing:
		if !smSharingEnabled {
			return "", fmt.Errorf("parsing annotation %q: value %q is not enabled on this cluster (sm-sharing support is disabled)",
				key, val)
		}
		return ComputeModeSMSharing, nil
	default:
		return "", fmt.Errorf("parsing annotation %q: invalid value %q, expected %q or %q",
			key, val, ComputeModeTimeSlicing, ComputeModeSMSharing)
	}
}

// quantityToMemoryMiB converts a Kubernetes Quantity string to the integer MiB
// value NVIDIA consumes from the injected memory environment variables.
func quantityToMemoryMiB(value string) (int64, error) {
	qty, err := resource.ParseQuantity(value)
	if err != nil {
		return 0, fmt.Errorf("invalid quantity %q: %w", value, err)
	}

	bytes := qty.Value()
	if bytes < 0 {
		return 0, fmt.Errorf("negative quantity: %s", value)
	}
	if bytes < minMemoryBytes {
		return 0, fmt.Errorf("quantity %q resolves to %d bytes, minimum is 1 MiB", value, bytes)
	}

	// NVIDIA interprets the injected memory environment values as whole MiB.
	return ceilDiv(bytes, bytesPerMiB), nil
}

// parseToMemoryMiB converts a Kubernetes Quantity string to the integer MiB
// string injected into NVIDIA memory environment variables.
func parseToMemoryMiB(value string) (string, error) {
	mib, err := quantityToMemoryMiB(value)
	if err != nil {
		return "", err
	}

	return strconv.FormatInt(mib, 10), nil
}

func ceilDiv(value, divisor int64) int64 {
	return 1 + (value-1)/divisor
}

// containerMemoryAnnotationKey builds the full annotation key for a container's
// GPU memory setting.
// Example: containerMemoryAnnotationKey("nvidia.com/container.", "trainer", "limit")
//
//	→ "nvidia.com/container.trainer.gpu-memory.limit"
func containerMemoryAnnotationKey(prefix, containerName, suffix string) string {
	return prefix + containerName + "." + memoryInfix + "." + suffix
}

// containerDevicesAnnotationKey builds the per-container GPU device-assignment
// annotation key.
// Example: containerDevicesAnnotationKey("nvidia.com/container.", "trainer")
//
//	→ "nvidia.com/container.trainer.gpus.devices"
func containerDevicesAnnotationKey(prefix, containerName string) string {
	return prefix + containerName + "." + devicesSuffix
}

// containerComputeModeAnnotationKey builds the per-container compute-mode
// annotation key.
// Example: containerComputeModeAnnotationKey("nvidia.com/container.", "trainer")
//
//	→ "nvidia.com/container.trainer.gpu-compute.mode"
func containerComputeModeAnnotationKey(prefix, containerName string) string {
	return prefix + containerName + "." + computeModeSuffix
}

// RequestAnnotationKey returns the annotation key fractiond looks up for a
// container's GPU-memory request (e.g. for diagnostic logging).
func RequestAnnotationKey(prefix, containerName string) string {
	return containerMemoryAnnotationKey(prefix, containerName, annotationSuffixRequest)
}

// DevicesAnnotationKey returns the annotation key fractiond looks up for a
// container's GPU device assignment.
func DevicesAnnotationKey(prefix, containerName string) string {
	return containerDevicesAnnotationKey(prefix, containerName)
}

// LimitAnnotationKey returns the annotation key fractiond looks up for a
// container's GPU-memory limit (e.g. for diagnostic logging of the annotation a
// non-fractioning container is missing).
func LimitAnnotationKey(prefix, containerName string) string {
	return containerMemoryAnnotationKey(prefix, containerName, annotationSuffixLimit)
}

// ComputeModeAnnotationKey returns the annotation key fractiond looks up for a
// container's compute-mode selection (e.g. for diagnostic logging).
func ComputeModeAnnotationKey(prefix, containerName string) string {
	return containerComputeModeAnnotationKey(prefix, containerName)
}
