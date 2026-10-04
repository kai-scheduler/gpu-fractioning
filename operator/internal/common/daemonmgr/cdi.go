// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// nriAnnotationDomain is the annotation domain the NVIDIA Container
	// Toolkit's NRI plugin watches for CDI device requests. It matches the
	// constant of the same name in the GPU Operator, which annotates its own
	// GPU-touching operands this way.
	nriAnnotationDomain = "nvidia.cdi.k8s.io"

	// ManagementCDIDeviceAll is the CDI device that grants a container the same
	// GPU access the GPU Operator gives its own management containers. It is
	// what the nvidia RuntimeClass provided before NRI mode, where the
	// RuntimeClass no longer exists.
	ManagementCDIDeviceAll = "management.nvidia.com/gpu=all"
)

// SetManagementCDIDevice requests the management CDI device for the named
// container by annotating the pod template — how a daemon container gets GPU
// access when the GPU Operator runs its NRI plugin, since the nvidia
// RuntimeClass is deleted in that mode.
//
// nriPluginEnabled is BuildOptions.NRIPluginEnabled: false leaves the pod
// template untouched. The device is not a parameter because there is only one —
// ManagementCDIDeviceAll — so passing it would just be a second way to say the
// same thing, and a way for a caller to say it wrongly. The annotation is
// merged rather than assigned, because fractiond already carries its
// prometheus.io/* scrape config here.
func SetManagementCDIDevice(podMeta *metav1.ObjectMeta, containerName string, nriPluginEnabled bool) {
	if !nriPluginEnabled {
		return
	}
	if podMeta.Annotations == nil {
		podMeta.Annotations = map[string]string{}
	}
	podMeta.Annotations[managementCDIDeviceAnnotation(containerName)] = ManagementCDIDeviceAll
}

// managementCDIDeviceAnnotation returns the pod annotation key that requests a
// CDI device for the named container. The NRI plugin resolves it per container,
// so the container name is part of the key rather than the value.
func managementCDIDeviceAnnotation(containerName string) string {
	return fmt.Sprintf("%s/container.%s", nriAnnotationDomain, containerName)
}
