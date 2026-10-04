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

// SetManagementCDIDevice requests a CDI device for the named container by
// annotating the pod template — how a daemon container gets GPU access when the
// GPU Operator runs its NRI plugin, since the nvidia RuntimeClass is deleted in
// that mode.
//
// device is BuildOptions.ManagementCDIDevice: empty means the GPU Operator is
// not in NRI mode and the pod template is left untouched. The annotation is
// merged rather than assigned, because fractiond already carries its
// prometheus.io/* scrape config here.
func SetManagementCDIDevice(podMeta *metav1.ObjectMeta, containerName, device string) {
	if device == "" {
		return
	}
	if podMeta.Annotations == nil {
		podMeta.Annotations = map[string]string{}
	}
	podMeta.Annotations[managementCDIDeviceAnnotation(containerName)] = device
}

// managementCDIDeviceAnnotation returns the pod annotation key that requests a
// CDI device for the named container. The NRI plugin resolves it per container,
// so the container name is part of the key rather than the value.
func managementCDIDeviceAnnotation(containerName string) string {
	return fmt.Sprintf("%s/container.%s", nriAnnotationDomain, containerName)
}
