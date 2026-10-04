// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// nriManagementCDIDeviceNamespacesEnv is the NVIDIA Container Toolkit env var
// listing the namespaces whose pods may request management CDI devices. It is
// set through the ClusterPolicy at spec.toolkit.env and read by the toolkit's
// NRI plugin. Only the GPU Operator's own namespace is permitted by default, so
// daemons installed elsewhere need an admin to extend it.
const nriManagementCDIDeviceNamespacesEnv = "NRI_MANAGEMENT_CDI_DEVICE_NAMESPACES"

var clusterPolicyGVK = schema.GroupVersionKind{
	Group:   "nvidia.com",
	Version: "v1",
	Kind:    "ClusterPolicy",
}

// ClusterPolicy is the outcome of the single per-reconcile read of the NVIDIA
// GPU Operator's ClusterPolicy: the object, its legitimate absence, or a
// failure to read it. Its methods answer what the rest of the reconcile needs
// to know about the GPU stack.
//
// It is a value rather than a plain return pair because two consumers need the
// same read — the DaemonSet builders, to decide how GPUs are injected into the
// daemon pods, and the dependency checker — and passing it keeps that to one
// API call per reconcile.
type ClusterPolicy struct {
	// Policy is nil when the GPU Operator exposes no ClusterPolicy: the CRD is
	// not installed, or an OpenShift install that reports through the OLM
	// ClusterServiceVersion instead.
	Policy *unstructured.Unstructured

	// Err is set when the read itself failed, which is distinct from a
	// ClusterPolicy that is legitimately absent. Carried rather than returned so
	// the reconcile can proceed and surface it on the Ready condition in the
	// same place it always has.
	Err error
}

// ReadClusterPolicy reads the NVIDIA GPU Operator ClusterPolicy. A missing CRD
// or no ClusterPolicy at all is not an error — both leave Policy nil.
func ReadClusterPolicy(ctx context.Context, reader client.Reader) ClusterPolicy {
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(clusterPolicyGVK.GroupVersion().WithKind(clusterPolicyGVK.Kind + "List"))

	if err := reader.List(ctx, list); err != nil {
		if meta.IsNoMatchError(err) || apierrors.IsNotFound(err) {
			return ClusterPolicy{}
		}
		return ClusterPolicy{Err: err}
	}

	if len(list.Items) == 0 {
		return ClusterPolicy{}
	}
	return ClusterPolicy{Policy: &list.Items[0]}
}

// NRIPluginEnabled reports whether the GPU Operator injects GPUs into its
// management containers through the NRI plugin rather than the nvidia
// RuntimeClass.
//
// This mirrors CDIConfigSpec.IsNRIPluginEnabled upstream, including its default:
// an unset spec.cdi.nriPluginEnabled means disabled. An absent or unreadable
// ClusterPolicy also reports false, which keeps the daemons on the RuntimeClass
// path — the behaviour that predates NRI support. A read failure is reported
// separately on the Ready condition, so it is not silently swallowed here.
func (p ClusterPolicy) NRIPluginEnabled() bool {
	if p.Policy == nil {
		return false
	}
	enabled, found, err := unstructured.NestedBool(p.Policy.Object, "spec", "cdi", "nriPluginEnabled")
	if err != nil || !found {
		return false
	}
	return enabled
}

// ManagementCDIDeviceNamespaces returns the namespaces the NVIDIA Container
// Toolkit permits to request management CDI devices, read from the toolkit's
// NRI_MANAGEMENT_CDI_DEVICE_NAMESPACES env var.
//
// The value is a comma-separated list. Entries are trimmed and empties dropped,
// so a trailing comma or padded list does not produce a namespace that can
// never match. The returned list does not include the GPU Operator's own
// namespace, which the toolkit permits implicitly.
func (p ClusterPolicy) ManagementCDIDeviceNamespaces() []string {
	if p.Policy == nil {
		return nil
	}
	env, found, err := unstructured.NestedSlice(p.Policy.Object, "spec", "toolkit", "env")
	if err != nil || !found {
		return nil
	}

	var namespaces []string
	for _, item := range env {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name, _, _ := unstructured.NestedString(entry, "name")
		if name != nriManagementCDIDeviceNamespacesEnv {
			continue
		}
		value, _, _ := unstructured.NestedString(entry, "value")
		for _, namespace := range strings.Split(value, ",") {
			if namespace = strings.TrimSpace(namespace); namespace != "" {
				namespaces = append(namespaces, namespace)
			}
		}
	}
	return namespaces
}
