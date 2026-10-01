// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package crdupgrade applies the CRDs embedded in the operator image.
//
// It exists because Helm never updates a chart's crds/ on upgrade, so the
// chart runs this as a pre-install/pre-upgrade hook Job instead.
package crdupgrade

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/operator/charts/crds"
)

const fieldOwner = client.FieldOwner("gpu-fractioning-crd-upgrader")

// Generous: a CRD is Established in well under a second, so this only covers a
// badly overloaded API server.
const establishedTimeout = 2 * time.Minute

// Run applies the embedded CRDs against the cluster this pod runs in.
func Run(ctx context.Context) error {
	cfg, err := ctrl.GetConfig()
	if err != nil {
		return fmt.Errorf("load kubeconfig: %w", err)
	}

	c, err := client.New(cfg, client.Options{})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	return Apply(ctx, c)
}

// Apply server-side applies the embedded CRDs and waits for each to be
// Established.
func Apply(ctx context.Context, c client.Client) error {
	objs, err := crds.Load()
	if err != nil {
		return err
	}
	return applyObjects(ctx, c, objs)
}

func applyObjects(ctx context.Context, c client.Client, objs []*unstructured.Unstructured) error {
	log := ctrl.LoggerFrom(ctx)

	for _, obj := range objs {
		name := obj.GetName()

		// ForceOwnership: on the first upgrade the schema fields are still
		// owned by Helm's field manager, so every one would be a conflict.
		if err := c.Apply(ctx, client.ApplyConfigurationFromUnstructured(obj), fieldOwner, client.ForceOwnership); err != nil {
			return fmt.Errorf("apply CRD %s: %w", name, err)
		}
		log.Info("applied CRD", "name", name)

		if err := waitEstablished(ctx, c, name); err != nil {
			return err
		}
		log.Info("CRD established", "name", name)
	}

	return nil
}

// waitEstablished blocks until the API server serves the kind, so the chart can
// apply a GpuFractioningConfig straight after this hook.
//
// This only really waits on a first install. On an upgrade the CRD is already
// Established and returns on the first poll: CRD conditions carry no
// observedGeneration unless the CRDObservedGenerationTracking feature gate is
// on, so there is nothing to tell us the new schema has been picked up.
func waitEstablished(ctx context.Context, c client.Client, name string) error {
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion("apiextensions.k8s.io/v1")
	obj.SetKind("CustomResourceDefinition")

	err := wait.PollUntilContextTimeout(ctx, time.Second, establishedTimeout, true,
		func(ctx context.Context) (bool, error) {
			if err := c.Get(ctx, client.ObjectKey{Name: name}, obj); err != nil {
				return false, err
			}
			// A name clash with another CRD never resolves, and its message
			// names what clashed. Without this it surfaces as a bare timeout
			// two minutes later with the reason sitting unread in the object.
			if cond, ok := condition(obj, "NamesAccepted"); ok && cond["status"] == "False" {
				return false, fmt.Errorf("names rejected: %v", cond["message"])
			}
			return isEstablished(obj), nil
		})
	if err != nil {
		return fmt.Errorf("wait for CRD %s to be established: %w", name, err)
	}
	return nil
}

func condition(obj *unstructured.Unstructured, want string) (map[string]any, bool) {
	conds, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil || !found {
		return nil, false
	}
	for _, c := range conds {
		cond, ok := c.(map[string]any)
		if ok && cond["type"] == want {
			return cond, true
		}
	}
	return nil, false
}

func isEstablished(obj *unstructured.Unstructured) bool {
	cond, ok := condition(obj, "Established")
	return ok && cond["status"] == "True"
}
