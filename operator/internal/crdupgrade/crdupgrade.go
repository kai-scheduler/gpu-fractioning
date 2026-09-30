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

// waitEstablished blocks until the API server serves the kind. The chart
// applies a GpuFractioningConfig immediately after this hook returns.
func waitEstablished(ctx context.Context, c client.Client, name string) error {
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion("apiextensions.k8s.io/v1")
	obj.SetKind("CustomResourceDefinition")

	err := wait.PollUntilContextTimeout(ctx, time.Second, establishedTimeout, true,
		func(ctx context.Context) (bool, error) {
			if err := c.Get(ctx, client.ObjectKey{Name: name}, obj); err != nil {
				return false, err
			}
			return isEstablished(obj), nil
		})
	if err != nil {
		return fmt.Errorf("wait for CRD %s to be established: %w", name, err)
	}
	return nil
}

func isEstablished(obj *unstructured.Unstructured) bool {
	conds, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil || !found {
		return false
	}
	for _, c := range conds {
		cond, ok := c.(map[string]any)
		if !ok {
			continue
		}
		if cond["type"] == "Established" && cond["status"] == "True" {
			return true
		}
	}
	return false
}
