// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package gpufractioningconfig provides typed CRUD and status helpers for the
// cluster-scoped GpuFractioningConfig custom resource, so the operator e2e suite can
// drive the CR's lifecycle (create/patch/delete) and read its status conditions
// the same way the operator's own reconciler does.
//
// The controller-runtime client in k8s/cluster is built with client-go's global
// scheme.Scheme (core + apps only), so this package registers the gpu-fractioning
// v1alpha1 types into that scheme in init(). Importing this package (which the
// operator suite does, via its fixtures) is therefore enough to make typed CR
// access work through the shared cluster client.
package gpufractioningconfig

import (
	"context"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
)

func init() {
	// Register GpuFractioningConfig into the global scheme the cluster client uses,
	// so typed Get/Create/Patch/Delete resolve. Must run before cluster.NewClient.
	runtime.Must(v1alpha1.AddToScheme(scheme.Scheme))
}

// DefaultName is the only name the CRD's CEL rule accepts for a GpuFractioningConfig.
const DefaultName = "default"

// Get returns the default GpuFractioningConfig.
func Get(ctx context.Context, c *cluster.Client) (*v1alpha1.GpuFractioningConfig, error) {
	return getByName(ctx, c, DefaultName)
}

// getByName returns the named GpuFractioningConfig (cluster-scoped). The CRD's CEL
// rule allows only "default", so this stays unexported — callers use Get.
func getByName(ctx context.Context, c *cluster.Client, name string) (*v1alpha1.GpuFractioningConfig, error) {
	var out v1alpha1.GpuFractioningConfig
	if err := c.Ctrl.Get(ctx, ctrlclient.ObjectKey{Name: name}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Create creates the given GpuFractioningConfig object.
func Create(ctx context.Context, c *cluster.Client, obj *v1alpha1.GpuFractioningConfig) error {
	return c.Ctrl.Create(ctx, obj)
}

// CreateDryRun runs a server-side dry-run create: the API server validates the
// object (schema + CEL) but persists nothing. Used by validation cases (e.g. a
// CR named other than "default") so they can assert rejection without mutating
// the live default CR.
func CreateDryRun(ctx context.Context, c *cluster.Client, obj *v1alpha1.GpuFractioningConfig) error {
	return c.Ctrl.Create(ctx, obj, ctrlclient.DryRunAll)
}

// CreateUnstructuredDryRun dry-run-creates an arbitrary (possibly invalid)
// GpuFractioningConfig expressed as a spec map. It exists for schema-negative cases
// that the typed struct can't express — e.g. omitting the required nodeSelector
// entirely (the typed field always serializes). Returns the API server's
// validation error.
func CreateUnstructuredDryRun(ctx context.Context, c *cluster.Client, name string, spec map[string]any) error {
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(v1alpha1.SchemeGroupVersion.WithKind("GpuFractioningConfig"))
	u.SetName(name)
	if err := unstructured.SetNestedField(u.Object, spec, "spec"); err != nil {
		return fmt.Errorf("build unstructured spec: %w", err)
	}
	return c.Ctrl.Create(ctx, u, ctrlclient.DryRunAll)
}

// Patch applies a JSON merge patch to the named CR. A field set to null in the
// patch is deleted (JSON-merge semantics), which config cases use to revert an
// override back to the shipped default.
func Patch(ctx context.Context, c *cluster.Client, name string, mergePatch []byte) error {
	obj := &v1alpha1.GpuFractioningConfig{ObjectMeta: metav1.ObjectMeta{Name: name}}
	return c.Ctrl.Patch(ctx, obj, ctrlclient.RawPatch(types.MergePatchType, mergePatch))
}

// PatchDryRun runs a server-side dry-run JSON-merge patch: the API server
// validates the change (schema + CEL — e.g. the nodeSelector immutability rule)
// but persists nothing. Used by validation cases so they can assert a patch is
// rejected without risking the live default CR (FX-STEADY) if the rule regressed.
func PatchDryRun(ctx context.Context, c *cluster.Client, name string, mergePatch []byte) error {
	obj := &v1alpha1.GpuFractioningConfig{ObjectMeta: metav1.ObjectMeta{Name: name}}
	return c.Ctrl.Patch(ctx, obj, ctrlclient.RawPatch(types.MergePatchType, mergePatch), ctrlclient.DryRunAll)
}

// Delete removes the named CR. A missing CR is treated as success.
func Delete(ctx context.Context, c *cluster.Client, name string) error {
	obj := &v1alpha1.GpuFractioningConfig{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if err := c.Ctrl.Delete(ctx, obj); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// WaitGone polls until the named CR no longer exists.
func WaitGone(ctx context.Context, c *cluster.Client, name string, timeout, interval time.Duration) error {
	return waiter.PollUntil(ctx, timeout, interval, fmt.Sprintf("GpuFractioningConfig/%s to be deleted", name),
		func(ctx context.Context) (bool, error) {
			_, err := getByName(ctx, c, name)
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		})
}

// Condition returns the named status condition and whether it is present.
func Condition(obj *v1alpha1.GpuFractioningConfig, condType string) (metav1.Condition, bool) {
	for _, cond := range obj.Status.Conditions {
		if cond.Type == condType {
			return cond, true
		}
	}
	return metav1.Condition{}, false
}

// WaitCondition polls the named CR until its condType condition reaches status,
// returning the CR observed at that point. observedGeneration is required to be
// current so a stale condition from a prior generation doesn't satisfy the wait.
func WaitCondition(ctx context.Context, c *cluster.Client, name, condType string, status metav1.ConditionStatus, timeout, interval time.Duration) (*v1alpha1.GpuFractioningConfig, error) {
	var latest *v1alpha1.GpuFractioningConfig
	err := waiter.PollUntil(ctx, timeout, interval,
		fmt.Sprintf("GpuFractioningConfig/%s condition %s=%s", name, condType, status),
		func(ctx context.Context) (bool, error) {
			obj, err := getByName(ctx, c, name)
			if err != nil {
				return false, err
			}
			latest = obj
			if obj.Status.ObservedGeneration != obj.Generation {
				return false, nil
			}
			cond, ok := Condition(obj, condType)
			return ok && cond.Status == status, nil
		})
	if err != nil {
		return latest, err
	}
	return latest, nil
}
