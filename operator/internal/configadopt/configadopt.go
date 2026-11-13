// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package configadopt gives an existing GpuFractioningConfig the ownership
// metadata Helm requires before it manages a resource it did not create.
//
// Releases up to 0.1.8 created the default CR as a post-install hook. Helm never
// stamps hook objects with its ownership annotations and never records them in
// the release manifest, so once the CR is an ordinary release resource the first
// upgrade finds an object that "exists and cannot be imported into the current
// release" and fails. The chart runs this as a pre-upgrade hook Job, so the
// upgrade that follows takes the CR over in place: same UID, same DaemonSets.
package configadopt

import (
	"context"
	"encoding/json"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
)

// Helm checks exactly these three before it imports an existing object into a
// release (pkg/action/validate.go, requireAdoption).
const (
	managedByLabel             = "app.kubernetes.io/managed-by"
	managedByHelm              = "Helm"
	releaseNameAnnotation      = "meta.helm.sh/release-name"
	releaseNamespaceAnnotation = "meta.helm.sh/release-namespace"

	hookAnnotation       = "helm.sh/hook"
	hookWeightAnnotation = "helm.sh/hook-weight"
)

// Release identifies the Helm release that is about to manage the object.
type Release struct {
	Name      string
	Namespace string
}

// Run adopts the named GpuFractioningConfig in the cluster this pod runs in.
func Run(ctx context.Context, name string, release Release) error {
	cfg, err := ctrl.GetConfig()
	if err != nil {
		return fmt.Errorf("load kubeconfig: %w", err)
	}

	scheme := runtime.NewScheme()
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	c, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	return Adopt(ctx, c, name, release)
}

// Adopt stamps the named GpuFractioningConfig with the release's ownership
// metadata and drops the hook annotations a previous chart left on it. A
// missing object is not an error: a fresh install has nothing to adopt. An
// object already owned by another release is refused rather than taken over.
func Adopt(ctx context.Context, c client.Client, name string, release Release) error {
	log := ctrl.LoggerFrom(ctx)

	var config v1alpha1.GpuFractioningConfig
	if err := c.Get(ctx, client.ObjectKey{Name: name}, &config); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("no GpuFractioningConfig to adopt", "name", name)
			return nil
		}
		return fmt.Errorf("get GpuFractioningConfig %q: %w", name, err)
	}

	if owner := config.Annotations[releaseNameAnnotation]; owner != "" && owner != release.Name {
		return fmt.Errorf("GpuFractioningConfig %q belongs to release %q, not %q", name, owner, release.Name)
	}
	if config.Labels[managedByLabel] == managedByHelm &&
		config.Annotations[releaseNameAnnotation] == release.Name &&
		config.Annotations[releaseNamespaceAnnotation] == release.Namespace &&
		config.Annotations[hookAnnotation] == "" {
		log.Info("GpuFractioningConfig already owned by the release", "name", name, "release", release.Name)
		return nil
	}

	patch, err := json.Marshal(map[string]any{
		"metadata": map[string]any{
			"labels": map[string]any{managedByLabel: managedByHelm},
			"annotations": map[string]any{
				releaseNameAnnotation:      release.Name,
				releaseNamespaceAnnotation: release.Namespace,
				hookAnnotation:             nil,
				hookWeightAnnotation:       nil,
			},
		},
	})
	if err != nil {
		return err
	}
	if err := c.Patch(ctx, &config, client.RawPatch(types.MergePatchType, patch)); err != nil {
		return fmt.Errorf("patch GpuFractioningConfig %q: %w", name, err)
	}
	log.Info("adopted GpuFractioningConfig into the release", "name", name, "release", release.Name, "namespace", release.Namespace)
	return nil
}
