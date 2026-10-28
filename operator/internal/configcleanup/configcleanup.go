// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package configcleanup deletes every GpuFractioningConfig and waits until the
// operator has run its finalizer and the objects are gone.
//
// Helm deletes a release's resources in kind order with custom resources last,
// so by the time it would delete the CR the operator Deployment is already gone
// and the finalizer would never be removed. The chart runs this as a pre-delete
// hook Job instead, while the operator is still running.
package configcleanup

import (
	"context"
	"fmt"
	"strings"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
)

// DefaultTimeout bounds the wait for the operator's cleanup. Removing node
// conditions and labels is a few API calls per node, so this only covers an
// operator that is not running at all.
const DefaultTimeout = 5 * time.Minute

const pollInterval = 2 * time.Second

// Run deletes the GpuFractioningConfigs in the cluster this pod runs in.
func Run(ctx context.Context, timeout time.Duration) error {
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

	return Delete(ctx, c, timeout)
}

// Delete requests deletion of every GpuFractioningConfig and waits until none
// is left. Objects whose deletion is already in progress are waited for too.
func Delete(ctx context.Context, c client.Client, timeout time.Duration) error {
	log := ctrl.LoggerFrom(ctx)

	var list v1alpha1.GpuFractioningConfigList
	if err := c.List(ctx, &list); err != nil {
		return fmt.Errorf("list GpuFractioningConfigs: %w", err)
	}
	for i := range list.Items {
		item := &list.Items[i]
		if !item.DeletionTimestamp.IsZero() {
			log.Info("GpuFractioningConfig deletion already in progress", "name", item.Name)
			continue
		}
		if err := c.Delete(ctx, item); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete GpuFractioningConfig %q: %w", item.Name, err)
		}
		log.Info("requested GpuFractioningConfig deletion", "name", item.Name)
	}

	var remaining []string
	err := wait.PollUntilContextTimeout(ctx, pollInterval, timeout, true, func(ctx context.Context) (bool, error) {
		var list v1alpha1.GpuFractioningConfigList
		if err := c.List(ctx, &list); err != nil {
			return false, err
		}
		remaining = remaining[:0]
		for i := range list.Items {
			remaining = append(remaining, list.Items[i].Name)
		}
		return len(remaining) == 0, nil
	})
	switch {
	case err == nil:
		log.Info("all GpuFractioningConfigs deleted")
		return nil
	case ctx.Err() != nil:
		return ctx.Err()
	case wait.Interrupted(err):
		return fmt.Errorf("GpuFractioningConfig(s) %s still exist after %s; the operator must be running to finish node cleanup and remove its finalizer",
			strings.Join(remaining, ", "), timeout)
	default:
		return fmt.Errorf("wait for GpuFractioningConfig deletion: %w", err)
	}
}
