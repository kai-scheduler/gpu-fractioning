// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package configcleanup

import (
	"context"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
)

const testFinalizer = "gpu-fractioning.nvidia.com/cleanup-node-conditions"

func newClient(objs ...client.Object) client.Client {
	scheme := runtime.NewScheme()
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

func config(name string, finalizers ...string) *v1alpha1.GpuFractioningConfig {
	return &v1alpha1.GpuFractioningConfig{
		ObjectMeta: metav1.ObjectMeta{Name: name, Finalizers: finalizers},
	}
}

func remaining(t *testing.T, c client.Client) []string {
	t.Helper()
	var list v1alpha1.GpuFractioningConfigList
	if err := c.List(context.Background(), &list); err != nil {
		t.Fatalf("list: %v", err)
	}
	names := make([]string, 0, len(list.Items))
	for i := range list.Items {
		names = append(names, list.Items[i].Name)
	}
	return names
}

func TestDeleteRemovesConfigsWithoutFinalizers(t *testing.T) {
	c := newClient(config("default"), config("second"))

	if err := Delete(context.Background(), c, time.Second); err != nil {
		t.Fatalf("Delete returned error: %v", err)
	}
	if left := remaining(t, c); len(left) != 0 {
		t.Fatalf("expected no GpuFractioningConfigs left, got %v", left)
	}
}

func TestDeleteWithNothingToDelete(t *testing.T) {
	if err := Delete(context.Background(), newClient(), time.Second); err != nil {
		t.Fatalf("Delete returned error: %v", err)
	}
}

// The fake client keeps an object with finalizers after Delete, with its
// deletionTimestamp set, until the last finalizer is removed, like the API
// server does. Play the operator: remove the finalizer once deletion starts.
func TestDeleteWaitsForTheFinalizer(t *testing.T) {
	c := newClient(config("default", testFinalizer))

	released := make(chan struct{})
	go func() {
		defer close(released)
		for {
			var cfg v1alpha1.GpuFractioningConfig
			if err := c.Get(context.Background(), client.ObjectKey{Name: "default"}, &cfg); err != nil {
				return
			}
			if !cfg.DeletionTimestamp.IsZero() {
				controllerutil.RemoveFinalizer(&cfg, testFinalizer)
				if err := c.Update(context.Background(), &cfg); err == nil {
					return
				}
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()

	if err := Delete(context.Background(), c, 30*time.Second); err != nil {
		t.Fatalf("Delete returned error: %v", err)
	}
	<-released
	if left := remaining(t, c); len(left) != 0 {
		t.Fatalf("expected no GpuFractioningConfigs left, got %v", left)
	}
}

func TestDeleteReportsWhatIsLeftOnTimeout(t *testing.T) {
	c := newClient(config("default", testFinalizer))

	err := Delete(context.Background(), c, 200*time.Millisecond)
	if err == nil {
		t.Fatal("expected a timeout error, got nil")
	}
	if !strings.Contains(err.Error(), "default") || !strings.Contains(err.Error(), "finalizer") {
		t.Fatalf("error should name the remaining object and the finalizer, got: %v", err)
	}
	if left := remaining(t, c); len(left) != 1 {
		t.Fatalf("expected the object to remain, got %v", left)
	}
}
