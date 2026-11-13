// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package configadopt

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
)

var release = Release{Name: "gpu-fractioning", Namespace: "gpu-fractioning"}

func newClient(objs ...client.Object) client.Client {
	scheme := runtime.NewScheme()
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

// hookCreated is the default CR as every release up to 0.1.8 left it: hook
// annotations only, no ownership metadata, and a UID the upgrade must keep.
func hookCreated() *v1alpha1.GpuFractioningConfig {
	return &v1alpha1.GpuFractioningConfig{ObjectMeta: metav1.ObjectMeta{
		Name: "default",
		UID:  types.UID("keep-me"),
		Annotations: map[string]string{
			hookAnnotation:       "post-install,post-upgrade",
			hookWeightAnnotation: "0",
			"other":              "stays",
		},
		Labels: map[string]string{"app.kubernetes.io/name": "gpu-fractioning"},
	}}
}

func get(t *testing.T, c client.Client) *v1alpha1.GpuFractioningConfig {
	t.Helper()
	var cfg v1alpha1.GpuFractioningConfig
	if err := c.Get(context.Background(), client.ObjectKey{Name: "default"}, &cfg); err != nil {
		t.Fatalf("get: %v", err)
	}
	return &cfg
}

func TestAdoptStampsOwnershipAndDropsHookAnnotations(t *testing.T) {
	c := newClient(hookCreated())

	if err := Adopt(context.Background(), c, "default", release); err != nil {
		t.Fatalf("Adopt returned error: %v", err)
	}

	cfg := get(t, c)
	if cfg.UID != types.UID("keep-me") {
		t.Fatalf("UID changed to %q", cfg.UID)
	}
	if cfg.Labels[managedByLabel] != managedByHelm || cfg.Labels["app.kubernetes.io/name"] != "gpu-fractioning" {
		t.Fatalf("unexpected labels: %v", cfg.Labels)
	}
	if cfg.Annotations[releaseNameAnnotation] != release.Name || cfg.Annotations[releaseNamespaceAnnotation] != release.Namespace {
		t.Fatalf("ownership annotations not set: %v", cfg.Annotations)
	}
	if _, found := cfg.Annotations[hookAnnotation]; found {
		t.Fatalf("hook annotation still present: %v", cfg.Annotations)
	}
	if _, found := cfg.Annotations[hookWeightAnnotation]; found {
		t.Fatalf("hook weight annotation still present: %v", cfg.Annotations)
	}
	if cfg.Annotations["other"] != "stays" {
		t.Fatalf("unrelated annotation lost: %v", cfg.Annotations)
	}
}

func TestAdoptIsANoOpWhenAlreadyOwned(t *testing.T) {
	owned := hookCreated()
	owned.Annotations = map[string]string{releaseNameAnnotation: release.Name, releaseNamespaceAnnotation: release.Namespace}
	owned.Labels[managedByLabel] = managedByHelm
	c := newClient(owned)
	before := get(t, c)

	if err := Adopt(context.Background(), c, "default", release); err != nil {
		t.Fatalf("Adopt returned error: %v", err)
	}
	if after := get(t, c); after.ResourceVersion != before.ResourceVersion {
		t.Fatalf("object was written although already owned")
	}
}

func TestAdoptWithNothingToAdopt(t *testing.T) {
	if err := Adopt(context.Background(), newClient(), "default", release); err != nil {
		t.Fatalf("Adopt returned error: %v", err)
	}
}

func TestAdoptRefusesAnotherReleasesObject(t *testing.T) {
	foreign := hookCreated()
	foreign.Annotations[releaseNameAnnotation] = "someone-else"
	c := newClient(foreign)

	if err := Adopt(context.Background(), c, "default", release); err == nil {
		t.Fatal("expected an error for an object owned by another release")
	}
	if cfg := get(t, c); cfg.Annotations[releaseNameAnnotation] != "someone-else" {
		t.Fatalf("foreign object was modified: %v", cfg.Annotations)
	}
}
