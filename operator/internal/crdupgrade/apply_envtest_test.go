// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package crdupgrade

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"

	"github.com/kai-scheduler/gpu-fractioning/operator/charts/crds"
)

const crdName = "gpufractioningconfigs.gpu-fractioning.kai.scheduler"

var testClient client.Client

func TestMain(m *testing.M) {
	// No CRDDirectoryPaths: installing the CRD is what is under test.
	env := &envtest.Environment{BinaryAssetsDirectory: envTestBinaryDir()}

	cfg, err := env.Start()
	if err != nil {
		fmt.Fprintf(os.Stderr, "start envtest: %v\nRun 'make setup-envtest' or use 'make test'.\n", err)
		os.Exit(1)
	}

	testClient, err = client.New(cfg, client.Options{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "build client: %v\n", err)
		os.Exit(1)
	}

	code := m.Run()
	_ = env.Stop()
	os.Exit(code)
}

func envTestBinaryDir() string {
	base := filepath.Join("..", "..", "bin", "k8s")
	entries, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if e.IsDir() {
			return filepath.Join(base, e.Name())
		}
	}
	return ""
}

// TestApply drives the hook Job's real path against an API server: first
// install, then the two cases an upgrade actually hits. Subtests share cluster
// state and must run in order.
func TestApply(t *testing.T) {
	ctx := context.Background()

	var installedRV string

	t.Run("installs the CRD and waits for it to be established", func(t *testing.T) {
		if err := Apply(ctx, testClient); err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		got := getCRD(t, ctx)
		if !isEstablished(got) {
			t.Error("CRD is not Established after Apply returned")
		}
		installedRV = got.GetResourceVersion()
	})

	t.Run("re-applying the same schema is a no-op", func(t *testing.T) {
		if err := Apply(ctx, testClient); err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		// A server-side apply that computes to the same object is not a write,
		// so the resourceVersion must not move. If this ever starts failing,
		// every helm upgrade is churning the CRD and bumping its generation.
		if rv := getCRD(t, ctx).GetResourceVersion(); rv != installedRV {
			t.Errorf("resourceVersion moved on an unchanged apply: %s -> %s", installedRV, rv)
		}
	})

	t.Run("applying a changed schema upgrades the CRD", func(t *testing.T) {
		objs, err := crds.Load()
		if err != nil {
			t.Fatalf("Load() error = %v", err)
		}
		upgraded := withExtraProperty(t, objs, "upgradeProbe")

		if err := applyObjects(ctx, testClient, upgraded); err != nil {
			t.Fatalf("applyObjects() error = %v", err)
		}

		got := getCRD(t, ctx)
		if rv := got.GetResourceVersion(); rv == installedRV {
			t.Errorf("resourceVersion unchanged at %s after a schema change", rv)
		}
		if !hasSpecProperty(t, got, "upgradeProbe") {
			t.Error("the new property is missing from the served schema")
		}
		if !isEstablished(got) {
			t.Error("CRD is not Established after the upgrade")
		}
	})
}

// TestApplyAdoptsHelmOwnedFields is the case the hook exists for: the CRD on
// the cluster was created by Helm, so its schema fields are owned by Helm's
// field manager. Without ForceOwnership every one comes back as a conflict.
func TestApplyAdoptsHelmOwnedFields(t *testing.T) {
	ctx := context.Background()

	objs, err := crds.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	// Delete and recreate under Helm's field manager to reproduce a fresh
	// `helm install`, which creates the CRD from the chart's crds/ directory.
	existing := getCRD(t, ctx)
	if err := testClient.Delete(ctx, existing); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	waitGone(t, ctx)

	helmOwned := objs[0].DeepCopy()
	if err := testClient.Create(ctx, helmOwned, client.FieldOwner("helm")); err != nil {
		t.Fatalf("Create() as helm error = %v", err)
	}

	if err := Apply(ctx, testClient); err != nil {
		t.Fatalf("Apply() over a Helm-owned CRD error = %v", err)
	}
	if !isEstablished(getCRD(t, ctx)) {
		t.Error("CRD is not Established after adoption")
	}
}

// waitGone blocks until the CRD is really gone: deletion is asynchronous, so a
// Create issued straight after a Delete races the finalizer.
func waitGone(t *testing.T, ctx context.Context) {
	t.Helper()
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion("apiextensions.k8s.io/v1")
	obj.SetKind("CustomResourceDefinition")

	err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, true,
		func(ctx context.Context) (bool, error) {
			err := testClient.Get(ctx, client.ObjectKey{Name: crdName}, obj)
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		})
	if err != nil {
		t.Fatalf("waiting for %s to be deleted: %v", crdName, err)
	}
}

func getCRD(t *testing.T, ctx context.Context) *unstructured.Unstructured {
	t.Helper()
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion("apiextensions.k8s.io/v1")
	obj.SetKind("CustomResourceDefinition")
	if err := testClient.Get(ctx, client.ObjectKey{Name: crdName}, obj); err != nil {
		t.Fatalf("Get(%s) error = %v", crdName, err)
	}
	return obj
}

// withExtraProperty returns the CRDs with an added string property under
// spec, standing in for a release that grows a GpuFractioningConfig field.
func withExtraProperty(t *testing.T, objs []*unstructured.Unstructured, name string) []*unstructured.Unstructured {
	t.Helper()
	out := make([]*unstructured.Unstructured, 0, len(objs))
	for _, obj := range objs {
		obj = obj.DeepCopy()
		versions, found, err := unstructured.NestedSlice(obj.Object, "spec", "versions")
		if err != nil || !found {
			t.Fatalf("%s: spec.versions not found: %v", obj.GetName(), err)
		}
		specProps(t, versions[0])[name] = map[string]any{"type": "string"}
		if err := unstructured.SetNestedSlice(obj.Object, versions, "spec", "versions"); err != nil {
			t.Fatalf("SetNestedSlice: %v", err)
		}
		out = append(out, obj)
	}
	return out
}

// specProps returns the live spec.properties map inside a CRD version, so the
// caller can mutate it in place.
func specProps(t *testing.T, version any) map[string]any {
	t.Helper()
	cur, ok := version.(map[string]any)
	if !ok {
		t.Fatalf("version is %T, want map", version)
	}
	for _, key := range []string{"schema", "openAPIV3Schema", "properties", "spec", "properties"} {
		next, ok := cur[key].(map[string]any)
		if !ok {
			t.Fatalf("%s is %T, want map", key, cur[key])
		}
		cur = next
	}
	return cur
}

func hasSpecProperty(t *testing.T, obj *unstructured.Unstructured, name string) bool {
	t.Helper()
	versions, found, err := unstructured.NestedSlice(obj.Object, "spec", "versions")
	if err != nil || !found {
		t.Fatalf("spec.versions not found: %v", err)
	}
	_, ok := specProps(t, versions[0])[name]
	return ok
}
