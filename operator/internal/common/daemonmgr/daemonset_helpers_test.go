// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"path/filepath"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// testDefaults mirrors what every managed daemon passes in: CPU, memory and
// ephemeral-storage requests plus a memory limit, and deliberately no CPU limit.
func testDefaults() corev1.ResourceRequirements {
	return DaemonResources("50m", "128Mi", "1Gi")
}

func quantityString(t *testing.T, list corev1.ResourceList, name corev1.ResourceName) string {
	t.Helper()
	q, ok := list[name]
	if !ok {
		return ""
	}
	return q.String()
}

// assertDefaultRequestsIntact is the check that matters most for any override:
// the three requests decide the pod's QoS class and what the scheduler reserves.
// Losing them turns a Burstable daemon into BestEffort, which makes it the first
// thing the kubelet evicts under node pressure — on the very node whose GPU
// fractioning it is responsible for.
func assertDefaultRequestsIntact(t *testing.T, got corev1.ResourceRequirements) {
	t.Helper()
	for name, want := range map[corev1.ResourceName]string{
		corev1.ResourceCPU:              "50m",
		corev1.ResourceMemory:           "128Mi",
		corev1.ResourceEphemeralStorage: daemonEphemeralStorageRequest,
	} {
		if got := quantityString(t, got.Requests, name); got != want {
			t.Errorf("requests[%s] = %q, want %q (an override must never drop a default request)", name, got, want)
		}
	}
}

func TestResolveDaemonResources_NilOverrideKeepsDefaults(t *testing.T) {
	got := ResolveDaemonResources(testDefaults(), nil)

	assertDefaultRequestsIntact(t, got)
	if got := quantityString(t, got.Limits, corev1.ResourceMemory); got != "1Gi" {
		t.Errorf("limits.memory = %q, want %q", got, "1Gi")
	}
	// No CPU limit: these are latency-sensitive privileged daemons and CFS
	// throttling one of them stalls MPS client setup for every GPU workload.
	if _, ok := got.Limits[corev1.ResourceCPU]; ok {
		t.Errorf("limits = %v, want no CPU limit", got.Limits)
	}
}

// A CRD block written but left empty (`resources: {}`) reaches the operator as a
// non-nil, all-zero override. It must be indistinguishable from no override at
// all; treating it as "the user asked for nothing" would strip every request.
func TestResolveDaemonResources_EmptyOverrideKeepsDefaults(t *testing.T) {
	for _, tt := range []struct {
		name     string
		override *corev1.ResourceRequirements
	}{
		{name: "zero struct", override: &corev1.ResourceRequirements{}},
		{
			name: "empty but non-nil lists",
			override: &corev1.ResourceRequirements{
				Requests: corev1.ResourceList{},
				Limits:   corev1.ResourceList{},
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := ResolveDaemonResources(testDefaults(), tt.override)

			assertDefaultRequestsIntact(t, got)
			if got := quantityString(t, got.Limits, corev1.ResourceMemory); got != "1Gi" {
				t.Errorf("limits.memory = %q, want %q", got, "1Gi")
			}
		})
	}
}

// The headline use case: "my node has more GPUs than the default was sized for,
// raise limits.memory". This is a one-line override in the CR, and if it were a
// replace rather than a merge the pod would come back with no requests at all.
func TestResolveDaemonResources_MemoryLimitOnlyOverrideKeepsRequests(t *testing.T) {
	got := ResolveDaemonResources(testDefaults(), &corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
	})

	assertDefaultRequestsIntact(t, got)
	if got := quantityString(t, got.Limits, corev1.ResourceMemory); got != "4Gi" {
		t.Errorf("limits.memory = %q, want %q", got, "4Gi")
	}
}

// The mirror image: raising a request must not silently drop the memory limit,
// which is the only thing standing between a leaking daemon and the node.
func TestResolveDaemonResources_RequestsOnlyOverrideKeepsLimit(t *testing.T) {
	got := ResolveDaemonResources(testDefaults(), &corev1.ResourceRequirements{
		Requests: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("512Mi")},
	})

	if got := quantityString(t, got.Limits, corev1.ResourceMemory); got != "1Gi" {
		t.Errorf("limits.memory = %q, want %q; a requests-only override dropped the default limit", got, "1Gi")
	}
	if got := quantityString(t, got.Requests, corev1.ResourceMemory); got != "512Mi" {
		t.Errorf("requests.memory = %q, want %q", got, "512Mi")
	}
	// The entries the override did not mention keep their defaults.
	if got := quantityString(t, got.Requests, corev1.ResourceCPU); got != "50m" {
		t.Errorf("requests.cpu = %q, want %q", got, "50m")
	}
	if got := quantityString(t, got.Requests, corev1.ResourceEphemeralStorage); got != daemonEphemeralStorageRequest {
		t.Errorf("requests.ephemeral-storage = %q, want %q", got, daemonEphemeralStorageRequest)
	}
}

// An override may introduce a resource the defaults never mention — a CPU limit
// is the interesting one, because the defaults omit it on purpose. Asking for it
// explicitly must be honoured rather than filtered out against the default set.
func TestResolveDaemonResources_OverrideAddsNewResourceNames(t *testing.T) {
	got := ResolveDaemonResources(testDefaults(), &corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:              resource.MustParse("2"),
			corev1.ResourceEphemeralStorage: resource.MustParse("2Gi"),
		},
	})

	if got := quantityString(t, got.Limits, corev1.ResourceCPU); got != "2" {
		t.Errorf("limits.cpu = %q, want %q", got, "2")
	}
	if got := quantityString(t, got.Limits, corev1.ResourceEphemeralStorage); got != "2Gi" {
		t.Errorf("limits.ephemeral-storage = %q, want %q", got, "2Gi")
	}
	// Adding names must not disturb the one limit the defaults do set.
	if got := quantityString(t, got.Limits, corev1.ResourceMemory); got != "1Gi" {
		t.Errorf("limits.memory = %q, want %q", got, "1Gi")
	}
	assertDefaultRequestsIntact(t, got)
}

// Defaults with no limits at all must still accept an override; the merge has to
// allocate rather than write into a nil map.
func TestResolveDaemonResources_OverrideOntoEmptyDefaults(t *testing.T) {
	got := ResolveDaemonResources(corev1.ResourceRequirements{}, &corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("2Gi")},
	})

	if got := quantityString(t, got.Limits, corev1.ResourceMemory); got != "2Gi" {
		t.Errorf("limits.memory = %q, want %q", got, "2Gi")
	}
	// Nothing on either side, so the list stays nil rather than becoming an
	// empty map: an empty `requests: {}` in the rendered DaemonSet is a spec
	// difference the reconciler would keep re-applying.
	if got.Requests != nil {
		t.Errorf("requests = %v, want nil when neither defaults nor override set any", got.Requests)
	}
}

// Every daemon in a single reconcile resolves against the same built-in default
// values. If the resolver hands back the caller's maps instead of copies, the
// first daemon's override leaks into the second one's container spec — and the
// symptom is an mpsd sized by whatever fractiond's CR happened to ask for.
func TestResolveDaemonResources_ReturnsFreshMaps(t *testing.T) {
	for _, tt := range []struct {
		name     string
		override *corev1.ResourceRequirements
	}{
		{name: "nil override", override: nil},
		{name: "empty override", override: &corev1.ResourceRequirements{}},
		{
			name: "memory limit override",
			override: &corev1.ResourceRequirements{
				Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			defaults := testDefaults()

			got := ResolveDaemonResources(defaults, tt.override)
			got.Limits[corev1.ResourceMemory] = resource.MustParse("999Gi")
			got.Requests[corev1.ResourceCPU] = resource.MustParse("31")
			delete(got.Requests, corev1.ResourceEphemeralStorage)

			if got := quantityString(t, defaults.Limits, corev1.ResourceMemory); got != "1Gi" {
				t.Errorf("defaults.limits.memory = %q after mutating the result, want %q", got, "1Gi")
			}
			if got := quantityString(t, defaults.Requests, corev1.ResourceCPU); got != "50m" {
				t.Errorf("defaults.requests.cpu = %q after mutating the result, want %q", got, "50m")
			}
			if _, ok := defaults.Requests[corev1.ResourceEphemeralStorage]; !ok {
				t.Error("defaults.requests lost ephemeral-storage after a delete on the result")
			}
		})
	}
}

// The override the caller passes in is the CR's own spec, still owned by the
// cached GpuFractioningConfig object. Writing through it would mutate the
// informer cache and make the next reconcile see resources the user never set.
func TestResolveDaemonResources_DoesNotMutateOverride(t *testing.T) {
	override := &corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
	}

	got := ResolveDaemonResources(testDefaults(), override)
	got.Limits[corev1.ResourceCPU] = resource.MustParse("8")

	if _, ok := override.Limits[corev1.ResourceCPU]; ok {
		t.Errorf("override.limits gained cpu = %v; the CR spec was mutated in place", override.Limits)
	}
	if len(override.Requests) != 0 {
		t.Errorf("override.requests = %v, want it left empty", override.Requests)
	}
}

// DefaultMPSDrainSocketPath is pinned to its literal value rather than compared
// against the constant it aliases, because every other assertion in the repo
// derives its expectation from that same constant — so they all move together
// and not one of them notices a change. Including the "cross-daemon agreement"
// test, whose two sides both read this constant.
//
// The value has to be pinned because it is NOT single-sourced across the repo,
// despite what the daemonpaths package doc says: the daemon binaries get their
// default from fractioning-manager/common/mpsdrain.DefaultSocketPath, which
// repeats the same literal instead of importing daemonpaths. Change one side
// only and the failure is completely silent — fractiond dials a socket mpsd
// never created, every StopContainer drain fails to connect, the container is
// stopped anyway by design, and a client killed mid-kernel wedges the MPS server
// for every other tenant of the GPU with nothing in the logs pointing at the
// cause. That is the exact scenario this whole feature exists to prevent.
func TestDefaultMPSDrainSocketPath_IsPinned(t *testing.T) {
	// Keep in sync with fractioning-manager/common/mpsdrain.DefaultSocketPath.
	const want = "/var/run/gpu-fractioning/drain/mpsd.sock"
	if DefaultMPSDrainSocketPath != want {
		t.Fatalf("DefaultMPSDrainSocketPath = %q, want %q; if this change is intentional, update "+
			"fractioning-manager/common/mpsdrain.DefaultSocketPath in the same commit or the drain silently stops happening",
			DefaultMPSDrainSocketPath, want)
	}
}

// The drain socket is mounted by directory, so the directory is what both
// daemons get DirectoryOrCreate access to inside a privileged pod. A default
// that shares a directory with anything else on the host would hand that pod
// far more of the node than the drain endpoint needs.
func TestDefaultMPSDrainSocketPath_LivesInItsOwnDirectory(t *testing.T) {
	if !filepath.IsAbs(DefaultMPSDrainSocketPath) {
		t.Fatalf("DefaultMPSDrainSocketPath = %q, want an absolute host path", DefaultMPSDrainSocketPath)
	}

	dir := filepath.Dir(DefaultMPSDrainSocketPath)
	for _, shared := range []string{"/", "/run", "/var", "/var/run", "/tmp", "/var/run/gpu-fractioning"} {
		if dir == shared {
			t.Fatalf("drain socket directory = %q, which is a shared host directory; a DirectoryOrCreate hostPath there exposes far more than the socket", dir)
		}
	}
}
