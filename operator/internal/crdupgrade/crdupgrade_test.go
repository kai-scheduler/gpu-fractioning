// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package crdupgrade

import (
	"slices"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	"github.com/kai-scheduler/gpu-fractioning/operator/charts/crds"
)

// wantCRDs is the exact set the hook Job is allowed to apply. Asserted as a set
// rather than a contains-check: the Job holds cluster-wide create on CRDs, so a
// stray manifest landing in crds/ would be applied to every user's cluster.
var wantCRDs = []string{"gpufractioningconfigs.gpu-fractioning.kai.scheduler"}

func TestLoadEmbedsExactlyTheChartCRDs(t *testing.T) {
	objs, err := crds.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	var got []string
	for _, obj := range objs {
		got = append(got, obj.GetName())
		if k := obj.GetKind(); k != "CustomResourceDefinition" {
			t.Errorf("%s: kind = %q, want CustomResourceDefinition", obj.GetName(), k)
		}
		if v := obj.GetAPIVersion(); v != "apiextensions.k8s.io/v1" {
			t.Errorf("%s: apiVersion = %q, want apiextensions.k8s.io/v1", obj.GetName(), v)
		}
		// The RBAC granted to the hook Job is pinned to these names, so an
		// unlisted CRD would fail at apply time on a real cluster anyway.
		if !slices.Contains(wantCRDs, obj.GetName()) {
			t.Errorf("unexpected CRD %q embedded; add it to wantCRDs and to the hook's ClusterRole resourceNames", obj.GetName())
		}
	}

	slices.Sort(got)
	want := slices.Clone(wantCRDs)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Errorf("Load() returned %v, want %v", got, want)
	}
}

// A server-side apply owns every field it sends, and status is the API
// server's to write.
func TestLoadOmitsStatus(t *testing.T) {
	objs, err := crds.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	for _, obj := range objs {
		if _, found, _ := unstructured.NestedMap(obj.Object, "status"); found {
			t.Errorf("%s: parsed object carries a status block; it must not be applied", obj.GetName())
		}
	}
}

func TestIsEstablished(t *testing.T) {
	conds := func(cs ...map[string]any) *unstructured.Unstructured {
		obj := &unstructured.Unstructured{Object: map[string]any{}}
		if cs != nil {
			s := make([]any, 0, len(cs))
			for _, c := range cs {
				s = append(s, c)
			}
			if err := unstructured.SetNestedSlice(obj.Object, s, "status", "conditions"); err != nil {
				t.Fatalf("SetNestedSlice: %v", err)
			}
		}
		return obj
	}

	tests := []struct {
		name string
		obj  *unstructured.Unstructured
		want bool
	}{
		{
			name: "established",
			obj:  conds(map[string]any{"type": "Established", "status": "True"}),
			want: true,
		},
		{
			name: "not yet established",
			obj:  conds(map[string]any{"type": "Established", "status": "False"}),
			want: false,
		},
		{
			name: "another condition is true",
			obj:  conds(map[string]any{"type": "NamesAccepted", "status": "True"}),
			want: false,
		},
		{
			name: "established alongside others",
			obj: conds(
				map[string]any{"type": "NamesAccepted", "status": "True"},
				map[string]any{"type": "Established", "status": "True"},
			),
			want: true,
		},
		{
			name: "no status at all",
			obj:  conds(),
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isEstablished(tt.obj); got != tt.want {
				t.Errorf("isEstablished() = %v, want %v", got, tt.want)
			}
		})
	}
}
