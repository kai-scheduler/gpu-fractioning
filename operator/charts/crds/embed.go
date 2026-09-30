// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package crds embeds the chart's generated CRDs so the crd-upgrader hook Job
// can apply them from inside the cluster.
//
// It lives in the chart because Go embeds only from its own package directory.
// Helm collects only .yaml/.yml/.json from crds/, so this file is inert there.
package crds

import (
	"embed"
	"fmt"
	"sort"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"
)

//go:embed *.yaml
var manifests embed.FS

const crdKind = "CustomResourceDefinition"

// Load parses the embedded manifests, ordered by filename.
//
// Unstructured, not the typed CRD: a server-side apply owns every field it
// sends, and the typed struct would add its zero values (an empty status, a
// null creationTimestamp) to that set.
func Load() ([]*unstructured.Unstructured, error) {
	entries, err := manifests.ReadDir(".")
	if err != nil {
		return nil, fmt.Errorf("read embedded crds directory: %w", err)
	}

	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)

	objs := make([]*unstructured.Unstructured, 0, len(names))
	for _, name := range names {
		data, err := manifests.ReadFile(name)
		if err != nil {
			return nil, fmt.Errorf("read embedded CRD %s: %w", name, err)
		}

		obj := &unstructured.Unstructured{}
		if err := yaml.Unmarshal(data, obj); err != nil {
			return nil, fmt.Errorf("parse embedded CRD %s: %w", name, err)
		}
		if obj.GetKind() != crdKind {
			return nil, fmt.Errorf("embedded manifest %s is a %s, want %s", name, obj.GetKind(), crdKind)
		}
		if obj.GetName() == "" {
			return nil, fmt.Errorf("embedded CRD %s has no metadata.name", name)
		}
		objs = append(objs, obj)
	}

	if len(objs) == 0 {
		return nil, fmt.Errorf("no CRD manifests embedded")
	}
	return objs, nil
}
