// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package driverlabel

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/pkg/driverinfo"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

func TestPatchNodeDriverMajor(t *testing.T) {
	clientset := fake.NewSimpleClientset(&corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-a"},
	})

	err := patchNodeDriverMajorWithClient(context.Background(), clientset.CoreV1().Nodes(), "node-a", 615)
	if err != nil {
		t.Fatalf("PatchNodeDriverMajor() error = %v", err)
	}

	actions := clientset.Actions()
	if len(actions) != 1 {
		t.Fatalf("actions length = %d, expected 1: %v", len(actions), actions)
	}
	action, ok := actions[0].(k8stesting.PatchAction)
	if !ok {
		t.Fatalf("action type = %T, expected PatchAction", actions[0])
	}
	if action.GetPatchType() != types.MergePatchType {
		t.Fatalf("patch type = %q, expected %q", action.GetPatchType(), types.MergePatchType)
	}
	if action.GetResource().Resource != "nodes" {
		t.Fatalf("resource = %q, expected nodes", action.GetResource().Resource)
	}
	if action.GetName() != "node-a" {
		t.Fatalf("node name = %q, expected node-a", action.GetName())
	}

	var gotBody map[string]any
	if err := json.Unmarshal(action.GetPatch(), &gotBody); err != nil {
		t.Fatalf("decode patch body: %v", err)
	}
	metadata := gotBody["metadata"].(map[string]any)
	labels := metadata["labels"].(map[string]any)
	if got := labels[driverinfo.NVIDIADriverMajorLabel]; got != "615" {
		t.Fatalf("driver label = %v, expected 615", got)
	}
}

func TestLabelCurrentNodeRequiresNodeName(t *testing.T) {
	t.Setenv(envNodeName, "")
	err := LabelCurrentNode(context.Background(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err == nil {
		t.Fatal("LabelCurrentNode() error = nil, expected error")
	}
}

func TestDriverMajorLabelPatchRejectsInvalidMajor(t *testing.T) {
	if _, err := driverMajorLabelPatch(0); err == nil {
		t.Fatal("driverMajorLabelPatch(0) error = nil, expected error")
	}
}
