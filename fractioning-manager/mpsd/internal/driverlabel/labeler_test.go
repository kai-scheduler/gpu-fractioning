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

func TestPatchPodDriverMajor(t *testing.T) {
	clientset := fake.NewSimpleClientset(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "mpsd-abc", Namespace: "gpu-fractioning"},
	})

	err := patchPodDriverMajorWithClient(context.Background(), clientset.CoreV1().Pods("gpu-fractioning"), "mpsd-abc", 615)
	if err != nil {
		t.Fatalf("patchPodDriverMajor() error = %v", err)
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
	if action.GetResource().Resource != "pods" {
		t.Fatalf("resource = %q, expected pods", action.GetResource().Resource)
	}
	if action.GetNamespace() != "gpu-fractioning" || action.GetName() != "mpsd-abc" {
		t.Fatalf("target = %s/%s, expected gpu-fractioning/mpsd-abc", action.GetNamespace(), action.GetName())
	}

	var gotBody map[string]any
	if err := json.Unmarshal(action.GetPatch(), &gotBody); err != nil {
		t.Fatalf("decode patch body: %v", err)
	}
	metadata := gotBody["metadata"].(map[string]any)
	if _, found := metadata["labels"]; found {
		t.Fatalf("patch touches labels: %v", metadata)
	}
	annotations := metadata["annotations"].(map[string]any)
	if got := annotations[driverinfo.NVIDIADriverMajorAnnotation]; got != "615" {
		t.Fatalf("driver annotation = %v, expected 615", got)
	}
}

func TestRecordDriverMajorRequiresPodIdentity(t *testing.T) {
	for name, env := range map[string]map[string]string{
		"no name":      {envPodName: "", envPodNamespace: "gpu-fractioning"},
		"no namespace": {envPodName: "mpsd-abc", envPodNamespace: ""},
	} {
		t.Run(name, func(t *testing.T) {
			for key, value := range env {
				t.Setenv(key, value)
			}
			_, err := RecordDriverMajor(context.Background(), slog.New(slog.NewTextHandler(io.Discard, nil)))
			if err == nil {
				t.Fatal("RecordDriverMajor() error = nil, expected error")
			}
		})
	}
}

func TestDriverMajorAnnotationPatchRejectsInvalidMajor(t *testing.T) {
	if _, err := driverMajorAnnotationPatch(0); err == nil {
		t.Fatal("driverMajorAnnotationPatch(0) error = nil, expected error")
	}
}
