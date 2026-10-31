// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package driverlabel

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/kai-scheduler/gpu-fractioning/pkg/toolkitinfo"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func TestParseToolkitVersion(t *testing.T) {
	for _, tt := range []struct{ output, want string }{
		{"NVIDIA Container Toolkit CLI version 1.19.1\ncommit: abc\n", "1.19.1"},
		{"NVIDIA Container Toolkit CLI version v1.20.1\n", "1.20.1"},
		{"NVIDIA Container Toolkit CLI version 1.20.1-rc.1", "1.20.1-rc.1"},
		{"NVIDIA Container Toolkit CLI version invalid", ""},
		{"unrelated binary version 1.20.1", ""},
		{"", ""},
	} {
		t.Run(tt.output, func(t *testing.T) {
			got, err := parseToolkitVersion(tt.output)
			if got != tt.want || (err != nil) != (tt.want == "") {
				t.Fatalf("parseToolkitVersion() = %q, %v; want %q", got, err, tt.want)
			}
		})
	}
}

func TestHostToolkitVersionUsesHostRoot(t *testing.T) {
	dir := t.TempDir()
	// A fake chroot verifies the host root and fixed command without requiring
	// root, Linux, a GPU, or an installed toolkit on the development machine.
	script := `#!/bin/sh
[ "$1" = /proc/1/root ] && [ "$2" = /bin/sh ] && [ "$3" = -c ] && [ "$4" = 'exec nvidia-ctk --version' ] || exit 1
[ "$PATH" = /usr/local/nvidia/toolkit:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin ] || exit 2
echo 'NVIDIA Container Toolkit CLI version 1.20.1'
`
	if err := os.WriteFile(filepath.Join(dir, "chroot"), []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	got, err := hostToolkitVersion(context.Background())
	if err != nil || got != "1.20.1" {
		t.Fatalf("hostToolkitVersion() = %q, %v", got, err)
	}
}

func TestHostToolkitVersionHonorsCancellation(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "chroot"), []byte("#!/bin/sh\nexec /bin/sleep 30\n"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := hostToolkitVersion(ctx); err == nil {
		t.Fatal("expected the timed-out host probe to fail")
	}
	if ctx.Err() == nil {
		t.Fatal("probe returned before the context timed out")
	}
}

func TestLabelToolkitVersionBypass(t *testing.T) {
	t.Setenv("CHECK_HOST_TOOLKIT_VERSION", "true")
	t.Setenv("SKIP_GPU_STACK_VERSION_CHECKS", "true")
	t.Setenv(envNodeName, "")
	// An explicit bypass must not require a Kubernetes client or host binaries.
	if err := LabelToolkitVersion(context.Background()); err != nil {
		t.Fatalf("bypassed toolkit validation returned %v", err)
	}
}

func TestLabelToolkitVersion(t *testing.T) {
	for _, tt := range []struct {
		name, version string
		probeErr      error
		wantErr       bool
	}{
		{name: "supported", version: "1.20.1"},
		{name: "newer", version: "1.21.0"},
		{name: "unsupported is published before refusing startup", version: "1.19.1", wantErr: true},
		{name: "prerelease below minimum", version: "1.20.1-rc.1", wantErr: true},
		{name: "failed probe clears stale label", probeErr: errors.New("host binary missing"), wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			client := fake.NewSimpleClientset(&corev1.Node{ObjectMeta: metav1.ObjectMeta{
				Name: "node-a", Labels: map[string]string{toolkitinfo.VersionLabel: "1.21.0", "keep": "value"},
			}}).CoreV1().Nodes()
			err := labelToolkitVersion(ctx, client, "node-a", func(context.Context) (string, error) {
				return tt.version, tt.probeErr
			})
			if (err != nil) != tt.wantErr {
				t.Fatalf("labelToolkitVersion() error = %v, wantErr = %t", err, tt.wantErr)
			}
			node, err := client.Get(ctx, "node-a", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if node.Labels[toolkitinfo.VersionLabel] != tt.version || node.Labels["keep"] != "value" {
				t.Fatalf("unexpected labels: %v", node.Labels)
			}
		})
	}
}

func TestLabelToolkitVersionProbeSelection(t *testing.T) {
	for _, value := range []string{"", "false", "true"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("CHECK_HOST_TOOLKIT_VERSION", value)
			t.Setenv("SKIP_GPU_STACK_VERSION_CHECKS", "false")
			t.Setenv(envNodeName, "")
			err := LabelToolkitVersion(context.Background())
			if value == "true" {
				if err == nil || err.Error() != "NODE_NAME is required" {
					t.Fatalf("enabled probe should require NODE_NAME, got %v", err)
				}
			} else if err != nil {
				t.Fatalf("skipped probe must not access the host or Kubernetes, got %v", err)
			}
		})
	}
}
