// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package driverlabel

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

// fakeChroot lets the tests exercise the real probe without root, Linux, or GPUs.
func fakeChroot(t *testing.T, script string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "chroot"), []byte("#!/bin/sh\n"+script), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
}

func TestHostToolkitVersionUsesSelectedHostBinary(t *testing.T) {
	const systemPath = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
	for _, tt := range []struct{ name, configured, binary, searchPath string }{
		{"external", "nvidia-ctk", "nvidia-ctk", systemPath},
		{"managed", "/usr/local/nvidia/toolkit/nvidia-ctk", "/usr/local/nvidia/toolkit/nvidia-ctk", systemPath},
		{"custom managed path", "/opt/nvidia tools/toolkit/nvidia-ctk", "/opt/nvidia tools/toolkit/nvidia-ctk", systemPath},
		{"no policy", "", "nvidia-ctk", systemPath + ":/usr/local/nvidia/toolkit"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOST_TOOLKIT_BINARY", tt.configured)
			fakeChroot(t, fmt.Sprintf(`[ "$#" = 3 ] && [ "$1" = /proc/1/root ] && [ "$2" = %q ] && [ "$3" = --version ] || exit 1
[ "$PATH" = %q ] || exit 2
printf 'NVIDIA Container Toolkit CLI version 1.20.1\n'
`, tt.binary, tt.searchPath))
			got, err := hostToolkitVersion(context.Background())
			if err != nil || got != "1.20.1" {
				t.Fatalf("hostToolkitVersion() = %q, %v", got, err)
			}
		})
	}
}

func TestHostToolkitVersionHonorsCancellation(t *testing.T) {
	fakeChroot(t, "exec /bin/sleep 30\n")
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := hostToolkitVersion(ctx); err == nil {
		t.Fatal("expected the timed-out host probe to fail")
	}
	if ctx.Err() == nil {
		t.Fatal("probe returned before the context timed out")
	}
}

func TestHostToolkitVersionRejectsRelativeBinary(t *testing.T) {
	t.Setenv("HOST_TOOLKIT_BINARY", "relative/toolkit/nvidia-ctk")
	fakeChroot(t, "exit 0\n")
	if _, err := hostToolkitVersion(context.Background()); err == nil || !strings.Contains(err.Error(), "absolute path") {
		t.Fatalf("expected a path validation error, got %v", err)
	}
}

func TestHostToolkitVersionDoesNotFallbackFromManagedPath(t *testing.T) {
	t.Setenv("HOST_TOOLKIT_BINARY", "/missing/toolkit/nvidia-ctk")
	fakeChroot(t, `[ "$2" = /missing/toolkit/nvidia-ctk ] && exit 127
printf 'NVIDIA Container Toolkit CLI version 1.20.1\n'
`)
	if _, err := hostToolkitVersion(context.Background()); err == nil || !strings.Contains(err.Error(), "/missing/toolkit/nvidia-ctk") {
		t.Fatalf("expected an error for the selected managed binary, got %v", err)
	}
}

func TestValidateAndLabelHostToolkit(t *testing.T) {
	for _, tt := range []struct {
		name, version string
		exitCode      int
		wantErr       bool
		wantLabel     string
	}{
		{name: "supported", version: "1.20.1", wantLabel: "1.20.1"},
		{name: "newer", version: "1.21.0", wantLabel: "1.21.0"},
		{name: "unsupported is published before refusing startup", version: "1.19.1", wantErr: true, wantLabel: "1.19.1"},
		{name: "prerelease below minimum", version: "1.20.1-rc.1", wantErr: true, wantLabel: "1.20.1-rc.1"},
		{name: "failed probe clears stale label", exitCode: 127, wantErr: true},
		{name: "invalid output clears stale label", version: "invalid", wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fakeChroot(t, fmt.Sprintf("printf 'NVIDIA Container Toolkit CLI version %%s\\n' %q\nexit %d\n", tt.version, tt.exitCode))
			ctx := context.Background()
			client := fake.NewSimpleClientset(&corev1.Node{ObjectMeta: metav1.ObjectMeta{
				Name: "node-a", Labels: map[string]string{toolkitinfo.VersionLabel: "1.21.0", "keep": "value"},
			}}).CoreV1().Nodes()
			err := validateAndLabelHostToolkit(ctx, client, "node-a")
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateAndLabelHostToolkit() error = %v, wantErr = %t", err, tt.wantErr)
			}
			node, err := client.Get(ctx, "node-a", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if node.Labels[toolkitinfo.VersionLabel] != tt.wantLabel || node.Labels["keep"] != "value" {
				t.Fatalf("unexpected labels: %v", node.Labels)
			}
			if tt.wantLabel == "" {
				if _, present := node.Labels[toolkitinfo.VersionLabel]; present {
					t.Fatal("stale label was not removed")
				}
			}
		})
	}
}

func TestValidateAndLabelHostToolkitRequiresLabelPublication(t *testing.T) {
	fakeChroot(t, "printf 'NVIDIA Container Toolkit CLI version 1.20.1\\n'\n")
	client := fake.NewSimpleClientset().CoreV1().Nodes()
	if err := validateAndLabelHostToolkit(context.Background(), client, "missing-node"); err == nil || !strings.Contains(err.Error(), "patch node") {
		t.Fatalf("label publication failure must prevent startup, got %v", err)
	}
}

func TestValidateHostToolkitBypass(t *testing.T) {
	for _, value := range []string{"", "false", "true"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("SKIP_GPU_STACK_VERSION_CHECKS", value)
			t.Setenv(envNodeName, "")
			err := ValidateHostToolkit(context.Background())
			if value != "true" {
				if err == nil || err.Error() != "NODE_NAME is required" {
					t.Fatalf("enabled probe should require NODE_NAME, got %v", err)
				}
			} else if err != nil {
				t.Fatalf("skipped probe must not access the host or Kubernetes, got %v", err)
			}
		})
	}
}
