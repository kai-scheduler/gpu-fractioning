// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
)

func newAMDPlugin(t *testing.T, failOpen bool) *Plugin {
	t.Helper()
	p, err := NewPlugin(Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		FailOpen:         failOpen,
		SupportSMSharing: true,
		MapDir:           t.TempDir(),
		Log:              slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil)),
		Vendor:           VendorAMD,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

func unified(adj *api.ContainerAdjustment) map[string]string {
	if adj == nil || adj.GetLinux().GetResources() == nil {
		return nil
	}
	return adj.GetLinux().GetResources().GetUnified()
}

func TestCreateContainerAMD(t *testing.T) {
	for _, tc := range []struct {
		name        string
		annotations map[string]string
		wantMin     string
		wantMax     string
		wantNil     bool
		wantErr     string
	}{
		{
			name: "request and limit on one GPU",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.request": "8Gi",
				"nvidia.com/container.main.gpu-memory.limit":   "11516Mi",
				"nvidia.com/container.main.gpus.devices":       "0000:03:00.0",
			},
			wantMin: "drm/0000:03:00.0/vram 8589934592",
			wantMax: "drm/0000:03:00.0/vram 12075401216",
		},
		{
			name: "limit only: the request defaults to the limit, on each of two GPUs",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "1Gi",
				"nvidia.com/container.main.gpus.devices":     "0000:03:00.0,0000:83:00.0",
			},
			wantMin: "drm/0000:03:00.0/vram 1073741824\ndrm/0000:83:00.0/vram 1073741824",
			wantMax: "drm/0000:03:00.0/vram 1073741824\ndrm/0000:83:00.0/vram 1073741824",
		},
		{
			name:        "no GPU memory annotations: untouched",
			annotations: map[string]string{"nvidia.com/container.main.gpus.devices": "0000:03:00.0"},
			wantNil:     true,
		},
		{
			name:        "another container's annotations: untouched",
			annotations: map[string]string{"nvidia.com/container.side.gpu-memory.limit": "1Gi", "nvidia.com/container.side.gpus.devices": "0000:03:00.0"},
			wantNil:     true,
		},
		{
			name:        "no device assignment: refused",
			annotations: map[string]string{"nvidia.com/container.main.gpu-memory.limit": "1Gi"},
			wantErr:     "no GPU assigned",
		},
		{
			name:        "device is not a PCI address: refused",
			annotations: map[string]string{"nvidia.com/container.main.gpu-memory.limit": "1Gi", "nvidia.com/container.main.gpus.devices": "GPU-1234"},
			wantErr:     "not a PCI address",
		},
		{
			name: "sm-sharing: refused (no MPS on AMD GPUs)",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "1Gi",
				"nvidia.com/container.main.gpus.devices":     "0000:03:00.0",
				"nvidia.com/container.main.gpu-compute.mode": "sm-sharing",
			},
			wantErr: "sm-sharing",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := &api.PodSandbox{Name: "test-pod", Annotations: tc.annotations}
			adj, _, err := newAMDPlugin(t, false).CreateContainer(context.Background(), pod, &api.Container{Name: "main"})
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tc.wantNil {
				if adj != nil {
					t.Fatalf("adjustment = %+v, want none", adj)
				}
				return
			}
			u := unified(adj)
			if u["dmem.min"] != tc.wantMin || u["dmem.max"] != tc.wantMax {
				t.Fatalf("unified = %q, want dmem.min %q and dmem.max %q", u, tc.wantMin, tc.wantMax)
			}
			if len(adj.Env) != 0 || len(adj.Mounts) != 0 {
				t.Fatalf("AMD adjustment adds no env or mounts, got env %v mounts %v", adj.Env, adj.Mounts)
			}
		})
	}
}

func TestCreateContainerAMDFailOpen(t *testing.T) {
	pod := &api.PodSandbox{Name: "test-pod", Annotations: map[string]string{"nvidia.com/container.main.gpu-memory.limit": "1Gi"}}
	adj, _, err := newAMDPlugin(t, true).CreateContainer(context.Background(), pod, &api.Container{Name: "main"})
	if err != nil || adj != nil {
		t.Fatalf("fail-open: got %+v, %v; want no adjustment and no error", adj, err)
	}
}

func TestNewPluginVendor(t *testing.T) {
	if _, err := NewPlugin(Config{MapDir: t.TempDir(), Vendor: VendorAMD, RetroactiveEnforcement: true}, nil); err == nil ||
		!strings.Contains(err.Error(), "not supported for AMD") {
		t.Fatalf("AMD with retroactive enforcement: err = %v", err)
	}
	if _, err := NewPlugin(Config{MapDir: t.TempDir(), Vendor: "intel"}, nil); err == nil || !strings.Contains(err.Error(), "unknown GPU vendor") {
		t.Fatalf("unknown vendor: err = %v", err)
	}
	p, err := NewPlugin(Config{MapDir: t.TempDir()}, nil)
	if err != nil || p.vendor != VendorNVIDIA {
		t.Fatalf("default vendor: %v, %v", p, err)
	}
}
