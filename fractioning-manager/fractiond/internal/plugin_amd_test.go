// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
)

// w7900Capacity is a dmem.capacity with two amdgpu GPUs' VRAM regions.
const w7900Capacity = "drm/0000:03:00.0/vram 51522830336\ndrm/0000:83:00.0/vram 51522830336\n"

type amdOptions struct {
	failOpen    bool
	retroactive bool
	capacity    *string // nil: no capacity file at all
	readiness   ReadinessSetter
}

func newAMDPlugin(t *testing.T, o amdOptions) *Plugin {
	t.Helper()
	capacityFile := filepath.Join(t.TempDir(), "dmem.capacity")
	if o.capacity != nil {
		if err := os.WriteFile(capacityFile, []byte(*o.capacity), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	p, err := NewPlugin(Config{
		AnnotationPrefix:       configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory:       configuration.DefaultMPSPipeDirectory,
		FailOpen:               o.failOpen,
		SupportSMSharing:       true,
		RetroactiveEnforcement: o.retroactive,
		MapDir:                 t.TempDir(),
		Log:                    slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil)),
		Readiness:              o.readiness,
		Vendor:                 VendorAMD,
		DmemCapacityFile:       capacityFile,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

func capacity(s string) *string { return &s }

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
		wantMax     string
		wantNil     bool
		wantErr     string
	}{
		{
			name: "request and limit on one GPU: only the limit is written",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.request": "8Gi",
				"nvidia.com/container.main.gpu-memory.limit":   "11516Mi",
				"nvidia.com/container.main.gpus.devices":       "0000:03:00.0",
			},
			wantMax: "drm/0000:03:00.0/vram 12075401216",
		},
		{
			name: "limit on each of two GPUs, one region per line",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "1Gi",
				"nvidia.com/container.main.gpus.devices":     "0000:03:00.0,0000:83:00.0",
			},
			wantMax: "drm/0000:03:00.0/vram 1073741824\ndrm/0000:83:00.0/vram 1073741824",
		},
		{
			name: "request only: the limit defaults to the request",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.request": "2Gi",
				"nvidia.com/container.main.gpus.devices":       "0000:03:00.0",
			},
			wantMax: "drm/0000:03:00.0/vram 2147483648",
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
			name:        "GPU without a dmem region on this node: refused with a clear error",
			annotations: map[string]string{"nvidia.com/container.main.gpu-memory.limit": "1Gi", "nvidia.com/container.main.gpus.devices": "0000:c1:00.0"},
			wantErr:     "GPU 0000:c1:00.0 has no dmem VRAM region",
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
			adj, _, err := newAMDPlugin(t, amdOptions{capacity: capacity(w7900Capacity)}).CreateContainer(context.Background(), pod, &api.Container{Name: "main"})
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
			if u["dmem.max"] != tc.wantMax {
				t.Fatalf("dmem.max = %q, want %q", u["dmem.max"], tc.wantMax)
			}
			if _, ok := u["dmem.min"]; ok || len(u) != 1 {
				t.Fatalf("unified = %q, want dmem.max only", u)
			}
			if len(adj.Env) != 0 || len(adj.Mounts) != 0 {
				t.Fatalf("AMD adjustment adds no env or mounts, got env %v mounts %v", adj.Env, adj.Mounts)
			}
		})
	}
}

func TestCreateContainerAMDFailOpen(t *testing.T) {
	for _, tc := range []struct {
		name        string
		annotations map[string]string
		capacity    *string
	}{
		{"no device assignment", map[string]string{"nvidia.com/container.main.gpu-memory.limit": "1Gi"}, capacity(w7900Capacity)},
		{"GPU without a dmem region (older kernel)", map[string]string{"nvidia.com/container.main.gpu-memory.limit": "1Gi", "nvidia.com/container.main.gpus.devices": "0000:03:00.0"}, capacity("")},
		{"dmem.capacity not mounted", map[string]string{"nvidia.com/container.main.gpu-memory.limit": "1Gi", "nvidia.com/container.main.gpus.devices": "0000:03:00.0"}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := &api.PodSandbox{Name: "test-pod", Annotations: tc.annotations}
			adj, _, err := newAMDPlugin(t, amdOptions{failOpen: true, capacity: tc.capacity}).CreateContainer(context.Background(), pod, &api.Container{Name: "main"})
			if err != nil || adj != nil {
				t.Fatalf("fail-open: got %+v, %v; want no adjustment and no error", adj, err)
			}
			// Fail-closed, the same container is refused by fractiond itself, before the runtime writes anything.
			if _, _, err := newAMDPlugin(t, amdOptions{capacity: tc.capacity}).CreateContainer(context.Background(), pod, &api.Container{Name: "main"}); err == nil {
				t.Fatal("fail-closed: want an error")
			}
		})
	}
}

type readinessRecorder struct{ ready []bool }

func (r *readinessRecorder) SetReady(ready bool) { r.ready = append(r.ready, ready) }

func TestSynchronizeAMDReadiness(t *testing.T) {
	for _, tc := range []struct {
		name      string
		capacity  *string
		wantReady bool
	}{
		{"GPU VRAM regions", capacity(w7900Capacity), true},
		{"dmem but no amdgpu VRAM region (kernel before 6.15, or another driver)", capacity("drm/0000:00:02.0/vram0 4294967296\n"), false},
		{"empty dmem.capacity", capacity(""), false},
		{"dmem.capacity not mounted", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &readinessRecorder{}
			p := newAMDPlugin(t, amdOptions{retroactive: true, capacity: tc.capacity, readiness: r})
			if _, err := p.Synchronize(context.Background(), nil, nil); err != nil {
				t.Fatal(err)
			}
			p.Flush()
			if len(r.ready) != 1 || r.ready[0] != tc.wantReady {
				t.Fatalf("readiness = %v, want [%v]", r.ready, tc.wantReady)
			}
		})
	}
}

func TestSynchronizeAMDReapply(t *testing.T) {
	limit := "drm/0000:03:00.0/vram 1073741824"
	pods := []*api.PodSandbox{
		{Id: "pod-frac", Name: "frac", Annotations: map[string]string{
			"nvidia.com/container.main.gpu-memory.limit": "1Gi",
			"nvidia.com/container.main.gpus.devices":     "0000:03:00.0",
		}},
		{Id: "pod-bad", Name: "bad", Annotations: map[string]string{
			"nvidia.com/container.main.gpu-memory.limit": "1Gi",
			"nvidia.com/container.main.gpus.devices":     "0000:c1:00.0",
		}},
		{Id: "pod-plain", Name: "plain"},
	}
	withLimit := &api.LinuxContainer{Resources: &api.LinuxResources{Unified: map[string]string{"dmem.max": limit}}}
	containers := []*api.Container{
		{Id: "uncapped", PodSandboxId: "pod-frac", Name: "main", State: api.ContainerState_CONTAINER_RUNNING},
		{Id: "created", PodSandboxId: "pod-frac", Name: "main", State: api.ContainerState_CONTAINER_CREATED},
		{Id: "capped", PodSandboxId: "pod-frac", Name: "main", State: api.ContainerState_CONTAINER_RUNNING, Linux: withLimit},
		{Id: "exited", PodSandboxId: "pod-frac", Name: "main", State: api.ContainerState_CONTAINER_STOPPED},
		{Id: "sidecar", PodSandboxId: "pod-frac", Name: "side", State: api.ContainerState_CONTAINER_RUNNING},
		{Id: "no-region", PodSandboxId: "pod-bad", Name: "main", State: api.ContainerState_CONTAINER_RUNNING},
		{Id: "plain", PodSandboxId: "pod-plain", Name: "main", State: api.ContainerState_CONTAINER_RUNNING},
		{Id: "orphan", PodSandboxId: "gone", Name: "main", State: api.ContainerState_CONTAINER_RUNNING},
	}

	updates, err := newAMDPlugin(t, amdOptions{retroactive: true, capacity: capacity(w7900Capacity)}).Synchronize(context.Background(), pods, containers)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, u := range updates {
		ids = append(ids, u.GetContainerId())
		if got := u.GetLinux().GetResources().GetUnified(); len(got) != 1 || got["dmem.max"] != limit {
			t.Errorf("update for %s: unified = %q, want dmem.max %q only", u.GetContainerId(), got, limit)
		}
		if !u.GetIgnoreFailure() {
			t.Errorf("update for %s is not ignore-failure: one failed update would fail the synchronization", u.GetContainerId())
		}
	}
	if got := strings.Join(ids, ","); got != "uncapped,created" {
		t.Fatalf("updated containers = %s, want uncapped,created", got)
	}

	// Retroactive enforcement off: the limits are left alone.
	updates, err = newAMDPlugin(t, amdOptions{capacity: capacity(w7900Capacity)}).Synchronize(context.Background(), pods, containers)
	if err != nil || len(updates) != 0 {
		t.Fatalf("retroactive enforcement off: updates = %v, %v; want none", updates, err)
	}
	// Not ready (no region): nothing is re-applied either.
	updates, err = newAMDPlugin(t, amdOptions{retroactive: true, capacity: capacity("")}).Synchronize(context.Background(), pods, containers)
	if err != nil || len(updates) != 0 {
		t.Fatalf("no dmem region: updates = %v, %v; want none", updates, err)
	}
}

func TestNewPluginVendor(t *testing.T) {
	// AMD with retroactive enforcement needs no stopper: it re-applies limits through NRI.
	p, err := NewPlugin(Config{MapDir: t.TempDir(), Vendor: " AMD ", RetroactiveEnforcement: true}, nil)
	if err != nil || p.vendor != VendorAMD || !p.reapplyAMD || p.sentinel != nil {
		t.Fatalf("AMD with retroactive enforcement: %+v, %v", p, err)
	}
	if _, err := NewPlugin(Config{MapDir: t.TempDir(), Vendor: "intel"}, nil); err == nil || !strings.Contains(err.Error(), "unknown GPU vendor") {
		t.Fatalf("unknown vendor: err = %v", err)
	}
	p, err = NewPlugin(Config{MapDir: t.TempDir()}, nil)
	if err != nil || p.vendor != VendorNVIDIA {
		t.Fatalf("default vendor: %v, %v", p, err)
	}
	// NVIDIA still requires a stopper for retroactive enforcement.
	if _, err := NewPlugin(Config{MapDir: t.TempDir(), Vendor: "NVIDIA", RetroactiveEnforcement: true}, nil); err == nil {
		t.Fatal("NVIDIA with retroactive enforcement and no stopper: want an error")
	}
	for in, want := range map[string]string{"": VendorNVIDIA, "AMD": VendorAMD, " amd\n": VendorAMD, "Nvidia": VendorNVIDIA} {
		if got := NormalizeVendor(in); got != want {
			t.Errorf("NormalizeVendor(%q) = %q, want %q", in, got, want)
		}
	}
}
