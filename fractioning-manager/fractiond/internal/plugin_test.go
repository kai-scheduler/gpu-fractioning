// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"bytes"
	"context"
	"log/slog"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/injection"
)

func TestCreateContainer(t *testing.T) {
	tests := []struct {
		name                     string
		annotations              map[string]string
		containerName            string
		expectedNil              bool
		expectedErr              bool
		expectedEnvKeys          []string
		expectedEnvVals          map[string]string
		expectedMount            bool
		expectedMountSource      string // defaults to configuration.DefaultMPSPipeDirectory when empty
		expectedMountDestination string // defaults to configuration.DefaultMPSPipeDirectory when empty
	}{
		{
			name: "both request and limit",
			annotations: map[string]string{
				"nvidia.com/container.trainer.gpu-memory.request": "2048Mi",
				"nvidia.com/container.trainer.gpu-memory.limit":   "4Gi",
			},
			containerName:   "trainer",
			expectedNil:     false,
			expectedEnvKeys: []string{injection.EnvGPUMemoryRequest, injection.EnvGPUMemoryLimit, injection.EnvMPSPipeDirectory},
			expectedEnvVals: map[string]string{injection.EnvGPUMemoryRequest: "2048", injection.EnvGPUMemoryLimit: "4096"},
			expectedMount:   true,
		},
		{
			name: "only limit defaults request to the limit",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "4Gi",
			},
			containerName:   "main",
			expectedNil:     false,
			expectedEnvKeys: []string{injection.EnvGPUMemoryRequest, injection.EnvGPUMemoryLimit, injection.EnvMPSPipeDirectory},
			expectedEnvVals: map[string]string{injection.EnvGPUMemoryRequest: "4096", injection.EnvGPUMemoryLimit: "4096"},
			expectedMount:   true,
		},
		{
			name: "only request defaults limit to the request",
			annotations: map[string]string{
				"nvidia.com/container.worker.gpu-memory.request": "1Gi",
			},
			containerName:   "worker",
			expectedNil:     false,
			expectedEnvKeys: []string{injection.EnvGPUMemoryRequest, injection.EnvGPUMemoryLimit, injection.EnvMPSPipeDirectory},
			expectedEnvVals: map[string]string{injection.EnvGPUMemoryRequest: "1024", injection.EnvGPUMemoryLimit: "1024"},
			expectedMount:   true,
		},
		{
			name: "injects assigned GPU devices alongside memory config",
			annotations: map[string]string{
				"nvidia.com/container.trainer.gpu-memory.limit": "4Gi",
				"nvidia.com/container.trainer.gpus.devices":     "GPU-abc123,GPU-def456",
			},
			containerName:   "trainer",
			expectedNil:     false,
			expectedEnvKeys: []string{injection.EnvGPUMemoryLimit, injection.EnvMPSPipeDirectory, injection.EnvVisibleDevices},
			expectedEnvVals: map[string]string{injection.EnvVisibleDevices: "GPU-abc123,GPU-def456"},
			expectedMount:   true,
		},
		{
			name: "device assignment without memory annotation is not injected",
			annotations: map[string]string{
				"nvidia.com/container.trainer.gpus.devices": "GPU-abc123",
			},
			containerName: "trainer",
			expectedNil:   true,
		},
		{
			name:          "no GPU annotations returns nil",
			annotations:   map[string]string{"some-other/annotation": "value"},
			containerName: "main",
			expectedNil:   true,
		},
		{
			name:          "nil annotations returns nil",
			annotations:   nil,
			containerName: "main",
			expectedNil:   true,
		},
		{
			name: "different container returns nil",
			annotations: map[string]string{
				"nvidia.com/container.other.gpu-memory.limit": "4Gi",
			},
			containerName: "main",
			expectedNil:   true,
		},
		{
			name: "malformed value returns error (fail-closed)",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "not-valid",
			},
			containerName: "main",
			expectedErr:   true,
		},
		{
			name: "below 1MB returns error (fail-closed)",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "500Ki",
			},
			containerName: "main",
			expectedErr:   true,
		},
		{
			name: "absent compute-mode annotation mounts default socket unchanged",
			annotations: map[string]string{
				"nvidia.com/container.trainer.gpu-memory.limit": "4Gi",
			},
			containerName:   "trainer",
			expectedNil:     false,
			expectedEnvKeys: []string{injection.EnvGPUMemoryLimit, injection.EnvMPSPipeDirectory},
			expectedEnvVals: map[string]string{injection.EnvMPSPipeDirectory: configuration.DefaultMPSPipeDirectory},
			expectedMount:   true,
		},
		{
			name: "explicit time-slicing mounts default socket unchanged",
			annotations: map[string]string{
				"nvidia.com/container.trainer.gpu-memory.limit": "4Gi",
				"nvidia.com/container.trainer.gpu-compute.mode": "time-slicing",
			},
			containerName:   "trainer",
			expectedNil:     false,
			expectedEnvKeys: []string{injection.EnvGPUMemoryLimit, injection.EnvMPSPipeDirectory},
			expectedEnvVals: map[string]string{injection.EnvMPSPipeDirectory: configuration.DefaultMPSPipeDirectory},
			expectedMount:   true,
		},
		{
			name: "sm-sharing routes to the shared server's default namespace",
			annotations: map[string]string{
				"nvidia.com/container.trainer.gpu-memory.limit": "4Gi",
				"nvidia.com/container.trainer.gpu-compute.mode": "sm-sharing",
			},
			containerName:   "trainer",
			expectedNil:     false,
			expectedEnvKeys: []string{injection.EnvGPUMemoryLimit, injection.EnvMPSPipeDirectory},
			expectedEnvVals: map[string]string{
				injection.EnvMPSPipeDirectory: configuration.ContainerMPSPipeDirectory,
			},
			expectedMount:            true,
			expectedMountSource:      filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath),
			expectedMountDestination: configuration.ContainerMPSPipeDirectory,
		},
		{
			name: "invalid compute mode value fails closed",
			annotations: map[string]string{
				"nvidia.com/container.main.gpu-memory.limit": "4Gi",
				"nvidia.com/container.main.gpu-compute.mode": "mig",
			},
			containerName: "main",
			expectedErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestPlugin(t)
			pod := &api.PodSandbox{
				Name:        "test-pod",
				Annotations: tt.annotations,
			}
			ctr := &api.Container{Name: tt.containerName}

			adj, _, err := p.CreateContainer(context.Background(), pod, ctr)

			if tt.expectedErr {
				if err == nil {
					t.Errorf("expected error, got adj=%+v", adj)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.expectedNil {
				if adj != nil {
					t.Errorf("expected nil adjustment, got %+v", adj)
				}
				return
			}

			if adj == nil {
				t.Fatal("expected non-nil adjustment, got nil")
			}

			envKeys := make(map[string]bool)
			envVals := make(map[string]string)
			for _, kv := range adj.Env {
				envKeys[kv.Key] = true
				envVals[kv.Key] = kv.Value
			}
			for _, key := range tt.expectedEnvKeys {
				if !envKeys[key] {
					t.Errorf("expected env var %q not found in adjustment", key)
				}
			}
			for key, expectedVal := range tt.expectedEnvVals {
				if envVals[key] != expectedVal {
					t.Errorf("env %q = %q, expected %q", key, envVals[key], expectedVal)
				}
			}

			if tt.expectedMount {
				wantSource := tt.expectedMountSource
				if wantSource == "" {
					wantSource = configuration.DefaultMPSPipeDirectory
				}
				wantDestination := tt.expectedMountDestination
				if wantDestination == "" {
					wantDestination = configuration.DefaultMPSPipeDirectory
				}
				if len(adj.Mounts) == 0 {
					t.Error("expected mount in adjustment, got none")
				} else {
					m := adj.Mounts[0]
					if m.Source != wantSource {
						t.Errorf("mount source = %q, expected %q", m.Source, wantSource)
					}
					if m.Destination != wantDestination {
						t.Errorf("mount destination = %q, expected %q", m.Destination, wantDestination)
					}
				}
			}
		})
	}
}

func TestCreateContainerOverwritesExistingVisibleDevices(t *testing.T) {
	p := newTestPlugin(t)
	pod := &api.PodSandbox{
		Name: "test-pod",
		Annotations: map[string]string{
			"nvidia.com/container.trainer.gpu-memory.limit": "4Gi",
			"nvidia.com/container.trainer.gpus.devices":     "GPU-assigned",
		},
	}
	// The container already carries NVIDIA_VISIBLE_DEVICES (e.g. "void" injected
	// by an admission plugin because the pod does not request nvidia.com/gpu).
	// Our assignment must overwrite it, leaving a single value once NRI applies
	// the adjustment.
	ctr := &api.Container{
		Name: "trainer",
		Env:  []string{injection.EnvVisibleDevices + "=void"},
	}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if adj == nil {
		t.Fatal("expected non-nil adjustment, got nil")
	}

	var setValues []string
	removed := false
	for _, kv := range adj.Env {
		if key, marked := kv.IsMarkedForRemoval(); marked {
			if key == injection.EnvVisibleDevices {
				removed = true
			}
			continue
		}
		if kv.Key == injection.EnvVisibleDevices {
			setValues = append(setValues, kv.Value)
		}
	}

	if !removed {
		t.Error("expected existing NVIDIA_VISIBLE_DEVICES to be removed before re-adding")
	}
	if len(setValues) != 1 {
		t.Fatalf("expected exactly one NVIDIA_VISIBLE_DEVICES set entry, got %d: %v", len(setValues), setValues)
	}
	if setValues[0] != "GPU-assigned" {
		t.Errorf("NVIDIA_VISIBLE_DEVICES = %q, want %q", setValues[0], "GPU-assigned")
	}
}

// TestCreateContainer_ComputeModeWithoutMemoryAnnotation covers a container
// that asks for a compute mode but never asks for GPU memory. It is not a
// fractioning container, so it is left alone and the mode has no effect —
// almost always a forgotten or misspelled memory annotation, so the skip log
// points at the relationship rather than reporting the memory annotations
// missing on their own.
func TestCreateContainer_ComputeModeWithoutMemoryAnnotation(t *testing.T) {
	var logBuf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		SupportSMSharing: true,
		MapDir:           t.TempDir(),
		Log:              log,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}

	pod := &api.PodSandbox{
		Name: "test-pod",
		Annotations: map[string]string{
			"nvidia.com/container.main.gpu-compute.mode": "sm-sharing",
		},
	}
	ctr := &api.Container{Name: "main"}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("a container without memory annotations must not be blocked, got: %v", err)
	}
	if adj != nil {
		t.Errorf("expected nil adjustment for a non-fractioning container, got %+v", adj)
	}
	if !strings.Contains(logBuf.String(), "gpu-compute.mode has no effect without them") {
		t.Errorf("expected the skip log to explain that the compute mode is inert, got: %s", logBuf.String())
	}
}

func TestCreateContainer_FailOpenLogsWarning(t *testing.T) {
	var logBuf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		FailOpen:         true,
		MapDir:           t.TempDir(),
		Log:              log,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}

	pod := &api.PodSandbox{
		Name: "test-pod",
		Annotations: map[string]string{
			"nvidia.com/container.main.gpu-memory.limit": "garbage-value",
		},
	}
	ctr := &api.Container{Name: "main"}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("fail-open should not return error, got: %v", err)
	}
	if adj != nil {
		t.Errorf("expected nil adjustment on parse error, got %+v", adj)
	}
	if logBuf.Len() == 0 {
		t.Error("expected warning log on parse failure, got nothing")
	}
}

func TestCreateContainer_FailOpenFallsBackToTimeSlicingOnInvalidComputeMode(t *testing.T) {
	var logBuf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		FailOpen:         true,
		MapDir:           t.TempDir(),
		Log:              log,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}

	pod := &api.PodSandbox{
		Name: "test-pod",
		Annotations: map[string]string{
			"nvidia.com/container.main.gpu-memory.limit": "4Gi",
			"nvidia.com/container.main.gpu-compute.mode": "mig",
		},
	}
	ctr := &api.Container{Name: "main"}

	// FailOpen must not block the container: an invalid compute-mode value
	// falls back to time-slicing (the default socket) rather than failing.
	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("fail-open should not return error, got: %v", err)
	}
	if adj == nil {
		t.Fatal("expected non-nil adjustment (memory annotation present), got nil")
	}
	if len(adj.Mounts) == 0 {
		t.Fatal("expected mount in adjustment, got none")
	}
	if got := adj.Mounts[0].Destination; got != configuration.DefaultMPSPipeDirectory {
		t.Errorf("mount destination = %q, expected fallback to default %q", got, configuration.DefaultMPSPipeDirectory)
	}
	if logBuf.Len() == 0 {
		t.Error("expected warning log on compute-mode parse failure, got nothing")
	}
}

// TestCreateContainer_SupportSMSharingDisabled verifies the sm-sharing
// chicken bit: when disabled, the annotation is rejected like any other
// invalid compute-mode value, both fail-closed (default) and fail-open
// (falls back to time-slicing) — mirroring
// TestCreateContainer_FailOpenFallsBackToTimeSlicingOnInvalidComputeMode.
func TestCreateContainer_SupportSMSharingDisabled(t *testing.T) {
	pod := &api.PodSandbox{
		Name: "test-pod",
		Annotations: map[string]string{
			"nvidia.com/container.main.gpu-memory.limit": "4Gi",
			"nvidia.com/container.main.gpu-compute.mode": "sm-sharing",
		},
	}
	ctr := &api.Container{Name: "main"}

	t.Run("fail-closed blocks the container", func(t *testing.T) {
		log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
		p, err := NewPlugin(Config{
			AnnotationPrefix: configuration.DefaultAnnotationPrefix,
			MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
			SupportSMSharing: false,
			MapDir:           t.TempDir(),
			Log:              log,
		}, nil)
		if err != nil {
			t.Fatalf("NewPlugin: %v", err)
		}

		if _, _, err := p.CreateContainer(context.Background(), pod, ctr); err == nil {
			t.Fatal("expected error blocking the container, got nil")
		}
	})

	t.Run("fail-open falls back to time-slicing", func(t *testing.T) {
		var logBuf bytes.Buffer
		log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelDebug}))
		p, err := NewPlugin(Config{
			AnnotationPrefix: configuration.DefaultAnnotationPrefix,
			MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
			FailOpen:         true,
			SupportSMSharing: false,
			MapDir:           t.TempDir(),
			Log:              log,
		}, nil)
		if err != nil {
			t.Fatalf("NewPlugin: %v", err)
		}

		adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
		if err != nil {
			t.Fatalf("fail-open should not return error, got: %v", err)
		}
		if adj == nil || len(adj.Mounts) == 0 {
			t.Fatalf("expected a mount falling back to time-slicing, got %+v", adj)
		}
		if got := adj.Mounts[0].Destination; got != configuration.DefaultMPSPipeDirectory {
			t.Errorf("mount destination = %q, expected fallback to default %q", got, configuration.DefaultMPSPipeDirectory)
		}
		if logBuf.Len() == 0 {
			t.Error("expected warning log on compute-mode parse failure, got nothing")
		}
	})
}

func newTestPlugin(t *testing.T) *Plugin {
	t.Helper()
	log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		FailOpen:         false,
		SupportSMSharing: true,
		MapDir:           t.TempDir(),
		Log:              log,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

// fakeStopper records the container IDs Synchronize asks to stop.
type fakeStopper struct {
	mu      sync.Mutex
	stopped []string
}

func (f *fakeStopper) Stop(_ context.Context, containerID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped = append(f.stopped, containerID)
	return nil
}

func (f *fakeStopper) stoppedIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.stopped...)
}

// enforcingPlugin builds a plugin with retroactive enforcement enabled, backed
// by the given fake stopper.
func enforcingPlugin(t *testing.T, stopper *fakeStopper) *Plugin {
	t.Helper()
	log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix:       configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory:       configuration.DefaultMPSPipeDirectory,
		MapDir:                 t.TempDir(),
		RetroactiveEnforcement: true,
		Log:                    log,
	}, stopper)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

// enforcingFailOpenPlugin is enforcingPlugin with the create hook's fail-open
// policy on, so the audit it drives inherits the same policy.
func enforcingFailOpenPlugin(t *testing.T, stopper *fakeStopper) *Plugin {
	t.Helper()
	log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix:       configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory:       configuration.DefaultMPSPipeDirectory,
		MapDir:                 t.TempDir(),
		RetroactiveEnforcement: true,
		FailOpen:               true,
		SupportSMSharing:       true,
		Log:                    log,
	}, stopper)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

// syncSnapshot returns a pod plus a fully-injected container and an uninjected
// container, both fractioning containers ("trainer") with a 4Gi limit annotation.
func syncSnapshot() ([]*api.PodSandbox, []*api.Container) {
	pods := []*api.PodSandbox{
		{Id: "p1", Name: "pod1", Namespace: "default", Annotations: map[string]string{
			configuration.DefaultAnnotationPrefix + "trainer.gpu-memory.limit": "4Gi",
		}},
		{Id: "p2", Name: "pod2", Namespace: "default", Annotations: map[string]string{
			configuration.DefaultAnnotationPrefix + "trainer.gpu-memory.limit": "4Gi",
		}},
	}
	containers := []*api.Container{
		{ // injected → not a violation (limit-only annotation ⇒ request env defaulted too)
			Id: "c1", Name: "trainer", PodSandboxId: "p1",
			State: api.ContainerState_CONTAINER_RUNNING,
			Env:   []string{injection.EnvMPSPipeDirectory + "=" + configuration.DefaultMPSPipeDirectory, injection.EnvGPUMemoryLimit + "=4096", injection.EnvGPUMemoryRequest + "=4096"},
			Mounts: []*api.Mount{{
				Source:      configuration.DefaultMPSPipeDirectory,
				Destination: configuration.DefaultMPSPipeDirectory,
				Type:        "bind",
			}},
		},
		{ // uninjected → violation
			Id: "c2", Name: "trainer", PodSandboxId: "p2",
			State: api.ContainerState_CONTAINER_RUNNING,
		},
	}
	return pods, containers
}

func TestSynchronizeEnforcesRetroactively(t *testing.T) {
	fake := &fakeStopper{}
	p := enforcingPlugin(t, fake)
	pods, containers := syncSnapshot()

	if _, err := p.Synchronize(context.Background(), pods, containers); err != nil {
		t.Fatalf("Synchronize returned error: %v", err)
	}
	p.Flush() // wait for async remediation

	if got := fake.stoppedIDs(); !slices.Equal(got, []string{"c2"}) {
		t.Fatalf("stopped = %v, want [c2]", got)
	}
}

func TestSynchronizeEnforcesMissingVisibleDevices(t *testing.T) {
	fake := &fakeStopper{}
	p := enforcingPlugin(t, fake)

	// A fractioning container whose pod was assigned a GPU device but which is running
	// without NVIDIA_VISIBLE_DEVICES: it holds all the memory injection yet cannot
	// see the correct GPU, so retroactive enforcement must stop it.
	pods := []*api.PodSandbox{
		{Id: "p1", Name: "pod1", Namespace: "default", Annotations: map[string]string{
			configuration.DefaultAnnotationPrefix + "trainer.gpu-memory.limit": "4Gi",
			"nvidia.com/container.trainer.gpus.devices":                        "GPU-abc123",
		}},
	}
	containers := []*api.Container{
		{
			Id: "c1", Name: "trainer", PodSandboxId: "p1",
			State: api.ContainerState_CONTAINER_RUNNING,
			Env:   []string{injection.EnvMPSPipeDirectory + "=" + configuration.DefaultMPSPipeDirectory, injection.EnvGPUMemoryLimit + "=4096"},
			Mounts: []*api.Mount{{
				Source:      configuration.DefaultMPSPipeDirectory,
				Destination: configuration.DefaultMPSPipeDirectory,
				Type:        "bind",
			}},
		},
	}

	if _, err := p.Synchronize(context.Background(), pods, containers); err != nil {
		t.Fatalf("Synchronize returned error: %v", err)
	}
	p.Flush() // wait for async remediation

	if got := fake.stoppedIDs(); !slices.Equal(got, []string{"c1"}) {
		t.Fatalf("stopped = %v, want [c1]", got)
	}
}

// TestSynchronizeFailOpenEnforcesBadComputeMode pins the two halves of the
// fail-open policy together. Creation with an unusable compute mode does not
// block the container; it falls back to time-slicing and still injects the
// memory limit. So a container that came up while fractiond was down, carrying
// valid memory annotations and a mode typo, is owed that injection and must be
// stopped for recreation — not written off as unparseable.
func TestSynchronizeFailOpenEnforcesBadComputeMode(t *testing.T) {
	fake := &fakeStopper{}
	p := enforcingFailOpenPlugin(t, fake)

	pods := []*api.PodSandbox{
		{Id: "p1", Name: "pod1", Namespace: "default", Annotations: map[string]string{
			configuration.DefaultAnnotationPrefix + "trainer.gpu-memory.limit": "4Gi",
			"nvidia.com/container.trainer.gpu-compute.mode":                    "mig",
		}},
	}
	containers := []*api.Container{
		{Id: "c1", Name: "trainer", PodSandboxId: "p1", State: api.ContainerState_CONTAINER_RUNNING},
	}

	if _, err := p.Synchronize(context.Background(), pods, containers); err != nil {
		t.Fatalf("Synchronize returned error: %v", err)
	}
	p.Flush() // wait for async remediation

	if got := fake.stoppedIDs(); !slices.Equal(got, []string{"c1"}) {
		t.Fatalf("stopped = %v, want [c1]: an uninjected container must not escape enforcement because of a compute-mode typo", got)
	}
}

func TestSynchronizeNoEnforcementWhenDisabled(t *testing.T) {
	fake := &fakeStopper{}
	// Default plugin: RetroactiveEnforcement is false.
	p := newTestPlugin(t)
	pods, containers := syncSnapshot()

	if _, err := p.Synchronize(context.Background(), pods, containers); err != nil {
		t.Fatalf("Synchronize returned error: %v", err)
	}
	p.Flush()

	if got := len(fake.stoppedIDs()); got != 0 {
		t.Fatalf("expected no stops when disabled, got %d", got)
	}
}

func TestNewPluginErrorsWhenEnabledWithoutStopper(t *testing.T) {
	// RetroactiveEnforcement set but nil stopper is a misconfiguration and must
	// surface as an error rather than silently disabling enforcement.
	log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix:       configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory:       configuration.DefaultMPSPipeDirectory,
		MapDir:                 t.TempDir(),
		RetroactiveEnforcement: true,
		Log:                    log,
	}, nil)
	if err == nil {
		t.Fatal("expected error when enforcement enabled without a stopper, got nil")
	}
	if p != nil {
		t.Fatalf("expected nil plugin on error, got %+v", p)
	}
}
