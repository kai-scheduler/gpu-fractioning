// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/containerd/nri/pkg/api"
	"github.com/containerd/nri/pkg/stub"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/mpsdrain"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/injection"
	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/daemonpaths"
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
	log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn}))
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
		t.Error("no warning was logged; fail-open silently degrades enforcement (a memory annotation it cannot parse leaves the container with no GPU limits at all), " +
			"so this log line is the only signal an operator gets that it happened")
	}
}

func TestCreateContainer_FailOpenFallsBackToTimeSlicingOnInvalidComputeMode(t *testing.T) {
	var logBuf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn}))
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
		t.Error("no warning was logged; fail-open silently degrades enforcement (the container was silently downgraded to time-slicing), " +
			"so this log line is the only signal an operator gets that it happened")
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
		log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn}))
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
			t.Error("no warning was logged; fail-open silently degrades enforcement (the container asked for sm-sharing on a cluster that has it disabled), " +
				"so this log line is the only signal an operator gets that it happened")
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
		FlushTimeout:     testFlushTimeout,
		Log:              log,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	flushBeforeCleanup(t, p)
	return p
}

// flushBeforeCleanup drains the plugin's mapping writes before the test's
// temporary directories are removed.
//
// CreateContainer records the container→pod mapping on a background worker,
// deliberately: the NRI hot path must never block on a filesystem write. A test
// that returns without draining it therefore leaves a write racing t.TempDir's
// RemoveAll, which surfaces as "TempDir RemoveAll cleanup: directory not empty"
// — a failure in whichever test happened to lose the race, with nothing in it
// pointing at the mapping worker. Cleanups run last-registered-first and
// t.TempDir registered its own before this one, so this always runs first.
func flushBeforeCleanup(t *testing.T, p *Plugin) {
	t.Helper()
	t.Cleanup(func() {
		if !p.Flush() {
			t.Errorf("the mapping flush did not complete within %v; the event processor is stuck", testFlushTimeout)
		}
	})
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

// reset clears the record between two audits of the same plugin.
func (f *fakeStopper) reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped = nil
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

// ---------------------------------------------------------------------------
// MPS-enforced caps: CUDA_MPS_PINNED_DEVICE_MEM_LIMIT and
// CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
// ---------------------------------------------------------------------------

const (
	limitAnnotation   = "nvidia.com/container.trainer.gpu-memory.limit"
	requestAnnotation = "nvidia.com/container.trainer.gpu-memory.request"
	devicesAnnotation = "nvidia.com/container.trainer.gpus.devices"
	modeAnnotation    = "nvidia.com/container.trainer.gpu-compute.mode"
	portionAnnotation = "nvidia.com/container.trainer.gpu-compute.portion"
)

// TestCreateContainerInjectsMPSEnforcementCaps pins exactly which env vars the
// create hook sets for each shape of pod annotation. The assertion is on the
// whole set, not on individual keys, because both directions are bugs: a
// missing cap is a tenant that can take memory or SMs from its neighbours on
// the same GPU, and a cap invented out of nothing (e.g. naming device ordinals
// the scheduler never assigned) caps the wrong hardware.
func TestCreateContainerInjectsMPSEnforcementCaps(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		wantEnv     map[string]string
		// wantAbsentEnv names keys the adjustment must NOT set. A cap that is
		// deliberately not applied has to be asserted positively: "the map does
		// not mention it" is also what a test that forgot to look would say.
		wantAbsentEnv   []string
		wantMountSource string
		wantMountDest   string
	}{
		{
			// The whole point of the pinned limit: MPS enforces it itself, so
			// it works on clusters whose container toolkit is too old to ship
			// the apply-cuda-memory-limits CDI hook that NVIDIA_GPU_MEMORY_LIMIT
			// depends on.
			name: "one assigned device is capped by ordinal",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:        "4096",
				injection.EnvGPUMemoryLimit:          "4096",
				injection.EnvMPSPinnedDeviceMemLimit: "0=4096M",
				injection.EnvVisibleDevices:          "GPU-abc123",
				injection.EnvMPSPipeDirectory:        configuration.DefaultMPSPipeDirectory,
			},
		},
		{
			// The cap is per GPU, and MPS only applies it to devices the list
			// names. A two-GPU container capped on device 0 alone can consume
			// all of device 1, which belongs to somebody else.
			name: "both assigned devices are named",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123,GPU-def456",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:        "4096",
				injection.EnvGPUMemoryLimit:          "4096",
				injection.EnvMPSPinnedDeviceMemLimit: "0=4096M,1=4096M",
				injection.EnvVisibleDevices:          "GPU-abc123,GPU-def456",
				injection.EnvMPSPipeDirectory:        configuration.DefaultMPSPipeDirectory,
			},
		},
		{
			name: "all eight devices of a full node are named",
			annotations: map[string]string{
				limitAnnotation:   "10Gi",
				devicesAnnotation: "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:        "10240",
				injection.EnvGPUMemoryLimit:          "10240",
				injection.EnvMPSPinnedDeviceMemLimit: "0=10240M,1=10240M,2=10240M,3=10240M,4=10240M,5=10240M,6=10240M,7=10240M",
				injection.EnvVisibleDevices:          "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7",
				injection.EnvMPSPipeDirectory:        configuration.DefaultMPSPipeDirectory,
			},
		},
		{
			// Without a device assignment the container's GPU ordinals are
			// unknown, so there is nothing to name in the per-device list. The
			// driver-level NVIDIA_GPU_MEMORY_LIMIT is still injected and is the
			// binding cap in that case; guessing "0=" here would cap whichever
			// GPU the container happened to enumerate first.
			name:        "no device assignment means no pinned memory cap",
			annotations: map[string]string{limitAnnotation: "4Gi"},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest: "4096",
				injection.EnvGPUMemoryLimit:   "4096",
				injection.EnvMPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
			},
		},
		{
			// The compute cap is what makes "half a GPU" mean half the SMs and
			// not just half the memory. It is scoped to sm-sharing: see
			// "time-slicing is never compute-capped" below.
			name: "compute portion becomes the MPS thread percentage",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				portionAnnotation: "0.5",
				modeAnnotation:    "sm-sharing",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:          "4096",
				injection.EnvGPUMemoryLimit:            "4096",
				injection.EnvMPSActiveThreadPercentage: "50",
				injection.EnvMPSPipeDirectory:          configuration.ContainerMPSPipeDirectory,
			},
			wantMountSource: filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath),
			wantMountDest:   configuration.ContainerMPSPipeDirectory,
		},
		{
			name: "a whole-GPU portion is 100 percent",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				portionAnnotation: "1",
				modeAnnotation:    "sm-sharing",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:          "4096",
				injection.EnvGPUMemoryLimit:            "4096",
				injection.EnvMPSActiveThreadPercentage: "100",
				injection.EnvMPSPipeDirectory:          configuration.ContainerMPSPipeDirectory,
			},
			wantMountSource: filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath),
			wantMountDest:   configuration.ContainerMPSPipeDirectory,
		},
		{
			// MPS reads 0 as "unlimited", so the smallest portion must floor at
			// 1 rather than round to nothing.
			name: "a portion too small to round still caps at 1 percent",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				portionAnnotation: "0.001",
				modeAnnotation:    "sm-sharing",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:          "4096",
				injection.EnvGPUMemoryLimit:            "4096",
				injection.EnvMPSActiveThreadPercentage: "1",
				injection.EnvMPSPipeDirectory:          configuration.ContainerMPSPipeDirectory,
			},
			wantMountSource: filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath),
			wantMountDest:   configuration.ContainerMPSPipeDirectory,
		},
		{
			// No portion annotated is the pre-existing world: memory capped,
			// compute uncapped. It must not silently become "100", which would
			// look like a cap while enforcing nothing.
			name:        "no compute portion means no thread percentage at all",
			annotations: map[string]string{limitAnnotation: "4Gi"},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest: "4096",
				injection.EnvGPUMemoryLimit:   "4096",
				injection.EnvMPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
			},
		},
		{
			// Memory enforcement must not depend on the compute mode. The two
			// cases below are the same pod annotated for the two modes; the MPS
			// pipe location and the compute cap are the only differences.
			//
			// time-slicing gets the memory caps and NO compute cap, even with a
			// portion annotated. The cap is a hard SM partition, so applying it
			// here would stop a lone time-slicing tenant from using an idle
			// card — the opposite of what time-slicing is for.
			name: "time-slicing is never compute-capped",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123",
				portionAnnotation: "0.25",
				modeAnnotation:    "time-slicing",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:        "4096",
				injection.EnvGPUMemoryLimit:          "4096",
				injection.EnvMPSPinnedDeviceMemLimit: "0=4096M",
				injection.EnvVisibleDevices:          "GPU-abc123",
				injection.EnvMPSPipeDirectory:        configuration.DefaultMPSPipeDirectory,
			},
			wantAbsentEnv: []string{injection.EnvMPSActiveThreadPercentage},
		},
		{
			// The default mode is time-slicing, so an annotated portion with no
			// mode is not capped either. This is the shape most pods have.
			name: "a portion with no mode annotation is not compute-capped",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				portionAnnotation: "0.25",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest: "4096",
				injection.EnvGPUMemoryLimit:   "4096",
				injection.EnvMPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
			},
			wantAbsentEnv: []string{injection.EnvMPSActiveThreadPercentage},
		},
		{
			name: "sm-sharing gets both caps",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123",
				portionAnnotation: "0.25",
				modeAnnotation:    "sm-sharing",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:          "4096",
				injection.EnvGPUMemoryLimit:            "4096",
				injection.EnvMPSPinnedDeviceMemLimit:   "0=4096M",
				injection.EnvMPSActiveThreadPercentage: "25",
				injection.EnvVisibleDevices:            "GPU-abc123",
				injection.EnvMPSPipeDirectory:          configuration.ContainerMPSPipeDirectory,
			},
			wantMountSource: filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath),
			wantMountDest:   configuration.ContainerMPSPipeDirectory,
		},
		{
			// The one shape that tells the two memory annotations apart. Every
			// other case here has request == limit (either annotated that way
			// or defaulted from the other), so all of them pass just as well
			// with the pinned cap derived from the request. This one does not:
			// MPS enforces CUDA_MPS_PINNED_DEVICE_MEM_LIMIT as a hard ceiling,
			// so building it from the request would cap a container at 2Gi that
			// the scheduler admitted at 4Gi — an allocation the workload was
			// charged for and cannot use, failing with an out-of-memory error
			// the pod spec does not explain.
			name: "a burstable container is pinned at its limit, not its request",
			annotations: map[string]string{
				requestAnnotation: "2Gi",
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123,GPU-def456",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:        "2048",
				injection.EnvGPUMemoryLimit:          "4096",
				injection.EnvMPSPinnedDeviceMemLimit: "0=4096M,1=4096M",
				injection.EnvVisibleDevices:          "GPU-abc123,GPU-def456",
				injection.EnvMPSPipeDirectory:        configuration.DefaultMPSPipeDirectory,
			},
		},
		{
			// A request-only pod has its limit defaulted from the request, and
			// the pinned cap must use that defaulted value — otherwise the
			// container is capped at nothing while its neighbours are capped.
			name: "request-only container is pinned at the defaulted limit",
			annotations: map[string]string{
				requestAnnotation: "2Gi",
				devicesAnnotation: "GPU-abc123",
			},
			wantEnv: map[string]string{
				injection.EnvGPUMemoryRequest:        "2048",
				injection.EnvGPUMemoryLimit:          "2048",
				injection.EnvMPSPinnedDeviceMemLimit: "0=2048M",
				injection.EnvVisibleDevices:          "GPU-abc123",
				injection.EnvMPSPipeDirectory:        configuration.DefaultMPSPipeDirectory,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestPlugin(t)
			pod := &api.PodSandbox{Name: "test-pod", Annotations: tt.annotations}
			ctr := &api.Container{Name: "trainer"}

			adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if adj == nil {
				t.Fatal("expected non-nil adjustment, got nil")
			}

			assertAdjustmentSets(t, adj, tt.wantEnv)
			for _, absent := range tt.wantAbsentEnv {
				for _, entry := range adjustmentEnv(adj) {
					if !entry.removal && entry.key == absent {
						t.Errorf("adjustment sets %s = %q, but this container must not carry that cap", absent, entry.value)
					}
				}
			}

			wantSource := tt.wantMountSource
			if wantSource == "" {
				wantSource = configuration.DefaultMPSPipeDirectory
			}
			wantDest := tt.wantMountDest
			if wantDest == "" {
				wantDest = configuration.DefaultMPSPipeDirectory
			}
			if len(adj.Mounts) != 1 {
				t.Fatalf("expected exactly one mount, got %d: %+v", len(adj.Mounts), adj.Mounts)
			}
			if adj.Mounts[0].Source != wantSource || adj.Mounts[0].Destination != wantDest {
				t.Errorf("mount = %q -> %q, want %q -> %q",
					adj.Mounts[0].Source, adj.Mounts[0].Destination, wantSource, wantDest)
			}
		})
	}
}

// TestCreateContainerInjectedCapsOverrideContainerProvidedValues is the bypass
// test. A pod author who writes CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 (or a
// huge NVIDIA_GPU_MEMORY_LIMIT, or a pinned limit naming an absurd size) into
// their own container spec is trying to hand themselves more of a shared GPU
// than the scheduler charged them for. Where the hook has a value of its own,
// that value must win — exactly once, with the container's copy gone.
func TestCreateContainerInjectedCapsOverrideContainerProvidedValues(t *testing.T) {
	p := newTestPlugin(t)
	pod := &api.PodSandbox{
		Name: "test-pod",
		Annotations: map[string]string{
			limitAnnotation:   "4Gi",
			devicesAnnotation: "GPU-abc123,GPU-def456",
			portionAnnotation: "0.25",
			// sm-sharing, because that is the only mode a compute cap applies
			// in — and the mode in which a pod author setting the env var
			// themselves would actually be trying to escape something.
			modeAnnotation: "sm-sharing",
		},
	}
	ctr := &api.Container{
		Name: "trainer",
		Env: []string{
			injection.EnvMPSActiveThreadPercentage + "=100",
			injection.EnvGPUMemoryLimit + "=999999",
			injection.EnvGPUMemoryRequest + "=999999",
			injection.EnvMPSPinnedDeviceMemLimit + "=0=999999M,1=999999M",
			injection.EnvVisibleDevices + "=all",
			"UNRELATED=keep-me",
		},
	}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if adj == nil {
		t.Fatal("expected non-nil adjustment, got nil")
	}

	want := map[string]string{
		injection.EnvGPUMemoryRequest:          "4096",
		injection.EnvGPUMemoryLimit:            "4096",
		injection.EnvMPSPinnedDeviceMemLimit:   "0=4096M,1=4096M",
		injection.EnvMPSActiveThreadPercentage: "25",
		injection.EnvVisibleDevices:            "GPU-abc123,GPU-def456",
	}

	entries := adjustmentEnv(adj)
	for key, wantValue := range want {
		// The removal has to come first: the runtime resolves the adjustment's
		// env list last-entry-wins per key, so an add followed by a removal
		// would delete the variable instead of overriding it.
		removeAt := indexOfRemoval(entries, key)
		addAt := indexOfAdd(entries, key)
		if removeAt < 0 {
			t.Errorf("%s: no removal of the container's own value was issued", key)
		}
		if addAt < 0 {
			t.Fatalf("%s: the adjustment never sets our value", key)
		}
		if removeAt >= 0 && removeAt > addAt {
			t.Errorf("%s: removal at index %d comes after the add at %d; the runtime would drop the variable", key, removeAt, addAt)
		}
		if n := countAdds(entries, key); n != 1 {
			t.Errorf("%s: adjustment sets the key %d times, want exactly 1", key, n)
		}
		if got := valueOfAdd(entries, key); got != wantValue {
			t.Errorf("%s = %q, want %q", key, got, wantValue)
		}
	}

	// And the end state the runtime actually produces: our value, once, with
	// the container's own copy gone and unrelated variables untouched.
	applied := applyAdjustment(ctr, adj)
	for key, wantValue := range want {
		values := envValues(applied, key)
		if len(values) != 1 {
			t.Errorf("after applying the adjustment, %s appears %d times: %v", key, len(values), values)
			continue
		}
		if values[0] != wantValue {
			t.Errorf("after applying the adjustment, %s = %q, want %q", key, values[0], wantValue)
		}
	}
	if values := envValues(applied, "UNRELATED"); len(values) != 1 || values[0] != "keep-me" {
		t.Errorf("an unrelated env var was disturbed: %v", values)
	}
}

// TestCreateContainerOverridesBareEnvKey covers the OCI form the presence check
// is easiest to get wrong: an env entry with no '=' at all, which is a
// legitimate way to declare a variable. Missing it would mean no removal is
// issued and — when another NRI plugin has claimed the same key — the whole
// adjustment is rejected, leaving the container with none of our limits.
func TestCreateContainerOverridesBareEnvKey(t *testing.T) {
	p := newTestPlugin(t)
	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}
	ctr := &api.Container{
		Name: "trainer",
		Env:  []string{injection.EnvMPSActiveThreadPercentage},
	}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	entries := adjustmentEnv(adj)
	if indexOfRemoval(entries, injection.EnvMPSActiveThreadPercentage) < 0 {
		t.Errorf("a bare %q declaration was not recognized as a pre-existing value", injection.EnvMPSActiveThreadPercentage)
	}
	if got := valueOfAdd(entries, injection.EnvMPSActiveThreadPercentage); got != "50" {
		t.Errorf("%s = %q, want %q", injection.EnvMPSActiveThreadPercentage, got, "50")
	}
}

// TestCreateContainerLeavesSelfImposedCapsAlone is the inverse of the bypass
// test, and it guards against over-reach. Where the hook has no value of its
// own — no compute portion annotated, or no device assignment to enumerate —
// the container's own setting is the only limit in play. Stripping it and
// putting nothing back would take a workload that voluntarily ran at 25% of the
// SMs and set it loose on the whole card.
func TestCreateContainerLeavesSelfImposedCapsAlone(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		key         string
		ownValue    string
	}{
		{
			name:        "no compute portion annotated",
			annotations: map[string]string{limitAnnotation: "4Gi", devicesAnnotation: "GPU-abc123"},
			key:         injection.EnvMPSActiveThreadPercentage,
			ownValue:    "25",
		},
		{
			name:        "no device assignment to enumerate",
			annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5"},
			key:         injection.EnvMPSPinnedDeviceMemLimit,
			ownValue:    "0=1024M",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestPlugin(t)
			pod := &api.PodSandbox{Name: "test-pod", Annotations: tt.annotations}
			ctr := &api.Container{Name: "trainer", Env: []string{tt.key + "=" + tt.ownValue}}

			adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if adj == nil {
				t.Fatal("expected non-nil adjustment, got nil")
			}

			entries := adjustmentEnv(adj)
			if at := indexOfRemoval(entries, tt.key); at >= 0 {
				t.Errorf("%s was removed even though the hook has no value to put back", tt.key)
			}
			if at := indexOfAdd(entries, tt.key); at >= 0 {
				t.Errorf("%s was set to %q, but nothing should have been injected for it", tt.key, valueOfAdd(entries, tt.key))
			}

			applied := applyAdjustment(ctr, adj)
			if values := envValues(applied, tt.key); len(values) != 1 || values[0] != tt.ownValue {
				t.Errorf("the container's own %s = %v, want it left as %q", tt.key, values, tt.ownValue)
			}
		})
	}
}

// TestCreateContainerComputePortionFailurePolicy covers both halves of the
// fail-open switch for an unusable compute portion. Fail-closed refuses to
// create the container at all; fail-open creates it with no compute cap but
// with everything else intact — a portion typo must not also cost the container
// its memory limit, which is the enforcement that was already working.
func TestCreateContainerComputePortionFailurePolicy(t *testing.T) {
	badPortions := []string{"abc", "", "  ", "0", "-0.5", "1.5", "50", "NaN", "Inf"}

	t.Run("fail-closed blocks the container", func(t *testing.T) {
		for _, portion := range badPortions {
			t.Run(portion, func(t *testing.T) {
				p := newTestPlugin(t)
				pod := &api.PodSandbox{
					Name:        "test-pod",
					Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: portion},
				}
				adj, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Name: "trainer"})
				if err == nil {
					t.Fatalf("expected creation to be blocked for portion %q, got adj=%+v", portion, adj)
				}
				if adj != nil {
					t.Errorf("expected no adjustment alongside the error, got %+v", adj)
				}
			})
		}
	})

	t.Run("fail-open injects everything except the compute cap", func(t *testing.T) {
		var logBuf bytes.Buffer
		log := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelWarn}))
		p, err := NewPlugin(Config{
			AnnotationPrefix: configuration.DefaultAnnotationPrefix,
			MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
			FailOpen:         true,
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
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123,GPU-def456",
				portionAnnotation: "not-a-number",
			},
		}

		adj, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Name: "trainer"})
		if err != nil {
			t.Fatalf("fail-open must not block the container, got: %v", err)
		}
		if adj == nil {
			t.Fatal("expected a non-nil adjustment, got nil")
		}

		assertAdjustmentSets(t, adj, map[string]string{
			injection.EnvGPUMemoryRequest:        "4096",
			injection.EnvGPUMemoryLimit:          "4096",
			injection.EnvMPSPinnedDeviceMemLimit: "0=4096M,1=4096M",
			injection.EnvVisibleDevices:          "GPU-abc123,GPU-def456",
			injection.EnvMPSPipeDirectory:        configuration.DefaultMPSPipeDirectory,
		})
		if len(adj.Mounts) != 1 {
			t.Errorf("expected the MPS mount to be injected anyway, got %+v", adj.Mounts)
		}
		if logBuf.Len() == 0 {
			t.Error("no warning was logged; fail-open silently degrades enforcement (the container is running uncapped on the SMs its portion was meant to limit), " +
				"so this log line is the only signal an operator gets that it happened")
		}
	})
}

// ---------------------------------------------------------------------------
// StopContainer: drain MPS clients before the runtime kills them.
// ---------------------------------------------------------------------------

// fakeDrainer stands in for mpsd's drain endpoint. It records every call and
// can be told to fail, to report a wedged MPS server, or to block until its
// context expires.
type fakeDrainer struct {
	mu        sync.Mutex
	calls     []mpsdrain.Request
	deadlines []time.Duration // time remaining on the call context, per call

	resp  mpsdrain.Response
	err   error
	block bool

	// journal, when set, records this call against a shared ordering log the
	// namespace tests use to assert a release lands after a drain.
	journal *[]string
	jmu     *sync.Mutex
}

func (f *fakeDrainer) Drain(ctx context.Context, req mpsdrain.Request) (mpsdrain.Response, error) {
	f.mu.Lock()
	f.calls = append(f.calls, req)
	remaining := time.Duration(-1)
	if deadline, ok := ctx.Deadline(); ok {
		remaining = time.Until(deadline)
	}
	f.deadlines = append(f.deadlines, remaining)
	block := f.block
	resp, err := f.resp, f.err
	f.mu.Unlock()
	if f.journal != nil {
		f.jmu.Lock()
		*f.journal = append(*f.journal, "drain")
		f.jmu.Unlock()
	}

	if block {
		<-ctx.Done()
		return mpsdrain.Response{}, ctx.Err()
	}
	return resp, err
}

func (f *fakeDrainer) requests() []mpsdrain.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]mpsdrain.Request(nil), f.calls...)
}

func (f *fakeDrainer) lastDeadline(t *testing.T) time.Duration {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.deadlines) == 0 {
		t.Fatal("the drainer was never called")
	}
	return f.deadlines[len(f.deadlines)-1]
}

// drainingPlugin builds a plugin whose StopContainer hook talks to the given
// fake drainer. requestTimeout of 0 leaves the NRI timeout source unset, which
// is how the plugin runs before its first Configure handshake.
func drainingPlugin(t *testing.T, drainer Drainer, drainTimeout, requestTimeout time.Duration) *Plugin {
	t.Helper()
	log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	cfg := Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		SupportSMSharing: true,
		MapDir:           t.TempDir(),
		Log:              log,
		Drainer:          drainer,
		DrainTimeout:     drainTimeout,
	}
	if requestTimeout > 0 {
		cfg.RequestTimeout = func() time.Duration { return requestTimeout }
	}
	p, err := NewPlugin(cfg, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

func fractionalPod() *api.PodSandbox {
	return &api.PodSandbox{
		Name:        "test-pod",
		Namespace:   "team-a",
		Annotations: map[string]string{limitAnnotation: "4Gi"},
	}
}

// TestStopContainerDrainsFractionalContainers checks the identity mpsd is given.
// mpsd matches MPS clients to the container by id, so a wrong or empty id
// drains nothing (or, worse, somebody else's clients); pod and namespace are
// what make the resulting log line actionable.
func TestStopContainerDrainsFractionalContainers(t *testing.T) {
	drainer := &fakeDrainer{resp: mpsdrain.Response{Drained: []int{4242}}}
	p := drainingPlugin(t, drainer, 0, time.Second)

	ctr := &api.Container{Id: "ctr-1", Name: "trainer"}
	updates, err := p.StopContainer(context.Background(), fractionalPod(), ctr)
	if err != nil {
		t.Fatalf("StopContainer returned an error: %v", err)
	}
	if updates != nil {
		t.Errorf("StopContainer returned container updates %+v; the hook must not mutate on stop", updates)
	}

	got := drainer.requests()
	if len(got) != 1 {
		t.Fatalf("expected exactly one drain call, got %d: %+v", len(got), got)
	}
	want := mpsdrain.Request{ContainerID: "ctr-1", Pod: "test-pod", Namespace: "team-a"}
	if got[0] != want {
		t.Errorf("drain request = %+v, want %+v", got[0], want)
	}
}

// TestStopContainerSkipsNonFractionalContainers keeps the hook off containers
// that never touched MPS. Draining one is not just wasted work: mpsd would be
// asked about a container it has no clients for on every pod deletion in the
// cluster.
func TestStopContainerSkipsNonFractionalContainers(t *testing.T) {
	tests := []struct {
		name string
		pod  *api.PodSandbox
		ctr  *api.Container
	}{
		{
			name: "no GPU annotations at all",
			pod:  &api.PodSandbox{Name: "test-pod", Annotations: map[string]string{"other": "value"}},
			ctr:  &api.Container{Id: "ctr-1", Name: "trainer"},
		},
		{
			name: "nil annotations",
			pod:  &api.PodSandbox{Name: "test-pod"},
			ctr:  &api.Container{Id: "ctr-1", Name: "trainer"},
		},
		{
			// A sibling of a fractional container is not itself fractional.
			name: "sibling container without its own annotation",
			pod:  fractionalPod(),
			ctr:  &api.Container{Id: "ctr-2", Name: "sidecar"},
		},
		{
			// Compute mode alone never made the container fractional at create
			// time either, so there is nothing on MPS to drain.
			name: "compute mode without memory annotations",
			pod:  &api.PodSandbox{Name: "test-pod", Annotations: map[string]string{modeAnnotation: "sm-sharing"}},
			ctr:  &api.Container{Id: "ctr-1", Name: "trainer"},
		},
		{
			// Unparseable annotations mean the create hook never injected
			// anything (fail-closed) or skipped the container (fail-open).
			name: "unparseable memory annotation",
			pod:  &api.PodSandbox{Name: "test-pod", Annotations: map[string]string{limitAnnotation: "not-a-quantity"}},
			ctr:  &api.Container{Id: "ctr-1", Name: "trainer"},
		},
		{
			// No id means mpsd has nothing to match clients against; calling it
			// anyway would ask it to drain "everything with an empty id".
			name: "empty container id",
			pod:  fractionalPod(),
			ctr:  &api.Container{Name: "trainer"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			drainer := &fakeDrainer{}
			p := drainingPlugin(t, drainer, 0, time.Second)

			if _, err := p.StopContainer(context.Background(), tt.pod, tt.ctr); err != nil {
				t.Fatalf("StopContainer returned an error: %v", err)
			}
			if got := drainer.requests(); len(got) != 0 {
				t.Errorf("expected no drain call, got %+v", got)
			}
		})
	}
}

// TestStopContainerNeverFailsTheStop is the contract that keeps a GPU problem
// from becoming a scheduling problem. Whatever mpsd says — an error, a wedged
// server, a nil response — the runtime must still be allowed to stop the
// container. Returning an error here would leave the pod terminating forever
// and the node's capacity held by something that is already dead.
func TestStopContainerNeverFailsTheStop(t *testing.T) {
	tests := []struct {
		name    string
		drainer *fakeDrainer
	}{
		{
			name:    "drain endpoint returns an error",
			drainer: &fakeDrainer{err: errors.New("dial unix: connection refused")},
		},
		{
			name:    "MPS was already wedged",
			drainer: &fakeDrainer{resp: mpsdrain.Response{Wedged: true, Message: "terminate timed out"}},
		},
		{
			name:    "wedged and restarted",
			drainer: &fakeDrainer{resp: mpsdrain.Response{Wedged: true, Restarted: true}},
		},
		{
			name:    "nothing to drain",
			drainer: &fakeDrainer{resp: mpsdrain.Response{}},
		},
		{
			name:    "drained and restarted",
			drainer: &fakeDrainer{resp: mpsdrain.Response{Drained: []int{1, 2}, Restarted: true}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := drainingPlugin(t, tt.drainer, 0, time.Second)

			updates, err := p.StopContainer(context.Background(), fractionalPod(), &api.Container{Id: "ctr-1", Name: "trainer"})
			if err != nil {
				t.Fatalf("StopContainer must never fail the stop, got: %v", err)
			}
			if updates != nil {
				t.Errorf("expected no container updates, got %+v", updates)
			}
			if got := tt.drainer.requests(); len(got) != 1 {
				t.Errorf("expected exactly one drain call, got %d", len(got))
			}
		})
	}
}

// TestStopContainerNoOps covers the arguments that must not reach the drainer
// at all. A nil drainer is the supported "drain disabled" configuration
// (--mps-drain-socket unset); nil pod/container are defensive, because a panic
// inside an NRI callback takes the whole plugin down and every container
// created while it is down is created without limits.
func TestStopContainerNoOps(t *testing.T) {
	drainer := &fakeDrainer{}

	t.Run("nil drainer", func(t *testing.T) {
		p := drainingPlugin(t, nil, 0, time.Second)
		if _, err := p.StopContainer(context.Background(), fractionalPod(), &api.Container{Id: "ctr-1", Name: "trainer"}); err != nil {
			t.Fatalf("StopContainer with drain disabled returned an error: %v", err)
		}
	})

	t.Run("nil pod", func(t *testing.T) {
		p := drainingPlugin(t, drainer, 0, time.Second)
		if _, err := p.StopContainer(context.Background(), nil, &api.Container{Id: "ctr-1", Name: "trainer"}); err != nil {
			t.Fatalf("StopContainer with a nil pod returned an error: %v", err)
		}
		if got := drainer.requests(); len(got) != 0 {
			t.Errorf("expected no drain call, got %+v", got)
		}
	})

	t.Run("nil container", func(t *testing.T) {
		p := drainingPlugin(t, drainer, 0, time.Second)
		if _, err := p.StopContainer(context.Background(), fractionalPod(), nil); err != nil {
			t.Fatalf("StopContainer with a nil container returned an error: %v", err)
		}
		if got := drainer.requests(); len(got) != 0 {
			t.Errorf("expected no drain call, got %+v", got)
		}
	})
}

// ---------------------------------------------------------------------------
// StopContainer must fit inside the runtime's NRI request deadline.
// ---------------------------------------------------------------------------

// TestStopContainerDrainBudget pins the deadline the drain call is given.
//
// This is not a latency nicety. containerd wraps every call into an NRI plugin
// in a context with plugin_request_timeout (default 2s) and treats
// context.DeadlineExceeded as fatal: it logs "closing plugin" and drops the
// connection. A StopContainer hook that outlives that deadline therefore
// disconnects fractiond from NRI — and while it is disconnected, every newly
// created fractional container is created with no limits injected at all, then
// stopped by retroactive enforcement once the plugin reconnects. A slow drain
// would turn routine pod deletion into a cluster-wide enforcement outage, so
// the drain gets half the budget and the hook gives up rather than overrun.
func TestStopContainerDrainBudget(t *testing.T) {
	tests := []struct {
		name           string
		drainTimeout   time.Duration
		requestTimeout time.Duration
		want           time.Duration
	}{
		{
			// The dangerous default-of-a-default: with no negotiated value the
			// plugin must fall back to the library default, not to zero. A zero
			// budget produces an already-expired context and fails every drain
			// instantly, so MPS never gets the terminate that protects the GPU.
			name:         "unset NRI timeout falls back to the library default",
			drainTimeout: 0,
			want:         stub.DefaultRequestTimeout / 2,
		},
		{
			name:           "a configured drain timeout longer than the budget is clamped",
			drainTimeout:   15 * time.Second,
			requestTimeout: 2 * time.Second,
			want:           time.Second,
		},
		{
			// Operators can shorten the drain below the NRI-derived bound; that
			// choice has to be honored, since it is the only knob they have.
			name:           "a configured drain timeout shorter than the budget wins",
			drainTimeout:   250 * time.Millisecond,
			requestTimeout: 2 * time.Second,
			want:           250 * time.Millisecond,
		},
		{
			// A cluster that raised containerd's plugin_request_timeout gets a
			// correspondingly longer drain, which is why the value is read from
			// the stub rather than hardcoded.
			name:           "a raised NRI timeout raises the budget",
			drainTimeout:   0,
			requestTimeout: 10 * time.Second,
			want:           5 * time.Second,
		},
		{
			// A source that reports nothing useful is the same situation as no
			// source at all — never a zero deadline.
			name:           "a non-positive NRI timeout falls back to the default",
			drainTimeout:   0,
			requestTimeout: -1,
			want:           stub.DefaultRequestTimeout / 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			drainer := &fakeDrainer{}
			log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
			cfg := Config{
				AnnotationPrefix: configuration.DefaultAnnotationPrefix,
				MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
				MapDir:           t.TempDir(),
				Log:              log,
				Drainer:          drainer,
				DrainTimeout:     tt.drainTimeout,
			}
			if tt.requestTimeout != 0 {
				cfg.RequestTimeout = func() time.Duration { return tt.requestTimeout }
			}
			p, err := NewPlugin(cfg, nil)
			if err != nil {
				t.Fatalf("NewPlugin: %v", err)
			}

			if _, err := p.StopContainer(context.Background(), fractionalPod(), &api.Container{Id: "ctr-1", Name: "trainer"}); err != nil {
				t.Fatalf("StopContainer returned an error: %v", err)
			}

			got := drainer.lastDeadline(t)
			if got <= 0 {
				t.Fatalf("drain was given a deadline of %v; an expired context fails every drain before it starts", got)
			}
			// The measured remaining time is a hair below the budget (the call
			// itself takes time), never above it.
			if got > tt.want {
				t.Errorf("drain deadline = %v, want at most %v", got, tt.want)
			}
			if slack := tt.want - got; slack > tt.want/10 {
				t.Errorf("drain deadline = %v, want close to %v", got, tt.want)
			}
		})
	}
}

// TestStopContainerGivesUpOnASlowDrain is the behaviour the budget buys: mpsd
// hanging must not hang the NRI callback. The hook returns when its own
// deadline fires, with no error — mpsd keeps draining in the background after
// the caller walks away, and kubelet's termination grace period runs after this
// returns, so the drain still completes; what it does not get is the right to
// hold the runtime's request open until the runtime drops the plugin.
func TestStopContainerGivesUpOnASlowDrain(t *testing.T) {
	// The hook is expected back at requestTimeout/2 (drainBudget), so the
	// assertion below has a whole half of this to absorb scheduling jitter. It
	// is deliberately not a tighter bound: the exact budget is pinned on the
	// injected deadline by TestStopContainerDrainBudget, and the only thing
	// worth measuring on the clock here is "did the hook come back before the
	// runtime's deadline", which must not become a flake under -race.
	const requestTimeout = 400 * time.Millisecond

	drainer := &fakeDrainer{block: true}
	p := drainingPlugin(t, drainer, 0, requestTimeout)

	// The call runs on its own goroutine with a hard backstop. Without the
	// clamp the hook waits on the drainer forever (the runtime's deadline lives
	// in a context this test does not supply), and a hung test is a much worse
	// signal than a failed assertion.
	type result struct {
		updates []*api.ContainerUpdate
		err     error
		elapsed time.Duration
	}
	done := make(chan result, 1)
	go func() {
		start := time.Now()
		updates, err := p.StopContainer(context.Background(), fractionalPod(), &api.Container{Id: "ctr-1", Name: "trainer"})
		done <- result{updates: updates, err: err, elapsed: time.Since(start)}
	}()

	select {
	case got := <-done:
		if got.err != nil {
			t.Fatalf("a drain that timed out must not fail the container stop, got: %v", got.err)
		}
		if got.updates != nil {
			t.Errorf("expected no container updates, got %+v", got.updates)
		}
		if got.elapsed >= requestTimeout {
			t.Errorf("StopContainer took %v, at or past the runtime's own %v deadline; the runtime would drop the plugin", got.elapsed, requestTimeout)
		}
	case <-time.After(20 * requestTimeout):
		t.Fatalf("StopContainer did not return within %v; a hook that outlives the NRI request deadline disconnects the plugin", 10*requestTimeout)
	}
}

// TestSetRequestTimeoutSource covers the reconnect wiring. The connect loop
// re-points the plugin at each new stub, and clearing it must restore the
// default rather than leave a zero.
func TestSetRequestTimeoutSource(t *testing.T) {
	drainer := &fakeDrainer{}
	p := drainingPlugin(t, drainer, 0, 0)

	pod, ctr := fractionalPod(), &api.Container{Id: "ctr-1", Name: "trainer"}

	if _, err := p.StopContainer(context.Background(), pod, ctr); err != nil {
		t.Fatalf("StopContainer: %v", err)
	}
	if got, want := drainer.lastDeadline(t), stub.DefaultRequestTimeout/2; got > want || want-got > want/10 {
		t.Errorf("before any handshake, drain deadline = %v, want about %v", got, want)
	}

	// A reconnect negotiates a longer timeout.
	p.SetRequestTimeoutSource(func() time.Duration { return 10 * time.Second })
	if _, err := p.StopContainer(context.Background(), pod, ctr); err != nil {
		t.Fatalf("StopContainer: %v", err)
	}
	if got, want := drainer.lastDeadline(t), 5*time.Second; got > want || want-got > want/10 {
		t.Errorf("after the handshake, drain deadline = %v, want about %v", got, want)
	}

	// Clearing the source reverts to the default, not to zero.
	p.SetRequestTimeoutSource(nil)
	if _, err := p.StopContainer(context.Background(), pod, ctr); err != nil {
		t.Fatalf("StopContainer: %v", err)
	}
	if got, want := drainer.lastDeadline(t), stub.DefaultRequestTimeout/2; got > want || want-got > want/10 {
		t.Errorf("after clearing the source, drain deadline = %v, want about %v", got, want)
	}
}

// ---------------------------------------------------------------------------
// The create hook and the retroactive audit must agree, exactly.
// ---------------------------------------------------------------------------

// TestInjectionSatisfiesRetroactiveAudit is the drift guard between the two
// halves of enforcement. buildAdjustment decides what a fractional container
// gets; audit.detector decides, on every NRI reconnect, whether a running
// container has it. They derive that from the same annotations through separate
// code, so they can disagree — and both directions are damaging:
//
//   - the audit expecting more than the hook injects stops perfectly healthy
//     containers on every containerd restart, a self-inflicted outage;
//   - the audit expecting less lets an under-enforced container keep running,
//     which is the leak this whole mechanism exists to close.
//
// Rather than restating the expectations a third time, this test takes the
// hook's own output, applies it to the container the way the runtime would, and
// requires the audit to find nothing wrong with the result — for every
// combination of annotations the scheduler can produce.
func TestInjectionSatisfiesRetroactiveAudit(t *testing.T) {
	memory := []struct {
		name string
		ann  map[string]string
	}{
		{"limit only", map[string]string{limitAnnotation: "4Gi"}},
		{"request only", map[string]string{requestAnnotation: "2Gi"}},
		{"request and limit", map[string]string{requestAnnotation: "2Gi", limitAnnotation: "4Gi"}},
	}
	devices := []struct {
		name string
		ann  map[string]string
	}{
		{"no devices", nil},
		{"one device", map[string]string{devicesAnnotation: "GPU-abc123"}},
		{"two devices", map[string]string{devicesAnnotation: "GPU-abc123,GPU-def456"}},
		{"eight devices", map[string]string{devicesAnnotation: "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7"}},
	}
	portions := []struct {
		name string
		ann  map[string]string
	}{
		{"no portion", nil},
		{"half", map[string]string{portionAnnotation: "0.5"}},
		{"whole", map[string]string{portionAnnotation: "1"}},
		{"floored", map[string]string{portionAnnotation: "0.001"}},
	}
	modes := []struct {
		name string
		ann  map[string]string
	}{
		{"no mode", nil},
		{"time-slicing", map[string]string{modeAnnotation: "time-slicing"}},
		{"sm-sharing", map[string]string{modeAnnotation: "sm-sharing"}},
	}

	for _, mem := range memory {
		for _, dev := range devices {
			for _, portion := range portions {
				for _, mode := range modes {
					name := strings.Join([]string{mem.name, dev.name, portion.name, mode.name}, "/")
					t.Run(name, func(t *testing.T) {
						podAnnotations := mergeAnnotations(mem.ann, dev.ann, portion.ann, mode.ann)

						stopper := &fakeStopper{}
						// One plugin for both halves: the audit is only
						// meaningful if it runs with the very configuration the
						// create hook ran with.
						p := enforcingSMSharingPlugin(t, stopper)

						pod := &api.PodSandbox{
							Id: "p1", Name: "pod1", Namespace: "default",
							Annotations: podAnnotations,
						}
						ctr := &api.Container{
							Id: "c1", Name: "trainer", PodSandboxId: "p1",
							State: api.ContainerState_CONTAINER_RUNNING,
						}

						adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
						if err != nil {
							t.Fatalf("CreateContainer: %v", err)
						}
						if adj == nil {
							t.Fatal("expected a fractional container to be adjusted, got a nil adjustment")
						}

						// Negative control first: without the injection the
						// audit must flag this container. Otherwise a detector
						// that silently skipped everything would make the
						// positive assertion below vacuous.
						if stopped := auditSnapshot(t, p, stopper, pod, ctr); !slices.Equal(stopped, []string{"c1"}) {
							t.Fatalf("an uninjected container was not flagged (stopped = %v); the rest of this case proves nothing", stopped)
						}

						// Now the real assertion: the hook's own output, applied
						// as the runtime would apply it, satisfies the audit.
						injected := &api.Container{
							Id: "c1", Name: "trainer", PodSandboxId: "p1",
							State:  api.ContainerState_CONTAINER_RUNNING,
							Env:    applyAdjustment(ctr, adj),
							Mounts: adj.Mounts,
						}
						if stopped := auditSnapshot(t, p, stopper, pod, injected); len(stopped) != 0 {
							t.Errorf("the audit would stop a container the create hook just injected (stopped = %v); env = %v, mounts = %+v",
								stopped, injected.Env, injected.Mounts)
						}
					})
				}
			}
		}
	}
}

// auditSnapshot runs one NRI Synchronize round through the plugin's retroactive
// enforcement and returns the container ids it decided to stop.
func auditSnapshot(t *testing.T, p *Plugin, stopper *fakeStopper, pod *api.PodSandbox, ctr *api.Container) []string {
	t.Helper()
	stopper.reset()
	if _, err := p.Synchronize(context.Background(), []*api.PodSandbox{pod}, []*api.Container{ctr}); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	p.Flush()
	return stopper.stoppedIDs()
}

// enforcingSMSharingPlugin is enforcingPlugin with the sm-sharing chicken bit
// on, so the matrix can exercise that mode on both sides at once.
func enforcingSMSharingPlugin(t *testing.T, stopper *fakeStopper) *Plugin {
	t.Helper()
	log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix:       configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory:       configuration.DefaultMPSPipeDirectory,
		MapDir:                 t.TempDir(),
		RetroactiveEnforcement: true,
		SupportSMSharing:       true,
		Log:                    log,
	}, stopper)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

func mergeAnnotations(maps ...map[string]string) map[string]string {
	out := map[string]string{}
	for _, m := range maps {
		for k, v := range m {
			out[k] = v
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Adjustment helpers: read an api.ContainerAdjustment the way the runtime does.
// ---------------------------------------------------------------------------

// envEntry is one entry of an adjustment's env list, with the removal marker
// already decoded.
type envEntry struct {
	key     string
	value   string
	removal bool
}

// adjustmentEnv decodes adj.Env in order. Order matters: the runtime resolves
// the list last-entry-wins per key, so "remove then add" and "add then remove"
// mean opposite things.
func adjustmentEnv(adj *api.ContainerAdjustment) []envEntry {
	var out []envEntry
	for _, kv := range adj.GetEnv() {
		if key, marked := kv.IsMarkedForRemoval(); marked {
			out = append(out, envEntry{key: key, removal: true})
			continue
		}
		out = append(out, envEntry{key: kv.Key, value: kv.Value})
	}
	return out
}

func indexOfRemoval(entries []envEntry, key string) int {
	for i, e := range entries {
		if e.removal && e.key == key {
			return i
		}
	}
	return -1
}

func indexOfAdd(entries []envEntry, key string) int {
	for i, e := range entries {
		if !e.removal && e.key == key {
			return i
		}
	}
	return -1
}

func countAdds(entries []envEntry, key string) int {
	n := 0
	for _, e := range entries {
		if !e.removal && e.key == key {
			n++
		}
	}
	return n
}

func valueOfAdd(entries []envEntry, key string) string {
	if i := indexOfAdd(entries, key); i >= 0 {
		return entries[i].value
	}
	return ""
}

// assertAdjustmentSets requires the adjustment to set exactly the given
// key/value pairs — no more, no fewer, and each exactly once. Asserting the
// whole set is deliberate: a test that only checks the keys it names would pass
// for an adjustment that also injects a cap nobody asked for.
func assertAdjustmentSets(t *testing.T, adj *api.ContainerAdjustment, want map[string]string) {
	t.Helper()

	entries := adjustmentEnv(adj)
	got := map[string]string{}
	for _, e := range entries {
		if e.removal {
			continue
		}
		if _, dup := got[e.key]; dup {
			t.Errorf("adjustment sets %q more than once", e.key)
		}
		got[e.key] = e.value
	}

	for key, wantValue := range want {
		gotValue, ok := got[key]
		if !ok {
			t.Errorf("adjustment does not set %q (want %q)", key, wantValue)
			continue
		}
		if gotValue != wantValue {
			t.Errorf("adjustment sets %s = %q, want %q", key, gotValue, wantValue)
		}
	}
	for key, gotValue := range got {
		if _, expected := want[key]; !expected {
			t.Errorf("adjustment sets unexpected env %s = %q", key, gotValue)
		}
	}
}

// applyAdjustment returns the container's env after the runtime applies adj,
// mirroring github.com/containerd/nri/pkg/adaptation/result.go adjustEnv: every
// key the adjustment mentions — whether it marks it for removal or sets it — is
// dropped from the container's own env first, then the adjustment's set values
// are appended in order. Reproducing it here rather than only inspecting
// adj.Env is what lets a test assert the end state a tenant would actually see,
// including duplicates a careless adjustment would produce.
func applyAdjustment(ctr *api.Container, adj *api.ContainerAdjustment) []string {
	if adj == nil {
		return slices.Clone(ctr.GetEnv())
	}

	mentioned := map[string]bool{}
	var adds []envEntry
	for _, e := range adjustmentEnv(adj) {
		mentioned[e.key] = true
		if !e.removal {
			adds = append(adds, e)
		}
	}

	var out []string
	for _, kv := range ctr.GetEnv() {
		key := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			key = kv[:i]
		}
		if mentioned[key] {
			continue
		}
		out = append(out, kv)
	}
	for _, e := range adds {
		out = append(out, e.key+"="+e.value)
	}
	return out
}

// envValues returns every value the env list carries for key, so a test can
// tell "set once" from "set twice".
func envValues(env []string, key string) []string {
	var out []string
	prefix := key + "="
	for _, kv := range env {
		if kv == key {
			out = append(out, "")
			continue
		}
		if strings.HasPrefix(kv, prefix) {
			out = append(out, strings.TrimPrefix(kv, prefix))
		}
	}
	return out
}

// TestAuditNoticesEveryInjectedEnv closes the direction
// TestInjectionSatisfiesRetroactiveAudit cannot see. That test proves the audit
// does not expect *more* than the hook injects; this one proves it does not
// expect *less*. It takes the hook's own output and, one variable at a time,
// puts the container back the way it would look if that single piece of
// injection had been lost — which is exactly the shape of a container that came
// up while fractiond was down. Every one of them has to be caught. A limit the
// hook bothers to inject but the audit never looks for is a limit that can go
// missing in production and never be noticed.
func TestAuditNoticesEveryInjectedEnv(t *testing.T) {
	pods := []struct {
		name        string
		annotations map[string]string
	}{
		{
			name: "time-slicing with both caps",
			annotations: map[string]string{
				requestAnnotation: "2Gi",
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123,GPU-def456",
				portionAnnotation: "0.5",
				modeAnnotation:    "time-slicing",
			},
		},
		{
			name: "sm-sharing with both caps",
			annotations: map[string]string{
				limitAnnotation:   "4Gi",
				devicesAnnotation: "GPU-abc123",
				portionAnnotation: "0.25",
				modeAnnotation:    "sm-sharing",
			},
		},
	}

	for _, tc := range pods {
		t.Run(tc.name, func(t *testing.T) {
			stopper := &fakeStopper{}
			p := enforcingSMSharingPlugin(t, stopper)

			pod := &api.PodSandbox{Id: "p1", Name: "pod1", Namespace: "default", Annotations: tc.annotations}
			bare := &api.Container{
				Id: "c1", Name: "trainer", PodSandboxId: "p1",
				State: api.ContainerState_CONTAINER_RUNNING,
			}

			adj, _, err := p.CreateContainer(context.Background(), pod, bare)
			if err != nil {
				t.Fatalf("CreateContainer: %v", err)
			}
			fullEnv := applyAdjustment(bare, adj)

			// Sanity: the fully injected container is clean, so any stop below
			// is attributable to the one variable that was dropped.
			injected := &api.Container{
				Id: "c1", Name: "trainer", PodSandboxId: "p1",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    fullEnv,
				Mounts: adj.Mounts,
			}
			if stopped := auditSnapshot(t, p, stopper, pod, injected); len(stopped) != 0 {
				t.Fatalf("the fully injected container was flagged (stopped = %v)", stopped)
			}

			for _, entry := range fullEnv {
				key := entry
				if i := strings.IndexByte(entry, '='); i >= 0 {
					key = entry[:i]
				}
				t.Run(key, func(t *testing.T) {
					degraded := &api.Container{
						Id: "c1", Name: "trainer", PodSandboxId: "p1",
						State:  api.ContainerState_CONTAINER_RUNNING,
						Env:    withoutEnv(fullEnv, key),
						Mounts: adj.Mounts,
					}
					if stopped := auditSnapshot(t, p, stopper, pod, degraded); !slices.Equal(stopped, []string{"c1"}) {
						t.Errorf("the audit does not notice a missing %s (stopped = %v); it can be lost in production without anyone finding out", key, stopped)
					}
				})
			}
		})
	}
}

// withoutEnv returns env with every entry for key removed.
func withoutEnv(env []string, key string) []string {
	var out []string
	prefix := key + "="
	for _, kv := range env {
		if kv == key || strings.HasPrefix(kv, prefix) {
			continue
		}
		out = append(out, kv)
	}
	return out
}

// ---------------------------------------------------------------------------
// NVML shim delivery: the read-only bind mount of the staged shadow
// libnvidia-ml.so.1 and the LD_LIBRARY_PATH merge that makes the loader find
// it ahead of the driver's own copy.
// ---------------------------------------------------------------------------

// shimPlugin is newTestPlugin with the NVML shim host directory configured, the
// state fractiond reaches once staging has succeeded.
func shimPlugin(t *testing.T) *Plugin {
	t.Helper()
	log := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p, err := NewPlugin(Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		NVMLShimHostDir:  daemonpaths.NVMLShimDir,
		SupportSMSharing: true,
		MapDir:           t.TempDir(),
		Log:              log,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	return p
}

// mountTo returns the adjustment's mount at the given in-container destination,
// or nil.
func mountTo(adj *api.ContainerAdjustment, destination string) *api.Mount {
	for _, m := range adj.GetMounts() {
		if m.GetDestination() == destination {
			return m
		}
	}
	return nil
}

// TestCreateContainerDeliversTheNVMLShim covers the delivery mechanism end to
// end for each shape of LD_LIBRARY_PATH a container can arrive with.
//
// The merge cases are the ones that matter. A GPU container comes out of the
// NVIDIA container runtime with the runtime's own driver directories already in
// LD_LIBRARY_PATH, so the overwrite semantics used for the MPS caps would strip
// the workload of the libraries its CUDA stack resolves through — a far worse
// outcome than the capped-reporting bug the shim exists to fix.
func TestCreateContainerDeliversTheNVMLShim(t *testing.T) {
	const shimDir = injection.ContainerNVMLShimDir

	tests := []struct {
		name string
		// containerEnv is the container's own env, exactly as its spec declares it.
		containerEnv []string
		// wantValue is the LD_LIBRARY_PATH the adjustment must set; "" means the
		// adjustment must not mention the key at all.
		wantValue string
		// wantApplied is the value a tenant sees once the runtime has applied
		// the adjustment.
		wantApplied string
	}{
		{
			name:        "no existing value gets only the shim directory",
			wantValue:   shimDir,
			wantApplied: shimDir,
		},
		{
			// Observed on a live GPU pod under the NVIDIA container runtime
			// (RuntimeClass nvidia), before any fractiond injection. This is
			// what every GPU container on the node actually looks like, so it
			// is the case the merge has to get right.
			name:         "the NVIDIA container runtime's own paths survive in order",
			containerEnv: []string{injection.EnvLDLibraryPath + "=/usr/local/nvidia/lib:/usr/local/nvidia/lib64"},
			wantValue:    shimDir + ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
			wantApplied:  shimDir + ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
		},
		{
			name:         "a workload's own paths survive in order",
			containerEnv: []string{injection.EnvLDLibraryPath + "=/foo:/bar"},
			wantValue:    shimDir + ":/foo:/bar",
			wantApplied:  shimDir + ":/foo:/bar",
		},
		{
			// Idempotency. The create hook runs again on every container
			// restart, and a merge that prepended unconditionally would grow
			// the search path without bound.
			name:         "the shim directory is not prepended twice",
			containerEnv: []string{injection.EnvLDLibraryPath + "=" + shimDir + ":/usr/local/nvidia/lib"},
			wantValue:    "",
			wantApplied:  shimDir + ":/usr/local/nvidia/lib",
		},
		{
			// A bare declaration with no '=' is a legitimate OCI form and means
			// an empty value, so the merge has nothing to preserve.
			name:         "a bare declaration is replaced by the shim directory",
			containerEnv: []string{injection.EnvLDLibraryPath},
			wantValue:    shimDir,
			wantApplied:  shimDir,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := shimPlugin(t)
			pod := &api.PodSandbox{
				Name:        "test-pod",
				Annotations: map[string]string{limitAnnotation: "4Gi", devicesAnnotation: "GPU-abc123"},
			}
			ctr := &api.Container{Name: "trainer", Env: append(slices.Clone(tt.containerEnv), "UNRELATED=keep-me")}

			adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if adj == nil {
				t.Fatal("expected non-nil adjustment, got nil")
			}

			// The mount is unconditional: it is what puts the library on the
			// path the loader is about to be pointed at, so it must be there
			// even in the case where the env needs no change.
			mount := mountTo(adj, injection.ContainerNVMLShimDir)
			if mount == nil {
				t.Fatalf("no mount at %q; the shim library never reaches the container. mounts = %+v",
					injection.ContainerNVMLShimDir, adj.GetMounts())
			}
			if mount.GetSource() != daemonpaths.NVMLShimDir {
				t.Errorf("shim mount source = %q, want %q", mount.GetSource(), daemonpaths.NVMLShimDir)
			}
			if mount.GetType() != "bind" {
				t.Errorf("shim mount type = %q, want %q", mount.GetType(), "bind")
			}
			// Read-only: the library is shared by every container on the node,
			// and fractiond replaces it by rename rather than in place, so
			// nothing needs write access through this mount.
			if want := []string{"bind", "ro"}; !slices.Equal(mount.GetOptions(), want) {
				t.Errorf("shim mount options = %v, want %v", mount.GetOptions(), want)
			}

			entries := adjustmentEnv(adj)
			if tt.wantValue == "" {
				if at := indexOfAdd(entries, injection.EnvLDLibraryPath); at >= 0 {
					t.Errorf("%s was set to %q, but the container's value already leads with the shim directory",
						injection.EnvLDLibraryPath, valueOfAdd(entries, injection.EnvLDLibraryPath))
				}
				if at := indexOfRemoval(entries, injection.EnvLDLibraryPath); at >= 0 {
					t.Errorf("%s was removed even though nothing was put back", injection.EnvLDLibraryPath)
				}
			} else {
				if got := valueOfAdd(entries, injection.EnvLDLibraryPath); got != tt.wantValue {
					t.Errorf("%s = %q, want %q", injection.EnvLDLibraryPath, got, tt.wantValue)
				}
				if n := countAdds(entries, injection.EnvLDLibraryPath); n != 1 {
					t.Errorf("%s is set %d times, want exactly 1", injection.EnvLDLibraryPath, n)
				}
				// Removal before add, for the reason setEnforcedEnv documents:
				// generate.AdjustEnv is last-entry-wins per key, so the
				// reversed pair deletes the variable instead of rewriting it —
				// which here would strip the NVIDIA runtime's driver paths.
				if len(tt.containerEnv) > 0 {
					removeAt := indexOfRemoval(entries, injection.EnvLDLibraryPath)
					addAt := indexOfAdd(entries, injection.EnvLDLibraryPath)
					if removeAt < 0 {
						t.Errorf("%s: no removal of the container's own value was issued", injection.EnvLDLibraryPath)
					}
					if addAt < 0 {
						t.Fatalf("%s: the adjustment never sets our value", injection.EnvLDLibraryPath)
					}
					if removeAt >= 0 && removeAt > addAt {
						t.Errorf("%s: removal at index %d comes after the add at %d; the runtime would drop the variable entirely",
							injection.EnvLDLibraryPath, removeAt, addAt)
					}
				}
			}

			applied := applyAdjustment(ctr, adj)
			values := envValues(applied, injection.EnvLDLibraryPath)
			if len(values) != 1 {
				t.Fatalf("after applying the adjustment, %s appears %d times: %v", injection.EnvLDLibraryPath, len(values), values)
			}
			if values[0] != tt.wantApplied {
				t.Errorf("after applying the adjustment, %s = %q, want %q", injection.EnvLDLibraryPath, values[0], tt.wantApplied)
			}
			if v := envValues(applied, "UNRELATED"); len(v) != 1 || v[0] != "keep-me" {
				t.Errorf("an unrelated env var was disturbed: %v", v)
			}
		})
	}
}

// TestCreateContainerShimDisabledTouchesNothing is the other half of the
// feature switch, and of the failure path: staging leaves the host directory
// unconfigured when it fails, so this is also what a node whose fractiond image
// carries no shim library must look like. No mount, and LD_LIBRARY_PATH left
// exactly as the container declared it — the caps still work without the shim,
// only the reporting view degrades.
func TestCreateContainerShimDisabledTouchesNothing(t *testing.T) {
	const runtimePath = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

	p := newTestPlugin(t)
	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", devicesAnnotation: "GPU-abc123", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}
	ctr := &api.Container{Name: "trainer", Env: []string{injection.EnvLDLibraryPath + "=" + runtimePath}}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if adj == nil {
		t.Fatal("expected non-nil adjustment, got nil")
	}

	if m := mountTo(adj, injection.ContainerNVMLShimDir); m != nil {
		t.Errorf("a shim mount was added with the feature off: %+v", m)
	}

	entries := adjustmentEnv(adj)
	if at := indexOfAdd(entries, injection.EnvLDLibraryPath); at >= 0 {
		t.Errorf("%s was set to %q with the feature off", injection.EnvLDLibraryPath, valueOfAdd(entries, injection.EnvLDLibraryPath))
	}
	if at := indexOfRemoval(entries, injection.EnvLDLibraryPath); at >= 0 {
		t.Errorf("%s was removed with the feature off", injection.EnvLDLibraryPath)
	}

	// The caps are unaffected: turning the shim off costs the reporting view
	// and nothing else.
	assertAdjustmentSets(t, adj, map[string]string{
		injection.EnvGPUMemoryRequest:          "4096",
		injection.EnvGPUMemoryLimit:            "4096",
		injection.EnvMPSPinnedDeviceMemLimit:   "0=4096M",
		injection.EnvMPSActiveThreadPercentage: "50",
		injection.EnvVisibleDevices:            "GPU-abc123",
		injection.EnvMPSPipeDirectory:          configuration.ContainerMPSPipeDirectory,
	})

	applied := applyAdjustment(ctr, adj)
	if values := envValues(applied, injection.EnvLDLibraryPath); len(values) != 1 || values[0] != runtimePath {
		t.Errorf("the container's own %s = %v, want it left as %q", injection.EnvLDLibraryPath, values, runtimePath)
	}
}

// TestCreateContainerShimLeavesTheMPSPipeMountAlone guards the interaction
// between the two mounts. The MPS pipe mount is what the caps themselves ride
// on; a second mount added beside it must not displace it or change its
// read-write access.
func TestCreateContainerShimLeavesTheMPSPipeMountAlone(t *testing.T) {
	p := shimPlugin(t)
	pod := &api.PodSandbox{Name: "test-pod", Annotations: map[string]string{limitAnnotation: "4Gi"}}
	ctr := &api.Container{Name: "trainer"}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pipe := mountTo(adj, configuration.DefaultMPSPipeDirectory)
	if pipe == nil {
		t.Fatalf("the MPS pipe mount is gone; mounts = %+v", adj.GetMounts())
	}
	if pipe.GetSource() != configuration.DefaultMPSPipeDirectory {
		t.Errorf("MPS pipe mount source = %q, want %q", pipe.GetSource(), configuration.DefaultMPSPipeDirectory)
	}
	if want := []string{"bind", "rw"}; !slices.Equal(pipe.GetOptions(), want) {
		t.Errorf("MPS pipe mount options = %v, want %v", pipe.GetOptions(), want)
	}
	if len(adj.GetMounts()) != 2 {
		t.Errorf("expected exactly the MPS pipe and shim mounts, got %+v", adj.GetMounts())
	}
}

// TestNVMLShimMountSourceAgreesWithTheStagingPath closes the gap the
// daemonpaths package exists for. fractiond stages the library to
// daemonpaths.NVMLShimDir and the operator hostPath-mounts the same constant
// into its pod; if the create hook ever bind-mounted a different directory, the
// containers would map an empty path and the only symptom would be nvidia-smi
// continuing to report the whole GPU — indistinguishable from the feature not
// being deployed.
func TestNVMLShimMountSourceAgreesWithTheStagingPath(t *testing.T) {
	p := shimPlugin(t)
	pod := &api.PodSandbox{Name: "test-pod", Annotations: map[string]string{limitAnnotation: "4Gi"}}

	adj, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Name: "trainer"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mount := mountTo(adj, injection.ContainerNVMLShimDir)
	if mount == nil {
		t.Fatalf("no shim mount; mounts = %+v", adj.GetMounts())
	}
	if mount.GetSource() != daemonpaths.NVMLShimDir {
		t.Errorf("shim mount source = %q but fractiond stages to %q; containers would mount a directory with no library in it",
			mount.GetSource(), daemonpaths.NVMLShimDir)
	}
}

// ── per-container MPS namespaces ────────────────────────────────────────────

// fakeProvisioner stands in for mpsd's namespace endpoints. It records every
// call in order, including against the shared journal a test uses to assert
// that a release lands after a drain.
type fakeProvisioner struct {
	mu sync.Mutex

	pipeDirectory string
	namespace     string
	provisionErr  error
	releaseResp   mpsdrain.ReleaseResponse
	releaseErr    error
	block         bool

	provisions []mpsdrain.ProvisionRequest
	releases   []mpsdrain.ReleaseRequest
	deadlines  []time.Duration

	journal *[]string
	jmu     *sync.Mutex
}

func (f *fakeProvisioner) note(entry string) {
	if f.journal == nil {
		return
	}
	f.jmu.Lock()
	defer f.jmu.Unlock()
	*f.journal = append(*f.journal, entry)
}

func (f *fakeProvisioner) Provision(ctx context.Context, req mpsdrain.ProvisionRequest) (mpsdrain.ProvisionResponse, error) {
	f.mu.Lock()
	f.provisions = append(f.provisions, req)
	remaining := time.Duration(-1)
	if deadline, ok := ctx.Deadline(); ok {
		remaining = time.Until(deadline)
	}
	f.deadlines = append(f.deadlines, remaining)
	block, err := f.block, f.provisionErr
	dir, namespace := f.pipeDirectory, f.namespace
	f.mu.Unlock()
	f.note("provision")

	if block {
		<-ctx.Done()
		return mpsdrain.ProvisionResponse{}, ctx.Err()
	}
	if err != nil {
		return mpsdrain.ProvisionResponse{}, err
	}
	return mpsdrain.ProvisionResponse{
		PipeDirectory:       dir,
		Namespace:           namespace,
		Server:              "shared",
		ActiveThreadPercent: req.ActiveThreadPercent,
	}, nil
}

func (f *fakeProvisioner) Release(_ context.Context, req mpsdrain.ReleaseRequest) (mpsdrain.ReleaseResponse, error) {
	f.mu.Lock()
	f.releases = append(f.releases, req)
	resp, err := f.releaseResp, f.releaseErr
	f.mu.Unlock()
	f.note("release")
	return resp, err
}

func (f *fakeProvisioner) provisionRequests() []mpsdrain.ProvisionRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.provisions)
}

func (f *fakeProvisioner) releaseRequests() []mpsdrain.ReleaseRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.releases)
}

const testNamespacePipeDir = "/run/nvidia-mps/shared/kai_abc123_0011223344"

// newProvisioningPlugin builds a plugin wired to a namespace provisioner.
func newProvisioningPlugin(t *testing.T, prov *fakeProvisioner, cfg Config) *Plugin {
	t.Helper()
	if cfg.AnnotationPrefix == "" {
		cfg.AnnotationPrefix = configuration.DefaultAnnotationPrefix
	}
	if cfg.MPSPipeDirectory == "" {
		cfg.MPSPipeDirectory = configuration.DefaultMPSPipeDirectory
	}
	if cfg.MapDir == "" {
		cfg.MapDir = t.TempDir()
	}
	if cfg.Log == nil {
		cfg.Log = slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil))
	}
	if cfg.FlushTimeout == 0 {
		cfg.FlushTimeout = testFlushTimeout
	}
	cfg.Provisioner = prov

	p, err := NewPlugin(cfg, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}
	flushBeforeCleanup(t, p)
	return p
}

// TestSMSharingContainerIsMountedOnlyItsOwnNamespaceDirectory is the whole
// mechanism, end to end on the fractiond side.
//
// The container is bind-mounted the leaf namespace directory mpsd read back
// from the control daemon — not the shared server directory it used to get.
// That distinction is the enforcement: the server directory carries the
// `default` namespace, which is uncapped, so a container mounted there has no
// ceiling at all while looking correctly routed.
func TestSMSharingContainerIsMountedOnlyItsOwnNamespaceDirectory(t *testing.T) {
	prov := &fakeProvisioner{pipeDirectory: testNamespacePipeDir, namespace: "kai_abc123_0011223344"}
	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true})

	pod := &api.PodSandbox{
		Name:      "test-pod",
		Namespace: "team-a",
		Annotations: map[string]string{
			limitAnnotation:   "4Gi",
			devicesAnnotation: "GPU-abc123",
			portionAnnotation: "0.25",
			modeAnnotation:    "sm-sharing",
		},
	}
	ctr := &api.Container{Id: "container-id", Name: "trainer"}

	adj, _, err := p.CreateContainer(context.Background(), pod, ctr)
	if err != nil {
		t.Fatalf("CreateContainer() error = %v", err)
	}
	if adj == nil {
		t.Fatal("expected an adjustment")
	}

	if len(adj.Mounts) != 1 {
		t.Fatalf("mounts = %+v, want exactly one", adj.Mounts)
	}
	if adj.Mounts[0].Source != testNamespacePipeDir {
		t.Errorf("mount source = %q, want the namespace directory %q", adj.Mounts[0].Source, testNamespacePipeDir)
	}
	if adj.Mounts[0].Source == filepath.Dir(testNamespacePipeDir) {
		t.Error("the server directory was mounted; its default namespace is uncapped")
	}
	if adj.Mounts[0].Destination != configuration.ContainerMPSPipeDirectory {
		t.Errorf("mount destination = %q, want %q", adj.Mounts[0].Destination, configuration.ContainerMPSPipeDirectory)
	}

	entries := adjustmentEnv(adj)
	if got := valueOfAdd(entries, injection.EnvMPSPipeDirectory); got != configuration.ContainerMPSPipeDirectory {
		t.Errorf("%s = %q, want the fixed in-container path %q", injection.EnvMPSPipeDirectory, got, configuration.ContainerMPSPipeDirectory)
	}
	// The env var is still injected, and is still only advisory: it can lower
	// the container's share, never raise it past the namespace's ceiling.
	if got := valueOfAdd(entries, injection.EnvMPSActiveThreadPercentage); got != "25" {
		t.Errorf("%s = %q, want %q", injection.EnvMPSActiveThreadPercentage, got, "25")
	}

	requests := prov.provisionRequests()
	if len(requests) != 1 {
		t.Fatalf("provision requests = %d, want 1", len(requests))
	}
	if requests[0].ContainerID != "container-id" {
		t.Errorf("provision container id = %q, want the runtime's id", requests[0].ContainerID)
	}
	if requests[0].ActiveThreadPercent != 25 {
		t.Errorf("provisioned cap = %d%%, want 25%%", requests[0].ActiveThreadPercent)
	}
	if !slices.Equal(requests[0].GPUUUIDs, []string{"GPU-abc123"}) {
		t.Errorf("provisioned GPU UUIDs = %v, want [GPU-abc123]", requests[0].GPUUUIDs)
	}
	if requests[0].Pod != "test-pod" || requests[0].Namespace != "team-a" {
		t.Errorf("provision request = %+v, want the pod identity carried for logging", requests[0])
	}
}

// TestSMSharingWithNoPortionIsStillGivenItsOwnNamespace.
//
// A container with no annotated portion asked for no cap, which is 100% — not a
// reason to put it in the shared `default` namespace. Keeping `default` empty
// is what lets mpsd cap it low as defence in depth without penalising anyone
// who legitimately belongs there, because nobody does.
func TestSMSharingWithNoPortionIsStillGivenItsOwnNamespace(t *testing.T) {
	prov := &fakeProvisioner{pipeDirectory: testNamespacePipeDir, namespace: "kai_abc123_0011223344"}
	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true})

	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", modeAnnotation: "sm-sharing"},
	}
	adj, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "container-id", Name: "trainer"})
	if err != nil {
		t.Fatalf("CreateContainer() error = %v", err)
	}

	requests := prov.provisionRequests()
	if len(requests) != 1 {
		t.Fatalf("provision requests = %d, want 1", len(requests))
	}
	if requests[0].ActiveThreadPercent != 100 {
		t.Errorf("provisioned cap = %d%%, want 100%% for a container that asked for no cap", requests[0].ActiveThreadPercent)
	}
	if adj.Mounts[0].Source != testNamespacePipeDir {
		t.Errorf("mount source = %q, want its own namespace directory", adj.Mounts[0].Source)
	}
	// No portion annotated still means no injected env var: "100" there would
	// look like a cap while enforcing nothing.
	if got := valueOfAdd(adjustmentEnv(adj), injection.EnvMPSActiveThreadPercentage); got != "" {
		t.Errorf("%s = %q, want it absent", injection.EnvMPSActiveThreadPercentage, got)
	}
}

// TestProvisioningFailureBlocksTheContainer, including under fail-open.
//
// FailOpen is about malformed input: one bad pod versus a node full of stuck
// pods. This is the enforcement plane being unavailable, where creating the
// container anyway puts an uncapped tenant on a GPU other tenants are using —
// the exact outcome the namespace exists to prevent. A pod that will not start
// is visible and fixable; a pod that quietly took the whole card is neither.
func TestProvisioningFailureBlocksTheContainer(t *testing.T) {
	for _, failOpen := range []bool{false, true} {
		prov := &fakeProvisioner{provisionErr: errors.New("mpsd is not answering")}
		p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true, FailOpen: failOpen})

		pod := &api.PodSandbox{
			Name: "test-pod",
			Annotations: map[string]string{
				limitAnnotation:   "4Gi",
				portionAnnotation: "0.5",
				modeAnnotation:    "sm-sharing",
			},
		}
		adj, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "container-id", Name: "trainer"})
		if err == nil {
			t.Fatalf("failOpen=%v: CreateContainer() error = nil, want the container refused", failOpen)
		}
		if adj != nil {
			t.Errorf("failOpen=%v: an adjustment was returned alongside the error: %+v", failOpen, adj)
		}
		if !strings.Contains(err.Error(), "mpsd is not answering") {
			t.Errorf("failOpen=%v: error %v does not carry the reason", failOpen, err)
		}
	}
}

// TestProvisioningRejectsAnEmptyPipeDirectory: an mpsd that answered without a
// path would otherwise produce a bind mount from "", which the runtime resolves
// to something no one intended.
func TestProvisioningRejectsAnEmptyPipeDirectory(t *testing.T) {
	prov := &fakeProvisioner{pipeDirectory: ""}
	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true})

	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}
	if _, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "container-id", Name: "trainer"}); err == nil {
		t.Fatal("CreateContainer() error = nil, want a refusal when no pipe directory came back")
	}
}

// TestTimeSlicingNeverProvisionsANamespace: the cap does not apply in
// time-slicing mode, so there is nothing to provision — and provisioning anyway
// would put a hard SM partition on a container that is supposed to get whatever
// the card is not using.
func TestTimeSlicingNeverProvisionsANamespace(t *testing.T) {
	for _, annotations := range []map[string]string{
		{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "time-slicing"},
		{limitAnnotation: "4Gi", portionAnnotation: "0.5"}, // no mode: the default is time-slicing
		{limitAnnotation: "4Gi"},
	} {
		prov := &fakeProvisioner{pipeDirectory: testNamespacePipeDir}
		p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true})

		adj, _, err := p.CreateContainer(context.Background(),
			&api.PodSandbox{Name: "test-pod", Annotations: annotations},
			&api.Container{Id: "container-id", Name: "trainer"})
		if err != nil {
			t.Fatalf("annotations %v: CreateContainer() error = %v", annotations, err)
		}
		if n := len(prov.provisionRequests()); n != 0 {
			t.Errorf("annotations %v: %d namespaces provisioned for a time-slicing container, want 0", annotations, n)
		}
		if adj.Mounts[0].Source != configuration.DefaultMPSPipeDirectory {
			t.Errorf("annotations %v: mount source = %q, want the node's MPS pipe directory", annotations, adj.Mounts[0].Source)
		}
	}
}

// TestSMSharingKillSwitchProvisionsNothing: with --support-sm-sharing=false the
// shared MPS server the namespaces live on does not exist, so the annotation is
// rejected like any other invalid value and nothing is provisioned. Under
// fail-open the container is created as a time-slicing one — uncapped, which is
// what turning the feature off means.
func TestSMSharingKillSwitchProvisionsNothing(t *testing.T) {
	pod := &api.PodSandbox{
		Name: "test-pod",
		Annotations: map[string]string{
			limitAnnotation:   "4Gi",
			portionAnnotation: "0.5",
			modeAnnotation:    "sm-sharing",
		},
	}

	// Fail-closed: the container is refused, as it was before namespaces
	// existed.
	prov := &fakeProvisioner{pipeDirectory: testNamespacePipeDir}
	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: false})
	if _, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "c", Name: "trainer"}); err == nil {
		t.Error("CreateContainer() error = nil, want the sm-sharing annotation rejected on a cluster with it disabled")
	}
	if n := len(prov.provisionRequests()); n != 0 {
		t.Errorf("%d namespaces provisioned with sm-sharing disabled, want 0", n)
	}

	// Fail-open: the container is created, in time-slicing mode, with no
	// compute cap of any kind.
	prov = &fakeProvisioner{pipeDirectory: testNamespacePipeDir}
	p = newProvisioningPlugin(t, prov, Config{SupportSMSharing: false, FailOpen: true})
	adj, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "c", Name: "trainer"})
	if err != nil {
		t.Fatalf("CreateContainer() under fail-open error = %v", err)
	}
	if n := len(prov.provisionRequests()); n != 0 {
		t.Errorf("%d namespaces provisioned with sm-sharing disabled (fail-open), want 0", n)
	}
	if got := valueOfAdd(adjustmentEnv(adj), injection.EnvMPSActiveThreadPercentage); got != "" {
		t.Errorf("%s = %q with sm-sharing disabled, want no compute cap at all", injection.EnvMPSActiveThreadPercentage, got)
	}
}

// TestWithoutAProvisionerSMSharingFallsBackToTheSharedSocket pins the
// behaviour of the kill switch on the namespace mechanism itself
// (--namespace-isolation=false): the pre-namespace mount, and a cap that is
// only the advisory env var. It is not a configuration to want, but it must
// work, because it is the way out if the mechanism itself goes wrong.
func TestWithoutAProvisionerSMSharingFallsBackToTheSharedSocket(t *testing.T) {
	p := newTestPlugin(t) // no provisioner

	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}
	adj, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "c", Name: "trainer"})
	if err != nil {
		t.Fatalf("CreateContainer() error = %v", err)
	}

	want := filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath)
	if adj.Mounts[0].Source != want {
		t.Errorf("mount source = %q, want the pre-namespace shared socket %q", adj.Mounts[0].Source, want)
	}
}

// TestReleaseHappensOnRemoveContainerAfterTheDrain is the ordering requirement.
//
// A namespace with an active client cannot be deleted, and at StopContainer
// time the container's processes are still alive — that hook is where they are
// drained, not where they end. RemoveContainer runs after the runtime has torn
// the container down, so the drain has already happened and the client is gone.
// Releasing from the earlier hook would make every delete fail on its first
// attempt and lean entirely on mpsd's retry.
func TestReleaseHappensOnRemoveContainerAfterTheDrain(t *testing.T) {
	var journal []string
	var jmu sync.Mutex

	drainer := &fakeDrainer{journal: &journal, jmu: &jmu}
	prov := &fakeProvisioner{
		pipeDirectory: testNamespacePipeDir,
		namespace:     "kai_abc123_0011223344",
		releaseResp:   mpsdrain.ReleaseResponse{Deleted: true, Namespace: "kai_abc123_0011223344"},
		journal:       &journal,
		jmu:           &jmu,
	}

	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true, Drainer: drainer})
	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}
	ctr := &api.Container{Id: "container-id", Name: "trainer"}

	if _, _, err := p.CreateContainer(context.Background(), pod, ctr); err != nil {
		t.Fatalf("CreateContainer() error = %v", err)
	}
	if _, err := p.StopContainer(context.Background(), pod, ctr); err != nil {
		t.Fatalf("StopContainer() error = %v", err)
	}
	if err := p.RemoveContainer(context.Background(), pod, ctr); err != nil {
		t.Fatalf("RemoveContainer() error = %v", err)
	}

	jmu.Lock()
	got := slices.Clone(journal)
	jmu.Unlock()
	want := []string{"provision", "drain", "release"}
	if !slices.Equal(got, want) {
		t.Errorf("call order = %v, want %v", got, want)
	}

	releases := prov.releaseRequests()
	if len(releases) != 1 || releases[0].ContainerID != "container-id" {
		t.Errorf("releases = %+v, want one for container-id", releases)
	}
}

// TestReleaseFailureDoesNotBreakTeardown: the container is already gone, there
// is nothing left to protect, and mpsd reclaims the namespaces of containers
// that have disappeared during its own sweep. A release that fails must cost a
// log line and nothing else.
func TestReleaseFailureDoesNotBreakTeardown(t *testing.T) {
	prov := &fakeProvisioner{releaseErr: errors.New("mpsd is not answering")}
	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true})

	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}
	if err := p.RemoveContainer(context.Background(), pod, &api.Container{Id: "container-id", Name: "trainer"}); err != nil {
		t.Errorf("RemoveContainer() error = %v, want the failure stepped over", err)
	}
	if n := len(prov.releaseRequests()); n != 1 {
		t.Fatalf("release attempts = %d, want 1: the failure path was never reached", n)
	}
}

// TestRemoveContainerSkipsContainersThatNeverHadANamespace: RemoveContainer
// fires for every container on the node, and most of them have nothing to do
// with GPUs. A socket round trip each would be a lot of nothing on the runtime's
// hot path.
func TestRemoveContainerSkipsContainersThatNeverHadANamespace(t *testing.T) {
	prov := &fakeProvisioner{}
	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true})

	if err := p.RemoveContainer(context.Background(),
		&api.PodSandbox{Name: "test-pod"},
		&api.Container{Id: "container-id", Name: "some-sidecar"}); err != nil {
		t.Fatalf("RemoveContainer() error = %v", err)
	}
	if n := len(prov.releaseRequests()); n != 0 {
		t.Errorf("%d releases for a container with no GPU annotations, want 0", n)
	}
}

// TestRemoveContainerReleasesWhenTheAnnotationsCannotBeRead: the skip above is
// an optimisation, and an optimisation that cannot answer must fall through to
// the safe side. A release for a container with no namespace is a no-op at the
// other end; a release skipped for one that has a namespace holds it until
// mpsd's sweep notices the container is gone.
func TestRemoveContainerReleasesWhenTheAnnotationsCannotBeRead(t *testing.T) {
	prov := &fakeProvisioner{}
	p := newProvisioningPlugin(t, prov, Config{SupportSMSharing: true})

	if err := p.RemoveContainer(context.Background(),
		&api.PodSandbox{Name: "test-pod", Annotations: map[string]string{limitAnnotation: "not-a-quantity"}},
		&api.Container{Id: "container-id", Name: "trainer"}); err != nil {
		t.Fatalf("RemoveContainer() error = %v", err)
	}
	if n := len(prov.releaseRequests()); n != 1 {
		t.Errorf("%d releases for a container whose annotations could not be read, want 1", n)
	}
}

// TestNamespaceCallsFitInsideTheNRIBudget: every call an NRI hook makes has to
// fit inside the runtime's request deadline, because overrunning it is treated
// as fatal and drops fractiond off NRI — and a plugin that is not registered
// injects nothing into any container on the node.
func TestNamespaceCallsFitInsideTheNRIBudget(t *testing.T) {
	prov := &fakeProvisioner{pipeDirectory: testNamespacePipeDir}
	p := newProvisioningPlugin(t, prov, Config{
		SupportSMSharing: true,
		RequestTimeout:   func() time.Duration { return 2 * time.Second },
	})

	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}
	if _, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "c", Name: "trainer"}); err != nil {
		t.Fatalf("CreateContainer() error = %v", err)
	}

	prov.mu.Lock()
	defer prov.mu.Unlock()
	if len(prov.deadlines) != 1 {
		t.Fatalf("deadlines recorded = %d, want 1", len(prov.deadlines))
	}
	if prov.deadlines[0] <= 0 {
		t.Fatal("the provision call carried no deadline; a slow mpsd would hold the NRI hook open")
	}
	if prov.deadlines[0] > time.Second {
		t.Errorf("provision deadline = %v, want at most half the 2s NRI request timeout", prov.deadlines[0])
	}
}

// TestProvisioningThatOverrunsTheBudgetBlocksTheContainer: giving up on mpsd is
// the same answer as mpsd refusing. The container does not start, because there
// is no namespace to put it in.
func TestProvisioningThatOverrunsTheBudgetBlocksTheContainer(t *testing.T) {
	prov := &fakeProvisioner{block: true}
	p := newProvisioningPlugin(t, prov, Config{
		SupportSMSharing: true,
		DrainTimeout:     20 * time.Millisecond,
	})

	pod := &api.PodSandbox{
		Name:        "test-pod",
		Annotations: map[string]string{limitAnnotation: "4Gi", portionAnnotation: "0.5", modeAnnotation: "sm-sharing"},
	}

	done := make(chan error, 1)
	go func() {
		_, _, err := p.CreateContainer(context.Background(), pod, &api.Container{Id: "c", Name: "trainer"})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("CreateContainer() error = nil, want the container refused when mpsd did not answer in time")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("CreateContainer() never returned; the provision call is not bounded")
	}
}
