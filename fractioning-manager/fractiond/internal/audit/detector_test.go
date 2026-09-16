// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package audit

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/configuration"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/injection"
)

func TestDetectorViolations(t *testing.T) {
	tests := []struct {
		name        string
		pod         *api.PodSandbox
		ctr         *api.Container
		wantMissing []string // nil ⇒ expect no violator
	}{
		{
			name: "fully injected container is not a violator",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", "2Gi"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    injectedEnv(true, true),
				Mounts: []*api.Mount{mpsMount()},
			},
		},
		{
			name: "missing MPS env is a violator",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    []string{injection.EnvGPUMemoryLimit + "=4096", injection.EnvGPUMemoryRequest + "=4096"},
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{"env:" + injection.EnvMPSPipeDirectory},
		},
		{
			name: "missing MPS mount is a violator",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				Env:   injectedEqualEnv(),
			},
			wantMissing: []string{defaultMountMissing},
		},
		{
			name: "missing limit env (limit annotated) is a violator",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				// request env present (defaulted from the limit), limit env missing.
				Env:    []string{injection.EnvMPSPipeDirectory + "=" + configuration.DefaultMPSPipeDirectory, injection.EnvGPUMemoryRequest + "=4096"},
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{"env:" + injection.EnvGPUMemoryLimit},
		},
		{
			name: "nothing injected reports every missing piece",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", "2Gi"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
			wantMissing: []string{
				"env:" + injection.EnvMPSPipeDirectory,
				"env:" + injection.EnvGPUMemoryLimit,
				"env:" + injection.EnvGPUMemoryRequest,
				defaultMountMissing,
			},
		},
		{
			name: "request-only fractioning container expects both request and limit env",
			pod:  fractioningPod("p", "pod", "trainer", "", "2Gi"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				// limit defaults from the request, so both memory envs are expected.
				Env:    []string{injection.EnvMPSPipeDirectory + "=" + configuration.DefaultMPSPipeDirectory},
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{"env:" + injection.EnvGPUMemoryLimit, "env:" + injection.EnvGPUMemoryRequest},
		},
		{
			name: "missing NVIDIA_VISIBLE_DEVICES env (device assigned) is a violator",
			pod:  withVisibleDevices(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "GPU-abc123"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    injectedEqualEnv(),
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{"env:" + injection.EnvVisibleDevices},
		},
		{
			name: "assigned device fully injected is not a violator",
			pod:  withVisibleDevices(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "GPU-abc123"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    append(injectedEqualEnv(), injection.EnvVisibleDevices+"=GPU-abc123"),
				Mounts: []*api.Mount{mpsMount()},
			},
		},
		{
			name: "no device assignment does not expect NVIDIA_VISIBLE_DEVICES",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    injectedEqualEnv(),
				Mounts: []*api.Mount{mpsMount()},
			},
		},
		{
			name: "created state is enforceable",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_CREATED,
			},
			wantMissing: []string{
				"env:" + injection.EnvMPSPipeDirectory,
				"env:" + injection.EnvGPUMemoryLimit,
				"env:" + injection.EnvGPUMemoryRequest,
				defaultMountMissing,
			},
		},
		{
			name: "non-fractioning container is skipped",
			pod:  &api.PodSandbox{Id: "p", Name: "pod", Annotations: map[string]string{"foo": "bar"}},
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
		},
		{
			name: "sibling without its own annotation is skipped",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "sidecar", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
		},
		{
			name: "stopped container is skipped even if uninjected",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_STOPPED,
			},
		},
		{
			name: "container with empty id is skipped",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
		},
		{
			name: "unparseable annotation is skipped, not stopped",
			pod:  fractioningPod("p", "pod", "trainer", "not-a-quantity", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
		},
		{
			name: "sm-sharing container mounted on the shared socket is not a violator",
			pod:  withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "sm-sharing"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    sharedSocketEqualEnv(),
				Mounts: []*api.Mount{sharedSocketMount()},
			},
		},
		{
			name: "sm-sharing container still on the default socket is a violator",
			pod:  withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "sm-sharing"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				// Injected as if time-slicing (the pre-upgrade/default shape):
				// CUDA_MPS_PIPE_DIRECTORY is present (presence-based env check
				// does not compare values), but the mount is still bound to the
				// default socket (wrong source and wrong destination), not the
				// shared one the sm-sharing annotation expects.
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    injectedEqualEnv(),
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{sharedMountMissing},
		},
		{
			name: "unparseable compute mode annotation is skipped, not stopped (fail-closed)",
			pod:  withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "mig"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newDetector().violators(
				[]*api.PodSandbox{tt.pod},
				[]*api.Container{tt.ctr},
			)

			if tt.wantMissing == nil {
				if len(got) != 0 {
					t.Fatalf("expected no violator, got %+v", got)
				}
				return
			}

			if len(got) != 1 {
				t.Fatalf("expected exactly one violator, got %d: %+v", len(got), got)
			}
			v := got[0]
			if v.containerID != tt.ctr.GetId() || v.container != tt.ctr.GetName() {
				t.Errorf("violator identity = %q/%q, want %q/%q", v.containerID, v.container, tt.ctr.GetId(), tt.ctr.GetName())
			}
			assertSameSet(t, v.missing, tt.wantMissing)
		})
	}
}

// TestDetectorSMSharingDisabledSkipsAnnotatedContainer verifies that when the
// sm-sharing chicken bit is off, a container annotated sm-sharing is skipped
// (unparseable, same as any other invalid compute-mode value) rather than
// flagged as a violator — mirroring how the plugin itself would reject/
// downgrade it, so a disabled feature can never produce a false violation.
func TestDetectorSMSharingDisabledSkipsAnnotatedContainer(t *testing.T) {
	pod := withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "sm-sharing")
	ctr := &api.Container{
		Id: "c", Name: "trainer", PodSandboxId: "p",
		State: api.ContainerState_CONTAINER_RUNNING,
	}

	got := newDetectorSMSharingDisabled().violators([]*api.PodSandbox{pod}, []*api.Container{ctr})
	if len(got) != 0 {
		t.Fatalf("expected no violator when sm-sharing is disabled cluster-wide, got %+v", got)
	}
}

// TestDetectorFailOpenAuditsUnusableComputeMode covers the fail-open half of
// the compute-mode policy. Creation under fail-open does not block a container
// whose mode annotation cannot be honored; it falls back to time-slicing and
// still injects the memory limit. The audit must reach the same conclusion,
// otherwise a mode typo would be enough to hide a container that is holding a
// GPU share with no memory limit at all — exactly what this audit is for.
func TestDetectorFailOpenAuditsUnusableComputeMode(t *testing.T) {
	tests := []struct {
		name     string
		detector detector
		mode     string
	}{
		{
			name:     "invalid value",
			detector: newDetectorFailOpen(),
			mode:     "mig",
		},
		{
			name:     "sm-sharing requested while disabled cluster-wide",
			detector: newDetectorFailOpenSMSharingDisabled(),
			mode:     "sm-sharing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", "2Gi"), "trainer", tt.mode)
			ctr := &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			}

			got := tt.detector.violators([]*api.PodSandbox{pod}, []*api.Container{ctr})
			if len(got) != 1 {
				t.Fatalf("expected the container to be audited against the time-slicing default, got %d violators: %+v", len(got), got)
			}
			// The expectations are the time-slicing ones (default mount), proving
			// the fallback rather than merely that something was reported.
			assertSameSet(t, got[0].missing, []string{
				"env:" + injection.EnvMPSPipeDirectory,
				"env:" + injection.EnvGPUMemoryLimit,
				"env:" + injection.EnvGPUMemoryRequest,
				defaultMountMissing,
			})
		})
	}
}

func TestDetectorViolationsSkipsContainerWithoutPod(t *testing.T) {
	// A container whose PodSandboxId matches no pod in the snapshot must be
	// skipped rather than panic or flagged.
	got := newDetector().violators(
		nil,
		[]*api.Container{{
			Id: "c", Name: "trainer", PodSandboxId: "missing",
			State: api.ContainerState_CONTAINER_RUNNING,
		}},
	)
	if len(got) != 0 {
		t.Fatalf("expected no violator for pod-less container, got %+v", got)
	}
}

func TestDetectorViolationsAcrossMultipleContainers(t *testing.T) {
	pods := []*api.PodSandbox{
		fractioningPod("p1", "pod1", "trainer", "4Gi", ""),
		fractioningPod("p2", "pod2", "trainer", "4Gi", ""),
		{Id: "p3", Name: "pod3", Annotations: map[string]string{"foo": "bar"}}, // non-fractioning
	}
	containers := []*api.Container{
		{ // p1: injected → ok
			Id: "c1", Name: "trainer", PodSandboxId: "p1",
			State: api.ContainerState_CONTAINER_RUNNING, Env: injectedEqualEnv(), Mounts: []*api.Mount{mpsMount()},
		},
		{ // p2: uninjected → violator
			Id: "c2", Name: "trainer", PodSandboxId: "p2",
			State: api.ContainerState_CONTAINER_RUNNING,
		},
		{ // p3: non-fractioning → skipped
			Id: "c3", Name: "app", PodSandboxId: "p3",
			State: api.ContainerState_CONTAINER_RUNNING,
		},
	}

	got := newDetector().violators(pods, containers)
	if len(got) != 1 {
		t.Fatalf("expected exactly one violator, got %d: %+v", len(got), got)
	}
	if got[0].containerID != "c2" {
		t.Fatalf("expected violator for c2, got %q", got[0].containerID)
	}
}

// fractioningPod builds a pod sandbox carrying the GPU memory annotations for the
// named container. Pass empty strings to omit an annotation.
func fractioningPod(id, name, containerName, limit, request string) *api.PodSandbox {
	ann := map[string]string{}
	if limit != "" {
		ann[annotations.LimitAnnotationKey(configuration.DefaultAnnotationPrefix, containerName)] = limit
	}
	if request != "" {
		ann[annotations.RequestAnnotationKey(configuration.DefaultAnnotationPrefix, containerName)] = request
	}
	return &api.PodSandbox{Id: id, Name: name, Namespace: "default", Uid: id + "-uid", Annotations: ann}
}

// withVisibleDevices records a GPU device assignment for the named container on
// the pod, as the scheduler would, so the detector expects NVIDIA_VISIBLE_DEVICES
// to be injected.
func withVisibleDevices(pod *api.PodSandbox, containerName, value string) *api.PodSandbox {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[annotations.DevicesAnnotationKey(configuration.DefaultAnnotationPrefix, containerName)] = value
	return pod
}

// withComputeMode records a compute-mode annotation for the named container on
// the pod, so the detector derives its expected mount from that mode instead
// of defaulting to time-slicing.
func withComputeMode(pod *api.PodSandbox, containerName, value string) *api.PodSandbox {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[annotations.ComputeModeAnnotationKey(configuration.DefaultAnnotationPrefix, containerName)] = value
	return pod
}

// injectedEnv returns the env keys buildAdjustment would add for the given
// request/limit presence, so tests can construct a fully-injected container.
func injectedEnv(limit, request bool) []string {
	env := []string{injection.EnvMPSPipeDirectory + "=" + configuration.DefaultMPSPipeDirectory}
	if limit {
		env = append(env, injection.EnvGPUMemoryLimit+"=4096")
	}
	if request {
		env = append(env, injection.EnvGPUMemoryRequest+"=2048")
	}
	return env
}

// injectedEqualEnv returns the fully-injected env for a container whose request
// and limit resolve to the same value — the real post-injection state of a
// request-only or limit-only pod after ApplyDefaults (both default to the 4Gi /
// 4096 MiB value injectedEnv uses for the limit). Use this instead of
// injectedEnv(true, true) for such pods so fixtures match reality.
func injectedEqualEnv() []string {
	return []string{
		injection.EnvMPSPipeDirectory + "=" + configuration.DefaultMPSPipeDirectory,
		injection.EnvGPUMemoryLimit + "=4096",
		injection.EnvGPUMemoryRequest + "=4096",
	}
}

func mpsMount() *api.Mount {
	return &api.Mount{
		Source:      configuration.DefaultMPSPipeDirectory,
		Destination: configuration.DefaultMPSPipeDirectory,
		Type:        "bind",
		Options:     []string{"bind", "rw"},
	}
}

// defaultMountMissing and sharedMountMissing are the "missing" entries
// missingInjection reports for the time-slicing and sm-sharing mounts,
// respectively — kept as the single source of truth for the "mount:<source>:
// <destination>" format instead of duplicating it across test cases.
var (
	defaultMountMissing = "mount:" + configuration.DefaultMPSPipeDirectory + ":" + configuration.DefaultMPSPipeDirectory
	sharedMountMissing  = "mount:" + filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath) + ":" + configuration.ContainerMPSPipeDirectory
)

// sharedSocketMount is the mount buildAdjustment produces for an sm-sharing
// container: the shared server's default-namespace socket on the host, bound
// to the fixed in-container MPS pipe path.
func sharedSocketMount() *api.Mount {
	return &api.Mount{
		Source:      filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath),
		Destination: configuration.ContainerMPSPipeDirectory,
		Type:        "bind",
		Options:     []string{"bind", "rw"},
	}
}

// sharedSocketEqualEnv is the fully-injected env for an sm-sharing container
// whose request and limit resolve to the same value (mirrors injectedEqualEnv
// but with CUDA_MPS_PIPE_DIRECTORY pointed at the in-container sm-sharing path).
func sharedSocketEqualEnv() []string {
	return []string{
		injection.EnvMPSPipeDirectory + "=" + configuration.ContainerMPSPipeDirectory,
		injection.EnvGPUMemoryLimit + "=4096",
		injection.EnvGPUMemoryRequest + "=4096",
	}
}

// newDetector is the fail-closed detector: an unusable compute-mode annotation
// means the container was never created by a healthy hook, so it is skipped.
func newDetector() detector {
	return detector{
		annotationPrefix: configuration.DefaultAnnotationPrefix,
		mpsPipeDirectory: configuration.DefaultMPSPipeDirectory,
		smSharingEnabled: true,
		failOpen:         false,
	}
}

func newDetectorSMSharingDisabled() detector {
	d := newDetector()
	d.smSharingEnabled = false
	return d
}

func newDetectorFailOpen() detector {
	d := newDetector()
	d.failOpen = true
	return d
}

func newDetectorFailOpenSMSharingDisabled() detector {
	d := newDetectorFailOpen()
	d.smSharingEnabled = false
	return d
}

func assertSameSet(t *testing.T, got, want []string) {
	t.Helper()
	g := slices.Clone(got)
	w := slices.Clone(want)
	slices.Sort(g)
	slices.Sort(w)
	if !slices.Equal(g, w) {
		t.Errorf("missing set = %v, want %v", g, w)
	}
}
