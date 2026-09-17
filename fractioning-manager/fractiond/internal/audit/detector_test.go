// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package audit

import (
	"path/filepath"
	"slices"
	"strings"
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
				Env:    append(injectedEqualEnv(), pinnedMemLimitEnv(1)),
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{"env:" + injection.EnvVisibleDevices},
		},
		{
			name: "assigned device fully injected is not a violator",
			pod:  withVisibleDevices(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "GPU-abc123"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				Env: append(injectedEqualEnv(),
					pinnedMemLimitEnv(1),
					injection.EnvVisibleDevices+"=GPU-abc123"),
				Mounts: []*api.Mount{mpsMount()},
			},
		},
		{
			// The MPS memory cap is per assigned GPU, so a two-GPU container
			// capped on only one of them is still under-enforced.
			name: "missing MPS memory cap (device assigned) is a violator",
			pod:  withVisibleDevices(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "GPU-abc123,GPU-def456"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    append(injectedEqualEnv(), injection.EnvVisibleDevices+"=GPU-abc123,GPU-def456"),
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{"env:" + injection.EnvMPSPinnedDeviceMemLimit},
		},
		{
			name: "missing MPS compute cap (portion annotated) is a violator",
			pod:  withComputePortion(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "0.5"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    injectedEqualEnv(),
				Mounts: []*api.Mount{mpsMount()},
			},
			wantMissing: []string{"env:" + injection.EnvMPSActiveThreadPercentage},
		},
		{
			name: "compute cap injected is not a violator",
			pod:  withComputePortion(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "0.5"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    append(injectedEqualEnv(), injection.EnvMPSActiveThreadPercentage+"=50"),
				Mounts: []*api.Mount{mpsMount()},
			},
		},
		{
			// Without the portion annotation there is no cap to enforce, so an
			// uncapped container is not retroactively stopped for it. The pod
			// carries a device assignment (and the container the matching
			// memory cap) so that the compute cap is the only thing absent —
			// otherwise this case would be indistinguishable from the
			// no-device-assignment one below and would prove nothing.
			name: "no compute portion does not expect a compute cap",
			pod:  withVisibleDevices(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "GPU-abc123"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				Env: append(injectedEqualEnv(),
					pinnedMemLimitEnv(1),
					injection.EnvVisibleDevices+"=GPU-abc123"),
				Mounts: []*api.Mount{mpsMount()},
			},
		},
		{
			// Everything at once, on the shared MPS server: the two caps must be
			// expected in sm-sharing exactly as in time-slicing. Memory
			// enforcement that quietly depended on the compute mode would leave
			// every sm-sharing tenant unaudited.
			name: "sm-sharing container with both caps is not a violator",
			pod: withComputePortion(
				withVisibleDevices(
					withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "sm-sharing"),
					"trainer", "GPU-abc123,GPU-def456"),
				"trainer", "0.5"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				Env: append(sharedSocketEqualEnv(),
					pinnedMemLimitEnv(2),
					injection.EnvMPSActiveThreadPercentage+"=50",
					injection.EnvVisibleDevices+"=GPU-abc123,GPU-def456"),
				Mounts: []*api.Mount{sharedSocketMount()},
			},
		},
		{
			// The same sm-sharing pod with neither cap: both must be reported,
			// alongside the mount, so remediation is driven by the whole
			// picture rather than by whichever check happens to run first.
			name: "sm-sharing container with no caps reports both",
			pod: withComputePortion(
				withVisibleDevices(
					withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "sm-sharing"),
					"trainer", "GPU-abc123,GPU-def456"),
				"trainer", "0.5"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
			wantMissing: []string{
				"env:" + injection.EnvMPSPipeDirectory,
				"env:" + injection.EnvGPUMemoryLimit,
				"env:" + injection.EnvGPUMemoryRequest,
				"env:" + injection.EnvVisibleDevices,
				"env:" + injection.EnvMPSPinnedDeviceMemLimit,
				"env:" + injection.EnvMPSActiveThreadPercentage,
				sharedMountMissing,
			},
		},
		{
			// Fail-closed: an unusable compute portion means the create hook
			// blocked the container, so a container running with one was not
			// injected by us and recreating it would hit the same parse error.
			// Skip rather than enter a stop/recreate loop.
			name: "unparseable compute portion is skipped, not stopped (fail-closed)",
			pod:  withComputePortion(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "50"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			},
		},
		{
			// The audit is presence-based by design: it asks whether the
			// variable is there, not what it says. Pinned here so nobody
			// mistakes it for a tamper check — a container that kept its own
			// value for a key the hook did not inject looks injected to the
			// audit, and the create hook is the only thing standing between a
			// tenant and a value of its choosing.
			name: "a present but wrong cap value is not a violation",
			pod:  withComputePortion(withVisibleDevices(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "GPU-abc123"), "trainer", "0.25"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				Env: append(injectedEqualEnv(),
					injection.EnvMPSPinnedDeviceMemLimit+"=0=999999M",
					injection.EnvMPSActiveThreadPercentage+"=100",
					injection.EnvVisibleDevices+"=all"),
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
			// The source half of the mount check, which the destination-only
			// cases above cannot see: every other mount case here differs in
			// BOTH source and destination, so all of them still pass if the
			// detector compares only the destination. This one does not.
			//
			// A time-slicing container bound to the *shared* server's socket at
			// the default in-container path is the shape a node left behind by
			// a half-rolled-back sm-sharing change: the path inside the
			// container is right, so the workload's CUDA runtime finds a
			// CUDA_MPS_PIPE_DIRECTORY and connects — to the wrong MPS server,
			// where its memory and thread caps were never registered. The audit
			// has to notice it is bound to the wrong host path.
			name: "time-slicing container bound to the shared socket is a violator",
			pod:  fractioningPod("p", "pod", "trainer", "4Gi", ""),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				Env:   injectedEqualEnv(),
				Mounts: []*api.Mount{{
					Source:      filepath.Join(configuration.DefaultMPSPipeDirectory, configuration.SharedMPSSocketPath),
					Destination: configuration.DefaultMPSPipeDirectory,
					Type:        "bind",
					Options:     []string{"bind", "rw"},
				}},
			},
			wantMissing: []string{defaultMountMissing},
		},
		{
			// The same blind spot from the other side: an sm-sharing container
			// whose mount lands on the right in-container path but comes from
			// some other host directory — what a pod hand-edited to mount its
			// own MPS pipe dir would look like. The destination matches the
			// expectation exactly; only the source betrays it.
			name: "sm-sharing container bound to a foreign host path is a violator",
			pod:  withComputeMode(fractioningPod("p", "pod", "trainer", "4Gi", ""), "trainer", "sm-sharing"),
			ctr: &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
				Env:   sharedSocketEqualEnv(),
				Mounts: []*api.Mount{{
					Source:      "/tmp/attacker-controlled-mps",
					Destination: configuration.ContainerMPSPipeDirectory,
					Type:        "bind",
					Options:     []string{"bind", "rw"},
				}},
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

// withComputePortion records a compute-portion annotation for the named
// container on the pod, as the scheduler would at bind time, so the detector
// expects CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to be injected.
func withComputePortion(pod *api.PodSandbox, containerName, value string) *api.PodSandbox {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[annotations.ComputePortionAnnotationKey(configuration.DefaultAnnotationPrefix, containerName)] = value
	return pod
}

// pinnedMemLimitEnv returns the MPS per-device memory cap buildAdjustment would
// inject for a container holding deviceCount GPUs at the 4Gi / 4096 MiB the
// fixtures use.
func pinnedMemLimitEnv(deviceCount int) string {
	return injection.EnvMPSPinnedDeviceMemLimit + "=" + injection.PinnedDeviceMemLimit("4096", deviceCount)
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

// TestDetectorFailOpenAuditsUnusableComputePortion is the compute-portion twin
// of TestDetectorFailOpenAuditsUnusableComputeMode, and it matters for the same
// reason. Under fail-open the create hook does not block a container whose
// portion annotation is unusable — it creates it without a compute cap and with
// every other limit intact. The audit has to reach the same conclusion, or a
// single bad portion value would be enough to hide a container that is holding
// a GPU share with no memory limit at all.
func TestDetectorFailOpenAuditsUnusableComputePortion(t *testing.T) {
	// One entry per way the portion can be unusable, because they take
	// different paths through the parser (syntax, range, blank).
	for _, portion := range []string{"abc", "50", "0", "-1", "", "NaN"} {
		t.Run(portion, func(t *testing.T) {
			pod := withComputePortion(fractioningPod("p", "pod", "trainer", "4Gi", "2Gi"), "trainer", portion)
			ctr := &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			}

			got := newDetectorFailOpen().violators([]*api.PodSandbox{pod}, []*api.Container{ctr})
			if len(got) != 1 {
				t.Fatalf("expected the container to be audited without a compute cap, got %d violators: %+v", len(got), got)
			}
			// CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is deliberately absent from the
			// expectations: fail-open creation produced no compute cap, so
			// demanding one here would stop a container for something the hook
			// never injected.
			assertSameSet(t, got[0].missing, []string{
				"env:" + injection.EnvMPSPipeDirectory,
				"env:" + injection.EnvGPUMemoryLimit,
				"env:" + injection.EnvGPUMemoryRequest,
				defaultMountMissing,
			})
		})
	}
}

// TestDetectorFailClosedSkipsUnusableComputePortion is the other half of that
// policy. Fail-closed creation refuses the container outright, so anything
// running with such an annotation was not created by a healthy hook; stopping
// it would only have kubelet recreate it into the same refusal, forever.
func TestDetectorFailClosedSkipsUnusableComputePortion(t *testing.T) {
	for _, portion := range []string{"abc", "50", "0", "-1", "", "NaN"} {
		t.Run(portion, func(t *testing.T) {
			pod := withComputePortion(fractioningPod("p", "pod", "trainer", "4Gi", "2Gi"), "trainer", portion)
			ctr := &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State: api.ContainerState_CONTAINER_RUNNING,
			}

			if got := newDetector().violators([]*api.PodSandbox{pod}, []*api.Container{ctr}); len(got) != 0 {
				t.Fatalf("expected the container to be skipped, got %+v", got)
			}
		})
	}
}

// TestDetectorPinnedCapScalesWithDeviceCount walks the device-assignment axis on
// its own. The expectation must follow the number of GPUs the scheduler
// assigned: zero devices means there is no per-device list to inject and
// demanding one would stop every single-tenant fractional container on the node,
// while any assignment at all means the cap is owed.
func TestDetectorPinnedCapScalesWithDeviceCount(t *testing.T) {
	tests := []struct {
		name           string
		visibleDevices string
		wantExpected   bool
	}{
		{name: "unassigned", visibleDevices: "", wantExpected: false},
		{name: "blank assignment", visibleDevices: "   ", wantExpected: false},
		{name: "one device", visibleDevices: "GPU-abc123", wantExpected: true},
		{name: "two devices", visibleDevices: "GPU-abc123,GPU-def456", wantExpected: true},
		{name: "eight devices", visibleDevices: "GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7", wantExpected: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := fractioningPod("p", "pod", "trainer", "4Gi", "")
			if tt.visibleDevices != "" {
				pod = withVisibleDevices(pod, "trainer", tt.visibleDevices)
			}
			// Everything injected except the pinned memory cap, so the pinned
			// cap is the only thing the result can be about.
			env := injectedEqualEnv()
			if strings.TrimSpace(tt.visibleDevices) != "" {
				env = append(env, injection.EnvVisibleDevices+"="+tt.visibleDevices)
			}
			ctr := &api.Container{
				Id: "c", Name: "trainer", PodSandboxId: "p",
				State:  api.ContainerState_CONTAINER_RUNNING,
				Env:    env,
				Mounts: []*api.Mount{mpsMount()},
			}

			got := newDetector().violators([]*api.PodSandbox{pod}, []*api.Container{ctr})
			if !tt.wantExpected {
				if len(got) != 0 {
					t.Fatalf("expected no violator for an unassigned container, got %+v", got)
				}
				return
			}
			if len(got) != 1 {
				t.Fatalf("expected the missing per-device memory cap to be reported, got %+v", got)
			}
			assertSameSet(t, got[0].missing, []string{"env:" + injection.EnvMPSPinnedDeviceMemLimit})
		})
	}
}
