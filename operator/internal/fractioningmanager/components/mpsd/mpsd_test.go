// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsd

import (
	"slices"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1alpha1 "github.com/kai-scheduler/kai-gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/common/daemonmgr"
)

const testDaemonServiceAccountName = "gpu-fractioning-daemon"

// Fixed args passed to NewMpsdDaemon in tests that don't exercise them
// directly (see TestDaemon_BuildDaemonSet_AuditLogDisabled and
// TestDaemon_BuildDaemonSet_SupportSMSharingDisabled for the dedicated
// disabled-value coverage of each).
const (
	testMpsdAuditLogTrue     = true
	testSupportSMSharingTrue = true
)

func TestDaemon_BuildDaemonSet_Basics(t *testing.T) {
	d := NewMpsdDaemon(nil, testMpsdAuditLogTrue, testSupportSMSharingTrue)

	if got := d.Name(); got != "mpsd" {
		t.Errorf("Name() = %q, want %q", got, "mpsd")
	}

	ds := d.BuildDaemonSet(defaultOpts())

	if ds.Name != "gpu-fractioning-mpsd" {
		t.Errorf("DaemonSet name = %q, want %q", ds.Name, "gpu-fractioning-mpsd")
	}
	if ds.Namespace != "gpu-fractioning-system" {
		t.Errorf("DaemonSet namespace = %q, want %q", ds.Namespace, "gpu-fractioning-system")
	}

	spec := ds.Spec.Template.Spec

	if spec.ServiceAccountName != testDaemonServiceAccountName {
		t.Errorf("serviceAccountName = %q, expected %s", spec.ServiceAccountName, testDaemonServiceAccountName)
	}

	// RuntimeClassName
	if spec.RuntimeClassName == nil || *spec.RuntimeClassName != daemonmgr.DefaultRuntimeClassName {
		t.Errorf("RuntimeClassName = %v, want nvidia", spec.RuntimeClassName)
	}

	// HostPID: without it the control daemon reads every client's PID as 0 and
	// memacct can't register them, so GPU memory limits go unenforced.
	if !spec.HostPID {
		t.Error("expected HostPID=true for mpsd")
	}

	// Node selector
	if spec.NodeSelector == nil || spec.NodeSelector["nvidia.com/gpu.present"] != "true" {
		t.Errorf("NodeSelector = %v, want nvidia.com/gpu.present=true", spec.NodeSelector)
	}

	// Container
	if len(spec.Containers) != 1 {
		t.Fatalf("expected 1 container, got %d", len(spec.Containers))
	}

	ctr := spec.Containers[0]
	if ctr.Image != "fake.io/org/mpsd:v0.1.0" {
		t.Errorf("container image = %q, want %q", ctr.Image, "fake.io/org/mpsd:v0.1.0")
	}
	if ctr.SecurityContext == nil || ctr.SecurityContext.Privileged == nil || !*ctr.SecurityContext.Privileged {
		t.Error("expected privileged=true")
	}

	// Probes
	if ctr.ReadinessProbe == nil {
		t.Fatal("expected readiness probe")
	}
	if ctr.ReadinessProbe.Exec == nil || len(ctr.ReadinessProbe.Exec.Command) == 0 {
		t.Error("expected exec readiness probe checking MPS control socket")
	}
	// Liveness reuses the socket check but with generous thresholds so the
	// supervisor can self-heal before a pod restart is escalated.
	if ctr.LivenessProbe == nil || ctr.LivenessProbe.Exec == nil {
		t.Fatal("expected an exec liveness probe on mpsd")
	}
	if ctr.LivenessProbe.FailureThreshold != 6 {
		t.Errorf("liveness FailureThreshold = %d, want 6", ctr.LivenessProbe.FailureThreshold)
	}

	// Resources: CPU + memory + ephemeral-storage requests, a memory limit, and
	// no CPU limit.
	if ctr.Resources.Requests.Cpu().IsZero() || ctr.Resources.Requests.Memory().IsZero() {
		t.Error("expected CPU and memory requests on mpsd")
	}
	if ctr.Resources.Requests.StorageEphemeral().IsZero() {
		t.Error("expected an ephemeral-storage request on mpsd")
	}
	if got := ctr.Resources.Limits.Memory().String(); got != mpsdMemLimit {
		t.Errorf("memory limit = %q, want %q", got, mpsdMemLimit)
	}

	// Volumes
	volumePaths := make(map[string]string)
	for _, v := range spec.Volumes {
		if v.HostPath != nil {
			volumePaths[v.Name] = v.HostPath.Path
		}
	}
	if volumePaths["mps-pipe"] != "/run/nvidia-mps" {
		t.Errorf("mps-pipe volume path = %q, want %q", volumePaths["mps-pipe"], "/run/nvidia-mps")
	}
	if volumePaths["mps-log"] != "/var/log/nvidia-mps" {
		t.Errorf("mps-log volume path = %q, want %q", volumePaths["mps-log"], "/var/log/nvidia-mps")
	}

	// Volume mounts
	mountPaths := make(map[string]string)
	for _, m := range ctr.VolumeMounts {
		mountPaths[m.Name] = m.MountPath
	}
	if mountPaths["mps-pipe"] != "/run/nvidia-mps" {
		t.Errorf("mps-pipe mount = %q, want %q", mountPaths["mps-pipe"], "/run/nvidia-mps")
	}
	if mountPaths["mps-log"] != "/var/log/nvidia-mps" {
		t.Errorf("mps-log mount = %q, want %q", mountPaths["mps-log"], "/var/log/nvidia-mps")
	}

	// Labels
	labels := ds.Labels
	if labels[daemonmgr.LabelManagedBy] != daemonmgr.ManagedByValue {
		t.Errorf("missing managed-by label, got %v", labels)
	}
	if labels[daemonmgr.LabelComponent] != "mpsd" {
		t.Errorf("missing component label, got %v", labels)
	}

	// No args when spec is nil
	// The drain socket is always passed, even when spec is nil: an explicit ""
	// is how the endpoint is switched off, which omitting the flag cannot say.
	wantArgs := []string{"--drain-socket", daemonmgr.DefaultMPSDrainSocketPath}
	if !slices.Equal(ctr.Args, wantArgs) {
		t.Errorf("args when spec is nil = %v, want %v", ctr.Args, wantArgs)
	}

	// Helm-driven audit-log toggle is injected as an env var.
	if got := envValue(ctr.Env, "MPS_MEMACCT_AUDIT_LOG"); got != "true" {
		t.Errorf("MPS_MEMACCT_AUDIT_LOG env = %q, want %q", got, "true")
	}
	if got := envFieldRef(ctr.Env, "NODE_NAME"); got != "spec.nodeName" {
		t.Errorf("NODE_NAME fieldRef = %q, want spec.nodeName", got)
	}
	if got := envValue(ctr.Env, "NVIDIA_VISIBLE_DEVICES"); got != "all" {
		t.Errorf("NVIDIA_VISIBLE_DEVICES env = %q, want %q", got, "all")
	}
	if got := envValue(ctr.Env, "NVIDIA_DRIVER_CAPABILITIES"); got != "compute,utility" {
		t.Errorf("NVIDIA_DRIVER_CAPABILITIES env = %q, want %q", got, "compute,utility")
	}
	if got := envValue(ctr.Env, "MPS_SUPPORT_SM_SHARING"); got != "true" {
		t.Errorf("MPS_SUPPORT_SM_SHARING env = %q, want %q", got, "true")
	}
}

func TestDaemon_BuildDaemonSet_AuditLogDisabled(t *testing.T) {
	ds := NewMpsdDaemon(nil, false, testSupportSMSharingTrue).BuildDaemonSet(defaultOpts())
	ctr := ds.Spec.Template.Spec.Containers[0]

	if got := envValue(ctr.Env, "MPS_MEMACCT_AUDIT_LOG"); got != "false" {
		t.Errorf("MPS_MEMACCT_AUDIT_LOG env = %q, want %q", got, "false")
	}
}

func TestDaemon_BuildDaemonSet_SupportSMSharingDisabled(t *testing.T) {
	ds := NewMpsdDaemon(nil, testMpsdAuditLogTrue, false).BuildDaemonSet(defaultOpts())
	ctr := ds.Spec.Template.Spec.Containers[0]

	if got := envValue(ctr.Env, "MPS_SUPPORT_SM_SHARING"); got != "false" {
		t.Errorf("MPS_SUPPORT_SM_SHARING env = %q, want %q", got, "false")
	}
}

func TestDaemon_BuildDaemonSet_FIPSOnly(t *testing.T) {
	tests := []struct {
		name     string
		fipsOnly bool
		// wantGODEBUG is the value expected on the container; empty means the
		// var must be absent.
		wantGODEBUG string
	}{
		{name: "disabled", fipsOnly: false, wantGODEBUG: ""},
		{name: "enabled", fipsOnly: true, wantGODEBUG: "fips140=only,tlsmlkem=0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := defaultOpts()
			opts.FIPSOnly = tt.fipsOnly
			ctr := NewMpsdDaemon(nil, testMpsdAuditLogTrue, testSupportSMSharingTrue).
				BuildDaemonSet(opts).Spec.Template.Spec.Containers[0]

			if got := envValue(ctr.Env, "GODEBUG"); got != tt.wantGODEBUG {
				t.Errorf("GODEBUG env = %q, want %q", got, tt.wantGODEBUG)
			}

			// mpsd's own env must survive injection: the NVIDIA vars get the
			// driver libraries mounted in, and NODE_NAME is how it labels its
			// node at startup.
			if got := envValue(ctr.Env, "NVIDIA_VISIBLE_DEVICES"); got != "all" {
				t.Errorf("NVIDIA_VISIBLE_DEVICES env = %q, want %q; FIPS injection clobbered existing env", got, "all")
			}
			if got := envValue(ctr.Env, "MPS_MEMACCT_AUDIT_LOG"); got != "true" {
				t.Errorf("MPS_MEMACCT_AUDIT_LOG env = %q, want %q; FIPS injection clobbered existing env", got, "true")
			}
			if got := envFieldRef(ctr.Env, "NODE_NAME"); got != "spec.nodeName" {
				t.Errorf("NODE_NAME fieldRef = %q, want spec.nodeName; FIPS injection clobbered existing env", got)
			}
		})
	}
}

func TestDaemon_BuildDaemonSet_RuntimeClass(t *testing.T) {
	custom := "custom-nvidia"

	tests := []struct {
		name             string
		runtimeClassName *string
		want             *string
	}{
		{
			name:             "default runtime class is propagated",
			runtimeClassName: ptr.To(daemonmgr.DefaultRuntimeClassName),
			want:             ptr.To(daemonmgr.DefaultRuntimeClassName),
		},
		{
			name:             "custom runtime class is propagated",
			runtimeClassName: &custom,
			want:             &custom,
		},
		{
			name:             "nil runtime class uses node default",
			runtimeClassName: nil,
			want:             nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := defaultOpts()
			opts.RuntimeClassName = tt.runtimeClassName
			spec := NewMpsdDaemon(nil, testMpsdAuditLogTrue, testSupportSMSharingTrue).BuildDaemonSet(opts).Spec.Template.Spec
			if tt.want == nil {
				if spec.RuntimeClassName != nil {
					t.Fatalf("RuntimeClassName = %q, want nil", *spec.RuntimeClassName)
				}
				return
			}
			if spec.RuntimeClassName == nil || *spec.RuntimeClassName != *tt.want {
				t.Fatalf("RuntimeClassName = %v, want %q", spec.RuntimeClassName, *tt.want)
			}
		})
	}
}

func envValue(env []corev1.EnvVar, name string) string {
	for _, e := range env {
		if e.Name == name {
			return e.Value
		}
	}
	return ""
}

func envFieldRef(env []corev1.EnvVar, name string) string {
	for _, e := range env {
		if e.Name == name && e.ValueFrom != nil && e.ValueFrom.FieldRef != nil {
			return e.ValueFrom.FieldRef.FieldPath
		}
	}
	return ""
}

func TestDaemon_BuildDaemonSet_Args(t *testing.T) {
	d := NewMpsdDaemon(&v1alpha1.MpsDaemonSpec{
		LogLevel:          "debug",
		Backoff:           &metav1.Duration{Duration: 10 * time.Second},
		MaxRetries:        ptr.To(int32(5)),
		StableThreshold:   &metav1.Duration{Duration: 10 * time.Minute},
		GracefulStopDelay: &metav1.Duration{Duration: 30 * time.Second},
	}, testMpsdAuditLogTrue, testSupportSMSharingTrue)

	ds := d.BuildDaemonSet(defaultOpts())
	args := ds.Spec.Template.Spec.Containers[0].Args

	expected := []string{
		"--log-level", "debug",
		"--backoff", "10s",
		"--max-retries", "5",
		"--stable-threshold", "10m0s",
		"--graceful-stop-delay", "30s",
		"--drain-socket", daemonmgr.DefaultMPSDrainSocketPath,
	}

	if len(args) != len(expected) {
		t.Fatalf("args length = %d, want %d\ngot:  %v\nwant: %v", len(args), len(expected), args, expected)
	}
	for i := range expected {
		if args[i] != expected[i] {
			t.Errorf("args[%d] = %q, want %q", i, args[i], expected[i])
		}
	}
}

func TestDaemon_BuildDaemonSet_TerminationGrace(t *testing.T) {
	// Default (no configured graceful-stop-delay): mpsd's 60s default + buffer,
	// and always ≥ the graceful-stop-delay so kubelet can't SIGKILL mpsd before
	// its MPS graceful-quit window completes (the driver-upgrade guarantee).
	grace := NewMpsdDaemon(nil, false, testSupportSMSharingTrue).BuildDaemonSet(defaultOpts()).Spec.Template.Spec.TerminationGracePeriodSeconds
	if grace == nil {
		t.Fatal("terminationGracePeriodSeconds is nil; mpsd inherits the 30s default and can be killed mid-quit")
	}
	if want := int64((defaultGracefulStopDelay + terminationGraceBuffer).Seconds()); *grace != want {
		t.Errorf("default grace = %d, want %d", *grace, want)
	}
	if *grace < int64(defaultGracefulStopDelay.Seconds()) {
		t.Errorf("grace %d < graceful-stop-delay %d", *grace, int64(defaultGracefulStopDelay.Seconds()))
	}

	// A configured graceful-stop-delay is tracked (plus buffer), still ≥ it.
	d := NewMpsdDaemon(&v1alpha1.MpsDaemonSpec{
		GracefulStopDelay: &metav1.Duration{Duration: 90 * time.Second},
	}, false, testSupportSMSharingTrue)
	grace = d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec.TerminationGracePeriodSeconds
	if want := int64((90*time.Second + terminationGraceBuffer).Seconds()); grace == nil || *grace != want {
		t.Fatalf("configured grace = %v, want %d", grace, want)
	}
	if *grace <= 90 {
		t.Errorf("grace %d must exceed the configured 90s graceful-stop-delay", *grace)
	}
}

func defaultOpts() daemonmgr.BuildOptions {
	return daemonmgr.BuildOptions{
		Namespace:          "gpu-fractioning-system",
		NodeSelector:       map[string]string{"nvidia.com/gpu.present": "true"},
		ServiceAccountName: testDaemonServiceAccountName,
		RuntimeClassName:   ptr.To(daemonmgr.DefaultRuntimeClassName),
		DefaultImages: map[string]daemonmgr.ImageSpec{
			"mpsd": {
				Repository: "fake.io/org/mpsd",
				Tag:        "v0.1.0",
			},
		},
	}
}

func mpsdContainer(t *testing.T, d daemonmgr.ManagedDaemon) corev1.Container {
	t.Helper()
	containers := d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec.Containers
	if len(containers) != 1 {
		t.Fatalf("expected 1 container, got %d", len(containers))
	}
	return containers[0]
}

func hostPathVolume(volumes []corev1.Volume, name string) *corev1.HostPathVolumeSource {
	for _, v := range volumes {
		if v.Name == name {
			return v.HostPath
		}
	}
	return nil
}

func mountPathFor(mounts []corev1.VolumeMount, name string) (string, bool) {
	for _, m := range mounts {
		if m.Name == name {
			return m.MountPath, true
		}
	}
	return "", false
}

// The default memory limit is sized by GPU count, not by mpsd's own footprint:
// the MPS control daemon holds one CUDA server context per GPU at roughly 50 MiB
// each. The previous 256Mi ran out around the sixth context on an 8-GPU node and
// failed as CUDA_ERROR_OUT_OF_MEMORY rather than an OOMKill, so it read like a
// GPU problem and nothing pointed at the pod limit. A silent revert to a small
// value would reintroduce exactly that misdiagnosis.
func TestDaemon_BuildDaemonSet_DefaultMemoryLimitFitsAnEightGPUNode(t *testing.T) {
	ctr := mpsdContainer(t, NewMpsdDaemon(nil, testMpsdAuditLogTrue, testSupportSMSharingTrue))

	limit := ctr.Resources.Limits.Memory()
	if got := limit.String(); got != "1Gi" {
		t.Errorf("default memory limit = %q, want %q", got, "1Gi")
	}
	// Restated in bytes so a change of unit (e.g. "1000M") that quietly shrinks
	// the budget below one context per GPU still fails.
	const eightGPUContexts = 8 * 50 * 1024 * 1024
	if limit.Value() < eightGPUContexts {
		t.Errorf("default memory limit = %d bytes, too small for 8 MPS server contexts (%d bytes)", limit.Value(), eightGPUContexts)
	}
}

// A CRD resources block merges over the defaults. If it replaced them, the
// common "my node has 16 GPUs, raise limits.memory" override would strip the
// CPU/memory/ephemeral-storage requests and drop mpsd to BestEffort — first in
// line for eviction on the node it is supposed to be fractioning.
func TestDaemon_BuildDaemonSet_ResourceOverrideMerges(t *testing.T) {
	tests := []struct {
		name         string
		override     *corev1.ResourceRequirements
		wantMemLimit string
		wantMemReq   string
	}{
		{
			name:         "no override keeps the built-in defaults",
			override:     nil,
			wantMemLimit: "1Gi",
			wantMemReq:   mpsdMemRequest,
		},
		{
			name: "limits-only override keeps the default requests",
			override: &corev1.ResourceRequirements{
				Limits: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
			},
			wantMemLimit: "4Gi",
			wantMemReq:   mpsdMemRequest,
		},
		{
			name: "requests-only override keeps the default memory limit",
			override: &corev1.ResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("512Mi")},
			},
			wantMemLimit: "1Gi",
			wantMemReq:   "512Mi",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctr := mpsdContainer(t, NewMpsdDaemon(&v1alpha1.MpsDaemonSpec{Resources: tt.override},
				testMpsdAuditLogTrue, testSupportSMSharingTrue))

			if got := ctr.Resources.Limits.Memory().String(); got != tt.wantMemLimit {
				t.Errorf("limits.memory = %q, want %q", got, tt.wantMemLimit)
			}
			if got := ctr.Resources.Requests.Memory().String(); got != tt.wantMemReq {
				t.Errorf("requests.memory = %q, want %q", got, tt.wantMemReq)
			}
			if got := ctr.Resources.Requests.Cpu().String(); got != mpsdCPURequest {
				t.Errorf("requests.cpu = %q, want %q; the override replaced instead of merging", got, mpsdCPURequest)
			}
			if ctr.Resources.Requests.StorageEphemeral().IsZero() {
				t.Error("requests.ephemeral-storage was dropped by the override")
			}
			if _, hasCPULimit := ctr.Resources.Limits[corev1.ResourceCPU]; hasCPULimit {
				t.Error("expected no CPU limit on mpsd")
			}
		})
	}
}

// mpsd creates the drain socket at startup, so the pod must mount the socket's
// parent DIRECTORY with DirectoryOrCreate. A HostPathType Socket mount on the
// file itself would make the kubelet refuse to start the very pod responsible
// for creating it — an unrecoverable chicken-and-egg on every fresh node.
func TestDaemon_BuildDaemonSet_DrainSocketMountsParentDirectory(t *testing.T) {
	tests := []struct {
		name     string
		spec     *v1alpha1.MpsDaemonSpec
		wantDir  string
		wantPath string
	}{
		{
			name:     "nil spec uses the shared default",
			spec:     nil,
			wantDir:  "/var/run/gpu-fractioning/drain",
			wantPath: daemonmgr.DefaultMPSDrainSocketPath,
		},
		{
			name:     "unset drainSocketPath uses the shared default",
			spec:     &v1alpha1.MpsDaemonSpec{},
			wantDir:  "/var/run/gpu-fractioning/drain",
			wantPath: daemonmgr.DefaultMPSDrainSocketPath,
		},
		{
			name:     "custom path is honoured",
			spec:     &v1alpha1.MpsDaemonSpec{DrainSocketPath: ptr.To("/custom/drain/mps.sock")},
			wantDir:  "/custom/drain",
			wantPath: "/custom/drain/mps.sock",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewMpsdDaemon(tt.spec, testMpsdAuditLogTrue, testSupportSMSharingTrue).BuildDaemonSet(defaultOpts())
			ctr := ds.Spec.Template.Spec.Containers[0]

			vol := hostPathVolume(ds.Spec.Template.Spec.Volumes, volumeDrainSock)
			if vol == nil {
				t.Fatalf("volume %q not found in %v", volumeDrainSock, ds.Spec.Template.Spec.Volumes)
			}
			if vol.Path == tt.wantPath {
				t.Fatalf("drain volume path = %q, which is the socket file itself; the kubelet would block the pod that creates it", vol.Path)
			}
			if vol.Path != tt.wantDir {
				t.Errorf("drain volume path = %q, want %q", vol.Path, tt.wantDir)
			}
			if vol.Type == nil {
				t.Errorf("drain volume type = nil, want %q", corev1.HostPathDirectoryOrCreate)
			} else if *vol.Type != corev1.HostPathDirectoryOrCreate {
				// Dereferenced: %v on the pointer prints an address, which says
				// nothing about which HostPathType was actually rendered.
				t.Errorf("drain volume type = %q, want %q", *vol.Type, corev1.HostPathDirectoryOrCreate)
			}

			mount, found := mountPathFor(ctr.VolumeMounts, volumeDrainSock)
			if !found {
				t.Fatalf("mount %q not found in %v", volumeDrainSock, ctr.VolumeMounts)
			}
			if mount != tt.wantDir {
				t.Errorf("drain mount path = %q, want %q", mount, tt.wantDir)
			}

			// The binary is still told the full socket path.
			if got := argValue(ctr.Args, "--drain-socket"); got != tt.wantPath {
				t.Errorf("--drain-socket = %q, want %q", got, tt.wantPath)
			}
		})
	}
}

// An explicit empty drainSocketPath is how an operator switches the drain
// endpoint off. The mount must go away, but the flag must still be passed with
// an empty value: omitting it entirely is indistinguishable from "unset", which
// the binary reads as "use your default" and silently re-enables the endpoint.
func TestDaemon_BuildDaemonSet_DrainSocketDisabled(t *testing.T) {
	ds := NewMpsdDaemon(&v1alpha1.MpsDaemonSpec{DrainSocketPath: ptr.To("")},
		testMpsdAuditLogTrue, testSupportSMSharingTrue).BuildDaemonSet(defaultOpts())
	ctr := ds.Spec.Template.Spec.Containers[0]

	if vol := hostPathVolume(ds.Spec.Template.Spec.Volumes, volumeDrainSock); vol != nil {
		t.Errorf("drain volume = %+v, want it absent when the endpoint is disabled", vol)
	}
	if _, found := mountPathFor(ctr.VolumeMounts, volumeDrainSock); found {
		t.Errorf("drain mount present in %v, want it absent when the endpoint is disabled", ctr.VolumeMounts)
	}

	var idx = -1
	for i, a := range ctr.Args {
		if a == "--drain-socket" {
			idx = i
		}
	}
	if idx == -1 {
		t.Fatalf("args = %v, want --drain-socket passed even when disabled", ctr.Args)
	}
	if idx+1 >= len(ctr.Args) || ctr.Args[idx+1] != "" {
		t.Errorf("args = %v, want --drain-socket followed by an empty value", ctr.Args)
	}
}

// The drain-tuning flags are only meaningful when the CR sets them; passing
// them unconditionally would pin the binary's defaults into the DaemonSet and
// silently override any future change to them in the mpsd image.
func TestDaemon_BuildDaemonSet_DrainTuningArgsOnlyWhenSet(t *testing.T) {
	ctr := mpsdContainer(t, NewMpsdDaemon(&v1alpha1.MpsDaemonSpec{}, testMpsdAuditLogTrue, testSupportSMSharingTrue))
	for _, flag := range []string{"--client-drain-timeout", "--recycle-when-idle"} {
		for _, a := range ctr.Args {
			if a == flag || (len(a) > len(flag) && a[:len(flag)+1] == flag+"=") {
				t.Errorf("args = %v, want no %s when the CR leaves it unset", ctr.Args, flag)
			}
		}
	}

	// Set: emitted in a stable order, after the always-present --drain-socket.
	ctr = mpsdContainer(t, NewMpsdDaemon(&v1alpha1.MpsDaemonSpec{
		GracefulStopDelay:  &metav1.Duration{Duration: 30 * time.Second},
		RecycleWhenIdle:    ptr.To(false),
		ClientDrainTimeout: &metav1.Duration{Duration: 45 * time.Second},
	}, testMpsdAuditLogTrue, testSupportSMSharingTrue))

	want := []string{
		"--graceful-stop-delay", "30s",
		"--drain-socket", daemonmgr.DefaultMPSDrainSocketPath,
		"--recycle-when-idle=false",
		"--client-drain-timeout", "45s",
	}
	if !slices.Equal(ctr.Args, want) {
		t.Errorf("args = %v, want %v", ctr.Args, want)
	}
}

// recycleWhenIdle defaults to true in the CRD, so the operator must render a
// configured false as an explicit =false rather than by dropping the flag.
func TestDaemon_BuildDaemonSet_RecycleWhenIdleIsExplicit(t *testing.T) {
	for _, tt := range []struct {
		name  string
		value bool
		want  string
	}{
		{name: "enabled", value: true, want: "--recycle-when-idle=true"},
		{name: "disabled", value: false, want: "--recycle-when-idle=false"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctr := mpsdContainer(t, NewMpsdDaemon(&v1alpha1.MpsDaemonSpec{RecycleWhenIdle: ptr.To(tt.value)},
				testMpsdAuditLogTrue, testSupportSMSharingTrue))
			if !slices.Contains(ctr.Args, tt.want) {
				t.Errorf("args = %v, want %q", ctr.Args, tt.want)
			}
		})
	}
}

func argValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}
