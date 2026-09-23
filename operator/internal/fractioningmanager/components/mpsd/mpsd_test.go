// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsd

import (
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/common/daemonmgr"
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
	// Without the gpu-operator's third-party GPU client label, a driver upgrade
	// never evicts mpsd and the driver unload fails while MPS holds the module.
	if spec.NodeSelector[daemonmgr.GPUDeployClientLabel] != daemonmgr.GPUDeployClientValue {
		t.Errorf("NodeSelector[%s] = %q, want %q",
			daemonmgr.GPUDeployClientLabel, spec.NodeSelector[daemonmgr.GPUDeployClientLabel], daemonmgr.GPUDeployClientValue)
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
	if len(ctr.Args) != 0 {
		t.Errorf("expected no args when spec is nil, got %v", ctr.Args)
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
