// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fractiond

import (
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1alpha1 "github.com/kai-scheduler/kai-gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/common/daemonmgr"
)

const testDaemonServiceAccountName = "gpu-fractioning-daemon"

// testSupportSMSharingTrue is the sm-sharing chicken bit passed to NewFractiondDaemon
// in tests that don't exercise it directly (see TestDaemon_BuildDaemonSet_SupportSMSharing
// for the dedicated true/false coverage).
const testSupportSMSharingTrue = true

func TestDaemon_BuildDaemonSet_Basics(t *testing.T) {
	d := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{Enabled: true}, testSupportSMSharingTrue)

	if got := d.Name(); got != "fractiond" {
		t.Errorf("Name() = %q, expected %q", got, "fractiond")
	}

	ds := d.BuildDaemonSet(defaultOpts())

	if ds.Name != "gpu-fractioning-fractiond" {
		t.Errorf("DaemonSet name = %q, expected %q", ds.Name, "gpu-fractioning-fractiond")
	}
	if ds.Namespace != "gpu-fractioning-system" {
		t.Errorf("DaemonSet namespace = %q, expected %q", ds.Namespace, "gpu-fractioning-system")
	}

	spec := ds.Spec.Template.Spec
	if !spec.HostPID {
		t.Error("expected HostPID=true")
	}
	if spec.ServiceAccountName != "" {
		t.Errorf("serviceAccountName = %q, expected empty", spec.ServiceAccountName)
	}

	// Node selector
	if spec.NodeSelector == nil || spec.NodeSelector["nvidia.com/gpu.present"] != "true" {
		t.Errorf("NodeSelector = %v, expected nvidia.com/gpu.present=true", spec.NodeSelector)
	}
	// The pod holds NVML/GPU handles through the metricsd sidecar, so it carries
	// the gpu-operator's third-party GPU client label like mpsd does.
	if spec.NodeSelector[daemonmgr.GPUDeployClientLabel] != daemonmgr.GPUDeployClientValue {
		t.Errorf("NodeSelector[%s] = %q, expected %q",
			daemonmgr.GPUDeployClientLabel, spec.NodeSelector[daemonmgr.GPUDeployClientLabel], daemonmgr.GPUDeployClientValue)
	}

	// Two containers: fractiond (the NRI plugin) + metricsd (the metrics sidecar).
	if len(spec.Containers) != 2 {
		t.Fatalf("expected 2 containers (fractiond + metricsd), got %d", len(spec.Containers))
	}

	ctr := containerByName(t, spec.Containers, "fractiond")
	if ctr.Image != "fake.io/org/fractiond:v0.1.0" {
		t.Errorf("container image = %q, want %q", ctr.Image, "fake.io/org/fractiond:v0.1.0")
	}
	if ctr.SecurityContext == nil || ctr.SecurityContext.Privileged == nil || !*ctr.SecurityContext.Privileged {
		t.Error("expected privileged=true")
	}

	// metricsd sidecar: image, privileged, metrics port, read-only shared map mount.
	metricsd := containerByName(t, spec.Containers, "metricsd")
	if metricsd.Image != "fake.io/org/metricsd:v0.1.0" {
		t.Errorf("metricsd image = %q, want %q", metricsd.Image, "fake.io/org/metricsd:v0.1.0")
	}
	if len(metricsd.Ports) != 1 || metricsd.Ports[0].ContainerPort != 2112 {
		t.Errorf("expected metricsd to expose port 2112, got %v", metricsd.Ports)
	}
	if !hasMount(metricsd.VolumeMounts, "map-dir", "/var/run/gpu-fractioning/map", true) {
		t.Errorf("expected metricsd to mount map-dir read-only, got %v", metricsd.VolumeMounts)
	}
	if !hasMount(ctr.VolumeMounts, "map-dir", "/var/run/gpu-fractioning/map", false) {
		t.Errorf("expected fractiond to mount map-dir read-write, got %v", ctr.VolumeMounts)
	}

	// The shared map dir is a hostPath so the fsstore survives pod restarts.
	var mapDirHostPath string
	for _, v := range spec.Volumes {
		if v.Name == "map-dir" && v.HostPath != nil {
			mapDirHostPath = v.HostPath.Path
		}
	}
	if mapDirHostPath != "/var/run/gpu-fractioning/map" {
		t.Errorf("expected map-dir to be a hostPath volume at /var/run/gpu-fractioning/map, got %q", mapDirHostPath)
	}

	// Volumes
	volumePaths := make(map[string]string)
	for _, v := range spec.Volumes {
		if v.HostPath != nil {
			volumePaths[v.Name] = v.HostPath.Path
		}
	}
	if volumePaths["nri-socket"] != "/var/run/nri" {
		t.Errorf("nri-socket volume path = %q, expected %q", volumePaths["nri-socket"], "/var/run/nri")
	}
	if volumePaths["mps-pipe"] != "/run/nvidia-mps" {
		t.Errorf("mps-pipe volume path = %q, expected %q", volumePaths["mps-pipe"], "/run/nvidia-mps")
	}
	// With a nil fractioningAgent spec, retroactive enforcement is off (the default
	// is materialized by the API server only when a fractioningAgent block exists),
	// so the CRI socket must not be mounted.
	if _, ok := volumePaths["cri-socket"]; ok {
		t.Errorf("cri-socket volume should be absent when fractioningAgent spec is nil, got %q", volumePaths["cri-socket"])
	}

	// Volume mounts
	mountPaths := make(map[string]string)
	for _, m := range ctr.VolumeMounts {
		mountPaths[m.Name] = m.MountPath
	}
	if mountPaths["nri-socket"] != "/var/run/nri" {
		t.Errorf("nri-socket mount = %q, expected %q", mountPaths["nri-socket"], "/var/run/nri")
	}
	if mountPaths["mps-pipe"] != "/run/nvidia-mps" {
		t.Errorf("mps-pipe mount = %q, expected %q", mountPaths["mps-pipe"], "/run/nvidia-mps")
	}
	if _, ok := mountPaths["cri-socket"]; ok {
		t.Errorf("cri-socket mount should be absent when fractioningAgent spec is nil, got %q", mountPaths["cri-socket"])
	}

	// Labels
	labels := ds.Labels
	if labels[daemonmgr.LabelManagedBy] != daemonmgr.ManagedByValue {
		t.Errorf("missing managed-by label, got %v", labels)
	}
	if labels[daemonmgr.LabelComponent] != "fractiond" {
		t.Errorf("missing component label, got %v", labels)
	}

	// The sm-sharing chicken bit is always passed, even when spec is nil (it is
	// an installation-time toggle, independent of the fractioningAgent CRD spec).
	args := ctr.Args
	if len(args) != 1 || args[0] != "--support-sm-sharing=true" {
		t.Errorf("expected only --support-sm-sharing=true when spec is nil, got %v", args)
	}
}

func TestDaemon_BuildDaemonSet_MetricsNVMLAccess(t *testing.T) {
	d := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{Enabled: true}, testSupportSMSharingTrue)
	spec := d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec

	// Pod must opt into the RuntimeClass so the NVIDIA container runtime injects
	// libnvidia-ml.so for metricsd; without this NVML returns ERROR_LIBRARY_NOT_FOUND.
	if spec.RuntimeClassName == nil || *spec.RuntimeClassName != daemonmgr.DefaultRuntimeClassName {
		t.Errorf("runtimeClassName = %v, expected \"nvidia\"", spec.RuntimeClassName)
	}

	metricsd := containerByName(t, spec.Containers, "metricsd")

	envMap := make(map[string]string, len(metricsd.Env))
	for _, e := range metricsd.Env {
		envMap[e.Name] = e.Value
	}
	if envMap["NVIDIA_VISIBLE_DEVICES"] != "all" {
		t.Errorf("NVIDIA_VISIBLE_DEVICES = %q, expected \"all\"", envMap["NVIDIA_VISIBLE_DEVICES"])
	}
	if envMap["NVIDIA_DRIVER_CAPABILITIES"] != "utility" {
		t.Errorf("NVIDIA_DRIVER_CAPABILITIES = %q, expected \"utility\"", envMap["NVIDIA_DRIVER_CAPABILITIES"])
	}
}

func TestDaemon_BuildDaemonSet_MetricsExtraNVMLVolumesOnlyWhenMetricsEnabled(t *testing.T) {
	d := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{
		Enabled: true,
		Volumes: []corev1.Volume{
			{
				Name:         "nvml-config",
				VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
			},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "nvml-config", MountPath: "/etc/nvml-mock", ReadOnly: true},
		},
	}, testSupportSMSharingTrue)
	spec := d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec

	var foundVolume bool
	for _, v := range spec.Volumes {
		if v.Name == "nvml-config" {
			foundVolume = true
		}
	}
	if !foundVolume {
		t.Fatalf("expected metricsAgent volume to be available to the pod, got %v", spec.Volumes)
	}

	metricsd := containerByName(t, spec.Containers, "metricsd")
	if !hasMount(metricsd.VolumeMounts, "nvml-config", "/etc/nvml-mock", true) {
		t.Fatalf("expected metricsd to mount metricsAgent volume, got %v", metricsd.VolumeMounts)
	}

	d = NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{
		Enabled: false,
		Volumes: []corev1.Volume{
			{
				Name:         "nvml-config",
				VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
			},
		},
	}, testSupportSMSharingTrue)
	spec = d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec
	for _, v := range spec.Volumes {
		if v.Name == "nvml-config" {
			t.Fatalf("metricsAgent volume should be absent when metricsd is disabled, got %v", spec.Volumes)
		}
	}
}

func TestDaemon_BuildDaemonSet_MetricNamesArgs(t *testing.T) {
	d := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{
		Enabled: true,
		MetricNames: &v1alpha1.MetricNamesSpec{
			GPUMemoryUsedBytes:      "custom_gpu_memory_used_bytes",
			GPUSMUtilizationPercent: "custom_gpu_utilization",
		},
	}, testSupportSMSharingTrue)
	spec := d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec
	metricsd := containerByName(t, spec.Containers, "metricsd")

	args := strings.Join(metricsd.Args, " ")
	for _, want := range []string{
		"--metric-name-gpu-memory-used-bytes custom_gpu_memory_used_bytes",
		"--metric-name-gpu-sm-utilization-percent custom_gpu_utilization",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("metricsd args %q missing %q", args, want)
		}
	}
	// unset normalized name must not add a flag (keeps the built-in default)
	if strings.Contains(args, "--metric-name-gpu-sm-utilization-percent-normalized") {
		t.Errorf("unexpected normalized metric-name flag in %q", args)
	}
}

func TestDaemon_BuildDaemonSet_MetricsRuntimeClassOverride(t *testing.T) {
	custom := "custom-nvidia"
	opts := defaultOpts()
	opts.RuntimeClassName = &custom

	d := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{Enabled: true}, testSupportSMSharingTrue)
	spec := d.BuildDaemonSet(opts).Spec.Template.Spec
	if spec.RuntimeClassName == nil || *spec.RuntimeClassName != "custom-nvidia" {
		t.Errorf("runtimeClassName = %v, expected %q", spec.RuntimeClassName, custom)
	}
}

func TestDaemon_BuildDaemonSet_MetricsRuntimeClassEmpty(t *testing.T) {
	opts := defaultOpts()
	opts.RuntimeClassName = nil

	d := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{Enabled: true}, testSupportSMSharingTrue)
	spec := d.BuildDaemonSet(opts).Spec.Template.Spec
	if spec.RuntimeClassName != nil {
		t.Errorf("runtimeClassName = %v, expected nil (node default)", spec.RuntimeClassName)
	}
}

func TestDaemon_BuildDaemonSet_MetricsDisabled(t *testing.T) {
	d := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{Enabled: false}, testSupportSMSharingTrue)

	spec := d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec
	if len(spec.InitContainers) != 0 {
		t.Fatalf("expected no init containers when metrics disabled, got %d", len(spec.InitContainers))
	}
	if len(spec.Containers) != 1 {
		t.Fatalf("expected only the fractiond container when metrics disabled, got %d", len(spec.Containers))
	}
	if spec.Containers[0].Name != "fractiond" {
		t.Errorf("expected sole container to be fractiond, got %q", spec.Containers[0].Name)
	}
	if spec.RuntimeClassName != nil {
		t.Errorf("runtimeClassName = %v, expected nil when only fractiond runs", spec.RuntimeClassName)
	}
}

// The readiness probe must hit the binary's /readyz endpoint so a fractiond
// that never registers with NRI reports Ready=False (and the node condition
// stays false) instead of a false-positive AllDaemonsReady.
func TestDaemon_BuildDaemonSet_ReadinessProbe(t *testing.T) {
	d := NewFractiondDaemon(nil, nil, testSupportSMSharingTrue)
	ds := d.BuildDaemonSet(defaultOpts())
	ctr := ds.Spec.Template.Spec.Containers[0]

	probe := ctr.ReadinessProbe
	if probe == nil {
		t.Fatal("expected a readiness probe on the fractiond container")
	}
	httpGet := probe.HTTPGet
	if httpGet == nil {
		t.Fatal("expected an httpGet readiness probe")
	}
	if httpGet.Path != "/readyz" {
		t.Errorf("probe path = %q, expected %q", httpGet.Path, "/readyz")
	}
	if httpGet.Port.IntValue() != defaultReadinessPort {
		t.Errorf("probe port = %d, expected %d", httpGet.Port.IntValue(), defaultReadinessPort)
	}

	var found bool
	for _, p := range ctr.Ports {
		if p.ContainerPort == defaultReadinessPort {
			found = true
		}
	}
	if !found {
		t.Errorf("expected container port %d to be declared, got %v", defaultReadinessPort, ctr.Ports)
	}

	// With no spec override, no --readiness-port arg is passed: the binary's
	// built-in default keeps older fractiond images (without the flag) working.
	for _, a := range ctr.Args {
		if a == "--readiness-port" {
			t.Errorf("expected no --readiness-port arg by default, got args %v", ctr.Args)
		}
	}
}

// Liveness on fractiond is a TCP connect (not /readyz) so a fractiond still
// retrying its NRI registration is not restart-looped; resources are always set.
func TestDaemon_BuildDaemonSet_FractiondLivenessAndResources(t *testing.T) {
	ctr := NewFractiondDaemon(nil, nil, testSupportSMSharingTrue).BuildDaemonSet(defaultOpts()).Spec.Template.Spec.Containers[0]

	if ctr.LivenessProbe == nil || ctr.LivenessProbe.TCPSocket == nil {
		t.Fatalf("expected a TCPSocket liveness probe, got %+v", ctr.LivenessProbe)
	}
	if got := ctr.LivenessProbe.TCPSocket.Port.IntValue(); got != defaultReadinessPort {
		t.Errorf("liveness port = %d, want %d", got, defaultReadinessPort)
	}
	assertDaemonResources(t, ctr, fractiondMemLimit)
}

// metricsd exposes readiness (/readyz) and liveness (/healthz) on its metrics
// port so a broken sidecar flips pod readiness / is restarted; resources are set.
func TestDaemon_BuildDaemonSet_MetricsdProbesAndResources(t *testing.T) {
	spec := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{Enabled: true}, testSupportSMSharingTrue).
		BuildDaemonSet(defaultOpts()).Spec.Template.Spec
	metricsd := containerByName(t, spec.Containers, "metricsd")

	if metricsd.ReadinessProbe == nil || metricsd.ReadinessProbe.HTTPGet == nil {
		t.Fatalf("expected an httpGet readiness probe on metricsd, got %+v", metricsd.ReadinessProbe)
	}
	if metricsd.LivenessProbe == nil || metricsd.LivenessProbe.HTTPGet == nil {
		t.Fatalf("expected an httpGet liveness probe on metricsd, got %+v", metricsd.LivenessProbe)
	}
	if metricsd.ReadinessProbe.HTTPGet.Path != metricsdReadyzPath {
		t.Errorf("readiness path = %q, want %q", metricsd.ReadinessProbe.HTTPGet.Path, metricsdReadyzPath)
	}
	if metricsd.LivenessProbe.HTTPGet.Path != metricsdHealthzPath {
		t.Errorf("liveness path = %q, want %q", metricsd.LivenessProbe.HTTPGet.Path, metricsdHealthzPath)
	}
	if got := int32(metricsd.ReadinessProbe.HTTPGet.Port.IntValue()); got != metricsPort {
		t.Errorf("metricsd probe port = %d, want %d", got, metricsPort)
	}
	assertDaemonResources(t, metricsd, metricsdMemLimit)
}

// assertDaemonResources checks the shared daemon resource contract: CPU, memory
// and ephemeral-storage requests set, a memory limit equal to wantMemLimit, and
// no CPU limit.
func assertDaemonResources(t *testing.T, ctr corev1.Container, wantMemLimit string) {
	t.Helper()
	if ctr.Resources.Requests.Cpu().IsZero() {
		t.Error("expected a CPU request")
	}
	if ctr.Resources.Requests.Memory().IsZero() {
		t.Error("expected a memory request")
	}
	if ctr.Resources.Requests.StorageEphemeral().IsZero() {
		t.Error("expected an ephemeral-storage request")
	}
	if got := ctr.Resources.Limits.Memory().String(); got != wantMemLimit {
		t.Errorf("memory limit = %q, want %q", got, wantMemLimit)
	}
	if _, hasCPULimit := ctr.Resources.Limits[corev1.ResourceCPU]; hasCPULimit {
		t.Error("expected no CPU limit on a system daemon")
	}
}

// When spec.readinessPort is set, the operator passes it to the binary and
// points the probe (and container port) at the same value.
func TestDaemon_BuildDaemonSet_ReadinessPortOverride(t *testing.T) {
	d := NewFractiondDaemon(&v1alpha1.FractioningAgentSpec{
		ReadinessPort: ptr.To(int32(9999)),
	}, nil, testSupportSMSharingTrue)
	ctr := d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec.Containers[0]

	if got := ctr.ReadinessProbe.HTTPGet.Port.IntValue(); got != 9999 {
		t.Errorf("probe port = %d, expected 9999", got)
	}
	if len(ctr.Ports) != 1 || ctr.Ports[0].ContainerPort != 9999 {
		t.Errorf("container ports = %v, expected readiness port 9999", ctr.Ports)
	}

	var argPort string
	for i, a := range ctr.Args {
		if a == "--readiness-port" && i+1 < len(ctr.Args) {
			argPort = ctr.Args[i+1]
		}
	}
	if argPort != "9999" {
		t.Errorf("--readiness-port arg = %q, expected %q (args %v)", argPort, "9999", ctr.Args)
	}
}

func TestDaemon_BuildDaemonSet_Args(t *testing.T) {
	d := NewFractiondDaemon(&v1alpha1.FractioningAgentSpec{
		AnnotationPrefix:       "custom.prefix.",
		FailOpen:               ptr.To(true),
		NRISocketPath:          "/custom/nri.sock",
		LogLevel:               "debug",
		RetryInterval:          &metav1.Duration{Duration: 10 * time.Second},
		StableThreshold:        &metav1.Duration{Duration: 10 * time.Minute},
		MaxRetries:             ptr.To(int32(5)),
		RetroactiveEnforcement: true,
		CRISocketPath:          "/custom/cri.sock",
	}, nil, testSupportSMSharingTrue)

	ds := d.BuildDaemonSet(defaultOpts())
	args := ds.Spec.Template.Spec.Containers[0].Args

	expected := []string{
		"--support-sm-sharing=true",
		"--annotation-prefix", "custom.prefix.",
		"--fail-open",
		"--socket-path", "/custom/nri.sock",
		"--log-level", "debug",
		"--retry-interval", "10s",
		"--stable-threshold", "10m0s",
		"--max-retries", "5",
		"--retroactive-enforcement=true",
		"--cri-socket", "/custom/cri.sock",
	}

	if len(args) != len(expected) {
		t.Fatalf("args length = %d, want %d\ngot:  %v\nwant: %v", len(args), len(expected), args, expected)
	}
	for i := range expected {
		if args[i] != expected[i] {
			t.Errorf("args[%d] = %q, want %q", i, args[i], expected[i])
		}
	}

	// The CRI socket's parent directory is mounted (DirectoryOrCreate), so a
	// missing/relocated socket can never block the pod from starting; the binary
	// still dials the full socket path passed via --cri-socket.
	var criMount string
	for _, m := range ds.Spec.Template.Spec.Containers[0].VolumeMounts {
		if m.Name == "cri-socket" {
			criMount = m.MountPath
		}
	}
	if criMount != "/custom" {
		t.Errorf("cri-socket mount = %q, want %q", criMount, "/custom")
	}

	var criVol *corev1.HostPathVolumeSource
	for _, v := range ds.Spec.Template.Spec.Volumes {
		if v.Name == "cri-socket" {
			criVol = v.HostPath
		}
	}
	if criVol == nil {
		t.Fatal("cri-socket volume not found")
	}
	if criVol.Path != "/custom" {
		t.Errorf("cri-socket volume path = %q, want %q", criVol.Path, "/custom")
	}
	if criVol.Type == nil || *criVol.Type != corev1.HostPathDirectoryOrCreate {
		t.Errorf("cri-socket volume type = %v, want %q", criVol.Type, corev1.HostPathDirectoryOrCreate)
	}
}

func TestDaemon_BuildDaemonSet_EnforcementDisabled(t *testing.T) {
	d := NewFractiondDaemon(&v1alpha1.FractioningAgentSpec{
		RetroactiveEnforcement: false,
	}, nil, testSupportSMSharingTrue)

	ds := d.BuildDaemonSet(defaultOpts())
	spec := ds.Spec.Template.Spec
	ctr := spec.Containers[0]

	// The CRI socket must not be mounted when enforcement is disabled.
	for _, v := range spec.Volumes {
		if v.Name == "cri-socket" {
			t.Error("cri-socket volume should be absent when enforcement is disabled")
		}
	}
	for _, m := range ctr.VolumeMounts {
		if m.Name == "cri-socket" {
			t.Error("cri-socket mount should be absent when enforcement is disabled")
		}
	}

	// The disable must be passed explicitly to override the binary default.
	foundDisable := false
	for _, a := range ctr.Args {
		if a == "--retroactive-enforcement=false" {
			foundDisable = true
		}
		if a == "--cri-socket" {
			t.Error("--cri-socket should not be set when enforcement is disabled")
		}
	}
	if !foundDisable {
		t.Errorf("expected --retroactive-enforcement=false in args, got %v", ctr.Args)
	}
}

// TestDaemon_BuildDaemonSet_SupportSMSharing verifies the Helm-injected
// sm-sharing chicken bit is always passed as an explicit arg, independent of
// whether a fractioningAgent spec is configured.
func TestDaemon_BuildDaemonSet_SupportSMSharing(t *testing.T) {
	for _, tt := range []struct {
		name             string
		supportSMSharing bool
		want             string
	}{
		{name: "enabled", supportSMSharing: true, want: "--support-sm-sharing=true"},
		{name: "disabled", supportSMSharing: false, want: "--support-sm-sharing=false"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			d := NewFractiondDaemon(&v1alpha1.FractioningAgentSpec{}, nil, tt.supportSMSharing)
			ctr := d.BuildDaemonSet(defaultOpts()).Spec.Template.Spec.Containers[0]

			var found bool
			for _, a := range ctr.Args {
				if a == tt.want {
					found = true
				}
			}
			if !found {
				t.Errorf("expected %q in args, got %v", tt.want, ctr.Args)
			}
		})
	}
}

// Both containers in this pod are asserted: fractiond and the metricsd sidecar
// are separate injection sites, and a pod where only one of them enforces FIPS
// is a half-configured install that nothing else would flag.
func TestDaemon_BuildDaemonSet_FIPSOnly(t *testing.T) {
	tests := []struct {
		name     string
		fipsOnly bool
		// wantGODEBUG is the value expected on every container; empty means the
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
			spec := NewFractiondDaemon(nil, &v1alpha1.MetricsAgentSpec{Enabled: true}, testSupportSMSharingTrue).
				BuildDaemonSet(opts).Spec.Template.Spec

			for _, name := range []string{"fractiond", "metricsd"} {
				ctr := containerByName(t, spec.Containers, name)
				if got := envValue(ctr.Env, "GODEBUG"); got != tt.wantGODEBUG {
					t.Errorf("%s GODEBUG env = %q, want %q", name, got, tt.wantGODEBUG)
				}
			}

			// metricsd is the only container here with pre-existing env, and
			// these two vars are what get libnvidia-ml.so mounted in, so
			// injection must append rather than replace.
			metricsd := containerByName(t, spec.Containers, "metricsd")
			if got := envValue(metricsd.Env, "NVIDIA_VISIBLE_DEVICES"); got != "all" {
				t.Errorf("NVIDIA_VISIBLE_DEVICES env = %q, want %q; FIPS injection clobbered existing env", got, "all")
			}
			if got := envValue(metricsd.Env, "NVIDIA_DRIVER_CAPABILITIES"); got != "utility" {
				t.Errorf("NVIDIA_DRIVER_CAPABILITIES env = %q, want %q; FIPS injection clobbered existing env", got, "utility")
			}

			// fractiond has no env of its own, so with enforcement off it must
			// stay nil rather than gain an empty list, keeping the default
			// DaemonSet spec unchanged.
			if !tt.fipsOnly {
				if ctr := containerByName(t, spec.Containers, "fractiond"); ctr.Env != nil {
					t.Errorf("fractiond env = %v, want nil when FIPS-only is off", ctr.Env)
				}
			}
		})
	}
}

func defaultOpts() daemonmgr.BuildOptions {
	return daemonmgr.BuildOptions{
		Namespace:          "gpu-fractioning-system",
		NodeSelector:       map[string]string{"nvidia.com/gpu.present": "true"},
		ServiceAccountName: testDaemonServiceAccountName,
		RuntimeClassName:   ptr.To(daemonmgr.DefaultRuntimeClassName),
		DefaultImages: map[string]daemonmgr.ImageSpec{
			"fractiond": {
				Repository: "fake.io/org/fractiond",
				Tag:        "v0.1.0",
			},
			"metricsd": {
				Repository: "fake.io/org/metricsd",
				Tag:        "v0.1.0",
			},
		},
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

func containerByName(t *testing.T, containers []corev1.Container, name string) corev1.Container {
	t.Helper()
	for _, c := range containers {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("container %q not found", name)
	return corev1.Container{}
}

func hasMount(mounts []corev1.VolumeMount, name, path string, readOnly bool) bool {
	for _, m := range mounts {
		if m.Name == name && m.MountPath == path && m.ReadOnly == readOnly {
			return true
		}
	}
	return false
}
