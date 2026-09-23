// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fractiond

import (
	"log/slog"
	"net"
	"path/filepath"
	"strconv"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/common/daemonmgr"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	daemonName           = "fractiond"
	metricsdName         = "metricsd"
	defaultMPSPipeDir    = "/run/nvidia-mps"
	defaultNRISocketDir  = "/var/run/nri"
	defaultCRISocketPath = "/run/containerd/containerd.sock"
	// containerPodMapDir is the shared handoff directory: fractiond writes the
	// container→pod mapping here and the metricsd sidecar reads it. Must match
	// fractiond's and metricsd's built-in default (fsstore.DefaultMapDir).
	containerPodMapDir = "/var/run/gpu-fractioning/map"
	// metricsPort is the Prometheus exporter port served by the metricsd sidecar.
	metricsPort = 2112

	volumeNRISocket = "nri-socket"
	volumeMPSPipe   = "mps-pipe"
	volumeMapDir    = "map-dir"
	volumeCRISocket = "cri-socket"

	// defaultReadinessPort must match the fractiond binary's default
	// (fractioning-manager/fractiond): it serves /readyz on --readiness-port. Used
	// when spec.readinessPort is unset, in which case no argument is passed to
	// the binary — keeping older fractiond images (without the flag) compatible.
	defaultReadinessPort = 8093
	readyzPath           = "/readyz"

	// metricsd liveness/readiness HTTP paths — must match the metricsd exporter's
	// HealthzPath/ReadyzPath.
	metricsdHealthzPath = "/healthz"
	metricsdReadyzPath  = "/readyz"

	// Resource requests/limits for the managed daemon containers. Requests set QoS
	// and let the scheduler account for the pods; only a memory limit is applied
	// (no CPU limit) so these latency-sensitive privileged daemons are never
	// CPU-throttled. Conservative defaults for v0.1.0 — see also mpsd.
	fractiondCPURequest = "50m"
	fractiondMemRequest = "64Mi"
	fractiondMemLimit   = "256Mi"

	metricsdCPURequest = "50m"
	metricsdMemRequest = "128Mi"
	metricsdMemLimit   = "512Mi"
)

// daemon implements daemonmgr.ManagedDaemon for the fractiond NRI plugin plus its
// metricsd metrics sidecar. Both run in a single DaemonSet pod.
type daemon struct {
	fractioningSpec  *v1alpha1.FractioningAgentSpec
	metricsSpec      *v1alpha1.MetricsAgentSpec
	supportSMSharing bool // Helm-injected sm-sharing chicken bit
}

// NewFractiondDaemon returns a ManagedDaemon for the fractiond NRI plugin. The
// metricsAgent spec (may be nil) configures the co-located metricsd sidecar.
// supportSMSharing is the Helm-injected sm-sharing chicken bit: when false,
// fractiond rejects the gpu-compute.mode: sm-sharing annotation like any other
// invalid value, since the shared MPS server mpsd would back it with is also
// disabled by the same toggle.
func NewFractiondDaemon(spec *v1alpha1.FractioningAgentSpec, metrics *v1alpha1.MetricsAgentSpec, supportSMSharing bool) daemonmgr.ManagedDaemon {
	return &daemon{fractioningSpec: spec, metricsSpec: metrics, supportSMSharing: supportSMSharing}
}

func (d *daemon) Name() string { return daemonName }

// BuildDaemonSet constructs the desired DaemonSet. It is a single DaemonSet pod
// with one or two regular containers:
//
//   - fractiond: the NRI plugin. Runs privileged with host PID visibility so it can
//     register with the host containerd's NRI endpoint and observe container
//     lifecycle. It writes the container→pod mapping to a shared hostPath.
//   - metricsd: the metrics sidecar. Reads the same mapping (read-only) and exports
//     per-pod GPU metrics on :2112. It relies on the pod's host PID namespace to
//     resolve NVML-reported host PIDs to cgroups (so no /host/proc mount is
//     needed), and runs privileged for NVML device access.
//
// Host paths mounted: the NRI socket directory (fractiond) and the MPS pipe
// directory (shared with mpsd). The map directory is a hostPath shared between
// fractiond and metricsd so the mapping can survive fractiond pod restarts.
func (d *daemon) BuildDaemonSet(opts daemonmgr.BuildOptions) *appsv1.DaemonSet {
	result := daemonmgr.BaseDaemonSet(daemonName, opts.Namespace)

	podSpec := &result.Spec.Template.Spec
	// HostPID is required so fractiond can observe container lifecycle events and
	// so metricsd can resolve NVML host PIDs through its own /proc.
	podSpec.NodeSelector = daemonmgr.DaemonNodeSelector(opts.NodeSelector)
	podSpec.HostPID = true

	fractiondContainer, fractiondVolumes := d.buildFractiondContainer(opts)
	podSpec.Volumes = append(podSpec.Volumes, fractiondVolumes...)

	metricsOn := d.metricsEnabled()
	if metricsOn {
		vol, mount := metricsSharedVolume()
		podSpec.Volumes = append(podSpec.Volumes, vol)
		fractiondContainer.VolumeMounts = append(fractiondContainer.VolumeMounts, mount)
		podSpec.RuntimeClassName = opts.RuntimeClassName
	}
	// fractiondContainer is a value type: append must happen after the map-dir mount
	// is added so the VolumeMount is included in the copy placed into the slice.
	podSpec.Containers = append(podSpec.Containers, fractiondContainer)
	if metricsOn {
		d.applyMetricsSidecar(result, opts)
	}

	return result
}

func (d *daemon) metricsEnabled() bool {
	return d.metricsSpec != nil && d.metricsSpec.Enabled
}

// metricsSharedVolume returns the map-dir hostPath volume and the corresponding
// mount for the fractiond container (writer side of the container→pod mapping handoff).
//
// A hostPath (rather than emptyDir) is used so the container→pod mapping
// survives fractiond DaemonSet pod restarts. With emptyDir, the mapping is lost
// when the pod is deleted, and k3s/containerd's NRI implementation does not
// replay Synchronize events for already-running containers when a new plugin
// registers — leaving metricsd unable to attribute GPU processes until new
// containers are created. A hostPath at the same path means the fsstore written
// by the previous fractiond pod is still present for the new one to read.
func metricsSharedVolume() (corev1.Volume, corev1.VolumeMount) {
	return corev1.Volume{
			Name: volumeMapDir,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: containerPodMapDir,
					Type: new(corev1.HostPathDirectoryOrCreate),
				},
			},
		},
		corev1.VolumeMount{Name: volumeMapDir, MountPath: containerPodMapDir}
}

// applyMetricsSidecar adds the metricsd container and Prometheus scrape
// annotations to the DaemonSet.
func (d *daemon) applyMetricsSidecar(result *appsv1.DaemonSet, opts daemonmgr.BuildOptions) {
	if d.metricsSpec != nil {
		result.Spec.Template.Spec.Volumes = append(result.Spec.Template.Spec.Volumes, d.metricsSpec.Volumes...)
	}
	result.Spec.Template.Spec.Containers = append(result.Spec.Template.Spec.Containers, d.buildMetricsdContainer(opts))

	if result.Spec.Template.Annotations == nil {
		result.Spec.Template.Annotations = map[string]string{}
	}
	result.Spec.Template.Annotations["prometheus.io/scrape"] = "true"
	result.Spec.Template.Annotations["prometheus.io/port"] = d.metricsAnnotationPort()
	result.Spec.Template.Annotations["prometheus.io/path"] = d.metricsAnnotationPath()
}

// metricsListenPort returns the TCP port metricsd actually listens on: the port
// of the configured metricsAgent.address, or metricsPort when unset/invalid. It
// is the single source of truth for the container port, the scrape annotation,
// and the health probes so they can never drift apart.
func (d *daemon) metricsListenPort() int32 {
	if d.metricsSpec != nil && d.metricsSpec.Address != "" {
		_, port, err := net.SplitHostPort(d.metricsSpec.Address)
		if err != nil {
			slog.Warn("invalid metricsAgent.address, falling back to default port", "address", d.metricsSpec.Address, "err", err)
		} else if port != "" {
			if p, perr := strconv.Atoi(port); perr == nil && p > 0 && p <= 65535 {
				return int32(p)
			}
			slog.Warn("invalid metricsAgent.address port, falling back to default", "address", d.metricsSpec.Address)
		}
	}
	return metricsPort
}

// metricsAnnotationPort returns the port string for the prometheus.io/port
// annotation, derived from metricsListenPort.
func (d *daemon) metricsAnnotationPort() string {
	return strconv.Itoa(int(d.metricsListenPort()))
}

// metricsAnnotationPath returns the path for the prometheus.io/path annotation.
func (d *daemon) metricsAnnotationPath() string {
	if d.metricsSpec != nil && d.metricsSpec.Path != "" {
		return d.metricsSpec.Path
	}
	return "/metrics"
}

// buildFractiondContainer returns the main fractiond container and its required
// host-path volumes. The map-dir mount is added by the caller when metricsd is enabled.
func (d *daemon) buildFractiondContainer(opts daemonmgr.BuildOptions) (corev1.Container, []corev1.Volume) {
	image := opts.DefaultImages[daemonName]

	// fractiond retries its NRI connection indefinitely by default, so the
	// process stays Running even when it never registers with containerd. The
	// binary serves /readyz on the readiness port, flipping to ready only once
	// the NRI plugin is registered and synchronized — probing it keeps the pod
	// (and the node's gpu-fractioning Ready condition) not-ready until fractiond
	// actually works.
	readinessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path: readyzPath,
				Port: intstr.FromInt32(d.readinessPort()),
			},
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       10,
	}

	// Liveness is a bare TCP connect to the readiness port, NOT a /readyz GET: the
	// endpoint reports not-ready for as long as fractiond is (legitimately) retrying
	// its NRI registration, so probing readiness for liveness would restart-loop a
	// pod that is simply waiting on containerd. A successful TCP connect proves the
	// process is alive and serving; a hung/dead fractiond fails it and is restarted.
	livenessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			TCPSocket: &corev1.TCPSocketAction{
				Port: intstr.FromInt32(d.readinessPort()),
			},
		},
		InitialDelaySeconds: 10,
		PeriodSeconds:       20,
		FailureThreshold:    3,
	}

	container := corev1.Container{
		Name:            daemonName,
		Image:           image.FullImage(),
		ImagePullPolicy: image.PullPolicy(),
		SecurityContext: daemonmgr.PrivilegedSecurityContext(),
		Args:            d.buildArgs(),
		ReadinessProbe:  readinessProbe,
		LivenessProbe:   livenessProbe,
		Resources:       daemonmgr.DaemonResources(fractiondCPURequest, fractiondMemRequest, fractiondMemLimit),
		Env:             daemonmgr.FIPSOnlyEnv(opts.FIPSOnly),
		Ports: []corev1.ContainerPort{
			{
				Name:          "readiness",
				ContainerPort: d.readinessPort(),
				Protocol:      corev1.ProtocolTCP,
			},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: volumeNRISocket, MountPath: d.nriSocketDir()},
			{Name: volumeMPSPipe, MountPath: defaultMPSPipeDir},
		},
	}

	// Host paths so the container can reach containerd's NRI endpoint and the MPS
	// control pipe shared with the mpsd daemon.
	volumes := []corev1.Volume{
		{
			Name: volumeNRISocket,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{Path: d.nriSocketDir()},
			},
		},
		{
			Name: volumeMPSPipe,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{Path: defaultMPSPipeDir},
			},
		},
	}

	// Retroactive enforcement stops offending containers via the CRI runtime
	// socket, so mount it into the (privileged) pod only when the feature is on.
	// We mount the socket's PARENT DIRECTORY (DirectoryOrCreate), not the socket
	// file with HostPathType: Socket: a Socket mount makes the kubelet refuse to
	// start the pod whenever the socket is absent or at a non-default path (k3s
	// and RKE2 keep containerd under /run/k3s/containerd), which would take down
	// fractiond's core NRI injection too — not just enforcement. With the directory
	// mount fractiond always starts; if the socket really isn't at the dialed path
	// the CRIStopper just logs a dial error and enforcement no-ops. The binary
	// still dials the full socket path.
	if d.retroactiveEnforcementEnabled() {
		dirType := corev1.HostPathDirectoryOrCreate
		criSocketDir := filepath.Dir(d.criSocketPath())
		container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
			Name:      volumeCRISocket,
			MountPath: criSocketDir,
		})
		volumes = append(volumes, corev1.Volume{
			Name: volumeCRISocket,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: criSocketDir,
					Type: &dirType,
				},
			},
		})
	}

	return container, volumes
}

// buildMetricsdContainer returns the metricsd sidecar. It reads the shared map
// directory (read-only) and exports GPU metrics on metricsPort. PID→pod
// attribution works via the pod's host PID namespace.
//
// NVIDIA_VISIBLE_DEVICES=all and NVIDIA_DRIVER_CAPABILITIES=utility are
// required so the NVIDIA container runtime mounts libnvidia-ml.so into the
// container at create time (the "utility" capability is sufficient for
// read-only NVML; "compute" is not needed).
func (d *daemon) buildMetricsdContainer(opts daemonmgr.BuildOptions) corev1.Container {
	image := opts.DefaultImages[metricsdName]
	port := d.metricsListenPort()

	// Readiness gates pod readiness on the exporter answering; liveness restarts a
	// wedged or dead exporter. Both hit metricsd's fixed health paths on the same
	// HTTP server that serves /metrics, so a broken sidecar is no longer invisible
	// to pod readiness.
	readinessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{Path: metricsdReadyzPath, Port: intstr.FromInt32(port)},
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       10,
	}
	livenessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{Path: metricsdHealthzPath, Port: intstr.FromInt32(port)},
		},
		InitialDelaySeconds: 15,
		PeriodSeconds:       20,
		FailureThreshold:    3,
	}

	c := corev1.Container{
		Name:            metricsdName,
		Image:           image.FullImage(),
		ImagePullPolicy: image.PullPolicy(),
		SecurityContext: daemonmgr.PrivilegedSecurityContext(),
		Args:            d.buildMetricsdArgs(),
		ReadinessProbe:  readinessProbe,
		LivenessProbe:   livenessProbe,
		Resources:       daemonmgr.DaemonResources(metricsdCPURequest, metricsdMemRequest, metricsdMemLimit),
		Env: []corev1.EnvVar{
			{Name: "NVIDIA_VISIBLE_DEVICES", Value: "all"},
			{Name: "NVIDIA_DRIVER_CAPABILITIES", Value: "utility"},
		},
		Ports: []corev1.ContainerPort{
			{Name: "metrics", ContainerPort: port, Protocol: corev1.ProtocolTCP},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: volumeMapDir, MountPath: containerPodMapDir, ReadOnly: true},
		},
	}
	if d.metricsSpec != nil {
		c.VolumeMounts = append(c.VolumeMounts, d.metricsSpec.VolumeMounts...)
	}
	// Appended, not assigned: the NVIDIA vars above are what get libnvidia-ml.so
	// mounted into the container, so replacing the slice would break NVML.
	c.Env = append(c.Env, daemonmgr.FIPSOnlyEnv(opts.FIPSOnly)...)
	return c
}

func (d *daemon) buildMetricsdArgs() []string {
	if d.metricsSpec == nil {
		return nil
	}
	var args []string
	if d.metricsSpec.LogLevel != "" {
		args = append(args, "--log-level", d.metricsSpec.LogLevel)
	}
	if d.metricsSpec.Address != "" {
		args = append(args, "--metrics-address", d.metricsSpec.Address)
	}
	if d.metricsSpec.Path != "" {
		args = append(args, "--metrics-path", d.metricsSpec.Path)
	}
	if n := d.metricsSpec.MetricNames; n != nil {
		if n.GPUMemoryUsedBytes != "" {
			args = append(args, "--metric-name-gpu-memory-used-bytes", n.GPUMemoryUsedBytes)
		}
		if n.GPUSMUtilizationPercent != "" {
			args = append(args, "--metric-name-gpu-sm-utilization-percent", n.GPUSMUtilizationPercent)
		}
		if n.GPUSMUtilizationPercentNormalized != "" {
			args = append(args, "--metric-name-gpu-sm-utilization-percent-normalized", n.GPUSMUtilizationPercentNormalized)
		}
	}
	return args
}

func (d *daemon) buildArgs() []string {
	// Always passed explicitly (independent of the fractioningAgent CRD spec
	// below): it is a Helm-installation-time toggle, not a per-CR setting.
	args := []string{"--support-sm-sharing=" + strconv.FormatBool(d.supportSMSharing)}

	spec := d.fractioningSpec
	if spec == nil {
		return args
	}

	// Pod annotation key prefix used to read per-container GPU memory requests.
	if spec.AnnotationPrefix != "" {
		args = append(args, "--annotation-prefix", spec.AnnotationPrefix)
	}
	// When set, allow container creation even if GPU fractioning setup fails.
	if spec.FailOpen != nil && *spec.FailOpen {
		args = append(args, "--fail-open")
	}
	// Path to the NRI socket file on the host for registering as a containerd plugin.
	if spec.NRISocketPath != "" {
		args = append(args, "--socket-path", spec.NRISocketPath)
	}
	// Logging verbosity (e.g. "debug", "info", "warn", "error").
	if spec.LogLevel != "" {
		args = append(args, "--log-level", spec.LogLevel)
	}
	// How long to wait between MPS control retries on transient failures.
	if spec.RetryInterval != nil {
		args = append(args, "--retry-interval", spec.RetryInterval.Duration.String())
	}
	// Duration a GPU must remain stable before fractiond considers it healthy.
	if spec.StableThreshold != nil {
		args = append(args, "--stable-threshold", spec.StableThreshold.Duration.String())
	}
	// Maximum number of MPS control retries before giving up.
	if spec.MaxRetries != nil {
		args = append(args, "--max-retries", strconv.Itoa(int(*spec.MaxRetries)))
	}
	// Port for the /readyz readiness endpoint. Only passed when explicitly
	// configured; otherwise the binary's default is left to match
	// defaultReadinessPort, so older images without the flag keep working.
	if spec.ReadinessPort != nil {
		args = append(args, "--readiness-port", strconv.Itoa(int(*spec.ReadinessPort)))
	}
	args = append(args, "--retroactive-enforcement="+strconv.FormatBool(spec.RetroactiveEnforcement))
	// Override the CRI socket path used to stop containers during enforcement.
	if spec.CRISocketPath != "" {
		args = append(args, "--cri-socket", spec.CRISocketPath)
	}

	return args
}

// readinessPort returns the port the readiness probe (and container port)
// target: the CRD override when set, the binary's default otherwise.
func (d *daemon) readinessPort() int32 {
	if d.fractioningSpec != nil && d.fractioningSpec.ReadinessPort != nil {
		return *d.fractioningSpec.ReadinessPort
	}
	return defaultReadinessPort
}

// nriSocketDir returns the directory containing the NRI socket.
// The volume mount needs the directory, not the socket file itself.
func (d *daemon) nriSocketDir() string {
	if d.fractioningSpec == nil || d.fractioningSpec.NRISocketPath == "" {
		return defaultNRISocketDir
	}

	return filepath.Dir(d.fractioningSpec.NRISocketPath)
}

// retroactiveEnforcementEnabled reports whether fractiond should stop offending
// containers on reconnect, and thus whether the CRI socket must be mounted. The
// authoritative default lives in the CRD schema (retroactiveEnforcement defaults
// to true via +kubebuilder:default), so the API server sets it to true for any
// CR that includes a fractioningAgent block. A nil spec (no fractioningAgent at all)
// leaves it off, mirroring metricsEnabled.
func (d *daemon) retroactiveEnforcementEnabled() bool {
	return d.fractioningSpec != nil && d.fractioningSpec.RetroactiveEnforcement
}

// criSocketPath returns the CRI runtime socket path fractiond uses to stop
// containers, mounted into the pod at the same path it is dialed on.
func (d *daemon) criSocketPath() string {
	if d.fractioningSpec == nil || d.fractioningSpec.CRISocketPath == "" {
		return defaultCRISocketPath
	}

	return d.fractioningSpec.CRISocketPath
}
