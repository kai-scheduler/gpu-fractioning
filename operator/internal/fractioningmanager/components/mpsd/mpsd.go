// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsd

import (
	"strconv"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/common/daemonmgr"
)

const (
	daemonName = "mpsd"

	defaultMPSPipeDir = "/run/nvidia-mps"
	defaultMPSLogDir  = "/var/log/nvidia-mps"

	volumeMPSPipe = "mps-pipe"
	volumeMPSLog  = "mps-log"

	envNodeName       = "NODE_NAME"
	envVisibleDevices = "NVIDIA_VISIBLE_DEVICES"
	envCapabilities   = "NVIDIA_DRIVER_CAPABILITIES"

	// Resource requests/limits for the mpsd container. See daemonmgr.DaemonResources
	// for the requests-plus-memory-limit rationale.
	mpsdCPURequest = "50m"
	mpsdMemRequest = "64Mi"
	mpsdMemLimit   = "256Mi"

	// defaultGracefulStopDelay mirrors mpsd's supervisor.DefaultGracefulStopDelay
	// (60s). It lives in a separate module, so it is duplicated here — keep in
	// sync. It is the effective --graceful-stop-delay when the CR leaves it unset.
	defaultGracefulStopDelay = 60 * time.Second
	// terminationGraceBuffer is added to the graceful stop delay so kubelet gives
	// mpsd time to finish quitting MPS before it SIGKILLs the pod on eviction.
	terminationGraceBuffer = 15 * time.Second
)

type daemon struct {
	spec             *v1alpha1.MpsDaemonSpec
	auditLog         bool // set  level of the memacct audit log
	supportSMSharing bool // Helm-injected sm-sharing chicken bit
}

// NewMpsdDaemon returns a ManagedDaemon for the MPS daemon supervisor.
// auditLog is the Helm-injected MPS memacct audit-log default. supportSMSharing
// is the Helm-injected sm-sharing chicken bit: when false, mpsd renders its MPS
// config with the shared server (context-share) disabled, exactly as it did
// before the sm-sharing feature existed.
func NewMpsdDaemon(spec *v1alpha1.MpsDaemonSpec, auditLog, supportSMSharing bool) daemonmgr.ManagedDaemon {
	return &daemon{spec: spec, auditLog: auditLog, supportSMSharing: supportSMSharing}
}

func (d *daemon) Name() string { return daemonName }

// BuildDaemonSet constructs the desired DaemonSet for mpsd.
// The pod runs privileged with the configured runtime class so it can label the
// node with the local NVIDIA driver major version via NVML, then start the
// nvidia-cuda-mps-control binary.
// Two host paths are mounted:
//   - The MPS pipe directory, shared with fractiond and GPU workload containers.
//   - The MPS log directory, for nvidia-cuda-mps-control daemon logs.
func (d *daemon) BuildDaemonSet(opts daemonmgr.BuildOptions) *appsv1.DaemonSet {
	result := daemonmgr.BaseDaemonSet(daemonName, opts.Namespace)

	result.Spec.Template.Spec.NodeSelector = opts.NodeSelector
	result.Spec.Template.Spec.ServiceAccountName = opts.ServiceAccountName
	result.Spec.Template.Spec.RuntimeClassName = opts.RuntimeClassName
	// HostPID is required for memacct to enforce GPU memory limits. The control
	// daemon identifies each client by the PID in the connecting socket's peer
	// credentials, and the kernel only translates a peer PID for a namespace it
	// can see — its own or a descendant. In its own pod namespace the daemon
	// therefore reads every workload container's PID as 0, memacct registration
	// fails ("failed to register client pid 0"), and the limit is silently not
	// applied. The host PID namespace is an ancestor of every container's, so
	// PIDs (and their cgroups) resolve. Workloads need no change.
	result.Spec.Template.Spec.HostPID = true
	// Give mpsd long enough to graceful-quit MPS before kubelet SIGKILLs it on
	// eviction (e.g. a driver-upgrade drain). Without this the pod inherits the
	// 30s default, shorter than the graceful stop delay, defeating the graceful
	// shutdown the driver-upgrade nodeAffinity relies on.
	result.Spec.Template.Spec.TerminationGracePeriodSeconds = ptr.To(d.terminationGraceSeconds())

	container, volumes := d.buildContainer(opts.DefaultImages[daemonName])
	container.Env = append(container.Env,
		corev1.EnvVar{
			Name: envNodeName,
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "spec.nodeName"},
			},
		},
		corev1.EnvVar{Name: envVisibleDevices, Value: "all"},
		corev1.EnvVar{Name: envCapabilities, Value: "compute,utility"},
		corev1.EnvVar{Name: "MPS_MEMACCT_AUDIT_LOG", Value: strconv.FormatBool(d.auditLog)},
		corev1.EnvVar{Name: "MPS_SUPPORT_SM_SHARING", Value: strconv.FormatBool(d.supportSMSharing)},
	)
	container.Env = append(container.Env, daemonmgr.FIPSOnlyEnv(opts.FIPSOnly)...)
	result.Spec.Template.Spec.Containers = append(result.Spec.Template.Spec.Containers, container)
	result.Spec.Template.Spec.Volumes = append(result.Spec.Template.Spec.Volumes, volumes...)

	return result
}

// terminationGraceSeconds is the effective --graceful-stop-delay (the CR value,
// else mpsd's 60s default) plus a teardown buffer, so the pod's termination
// grace always covers mpsd's graceful MPS quit window.
func (d *daemon) terminationGraceSeconds() int64 {
	grace := defaultGracefulStopDelay
	if d.spec != nil && d.spec.GracefulStopDelay != nil {
		grace = d.spec.GracefulStopDelay.Duration
	}
	return int64((grace + terminationGraceBuffer).Seconds())
}

func (d *daemon) buildContainer(image daemonmgr.ImageSpec) (corev1.Container, []corev1.Volume) {
	hostPathDirOrCreate := corev1.HostPathDirectoryOrCreate

	// The MPS control socket at /run/nvidia-mps/control is created by
	// nvidia-cuda-mps-control once it finishes initialising. Checking for
	// its existence is the lightest and most reliable readiness signal:
	// no GPU driver calls, no extra binaries, and it directly proves the
	// daemon is accepting client connections.
	mpsControlSocketProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{"test", "-S", defaultMPSPipeDir + "/control"},
			},
		},
		InitialDelaySeconds: 5,
		PeriodSeconds:       10,
	}

	// Liveness reuses the control-socket check but with deliberately generous
	// thresholds: mpsd's supervisor already restarts nvidia-cuda-mps-control on
	// its own, so we only escalate to a pod restart when the socket stays absent
	// for roughly 3 minutes (60s initial + 6 × 30s) — long enough for the
	// supervisor to self-heal, but a backstop when MPS is genuinely wedged.
	mpsLivenessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{"test", "-S", defaultMPSPipeDir + "/control"},
			},
		},
		InitialDelaySeconds: 60,
		PeriodSeconds:       30,
		FailureThreshold:    6,
	}

	container := corev1.Container{
		Name:            daemonName,
		Image:           image.FullImage(),
		ImagePullPolicy: image.PullPolicy(),
		SecurityContext: daemonmgr.PrivilegedSecurityContext(),
		Args:            d.buildArgs(),
		ReadinessProbe:  mpsControlSocketProbe,
		LivenessProbe:   mpsLivenessProbe,
		Resources:       daemonmgr.DaemonResources(mpsdCPURequest, mpsdMemRequest, mpsdMemLimit),
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      volumeMPSPipe,
				MountPath: defaultMPSPipeDir,
			},
			{
				Name:      volumeMPSLog,
				MountPath: defaultMPSLogDir,
			},
		},
	}

	volumes := []corev1.Volume{
		{
			Name: volumeMPSPipe,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: defaultMPSPipeDir,
					Type: &hostPathDirOrCreate,
				},
			},
		},
		{
			Name: volumeMPSLog,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{
					Path: defaultMPSLogDir,
					Type: &hostPathDirOrCreate,
				},
			},
		},
	}

	return container, volumes
}

func (d *daemon) buildArgs() []string {
	spec := d.spec
	if spec == nil {
		return nil
	}

	var args []string

	if spec.LogLevel != "" {
		args = append(args, "--log-level", spec.LogLevel)
	}
	if spec.Backoff != nil {
		args = append(args, "--backoff", spec.Backoff.Duration.String())
	}
	if spec.MaxRetries != nil {
		args = append(args, "--max-retries", strconv.Itoa(int(*spec.MaxRetries)))
	}
	if spec.StableThreshold != nil {
		args = append(args, "--stable-threshold", spec.StableThreshold.Duration.String())
	}
	if spec.GracefulStopDelay != nil {
		args = append(args, "--graceful-stop-delay", spec.GracefulStopDelay.Duration.String())
	}

	return args
}
