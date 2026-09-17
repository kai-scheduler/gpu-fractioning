// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// GpuFractioningConfigSpec defines the desired state of GpuFractioningConfig.
type GpuFractioningConfigSpec struct {
	// nodeSelector determines which nodes the managed DaemonSets (fractiond, mpsd)
	// target. All daemons share the same selector.
	// Example: {"nvidia.com/gpu.present": "true"} (set by the Helm chart's default CR).
	//
	// The field is immutable (enforced by CEL, no webhook needed): changing it on
	// a live CR would strand stale node conditions on nodes removed from the
	// target set, so retargeting requires deleting and recreating the
	// GpuFractioningConfig (which cleans those conditions up).
	//
	// +required
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="nodeSelector is immutable; delete and recreate the GpuFractioningConfig to change it"
	NodeSelector map[string]string `json:"nodeSelector"`

	// runtimeClassName sets the RuntimeClass for daemon pods that need NVIDIA
	// GPU/NVML access. Defaults to "nvidia". Set to "" to use the node's default
	// runtime class (only safe when the default runtime already provides NVIDIA
	// GPU/NVML access).
	// +optional
	RuntimeClassName *string `json:"runtimeClassName,omitempty"`

	// fractioningAgent configures the NRI-based fractioning agent (fractiond).
	// +optional
	FractioningAgent *FractioningAgentSpec `json:"fractioningAgent,omitempty"`

	// metricsAgent configures the GPU metrics exporter (metricsd), which runs as
	// a sidecar container in the same DaemonSet as fractiond and reads the
	// container→pod mapping fractiond writes to a shared volume.
	// +optional
	MetricsAgent *MetricsAgentSpec `json:"metricsAgent,omitempty"`

	// mpsDaemon configures the MPS daemon supervisor (mpsd).
	// +optional
	MpsDaemon *MpsDaemonSpec `json:"mpsDaemon,omitempty"`
}

// FractioningAgentSpec configures the fractiond DaemonSet managed by the controller.
type FractioningAgentSpec struct {
	// AnnotationPrefix is the leading prefix for the scheduler's per-container
	// GPU-fractioning annotations, e.g. "<prefix><container>.gpu-memory.request" and
	// "<prefix><container>.gpus.devices".
	// Default: "nvidia.com/container."
	// +optional
	AnnotationPrefix string `json:"annotationPrefix,omitempty"`

	// failOpen controls behavior when annotation parsing fails.
	// If true, the container is created without GPU memory limits.
	// Default: false
	// +optional
	FailOpen *bool `json:"failOpen,omitempty"`

	// nriSocketPath is the path to the NRI Unix socket used to communicate
	// with the container runtime (containerd/CRI-O). Default: /var/run/nri/nri.sock.
	// +optional
	NRISocketPath string `json:"nriSocketPath,omitempty"`

	// retroactiveEnforcement controls whether fractiond, on (re)connect to NRI,
	// stops GPU-fractioning containers found running without passing through the NRI
	// plugin. Those containers lack the proper GPU-fractioning setup and may harm
	// other containers on the GPU. Default: true.
	// +kubebuilder:default=true
	// +optional
	RetroactiveEnforcement bool `json:"retroactiveEnforcement"`

	// criSocketPath is the path to the CRI runtime socket fractiond uses to stop
	// containers during retroactive enforcement. It is mounted into the pod only
	// when retroactiveEnforcement is enabled. Default: /run/containerd/containerd.sock.
	// +optional
	CRISocketPath string `json:"criSocketPath,omitempty"`

	// logLevel controls the logging verbosity of the fractioning agent.
	// One of debug, info, warn, error. Default: info.
	// +optional
	LogLevel string `json:"logLevel,omitempty"`

	// retryInterval is the initial wait duration before retrying a failed NRI connection.
	// Backoff doubles on each failure, capped at 60s. Default: 5s.
	// +optional
	RetryInterval *metav1.Duration `json:"retryInterval,omitempty"`

	// stableThreshold is how long an NRI connection must stay up to be considered stable.
	// Once stable, the retry budget and backoff reset. Default: 5m.
	// +optional
	StableThreshold *metav1.Duration `json:"stableThreshold,omitempty"`

	// maxRetries is the maximum number of NRI connection retries before giving up.
	// 0 means unlimited. Default: 0.
	// +optional
	MaxRetries *int32 `json:"maxRetries,omitempty"`

	// readinessPort is the port fractiond serves its /readyz endpoint on. When set,
	// the operator passes --readiness-port to the binary and points the readiness
	// probe at it. When unset, no argument is passed and the probe targets the
	// binary's built-in default (8093), keeping older fractiond images compatible.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +optional
	ReadinessPort *int32 `json:"readinessPort,omitempty"`

	// mpsDrainSocketPath is the host path of the unix socket mpsd serves its
	// MPS client-drain endpoint on. fractiond calls it from its StopContainer
	// hook to drain a container's MPS clients before the runtime kills them,
	// which is what keeps a killed-mid-kernel client from wedging the MPS
	// server for every other tenant of the GPU. Must match mpsDaemon.drainSocketPath.
	// Default: /var/run/gpu-fractioning/drain/mpsd.sock. Set to "" to disable
	// the drain call.
	// +optional
	MPSDrainSocketPath *string `json:"mpsDrainSocketPath,omitempty"`

	// mpsDrainTimeout bounds how long StopContainer waits for mpsd to drain the
	// container's MPS clients. The hook always lets the stop proceed when it
	// expires — a slow drain delays a container stop, it never blocks it.
	// Default: 15s.
	// +optional
	MPSDrainTimeout *metav1.Duration `json:"mpsDrainTimeout,omitempty"`

	// resources overrides the compute resources of the fractiond container.
	// Merged over the built-in defaults, so setting only limits.memory keeps
	// the default requests.
	//
	// claims is rejected rather than ignored: the type carries it, but a claim
	// entry has to name an entry in the pod's spec.resourceClaims, which the
	// operator builds and never populates. Accepting it would silently drop it.
	// +optional
	// +kubebuilder:validation:XValidation:rule="!has(self.claims) || size(self.claims) == 0",message="resources.claims is not supported: the operator builds the daemon pod spec and never populates spec.resourceClaims, so a claim entry could not resolve"
	Resources *corev1.ResourceRequirements `json:"resources,omitempty"`
}

// MetricsAgentSpec configures the metricsd sidecar in the fractiond DaemonSet.
type MetricsAgentSpec struct {
	// enabled controls whether the metricsd sidecar is added to the DaemonSet.
	// When false, no metrics container is deployed.
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// logLevel controls the logging verbosity of the metrics agent.
	// One of debug, info, warn, error. Default: info.
	// +optional
	LogLevel string `json:"logLevel,omitempty"`

	// address is the Prometheus exporter listen address. Default: :2112.
	// +optional
	Address string `json:"address,omitempty"`

	// path is the HTTP path on which metrics are served. Default: /metrics.
	// +optional
	Path string `json:"path,omitempty"`

	// volumes are additional volumes to add to the fractiond DaemonSet pod. Use this
	// to inject environment-specific config (e.g. an alternative NVML backend in
	// test clusters) without encoding test knowledge in the operator itself.
	// +optional
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// volumeMounts are additional volume mounts to add to the metricsd container.
	// Each entry must reference a volume name declared in Volumes.
	// +optional
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`

	// metricNames overrides the exported Prometheus metric names. Empty fields
	// keep the built-in gpu_fractioning_* defaults. Set these to make metricsd emit
	// the names a downstream consumer expects. The metric labels
	// (namespace, pod, pod_uuid, gpu_uuid, gpu) are fixed.
	// +optional
	MetricNames *MetricNamesSpec `json:"metricNames,omitempty"`

	// resources overrides the compute resources of the metricsd container.
	// Merged over the built-in defaults. claims is rejected, see
	// FractioningAgentSpec.Resources.
	// +optional
	// +kubebuilder:validation:XValidation:rule="!has(self.claims) || size(self.claims) == 0",message="resources.claims is not supported: the operator builds the daemon pod spec and never populates spec.resourceClaims, so a claim entry could not resolve"
	Resources *corev1.ResourceRequirements `json:"resources,omitempty"`
}

// MetricNamesSpec overrides the Prometheus metric names metricsd exports.
type MetricNamesSpec struct {
	// gpuMemoryUsedBytes overrides the per-pod GPU memory-used-bytes metric name.
	// +optional
	GPUMemoryUsedBytes string `json:"gpuMemoryUsedBytes,omitempty"`

	// gpuSmUtilizationPercent overrides the per-pod GPU SM-utilization metric name.
	// +optional
	GPUSMUtilizationPercent string `json:"gpuSmUtilizationPercent,omitempty"`

	// gpuSmUtilizationPercentNormalized overrides the per-pod normalized GPU
	// SM-utilization metric name.
	// +optional
	GPUSMUtilizationPercentNormalized string `json:"gpuSmUtilizationPercentNormalized,omitempty"`
}

// MpsDaemonSpec configures the mpsd DaemonSet managed by the controller.
// mpsd supervises nvidia-cuda-mps-control, providing GPU multi-process
// service to containers sharing a GPU.
type MpsDaemonSpec struct {
	// logLevel controls the logging verbosity of the MPS daemon supervisor.
	// One of debug, info, warn, error. Default: info.
	// +optional
	LogLevel string `json:"logLevel,omitempty"`

	// backoff is the initial delay before restarting the MPS daemon after a crash.
	// Doubles on each failure, capped at 60s. Default: 5s.
	// +optional
	Backoff *metav1.Duration `json:"backoff,omitempty"`

	// maxRetries is the maximum number of MPS daemon restart attempts before giving up.
	// 0 means unlimited. Default: 0.
	// +optional
	MaxRetries *int32 `json:"maxRetries,omitempty"`

	// stableThreshold is how long the MPS daemon must run to be considered stable.
	// Once stable, the retry budget and backoff reset. Default: 5m.
	// +optional
	StableThreshold *metav1.Duration `json:"stableThreshold,omitempty"`

	// gracefulStopDelay is how long to wait after sending "quit" to the MPS daemon
	// before force-killing it. Default: 60s.
	// +optional
	GracefulStopDelay *metav1.Duration `json:"gracefulStopDelay,omitempty"`

	// drainSocketPath is the host path of the unix socket mpsd serves its MPS
	// client-drain endpoint on, and the path fractiond calls. Must match
	// fractioningAgent.mpsDrainSocketPath. Default:
	// /var/run/gpu-fractioning/drain/mpsd.sock. Set to "" to disable the
	// endpoint (and with it the drain-before-kill protection).
	// +optional
	DrainSocketPath *string `json:"drainSocketPath,omitempty"`

	// recycleWhenIdle restarts the MPS control daemon once the last MPS client
	// on the node disconnects through a drain. An MPS server that lost a client
	// mid-kernel keeps an unconsumed fault that hangs every future client, and
	// the only recovery is an MPS restart; bouncing it while no client is
	// attached makes that recovery free. Default: true.
	// +optional
	RecycleWhenIdle *bool `json:"recycleWhenIdle,omitempty"`

	// clientDrainTimeout bounds how long mpsd waits for a single MPS client to
	// finish draining after it is terminated. Exceeding it is treated as a
	// wedged server and escalates to an MPS restart. Default: 30s.
	// +optional
	ClientDrainTimeout *metav1.Duration `json:"clientDrainTimeout,omitempty"`

	// resources overrides the compute resources of the mpsd container. The
	// built-in default is sized for a node with up to 8 GPUs: the MPS control
	// daemon holds a CUDA server context per GPU, each costing roughly 50 MiB
	// of host memory. Raise limits.memory on nodes with more GPUs than that.
	// Merged over the built-in defaults. claims is rejected, see
	// FractioningAgentSpec.Resources.
	// +optional
	// +kubebuilder:validation:XValidation:rule="!has(self.claims) || size(self.claims) == 0",message="resources.claims is not supported: the operator builds the daemon pod spec and never populates spec.resourceClaims, so a claim entry could not resolve"
	Resources *corev1.ResourceRequirements `json:"resources,omitempty"`
}

// GpuFractioningConfigStatus defines the observed state of GpuFractioningConfig.
type GpuFractioningConfigStatus struct {
	// observedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// conditions represent the latest available observations of the config's state.
	// Known condition types: FractiondReady, MpsdReady, Ready, DriverUpgradeInProgress.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:validation:XValidation:rule="self.metadata.name == 'default'",message="GpuFractioningConfig must be named 'default'"

// GpuFractioningConfig is the Schema for the gpufractioningconfigs API
type GpuFractioningConfig struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec defines the desired state of GpuFractioningConfig
	// +required
	Spec GpuFractioningConfigSpec `json:"spec"`

	// status defines the observed state of GpuFractioningConfig
	// +optional
	Status GpuFractioningConfigStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// GpuFractioningConfigList contains a list of GpuFractioningConfig
type GpuFractioningConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []GpuFractioningConfig `json:"items"`
}
