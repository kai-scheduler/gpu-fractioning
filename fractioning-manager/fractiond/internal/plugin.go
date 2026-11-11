// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/fsstore"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/amd"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/audit"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/events"
	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/injection"
)

// ReadinessSetter receives the plugin's NRI registration state: true once the
// runtime delivers Synchronize, false when it disconnects. *readiness.State
// implements it; the plugin only needs the setter half, so it depends on this
// interface rather than the readiness package.
type ReadinessSetter interface {
	SetReady(bool)
}

const (
	// DefaultPluginName NRI plugin registration defaults.
	DefaultPluginName = "gpu-fractioning"
	DefaultPluginIdx  = "10"
)

// Config configures a Plugin. The first fields drive container mutation (the
// existing GPU-memory/MPS behaviour); the mapping fields drive the container→pod
// mapping handoff consumed by the metricsd sidecar.
type Config struct {
	// AnnotationPrefix is the annotation prefix for GPU memory config (mutation).
	AnnotationPrefix string
	// MPSPipeDirectory is the MPS pipe directory bind-mounted into GPU containers.
	MPSPipeDirectory string
	// FailOpen skips a container on parse error instead of blocking it.
	FailOpen bool

	// SupportSMSharing is the cluster's installation-time sm-sharing chicken
	// bit (Helm value -> operator -> --support-sm-sharing). When false, the
	// gpu-compute.mode: sm-sharing annotation is rejected like any other
	// invalid value, since mpsd's shared MPS server is disabled by the same
	// toggle.
	SupportSMSharing bool

	// InjectCDIDevice requests a container's assigned GPU as a CDI device in
	// addition to NVIDIA_VISIBLE_DEVICES (operator -> --inject-cdi-device, from
	// the GPU Operator's spec.cdi.nriPluginEnabled).
	InjectCDIDevice bool

	// RetroactiveEnforcement enables the audit pass on NRI (re)connect: any
	// GPU-fractioning container found running without the expected injection is
	// stopped so kubelet recreates it through a healthy CreateContainer hook.
	// Requires a non-nil stopper passed to NewPlugin; otherwise NewPlugin
	// returns an error.
	RetroactiveEnforcement bool

	// MapDir is the shared dir for the container→pod mapping handoff.
	MapDir string
	// LogPodEvents logs each recorded/removed mapping event.
	LogPodEvents bool

	// Log is the logger used by the plugin; defaults to slog.Default() when nil.
	Log *slog.Logger

	// Readiness, when non-nil, is flipped to ready on Synchronize (the runtime
	// delivered the full container state, so registration succeeded) and back to
	// not-ready on Shutdown (the runtime is disconnecting).
	Readiness ReadinessSetter

	// Vendor selects how a container's GPU memory is enforced: VendorNVIDIA (the
	// default, when empty) injects the NVIDIA_GPU_MEMORY_* env vars and the MPS
	// pipe; VendorAMD sets dmem.max on the container's cgroup (see package amd).
	// The value is matched case-insensitively. With VendorAMD, sm-sharing is
	// rejected (AMD GPUs have no MPS), and RetroactiveEnforcement re-applies the
	// limit to running containers on Synchronize instead of stopping them, so it
	// needs no stopper.
	Vendor string

	// DmemCapacityFile is the host's root-cgroup dmem.capacity file, read with
	// VendorAMD to check that each GPU's VRAM is a dmem region on this node
	// (DefaultDmemCapacityFile when empty). The root-only file is not visible
	// in the plugin's own cgroup namespace, so the host's /sys/fs/cgroup is
	// mounted read-only.
	DmemCapacityFile string
}

// GPU vendors the plugin enforces memory for.
const (
	VendorNVIDIA = "nvidia"
	VendorAMD    = "amd"
)

// DefaultDmemCapacityFile is where the AMD backend reads the host's
// dmem.capacity: the host's /sys/fs/cgroup mounted read-only at /host/sys/fs/cgroup.
const DefaultDmemCapacityFile = "/host/sys/fs/cgroup/dmem.capacity"

// NormalizeVendor returns the vendor name as the plugin compares it: trimmed and
// lower case, with empty meaning VendorNVIDIA.
func NormalizeVendor(vendor string) string {
	vendor = strings.ToLower(strings.TrimSpace(vendor))
	if vendor == "" {
		return VendorNVIDIA
	}
	return vendor
}

// Plugin implements the GPU fractioning NRI handler logic. It has two independent
// jobs on the container lifecycle:
//
//  1. Mutation: on CreateContainer it evaluates the pod's GPU-memory annotations
//     and injects the NVIDIA_GPU_MEMORY_* env vars, CUDA_MPS_PIPE_DIRECTORY, and
//     the MPS pipe bind mount. When the scheduler also recorded a GPU
//     device-assignment annotation on the pod it injects NVIDIA_VISIBLE_DEVICES
//     so the fractional container sees the GPU the scheduler picked. With
//     Vendor set to VendorAMD it instead sets the container's dmem.max for each
//     assigned GPU (see package amd).
//  2. Mapping: it records a container→pod mapping (plus assigned GPU devices and
//     requested fraction) to a shared directory via an async events processor and
//     fsstore writer. The metricsd sidecar reads that mapping to attribute GPU
//     processes to pods. The mapping path never mutates the container.
//
// The mapping work stays off the NRI hot path: handlers capture the runtime
// objects in a closure and hand it to the events processor, which does the
// api.* → store.ContainerInfo conversion on its own worker goroutine and drops
// events rather than blocking if it falls behind. Mapping failures are
// fire-and-forget, so they can never break container mutation.
//
// It satisfies stub.ConfigureInterface, stub.SynchronizeInterface,
// stub.CreateContainerInterface, stub.RemoveContainerInterface and
// stub.ShutdownInterface.
type Plugin struct {
	AnnotationPrefix string
	MPSPipeDirectory string
	FailOpen         bool
	SupportSMSharing bool
	InjectCDIDevice  bool
	Log              *slog.Logger

	vendor    string
	events    *events.Processor
	adapter   adapter
	readiness ReadinessSetter

	// sentinel runs retroactive enforcement on Synchronize; nil when disabled
	// or with VendorAMD.
	sentinel *audit.Sentinel

	// reapplyAMD re-applies dmem.max to running AMD GPU containers on
	// Synchronize (VendorAMD with RetroactiveEnforcement).
	reapplyAMD bool
	// dmemCapacity reads the node's dmem regions (VendorAMD).
	dmemCapacity func() (map[string]int64, error)
}

// NewPlugin creates a Plugin from cfg. Empty mapping defaults are filled in so a
// minimal caller still gets a working handoff directory.
//
// stopper backs retroactive enforcement: when cfg.RetroactiveEnforcement is true
// the plugin audits each NRI Synchronize snapshot and stops GPU-fractioning
// containers missing injection. Enabling the flag without a stopper is a
// misconfiguration and returns an error rather than silently doing nothing;
// when the flag is off, stopper is ignored (pass nil).
func NewPlugin(cfg Config, stopper audit.ContainerStopper) (*Plugin, error) {
	log := cfg.Log
	if log == nil {
		log = slog.Default()
	}
	mapDir := cfg.MapDir
	if mapDir == "" {
		mapDir = fsstore.DefaultMapDir
	}

	writer := fsstore.NewWriter(mapDir, log)
	logPodEvents := cfg.LogPodEvents
	proc := events.NewProcessor(writer, log, events.Options{
		LogEvents: func() bool { return logPodEvents },
	})

	vendor := NormalizeVendor(cfg.Vendor)
	var dmemCapacity func() (map[string]int64, error)
	switch vendor {
	case VendorNVIDIA:
	case VendorAMD:
		cfg.SupportSMSharing = false // no MPS on AMD GPUs: sm-sharing is rejected like an invalid mode
		capacityFile := cfg.DmemCapacityFile
		if capacityFile == "" {
			capacityFile = DefaultDmemCapacityFile
		}
		dmemCapacity = func() (map[string]int64, error) { return readDmemCapacity(capacityFile) }
	default:
		return nil, fmt.Errorf("unknown GPU vendor %q (use %q or %q)", cfg.Vendor, VendorNVIDIA, VendorAMD)
	}

	var sentinel *audit.Sentinel
	if cfg.RetroactiveEnforcement && vendor == VendorNVIDIA {
		if stopper == nil {
			return nil, fmt.Errorf("retroactive enforcement enabled but no container stopper was provided")
		}
		sentinel = audit.NewSentinel(audit.Config{
			AnnotationPrefix: cfg.AnnotationPrefix,
			MPSPipeDirectory: cfg.MPSPipeDirectory,
			SMSharingEnabled: cfg.SupportSMSharing,
			FailOpen:         cfg.FailOpen,
		}, stopper, log)
	}

	return &Plugin{
		AnnotationPrefix: cfg.AnnotationPrefix,
		MPSPipeDirectory: cfg.MPSPipeDirectory,
		FailOpen:         cfg.FailOpen,
		SupportSMSharing: cfg.SupportSMSharing,
		InjectCDIDevice:  cfg.InjectCDIDevice,
		Log:              log,
		vendor:           vendor,
		events:           proc,
		adapter:          adapter{annotationPrefix: cfg.AnnotationPrefix, log: log},
		readiness:        cfg.Readiness,
		sentinel:         sentinel,
		reapplyAMD:       vendor == VendorAMD && cfg.RetroactiveEnforcement,
		dmemCapacity:     dmemCapacity,
	}, nil
}

// readDmemCapacity reads and parses a dmem.capacity file.
func readDmemCapacity(path string) (map[string]int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read the host's dmem.capacity (mount the host's /sys/fs/cgroup read-only, or set --dmem-capacity-file): %w", err)
	}
	defer func() { _ = f.Close() }()
	return amd.ParseCapacity(f)
}

// setReady updates the shared readiness state, if one was configured.
func (p *Plugin) setReady(ready bool) {
	if p.readiness != nil {
		p.readiness.SetReady(ready)
	}
}

// Configure subscribes to every NRI event this plugin implements (returning a
// zero event mask asks the runtime for all of them). The plugin does not consume
// NRI-provided configuration.
func (p *Plugin) Configure(ctx context.Context, _, runtime, version string) (api.EventMask, error) {
	p.Log.InfoContext(ctx, "configured NRI plugin", "runtime", runtime, "runtimeVersion", version)
	return 0, nil
}

// Synchronize rebuilds the full container→pod mapping from the runtime's current
// container set on (re)connect. Pre-existing pods arrive here, not via
// CreateContainer. The conversion runs on the events worker; the handler returns
// no container updates (this plugin does not mutate on sync).
//
// Synchronize only fires after the plugin successfully registered with the
// runtime, so it doubles as the readiness signal.
//
// When retroactive enforcement is enabled it also audits the snapshot for
// GPU-fractioning containers that started without injection (while the agent was
// down) and stops them off the hot path. Detection is synchronous and cheap;
// the stopping happens on a background goroutine so it never stalls this NRI
// callback.
//
// With VendorAMD the plugin is ready only once the node has a dmem region for
// GPU VRAM, and retroactive enforcement returns container updates instead: the
// limit is a cgroup file, so a running container that has no dmem.max (it was
// created while fractiond was down) gets it set in place, without a restart.
func (p *Plugin) Synchronize(ctx context.Context, pods []*api.PodSandbox, containers []*api.Container) ([]*api.ContainerUpdate, error) {
	p.Log.InfoContext(ctx, "synchronizing container mapping with runtime",
		"pods", len(pods), "containers", len(containers))
	p.events.Synchronize(func() []store.ContainerInfo {
		infos := p.adapter.containers(pods, containers)
		p.Log.Info("rebuilt container mapping from runtime sync",
			"recordedContainers", len(infos), "totalContainers", len(containers))
		return infos
	})
	if p.vendor == VendorAMD {
		ready := p.checkAMDNode(ctx)
		p.setReady(ready)
		if !p.reapplyAMD || !ready {
			return nil, nil
		}
		return p.amdReapply(ctx, pods, containers), nil
	}
	p.setReady(true)
	if p.sentinel != nil {
		p.sentinel.Audit(ctx, pods, containers)
	}
	return nil, nil
}

// checkAMDNode reports whether this node can enforce AMD GPU memory: its
// dmem.capacity lists at least one GPU VRAM region. amdgpu registers one per GPU
// from Linux 6.15 on; on an older kernel, or with amdgpu not bound, there is none,
// and every limit the runtime tried to write would fail.
func (p *Plugin) checkAMDNode(ctx context.Context) bool {
	regions, err := p.dmemCapacity()
	if err != nil {
		p.Log.ErrorContext(ctx, "not ready: cannot check the node's dmem regions", "error", err)
		return false
	}
	vram := amd.VRAMRegions(regions)
	if len(vram) == 0 {
		p.Log.ErrorContext(ctx, "not ready: no GPU VRAM is a dmem region on this node; AMD GPU memory limits need Linux 6.15+ with amdgpu bound to the GPUs and cgroup v2",
			"dmemRegions", len(regions))
		return false
	}
	p.Log.InfoContext(ctx, "AMD GPU VRAM dmem regions found", "regions", vram)
	return true
}

// amdReapply returns an update setting dmem.max on each created, running or
// paused fractional AMD GPU container whose cgroup does not carry the limit its
// annotations ask for. Updates are marked ignore-failure, so one container the
// runtime cannot update does not fail the whole synchronization.
func (p *Plugin) amdReapply(ctx context.Context, pods []*api.PodSandbox, containers []*api.Container) []*api.ContainerUpdate {
	podByID := make(map[string]*api.PodSandbox, len(pods))
	for _, pod := range pods {
		podByID[pod.GetId()] = pod
	}
	var updates []*api.ContainerUpdate
	for _, ctr := range containers {
		switch ctr.GetState() {
		case api.ContainerState_CONTAINER_CREATED, api.ContainerState_CONTAINER_RUNNING, api.ContainerState_CONTAINER_PAUSED:
		default:
			continue
		}
		pod := podByID[ctr.GetPodSandboxId()]
		if pod == nil {
			continue
		}
		value, ok, err := p.amdMaxValue(pod, ctr)
		if err != nil {
			p.Log.WarnContext(ctx, "cannot re-apply the AMD GPU memory limit to a running container; it stays uncapped",
				"container", ctr.GetName(), "pod", pod.GetName(), "error", err)
			continue
		}
		if !ok || ctr.GetLinux().GetResources().GetUnified()[amd.UnifiedMax] == value {
			continue
		}
		u := &api.ContainerUpdate{}
		u.SetContainerId(ctr.GetId())
		u.AddLinuxUnified(amd.UnifiedMax, value)
		u.SetIgnoreFailure()
		updates = append(updates, u)
		p.Log.InfoContext(ctx, "re-applying the AMD GPU memory limit to a container created without it",
			"container", ctr.GetName(), "pod", pod.GetName(), "limit", value)
	}
	return updates
}

// CreateContainer evaluates a container's pod annotations and returns an
// adjustment if GPU memory fractioning is configured, and records the container→pod
// mapping for the metrics sidecar.
//
// If annotation parsing fails and FailOpen is true, the error is logged and nil
// is returned. If FailOpen is false (default), the error is returned and
// container creation is blocked — in which case the mapping is NOT recorded,
// since the container will not exist.
func (p *Plugin) CreateContainer(_ context.Context, pod *api.PodSandbox, ctr *api.Container) (*api.ContainerAdjustment, []*api.ContainerUpdate, error) {
	adj, err := p.buildAdjustment(pod, ctr)
	if err != nil {
		// Fail-closed: the runtime will refuse to create the container, so do
		// not record a mapping for it.
		return nil, nil, err
	}

	// The container will be created — record its mapping off the hot path. The
	// adapter drops non-fractional-GPU containers (ok=false), so this is a no-op
	// for containers the metrics sidecar does not care about.
	p.events.Upsert(func() (store.ContainerInfo, bool) {
		return p.adapter.container(pod, ctr)
	})

	return adj, nil, nil
}

// buildAdjustment contains the GPU-memory / MPS mutation logic. It returns a nil
// adjustment when the container has no GPU memory annotations, and an error only
// when annotation parsing fails while FailOpen is false. For a GPU-fractioning
// container it additionally injects NVIDIA_VISIBLE_DEVICES from the container's
// device-assignment annotation when present, requests that same GPU as a CDI
// device if InjectCDIDevice is set, and routes the MPS pipe mount to either the
// default or the shared MPS server based on the container's compute-mode
// annotation (see injection.MPSPipeMount).
func (p *Plugin) buildAdjustment(pod *api.PodSandbox, ctr *api.Container) (*api.ContainerAdjustment, error) {
	gpuMemoryCfg, err := annotations.ParseGPUMemoryAnnotations(pod.Annotations, ctr.Name, p.AnnotationPrefix)
	if err != nil {
		p.Log.Warn("failed to parse GPU memory annotations",
			"container", ctr.Name,
			"pod", pod.Name,
			"error", err,
		)
		if !p.FailOpen {
			return nil, fmt.Errorf("container %q in pod %q: %w", ctr.Name, pod.Name, err)
		}
		return nil, nil
	}

	if gpuMemoryCfg.IsEmpty() {
		p.Log.Debug("no GPU memory annotations; gpu-compute.mode has no effect without them", "container", ctr.Name, "pod", pod.Name)
		return nil, nil
	}

	// A container that specified only a request or only a limit gets the missing
	// value defaulted from the other (request == limit), so the memory limit is
	// always enforced and the request is always populated for metrics.
	gpuMemoryCfg = gpuMemoryCfg.ApplyDefaults()

	computeMode, err := annotations.ParseComputeMode(pod.Annotations, ctr.Name, p.AnnotationPrefix, p.SupportSMSharing)
	if err != nil {
		p.Log.Warn("failed to parse GPU compute mode annotation",
			"container", ctr.Name,
			"pod", pod.Name,
			"error", err,
		)
		if !p.FailOpen {
			return nil, fmt.Errorf("container %q in pod %q: %w", ctr.Name, pod.Name, err)
		}
		p.Log.Warn("defaulting to time-slicing compute mode (fail-open)",
			"container", ctr.Name,
			"pod", pod.Name,
		)
		computeMode = annotations.ComputeModeTimeSlicing
	}
	if p.vendor == VendorAMD {
		return p.amdAdjustment(pod, ctr, gpuMemoryCfg)
	}
	pipeSource, pipeDestination := injection.MPSPipeMount(p.MPSPipeDirectory, computeMode)

	adj := &api.ContainerAdjustment{}

	if gpuMemoryCfg.Request != "" {
		adj.AddEnv(injection.EnvGPUMemoryRequest, gpuMemoryCfg.Request)
	}
	if gpuMemoryCfg.Limit != "" {
		adj.AddEnv(injection.EnvGPUMemoryLimit, gpuMemoryCfg.Limit)
	}
	adj.AddEnv(injection.EnvMPSPipeDirectory, pipeDestination)

	adj.AddMount(&api.Mount{
		Source:      pipeSource,
		Destination: pipeDestination,
		Type:        "bind",
		Options:     []string{"bind", "rw"},
	})

	// Promote the scheduler's GPU device assignment to NVIDIA_VISIBLE_DEVICES.
	// A fractional container does not request the nvidia.com/gpu resource, so the
	// NVIDIA device plugin never sets this env var; without it the container would
	// see all GPUs or none. When the annotation is absent we leave the env var
	// untouched (the device plugin or the image may already set it).
	//
	// The container may already carry NVIDIA_VISIBLE_DEVICES (e.g. set to "void"
	// by an admission plugin precisely because the pod does not request
	// nvidia.com/gpu). Our assignment must win: remove any existing value first so
	// NRI applies the override instead of rejecting it as a conflict, and the
	// container ends up with a single, correct value rather than a duplicate.
	visibleDevices := annotations.ParseVisibleDevices(pod.Annotations, ctr.Name, p.AnnotationPrefix)
	var cdiDevices []string
	if visibleDevices != "" {
		if containerHasEnv(ctr, injection.EnvVisibleDevices) {
			adj.RemoveEnv(injection.EnvVisibleDevices)
		}
		adj.AddEnv(injection.EnvVisibleDevices, visibleDevices)

		if p.InjectCDIDevice {
			cdiDevices = injection.CDIDeviceNames(visibleDevices)
			for _, name := range cdiDevices {
				adj.AddCDIDevice(&api.CDIDevice{Name: name})
			}
		}
	}

	p.Log.Info("adjusting container with GPU memory config",
		"container", ctr.Name,
		"pod", pod.Name,
		"request", gpuMemoryCfg.Request,
		"limit", gpuMemoryCfg.Limit,
		"visibleDevices", visibleDevices,
		"cdiDevices", cdiDevices,
		"computeMode", computeMode,
	)

	return adj, nil
}

// amdAdjustment enforces an AMD GPU container's memory with the dmem cgroup
// controller: the runtime writes dmem.max (the limit) for each assigned GPU's
// VRAM region on the container's cgroup, and the kernel refuses allocations past
// it. The GPU itself reaches the container through the device plugin or the
// scheduler's own device injection; no env var or mount is added. Errors,
// including a GPU whose VRAM is not a dmem region on this node, follow FailOpen
// like annotation errors.
func (p *Plugin) amdAdjustment(pod *api.PodSandbox, ctr *api.Container, cfg annotations.GPUMemoryConfig) (*api.ContainerAdjustment, error) {
	devices, err := amd.Devices(annotations.ParseVisibleDevices(pod.Annotations, ctr.Name, p.AnnotationPrefix))
	var maxValue string
	if err == nil {
		maxValue, err = amd.MaxValue(cfg.Limit, devices)
	}
	if err == nil {
		err = p.checkAMDRegions(devices)
	}
	if err != nil {
		p.Log.Warn("cannot limit AMD GPU memory", "container", ctr.Name, "pod", pod.Name, "error", err)
		if !p.FailOpen {
			return nil, fmt.Errorf("container %q in pod %q: %w", ctr.Name, pod.Name, err)
		}
		p.Log.Warn("starting the container without an AMD GPU memory limit (fail-open)", "container", ctr.Name, "pod", pod.Name)
		return nil, nil
	}
	adj := &api.ContainerAdjustment{}
	adj.AddLinuxUnified(amd.UnifiedMax, maxValue)
	p.Log.Info("adjusting container with AMD GPU memory limits",
		"container", ctr.Name,
		"pod", pod.Name,
		"requestMiB", cfg.Request,
		"limitMiB", cfg.Limit,
		"devices", devices,
	)
	return adj, nil
}

// checkAMDRegions returns an error unless each device's VRAM is a dmem region on
// this node. Without the region the runtime's write of dmem.max fails inside
// the runtime, after this plugin has returned, as an opaque container-creation
// error that --fail-open cannot catch; checking here makes the failure clear and
// lets fail-open apply.
func (p *Plugin) checkAMDRegions(devices []string) error {
	regions, err := p.dmemCapacity()
	if err != nil {
		return err
	}
	for _, d := range devices {
		if _, ok := regions[amd.Region(d)]; !ok {
			return fmt.Errorf("GPU %s has no dmem VRAM region on this node (dmem.capacity lists %v); AMD GPU memory limits need Linux 6.15+ with amdgpu bound to the GPU",
				d, amd.VRAMRegions(regions))
		}
	}
	return nil
}

// amdMaxValue returns the dmem.max value a container's annotations ask for; ok
// is false for a container without GPU-memory annotations.
func (p *Plugin) amdMaxValue(pod *api.PodSandbox, ctr *api.Container) (value string, ok bool, err error) {
	cfg, err := annotations.ParseGPUMemoryAnnotations(pod.GetAnnotations(), ctr.GetName(), p.AnnotationPrefix)
	if err != nil {
		return "", false, err
	}
	if cfg.IsEmpty() {
		return "", false, nil
	}
	cfg = cfg.ApplyDefaults()
	devices, err := amd.Devices(annotations.ParseVisibleDevices(pod.GetAnnotations(), ctr.GetName(), p.AnnotationPrefix))
	if err == nil {
		value, err = amd.MaxValue(cfg.Limit, devices)
	}
	if err == nil {
		err = p.checkAMDRegions(devices)
	}
	if err != nil {
		return "", false, err
	}
	return value, true, nil
}

// containerHasEnv reports whether the container's spec already defines the given
// environment variable (as "KEY=VALUE" or a bare "KEY"), so the caller can
// replace it rather than append a duplicate.
func containerHasEnv(ctr *api.Container, key string) bool {
	prefix := key + "="
	for _, kv := range ctr.GetEnv() {
		if kv == key || strings.HasPrefix(kv, prefix) {
			return true
		}
	}
	return false
}

// RemoveContainer drops the container's mapping when the runtime removes it.
func (p *Plugin) RemoveContainer(_ context.Context, _ *api.PodSandbox, ctr *api.Container) error {
	p.events.Delete(ctr.GetId())
	return nil
}

// Shutdown flushes any queued mapping events and waits for in-flight
// remediation when the runtime disconnects, and marks the plugin not-ready until
// the next successful Synchronize.
func (p *Plugin) Shutdown(_ context.Context) {
	p.setReady(false)
	p.events.Flush()
	if p.sentinel != nil {
		p.sentinel.Wait()
	}
}

// Flush blocks until all queued mapping events have been applied and any
// in-flight remediation has finished. Used by tests and graceful shutdown.
func (p *Plugin) Flush() {
	p.events.Flush()
	if p.sentinel != nil {
		p.sentinel.Wait()
	}
}
