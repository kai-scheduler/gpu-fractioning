// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/containerd/nri/pkg/api"
	"github.com/containerd/nri/pkg/stub"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/mapping/fsstore"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/mapping/store"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/mpsdrain"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/annotations"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/audit"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/events"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/fractiond/internal/injection"
)

// ReadinessSetter receives the plugin's NRI registration state: true once the
// runtime delivers Synchronize, false when it disconnects. *readiness.State
// implements it; the plugin only needs the setter half, so it depends on this
// interface rather than the readiness package.
type ReadinessSetter interface {
	SetReady(bool)
}

// Drainer drains a container's MPS clients before the runtime stops it.
// *mpsdrain.Client is the production implementation, talking to mpsd over its
// unix socket; the plugin depends on the interface so the StopContainer hook is
// testable without a socket.
type Drainer interface {
	Drain(ctx context.Context, req mpsdrain.Request) (mpsdrain.Response, error)
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

	// Drainer, when non-nil, is called from StopContainer to drain a
	// GPU-fractioning container's MPS clients before the runtime stops them.
	// Nil disables the drain.
	Drainer Drainer

	// DrainTimeout is the drain call timeout the Drainer was configured with
	// (fractiond's --mps-drain-timeout). StopContainer bounds its call by the
	// smaller of this and half the NRI request timeout, so the configured value
	// can only ever shorten the call, never push it past the runtime's own
	// deadline. Zero means "no configured bound", leaving only the NRI-derived
	// one.
	DrainTimeout time.Duration

	// RequestTimeout reports the NRI request timeout currently negotiated with
	// the runtime — stub.Stub.RequestTimeout. It is a function, not a value,
	// because the real number is only known after the Configure handshake and
	// changes across reconnects. Nil (or a non-positive result) falls back to
	// stub.DefaultRequestTimeout; it must never be treated as zero, which would
	// give every drain an already-expired deadline.
	RequestTimeout func() time.Duration
}

// Plugin implements the GPU fractioning NRI handler logic. It has two independent
// jobs on the container lifecycle:
//
//  1. Mutation: on CreateContainer it evaluates the pod's GPU-memory annotations
//     and injects the NVIDIA_GPU_MEMORY_* env vars, CUDA_MPS_PIPE_DIRECTORY, and
//     the MPS pipe bind mount. When the scheduler also recorded a GPU
//     device-assignment annotation on the pod it injects NVIDIA_VISIBLE_DEVICES
//     so the fractional container sees the GPU the scheduler picked.
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
// stub.CreateContainerInterface, stub.StopContainerInterface,
// stub.RemoveContainerInterface and stub.ShutdownInterface.
type Plugin struct {
	AnnotationPrefix string
	MPSPipeDirectory string
	FailOpen         bool
	SupportSMSharing bool
	Log              *slog.Logger

	events    *events.Processor
	adapter   adapter
	readiness ReadinessSetter
	drainer   Drainer

	// drainTimeout is Config.DrainTimeout; see drainBudget.
	drainTimeout time.Duration
	// requestTimeout holds Config.RequestTimeout (nil when unset). It is
	// atomic because SetRequestTimeoutSource is called from the connect loop
	// on every (re)connect while NRI callbacks read it.
	requestTimeout atomic.Pointer[requestTimeoutFunc]

	// sentinel runs retroactive enforcement on Synchronize; nil when disabled.
	sentinel *audit.Sentinel
}

// requestTimeoutFunc is a named type so the func can live in an
// atomic.Pointer (which needs a concrete element type).
type requestTimeoutFunc func() time.Duration

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

	var sentinel *audit.Sentinel
	if cfg.RetroactiveEnforcement {
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

	p := &Plugin{
		AnnotationPrefix: cfg.AnnotationPrefix,
		MPSPipeDirectory: cfg.MPSPipeDirectory,
		FailOpen:         cfg.FailOpen,
		SupportSMSharing: cfg.SupportSMSharing,
		Log:              log,
		events:           proc,
		adapter:          adapter{annotationPrefix: cfg.AnnotationPrefix, log: log},
		readiness:        cfg.Readiness,
		drainer:          cfg.Drainer,
		drainTimeout:     cfg.DrainTimeout,
		sentinel:         sentinel,
	}
	p.SetRequestTimeoutSource(cfg.RequestTimeout)

	return p, nil
}

// SetRequestTimeoutSource points the plugin at the live NRI request timeout,
// normally stub.Stub.RequestTimeout. The connect loop calls it after building
// each stub so the plugin picks up the value the runtime negotiated on that
// connection; passing nil reverts to stub.DefaultRequestTimeout.
func (p *Plugin) SetRequestTimeoutSource(fn func() time.Duration) {
	if fn == nil {
		p.requestTimeout.Store(nil)
		return
	}
	f := requestTimeoutFunc(fn)
	p.requestTimeout.Store(&f)
}

// nriRequestTimeout is the deadline the runtime puts on every call into this
// plugin (containerd's plugin_request_timeout, default 2s, renegotiated on each
// Configure). Overrunning it is not a slow hook: the runtime treats
// context.DeadlineExceeded as fatal, closes the connection and drops the
// plugin, so fractiond stops injecting limits into new containers until it
// reconnects. Every blocking call made from an NRI callback has to fit inside
// it.
//
// An unset source, or one reporting a non-positive value, falls back to the
// library default rather than to zero — a zero here would expire every derived
// deadline before the call even started.
func (p *Plugin) nriRequestTimeout() time.Duration {
	if fn := p.requestTimeout.Load(); fn != nil {
		if timeout := (*fn)(); timeout > 0 {
			return timeout
		}
	}
	return stub.DefaultRequestTimeout
}

// drainBudget is how long StopContainer may wait for mpsd: the smaller of the
// configured drain timeout and half the NRI request timeout.
//
// Half, not all of it, so the response still has time to travel back and the
// rest of the hook to run before the runtime's own deadline fires. Giving up
// early is safe: mpsd keeps draining in the background after the caller walks
// away, and the part that actually protects the GPU — the MPS terminate that
// stops the client submitting new work — has already been issued by then.
// Kubelet's termination grace period runs after this hook returns, so the drain
// still has time to finish; what it does not have is permission to hold an NRI
// callback open.
func (p *Plugin) drainBudget() time.Duration {
	budget := p.nriRequestTimeout() / 2
	if budget <= 0 {
		// Only reachable if the runtime negotiated an absurdly small timeout.
		// An expired deadline would fail every drain instantly, which is worse
		// than briefly risking the runtime's patience.
		budget = stub.DefaultRequestTimeout / 2
	}
	if p.drainTimeout > 0 && p.drainTimeout < budget {
		budget = p.drainTimeout
	}
	return budget
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
func (p *Plugin) Synchronize(ctx context.Context, pods []*api.PodSandbox, containers []*api.Container) ([]*api.ContainerUpdate, error) {
	p.Log.InfoContext(ctx, "synchronizing container mapping with runtime",
		"pods", len(pods), "containers", len(containers))
	p.events.Synchronize(func() []store.ContainerInfo {
		infos := p.adapter.containers(pods, containers)
		p.Log.Info("rebuilt container mapping from runtime sync",
			"recordedContainers", len(infos), "totalContainers", len(containers))
		return infos
	})
	p.setReady(true)
	if p.sentinel != nil {
		p.sentinel.Audit(ctx, pods, containers)
	}
	return nil, nil
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
// device-assignment annotation when present, and routes the MPS pipe mount to
// either the default or the shared MPS server based on the container's
// compute-mode annotation (see injection.MPSPipeMount).
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
	pipeSource, pipeDestination := injection.MPSPipeMount(p.MPSPipeDirectory, computeMode)

	computePercent, hasComputeCap, err := annotations.ParseComputePortion(pod.Annotations, ctr.Name, p.AnnotationPrefix)
	if err != nil {
		p.Log.Warn("failed to parse GPU compute portion annotation",
			"container", ctr.Name,
			"pod", pod.Name,
			"error", err,
		)
		if !p.FailOpen {
			return nil, fmt.Errorf("container %q in pod %q: %w", ctr.Name, pod.Name, err)
		}
		p.Log.Warn("proceeding without a GPU compute cap (fail-open)",
			"container", ctr.Name,
			"pod", pod.Name,
		)
		hasComputeCap = false
	}

	// The scheduler's device assignment is read before the adjustment is built:
	// besides becoming NVIDIA_VISIBLE_DEVICES below, its length is how many
	// per-device entries the MPS memory limit needs.
	visibleDevices := annotations.ParseVisibleDevices(pod.Annotations, ctr.Name, p.AnnotationPrefix)
	pinnedMemLimit := injection.PinnedDeviceMemLimit(gpuMemoryCfg.Limit, injection.DeviceCount(visibleDevices))

	adj := &api.ContainerAdjustment{}

	// Each limit-carrying env var is set through setEnforcedEnv, which drops the
	// container's own value for that key before writing ours. The removal is
	// scoped to the keys actually being injected: a key we have no value for is
	// left exactly as the container declared it, so a workload that voluntarily
	// caps itself (e.g. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 baked into its
	// image, on a pod the scheduler gave no compute portion) keeps that cap
	// instead of being freed from it by an enforcement pass that had nothing to
	// put back.
	if gpuMemoryCfg.Request != "" {
		setEnforcedEnv(adj, ctr, injection.EnvGPUMemoryRequest, gpuMemoryCfg.Request)
	}
	if gpuMemoryCfg.Limit != "" {
		setEnforcedEnv(adj, ctr, injection.EnvGPUMemoryLimit, gpuMemoryCfg.Limit)
	}
	if pinnedMemLimit != "" {
		setEnforcedEnv(adj, ctr, injection.EnvMPSPinnedDeviceMemLimit, pinnedMemLimit)
	}
	if hasComputeCap {
		setEnforcedEnv(adj, ctr, injection.EnvMPSActiveThreadPercentage, strconv.Itoa(computePercent))
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
	if visibleDevices != "" {
		setEnforcedEnv(adj, ctr, injection.EnvVisibleDevices, visibleDevices)
	}

	p.Log.Info("adjusting container with GPU memory config",
		"container", ctr.Name,
		"pod", pod.Name,
		"request", gpuMemoryCfg.Request,
		"limit", gpuMemoryCfg.Limit,
		"mpsPinnedMemoryLimit", pinnedMemLimit,
		"mpsActiveThreadPercent", computePercent,
		"visibleDevices", visibleDevices,
		"computeMode", computeMode,
	)

	return adj, nil
}

// setEnforcedEnv records "set key=value, and mine wins" on the adjustment.
//
// Against the container's own spec the AddEnv alone would already be enough:
// the runtime strips every key an adjustment sets from the creation request
// before appending the adjustment's value, so the container cannot keep its
// own and cannot end up with a duplicate. The RemoveEnv matters against *other
// NRI plugins* — env keys are owned per plugin per container, and claiming a
// key another plugin already claimed is a hard error that fails the whole
// adjustment, i.e. the container would be created with none of our limits.
// Clearing the key first transfers that ownership instead of colliding with
// it. Emitting it only when the container actually declares the key keeps the
// adjustment free of no-op removals.
//
// The removal is emitted before the add. Under nri v0.12.1's create path that
// ordering happens not to matter — result.adjustEnv partitions the list into
// removals and adds in one pass before applying either, and drops the removal
// markers rather than forwarding them — but the generator that applies an
// accumulated adjustment to an OCI spec (runtime-tools/generate.AdjustEnv) is
// last-entry-wins per key, where a reversed pair would delete the variable and
// never restore it. Emitting them in the order that is correct under both costs
// nothing and does not depend on which path a future runtime takes.
func setEnforcedEnv(adj *api.ContainerAdjustment, ctr *api.Container, key, value string) {
	if containerHasEnv(ctr, key) {
		adj.RemoveEnv(key)
	}
	adj.AddEnv(key, value)
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

// StopContainer drains the container's MPS clients before the runtime stops it.
//
// This hook runs while the container is still alive, which is the whole reason
// it exists: an MPS client killed with GPU work in flight orphans that work,
// which takes down every other client on the same GPU and can leave the MPS
// server permanently unable to accept new ones. NVIDIA's supported workaround
// is to terminate the client through the MPS control daemon first — that blocks
// further submissions and drains what is outstanding — and only then kill the
// process. Doing it here makes that automatic for every fractional container,
// with no preStop hook or cooperation required from the workload.
//
// mpsd, not fractiond, owns the MPS control daemon, so the actual draining is a
// call to its endpoint (see the mpsdrain package). The hook returns no
// container updates and never an error: a drain that fails is a GPU-health
// problem, and refusing to stop a container over it would turn that into a
// stuck pod.
//
// The call is bounded by drainBudget rather than by the drain client's own
// timeout alone, because this hook runs inside the runtime's NRI request
// deadline and blowing through that disconnects the plugin — see
// nriRequestTimeout.
func (p *Plugin) StopContainer(ctx context.Context, pod *api.PodSandbox, ctr *api.Container) ([]*api.ContainerUpdate, error) {
	if p.drainer == nil || pod == nil || ctr == nil || ctr.GetId() == "" {
		return nil, nil
	}

	// Only fractional containers reach MPS, and this is the same test the
	// create hook uses to decide whether a container is one.
	cfg, err := annotations.ParseGPUMemoryAnnotations(pod.Annotations, ctr.Name, p.AnnotationPrefix)
	if err != nil || cfg.IsEmpty() {
		return nil, nil
	}

	ctx, cancel := context.WithTimeout(ctx, p.drainBudget())
	defer cancel()

	resp, err := p.drainer.Drain(ctx, mpsdrain.Request{
		ContainerID: ctr.GetId(),
		Pod:         pod.GetName(),
		Namespace:   pod.GetNamespace(),
	})
	if err != nil {
		p.Log.Warn("failed to drain MPS clients before container stop; stopping anyway",
			"container", ctr.Name,
			"pod", pod.Name,
			"error", err,
		)
		return nil, nil
	}

	if resp.Wedged {
		p.Log.Error("MPS clients could not be drained before container stop; MPS was already wedged",
			"container", ctr.Name,
			"pod", pod.Name,
			"mpsRestarted", resp.Restarted,
			"message", resp.Message,
		)
		return nil, nil
	}

	if len(resp.Drained) > 0 || resp.Restarted {
		p.Log.Info("drained MPS clients before container stop",
			"container", ctr.Name,
			"pod", pod.Name,
			"drainedClients", resp.Drained,
			"mpsRestarted", resp.Restarted,
		)
	}

	return nil, nil
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
