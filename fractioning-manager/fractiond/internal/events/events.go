// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package events decouples the ingestion of container lifecycle events from
// their application to storage. Producers (e.g. the NRI plugin handlers) hand the
// Processor an adapter that yields the container→pod mapping and return
// immediately; a single background worker runs each adapter and applies the
// result to a store.Writer in order. The runtime hot path is never blocked and
// create/delete ordering is preserved without locking.
//
// The Processor's API is deliberately free of any runtime (NRI) types: it speaks
// only store.ContainerInfo and the Adapter/SyncAdapter function types. The work
// of turning a runtime-specific event into a store.ContainerInfo — "calculating
// the mapping" — is supplied by the caller as an adapter and executed on the
// worker goroutine, so even that conversion stays off the producer's hot path.
// The writer in production is the filesystem mapping store (internal/fsstore),
// but the Processor depends only on the store.Writer interface and neither knows
// nor cares how the mapping is persisted.
package events

import (
	"log/slog"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

// defaultQueueDepth buffers events so the producer rarely hits the drop path.
// Sized well above realistic per-node container churn; a full queue causes the
// incoming event to be dropped rather than blocking the NRI callback goroutine.
const defaultQueueDepth = 10 * 1024

// adapter produces the mapping for a single container. It is run on the worker
// goroutine, off the producer's hot path. ok is false when the event carries no
// metrics-relevant container (e.g. no GPU assigned), in which case it is dropped.
type adapter func() (store.ContainerInfo, bool)

// syncAdapter produces the full set of current container mappings for a resync.
// It is run on the worker goroutine.
type syncAdapter func() []store.ContainerInfo

// kind is the type of mapping change an event carries.
type kind int

const (
	upsert kind = iota
	remove
	replace
)

// event is a pending mapping change. The callback is executed on the worker
// goroutine, keeping the producer's NRI hot path non-blocking.
type event struct {
	kind          kind
	adapt         adapter     // upsert
	syncAdapt     syncAdapter // replace
	containerID   string      // remove
	fromReconnect bool        // replace: came from an NRI Synchronize (reconnect); skip Replace when infos is empty
}

// Options configures a Processor.
type Options struct {
	// LogEvents, if non-nil, is consulted on each applied event to decide whether
	// to emit a log line. It is read on the worker goroutine so it reflects the
	// current value (e.g. a live config flag).
	LogEvents func() bool
}

// Processor applies events to storage on a single background goroutine.
type Processor struct {
	writer    store.Writer
	log       *slog.Logger
	logEvents func() bool
	queue     chan item
}

// item is a queued event or a flush barrier (done != nil).
type item struct {
	event event
	done  chan struct{}
}

// NewProcessor starts the worker and returns a ready Processor. The worker runs
// until the process exits; callers drain pending work with Flush (e.g. on
// shutdown) but do not need to stop it explicitly.
func NewProcessor(writer store.Writer, logger *slog.Logger, opts Options) *Processor {
	if logger == nil {
		logger = slog.Default()
	}
	p := &Processor{
		writer:    writer,
		log:       logger,
		logEvents: opts.LogEvents,
		queue:     make(chan item, defaultQueueDepth),
	}
	go p.run()
	return p
}

// Upsert schedules a single container's mapping to be (re)written. adapt is run
// on the worker goroutine and may return ok=false to drop the event. Returns
// immediately; drops the event if the queue is full rather than blocking the
// NRI callback.
func (p *Processor) Upsert(adapt func() (store.ContainerInfo, bool)) {
	p.enqueue(item{event: event{kind: upsert, adapt: adapt}})
}

// Delete schedules removal of a container's mapping by ID. Returns immediately;
// drops the event if the queue is full rather than blocking the NRI callback.
func (p *Processor) Delete(containerID string) {
	p.enqueue(item{event: event{kind: remove, containerID: containerID}})
}

// Synchronize schedules a full replace of the mapping set (used on runtime
// resync). adapt is run on the worker goroutine and returns the complete current
// container set. Returns immediately; drops the event if the queue is full
// rather than blocking the NRI callback.
//
// If the resulting container list is empty the Replace is skipped: an empty
// Synchronize indicates containerd restarted and has not yet replayed existing
// containers (observed in k3s/k3d). Pruning would wipe all mapping files and
// cause a metric gap for every running workload; skipping preserves attribution
// across the reconnect window. A subsequent non-empty Synchronize, or
// individual Upsert/Delete events, will reconcile the directory.
//
// This is intentionally different from a non-reconnect Replace(nil): genuine
// transitions to zero containers arrive via Delete events, not Synchronize.
func (p *Processor) Synchronize(adapt func() []store.ContainerInfo) {
	p.enqueue(item{event: event{kind: replace, syncAdapt: adapt, fromReconnect: true}})
}

// enqueue sends an item to the worker without blocking. If the queue is full the
// event is dropped and a warning is logged. A full queue means the worker is
// falling behind or has panicked; dropping is preferable to stalling the NRI
// runtime callback goroutine.
func (p *Processor) enqueue(it item) {
	select {
	case p.queue <- it:
	default:
		p.log.Warn("event queue full; dropping event")
	}
}

// Flush blocks until every event queued so far has been applied. Used by
// graceful shutdown and tests; it is a no-op marker in the event stream.
func (p *Processor) Flush() {
	done := make(chan struct{})
	p.queue <- item{done: done}
	<-done
}

func (p *Processor) run() {
	for it := range p.queue {
		if it.done != nil {
			close(it.done)
			continue
		}
		p.safeApply(it.event)
	}
}

func (p *Processor) safeApply(ev event) {
	defer func() {
		if r := recover(); r != nil {
			p.log.Error("panic in event processor; event dropped", "panic", r)
		}
	}()
	p.apply(ev)
}

func (p *Processor) apply(ev event) {
	logEnabled := p.logEvents != nil && p.logEvents()
	switch ev.kind {
	case upsert:
		info, ok := ev.adapt()
		if !ok {
			return
		}
		p.writer.Upsert(info)
		if logEnabled {
			p.logContainer("recorded container mapping", info)
		}
	case remove:
		p.writer.Delete(ev.containerID)
		if logEnabled {
			p.log.Info("removed container from metrics mapping", "containerID", ev.containerID)
		}
	case replace:
		infos := ev.syncAdapt()
		if ev.fromReconnect && len(infos) == 0 {
			p.log.Debug("NRI Synchronize: skipping empty reconnect sync to preserve existing mapping files")
			return
		}
		p.writer.Replace(infos)
		if logEnabled {
			for _, info := range infos {
				p.logContainer("synchronized container mapping", info)
			}
		}
	}
}

func (p *Processor) logContainer(msg string, info store.ContainerInfo) {
	p.log.Info(msg,
		"pod", info.Pod,
		"namespace", info.Namespace,
		"podUID", info.PodUID,
		"container", info.Container,
		"containerID", info.ContainerID,
		"gpuDevices", info.GPUDevices,
	)
}
