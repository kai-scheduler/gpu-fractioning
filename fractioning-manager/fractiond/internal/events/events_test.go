// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package events

import (
	"io"
	"log/slog"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/mapping/store"
)

type fakeWriter struct {
	mu       sync.Mutex
	upserts  []store.ContainerInfo
	deletes  []string
	replaces [][]store.ContainerInfo
}

func (w *fakeWriter) Upsert(info store.ContainerInfo) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.upserts = append(w.upserts, info)
}

func (w *fakeWriter) Delete(id string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.deletes = append(w.deletes, id)
}

func (w *fakeWriter) Replace(infos []store.ContainerInfo) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.replaces = append(w.replaces, infos)
}

func newTestProcessor(w store.Writer, opts Options) *Processor {
	return NewProcessor(w, slog.New(slog.NewTextHandler(io.Discard, nil)), opts)
}

// upsertOf returns an adapter that yields info unconditionally.
func upsertOf(info store.ContainerInfo) adapter {
	return func() (store.ContainerInfo, bool) { return info, true }
}

func TestProcessorAppliesEventsInOrder(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "a"}))
	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "b"}))
	p.Delete("a")
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.upserts) != 2 || w.upserts[0].ContainerID != "a" || w.upserts[1].ContainerID != "b" {
		t.Fatalf("unexpected upserts: %#v", w.upserts)
	}
	if len(w.deletes) != 1 || w.deletes[0] != "a" {
		t.Fatalf("unexpected deletes: %#v", w.deletes)
	}
}

func TestProcessorDropsUpsertWhenAdapterReturnsNotOK(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	p.Upsert(func() (store.ContainerInfo, bool) { return store.ContainerInfo{ContainerID: "skip"}, false })
	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "keep"}))
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.upserts) != 1 || w.upserts[0].ContainerID != "keep" {
		t.Fatalf("expected only the ok adapter to be written, got %#v", w.upserts)
	}
}

func TestProcessorReplace(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	p.Synchronize(func() []store.ContainerInfo {
		return []store.ContainerInfo{{ContainerID: "x"}, {ContainerID: "y"}}
	})
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.replaces) != 1 || len(w.replaces[0]) != 2 {
		t.Fatalf("unexpected replaces: %#v", w.replaces)
	}
}

func TestProcessorFlushIsBarrier(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	for range 50 {
		p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "c"}))
	}
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.upserts) != 50 {
		t.Fatalf("expected 50 upserts applied before Flush returned, got %d", len(w.upserts))
	}
}

func TestProcessorConsultsLogPredicate(t *testing.T) {
	w := &fakeWriter{}
	var mu sync.Mutex
	calls := 0
	p := newTestProcessor(w, Options{LogEvents: func() bool {
		mu.Lock()
		defer mu.Unlock()
		calls++
		return false
	}})

	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "a"}))
	p.Flush()

	mu.Lock()
	defer mu.Unlock()
	if calls == 0 {
		t.Fatal("expected the LogEvents predicate to be consulted on apply")
	}
}

// ---------------------------------------------------------------------------
// Flush must be bounded.
//
// Flush waits on two things that can both stop happening at once: room in the
// queue, and the worker reaching the barrier. The worker's actual work is a
// write into the shared mapping directory — a hostPath, which on a wedged
// filesystem blocks in uninterruptible sleep and never returns. Once that
// happens the queue fills behind it and an unbounded Flush blocks on the send
// forever.
//
// Its callers are Plugin.Shutdown — served as an NRI request, where overrunning
// containerd's plugin_request_timeout is fatal — and the last statement of
// fractiond's main, where blocking leaves the pod in Terminating until kubelet
// SIGKILLs it. Losing the last few mapping records instead is recoverable:
// metricsd rebuilds them from the next NRI Synchronize.
// ---------------------------------------------------------------------------

// gatedWriter parks the worker inside each write until it is handed a token,
// standing in for an fsstore write that is not coming back. Closing tokens
// releases everything, so a test can never leave the worker wedged.
type gatedWriter struct {
	tokens  chan struct{}
	entered chan struct{} // buffered: one send per write entered
}

func newGatedWriter(t *testing.T) *gatedWriter {
	t.Helper()
	w := &gatedWriter{
		tokens:  make(chan struct{}),
		entered: make(chan struct{}, 64),
	}
	t.Cleanup(func() { close(w.tokens) })
	return w
}

func (w *gatedWriter) block() {
	select {
	case w.entered <- struct{}{}:
	default:
	}
	<-w.tokens
}

func (w *gatedWriter) Upsert(store.ContainerInfo)    { w.block() }
func (w *gatedWriter) Delete(string)                 { w.block() }
func (w *gatedWriter) Replace([]store.ContainerInfo) { w.block() }

// release lets exactly one blocked write finish.
func (w *gatedWriter) release(t *testing.T) {
	t.Helper()
	select {
	case w.tokens <- struct{}{}:
	case <-time.After(5 * time.Second):
		t.Fatal("no write was waiting for a token")
	}
}

// flushDuration runs Flush on its own goroutine and reports how long it took,
// failing rather than hanging the suite if it never returns.
func flushDuration(t *testing.T, p *Processor, hardLimit time.Duration) time.Duration {
	t.Helper()
	done := make(chan time.Duration, 1)
	go func() {
		start := time.Now()
		p.Flush()
		done <- time.Since(start)
	}()
	select {
	case took := <-done:
		return took
	case <-time.After(hardLimit):
		t.Fatalf("Flush did not return within %v; this is the hang that leaves fractiond's pod in Terminating "+
			"until kubelet SIGKILLs it, and blows containerd's NRI request deadline on the way", hardLimit)
		return 0
	}
}

// fillQueue blocks the worker inside its first write and then packs the queue,
// which is the state an unbounded Flush cannot get out of.
func fillQueue(t *testing.T, p *Processor, w *gatedWriter) {
	t.Helper()
	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "stuck"}))
	select {
	case <-w.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the worker never entered a write")
	}
	for i := 0; i <= cap(p.queue); i++ {
		p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "queued"}))
	}
	if len(p.queue) != cap(p.queue) {
		t.Fatalf("queue holds %d of %d events; the test needs it full", len(p.queue), cap(p.queue))
	}
}

// TestFlushGivesUpWhenTheQueueIsFull covers the send half: with the worker
// stalled and the queue packed, the barrier cannot even be deposited. This is
// the branch an unbounded Flush blocks on forever.
func TestFlushGivesUpWhenTheQueueIsFull(t *testing.T) {
	const budget = 150 * time.Millisecond

	w := newGatedWriter(t)
	p := newTestProcessor(w, Options{FlushTimeout: budget})
	fillQueue(t, p, w)

	if took := flushDuration(t, p, 20*budget); took > 3*budget {
		t.Errorf("Flush returned after %v with a %v budget", took, budget)
	}
}

// TestFlushGivesUpWhenTheWorkerNeverReachesTheBarrier covers the wait half: the
// queue has room, so the barrier goes in, but the worker is stuck ahead of it
// and never gets there.
func TestFlushGivesUpWhenTheWorkerNeverReachesTheBarrier(t *testing.T) {
	const budget = 150 * time.Millisecond

	w := newGatedWriter(t)
	p := newTestProcessor(w, Options{FlushTimeout: budget})

	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "stuck"}))
	select {
	case <-w.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the worker never entered a write")
	}

	if took := flushDuration(t, p, 20*budget); took > 3*budget {
		t.Errorf("Flush returned after %v with a %v budget", took, budget)
	}
}

// TestFlushSpendsOneBudgetAcrossBothWaits is the reason the timer is created
// once and shared rather than per stage. A caller's constraint is the total
// time it may spend in Flush — an NRI request deadline, or the seconds before
// kubelet loses patience with a Terminating pod — and two budgets let a
// half-stalled processor spend it twice: a send that only just squeaks through
// hands the barrier wait a brand-new, full budget.
//
// The arrangement makes the send take most of a budget and then succeed, so the
// two designs separate cleanly: one shared budget returns at about the budget,
// two return at about one and a half.
func TestFlushSpendsOneBudgetAcrossBothWaits(t *testing.T) {
	const budget = 400 * time.Millisecond

	w := newGatedWriter(t)
	p := newTestProcessor(w, Options{FlushTimeout: budget})
	fillQueue(t, p, w)

	done := make(chan time.Duration, 1)
	go func() {
		start := time.Now()
		p.Flush()
		done <- time.Since(start)
	}()

	// Halfway through the budget, let exactly one write finish. The worker
	// takes the next event off the queue, which frees the slot Flush's send is
	// waiting for — and then blocks again, so the barrier is still unreachable
	// behind a full queue of events.
	time.Sleep(budget / 2)
	w.release(t)

	select {
	case took := <-done:
		if took > budget*3/2 {
			t.Errorf("Flush took %v with a %v budget; the send and the wait each got their own timer, "+
				"so a caller can be held for twice the budget it asked for", took, budget)
		}
		if took < budget/2 {
			t.Errorf("Flush returned after only %v; it gave up before its %v budget was spent", took, budget)
		}
	case <-time.After(20 * budget):
		t.Fatalf("Flush did not return within %v", 20*budget)
	}
}

// TestFlushStillFlushesWhenTheWorkerIsHealthy is the other half of the bound: a
// timeout is only an acceptable fix if it never fires in the normal case. A
// Flush that quietly became "wait a bit, then return" would drop mapping
// records on every graceful shutdown, which is worse than the hang it replaced,
// because nothing would ever notice.
func TestFlushStillFlushesWhenTheWorkerIsHealthy(t *testing.T) {
	w := &fakeWriter{}
	// Long enough that a healthy worker cannot plausibly miss it, so a failure
	// here means Flush stopped being a barrier, not that CI was slow.
	p := newTestProcessor(w, Options{FlushTimeout: 30 * time.Second})

	const events = 500
	for i := range events {
		p.Upsert(upsertOf(store.ContainerInfo{ContainerID: strconv.Itoa(i)}))
	}
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.upserts) != events {
		t.Fatalf("after Flush the writer saw %d of %d upserts; Flush is no longer a barrier", len(w.upserts), events)
	}
	// And in order, which is the property the single worker exists to provide.
	for i, got := range w.upserts {
		if got.ContainerID != strconv.Itoa(i) {
			t.Fatalf("upsert %d has id %q, want %q", i, got.ContainerID, strconv.Itoa(i))
		}
	}
}

// TestNewProcessorDefaultsTheFlushBudget guards the wiring: production never
// sets FlushTimeout, so a zero left in place would make the timer fire
// immediately and turn every Flush into a no-op.
func TestNewProcessorDefaultsTheFlushBudget(t *testing.T) {
	for _, tt := range []struct {
		name string
		opt  time.Duration
		want time.Duration
	}{
		{name: "unset", opt: 0, want: DefaultFlushTimeout},
		{name: "negative", opt: -time.Second, want: DefaultFlushTimeout},
		{name: "explicit", opt: 42 * time.Millisecond, want: 42 * time.Millisecond},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestProcessor(&fakeWriter{}, Options{FlushTimeout: tt.opt})
			if p.flushTimeout != tt.want {
				t.Errorf("flushTimeout = %v, want %v", p.flushTimeout, tt.want)
			}
		})
	}
}

// TestDefaultFlushTimeoutFitsInsideTheNRIRequestDeadline pins the constant to
// the budget that actually constrains it. Plugin.Shutdown calls Flush from an
// NRI callback, and containerd wraps those in plugin_request_timeout (2s by
// default), treating the resulting DeadlineExceeded as fatal.
func TestDefaultFlushTimeoutFitsInsideTheNRIRequestDeadline(t *testing.T) {
	const nriPluginRequestTimeout = 2 * time.Second

	if DefaultFlushTimeout <= 0 {
		t.Fatalf("DefaultFlushTimeout = %v; a non-positive budget makes every Flush a no-op", DefaultFlushTimeout)
	}
	// Half, not all: Shutdown also waits on in-flight retroactive remediation
	// after this returns.
	if DefaultFlushTimeout > nriPluginRequestTimeout/2 {
		t.Errorf("DefaultFlushTimeout = %v, which leaves less than half of containerd's %v NRI request timeout "+
			"for the rest of Plugin.Shutdown", DefaultFlushTimeout, nriPluginRequestTimeout)
	}
}
