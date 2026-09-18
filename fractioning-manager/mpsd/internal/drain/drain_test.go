// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package drain

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const testContainerID = "3f5a1c9e7b2d4086af1c2e3d4b5a69780f1e2d3c4b5a69788f7e6d5c4b3a2910"

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// fakeControl stands in for the MPS control daemon. Every terminate is
// recorded, because "which PIDs did we terminate" is the safety-critical
// output of this package.
type fakeControl struct {
	mu sync.Mutex

	// listResults is consumed one entry per ClientPIDs call; the last entry is
	// reused once exhausted, so a test only has to describe the calls it cares
	// about.
	listResults []listResult
	listCalls   int

	terminateErr  map[int]error
	terminateFunc func(ctx context.Context, pid int) error
	terminated    []int
	timeouts      []time.Duration

	// clientPIDsFunc overrides listResults entirely, for tests about the
	// deadline a list is given rather than the answer it returns.
	clientPIDsFunc func(ctx context.Context) ([]int, error)
}

type listResult struct {
	pids []int
	err  error
}

func (f *fakeControl) ClientPIDs(ctx context.Context) ([]int, error) {
	f.mu.Lock()
	f.listCalls++
	listFunc := f.clientPIDsFunc
	f.mu.Unlock()
	if listFunc != nil {
		// Called without the lock: these implementations block until their
		// context expires, and holding the mutex would deadlock every other
		// accessor for the duration.
		return listFunc(ctx)
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.listResults) == 0 {
		return nil, nil
	}
	i := min(f.listCalls-1, len(f.listResults)-1)
	return f.listResults[i].pids, f.listResults[i].err
}

func (f *fakeControl) TerminateClient(ctx context.Context, pid int, timeout time.Duration) error {
	f.mu.Lock()
	terminateFunc := f.terminateFunc
	err := f.terminateErr[pid]
	f.terminated = append(f.terminated, pid)
	f.timeouts = append(f.timeouts, timeout)
	f.mu.Unlock()

	if terminateFunc != nil {
		return terminateFunc(ctx, pid)
	}
	return err
}

func (f *fakeControl) terminatedPIDs() []int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.terminated)
}

func (f *fakeControl) listCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.listCalls
}

type restartCall struct {
	reason   string
	graceful bool
}

type fakeRestarter struct {
	mu    sync.Mutex
	calls []restartCall
	err   error
}

func (f *fakeRestarter) Restart(reason string, graceful bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, restartCall{reason: reason, graceful: graceful})
	return f.err
}

func (f *fakeRestarter) restarts() []restartCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

func staticLookup(pids []int, err error) PIDLookup {
	return func(string, string) ([]int, error) { return pids, err }
}

func TestDrainDecisionTable(t *testing.T) {
	tests := []struct {
		name            string
		control         *fakeControl
		lookup          PIDLookup
		restarter       *fakeRestarter
		noRestarter     bool
		recycleWhenIdle bool

		wantDrained    []int
		wantWedged     bool
		wantRestarted  bool
		wantMessage    string // substring; "" means the message must be empty
		wantTerminated []int
		wantRestarts   []restartCall
	}{
		{
			// The normal case on a node whose GPU nobody is using. Restarting
			// MPS here would be a gratuitous outage.
			name:           "no clients attached does nothing",
			control:        &fakeControl{listResults: []listResult{{}}},
			lookup:         staticLookup([]int{100}, nil),
			restarter:      &fakeRestarter{},
			wantTerminated: []int{},
			wantRestarts:   []restartCall{},
		},
		{
			// A control daemon that cannot be reached must not block the stop.
			name:           "ClientPIDs error reports and gives up",
			control:        &fakeControl{listResults: []listResult{{err: errors.New("control daemon is not listening")}}},
			lookup:         staticLookup([]int{100}, nil),
			restarter:      &fakeRestarter{},
			wantMessage:    "listing MPS clients: control daemon is not listening",
			wantTerminated: []int{},
			wantRestarts:   []restartCall{},
		},
		{
			// If /proc cannot be read we do not know which clients are ours, and
			// guessing would terminate another tenant's work.
			name:           "LookupPIDs error terminates nothing",
			control:        &fakeControl{listResults: []listResult{{pids: []int{100, 200}}}},
			lookup:         staticLookup(nil, errors.New("open /proc: permission denied")),
			restarter:      &fakeRestarter{},
			wantMessage:    "resolving container PIDs: open /proc: permission denied",
			wantTerminated: []int{},
			wantRestarts:   []restartCall{},
		},
		{
			// A sidecar, an init container, or a worker that died before CUDA
			// init. Extremely common and completely uninteresting.
			name:           "clients attached but none in the container",
			control:        &fakeControl{listResults: []listResult{{pids: []int{100, 200}}}},
			lookup:         staticLookup([]int{300, 400}, nil),
			restarter:      &fakeRestarter{},
			wantTerminated: []int{},
			wantRestarts:   []restartCall{},
		},
		{
			name:           "container has no PIDs at all",
			control:        &fakeControl{listResults: []listResult{{pids: []int{100}}}},
			lookup:         staticLookup(nil, nil),
			restarter:      &fakeRestarter{},
			wantTerminated: []int{},
			wantRestarts:   []restartCall{},
		},
		{
			name:           "happy path drains only the container's clients",
			control:        &fakeControl{listResults: []listResult{{pids: []int{100, 200, 300}}}},
			lookup:         staticLookup([]int{200, 300, 999}, nil),
			restarter:      &fakeRestarter{},
			wantDrained:    []int{200, 300},
			wantTerminated: []int{200, 300},
			wantRestarts:   []restartCall{},
		},
		{
			// A terminate that fails is the wedge signature, and a wedge can
			// only be cleared by killing the MPS servers outright — a graceful
			// restart would ask a wedged daemon to quit and block on it.
			name: "terminate failure sets Wedged and forces a non-graceful restart",
			control: &fakeControl{
				listResults:  []listResult{{pids: []int{100, 200, 300}}},
				terminateErr: map[int]error{200: context.DeadlineExceeded},
			},
			lookup:         staticLookup([]int{100, 200, 300}, nil),
			restarter:      &fakeRestarter{},
			wantDrained:    []int{100},
			wantWedged:     true,
			wantRestarted:  true,
			wantMessage:    "draining client 200",
			wantTerminated: []int{100, 200}, // 300 is not attempted
			wantRestarts:   []restartCall{{reason: "MPS client drain for container " + testContainerID + " did not complete", graceful: false}},
		},
		{
			// Without a Restarter the endpoint still has to answer, and answer
			// honestly: nothing was restarted.
			name: "wedged with no restarter reports Restarted=false",
			control: &fakeControl{
				listResults:  []listResult{{pids: []int{100}}},
				terminateErr: map[int]error{100: errors.New("exit status 1")},
			},
			lookup:         staticLookup([]int{100}, nil),
			noRestarter:    true,
			wantWedged:     true,
			wantMessage:    "draining client 100",
			wantTerminated: []int{100},
		},
		{
			// A Restarter that fails must not be reported as a successful
			// recovery: the node is still wedged and somebody has to know.
			name: "restarter failure reports Restarted=false",
			control: &fakeControl{
				listResults:  []listResult{{pids: []int{100}}},
				terminateErr: map[int]error{100: errors.New("exit status 1")},
			},
			lookup:         staticLookup([]int{100}, nil),
			restarter:      &fakeRestarter{err: errors.New("MPS daemon is not running")},
			wantWedged:     true,
			wantMessage:    "draining client 100",
			wantTerminated: []int{100},
			wantRestarts:   []restartCall{{reason: "MPS client drain for container " + testContainerID + " did not complete", graceful: false}},
		},
		{
			// The cheapest possible moment to clear a latent fault: nothing is
			// attached, so the restart costs nobody anything.
			name: "recycleWhenIdle restarts gracefully once the node is empty",
			control: &fakeControl{listResults: []listResult{
				{pids: []int{100}},
				{}, // nothing left after the drain
			}},
			lookup:          staticLookup([]int{100}, nil),
			restarter:       &fakeRestarter{},
			recycleWhenIdle: true,
			wantDrained:     []int{100},
			wantRestarted:   true,
			wantTerminated:  []int{100},
			wantRestarts:    []restartCall{{reason: "no MPS clients remain after drain", graceful: true}},
		},
		{
			// Somebody else is still using the GPU. Restarting would kill their
			// work, which is exactly the fault this feature prevents.
			name: "recycleWhenIdle does not restart while other clients remain",
			control: &fakeControl{listResults: []listResult{
				{pids: []int{100, 500}},
				{pids: []int{500}},
			}},
			lookup:          staticLookup([]int{100}, nil),
			restarter:       &fakeRestarter{},
			recycleWhenIdle: true,
			wantDrained:     []int{100},
			wantTerminated:  []int{100},
			wantRestarts:    []restartCall{},
		},
		{
			// Not knowing whether the node is idle is not a reason to restart
			// MPS; it is a reason to leave it alone.
			name: "recycleWhenIdle skips the recycle when the re-check fails",
			control: &fakeControl{listResults: []listResult{
				{pids: []int{100}},
				{err: errors.New("control daemon went away")},
			}},
			lookup:          staticLookup([]int{100}, nil),
			restarter:       &fakeRestarter{},
			recycleWhenIdle: true,
			wantDrained:     []int{100},
			wantTerminated:  []int{100},
			wantRestarts:    []restartCall{},
		},
		{
			name: "recycleWhenIdle off leaves an idle node alone",
			control: &fakeControl{listResults: []listResult{
				{pids: []int{100}},
				{},
			}},
			lookup:         staticLookup([]int{100}, nil),
			restarter:      &fakeRestarter{},
			wantDrained:    []int{100},
			wantTerminated: []int{100},
			wantRestarts:   []restartCall{},
		},
		{
			// A wedge already forced a restart; recycling on top of it would
			// restart MPS twice in a row and re-run the idle check against a
			// daemon that is being killed.
			name: "a wedge short-circuits the idle recycle",
			control: &fakeControl{
				listResults:  []listResult{{pids: []int{100}}, {}},
				terminateErr: map[int]error{100: context.DeadlineExceeded},
			},
			lookup:          staticLookup([]int{100}, nil),
			restarter:       &fakeRestarter{},
			recycleWhenIdle: true,
			wantWedged:      true,
			wantRestarted:   true,
			wantMessage:     "draining client 100",
			wantTerminated:  []int{100},
			wantRestarts:    []restartCall{{reason: "MPS client drain for container " + testContainerID + " did not complete", graceful: false}},
		},
		{
			// The MPS control daemon lists candidate PIDs loosely (see
			// mpsctl.parsePIDs). Only a PID that is both attached and inside the
			// container may ever be terminated.
			name:           "a PID in the container but not attached is never terminated",
			control:        &fakeControl{listResults: []listResult{{pids: []int{100}}}},
			lookup:         staticLookup([]int{100, 101, 102}, nil),
			restarter:      &fakeRestarter{},
			wantDrained:    []int{100},
			wantTerminated: []int{100},
			wantRestarts:   []restartCall{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := Options{
				Control:            tt.control,
				ProcRoot:           "/unused",
				LookupPIDs:         tt.lookup,
				ClientDrainTimeout: 50 * time.Millisecond,
				RecycleWhenIdle:    tt.recycleWhenIdle,
				Log:                quietLogger(),
			}
			if !tt.noRestarter {
				opts.Restarter = tt.restarter
			}
			s := New(opts)

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			got := s.Drain(ctx, Request{ContainerID: testContainerID, Pod: "p", Namespace: "ns"})

			if !slices.Equal(got.Drained, tt.wantDrained) {
				t.Errorf("Drained = %v, want %v", got.Drained, tt.wantDrained)
			}
			if got.Wedged != tt.wantWedged {
				t.Errorf("Wedged = %t, want %t", got.Wedged, tt.wantWedged)
			}
			if got.Restarted != tt.wantRestarted {
				t.Errorf("Restarted = %t, want %t", got.Restarted, tt.wantRestarted)
			}
			if tt.wantMessage == "" {
				if got.Message != "" {
					t.Errorf("Message = %q, want empty", got.Message)
				}
			} else if !strings.Contains(got.Message, tt.wantMessage) {
				t.Errorf("Message = %q, want it to contain %q", got.Message, tt.wantMessage)
			}
			if terminated := tt.control.terminatedPIDs(); !slices.Equal(terminated, tt.wantTerminated) {
				t.Errorf("terminated = %v, want %v", terminated, tt.wantTerminated)
			}
			if !tt.noRestarter {
				if restarts := tt.restarter.restarts(); !slices.Equal(restarts, tt.wantRestarts) {
					t.Errorf("restarts = %v, want %v", restarts, tt.wantRestarts)
				}
			}
		})
	}
}

func TestIntersectPreservesAttachedOrder(t *testing.T) {
	// The terminate order is the attach order, which is the order the control
	// daemon reports. Reordering would drain a client that is still feeding
	// another one's work first.
	tests := []struct {
		name          string
		attached      []int
		containerPIDs []int
		want          []int
	}{
		{name: "both empty", attached: nil, containerPIDs: nil},
		{name: "attached empty", containerPIDs: []int{1, 2}},
		{name: "container empty", attached: []int{1, 2}},
		{name: "disjoint", attached: []int{1, 2}, containerPIDs: []int{3, 4}},
		{name: "order follows attached, not the container", attached: []int{9, 3, 7}, containerPIDs: []int{7, 3, 9}, want: []int{9, 3, 7}},
		{name: "partial overlap", attached: []int{1, 2, 3, 4}, containerPIDs: []int{4, 2}, want: []int{2, 4}},
		{
			// The control daemon can list the same PID twice; terminating it
			// twice would fail the second time and be read as a wedge.
			name:     "a duplicate in attached is passed through as-is",
			attached: []int{5, 5}, containerPIDs: []int{5}, want: []int{5, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := intersect(tt.attached, tt.containerPIDs)
			if !slices.Equal(got, tt.want) {
				t.Errorf("intersect(%v, %v) = %v, want %v", tt.attached, tt.containerPIDs, got, tt.want)
			}
		})
	}
}

func TestDrainPassesTheConfiguredTimeoutToTerminate(t *testing.T) {
	// The per-client drain window is the whole reason a terminate is allowed to
	// block; handing the control wrapper a zero would make it fall back to the
	// much shorter command timeout and kill the client mid-kernel.
	control := &fakeControl{listResults: []listResult{{pids: []int{100}}}}
	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: 1234 * time.Millisecond,
		Log:                quietLogger(),
	})

	s.Drain(context.Background(), Request{ContainerID: testContainerID})

	control.mu.Lock()
	defer control.mu.Unlock()
	if len(control.timeouts) != 1 || control.timeouts[0] != 1234*time.Millisecond {
		t.Errorf("terminate timeouts = %v, want [1.234s]", control.timeouts)
	}
}

func TestCallerCancellationDoesNotCancelTheDrain(t *testing.T) {
	// The caller is an NRI hook on a ~1s leash (mpsdrain.DefaultCallTimeout)
	// while a genuine drain blocks for as long as the client's GPU work takes.
	// If the caller's deadline cancelled the terminate, mpsd would itself be
	// killing the client mid-kernel — the fault this endpoint exists to prevent.
	terminateStarted := make(chan struct{})
	terminateFinished := make(chan struct{})
	releaseTerminate := make(chan struct{})

	control := &fakeControl{
		listResults: []listResult{
			{pids: []int{100}},
			{}, // idle once the drain completes
		},
		terminateFunc: func(ctx context.Context, pid int) error {
			close(terminateStarted)
			select {
			case <-releaseTerminate:
				close(terminateFinished)
				return nil
			case <-ctx.Done():
				// Would mean the caller's cancellation reached the terminate.
				return fmt.Errorf("terminate was cancelled: %w", ctx.Err())
			}
		},
	}
	restarter := &fakeRestarter{}
	s := New(Options{
		Control:            control,
		Restarter:          restarter,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: 5 * time.Second,
		RecycleWhenIdle:    true,
		Log:                quietLogger(),
	})

	callerCtx, cancelCaller := context.WithCancel(context.Background())
	responses := make(chan Response, 1)
	go func() { responses <- s.Drain(callerCtx, Request{ContainerID: testContainerID}) }()

	<-terminateStarted
	cancelCaller()

	// The handler has to get out of the way promptly rather than hold the NRI
	// request open until the drain is done.
	select {
	case resp := <-responses:
		if resp.Wedged {
			t.Errorf("Wedged = true; the caller giving up is not evidence of a wedge")
		}
		if !strings.Contains(resp.Message, "continues in the background") {
			t.Errorf("Message = %q, want it to say the drain continues", resp.Message)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Drain() blocked past the caller's cancellation")
	}

	// And the drain itself carries on.
	close(releaseTerminate)
	select {
	case <-terminateFinished:
	case <-time.After(2 * time.Second):
		t.Fatal("the terminate did not complete after the caller went away")
	}

	// The decision that depends on the drain — recycling an idle node — still
	// has to be taken, in the background, on the real result.
	deadline := time.After(2 * time.Second)
	for {
		if restarts := restarter.restarts(); len(restarts) == 1 {
			if restarts[0] != (restartCall{reason: "no MPS clients remain after drain", graceful: true}) {
				t.Fatalf("restart = %+v, want a graceful idle recycle", restarts[0])
			}
			break
		}
		select {
		case <-deadline:
			t.Fatalf("the background drain never reached the recycle decision (restarts = %v)", restarter.restarts())
		case <-time.After(5 * time.Millisecond):
		}
	}

	s.waitForDrains(2 * time.Second)
}

func TestConcurrentDrainsForOneContainerShareOneTerminate(t *testing.T) {
	// fractiond's call now times out long before a drain can finish, so a retry
	// (or the retroactive stop after an NRI reconnect) can arrive while the
	// first drain is still running. Terminating the same PID twice fails the
	// second time, and a failed terminate is read as a wedge — so a duplicate
	// call would restart MPS and kill every other tenant's GPU work.
	release := make(chan struct{})
	control := &fakeControl{
		listResults: []listResult{{pids: []int{100}}},
		terminateFunc: func(ctx context.Context, _ int) error {
			<-release
			return nil
		},
	}
	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: 5 * time.Second,
		Log:                quietLogger(),
	})

	const callers = 8
	responses := make(chan Response, callers)
	var started sync.WaitGroup
	started.Add(callers)
	for range callers {
		go func() {
			started.Done()
			responses <- s.Drain(context.Background(), Request{ContainerID: testContainerID})
		}()
	}
	started.Wait()

	// Give the callers a moment to pile onto the same drain, then let it finish.
	time.Sleep(10 * time.Millisecond)
	close(release)

	for range callers {
		select {
		case resp := <-responses:
			if !slices.Equal(resp.Drained, []int{100}) {
				t.Errorf("Drained = %v, want [100] for every caller sharing the drain", resp.Drained)
			}
			if resp.Wedged {
				t.Error("Wedged = true; a shared drain must not look like a failure")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("a caller never got its response")
		}
	}

	if terminated := control.terminatedPIDs(); !slices.Equal(terminated, []int{100}) {
		t.Errorf("terminated = %v, want exactly one terminate for %d concurrent callers", terminated, callers)
	}
}

func TestDrainsForDifferentContainersDoNotShare(t *testing.T) {
	// De-duplication is per container id: two pods being stopped at once must
	// each get their own clients drained.
	control := &fakeControl{listResults: []listResult{{pids: []int{100, 200}}}}
	lookup := func(_, containerID string) ([]int, error) {
		if containerID == testContainerID {
			return []int{100}, nil
		}
		return []int{200}, nil
	}
	s := New(Options{Control: control, LookupPIDs: lookup, ClientDrainTimeout: time.Second, Log: quietLogger()})

	first := s.Drain(context.Background(), Request{ContainerID: testContainerID})
	second := s.Drain(context.Background(), Request{ContainerID: "bbbb1111cccc2222dddd3333eeee4444ffff5555aaaa6666bbbb7777cccc8888"})

	if !slices.Equal(first.Drained, []int{100}) || !slices.Equal(second.Drained, []int{200}) {
		t.Errorf("drained %v and %v, want [100] and [200]", first.Drained, second.Drained)
	}
}

func TestSequentialDrainsForTheSameContainerBothRun(t *testing.T) {
	// The de-duplication must be scoped to drains actually in flight. A second
	// stop of the same container id later (a restarting pod reusing an id, a
	// retry after mpsd answered) has to do real work, not silently return the
	// previous result.
	control := &fakeControl{listResults: []listResult{{pids: []int{100}}}}
	s := New(Options{Control: control, LookupPIDs: staticLookup([]int{100}, nil), ClientDrainTimeout: time.Second, Log: quietLogger()})

	s.Drain(context.Background(), Request{ContainerID: testContainerID})
	s.Drain(context.Background(), Request{ContainerID: testContainerID})

	if terminated := control.terminatedPIDs(); !slices.Equal(terminated, []int{100, 100}) {
		t.Errorf("terminated = %v, want two independent drains", terminated)
	}
	if got := control.listCount(); got != 2 {
		t.Errorf("ClientPIDs called %d times, want 2", got)
	}
}

func TestNewAppliesDefaults(t *testing.T) {
	// Every one of these has a production consequence if it is left zero: no
	// socket to dial, /proc unreadable, a zero drain timeout that kills clients
	// mid-kernel, and a nil lookup that panics on the first drain.
	s := New(Options{Control: &fakeControl{}})

	if s.socketPath != DefaultSocketPath {
		t.Errorf("socketPath = %q, want %q", s.socketPath, DefaultSocketPath)
	}
	if s.procRoot != "/proc" {
		t.Errorf("procRoot = %q, want /proc", s.procRoot)
	}
	if s.clientDrainTimeout != DefaultClientDrainTimeout {
		t.Errorf("clientDrainTimeout = %v, want %v", s.clientDrainTimeout, DefaultClientDrainTimeout)
	}
	if s.lookupPIDs == nil {
		t.Error("lookupPIDs = nil, want procfs.PIDsInContainer")
	}
	if s.log == nil {
		t.Error("log = nil, want slog.Default()")
	}
	if s.inFlight == nil {
		t.Error("inFlight = nil, want an initialised map")
	}

	negative := New(Options{Control: &fakeControl{}, ClientDrainTimeout: -time.Second})
	if negative.clientDrainTimeout != DefaultClientDrainTimeout {
		t.Errorf("a negative timeout gave %v, want the default", negative.clientDrainTimeout)
	}
}

func TestDrainDoesNotPanicWithoutARestarter(t *testing.T) {
	// Restarter is documented as optional, and the nil path is only reached on
	// the failure branch nobody exercises by hand.
	control := &fakeControl{
		listResults:  []listResult{{pids: []int{100}}, {}},
		terminateErr: map[int]error{100: errors.New("boom")},
	}
	s := New(Options{
		Control:         control,
		LookupPIDs:      staticLookup([]int{100}, nil),
		RecycleWhenIdle: true,
		Log:             quietLogger(),
	})

	resp := s.Drain(context.Background(), Request{ContainerID: testContainerID})
	if !resp.Wedged || resp.Restarted {
		t.Errorf("response = %+v, want Wedged without Restarted", resp)
	}
}

func TestDrainWithAnIdleRecycleAndNoRestarter(t *testing.T) {
	control := &fakeControl{listResults: []listResult{{pids: []int{100}}, {}}}
	s := New(Options{
		Control:         control,
		LookupPIDs:      staticLookup([]int{100}, nil),
		RecycleWhenIdle: true,
		Log:             quietLogger(),
	})

	resp := s.Drain(context.Background(), Request{ContainerID: testContainerID})
	if resp.Restarted {
		t.Errorf("Restarted = true with no restarter configured")
	}
	if !slices.Equal(resp.Drained, []int{100}) {
		t.Errorf("Drained = %v, want [100]", resp.Drained)
	}
}

func TestWaitForDrainsGivesUpRatherThanHanging(t *testing.T) {
	// mpsd must still be able to exit when the control daemon has stopped
	// answering; a shutdown that blocks forever is a pod stuck in Terminating.
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })

	control := &fakeControl{
		listResults:   []listResult{{pids: []int{100}}},
		terminateFunc: func(context.Context, int) error { <-release; return nil },
	}
	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: time.Hour,
		Log:                quietLogger(),
	})

	callerCtx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	s.Drain(callerCtx, Request{ContainerID: testContainerID})

	start := time.Now()
	s.waitForDrains(50 * time.Millisecond)
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("waitForDrains took %v, want it to give up after ~50ms", elapsed)
	}
}

// TestEachClientGetsItsOwnDrainWindow pins the "one budget per step, not one
// shared budget" rule in runDrain.
//
// A container can hold several MPS clients, and every one of them is entitled
// to the full drain window: the terminate blocks for as long as that client's
// outstanding GPU work takes, which is the whole point of issuing it. Share one
// budget across them and the second client is cut short by however long the
// first one legitimately took — reported as a failed terminate, which mpsd
// reads as a wedged server and answers with an MPS restart that kills every
// other tenant's work on the node. A drain that succeeded twice would take the
// GPU down.
//
// The assertion is on the deadlines the terminates are handed, not on elapsed
// wall-clock time: under per-step budgets the second terminate's deadline is
// strictly later than the first's (its window started later), while a shared
// budget hands both the very same instant. That difference is exact, so this
// test cannot flake on a loaded machine.
func TestEachClientGetsItsOwnDrainWindow(t *testing.T) {
	const window = time.Second
	// Long enough to be unambiguous against a monotonic clock, short enough
	// that it costs the suite nothing.
	const firstClientWork = 20 * time.Millisecond

	var mu sync.Mutex
	var deadlines []time.Time

	control := &fakeControl{
		listResults: []listResult{{pids: []int{100, 101}}},
		terminateFunc: func(ctx context.Context, pid int) error {
			deadline, ok := ctx.Deadline()
			if !ok {
				return fmt.Errorf("terminate of %d got a context with no deadline", pid)
			}
			mu.Lock()
			deadlines = append(deadlines, deadline)
			mu.Unlock()

			if pid != 100 {
				return nil
			}
			// The first client's work is real: it is what consumes the shared
			// budget in the broken arrangement.
			select {
			case <-time.After(firstClientWork):
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		},
	}

	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100, 101}, nil),
		ClientDrainTimeout: window,
		Log:                quietLogger(),
	})

	resp := s.Drain(context.Background(), Request{ContainerID: testContainerID})

	if resp.Wedged {
		t.Errorf("Wedged = true (%q); both clients drained, nothing was wedged", resp.Message)
	}
	if !slices.Equal(resp.Drained, []int{100, 101}) {
		t.Errorf("Drained = %v, want [100 101]", resp.Drained)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(deadlines) != 2 {
		t.Fatalf("recorded %d terminate deadlines, want 2", len(deadlines))
	}
	if !deadlines[1].After(deadlines[0]) {
		t.Errorf("the second client's terminate deadline (%v) is not later than the first's (%v); "+
			"the clients are sharing one budget, so a slow first client eats the second one's drain window",
			deadlines[1], deadlines[0])
	}
	// And the second window is a whole window, not the remains of the first.
	if got := deadlines[1].Sub(deadlines[0]); got < firstClientWork/2 {
		t.Errorf("the second terminate's window started only %v after the first's, want at least %v",
			got, firstClientWork/2)
	}
}

// TestDrainWithoutAControlClientDoesNotPanic covers the guard at the top of
// Drain. Options.Control is documented as required and the production wiring
// always sets it, but this code runs on the container-stop path: a nil
// dereference here would panic the goroutine and take mpsd down every time a
// fractional pod is deleted — losing the MPS supervisor, the drain endpoint and
// the driver labeller over a misconfiguration whose correct outcome is one
// logged error and a container that stops undrained.
func TestDrainWithoutAControlClientDoesNotPanic(t *testing.T) {
	s := New(Options{
		LookupPIDs: staticLookup([]int{100}, nil),
		Log:        quietLogger(),
	})

	resp := s.Drain(context.Background(), Request{ContainerID: testContainerID})

	if len(resp.Drained) != 0 || resp.Wedged || resp.Restarted {
		t.Errorf("response = %+v, want an empty response", resp)
	}
	if resp.Message == "" {
		t.Error("Message is empty; a drain that could not run at all has to say so, or the misconfiguration is invisible")
	}
	// The guard must short-circuit rather than start a background drain that
	// would dereference the nil control itself.
	s.mu.Lock()
	inFlight := len(s.inFlight)
	s.mu.Unlock()
	if inFlight != 0 {
		t.Errorf("inFlight = %d, want 0: no background drain should have been started", inFlight)
	}
	s.waitForDrains(time.Second)
}

// ── budgeting the client list (the hook's terminate has to actually happen) ──

// TestContainerPIDsAreResolvedBeforeTheControlDaemonIsAsked.
//
// The drain runs inside an NRI StopContainer hook with a budget of around a
// second. Resolving the container's PIDs is a local /proc scan that cannot
// block on MPS and can end the drain on its own, so doing it first means the
// control daemon is only consulted when there is something to consult it about.
// With the order reversed, a container that never touched the GPU — a sidecar,
// an init container, most containers on most nodes — spent the hook's whole
// budget on a `client list` whose answer could not matter.
func TestContainerPIDsAreResolvedBeforeTheControlDaemonIsAsked(t *testing.T) {
	control := &fakeControl{listResults: []listResult{{pids: []int{100, 200}}}}
	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup(nil, nil),
		ClientDrainTimeout: time.Second,
		Log:                quietLogger(),
	})

	resp := s.Drain(context.Background(), Request{ContainerID: testContainerID})

	if len(resp.Drained) != 0 {
		t.Errorf("drained %v, want nothing for a container with no processes", resp.Drained)
	}
	if got := control.listCount(); got != 0 {
		t.Errorf("ClientPIDs called %d times for a container with no processes, want 0", got)
	}
}

// TestASlowClientListStillLeadsToATerminate is the defect this budgeting
// exists for.
//
// `client list` used to get the full drain budget (30s by default) while the
// caller's hook has about a second. One slow list therefore consumed the entire
// hook window before a single terminate was issued: the hook returned having
// protected nothing, and the container was killed with GPU work in flight —
// the orphaned-client wedge this endpoint exists to prevent, caused by the
// endpoint.
//
// Now the first attempt is bounded by the much shorter list budget, and only
// the retry gets the drain budget. The terminate still happens, because the
// drain goroutine outlives the caller by design.
func TestASlowClientListStillLeadsToATerminate(t *testing.T) {
	const listTimeout = 20 * time.Millisecond

	var attempts atomic.Int32
	control := &fakeControl{}
	control.clientPIDsFunc = func(ctx context.Context) ([]int, error) {
		// The first attempt never answers; the retry does. A control daemon
		// that is loaded rather than broken behaves like this.
		if attempts.Add(1) == 1 {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return []int{100}, nil
	}

	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: 5 * time.Second,
		ClientListTimeout:  listTimeout,
		Log:                quietLogger(),
	})

	start := time.Now()
	resp := s.Drain(context.Background(), Request{ContainerID: testContainerID})
	elapsed := time.Since(start)

	if !slices.Equal(resp.Drained, []int{100}) {
		t.Fatalf("drained %v, want [100]: a slow list must not mean no terminate", resp.Drained)
	}
	if !slices.Equal(control.terminatedPIDs(), []int{100}) {
		t.Errorf("terminated %v, want [100]", control.terminatedPIDs())
	}
	// The first attempt must have been cut off near the list budget rather than
	// running to the drain budget, which is what leaves room for the terminate.
	if elapsed > time.Second {
		t.Errorf("the drain took %v; the first list was not bounded by the %v list budget", elapsed, listTimeout)
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("ClientPIDs attempted %d times, want 2 (a short one, then the retry)", got)
	}
}

// TestTheFirstClientListGetsTheShortBudget pins the deadline the control
// daemon actually sees. A list handed the drain budget is the defect above; a
// list handed no deadline at all is worse.
func TestTheFirstClientListGetsTheShortBudget(t *testing.T) {
	control := &fakeControl{}
	var firstDeadline time.Duration
	control.clientPIDsFunc = func(ctx context.Context) ([]int, error) {
		if deadline, ok := ctx.Deadline(); ok {
			firstDeadline = time.Until(deadline)
		}
		return nil, nil
	}

	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: 30 * time.Second,
		ClientListTimeout:  250 * time.Millisecond,
		Log:                quietLogger(),
	})
	s.Drain(context.Background(), Request{ContainerID: testContainerID})

	if firstDeadline <= 0 {
		t.Fatal("the first client list carried no deadline")
	}
	if firstDeadline > 250*time.Millisecond {
		t.Errorf("the first client list got %v, want at most the 250ms list budget", firstDeadline)
	}
}

// TestAFailedClientListIsNotRetried: only slowness is worth waiting longer for.
// A control daemon that is not listening will not start listening within the
// drain budget, and retrying would spend it finding that out again.
func TestAFailedClientListIsNotRetried(t *testing.T) {
	control := &fakeControl{listResults: []listResult{{err: errors.New("control daemon is not listening")}}}
	s := New(Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: time.Second,
		ClientListTimeout:  10 * time.Millisecond,
		Log:                quietLogger(),
	})

	s.Drain(context.Background(), Request{ContainerID: testContainerID})

	if got := control.listCount(); got != 1 {
		t.Errorf("ClientPIDs called %d times for an outright failure, want 1", got)
	}
}

// TestClientListBudgetIsNeverLongerThanTheDrainBudget: the retry has to be a
// lengthening. A misconfiguration that inverted them would make the second
// attempt shorter than the first, so a list that was merely slow would fail
// twice and report a wedge that is not there.
func TestClientListBudgetIsNeverLongerThanTheDrainBudget(t *testing.T) {
	s := New(Options{Control: &fakeControl{}, ClientDrainTimeout: time.Second, ClientListTimeout: time.Minute})
	if s.clientListTimeout > s.clientDrainTimeout {
		t.Errorf("clientListTimeout = %v, clientDrainTimeout = %v; the retry would be shorter than the first attempt",
			s.clientListTimeout, s.clientDrainTimeout)
	}
}
