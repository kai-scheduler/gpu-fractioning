// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsns

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeControl stands in for nvidia-cuda-mps-control at the same seam
// *mpsctl.Control sits at. It models the daemon behaviours the manager exists
// to cope with — a namespace that already exists, a namespace with an attached
// client refusing `set` and `delete`, a daemon that lost every namespace to a
// restart — rather than the CLI's text output, which control_test.go covers.
//
// Nothing here shells out. Running a real control daemon in a unit test would
// need a GPU, and a fake CLI on $PATH would only assert what the fake printed.
type fakeControl struct {
	mu sync.Mutex

	// namespaces holds the active-thread percentage of each existing
	// namespace, by "<server>/<name>".
	namespaces map[string]int
	// busy names the namespaces that have an attached client: `namespace set`
	// and `namespace delete` both fail on those, which is the behaviour the
	// whole provision-before-start and retry-after-teardown design is built
	// around.
	busy map[string]bool
	// pipeRoot is the directory namespaces are reported under.
	pipeRoot string

	// failCreate, failSet and failDelete inject failures.
	failCreate, failSet, failDelete error
	// failList makes `namespace list` fail.
	failList error

	calls []string
}

func newFakeControl() *fakeControl {
	return &fakeControl{
		namespaces: map[string]int{"shared/default": 100},
		busy:       map[string]bool{},
		pipeRoot:   "/run/nvidia-mps",
	}
}

func (f *fakeControl) key(server, name string) string { return server + "/" + name }

func (f *fakeControl) record(format string, args ...any) {
	f.calls = append(f.calls, fmt.Sprintf(format, args...))
}

func (f *fakeControl) CreateNamespace(_ context.Context, server, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.record("create %s/%s", server, name)
	if f.failCreate != nil {
		return f.failCreate
	}
	if _, exists := f.namespaces[f.key(server, name)]; exists {
		return fmt.Errorf("namespace %q already exists", name)
	}
	f.namespaces[f.key(server, name)] = 100
	return nil
}

func (f *fakeControl) SetNamespaceActiveThreadPercentage(_ context.Context, server, name string, percent int) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.record("set %s/%s=%d", server, name, percent)
	if f.failSet != nil {
		return f.failSet
	}
	if _, exists := f.namespaces[f.key(server, name)]; !exists {
		return fmt.Errorf("namespace %q does not exist", name)
	}
	if f.busy[f.key(server, name)] {
		return fmt.Errorf("namespace %q has an active client", name)
	}
	f.namespaces[f.key(server, name)] = percent
	return nil
}

func (f *fakeControl) DeleteNamespace(_ context.Context, server, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.record("delete %s/%s", server, name)
	if f.failDelete != nil {
		return f.failDelete
	}
	if _, exists := f.namespaces[f.key(server, name)]; !exists {
		return fmt.Errorf("namespace %q does not exist", name)
	}
	if f.busy[f.key(server, name)] {
		return fmt.Errorf("namespace %q has an active client", name)
	}
	delete(f.namespaces, f.key(server, name))
	return nil
}

func (f *fakeControl) NamespacePipeDirectory(_ context.Context, server, name string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.record("get %s/%s", server, name)
	if _, exists := f.namespaces[f.key(server, name)]; !exists {
		return "", fmt.Errorf("namespace %q does not exist", name)
	}
	return filepath.Join(f.pipeRoot, server, name), nil
}

func (f *fakeControl) NamespaceNames(_ context.Context, server string) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.record("list %s", server)
	if f.failList != nil {
		return nil, f.failList
	}
	var names []string
	for key := range f.namespaces {
		if prefix := server + "/"; strings.HasPrefix(key, prefix) {
			names = append(names, strings.TrimPrefix(key, prefix))
		}
	}
	slices.Sort(names)
	return names, nil
}

// percentOf returns a namespace's cap, or -1 when it does not exist.
func (f *fakeControl) percentOf(server, name string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	if percent, ok := f.namespaces[f.key(server, name)]; ok {
		return percent
	}
	return -1
}

func (f *fakeControl) exists(server, name string) bool { return f.percentOf(server, name) >= 0 }

func (f *fakeControl) setBusy(server, name string, busy bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.busy[f.key(server, name)] = busy
}

// dropEverything is what an MPS restart looks like from the manager's side:
// every namespace is gone while the containers that were using them keep
// running.
func (f *fakeControl) dropEverything() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.namespaces = map[string]int{"shared/default": 100}
}

func (f *fakeControl) callLog() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

func quietLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// testClock is a manually advanced clock, so the grace period that protects a
// freshly provisioned namespace can be tested without waiting one out.
type testClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *testClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// testManager builds a Manager over the fake daemon, persisting to a throwaway
// file. live reports container liveness; nil means "never prune for being
// dead".
func testManager(t *testing.T, control Control, live LivenessFunc, clock *testClock) *Manager {
	t.Helper()
	m, err := New(Options{
		Control:   control,
		Server:    "shared",
		StatePath: filepath.Join(t.TempDir(), "namespaces.json"),
		Live:      live,
		Now:       clock.Now,
		Log:       quietLogger(),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return m
}

func newClock() *testClock {
	return &testClock{now: time.Date(2026, 9, 18, 12, 0, 0, 0, time.UTC)}
}

const testContainerID = "a1b2c3d4e5f60718293a4b5c6d7e8f901a2b3c4d5e6f708192a3b4c5d6e7f809"

// TestProvisionCapsTheNamespaceBeforeHandingItOut is the core of the feature:
// the container is given a namespace that already carries its ceiling, and the
// pipe directory it is given is the namespace's own leaf directory — not the
// server's, which is where the uncapped `default` namespace lives.
func TestProvisionCapsTheNamespaceBeforeHandingItOut(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 25})
	if err != nil {
		t.Fatalf("Provision() error = %v", err)
	}

	name, _ := Name(testContainerID)
	if lease.Namespace != name {
		t.Errorf("lease namespace = %q, want %q", lease.Namespace, name)
	}
	if got := control.percentOf("shared", name); got != 25 {
		t.Errorf("namespace cap = %d%%, want 25%%", got)
	}
	if want := "/run/nvidia-mps/shared/" + name; lease.PipeDirectory != want {
		t.Errorf("pipe directory = %q, want %q", lease.PipeDirectory, want)
	}
	if lease.PipeDirectory == "/run/nvidia-mps/shared" {
		t.Error("the server directory was handed out; a container mounted there is in the uncapped default namespace")
	}

	// The cap must be applied, not merely requested, before the pipe directory
	// is returned — a container that starts against an uncapped namespace can
	// have taken the card before the set lands.
	calls := control.callLog()
	setAt, getAt := -1, -1
	for i, call := range calls {
		if strings.HasPrefix(call, "set shared/"+name) {
			setAt = i
		}
		if strings.HasPrefix(call, "get shared/"+name) {
			getAt = i
		}
	}
	if setAt < 0 {
		t.Fatalf("no cap was applied: %v", calls)
	}
	if getAt < 0 {
		t.Fatalf("the pipe directory was never read back: %v", calls)
	}
}

// TestProvisionIsIdempotent: the NRI CreateContainer hook can be retried, and a
// container can be recreated with the same id. A second provision must return
// the same namespace without creating another, and — critically — without
// re-issuing `namespace set`, which fails the moment a client is attached and
// would turn a harmless retry into a container that cannot start.
func TestProvisionIsIdempotent(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	first, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 40})
	if err != nil {
		t.Fatalf("first Provision() error = %v", err)
	}

	// The container has started and attached a client, so `set` would now fail.
	control.setBusy("shared", first.Namespace, true)

	for attempt := 2; attempt <= 4; attempt++ {
		again, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 40})
		if err != nil {
			t.Fatalf("Provision() attempt %d error = %v", attempt, err)
		}
		if again.Namespace != first.Namespace || again.PipeDirectory != first.PipeDirectory ||
			again.ActiveThreadPercent != first.ActiveThreadPercent || !again.CreatedAt.Equal(first.CreatedAt) {
			t.Errorf("Provision() attempt %d = %+v, want the first lease %+v", attempt, again, first)
		}
	}

	if n := len(m.Leases()); n != 1 {
		t.Errorf("leases = %d, want 1: a retry leaked a namespace", n)
	}
	if got := control.percentOf("shared", first.Namespace); got != 40 {
		t.Errorf("namespace cap = %d%%, want it untouched at 40%%", got)
	}
}

// TestProvisionAdoptsANamespaceTheDaemonAlreadyHas: mpsd can lose its lease
// file (a fresh container filesystem, a wiped runtime directory) while the
// control daemon keeps its namespaces. The daemon is the authority, so a
// provision has to adopt what is there rather than fail trying to create it.
func TestProvisionAdoptsANamespaceTheDaemonAlreadyHas(t *testing.T) {
	control := newFakeControl()
	name, _ := Name(testContainerID)
	if err := control.CreateNamespace(context.Background(), "shared", name); err != nil {
		t.Fatal(err)
	}
	if err := control.SetNamespaceActiveThreadPercentage(context.Background(), "shared", name, 30); err != nil {
		t.Fatal(err)
	}

	control.calls = nil // ignore the setup above; only the manager's calls matter

	m := testManager(t, control, nil, newClock())
	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 30})
	if err != nil {
		t.Fatalf("Provision() error = %v", err)
	}
	if lease.Namespace != name {
		t.Errorf("lease namespace = %q, want the existing %q", lease.Namespace, name)
	}
	for _, call := range control.callLog() {
		if strings.HasPrefix(call, "create ") {
			t.Errorf("a create was issued for a namespace that already existed: %v", control.callLog())
			break
		}
	}
	// The cap IS re-applied. The manager has no record of this namespace, so it
	// has no grounds to believe the cap on it is the right one — and it is
	// reached here from CreateContainer, before the container has a client, so
	// the set can still succeed.
	if got := control.percentOf("shared", name); got != 30 {
		t.Errorf("adopted namespace cap = %d%%, want 30%%", got)
	}
}

// TestProvisionRefusesToChangeALiveCap: `namespace set` fails while a client is
// attached, so a cap cannot be changed under a running workload. Reporting that
// is the point — a silent no-op would leave a pod that looks reconfigured,
// is not, and gives no indication anywhere that the number in its annotation is
// not the number MPS is enforcing.
func TestProvisionRefusesToChangeALiveCap(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	first, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 25})
	if err != nil {
		t.Fatalf("Provision() error = %v", err)
	}

	_, err = m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 75})
	if err == nil {
		t.Fatal("Provision() with a different cap = nil error, want a refusal")
	}
	if !errors.Is(err, ErrCapChangeNeedsRestart) {
		t.Errorf("error %v is not ErrCapChangeNeedsRestart; the caller cannot tell it from a transient failure", err)
	}
	if got := control.percentOf("shared", first.Namespace); got != 25 {
		t.Errorf("namespace cap = %d%%, want it left at 25%% rather than half-changed", got)
	}
}

// TestProvisionRejectsAnOutOfRangeCap: 0 is "unlimited" to MPS, so it is the
// opposite of a cap, and over 100 is not a share of a card. Neither may reach
// the daemon or produce a lease.
func TestProvisionRejectsAnOutOfRangeCap(t *testing.T) {
	for _, percent := range []int{-10, 0, 101} {
		control := newFakeControl()
		m := testManager(t, control, nil, newClock())

		if _, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: percent}); err == nil {
			t.Errorf("Provision() with %d%% = nil error, want a rejection", percent)
		}
		if len(m.Leases()) != 0 {
			t.Errorf("Provision() with %d%% left a lease behind", percent)
		}
	}
}

// TestProvisionRejectsAPipeDirectoryThatIsTooLong is the sun_path guard, which
// has to fire here because it cannot fire anywhere useful later: the control
// daemon's own failure for an over-long pipe directory is "Failed to initialize
// MPS control daemon" with no log at all.
//
// The half-created namespace is cleaned up, so a node with a too-long pipe
// directory does not accumulate one namespace per rejected container.
func TestProvisionRejectsAPipeDirectoryThatIsTooLong(t *testing.T) {
	control := newFakeControl()
	control.pipeRoot = "/" + strings.Repeat("d", MaxPipeDirLength)
	m := testManager(t, control, nil, newClock())

	_, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50})
	if err == nil {
		t.Fatal("Provision() error = nil, want the path-length guard to fire")
	}
	if !strings.Contains(err.Error(), "characters") {
		t.Errorf("error %v does not explain the length problem", err)
	}

	name, _ := Name(testContainerID)
	if control.exists("shared", name) {
		t.Error("the rejected namespace was left behind on the daemon")
	}
	if len(m.Leases()) != 0 {
		t.Error("the rejected provision left a lease behind")
	}
}

// TestProvisionCleansUpWhenTheCapCannotBeApplied: a namespace with no cap is
// worse than no namespace, because a container routed to it is uncapped while
// looking correctly isolated. If the cap cannot be set, the namespace goes and
// the container is refused.
func TestProvisionCleansUpWhenTheCapCannotBeApplied(t *testing.T) {
	control := newFakeControl()
	control.failSet = errors.New("the control daemon said no")
	m := testManager(t, control, nil, newClock())

	if _, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50}); err == nil {
		t.Fatal("Provision() error = nil, want the failed cap to be reported")
	}

	name, _ := Name(testContainerID)
	if control.exists("shared", name) {
		t.Error("an uncapped namespace was left behind for a container that was refused")
	}
	if len(m.Leases()) != 0 {
		t.Error("the failed provision left a lease behind")
	}
}

// TestReleaseDeletesTheNamespace is the ordinary teardown, where the container
// is gone and the namespace can go with it.
func TestReleaseDeletesTheNamespace(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50})
	if err != nil {
		t.Fatal(err)
	}

	deleted, err := m.Release(context.Background(), testContainerID)
	if err != nil {
		t.Fatalf("Release() error = %v", err)
	}
	if !deleted {
		t.Error("Release() reported the namespace was not deleted")
	}
	if control.exists("shared", lease.Namespace) {
		t.Error("the namespace is still on the daemon after a successful release")
	}
	if len(m.Leases()) != 0 {
		t.Error("the lease survived its release")
	}
}

// TestReleaseRetriesWhileAClientIsStillAttached: a namespace with an attached
// client cannot be deleted, and that is the normal case rather than a failure —
// a release arrives while the container's teardown is still in progress.
// Reporting it as a failure would make fractiond log an error on every pod
// deletion; forgetting it would leak the namespace.
func TestReleaseRetriesWhileAClientIsStillAttached(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	m := testManager(t, control, nil, clock)

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50})
	if err != nil {
		t.Fatal(err)
	}
	control.setBusy("shared", lease.Namespace, true)

	deleted, err := m.Release(context.Background(), testContainerID)
	if err != nil {
		t.Fatalf("Release() error = %v, want a clean report that it is not gone yet", err)
	}
	if deleted {
		t.Error("Release() reported a deletion the daemon refused")
	}
	if !control.exists("shared", lease.Namespace) {
		t.Fatal("test data error: the namespace should still exist")
	}

	// The client detaches, and the sweep finishes the job.
	control.setBusy("shared", lease.Namespace, false)
	m.Sweep(context.Background())

	if control.exists("shared", lease.Namespace) {
		t.Error("the sweep did not delete the namespace once its client was gone")
	}
	if len(m.Leases()) != 0 {
		t.Error("the lease survived the sweep that deleted its namespace")
	}
}

// TestReleaseOfAnUnknownContainerIsANoOp: fractiond releases from
// RemoveContainer for every container it saw, including ones that never had a
// namespace. That must be silent.
func TestReleaseOfAnUnknownContainerIsANoOp(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	deleted, err := m.Release(context.Background(), "never-provisioned")
	if err != nil {
		t.Fatalf("Release() error = %v", err)
	}
	if deleted {
		t.Error("Release() claimed to delete a namespace that never existed")
	}
	for _, call := range control.callLog() {
		if strings.HasPrefix(call, "delete ") {
			t.Errorf("a delete was issued for a container with no namespace: %v", control.callLog())
		}
	}
}

// TestReconcileCapsTheDefaultNamespace is the defence in depth. Nothing is
// routed to `default` — every sm-sharing container gets a namespace of its own
// — so the only way to be in it is a mistake, and the cap is what stops that
// mistake from being "took the whole card".
func TestReconcileCapsTheDefaultNamespace(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	if got := control.percentOf("shared", DefaultNamespace); got != 100 {
		t.Fatalf("test data error: the default namespace starts at %d%%, want the uncapped 100%%", got)
	}
	if err := m.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}
	if got := control.percentOf("shared", DefaultNamespace); got != DefaultDefaultNamespaceATP {
		t.Errorf("default namespace cap = %d%%, want %d%%", got, DefaultDefaultNamespaceATP)
	}
}

// TestReconcileRecreatesNamespacesAfterAnMPSRestart: mpsd restarts the control
// daemon to clear a wedge, and after the last client leaves when recycling is
// on. That destroys every namespace while the containers using them are still
// running and still bind-mounted to their pipe directories. Recreating them at
// the same names restores the same paths; not doing so would leave every
// running fractional container pointed at a directory the daemon no longer
// owns.
func TestReconcileRecreatesNamespacesAfterAnMPSRestart(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	m := testManager(t, control, func(string) (bool, error) { return true, nil }, clock)

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 35})
	if err != nil {
		t.Fatal(err)
	}

	control.dropEverything()
	if control.exists("shared", lease.Namespace) {
		t.Fatal("test data error: the namespace should be gone")
	}

	if err := m.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}

	if !control.exists("shared", lease.Namespace) {
		t.Fatal("the namespace was not recreated; the container is pointed at a directory that no longer exists")
	}
	if got := control.percentOf("shared", lease.Namespace); got != 35 {
		t.Errorf("recreated namespace cap = %d%%, want the original 35%%", got)
	}
	// Same name means the same pipe directory, which is what the running
	// container is already mounted to.
	dir, err := control.NamespacePipeDirectory(context.Background(), "shared", lease.Namespace)
	if err != nil {
		t.Fatal(err)
	}
	if dir != lease.PipeDirectory {
		t.Errorf("recreated pipe directory = %q, want the original %q", dir, lease.PipeDirectory)
	}
}

// TestReconcileDeletesOrphanedNamespaces: an mpsd that lost its lease file
// cannot release what it does not know about, and fractiond's releases during
// an mpsd outage go nowhere. Namespaces carrying this package's prefix that no
// lease claims are therefore reclaimed — this is the "mpsd restarting must not
// strand namespaces" case.
func TestReconcileDeletesOrphanedNamespaces(t *testing.T) {
	control := newFakeControl()
	orphan, _ := Name("a-container-that-is-long-gone")
	if err := control.CreateNamespace(context.Background(), "shared", orphan); err != nil {
		t.Fatal(err)
	}

	m := testManager(t, control, nil, newClock())
	if err := m.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}

	if control.exists("shared", orphan) {
		t.Error("an orphaned namespace survived reconciliation")
	}
}

// TestReconcileNeverTouchesNamespacesItDoesNotOwn is the safety property that
// makes the permissive list parse acceptable. Reconciliation deletes; if its
// idea of "mine" were wide, it would delete another feature's namespace, a
// namespace a human created while debugging, or `default` itself — and a
// deleted namespace takes its containers' GPU access with it.
func TestReconcileNeverTouchesNamespacesItDoesNotOwn(t *testing.T) {
	control := newFakeControl()
	strangers := []string{"default", "someone_elses", "kaiabc", "n100", "shared"}
	for _, name := range strangers {
		if name == "default" {
			continue // already there
		}
		if err := control.CreateNamespace(context.Background(), "shared", name); err != nil {
			t.Fatal(err)
		}
	}

	m := testManager(t, control, nil, newClock())
	if err := m.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}

	for _, name := range strangers {
		if !control.exists("shared", name) {
			t.Errorf("reconciliation deleted namespace %q, which it does not own", name)
		}
	}
}

// TestReconcileKeepsLeasesForContainersThatAreStillRunning: the sweep's whole
// job is reclaiming namespaces whose containers are gone, so it must be certain
// about which those are. A namespace deleted under a running container costs it
// its GPU.
func TestReconcileKeepsLeasesForContainersThatAreStillRunning(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	m := testManager(t, control, func(string) (bool, error) { return true, nil }, clock)

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50})
	if err != nil {
		t.Fatal(err)
	}

	clock.advance(24 * time.Hour)
	if err := m.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}

	if !control.exists("shared", lease.Namespace) {
		t.Error("a running container's namespace was reclaimed")
	}
	if len(m.Leases()) != 1 {
		t.Error("a running container's lease was pruned")
	}
}

// TestReconcileReclaimsNamespacesOfDeadContainers is the other half: fractiond
// crashing between a container's teardown and its release leaves a lease with
// no container. Nothing else would ever clean it up.
func TestReconcileReclaimsNamespacesOfDeadContainers(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	alive := true
	m := testManager(t, control, func(string) (bool, error) { return alive, nil }, clock)

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50})
	if err != nil {
		t.Fatal(err)
	}

	alive = false
	clock.advance(DefaultOrphanGrace + time.Minute)
	if err := m.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}

	if control.exists("shared", lease.Namespace) {
		t.Error("a dead container's namespace was not reclaimed")
	}
	if len(m.Leases()) != 0 {
		t.Error("a dead container's lease survived")
	}
}

// TestOrphanGraceProtectsAContainerThatHasNotStartedYet: liveness is "the
// container has a process", and a container that has been created but has not
// started one looks exactly like a container that is gone. Provisioning happens
// in CreateContainer — strictly inside that window — so without a grace period
// an mpsd restart landing there would delete the namespace of a container that
// is about to use it.
func TestOrphanGraceProtectsAContainerThatHasNotStartedYet(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	m := testManager(t, control, func(string) (bool, error) { return false, nil }, clock)

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50})
	if err != nil {
		t.Fatal(err)
	}

	clock.advance(DefaultOrphanGrace - time.Second)
	m.Sweep(context.Background())
	if !control.exists("shared", lease.Namespace) {
		t.Fatal("a namespace was reclaimed inside the grace period, before its container could start")
	}

	clock.advance(2 * time.Second)
	m.Sweep(context.Background())
	if control.exists("shared", lease.Namespace) {
		t.Error("a namespace survived past the grace period with no container behind it")
	}
}

// TestUnknownLivenessKeepsTheNamespace: a /proc scan can fail for reasons that
// have nothing to do with the container. "I could not tell" must not be read as
// "it is dead", because the two have wildly different costs — one leaks a name,
// the other takes a running workload's GPU away.
func TestUnknownLivenessKeepsTheNamespace(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	m := testManager(t, control, func(string) (bool, error) { return false, errors.New("procfs is unreadable") }, clock)

	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50})
	if err != nil {
		t.Fatal(err)
	}

	clock.advance(DefaultOrphanGrace * 10)
	m.Sweep(context.Background())

	if !control.exists("shared", lease.Namespace) {
		t.Error("a namespace was reclaimed on the strength of a liveness check that failed")
	}
}

// TestLeasesSurviveAnMpsdRestart: the lease file is what lets a restarted mpsd
// know which namespaces belong to which containers. Without it, reconciliation
// would treat every namespace on the daemon as an orphan and delete the lot,
// taking the GPU away from every running fractional container on the node.
func TestLeasesSurviveAnMpsdRestart(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	statePath := filepath.Join(t.TempDir(), "namespaces.json")

	first, err := New(Options{Control: control, Server: "shared", StatePath: statePath, Now: clock.Now, Log: quietLogger()})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := first.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 60})
	if err != nil {
		t.Fatal(err)
	}

	// mpsd restarts: a new Manager over the same state file and the same
	// daemon, with the container still running.
	second, err := New(Options{
		Control:   control,
		Server:    "shared",
		StatePath: statePath,
		Live:      func(string) (bool, error) { return true, nil },
		Now:       clock.Now,
		Log:       quietLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := second.Reconcile(context.Background()); err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}

	if !control.exists("shared", lease.Namespace) {
		t.Fatal("a restarted mpsd deleted a running container's namespace as an orphan")
	}
	leases := second.Leases()
	if len(leases) != 1 || leases[0].ContainerID != testContainerID {
		t.Fatalf("leases after restart = %+v, want the one for %q", leases, testContainerID)
	}
	if leases[0].ActiveThreadPercent != 60 {
		t.Errorf("recovered cap = %d%%, want 60%%", leases[0].ActiveThreadPercent)
	}

	// And the restarted manager can still release it.
	deleted, err := second.Release(context.Background(), testContainerID)
	if err != nil || !deleted {
		t.Errorf("Release() after restart = (%v, %v), want (true, nil)", deleted, err)
	}
}

// TestReconcileContinuesAfterAFailure: every step is independent, and a
// half-reconciled daemon is better than one that stopped at the first error. In
// particular a `default` namespace that cannot be capped must not stop the
// namespaces containers actually use from being restored.
func TestReconcileContinuesAfterAFailure(t *testing.T) {
	control := newFakeControl()
	clock := newClock()
	statePath := filepath.Join(t.TempDir(), "namespaces.json")

	m, err := New(Options{
		Control:   control,
		Server:    "shared",
		StatePath: statePath,
		Live:      func(string) (bool, error) { return true, nil },
		Now:       clock.Now,
		Log:       quietLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 20})
	if err != nil {
		t.Fatal(err)
	}
	control.dropEverything()

	// The default namespace is gone too, so capping it fails — the shape a
	// daemon mid-restart has.
	delete(control.namespaces, "shared/default")

	err = m.Reconcile(context.Background())
	if err == nil {
		t.Error("Reconcile() error = nil, want the failed default-namespace cap reported")
	}
	if !control.exists("shared", lease.Namespace) {
		t.Error("a failure capping the default namespace stopped the real namespaces from being restored")
	}
}

// TestConcurrentProvisionsDoNotRace exercises the manager's serialisation under
// -race. Two NRI hooks for different containers run on different goroutines,
// and a retried hook for the same container can overlap its own first attempt.
func TestConcurrentProvisionsDoNotRace(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	ids := []string{testContainerID, "b" + testContainerID[1:], "c" + testContainerID[1:]}

	var wg sync.WaitGroup
	errs := make(chan error, len(ids)*4)
	for _, id := range ids {
		for range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if _, err := m.Provision(context.Background(), Request{ContainerID: id, ActiveThreadPercent: 30}); err != nil {
					errs <- err
				}
			}()
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent Provision() error = %v", err)
	}

	if n := len(m.Leases()); n != len(ids) {
		t.Errorf("leases = %d, want %d: concurrent provisions leaked or lost one", n, len(ids))
	}
}

// TestNewRejectsAnIncompleteConfiguration: a manager with no control daemon or
// no server would fail on its first call, on the container-create path.
func TestNewRejectsAnIncompleteConfiguration(t *testing.T) {
	if _, err := New(Options{Server: "shared"}); err == nil {
		t.Error("New() with no control client = nil error, want a rejection")
	}
	if _, err := New(Options{Control: newFakeControl()}); err == nil {
		t.Error("New() with no server = nil error, want a rejection")
	}
}

// TestStateFileSurvivesCorruption: a state file that cannot be decoded must not
// stop mpsd from provisioning. The daemon is the authority on what exists, and
// reconciliation only ever deletes namespaces this package generated the names
// of, so starting from an empty set is safe.
func TestStateFileSurvivesCorruption(t *testing.T) {
	control := newFakeControl()
	statePath := filepath.Join(t.TempDir(), "namespaces.json")
	if err := os.WriteFile(statePath, []byte("{not json at all"), 0o600); err != nil {
		t.Fatal(err)
	}

	m, err := New(Options{Control: control, Server: "shared", StatePath: statePath, Now: newClock().Now, Log: quietLogger()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Provision(context.Background(), Request{ContainerID: testContainerID, ActiveThreadPercent: 50}); err != nil {
		t.Errorf("Provision() with a corrupt state file error = %v, want it to carry on", err)
	}
}

// TestGPUUUIDsAreRecordedNotInterpreted: the assignment is carried for
// diagnostics. An MPS namespace belongs to a server rather than a device, so
// nothing here may quietly behave as though the cap were per GPU.
func TestGPUUUIDsAreRecordedNotInterpreted(t *testing.T) {
	control := newFakeControl()
	m := testManager(t, control, nil, newClock())

	uuids := []string{"GPU-abc123", "GPU-def456"}
	lease, err := m.Provision(context.Background(), Request{
		ContainerID:         testContainerID,
		ActiveThreadPercent: 50,
		GPUUUIDs:            uuids,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(lease.GPUUUIDs, uuids) {
		t.Errorf("lease GPU UUIDs = %v, want %v", lease.GPUUUIDs, uuids)
	}
	if got := control.percentOf("shared", lease.Namespace); got != 50 {
		t.Errorf("cap = %d%%, want 50%% regardless of how many GPUs were assigned", got)
	}

	// The caller's slice must not be aliased into the lease: it is the NRI
	// hook's, and it goes back to the runtime.
	uuids[0] = "mutated"
	if m.Leases()[0].GPUUUIDs[0] == "mutated" {
		t.Error("the lease aliases the caller's slice")
	}
}
