// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package mpsns provisions one MPS namespace per fractional container and caps
// it, so a container's compute allotment is something MPS enforces rather than
// something the container is asked to respect.
//
// # Why a namespace
//
// The compute cap used to travel to the container as
// CUDA_MPS_ACTIVE_THREAD_PERCENTAGE and nothing else. That variable is read by
// the CUDA runtime inside the container, which means any process in the
// container can re-export it before cuInit and take the whole card. mpsd — the
// only component with authority over MPS — never learned the per-container
// portion at all.
//
// A per-namespace active-thread percentage is a different thing entirely: it is
// an authoritative ceiling. Measured on an RTX PRO 6000 (188 SMs, driver 615):
// with the namespace set to 25%, a client in it got 46 SMs, and exporting
// CUDA_MPS_ACTIVE_THREAD_PERCENTAGE of 100, 75 or 50 still got 46. Only a
// LOWER value took effect (10 → 18 SMs). The invariant is
//
//	effective = min(client env, namespace ceiling)
//
// so the container can lower its own share and can never raise it. It is
// genuinely per-namespace, not per-server: a sibling namespace on the same
// server was unaffected, and heterogeneous caps (25% and 75%) held
// concurrently on one server.
//
// # What makes it a boundary
//
//   - The control daemon runs multiuser (-m). This is not optional. Without it
//     a server only accepts clients whose uid matches its owner, which forces
//     the pod to run as the owner and hands it owner privileges over the
//     control API. With it, `server create --uid=<non-zero>` is refused
//     outright — every server is root-owned — and non-root clients attach and
//     are capped. Verified with uids 60000 and 70000 concurrently on one
//     root-owned server.
//   - A non-root, non-owner client cannot raise its own cap: `namespace set
//     --active-thread-percentage=100` from inside such a container fails with
//     "Insufficient privileges for UID 60000. 'NAMESPACE SET' requires server
//     owner." The same holds for `server set`, `namespace create/delete` and
//     `sm-partition create`.
//   - The container is bind-mounted ONLY its own leaf namespace directory. The
//     `default` namespace of every server is uncapped and has its own control
//     socket, so a container handed the server directory instead would be
//     uncapped; mpsd additionally caps `default` itself, so that mistake costs
//     a small share rather than the card.
//
// SM partitions are deliberately NOT used. They are exact and hardware-
// enforced, but CUDA_MPS_SM_PARTITION is chosen by the client and is not
// validated against the namespace: a container in a 40-SM namespace was
// measured taking a 136-SM partition. They cannot be a security boundary.
//
// # Known residual gap
//
// Any uid can run `namespace list`, `server list` or `sm-partition list --all`
// through its own namespace socket and read every other tenant's cap and pipe
// path. That is information disclosure; no mutation is possible from a
// non-owner uid. It is documented rather than fixed.
package mpsns

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultNamespace is the namespace every MPS server has, which is what a
	// client that names none lands in. It is uncapped (100%) unless something
	// caps it, which is why mpsd does.
	DefaultNamespace = "default"

	// DefaultDefaultNamespaceATP is the ceiling mpsd puts on that namespace.
	//
	// Nothing is supposed to be in it: every sm-sharing container is routed to
	// a namespace of its own. So this is purely defence in depth against a
	// container that reaches the server socket some other way — a mount that
	// was built wrong, an image that hardcodes a pipe directory, a future
	// change that forgets. Low enough that such a container cannot take the
	// card, high enough that it runs and can be noticed rather than appearing
	// to hang.
	DefaultDefaultNamespaceATP = 10

	// DefaultOrphanGrace is how long a lease is protected from being pruned as
	// an orphan. Liveness is "the container has at least one process", and a
	// container that has been created but has not started one yet looks
	// identical to a container that is gone. Provisioning happens in the NRI
	// CreateContainer hook, i.e. strictly before the first process exists, so
	// without a grace window an mpsd restart landing in that gap would delete a
	// namespace a container is about to use.
	DefaultOrphanGrace = 5 * time.Minute

	// DefaultSweepInterval is how often pending deletes are retried. A release
	// arrives while the container's client may still be attached, and an
	// attached client blocks the delete, so the first attempt often fails by
	// design.
	DefaultSweepInterval = 30 * time.Second
)

// ErrCapChangeNeedsRestart is returned when a container asks for a different
// cap than the one its live namespace already carries.
//
// It is an error rather than a silent no-op on purpose. `namespace set` fails
// while the namespace has an active client, so a cap cannot be changed under a
// running workload; reporting the container as un-provisionable makes the
// operator restart it, which is the only thing that actually applies the new
// number. Swallowing it would leave a pod that looks reconfigured and is not.
var ErrCapChangeNeedsRestart = errors.New("the MPS namespace already has a different compute cap; the container must be restarted to change it")

// Control is the slice of the MPS control daemon this package drives.
// *mpsctl.Control implements it; tests substitute a fake at this seam rather
// than shelling out to a real nvidia-cuda-mps-control.
type Control interface {
	CreateNamespace(ctx context.Context, server, name string) error
	SetNamespaceActiveThreadPercentage(ctx context.Context, server, name string, percent int) error
	DeleteNamespace(ctx context.Context, server, name string) error
	NamespacePipeDirectory(ctx context.Context, server, name string) (string, error)
	NamespaceNames(ctx context.Context, server string) ([]string, error)
}

// LivenessFunc reports whether a container still exists on the node.
type LivenessFunc func(containerID string) (bool, error)

// Request is one provisioning request.
type Request struct {
	// ContainerID is the runtime's container id.
	ContainerID string
	// ActiveThreadPercent is the ceiling to put on the namespace, 1..100.
	ActiveThreadPercent int
	// GPUUUIDs are the devices the scheduler assigned (diagnostic; see
	// Lease.GPUUUIDs).
	GPUUUIDs []string
	// Pod and Namespace are carried for logging only.
	Pod, Namespace string
}

// Options configures a Manager.
type Options struct {
	// Control drives the control daemon. Required.
	Control Control
	// Server is the MPS server namespaces are created on. Required.
	Server string
	// StatePath is the file the lease set is persisted to. Empty disables
	// persistence, which costs reconciliation its memory across an mpsd
	// restart.
	StatePath string
	// DefaultNamespaceATP is the ceiling put on the server's `default`
	// namespace. Zero uses DefaultDefaultNamespaceATP; negative leaves the
	// namespace alone.
	DefaultNamespaceATP int
	// Live reports whether a container still exists. Required for orphan
	// pruning; without it a lease is never pruned for being dead, only for
	// being released.
	Live LivenessFunc
	// OrphanGrace protects young leases from being pruned. Zero uses
	// DefaultOrphanGrace.
	OrphanGrace time.Duration
	// Now is the clock (tests).
	Now func() time.Time
	// Log defaults to slog.Default().
	Log *slog.Logger
}

// Manager owns the node's per-container MPS namespaces.
//
// Every exported method is safe for concurrent use and serialises on one mutex.
// That is deliberate: the control daemon is a single process driven through a
// CLI, the operations are short, and two concurrent provisions for the same
// container racing between "does it exist" and "create it" would produce a
// create failure that looks like a real one.
type Manager struct {
	control             Control
	server              string
	store               store
	defaultNamespaceATP int
	live                LivenessFunc
	orphanGrace         time.Duration
	now                 func() time.Time
	log                 *slog.Logger

	mu     sync.Mutex
	leases map[string]Lease // by container id
	loaded bool
}

// New builds a Manager. It does not touch the control daemon; call Reconcile
// once the daemon is up (and again after every restart of it).
func New(opts Options) (*Manager, error) {
	if opts.Control == nil {
		return nil, fmt.Errorf("MPS namespace manager needs a control daemon client")
	}
	if opts.Server == "" {
		return nil, fmt.Errorf("MPS namespace manager needs a server name")
	}
	atp := opts.DefaultNamespaceATP
	if atp == 0 {
		atp = DefaultDefaultNamespaceATP
	}
	grace := opts.OrphanGrace
	if grace <= 0 {
		grace = DefaultOrphanGrace
	}
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	log := opts.Log
	if log == nil {
		log = slog.Default()
	}

	return &Manager{
		control:             opts.Control,
		server:              opts.Server,
		store:               store{path: opts.StatePath},
		defaultNamespaceATP: atp,
		live:                opts.Live,
		orphanGrace:         grace,
		now:                 now,
		log:                 log,
		leases:              map[string]Lease{},
	}, nil
}

// Server returns the MPS server namespaces are created on.
func (m *Manager) Server() string { return m.server }

// Provision returns the container's namespace, creating and capping it if it
// does not have one yet, and returns the pipe directory the container must be
// pointed at.
//
// It is idempotent, because the NRI CreateContainer hook it is called from can
// be retried and a container can be recreated with the same id. Repeating a
// provision with the same cap re-reads the pipe directory and returns it; it
// does not re-issue `namespace set`, which would fail as soon as a client had
// attached. Asking for a DIFFERENT cap on a namespace that already exists
// returns ErrCapChangeNeedsRestart rather than pretending to apply it.
//
// The daemon, not the lease file, is the authority on whether a namespace
// exists: a create is attempted only when the pipe directory cannot be read,
// and a create failure is not fatal if the directory can be read afterwards.
// That is what makes it survive an mpsd that lost its state and a daemon that
// kept its namespaces, and vice versa.
func (m *Manager) Provision(ctx context.Context, req Request) (Lease, error) {
	if req.ActiveThreadPercent < 1 || req.ActiveThreadPercent > 100 {
		return Lease{}, fmt.Errorf("provisioning an MPS namespace for container %q: active thread percentage %d is out of range, expected 1..100",
			req.ContainerID, req.ActiveThreadPercent)
	}
	name, err := Name(req.ContainerID)
	if err != nil {
		return Lease{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.ensureLoaded(); err != nil {
		// A state file we cannot read must not stop a container from getting
		// its cap: the daemon is the authority below, and reconciliation will
		// sort the bookkeeping out.
		m.log.Warn("could not load the MPS namespace state; continuing from an empty set", "error", err)
	}

	existing, known := m.leases[req.ContainerID]
	if known && !existing.released() && existing.ActiveThreadPercent != req.ActiveThreadPercent {
		return Lease{}, fmt.Errorf("container %q holds MPS namespace %q capped at %d%%, not %d%%: %w",
			req.ContainerID, existing.Namespace, existing.ActiveThreadPercent, req.ActiveThreadPercent, ErrCapChangeNeedsRestart)
	}

	// Ask the daemon whether the namespace is already there. This is the check
	// that makes a retried hook cheap and a lost state file harmless.
	pipeDir, err := m.control.NamespacePipeDirectory(ctx, m.server, name)
	fresh := false
	if err != nil {
		if createErr := m.control.CreateNamespace(ctx, m.server, name); createErr != nil {
			// Inconclusive, not fatal: a concurrent creator (or a stale lease)
			// may have created it a moment ago. The read below settles it.
			m.log.Debug("creating the MPS namespace failed; re-reading it before giving up",
				"namespace", name, "error", createErr)
		}
		pipeDir, err = m.control.NamespacePipeDirectory(ctx, m.server, name)
		if err != nil {
			return Lease{}, fmt.Errorf("provisioning MPS namespace %q for container %q: %w", name, req.ContainerID, err)
		}
		fresh = true
	}

	// Guard the length before anything is handed out. A path that overruns
	// sun_path makes the daemon fail to initialise with no log at all, so the
	// only chance to say something useful about it is here.
	if err := CheckPipeDirLength(pipeDir); err != nil {
		if fresh {
			m.deleteNamespace(ctx, name)
		}
		return Lease{}, fmt.Errorf("provisioning MPS namespace %q for container %q: %w", name, req.ContainerID, err)
	}

	// The cap is applied only when the namespace is new to us, because
	// `namespace set` fails once a client has attached and a retried hook for a
	// container that is already running would otherwise turn into a spurious
	// provisioning failure. A namespace this manager created always gets its
	// cap before the container's first CUDA process exists — that is the whole
	// reason provisioning lives in CreateContainer.
	if fresh || !known || existing.released() {
		if err := m.control.SetNamespaceActiveThreadPercentage(ctx, m.server, name, req.ActiveThreadPercent); err != nil {
			if fresh {
				m.deleteNamespace(ctx, name)
			}
			return Lease{}, fmt.Errorf("capping MPS namespace %q for container %q: %w", name, req.ContainerID, err)
		}
	}

	lease := Lease{
		ContainerID:         req.ContainerID,
		Namespace:           name,
		Server:              m.server,
		PipeDirectory:       pipeDir,
		ActiveThreadPercent: req.ActiveThreadPercent,
		GPUUUIDs:            slices.Clone(req.GPUUUIDs),
		CreatedAt:           m.now(),
	}
	if known && !existing.released() {
		lease.CreatedAt = existing.CreatedAt
	}
	m.leases[req.ContainerID] = lease
	m.persist()

	m.log.Info("provisioned MPS namespace",
		"containerId", req.ContainerID,
		"pod", req.Pod,
		"podNamespace", req.Namespace,
		"namespace", name,
		"server", m.server,
		"pipeDirectory", pipeDir,
		"activeThreadPercent", req.ActiveThreadPercent,
		"gpuUuids", req.GPUUUIDs,
		"created", fresh,
	)
	return lease, nil
}

// Release gives up the container's namespace.
//
// It reports deleted=false without an error when the daemon will not delete the
// namespace yet. That is the expected case rather than a failure: a namespace
// with an attached client cannot be deleted, and the container's client is
// often still attached when its teardown reaches here. The lease is marked and
// Sweep retries until it goes, so a refusal costs a delay and never a leak.
//
// Releasing a container that holds no namespace is a no-op, so a teardown path
// can call it unconditionally.
func (m *Manager) Release(ctx context.Context, containerID string) (deleted bool, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.ensureLoaded(); err != nil {
		m.log.Warn("could not load the MPS namespace state; continuing from an empty set", "error", err)
	}

	lease, known := m.leases[containerID]
	if !known {
		return false, nil
	}

	if err := m.control.DeleteNamespace(ctx, lease.Server, lease.Namespace); err != nil {
		now := m.now()
		lease.ReleasedAt = &now
		m.leases[containerID] = lease
		m.persist()
		m.log.Info("MPS namespace could not be deleted yet; it will be retried",
			"containerId", containerID, "namespace", lease.Namespace, "error", err)
		return false, nil
	}

	delete(m.leases, containerID)
	m.persist()
	m.log.Info("released MPS namespace", "containerId", containerID, "namespace", lease.Namespace)
	return true, nil
}

// Reconcile brings the control daemon and the lease set back into agreement.
// It must run once the daemon is up and again after every restart of it, and it
// does four things, in this order:
//
//  1. Caps the server's `default` namespace, so the one namespace nothing is
//     routed to cannot hand out the whole card if something reaches it anyway.
//  2. Prunes leases whose container is gone (and whose grace period has
//     elapsed), deleting their namespaces. This is what stops an mpsd restart —
//     during which fractiond's releases go nowhere — from stranding namespaces
//     for dead containers.
//  3. Recreates namespaces for surviving leases that the daemon does not have.
//     An MPS restart destroys every namespace while the containers using them
//     keep running; recreating them at the same names restores the same pipe
//     directories those containers are already bind-mounted to.
//  4. Deletes namespaces carrying this package's prefix that no lease claims.
//     Only that prefix — a namespace created by anything else is never a
//     candidate, whatever the list output looked like.
//
// Every step is best-effort and independent: a failure is logged and the rest
// still runs, because a half-reconciled daemon is strictly better than one that
// stopped at the first error.
func (m *Manager) Reconcile(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var problems []error
	if err := m.ensureLoaded(); err != nil {
		problems = append(problems, err)
	}

	if m.defaultNamespaceATP > 0 {
		if err := m.control.SetNamespaceActiveThreadPercentage(ctx, m.server, DefaultNamespace, m.defaultNamespaceATP); err != nil {
			problems = append(problems, fmt.Errorf("capping the %q namespace: %w", DefaultNamespace, err))
		} else {
			m.log.Info("capped the default MPS namespace as defence in depth",
				"server", m.server, "namespace", DefaultNamespace, "activeThreadPercent", m.defaultNamespaceATP)
		}
	}

	m.pruneDeadLeases(ctx)

	existing, err := m.control.NamespaceNames(ctx, m.server)
	if err != nil {
		problems = append(problems, fmt.Errorf("listing MPS namespaces: %w", err))
		m.persist()
		return errors.Join(problems...)
	}
	present := make(map[string]struct{}, len(existing))
	for _, name := range existing {
		present[name] = struct{}{}
	}

	// Restore the namespaces live leases expect.
	claimed := make(map[string]struct{}, len(m.leases))
	for id, lease := range m.leases {
		claimed[lease.Namespace] = struct{}{}
		if lease.released() {
			continue
		}
		if _, ok := present[lease.Namespace]; ok {
			continue
		}
		if err := m.restore(ctx, lease); err != nil {
			problems = append(problems, err)
			continue
		}
		m.log.Info("recreated an MPS namespace the control daemon had lost",
			"containerId", id, "namespace", lease.Namespace, "activeThreadPercent", lease.ActiveThreadPercent)
	}

	// Delete the ones nothing claims.
	for _, name := range existing {
		if !IsManagedName(name) {
			continue
		}
		if _, ok := claimed[name]; ok {
			continue
		}
		if err := m.control.DeleteNamespace(ctx, m.server, name); err != nil {
			problems = append(problems, fmt.Errorf("deleting orphaned MPS namespace %q: %w", name, err))
			continue
		}
		m.log.Info("deleted an orphaned MPS namespace", "namespace", name, "server", m.server)
	}

	m.persist()
	return errors.Join(problems...)
}

// Sweep retries the deletes that were refused because a client was still
// attached, and prunes leases whose container has gone away without a release
// (fractiond crashing between a container's teardown and its release call is
// the case that matters).
func (m *Manager) Sweep(ctx context.Context) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.ensureLoaded(); err != nil {
		m.log.Warn("could not load the MPS namespace state; skipping the sweep", "error", err)
		return
	}
	m.pruneDeadLeases(ctx)
	m.persist()
}

// SweepEvery runs Sweep on a ticker until ctx is cancelled.
func (m *Manager) SweepEvery(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = DefaultSweepInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.Sweep(ctx)
		}
	}
}

// Leases returns a copy of the current lease set, for diagnostics and tests.
func (m *Manager) Leases() []Lease {
	m.mu.Lock()
	defer m.mu.Unlock()
	_ = m.ensureLoaded()
	out := make([]Lease, 0, len(m.leases))
	for _, lease := range m.leases {
		out = append(out, lease)
	}
	slices.SortFunc(out, func(a, b Lease) int { return strings.Compare(a.ContainerID, b.ContainerID) })
	return out
}

// pruneDeadLeases deletes the namespaces of leases that were released, or whose
// container no longer exists. Callers hold m.mu.
func (m *Manager) pruneDeadLeases(ctx context.Context) {
	now := m.now()
	for id, lease := range m.leases {
		if !lease.released() && !m.isOrphan(id, lease, now) {
			continue
		}
		if err := m.control.DeleteNamespace(ctx, lease.Server, lease.Namespace); err != nil {
			if !lease.released() {
				// Mark it so the reason it is still here is recorded, and so a
				// later sweep retries it without re-deciding liveness.
				released := now
				lease.ReleasedAt = &released
				m.leases[id] = lease
			}
			m.log.Debug("MPS namespace is still in use; deferring its deletion",
				"containerId", id, "namespace", lease.Namespace, "error", err)
			continue
		}
		delete(m.leases, id)
		m.log.Info("reclaimed the MPS namespace of a container that is gone",
			"containerId", id, "namespace", lease.Namespace)
	}
}

// isOrphan reports whether the lease's container has gone away. Callers hold
// m.mu.
//
// Liveness is only consulted after the grace period, because "the container has
// no processes" is also what a container looks like between being created and
// starting — and provisioning happens strictly inside that window.
func (m *Manager) isOrphan(containerID string, lease Lease, now time.Time) bool {
	if m.live == nil {
		return false
	}
	if now.Sub(lease.CreatedAt) < m.orphanGrace {
		return false
	}
	live, err := m.live(containerID)
	if err != nil {
		// Unknown is not dead. Deleting a namespace out from under a running
		// container costs it its GPU; keeping one costs a name.
		m.log.Warn("could not determine whether a container is still running; keeping its MPS namespace",
			"containerId", containerID, "error", err)
		return false
	}
	return !live
}

// restore recreates a namespace the daemon has lost and re-applies its cap.
// Callers hold m.mu.
func (m *Manager) restore(ctx context.Context, lease Lease) error {
	if err := m.control.CreateNamespace(ctx, lease.Server, lease.Namespace); err != nil {
		return fmt.Errorf("recreating MPS namespace %q for container %q: %w", lease.Namespace, lease.ContainerID, err)
	}
	if err := m.control.SetNamespaceActiveThreadPercentage(ctx, lease.Server, lease.Namespace, lease.ActiveThreadPercent); err != nil {
		return fmt.Errorf("re-capping MPS namespace %q for container %q: %w", lease.Namespace, lease.ContainerID, err)
	}
	return nil
}

// deleteNamespace drops a namespace created moments ago on a path that then
// failed, so a rejected provision does not leave one behind. Callers hold m.mu.
func (m *Manager) deleteNamespace(ctx context.Context, name string) {
	if err := m.control.DeleteNamespace(ctx, m.server, name); err != nil {
		m.log.Warn("could not remove a half-provisioned MPS namespace", "namespace", name, "error", err)
	}
}

// ensureLoaded reads the persisted lease set once. Callers hold m.mu.
func (m *Manager) ensureLoaded() error {
	if m.loaded {
		return nil
	}
	m.loaded = true
	leases, err := m.store.load()
	if err != nil {
		return err
	}
	for _, lease := range leases {
		m.leases[lease.ContainerID] = lease
	}
	return nil
}

// persist writes the lease set back. Callers hold m.mu.
//
// A failure is logged, not returned: the namespaces themselves are already
// correct on the daemon, and the only thing lost is the ability to reconcile
// them after a restart. Failing a container's provisioning over it would turn a
// bookkeeping problem into an outage.
func (m *Manager) persist() {
	leases := make([]Lease, 0, len(m.leases))
	for _, lease := range m.leases {
		leases = append(leases, lease)
	}
	if err := m.store.save(leases); err != nil {
		m.log.Error("could not persist the MPS namespace state; namespaces will not be reconciled after an mpsd restart",
			"path", m.store.path, "error", err)
	}
}
