// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package drain serves mpsd's MPS client-drain endpoint.
//
// An MPS client killed while it still has GPU work in flight orphans that work.
// Every other client on the same GPU then fails with cudaErrorIllegalAddress —
// a hardware boundary that MPS namespacing does not contain — and if nothing
// consumes the resulting fault (the common case when the dying client was the
// GPU's only tenant), every client that connects afterwards hangs forever
// inside CUDA init. The control plane reports healthy throughout: the server
// lists as Ready, the client lists as attached, and nothing is logged. Only
// restarting MPS clears it. NVIDIA has confirmed this as a known limitation
// with a fix planned for CUDA 13.6 / r625.
//
// Everything here is a workaround for that, in two halves:
//
//   - Prevention. NVIDIA's supported workaround is to terminate the client
//     through the control daemon first, which blocks new submissions and drains
//     the outstanding work, and only then kill the process. This endpoint does
//     that automatically: fractiond calls it from its NRI StopContainer hook, so
//     the drain happens before the runtime stops a fractional container, with no
//     cooperation required from the workload.
//
//   - Recovery. A terminate that does not return means the server is already
//     wedged, which escalates to an MPS restart. And because a drained node is
//     the cheapest possible moment to restart MPS, the last client leaving
//     optionally recycles it too — so a fault picked up during one run of a
//     benchmark sweep cannot survive into the next.
package drain

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/mpsdrain"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/procfs"
)

const (
	// Path and DefaultSocketPath are the endpoint's wire contract, shared with
	// the fractiond side that calls it.
	Path              = mpsdrain.Path
	DefaultSocketPath = mpsdrain.DefaultSocketPath

	// DefaultClientDrainTimeout bounds one client's terminate-and-drain. It is
	// generous on purpose: the call is supposed to block while real GPU work
	// finishes, and cutting it short would kill the client mid-kernel — the very
	// thing being avoided. Exceeding it is treated as a wedged server.
	DefaultClientDrainTimeout = 30 * time.Second

	// socketPerm keeps the endpoint root-only. Both daemons run as root, and
	// anything else reaching it could terminate other tenants' GPU work.
	socketPerm = 0o600

	dirPerm = 0o755
)

// Request and Response are the endpoint's wire types. They are defined in the
// shared package so fractiond, which lives outside this module path, can use
// exactly the same shapes.
type (
	Request  = mpsdrain.Request
	Response = mpsdrain.Response
)

// MPSControl is the slice of the control daemon this package needs.
type MPSControl interface {
	ClientPIDs(ctx context.Context) ([]int, error)
	TerminateClient(ctx context.Context, pid int, timeout time.Duration) error
}

// Restarter restarts the supervised MPS control daemon. *internal.Supervisor
// implements it.
type Restarter interface {
	Restart(reason string, graceful bool) error
}

// PIDLookup returns the host PIDs belonging to a container.
type PIDLookup func(procRoot, containerID string) ([]int, error)

// Options configures a Server.
type Options struct {
	// SocketPath is the unix socket to listen on. Empty uses DefaultSocketPath.
	SocketPath string
	// Control talks to the MPS control daemon. Required.
	Control MPSControl
	// Restarter restarts MPS on a wedge or an idle recycle. Optional: without
	// one, the endpoint still drains, it just cannot recover.
	Restarter Restarter
	// ProcRoot is the procfs mount used to map a container to its PIDs.
	ProcRoot string
	// ClientDrainTimeout bounds one client's drain. Zero uses the default.
	ClientDrainTimeout time.Duration
	// RecycleWhenIdle restarts MPS once a drain leaves no clients attached.
	RecycleWhenIdle bool
	// LookupPIDs overrides container→PID resolution (tests).
	LookupPIDs PIDLookup
	// Log defaults to slog.Default().
	Log *slog.Logger
}

// Server serves the drain endpoint.
type Server struct {
	socketPath         string
	control            MPSControl
	restarter          Restarter
	procRoot           string
	clientDrainTimeout time.Duration
	recycleWhenIdle    bool
	lookupPIDs         PIDLookup
	log                *slog.Logger

	// mu guards inFlight. running tracks the background drains so shutdown can
	// wait for them.
	mu       sync.Mutex
	inFlight map[string]*drainCall
	running  sync.WaitGroup
}

// drainCall is one background drain. Its response is written before done is
// closed, so every waiter that observes the close also observes the response.
type drainCall struct {
	done chan struct{}
	resp Response
}

// New builds a Server from opts.
func New(opts Options) *Server {
	socketPath := opts.SocketPath
	if socketPath == "" {
		socketPath = DefaultSocketPath
	}
	procRoot := opts.ProcRoot
	if procRoot == "" {
		procRoot = procfs.DefaultRoot
	}
	timeout := opts.ClientDrainTimeout
	if timeout <= 0 {
		timeout = DefaultClientDrainTimeout
	}
	lookup := opts.LookupPIDs
	if lookup == nil {
		lookup = procfs.PIDsInContainer
	}
	log := opts.Log
	if log == nil {
		log = slog.Default()
	}

	return &Server{
		socketPath:         socketPath,
		control:            opts.Control,
		restarter:          opts.Restarter,
		procRoot:           procRoot,
		clientDrainTimeout: timeout,
		recycleWhenIdle:    opts.RecycleWhenIdle,
		lookupPIDs:         lookup,
		log:                log,
		inFlight:           map[string]*drainCall{},
	}
}

// Serve listens on the unix socket and serves until ctx is cancelled. The
// socket file is removed on the way in (a previous mpsd may have left one) and
// on the way out.
func (s *Server) Serve(ctx context.Context) error {
	if err := os.MkdirAll(filepath.Dir(s.socketPath), dirPerm); err != nil {
		return fmt.Errorf("creating drain socket directory: %w", err)
	}
	// A leftover socket file from a crashed mpsd would make Listen fail with
	// "address already in use" even though nothing is listening on it.
	if err := os.Remove(s.socketPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("removing stale drain socket: %w", err)
	}

	listener, err := net.Listen("unix", s.socketPath)
	if err != nil {
		return fmt.Errorf("listening on drain socket %q: %w", s.socketPath, err)
	}
	if err := os.Chmod(s.socketPath, socketPerm); err != nil {
		_ = listener.Close()
		return fmt.Errorf("setting drain socket permissions: %w", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc(Path, s.handle)
	server := &http.Server{
		Handler: mux,
		// The handler's own work is bounded by the per-client drain timeout, so
		// these only guard against a caller that stalls mid-request.
		ReadHeaderTimeout: 5 * time.Second,
	}

	// serveDone stops the shutdown watcher from outliving Serve when Serve
	// returns for some reason other than a cancelled context.
	serveDone := make(chan struct{})
	defer close(serveDone)

	go func() {
		select {
		case <-ctx.Done():
		case <-serveDone:
			return
		}
		// The drain calls in flight are the interesting part of a shutdown: a
		// container being stopped right now still wants its clients drained, so
		// close gracefully and let them finish.
		shutdownCtx, cancel := context.WithTimeout(context.Background(), s.clientDrainTimeout)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			s.log.Warn("drain server shutdown", "error", err)
		}
	}()

	s.log.Info("serving MPS drain endpoint",
		"socket", s.socketPath,
		"clientDrainTimeout", s.clientDrainTimeout,
		"recycleWhenIdle", s.recycleWhenIdle,
	)

	defer func() {
		if err := os.Remove(s.socketPath); err != nil && !os.IsNotExist(err) {
			s.log.Warn("failed to remove drain socket", "path", s.socketPath, "error", err)
		}
	}()
	// Drains outlive the request that started them (see Drain), so waiting for
	// the HTTP server alone is not enough: exiting with a terminate in flight
	// would abandon a half-drained client, which is the wedge this whole
	// endpoint exists to avoid.
	defer s.waitForDrains(s.clientDrainTimeout)

	if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("serving drain endpoint: %w", err)
	}
	return nil
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "only POST is supported", http.StatusMethodNotAllowed)
		return
	}

	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decoding drain request: %v", err), http.StatusBadRequest)
		return
	}
	if req.ContainerID == "" {
		http.Error(w, "containerId is required", http.StatusBadRequest)
		return
	}

	response := s.Drain(r.Context(), req)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		s.log.Warn("failed to write drain response", "error", err)
	}
}

// Drain terminates the container's MPS clients and, when warranted, restarts
// MPS. It is the testable core of the endpoint.
//
// A failure to drain is reported in the response rather than returned as an
// error: the caller is about to stop a container either way, and blocking that
// on a drain problem would turn a degraded GPU into a stuck pod.
//
// ctx says how long the *caller* is prepared to wait, and nothing more. The
// termination itself runs on a context of its own and finishes even if the
// caller gives up first. That asymmetry is the point: the caller is an NRI
// StopContainer hook with a ~2s runtime deadline it must not overrun (see
// mpsdrain.DefaultCallTimeout), while a real drain blocks for as long as the
// client's outstanding GPU work takes. Cancelling the terminate when the hook
// times out would abandon the client mid-kernel — the exact failure this
// endpoint exists to prevent, caused by us. Issuing the terminate is what stops
// new submissions, and the kubelet's termination grace period still has to
// elapse before anything kills the process, so there is room to finish.
func (s *Server) Drain(ctx context.Context, req Request) Response {
	// A Server built without a Control cannot drain anything. Options.Control is
	// documented as required and the production wiring always sets it, but this
	// runs on the container-stop path: a panic here would take mpsd down every
	// time a fractional pod is deleted, which is a far worse failure than the
	// misconfiguration it would be reporting.
	if s.control == nil {
		s.log.Error("MPS drain endpoint has no control daemon client; stopping container without draining",
			"containerId", req.ContainerID)
		return Response{Message: "drain endpoint is not configured with an MPS control client"}
	}

	call := s.beginDrain(req)

	select {
	case <-call.done:
		return call.resp
	case <-ctx.Done():
		s.log.Info("drain caller gave up; the drain continues in the background",
			"containerId", req.ContainerID, "error", ctx.Err())
		return Response{Message: fmt.Sprintf("drain of container %s continues in the background: %v", req.ContainerID, ctx.Err())}
	}
}

// beginDrain returns the background drain for this container, starting one if
// none is running.
//
// Concurrent calls for the same container share one drain. The caller's own
// timeout is now much shorter than a drain can take, so a retry — or the
// retroactive stop fractiond performs after an NRI reconnect — can easily
// arrive while the first drain is still in flight. Issuing `client terminate`
// twice for the same PID would fail the second time, and a failed terminate is
// read as a wedged server: mpsd would restart MPS, killing every other tenant's
// GPU work on the node, because a drain succeeded twice.
func (s *Server) beginDrain(req Request) *drainCall {
	s.mu.Lock()
	if call, running := s.inFlight[req.ContainerID]; running {
		s.mu.Unlock()
		s.log.Debug("joining an MPS drain already in flight", "containerId", req.ContainerID)
		return call
	}
	call := &drainCall{done: make(chan struct{})}
	s.inFlight[req.ContainerID] = call
	s.running.Add(1)
	s.mu.Unlock()

	go func() {
		defer s.running.Done()
		defer func() {
			s.mu.Lock()
			delete(s.inFlight, req.ContainerID)
			s.mu.Unlock()
			close(call.done)
		}()
		// Deliberately not derived from the caller's context. Every step is
		// bounded on its own (see runDrain), so this cannot run forever, and
		// Serve waits for it before the process exits.
		call.resp = s.runDrain(context.Background(), req)
	}()

	return call
}

// waitForDrains blocks until the background drains finish, giving up after
// timeout so a control daemon that never answers cannot stop mpsd from exiting.
func (s *Server) waitForDrains(timeout time.Duration) {
	finished := make(chan struct{})
	go func() {
		s.running.Wait()
		close(finished)
	}()

	select {
	case <-finished:
	case <-time.After(timeout):
		s.log.Warn("gave up waiting for in-flight MPS drains", "timeout", timeout)
	}
}

// runDrain is the drain itself. Each step gets its own deadline rather than
// sharing one budget: a container can hold several MPS clients and each is
// entitled to the full drain window, so a shared budget would cut the second
// client short and report a wedge that is not there.
func (s *Server) runDrain(ctx context.Context, req Request) Response {
	log := s.log.With("containerId", req.ContainerID, "pod", req.Pod, "namespace", req.Namespace)

	attached, err := s.listClients(ctx)
	if err != nil {
		log.Warn("cannot list MPS clients; stopping container without draining", "error", err)
		return Response{Message: fmt.Sprintf("listing MPS clients: %v", err)}
	}
	if len(attached) == 0 {
		log.Debug("no MPS clients attached; nothing to drain")
		return Response{}
	}

	containerPIDs, err := s.lookupPIDs(s.procRoot, req.ContainerID)
	if err != nil {
		log.Warn("cannot resolve container PIDs; stopping container without draining", "error", err)
		return Response{Message: fmt.Sprintf("resolving container PIDs: %v", err)}
	}

	targets := intersect(attached, containerPIDs)
	if len(targets) == 0 {
		// Normal for a container that never opened a CUDA context — a sidecar,
		// an init container, or a worker that failed before reaching the GPU.
		log.Debug("container has no MPS clients; nothing to drain", "attachedClients", len(attached))
		return Response{}
	}

	response := Response{}
	for _, pid := range targets {
		log.Info("draining MPS client before container stop", "pid", pid)
		if err := s.terminate(ctx, pid); err != nil {
			// Not being able to terminate is exactly the wedge signature: the
			// command blocks until the client's work drains, so a timeout means
			// work that will never drain.
			log.Error("failed to drain MPS client; treating the MPS server as wedged",
				"pid", pid, "error", err)
			response.Wedged = true
			response.Message = fmt.Sprintf("draining client %d: %v", pid, err)
			break
		}
		response.Drained = append(response.Drained, pid)
	}

	if response.Wedged {
		response.Restarted = s.restart(fmt.Sprintf("MPS client drain for container %s did not complete", req.ContainerID), false)
		return response
	}

	if s.recycleWhenIdle {
		remaining, err := s.listClients(ctx)
		if err != nil {
			log.Warn("cannot re-check MPS clients; skipping idle recycle", "error", err)
			return response
		}
		if len(remaining) == 0 {
			// Nothing is attached, so this restart costs nobody anything — and
			// it is the only way to be sure a fault left by an earlier client
			// does not hang the next one.
			response.Restarted = s.restart("no MPS clients remain after drain", true)
		}
	}

	return response
}

// listClients lists the attached MPS clients under its own deadline, so a
// control daemon that never answers cannot hold the drain goroutine open.
func (s *Server) listClients(ctx context.Context) ([]int, error) {
	ctx, cancel := context.WithTimeout(ctx, s.clientDrainTimeout)
	defer cancel()
	return s.control.ClientPIDs(ctx)
}

// terminate drains one client. The timeout is passed twice on purpose: as the
// argument the control wrapper uses to bound the command, and as a context
// deadline so an implementation that ignores the argument is still bounded.
func (s *Server) terminate(ctx context.Context, pid int) error {
	ctx, cancel := context.WithTimeout(ctx, s.clientDrainTimeout)
	defer cancel()
	return s.control.TerminateClient(ctx, pid, s.clientDrainTimeout)
}

// restart asks the Restarter to bounce MPS, reporting whether it was able to.
func (s *Server) restart(reason string, graceful bool) bool {
	if s.restarter == nil {
		s.log.Warn("MPS restart requested but no restarter is configured", "reason", reason)
		return false
	}
	if err := s.restarter.Restart(reason, graceful); err != nil {
		s.log.Error("failed to restart MPS", "reason", reason, "error", err)
		return false
	}
	return true
}

// intersect returns the members of attached that also appear in containerPIDs,
// in attached order.
func intersect(attached, containerPIDs []int) []int {
	if len(attached) == 0 || len(containerPIDs) == 0 {
		return nil
	}

	inContainer := make(map[int]struct{}, len(containerPIDs))
	for _, pid := range containerPIDs {
		inContainer[pid] = struct{}{}
	}

	var targets []int
	for _, pid := range attached {
		if _, ok := inContainer[pid]; ok {
			targets = append(targets, pid)
		}
	}
	return targets
}
