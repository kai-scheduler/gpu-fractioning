// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package drain

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/common/mpsdrain"
	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/mpsns"
)

// serveOnSocket starts a Server on a socket in t.TempDir() and returns its
// path. The server is stopped and joined by t.Cleanup, so a test that leaves it
// running fails loudly instead of leaking into the next one.
func serveOnSocket(t *testing.T, opts Options) (string, func()) {
	t.Helper()

	socketPath := filepath.Join(t.TempDir(), "mpsd.sock")
	opts.SocketPath = socketPath
	if opts.Log == nil {
		opts.Log = quietLogger()
	}
	s := New(opts)

	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- s.Serve(ctx) }()

	waitForSocket(t, socketPath)

	stopped := false
	stop := func() {
		if stopped {
			return
		}
		stopped = true
		cancel()
		select {
		case err := <-served:
			if err != nil {
				t.Errorf("Serve() returned error: %v", err)
			}
		case <-time.After(10 * time.Second):
			t.Error("Serve() did not return within 10s of cancellation")
		}
	}
	t.Cleanup(stop)

	return socketPath, stop
}

func waitForSocket(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("drain socket %q never appeared", path)
}

// rawClient speaks HTTP over the endpoint's unix socket without going through
// mpsdrain.Client, so a test can send requests the real client would never
// produce.
func rawClient(t *testing.T, socketPath string) *http.Client {
	t.Helper()
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
		},
	}
	t.Cleanup(transport.CloseIdleConnections)
	return &http.Client{Transport: transport}
}

func TestServeEndToEndWithTheRealClient(t *testing.T) {
	control := &fakeControl{listResults: []listResult{{pids: []int{100, 200}}, {}}}
	restarter := &fakeRestarter{}
	socketPath, _ := serveOnSocket(t, Options{
		Control:            control,
		Restarter:          restarter,
		LookupPIDs:         staticLookup([]int{200}, nil),
		ClientDrainTimeout: 2 * time.Second,
	})

	client := mpsdrain.NewClient(socketPath, 5*time.Second, quietLogger())
	if client.SocketPath() != socketPath {
		t.Errorf("SocketPath() = %q, want %q", client.SocketPath(), socketPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resp, err := client.Drain(ctx, mpsdrain.Request{ContainerID: testContainerID, Pod: "trainer-0", Namespace: "team-a"})
	if err != nil {
		t.Fatalf("Drain() error = %v", err)
	}
	if !slices.Equal(resp.Drained, []int{200}) {
		t.Errorf("Drained = %v, want [200] — only the stopping container's client", resp.Drained)
	}
	if resp.Wedged || resp.Restarted {
		t.Errorf("response = %+v, want a clean drain", resp)
	}
	if terminated := control.terminatedPIDs(); !slices.Equal(terminated, []int{200}) {
		t.Errorf("terminated = %v, want [200]", terminated)
	}
}

func TestServeSurfacesAWedgeToTheRealClient(t *testing.T) {
	// The wedge flags are the only signal fractiond has that the node's GPU is
	// in trouble; they have to survive the round trip.
	control := &fakeControl{
		listResults:  []listResult{{pids: []int{100}}},
		terminateErr: map[int]error{100: context.DeadlineExceeded},
	}
	socketPath, _ := serveOnSocket(t, Options{
		Control:            control,
		Restarter:          &fakeRestarter{},
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: time.Second,
	})

	resp, err := mpsdrain.NewClient(socketPath, 5*time.Second, quietLogger()).
		Drain(context.Background(), mpsdrain.Request{ContainerID: testContainerID})
	if err != nil {
		t.Fatalf("Drain() error = %v", err)
	}
	if !resp.Wedged || !resp.Restarted {
		t.Errorf("response = %+v, want Wedged and Restarted", resp)
	}
	if !strings.Contains(resp.Message, "draining client 100") {
		t.Errorf("Message = %q, want it to name the client that would not drain", resp.Message)
	}
}

func TestServeRejectsBadRequests(t *testing.T) {
	socketPath, _ := serveOnSocket(t, Options{
		Control:    &fakeControl{},
		LookupPIDs: staticLookup(nil, nil),
	})
	client := rawClient(t, socketPath)

	tests := []struct {
		name     string
		method   string
		body     string
		wantCode int
		wantBody string
	}{
		{
			// A GET reaching this socket is not fractiond; answering it would
			// let a probe terminate GPU work.
			name:   "GET is rejected",
			method: http.MethodGet, body: "",
			wantCode: http.StatusMethodNotAllowed, wantBody: "only POST is supported",
		},
		{name: "PUT is rejected", method: http.MethodPut, body: "{}", wantCode: http.StatusMethodNotAllowed},
		{name: "DELETE is rejected", method: http.MethodDelete, body: "{}", wantCode: http.StatusMethodNotAllowed},
		{
			name:   "malformed JSON is a bad request, not a panic",
			method: http.MethodPost, body: "{not json",
			wantCode: http.StatusBadRequest, wantBody: "decoding drain request",
		},
		{
			name:   "empty body is a bad request",
			method: http.MethodPost, body: "",
			wantCode: http.StatusBadRequest, wantBody: "decoding drain request",
		},
		{
			name:   "a JSON array is a bad request",
			method: http.MethodPost, body: `["containerId"]`,
			wantCode: http.StatusBadRequest, wantBody: "decoding drain request",
		},
		{
			// Without an id there is nothing to intersect against, and a blank
			// id must never be allowed to reach the procfs lookup.
			name:   "missing containerId is rejected",
			method: http.MethodPost, body: `{"pod":"p"}`,
			wantCode: http.StatusBadRequest, wantBody: "containerId is required",
		},
		{
			name:   "explicitly empty containerId is rejected",
			method: http.MethodPost, body: `{"containerId":""}`,
			wantCode: http.StatusBadRequest, wantBody: "containerId is required",
		},
		{
			// Wrong types must not become a 500 or a panic in the handler.
			name:   "containerId of the wrong type is a bad request",
			method: http.MethodPost, body: `{"containerId":12345}`,
			wantCode: http.StatusBadRequest, wantBody: "decoding drain request",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := http.NewRequest(tt.method, "http://mpsd"+Path, strings.NewReader(tt.body))
			if err != nil {
				t.Fatal(err)
			}
			resp, err := client.Do(req)
			if err != nil {
				t.Fatalf("request error = %v", err)
			}
			defer func() { _ = resp.Body.Close() }()

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != tt.wantCode {
				t.Errorf("status = %d, want %d (body %q)", resp.StatusCode, tt.wantCode, body)
			}
			if tt.wantBody != "" && !strings.Contains(string(body), tt.wantBody) {
				t.Errorf("body = %q, want it to contain %q", body, tt.wantBody)
			}
		})
	}
}

func TestServeStillDrainsWhenTheCallerTimesOut(t *testing.T) {
	// End-to-end version of the NRI deadline problem: fractiond gives up after
	// its short timeout, and the drain still has to finish rather than leaving
	// the client half-terminated.
	release := make(chan struct{})
	terminated := make(chan int, 1)
	control := &fakeControl{
		listResults: []listResult{{pids: []int{100}}},
		terminateFunc: func(ctx context.Context, pid int) error {
			select {
			case <-release:
				terminated <- pid
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		},
	}
	socketPath, stop := serveOnSocket(t, Options{
		Control:            control,
		LookupPIDs:         staticLookup([]int{100}, nil),
		ClientDrainTimeout: 5 * time.Second,
	})

	client := mpsdrain.NewClient(socketPath, 50*time.Millisecond, quietLogger())

	start := time.Now()
	_, err := client.Drain(context.Background(), mpsdrain.Request{ContainerID: testContainerID})
	if err == nil {
		t.Fatal("Drain() error = nil, want the caller's own timeout")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error = %v, want a deadline error the caller can recognise", err)
	}
	// The caller must be released on its own schedule, not the drain's.
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("the caller was held for %v, far past its 50ms timeout", elapsed)
	}

	close(release)
	select {
	case pid := <-terminated:
		if pid != 100 {
			t.Errorf("terminated %d, want 100", pid)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the drain was abandoned when the caller timed out")
	}

	stop()
}

func TestServeShutdownIsClean(t *testing.T) {
	// A leftover socket file makes the next mpsd fail to bind ("address already
	// in use") even though nothing is listening, and a leaked listener fd would
	// do the same.
	before := runtime.NumGoroutine()

	socketPath, stop := serveOnSocket(t, Options{
		Control:            &fakeControl{},
		LookupPIDs:         staticLookup(nil, nil),
		ClientDrainTimeout: 200 * time.Millisecond,
	})

	resp, err := mpsdrain.NewClient(socketPath, 2*time.Second, quietLogger()).
		Drain(context.Background(), mpsdrain.Request{ContainerID: testContainerID})
	if err != nil {
		t.Fatalf("Drain() error = %v", err)
	}
	if resp.Wedged || resp.Restarted || len(resp.Drained) != 0 {
		t.Errorf("response = %+v, want an empty drain", resp)
	}

	stop()

	if _, err := os.Stat(socketPath); !os.IsNotExist(err) {
		t.Errorf("socket %q still exists after shutdown (stat err = %v)", socketPath, err)
	}
	// Binding the same path again proves the listener fd really went away.
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("cannot rebind %q after shutdown: %v", socketPath, err)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}

	assertGoroutinesSettle(t, before)
}

func TestServeRemovesAStaleSocketOnStartup(t *testing.T) {
	// mpsd crashing leaves the socket file behind; refusing to start because of
	// it would mean a node that never recovers.
	dir := t.TempDir()
	socketPath := filepath.Join(dir, "mpsd.sock")
	if err := os.WriteFile(socketPath, []byte("stale"), 0o600); err != nil {
		t.Fatal(err)
	}

	s := New(Options{
		SocketPath:         socketPath,
		Control:            &fakeControl{},
		LookupPIDs:         staticLookup(nil, nil),
		ClientDrainTimeout: 200 * time.Millisecond,
		Log:                quietLogger(),
	})

	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- s.Serve(ctx) }()
	waitForSocket(t, socketPath)
	cancel()

	select {
	case err := <-served:
		if err != nil {
			t.Errorf("Serve() error = %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Serve() did not return")
	}
}

func TestServeSocketIsRootOnly(t *testing.T) {
	// Anything that can reach this socket can terminate other tenants' GPU
	// work, so the permissions are part of the security boundary.
	socketPath, _ := serveOnSocket(t, Options{
		Control:            &fakeControl{},
		LookupPIDs:         staticLookup(nil, nil),
		ClientDrainTimeout: 200 * time.Millisecond,
	})

	info, err := os.Stat(socketPath)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != socketPerm {
		t.Errorf("socket mode = %#o, want %#o", perm, socketPerm)
	}
}

func TestServeFailsWhenTheSocketCannotBeCreated(t *testing.T) {
	// A misconfigured mount must fail loudly at startup rather than leave mpsd
	// running with an endpoint nobody can reach.
	file := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	s := New(Options{
		SocketPath: filepath.Join(file, "sub", "mpsd.sock"),
		Control:    &fakeControl{},
		LookupPIDs: staticLookup(nil, nil),
		Log:        quietLogger(),
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.Serve(ctx); err == nil {
		t.Error("Serve() error = nil, want a failure for an unusable socket path")
	}
}

func TestClientAgainstAServerThatIsNotThere(t *testing.T) {
	// mpsd not running yet (or its socket not mounted) must be an immediate,
	// legible error rather than a hang on the container-stop path.
	missing := filepath.Join(t.TempDir(), "absent.sock")

	_, err := mpsdrain.NewClient(missing, time.Second, quietLogger()).
		Drain(context.Background(), mpsdrain.Request{ContainerID: testContainerID})
	if err == nil {
		t.Fatal("Drain() error = nil, want a dial failure")
	}
	if !strings.Contains(err.Error(), missing) {
		t.Errorf("error %v does not name the socket it failed to reach", err)
	}
}

func TestRequestBodyUsesTheAgreedFieldNames(t *testing.T) {
	// fractiond and mpsd are separate binaries in separate images, so the JSON
	// field names are a wire contract: renaming containerId to container_id on
	// one side turns every drain into "containerId is required".
	bodies := make(chan []byte, 1)
	socketPath := filepath.Join(t.TempDir(), "mpsd.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{
		ReadHeaderTimeout: time.Second,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			bodies <- body
			_, _ = w.Write([]byte(`{"drained":[7],"wedged":true,"restarted":true,"message":"hi"}`))
		}),
	}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })

	resp, err := mpsdrain.NewClient(socketPath, 2*time.Second, quietLogger()).
		Drain(context.Background(), mpsdrain.Request{ContainerID: testContainerID, Pod: "p", Namespace: "ns"})
	if err != nil {
		t.Fatalf("Drain() error = %v", err)
	}

	body := <-bodies
	for _, field := range []string{`"containerId":"` + testContainerID + `"`, `"pod":"p"`, `"namespace":"ns"`} {
		if !bytes.Contains(body, []byte(field)) {
			t.Errorf("request body %s is missing %s", body, field)
		}
	}
	if !slices.Equal(resp.Drained, []int{7}) || !resp.Wedged || !resp.Restarted || resp.Message != "hi" {
		t.Errorf("response = %+v, want every field decoded", resp)
	}
}

// assertGoroutinesSettle waits for the goroutine count to come back to roughly
// where it started. A drain server that leaks a goroutine per shutdown would
// accumulate them for the life of the node.
func assertGoroutinesSettle(t *testing.T, before int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= before+2 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	buf := make([]byte, 1<<16)
	buf = buf[:runtime.Stack(buf, true)]
	t.Errorf("goroutines went from %d to %d and did not settle:\n%s", before, runtime.NumGoroutine(), buf)
}

// ── namespace provisioning over the same socket ─────────────────────────────

// fakeNamespaces stands in for the namespace manager at the endpoint's seam.
type fakeNamespaces struct {
	mu sync.Mutex

	lease        mpsns.Lease
	provisionErr error
	releaseErr   error
	deleted      bool

	provisioned []mpsns.Request
	released    []string
}

func (f *fakeNamespaces) Provision(_ context.Context, req mpsns.Request) (mpsns.Lease, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.provisioned = append(f.provisioned, req)
	if f.provisionErr != nil {
		return mpsns.Lease{}, f.provisionErr
	}
	return f.lease, nil
}

func (f *fakeNamespaces) Release(_ context.Context, containerID string) (bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.released = append(f.released, containerID)
	return f.deleted, f.releaseErr
}

func (f *fakeNamespaces) requests() []mpsns.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.provisioned)
}

func (f *fakeNamespaces) releases() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.released)
}

// TestProvisionEndToEndWithTheRealClient: the provision call travels the same
// unix socket as the drain, and what comes back is the leaf namespace pipe
// directory the container is bind-mounted. Deliberately end-to-end through the
// real client, because the thing most likely to break silently is the wire
// shape between two binaries in two images.
func TestProvisionEndToEndWithTheRealClient(t *testing.T) {
	namespaces := &fakeNamespaces{lease: mpsns.Lease{
		ContainerID:         testContainerID,
		Namespace:           "kai_3f5a1c9e7b2d4086_abc123",
		Server:              "shared",
		PipeDirectory:       "/run/nvidia-mps/shared/kai_3f5a1c9e7b2d4086_abc123",
		ActiveThreadPercent: 25,
	}}
	socketPath, _ := serveOnSocket(t, Options{Control: &fakeControl{}, Namespaces: namespaces})

	client := mpsdrain.NewClient(socketPath, 5*time.Second, quietLogger())
	resp, err := client.Provision(context.Background(), mpsdrain.ProvisionRequest{
		ContainerID:         testContainerID,
		ActiveThreadPercent: 25,
		GPUUUIDs:            []string{"GPU-abc123"},
		Pod:                 "trainer-0",
		Namespace:           "team-a",
	})
	if err != nil {
		t.Fatalf("Provision() error = %v", err)
	}

	if resp.PipeDirectory != namespaces.lease.PipeDirectory {
		t.Errorf("PipeDirectory = %q, want %q", resp.PipeDirectory, namespaces.lease.PipeDirectory)
	}
	if resp.Namespace != namespaces.lease.Namespace || resp.Server != "shared" || resp.ActiveThreadPercent != 25 {
		t.Errorf("response = %+v, want the lease's namespace, server and cap", resp)
	}

	got := namespaces.requests()
	if len(got) != 1 {
		t.Fatalf("provision requests = %d, want 1", len(got))
	}
	if got[0].ContainerID != testContainerID || got[0].ActiveThreadPercent != 25 {
		t.Errorf("request = %+v, want the container id and cap intact", got[0])
	}
	if !slices.Equal(got[0].GPUUUIDs, []string{"GPU-abc123"}) {
		t.Errorf("request GPU UUIDs = %v, want [GPU-abc123]", got[0].GPUUUIDs)
	}
}

// TestProvisionFailureIsAnError, and must be: a container whose namespace could
// not be provisioned has no enforceable compute cap, and the caller has to be
// able to refuse to create it. A failure dressed up as a successful response
// with an empty pipe directory would be a container created uncapped.
func TestProvisionFailureIsAnError(t *testing.T) {
	namespaces := &fakeNamespaces{provisionErr: errors.New("the control daemon refused")}
	socketPath, _ := serveOnSocket(t, Options{Control: &fakeControl{}, Namespaces: namespaces})

	client := mpsdrain.NewClient(socketPath, 5*time.Second, quietLogger())
	resp, err := client.Provision(context.Background(), mpsdrain.ProvisionRequest{
		ContainerID: testContainerID, ActiveThreadPercent: 25,
	})
	if err == nil {
		t.Fatalf("Provision() = %+v, nil; want the failure reported", resp)
	}
	if !strings.Contains(err.Error(), "the control daemon refused") {
		t.Errorf("error %v does not carry the reason the caller has to log", err)
	}
	if resp.PipeDirectory != "" {
		t.Errorf("PipeDirectory = %q alongside an error; a caller that ignored the error would mount it", resp.PipeDirectory)
	}
}

// TestProvisionWithoutANamespaceManagerIsRefused: a node where namespace
// isolation is off, or misconfigured, must say so rather than quietly hand back
// nothing. fractiond turns this into a refusal to create the container, which
// is the safe direction — the unsafe one is an sm-sharing container running
// with no cap.
func TestProvisionWithoutANamespaceManagerIsRefused(t *testing.T) {
	socketPath, _ := serveOnSocket(t, Options{Control: &fakeControl{}})

	client := mpsdrain.NewClient(socketPath, 5*time.Second, quietLogger())
	if _, err := client.Provision(context.Background(), mpsdrain.ProvisionRequest{
		ContainerID: testContainerID, ActiveThreadPercent: 25,
	}); err == nil {
		t.Fatal("Provision() error = nil on a node with no namespace manager, want a refusal")
	}
}

// TestReleaseEndToEndWithTheRealClient covers both outcomes of a release: the
// namespace is gone, or it is not gone yet because a client is still attached.
// The second is not an error — the caller is on a teardown path and mpsd
// retries — so it has to arrive as a successful response that says so.
func TestReleaseEndToEndWithTheRealClient(t *testing.T) {
	for _, deleted := range []bool{true, false} {
		namespaces := &fakeNamespaces{deleted: deleted}
		socketPath, _ := serveOnSocket(t, Options{Control: &fakeControl{}, Namespaces: namespaces})

		client := mpsdrain.NewClient(socketPath, 5*time.Second, quietLogger())
		resp, err := client.Release(context.Background(), mpsdrain.ReleaseRequest{ContainerID: testContainerID})
		if err != nil {
			t.Fatalf("Release() error = %v", err)
		}
		if resp.Deleted != deleted {
			t.Errorf("Deleted = %v, want %v", resp.Deleted, deleted)
		}
		if !deleted && resp.Message == "" {
			t.Error("a release that did not delete said nothing about why")
		}
		if got := namespaces.releases(); !slices.Equal(got, []string{testContainerID}) {
			t.Errorf("released = %v, want [%s]", got, testContainerID)
		}
	}
}

// TestReleaseWithoutANamespaceManagerIsNotAnError: fractiond releases from
// RemoveContainer unconditionally. On a node with isolation off that must be a
// no-op, not an error logged on every container removal.
func TestReleaseWithoutANamespaceManagerIsNotAnError(t *testing.T) {
	socketPath, _ := serveOnSocket(t, Options{Control: &fakeControl{}})

	client := mpsdrain.NewClient(socketPath, 5*time.Second, quietLogger())
	resp, err := client.Release(context.Background(), mpsdrain.ReleaseRequest{ContainerID: testContainerID})
	if err != nil {
		t.Fatalf("Release() error = %v, want a quiet no-op", err)
	}
	if resp.Deleted {
		t.Error("Deleted = true on a node with no namespace manager")
	}
}

// TestNamespaceEndpointsRejectMalformedRequests: both endpoints are reachable
// only over a root-only unix socket, but a missing container id would otherwise
// be answered with a namespace derived from an empty string — the same name for
// every such request.
func TestNamespaceEndpointsRejectMalformedRequests(t *testing.T) {
	socketPath, _ := serveOnSocket(t, Options{Control: &fakeControl{}, Namespaces: &fakeNamespaces{}})
	client := rawClient(t, socketPath)

	for _, path := range []string{mpsdrain.PathProvision, mpsdrain.PathRelease} {
		resp, err := client.Post("http://mpsd"+path, "application/json", strings.NewReader(`{"activeThreadPercent":50}`))
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("POST %s with no container id = %s, want 400", path, resp.Status)
		}

		resp, err = client.Post("http://mpsd"+path, "application/json", strings.NewReader("{not json"))
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("POST %s with a malformed body = %s, want 400", path, resp.Status)
		}

		getResp, err := client.Get("http://mpsd" + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		_ = getResp.Body.Close()
		if getResp.StatusCode != http.StatusMethodNotAllowed {
			t.Errorf("GET %s = %s, want 405", path, getResp.Status)
		}
	}
}
