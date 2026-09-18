// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsdrain

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/daemonpaths"
)

const testContainerID = "3f5a1c9e7b2d4086af1c2e3d4b5a69780f1e2d3c4b5a69788f7e6d5c4b3a2910"

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// serveUnix runs handler on a unix socket in t.TempDir() and returns its path.
func serveUnix(t *testing.T, handler http.HandlerFunc) string {
	t.Helper()

	socketPath := filepath.Join(t.TempDir(), "mpsd.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatal(err)
	}
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 2 * time.Second}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })

	return socketPath
}

// acceptAndStall accepts connections and never answers them, which is what a
// wedged mpsd looks like from the caller's side.
func acceptAndStall(t *testing.T) string {
	t.Helper()

	socketPath := filepath.Join(t.TempDir(), "mpsd.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	var conns []net.Conn
	done := make(chan struct{})

	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			mu.Lock()
			conns = append(conns, conn)
			mu.Unlock()
			select {
			case <-done:
				return
			default:
			}
		}
	}()

	t.Cleanup(func() {
		close(done)
		_ = listener.Close()
		mu.Lock()
		defer mu.Unlock()
		for _, conn := range conns {
			_ = conn.Close()
		}
	})

	return socketPath
}

func TestClientDrainHappyPath(t *testing.T) {
	socketPath := serveUnix(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != Path {
			t.Errorf("got %s %s, want POST %s", r.Method, r.URL.Path, Path)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
		}
		_, _ = w.Write([]byte(`{"drained":[100,200],"wedged":false,"restarted":true}`))
	})

	resp, err := NewClient(socketPath, 2*time.Second, quietLogger()).
		Drain(context.Background(), Request{ContainerID: testContainerID})
	if err != nil {
		t.Fatalf("Drain() error = %v", err)
	}
	if !slices.Equal(resp.Drained, []int{100, 200}) || resp.Wedged || !resp.Restarted {
		t.Errorf("response = %+v, want drained [100 200] and Restarted", resp)
	}
}

func TestClientRejectsNon200Responses(t *testing.T) {
	// A non-200 means mpsd did not drain anything. Decoding the body anyway
	// would report an empty-but-successful drain and hide a broken endpoint on
	// every container stop.
	tests := []struct {
		name string
		code int
		body string
	}{
		{name: "bad request", code: http.StatusBadRequest, body: "containerId is required\n"},
		{name: "method not allowed", code: http.StatusMethodNotAllowed, body: "only POST is supported\n"},
		{name: "internal error", code: http.StatusInternalServerError, body: "boom"},
		{name: "not found - wrong path or a different server on the socket", code: http.StatusNotFound, body: "404 page not found\n"},
		{
			// The dangerous one: a valid-looking body behind a failure status.
			name: "a valid response body behind a failure status is still a failure",
			code: http.StatusServiceUnavailable, body: `{"drained":[1,2,3]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			socketPath := serveUnix(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.code)
				_, _ = w.Write([]byte(tt.body))
			})

			resp, err := NewClient(socketPath, 2*time.Second, quietLogger()).
				Drain(context.Background(), Request{ContainerID: testContainerID})
			if err == nil {
				t.Fatalf("Drain() error = nil, want a failure for status %d", tt.code)
			}
			if len(resp.Drained) != 0 || resp.Wedged || resp.Restarted {
				t.Errorf("response = %+v, want the zero value on failure", resp)
			}
			if !strings.Contains(err.Error(), http.StatusText(tt.code)) {
				t.Errorf("error %v does not mention the status", err)
			}
			if !strings.Contains(err.Error(), strings.TrimSpace(tt.body)) {
				t.Errorf("error %v dropped the server's explanation %q", err, tt.body)
			}
		})
	}
}

func TestClientRejectsGarbageBodies(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "not JSON at all", body: "<html>gateway</html>"},
		{name: "truncated JSON", body: `{"drained":[1,2`},
		{name: "a JSON array where an object is expected", body: `[1,2,3]`},
		{name: "wrong field type", body: `{"drained":"all of them"}`},
		{name: "empty body", body: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			socketPath := serveUnix(t, func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(tt.body))
			})

			_, err := NewClient(socketPath, 2*time.Second, quietLogger()).
				Drain(context.Background(), Request{ContainerID: testContainerID})
			if err == nil {
				t.Fatalf("Drain() error = nil, want a decode failure for %q", tt.body)
			}
			if !strings.Contains(err.Error(), "decoding drain response") {
				t.Errorf("error = %v, want a decode failure", err)
			}
		})
	}
}

func TestClientDoesNotSwallowAnOversizedBody(t *testing.T) {
	// Something other than mpsd on the socket must not be able to make the
	// caller allocate without bound on the container-stop path.
	socketPath := serveUnix(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"message":"`))
		chunk := strings.Repeat("A", 4096)
		for range 64 { // 256 KiB, four times the cap
			_, _ = w.Write([]byte(chunk))
		}
		_, _ = w.Write([]byte(`"}`))
	})

	_, err := NewClient(socketPath, 5*time.Second, quietLogger()).
		Drain(context.Background(), Request{ContainerID: testContainerID})
	if err == nil {
		t.Fatal("Drain() error = nil, want the truncated body to fail to decode")
	}
	if !strings.Contains(err.Error(), "decoding drain response") {
		t.Errorf("error = %v, want a decode failure", err)
	}
}

func TestClientAcceptsABodyJustUnderTheCap(t *testing.T) {
	// The cap must not reject a legitimately large drain: a container with many
	// MPS clients still has to get its response.
	pids := make([]int, 0, 2000)
	var body strings.Builder
	body.WriteString(`{"drained":[`)
	for i := range 2000 {
		if i > 0 {
			body.WriteString(",")
		}
		pid := 100000 + i
		pids = append(pids, pid)
		body.WriteString(strconv.Itoa(pid))
	}
	body.WriteString(`]}`)
	if body.Len() >= maxResponseBytes {
		t.Fatalf("test fixture is %d bytes, which is already over the %d cap", body.Len(), maxResponseBytes)
	}

	socketPath := serveUnix(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(body.String()))
	})

	resp, err := NewClient(socketPath, 5*time.Second, quietLogger()).
		Drain(context.Background(), Request{ContainerID: testContainerID})
	if err != nil {
		t.Fatalf("Drain() error = %v", err)
	}
	if !slices.Equal(resp.Drained, pids) {
		t.Errorf("decoded %d PIDs, want %d", len(resp.Drained), len(pids))
	}
}

func TestClientTimesOutRatherThanHanging(t *testing.T) {
	// A wedged mpsd that accepts the connection and never answers is the exact
	// shape of the failure this feature is about. Hanging here would hold the
	// NRI hook open past the runtime's plugin request timeout, which closes the
	// plugin connection and stops fractiond injecting limits at all.
	socketPath := acceptAndStall(t)

	client := NewClient(socketPath, 50*time.Millisecond, quietLogger())
	start := time.Now()

	done := make(chan error, 1)
	go func() {
		_, err := client.Drain(context.Background(), Request{ContainerID: testContainerID})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Drain() error = nil, want a timeout")
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("error = %v, want a recognisable context.DeadlineExceeded", err)
		}
		if elapsed := time.Since(start); elapsed > 2*time.Second {
			t.Errorf("took %v to give up on a 50ms timeout", elapsed)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Drain() hung on a server that never answers")
	}
}

func TestClientHonoursACallerDeadlineShorterThanItsOwn(t *testing.T) {
	// fractiond clamps to whatever the runtime negotiated; the tighter of the
	// two deadlines has to win.
	socketPath := acceptAndStall(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	start := time.Now()
	if _, err := NewClient(socketPath, time.Hour, quietLogger()).Drain(ctx, Request{ContainerID: testContainerID}); err == nil {
		t.Fatal("Drain() error = nil, want the caller's deadline to apply")
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("took %v; the caller's 30ms deadline was ignored", elapsed)
	}
}

func TestClientReportsAnAlreadyCancelledContext(t *testing.T) {
	socketPath := serveUnix(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := NewClient(socketPath, time.Second, quietLogger()).Drain(ctx, Request{ContainerID: testContainerID}); !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want context.Canceled", err)
	}
}

func TestClientReportsAMissingSocket(t *testing.T) {
	// mpsd not running, or its socket not mounted into fractiond. Both are
	// startup mistakes that have to be legible in a log line.
	missing := filepath.Join(t.TempDir(), "absent.sock")

	_, err := NewClient(missing, time.Second, quietLogger()).
		Drain(context.Background(), Request{ContainerID: testContainerID})
	if err == nil {
		t.Fatal("Drain() error = nil, want a dial failure")
	}
	if !strings.Contains(err.Error(), missing) {
		t.Errorf("error %v does not name the socket", err)
	}
	if !strings.Contains(err.Error(), "calling mpsd drain endpoint") {
		t.Errorf("error %v does not say what it was doing", err)
	}
}

func TestClientReportsASocketThatIsAPlainFile(t *testing.T) {
	// A bind mount pointing at a file instead of the socket is a realistic
	// misconfiguration and must not become a hang.
	path := filepath.Join(t.TempDir(), "mpsd.sock")
	if err := os.WriteFile(path, []byte("not a socket"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := NewClient(path, time.Second, quietLogger()).Drain(context.Background(), Request{ContainerID: testContainerID}); err == nil {
		t.Error("Drain() error = nil, want a dial failure")
	}
}

func TestNewClientDefaults(t *testing.T) {
	tests := []struct {
		name        string
		socketPath  string
		timeout     time.Duration
		wantSocket  string
		wantTimeout time.Duration
	}{
		{name: "all defaults", wantSocket: DefaultSocketPath, wantTimeout: DefaultCallTimeout},
		{name: "zero timeout", socketPath: "/tmp/x.sock", timeout: 0, wantSocket: "/tmp/x.sock", wantTimeout: DefaultCallTimeout},
		{
			// A negative timeout must not mean "no deadline": that is the hang
			// that takes fractiond off NRI.
			name: "negative timeout", socketPath: "/tmp/x.sock", timeout: -time.Hour,
			wantSocket: "/tmp/x.sock", wantTimeout: DefaultCallTimeout,
		},
		{name: "explicit values are kept", socketPath: "/tmp/y.sock", timeout: 3 * time.Second, wantSocket: "/tmp/y.sock", wantTimeout: 3 * time.Second},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewClient(tt.socketPath, tt.timeout, nil)
			if c.SocketPath() != tt.wantSocket {
				t.Errorf("SocketPath() = %q, want %q", c.SocketPath(), tt.wantSocket)
			}
			if c.timeout != tt.wantTimeout {
				t.Errorf("timeout = %v, want %v", c.timeout, tt.wantTimeout)
			}
			if c.log == nil {
				t.Error("log = nil; a nil logger is a panic on the first failure")
			}
		})
	}
}

func TestDefaultCallTimeoutFitsInsideTheNRIPluginDeadline(t *testing.T) {
	// containerd wraps every NRI plugin request in plugin_request_timeout (2s by
	// default) and treats the resulting context.DeadlineExceeded as fatal: it
	// closes the plugin connection. While fractiond is disconnected, newly
	// created fractional containers get no limits injected at all. The default
	// drain call therefore has to give up with room to spare.
	const nriPluginRequestTimeout = 2 * time.Second

	if DefaultCallTimeout >= nriPluginRequestTimeout {
		t.Fatalf("DefaultCallTimeout = %v, which is not inside containerd's %v plugin request timeout; "+
			"a slow drain would disconnect fractiond from NRI and drop enforcement for every new container",
			DefaultCallTimeout, nriPluginRequestTimeout)
	}
	// Leave headroom for the rest of the StopContainer hook (the CRI stop, the
	// bookkeeping) rather than spending the whole budget on the drain.
	if DefaultCallTimeout > nriPluginRequestTimeout/2 {
		t.Errorf("DefaultCallTimeout = %v leaves less than half the %v budget for the rest of the hook",
			DefaultCallTimeout, nriPluginRequestTimeout)
	}
}

// TestDefaultSocketPathIsPinned is the daemon-side half of a pin the operator
// module also carries (daemonmgr.TestDefaultMPSDrainSocketPath_IsPinned).
//
// Three things have to name the same file, in two Go modules:
//
//  1. mpsd serves the endpoint there   — mpsd --drain-socket, defaulted from
//     drain.DefaultSocketPath, which aliases this constant;
//  2. fractiond dials it               — fractiond --mps-drain-socket,
//     defaulted from this constant directly;
//  3. the operator mounts it into both pods — daemonmgr.DefaultMPSDrainSocketPath.
//
// All three now read pkg/daemonpaths.MPSDrainSocket, so they cannot drift from
// each other. What they can still do is drift from the path an already-deployed
// operator mounted, and that failure is completely silent: fractiond's
// drain-before-stop call fails to connect on every container stop, the container
// is stopped anyway (by design — a drain problem must never become a stuck pod),
// and a client killed mid-kernel wedges the MPS server for every other tenant of
// the GPU. Nothing logs an error that points at the cause.
//
// So the literal is written out once more here, on purpose, as a tripwire: a
// change to the shared constant has to be a deliberate edit to this line too,
// and it is caught from both modules rather than from neither.
func TestDefaultSocketPathIsPinned(t *testing.T) {
	const want = "/var/run/gpu-fractioning/drain/mpsd.sock"

	if DefaultSocketPath != want {
		t.Fatalf("DefaultSocketPath = %q, want %q.\n"+
			"This path is a wire contract between three consumers: mpsd's --drain-socket, "+
			"fractiond's --mps-drain-socket, and the operator's hostPath mount "+
			"(operator/internal/common/daemonmgr.DefaultMPSDrainSocketPath). "+
			"If the change is intentional, roll all of them plus any deployed DaemonSets in the same "+
			"commit — a half-applied change silently disables the MPS drain instead of failing.",
			DefaultSocketPath, want)
	}

	// Sourced from the shared package, not repeated per module: that is what
	// keeps the operator and the daemons from drifting apart in the first place.
	if DefaultSocketPath != daemonpaths.MPSDrainSocket {
		t.Errorf("DefaultSocketPath = %q but daemonpaths.MPSDrainSocket = %q; "+
			"the daemon side has stopped reading the shared constant",
			DefaultSocketPath, daemonpaths.MPSDrainSocket)
	}
}

// TestDrainPackageAliasResolvesToTheSharedPath follows the chain mpsd's flag
// default actually travels: drain.DefaultSocketPath is an alias of this
// package's constant, which is an alias of the shared one. An alias that got
// re-typed as a literal somewhere along the way would still compile and still
// look right in review.
func TestDefaultSocketPathLivesInItsOwnDirectory(t *testing.T) {
	if !filepath.IsAbs(DefaultSocketPath) {
		t.Fatalf("DefaultSocketPath = %q, want an absolute host path", DefaultSocketPath)
	}
	// The operator mounts the socket's *directory* with DirectoryOrCreate into
	// two privileged pods, so a default sitting directly in a shared host
	// directory would hand those pods far more of the node than the endpoint
	// needs.
	dir := filepath.Dir(DefaultSocketPath)
	for _, shared := range []string{"/", "/run", "/var", "/var/run", "/tmp", "/var/run/gpu-fractioning"} {
		if dir == shared {
			t.Fatalf("drain socket directory = %q, a shared host directory", dir)
		}
	}
}
