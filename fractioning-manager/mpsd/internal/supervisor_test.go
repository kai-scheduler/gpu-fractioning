// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"context"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSupervisor_RunsUntilContextExpires(t *testing.T) {
	script := filepath.Join(t.TempDir(), "fake-mps")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	s := NewSupervisor(SupervisorConfig{
		MPSBinary: script,
		PipeDir:   filepath.Join(t.TempDir(), "pipe"),
		LogDir:    filepath.Join(t.TempDir(), "log"),
		Backoff:   100 * time.Millisecond,
		Stdout:    io.Discard,
		Stderr:    io.Discard,
	}, testLogger())

	err := s.Run(ctx)
	if err != nil {
		t.Errorf("Run() returned error: %v", err)
	}
}

func TestSupervisor_RunOnceThenStop(t *testing.T) {
	script := filepath.Join(t.TempDir(), "fake-mps")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:       script,
		PipeDir:         filepath.Join(t.TempDir(), "pipe"),
		LogDir:          filepath.Join(t.TempDir(), "log"),
		Backoff:         10 * time.Millisecond,
		MaxRetries:      1,
		StableThreshold: time.Hour,
		Stdout:          io.Discard,
		Stderr:          io.Discard,
	}, testLogger())

	err := s.Run(ctx)
	if err == nil {
		t.Fatal("expected error after single retry, got nil")
	}
	if !strings.Contains(err.Error(), "1 attempts") {
		t.Errorf("expected error to mention '1 attempts', got: %v", err)
	}
}

func TestSupervisor_GracefulShutdown(t *testing.T) {
	script := filepath.Join(t.TempDir(), "fake-mps")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nsleep 60\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         script,
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           100 * time.Millisecond,
		GracefulStopDelay: 2 * time.Second,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	done := make(chan error, 1)
	go func() {
		done <- s.Run(ctx)
	}()

	time.Sleep(200 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run() returned error on graceful shutdown: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run() did not return within 5s after cancel")
	}
}

func TestSupervisor_GracefulShutdown_SIGKILLAfterDelay(t *testing.T) {
	// Script traps SIGTERM and ignores it — only SIGKILL (after WaitDelay) can stop it.
	// Uses a busy loop (shell built-in) instead of `sleep` to avoid orphaned child
	// processes that hold stdout/stderr pipes open after SIGKILL.
	script := filepath.Join(t.TempDir(), "fake-mps")
	if err := os.WriteFile(script, []byte("#!/bin/sh\ntrap '' TERM\nwhile true; do :; done\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	gracePeriod := 1 * time.Second
	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         script,
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           100 * time.Millisecond,
		GracefulStopDelay: gracePeriod,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	done := make(chan error, 1)
	go func() {
		done <- s.Run(ctx)
	}()

	time.Sleep(200 * time.Millisecond)
	start := time.Now()
	cancel()

	select {
	case err := <-done:
		elapsed := time.Since(start)
		if err != nil {
			t.Errorf("Run() returned error: %v", err)
		}
		if elapsed < gracePeriod {
			t.Errorf("process exited in %v, expected at least %v (SIGKILL after WaitDelay)", elapsed, gracePeriod)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Run() did not return within 10s — SIGKILL path may be broken")
	}
}

func TestSupervisor_MaxRetriesExhausted(t *testing.T) {
	// Script that automatically fails
	script := filepath.Join(t.TempDir(), "fake-mps")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:       script,
		PipeDir:         filepath.Join(t.TempDir(), "pipe"),
		LogDir:          filepath.Join(t.TempDir(), "log"),
		Backoff:         10 * time.Millisecond,
		MaxRetries:      3,
		StableThreshold: time.Hour,
		Stdout:          io.Discard,
		Stderr:          io.Discard,
	}, testLogger())

	err := s.Run(ctx)
	if err == nil {
		t.Fatal("expected error after exhausting retries, got nil")
	}
	if !strings.Contains(err.Error(), "3 attempts") {
		t.Errorf("expected error to mention '3 attempts', got: %v", err)
	}
}

func TestSupervisor_StableThresholdResetsRetryBudget(t *testing.T) {
	// Script sleeps 100ms then exits — always exceeds the 50ms StableThreshold,
	// so the retry budget resets every time and MaxRetries is never exhausted.
	script := filepath.Join(t.TempDir(), "fake-mps")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nsleep 0.1\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:       script,
		PipeDir:         filepath.Join(t.TempDir(), "pipe"),
		LogDir:          filepath.Join(t.TempDir(), "log"),
		Backoff:         10 * time.Millisecond,
		MaxRetries:      2,
		StableThreshold: 50 * time.Millisecond,
		Stdout:          io.Discard,
		Stderr:          io.Discard,
	}, testLogger())

	err := s.Run(ctx)
	if err != nil {
		t.Errorf("expected nil (context timeout) because stable runs should reset the retry budget, got: %v", err)
	}
}

func TestBuildMPSArgs(t *testing.T) {
	tests := []struct {
		name        string
		controlPort string
		configPath  string
		multiuser   bool
		want        []string
	}{
		{
			name:        "port, config and multiuser (production invocation)",
			controlPort: DefaultMPSControlPort,
			configPath:  DefaultMPSConfigPath,
			multiuser:   true,
			want:        []string{"-p", "3", "-m", "-f", "-a", "/etc/nvidia-mps/mps-control.toml"},
		},
		{
			// sm-sharing disabled: the shared server is gone, so -m must go
			// with it, restoring the exact pre-feature invocation.
			name:        "sm-sharing disabled omits -m",
			controlPort: DefaultMPSControlPort,
			configPath:  DefaultMPSConfigPath,
			multiuser:   false,
			want:        []string{"-p", "3", "-f", "-a", "/etc/nvidia-mps/mps-control.toml"},
		},
		{
			name:        "empty port omits -p",
			controlPort: "",
			configPath:  DefaultMPSConfigPath,
			multiuser:   true,
			want:        []string{"-m", "-f", "-a", "/etc/nvidia-mps/mps-control.toml"},
		},
		{
			name:        "empty config omits -a",
			controlPort: DefaultMPSControlPort,
			configPath:  "",
			multiuser:   true,
			want:        []string{"-p", "3", "-m", "-f"},
		},
		{
			name:        "both empty leaves only -m -f",
			controlPort: "",
			configPath:  "",
			multiuser:   true,
			want:        []string{"-m", "-f"},
		},
		{
			name:        "everything off leaves only -f",
			controlPort: "",
			configPath:  "",
			multiuser:   false,
			want:        []string{"-f"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildMPSArgs(tt.controlPort, tt.configPath, tt.multiuser)
			if !slices.Equal(got, tt.want) {
				t.Errorf("buildMPSArgs(%q, %q, %t) = %v, want %v", tt.controlPort, tt.configPath, tt.multiuser, got, tt.want)
			}
		})
	}
}

func TestSupervisor_SetupWritesConfig(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "nested", "mps-control.toml")
	content := MPSConfig{MemacctEnabled: true, MemacctAuditLog: true}.TOML()

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:     "/bin/true",
		ConfigPath:    configPath,
		ConfigContent: content,
		PipeDir:       filepath.Join(t.TempDir(), "pipe"),
		LogDir:        filepath.Join(t.TempDir(), "log"),
		Stdout:        io.Discard,
		Stderr:        io.Discard,
	}, testLogger())

	if err := s.setup(); err != nil {
		t.Fatalf("setup() error: %v", err)
	}

	got, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("reading config: %v", err)
	}
	if string(got) != content {
		t.Errorf("config file =\n%q\nwant\n%q", got, content)
	}
}

func TestSupervisor_SetupSkipsConfigWhenEmpty(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "mps-control.toml")

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:  "/bin/true",
		ConfigPath: configPath,
		// ConfigContent empty -> no file written
		PipeDir: filepath.Join(t.TempDir(), "pipe"),
		LogDir:  filepath.Join(t.TempDir(), "log"),
		Stdout:  io.Discard,
		Stderr:  io.Discard,
	}, testLogger())

	if err := s.setup(); err != nil {
		t.Fatalf("setup() error: %v", err)
	}
	if _, err := os.Stat(configPath); !os.IsNotExist(err) {
		t.Errorf("expected no config file, stat err = %v", err)
	}
}

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

// ── Supervisor.Restart ───────────────────────────────────────────────────
//
// Restart is the recovery half of the MPS wedge problem and is called from the
// drain endpoint's goroutines while the Run loop is blocked in Wait, so its
// failure modes are the ones nobody sees in a manual test.

// killRecorder stands in for SIGKILL so a test can assert which PIDs a hard
// restart would have killed without killing anything.
type killRecorder struct {
	mu   sync.Mutex
	pids []int
	err  error
}

func (k *killRecorder) kill(pid int) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.pids = append(k.pids, pid)
	return k.err
}

func (k *killRecorder) killed() []int {
	k.mu.Lock()
	defer k.mu.Unlock()
	return slices.Clone(k.pids)
}

// fakeProcWithComms builds a procfs whose processes have the given comm names.
// fakeProcExecutables builds a /proc stand-in where each entry is identified by
// its exe symlink, the way killStrayServers reads it. comm is deliberately NOT
// written: the kernel truncates it to 15 characters, so writing it here would
// let a test pass against a lookup that could never work on a real node.
func fakeProcExecutables(t *testing.T, executables map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for dir, executable := range executables {
		path := filepath.Join(root, dir)
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join("/usr/bin", executable), filepath.Join(path, "exe")); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// blockingScript is an MPS stand-in that waits on stdin exactly like the real
// control daemon run with -f, so "quit" and SIGKILL behave as they would in
// production.
func blockingScript(t *testing.T) string {
	t.Helper()
	return writeScript(t, "#!/bin/sh\nread line\nexit 0\n")
}

func writeScript(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fake-mps")
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// waitForMPS waits for the supervisor to be running a daemon that is not
// previous, which is how a test observes that a restart actually took.
func waitForMPS(t *testing.T, s *Supervisor, previous *runningMPS) *runningMPS {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		current := s.current
		s.mu.Unlock()
		if current != nil && current != previous {
			return current
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("the supervisor never brought up a new MPS daemon")
	return nil
}

func waitUntilStopped(t *testing.T, s *Supervisor, current *runningMPS, within time.Duration) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if !s.isCurrent(current) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("the MPS daemon was still running %v after the restart", within)
}

func TestSupervisor_RestartWithNoDaemonRunningIsAnError(t *testing.T) {
	// Drain calls Restart from an HTTP handler. Before Run has started anything
	// — or after it has given up — there is no process to act on, and a nil
	// deref there would take mpsd down on the container-stop path.
	s := NewSupervisor(SupervisorConfig{
		MPSBinary:   "/bin/true",
		PipeDir:     filepath.Join(t.TempDir(), "pipe"),
		LogDir:      filepath.Join(t.TempDir(), "log"),
		ProcRoot:    t.TempDir(),
		KillProcess: (&killRecorder{}).kill,
		Stdout:      io.Discard,
		Stderr:      io.Discard,
	}, testLogger())

	for _, graceful := range []bool{true, false} {
		err := s.Restart("nothing to restart", graceful)
		if err == nil {
			t.Errorf("Restart(graceful=%t) error = nil, want a failure", graceful)
		} else if !strings.Contains(err.Error(), "not running") {
			t.Errorf("Restart(graceful=%t) error = %v, want it to say the daemon is not running", graceful, err)
		}
	}

	// And a failed Restart must not leave a request behind for the next run to
	// pick up, or a real crash would be silently excused.
	if reason, requested := s.takeRestartRequest(); requested {
		t.Errorf("a failed Restart left a pending request %q", reason)
	}
}

func TestSupervisor_GracefulRestartSendsQuit(t *testing.T) {
	// "quit" on stdin is the NVIDIA-documented way to stop the control daemon:
	// it drains active clients and shuts the MPS servers down cleanly. Killing
	// instead would strand exactly the GPU state a restart is meant to clear.
	quitLog := filepath.Join(t.TempDir(), "stdin.log")
	script := writeScript(t, "#!/bin/sh\nread line\nprintf '%s\\n' \"$line\" >> "+quitLog+"\nexit 0\n")

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         script,
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           10 * time.Millisecond,
		GracefulStopDelay: 5 * time.Second,
		ProcRoot:          t.TempDir(),
		KillProcess:       (&killRecorder{}).kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)
	if err := s.Restart("idle recycle", true); err != nil {
		t.Fatalf("Restart() error = %v", err)
	}
	waitUntilStopped(t, s, first, 10*time.Second)

	deadline := time.Now().Add(10 * time.Second)
	for {
		content, err := os.ReadFile(quitLog)
		if err == nil && strings.Contains(string(content), "quit") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the daemon never received \"quit\" on stdin (log = %q, err = %v)", content, err)
		}
		time.Sleep(time.Millisecond)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run() error = %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
}

func TestSupervisor_GracefulRestartEscalatesToKill(t *testing.T) {
	// A control daemon with a client still running a kernel blocks on "quit".
	// Nothing else would ever stop it: the Run loop is inside Wait, and the
	// exec package's WaitDelay only fires on a context cancellation, which a
	// restart is not. Without the escalation the node stays wedged forever.
	script := writeScript(t, "#!/bin/sh\ntrap '' TERM\nwhile true; do :; done\n")

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         script,
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           10 * time.Millisecond,
		GracefulStopDelay: 100 * time.Millisecond,
		ProcRoot:          t.TempDir(),
		KillProcess:       (&killRecorder{}).kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)
	start := time.Now()
	if err := s.Restart("wedged server", true); err != nil {
		t.Fatalf("Restart() error = %v", err)
	}

	waitUntilStopped(t, s, first, 10*time.Second)
	if elapsed := time.Since(start); elapsed < 100*time.Millisecond {
		t.Errorf("the daemon died after %v, before the graceful stop delay had a chance to elapse", elapsed)
	}

	// And the supervisor brings a fresh one up rather than treating the kill as
	// a reason to stop.
	waitForMPS(t, s, first)

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run() error = %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
}

func TestSupervisor_HardRestartKillsTheDaemonAndEveryMPSServer(t *testing.T) {
	// The per-GPU MPS servers are what actually hold the wedged state, and they
	// outlive their control daemon. Restarting only the daemon would report
	// success and change nothing.
	kills := &killRecorder{}
	procRoot := fakeProcExecutables(t, map[string]string{
		"111":     mpsServerExecutable,
		"222":     mpsServerExecutable,
		"333":     "bash",
		"444":     mpsServerExecutable + "-extra",
		"notapid": mpsServerExecutable,
	})

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         blockingScript(t),
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           10 * time.Millisecond,
		GracefulStopDelay: time.Second,
		ProcRoot:          procRoot,
		KillProcess:       kills.kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)
	if err := s.Restart("wedged server", false); err != nil {
		t.Fatalf("Restart() error = %v", err)
	}

	if killed := kills.killed(); !slices.Equal(killed, []int{111, 222}) {
		t.Errorf("killed %v, want [111 222] — every nvidia-cuda-mps-server and nothing else", killed)
	}
	waitUntilStopped(t, s, first, 10*time.Second)

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run() error = %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
}

func TestSupervisor_HardRestartSurvivesAFailingKill(t *testing.T) {
	// A server that exits between the scan and the kill is normal. Giving up on
	// the rest of the list would leave other GPUs' servers wedged.
	kills := &killRecorder{err: os.ErrPermission}
	procRoot := fakeProcExecutables(t, map[string]string{"111": mpsServerExecutable, "222": mpsServerExecutable})

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         blockingScript(t),
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           10 * time.Millisecond,
		GracefulStopDelay: time.Second,
		ProcRoot:          procRoot,
		KillProcess:       kills.kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)
	if err := s.Restart("wedged server", false); err != nil {
		t.Fatalf("Restart() error = %v", err)
	}
	if killed := kills.killed(); !slices.Equal(killed, []int{111, 222}) {
		t.Errorf("killed %v, want the second kill to be attempted after the first failed", killed)
	}
	waitUntilStopped(t, s, first, 10*time.Second)

	cancel()
	<-done
}

func TestSupervisor_HardRestartWithAnUnreadableProcRoot(t *testing.T) {
	// Failing to scan /proc must not stop the control daemon from being killed,
	// and must not panic — the daemon is the part we can definitely fix.
	kills := &killRecorder{}
	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         blockingScript(t),
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           10 * time.Millisecond,
		GracefulStopDelay: time.Second,
		ProcRoot:          filepath.Join(t.TempDir(), "no-such-proc"),
		KillProcess:       kills.kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)
	if err := s.Restart("wedged server", false); err != nil {
		t.Fatalf("Restart() error = %v", err)
	}
	waitUntilStopped(t, s, first, 10*time.Second)
	if killed := kills.killed(); len(killed) != 0 {
		t.Errorf("killed %v with an unreadable procfs, want nothing", killed)
	}

	cancel()
	<-done
}

func TestSupervisor_IntentionalRestartDoesNotConsumeTheRetryBudget(t *testing.T) {
	// mpsd recycles MPS after every drained workload. If that counted as a
	// crash, MaxRetries would be exhausted on a busy node and the supervisor
	// would exit the process — taking MPS down for every tenant because pods
	// were being deleted normally.
	//
	// MaxRetries is 1, so a single restart proves it: one restart counted as a
	// crash is already over budget. Backoff is long, so a restart that took the
	// crash path would also be visible as a 5s stall instead of the 1s
	// intentional-restart pause.
	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         blockingScript(t),
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           5 * time.Second,
		MaxRetries:        1,
		StableThreshold:   time.Hour,
		GracefulStopDelay: time.Second,
		ProcRoot:          t.TempDir(),
		KillProcess:       (&killRecorder{}).kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)
	start := time.Now()
	if err := s.Restart("no MPS clients remain after drain", false); err != nil {
		t.Fatalf("Restart() error = %v", err)
	}

	second := waitForMPS(t, s, first)
	restartTook := time.Since(start)

	select {
	case err := <-done:
		t.Fatalf("Run() returned %v; an intentional restart was counted against MaxRetries", err)
	default:
	}
	if second == first {
		t.Fatal("the supervisor did not bring up a new daemon")
	}
	if restartTook > 3*time.Second {
		t.Errorf("the restart took %v; that is the crash backoff (%v), not the intentional-restart pause",
			restartTook, 5*time.Second)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run() error = %v, want a clean shutdown", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
}

func TestSupervisor_RestartRequestIsConsumedExactlyOnce(t *testing.T) {
	// The request is cleared by the Run loop after the restart it belongs to.
	// If it survived, the *next* exit — a real crash — would be excused as
	// intentional and the retry budget would never be spent, so a daemon that
	// cannot start would be restarted forever instead of failing the pod.
	s := NewSupervisor(SupervisorConfig{
		MPSBinary:   "/bin/true",
		PipeDir:     filepath.Join(t.TempDir(), "pipe"),
		LogDir:      filepath.Join(t.TempDir(), "log"),
		ProcRoot:    t.TempDir(),
		KillProcess: (&killRecorder{}).kill,
		Stdout:      io.Discard,
		Stderr:      io.Discard,
	}, testLogger())

	s.mu.Lock()
	s.restartRequested = true
	s.restartReason = "idle recycle"
	s.mu.Unlock()

	reason, requested := s.takeRestartRequest()
	if !requested || reason != "idle recycle" {
		t.Fatalf("takeRestartRequest() = %q, %t; want the pending request", reason, requested)
	}
	reason, requested = s.takeRestartRequest()
	if requested || reason != "" {
		t.Errorf("takeRestartRequest() = %q, %t on the second call; the request was not consumed", reason, requested)
	}
}

func TestSupervisor_RestartWithAnEmptyReasonIsStillIntentional(t *testing.T) {
	// The reason is a log string, not the signal. Treating an empty one as "no
	// restart was requested" would make the following exit look like a crash,
	// consuming the retry budget for a restart mpsd asked for itself.
	s := NewSupervisor(SupervisorConfig{
		MPSBinary:   "/bin/true",
		PipeDir:     filepath.Join(t.TempDir(), "pipe"),
		LogDir:      filepath.Join(t.TempDir(), "log"),
		ProcRoot:    t.TempDir(),
		KillProcess: (&killRecorder{}).kill,
		Stdout:      io.Discard,
		Stderr:      io.Discard,
	}, testLogger())

	// A handle whose process was never started: hardStop has nothing to kill,
	// which keeps this test to the bookkeeping it is about.
	s.setCurrent(&runningMPS{cmd: exec.Command("/bin/true"), stdin: io.Discard})

	if err := s.Restart("", false); err != nil {
		t.Fatalf("Restart(\"\") error = %v", err)
	}
	if _, requested := s.takeRestartRequest(); !requested {
		t.Error("an empty reason was not recorded as a restart request; the next exit would be counted as a crash")
	}
}

func TestSupervisor_ConcurrentRestarts(t *testing.T) {
	// The drain endpoint serves requests concurrently, so several container
	// stops can decide to restart MPS at the same moment. They must not race on
	// the process handle or double-kill something that has already been
	// replaced.
	kills := &killRecorder{}
	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         blockingScript(t),
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           10 * time.Millisecond,
		GracefulStopDelay: 50 * time.Millisecond,
		StableThreshold:   time.Hour,
		ProcRoot:          fakeProcExecutables(t, map[string]string{"111": mpsServerExecutable}),
		KillProcess:       kills.kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// "not running" is a legitimate answer for a caller that arrives
			// between two runs; anything else is not.
			if err := s.Restart("concurrent recycle", i%2 == 0); err != nil && !strings.Contains(err.Error(), "not running") {
				t.Errorf("Restart() error = %v", err)
			}
		}()
	}
	wg.Wait()

	waitUntilStopped(t, s, first, 10*time.Second)
	waitForMPS(t, s, first)

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run() error = %v; concurrent restarts must not exhaust the retry budget", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
}

// TestSupervisor_IntentionalRestartsDoNotAccumulateAgainstTheCrashBudget is the
// assertion its single-restart sibling above cannot make.
//
// TestSupervisor_IntentionalRestartDoesNotConsumeTheRetryBudget only proves the
// Run loop takes the fast path for one requested restart: the MaxRetries check
// lives on the crash branch, which that path skips entirely, so a supervisor
// that silently carried the attempt counter forward across every recycle would
// still pass it. The damage from that bug only lands later — mpsd recycles MPS
// after every drained workload, so on a busy node the counter creeps up all day
// and then the first genuine crash, which should have had a full retry budget,
// takes the process down instead. Every tenant of the node loses MPS because
// pods were being deleted normally.
//
// So this drives the whole sequence: MaxRetries worth of recycles, then real
// crashes, and counts how many times the daemon was actually started. With the
// budget properly reset, the crashes get all MaxRetries attempts; with the
// recycles counted against it, the first crash is already over budget and the
// daemon is started one time fewer.
func TestSupervisor_IntentionalRestartsDoNotAccumulateAgainstTheCrashBudget(t *testing.T) {
	const maxRetries = 2

	dir := t.TempDir()
	counter := filepath.Join(dir, "starts")
	// Each invocation records itself before doing anything else, so the test
	// can tell "the supervisor started a daemon" from "the daemon actually
	// ran". Invocations 1..maxRetries then block like the real control daemon
	// so they can be restarted on request; every one after that exits non-zero
	// immediately, which is a crash and the only thing allowed to spend the
	// retry budget.
	script := writeScript(t, "#!/bin/sh\n"+
		"n=$(cat "+counter+" 2>/dev/null || echo 0)\n"+
		"n=$((n+1))\n"+
		"echo $n > "+counter+"\n"+
		"if [ \"$n\" -gt "+strconv.Itoa(maxRetries)+" ]; then exit 1; fi\n"+
		"read line\n"+
		"exit 0\n")

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         script,
		PipeDir:           filepath.Join(dir, "pipe"),
		LogDir:            filepath.Join(dir, "log"),
		Backoff:           10 * time.Millisecond,
		MaxRetries:        maxRetries,
		StableThreshold:   time.Hour,
		GracefulStopDelay: time.Second,
		ProcRoot:          t.TempDir(),
		KillProcess:       (&killRecorder{}).kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	// maxRetries recycles, exactly as a busy node's drain endpoint would issue
	// them. None of these may cost the supervisor an attempt.
	previous := waitForMPS(t, s, nil)
	for i := 1; i <= maxRetries; i++ {
		// Wait for the daemon to have recorded its own start before killing it,
		// or the recycle races the process and the run is never counted.
		waitForStartCount(t, counter, i)
		if err := s.Restart("no MPS clients remain after drain", false); err != nil {
			t.Fatalf("Restart() %d error = %v", i, err)
		}
		if i < maxRetries {
			previous = waitForMPS(t, s, previous)
		}
	}

	// From here the daemon only crashes, and the supervisor should spend its
	// full budget on those crashes before giving up.
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Run() returned nil; a daemon that only crashes must eventually fail the pod")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Run() never returned after the daemon started crashing")
	}

	starts := readStartCount(t, counter)
	want := maxRetries * 2 // maxRetries recycles + maxRetries crash attempts
	if starts != want {
		t.Errorf("the daemon was started %d times, want %d: %d recycle(s) plus a full budget of %d crash attempt(s). "+
			"Fewer means the recycles were charged to the retry budget, so routine pod deletion can exhaust it and take mpsd down",
			starts, want, maxRetries, maxRetries)
	}
}

// waitForStartCount blocks until the fake daemon has recorded at least n starts.
func waitForStartCount(t *testing.T, path string, n int) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(path)
		if err == nil {
			if got, err := strconv.Atoi(strings.TrimSpace(string(raw))); err == nil && got >= n {
				return
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("the fake MPS daemon never reached %d starts", n)
}

func readStartCount(t *testing.T, path string) int {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading start counter: %v", err)
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil {
		t.Fatalf("parsing start counter %q: %v", raw, err)
	}
	return n
}

// TestSupervisor_GracefulRestartWatchdogStopsWhenTheDaemonQuits covers the
// escalation watchdog's exit path.
//
// A graceful Restart spawns a goroutine that waits out GracefulStopDelay and
// kills the daemon if the "quit" did not take. In the normal case the quit does
// take, and the watchdog has to notice and return — that is what the
// <-current.done case is for. Without it the goroutine sleeps out the full
// delay every single time, and mpsd recycles MPS after every drained workload:
// on a node running a benchmark sweep that is one sleeping goroutine (each
// pinning a runningMPS and a timer) per completed job for a minute at a time.
// It never surfaces as a crash, only as memory that grows with pod churn.
//
// The assertion reads the goroutine dump rather than counting goroutines,
// because it has to name *this* goroutine: a count would be satisfied by any
// unrelated goroutine exiting at the right moment.
func TestSupervisor_GracefulRestartWatchdogStopsWhenTheDaemonQuits(t *testing.T) {
	// Long enough that a watchdog which sleeps out the delay is still asleep
	// when this test finishes asserting.
	const gracefulStopDelay = 30 * time.Second

	s := NewSupervisor(SupervisorConfig{
		MPSBinary:         blockingScript(t),
		PipeDir:           filepath.Join(t.TempDir(), "pipe"),
		LogDir:            filepath.Join(t.TempDir(), "log"),
		Backoff:           10 * time.Millisecond,
		StableThreshold:   time.Hour,
		GracefulStopDelay: gracefulStopDelay,
		ProcRoot:          t.TempDir(),
		KillProcess:       (&killRecorder{}).kill,
		Stdout:            io.Discard,
		Stderr:            io.Discard,
	}, testLogger())

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	first := waitForMPS(t, s, nil)
	if watchdogGoroutines() != 0 {
		t.Fatalf("a restart watchdog was already running before any Restart call; this test cannot attribute what it sees")
	}

	if err := s.Restart("no MPS clients remain after drain", true); err != nil {
		t.Fatalf("Restart() error = %v", err)
	}

	// The daemon accepts the quit and the run ends, which is the event the
	// watchdog is supposed to be waiting on.
	waitUntilStopped(t, s, first, 10*time.Second)

	// Give it a generous window to notice, but far less than the delay it would
	// otherwise sleep out.
	deadline := time.Now().Add(5 * time.Second)
	for {
		if watchdogGoroutines() == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the graceful-restart watchdog was still parked 5s after the daemon quit; "+
				"it is sleeping out the whole %v graceful stop delay, so every MPS recycle leaks a goroutine",
				gracefulStopDelay)
		}
		time.Sleep(5 * time.Millisecond)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run() error = %v, want a clean shutdown", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("Run() did not return after cancellation")
	}
}

// watchdogGoroutines counts the goroutines currently running the escalation
// closure inside Supervisor.Restart, by name, from the runtime's own dump.
func watchdogGoroutines() int {
	buf := make([]byte, 1<<20)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			return strings.Count(string(buf[:n]), "(*Supervisor).Restart.func")
		}
		buf = make([]byte, 2*len(buf))
	}
}
