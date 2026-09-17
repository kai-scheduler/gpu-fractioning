// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/kai-scheduler/kai-gpu-fractioning/fractioning-manager/mpsd/internal/procfs"
)

const (
	DefaultMPSBinary       = "/usr/bin/nvidia-cuda-mps-control"
	DefaultLogDir          = "/var/log/nvidia-mps"
	DefaultBackoff         = 5 * time.Second  // initial wait before first restart
	DefaultMaxRetries      = 5                // after 5 consecutive restarts, the supervisor exits the process. 0 means unlimited restart attempts
	DefaultStableThreshold = 5 * time.Minute  // uptime required to reset the retry budget
	maxBackoff             = 60 * time.Second // upper bound for exponential backoff
	dirPerm                = 0o755            // rwxr-xr-x — used for runtime directories (pipe, log)
	// mpsControlSocket is the fixed filename nvidia-cuda-mps-control creates
	// inside CUDA_MPS_PIPE_DIRECTORY. It is not user-configurable.
	mpsControlSocket         = "control"
	DefaultGracefulStopDelay = 60 * time.Second // time to wait after "quit" before SIGKILL

	DefaultMPSControlPort = "3" // protocol version 3

	// mpsServerExecutable is the executable name of the per-GPU MPS servers the
	// control daemon spawns. A hard restart kills them explicitly: they are what
	// holds the GPU state, and one that outlives its control daemon keeps
	// whatever fault wedged it. Matched against the process's executable rather
	// than its comm, which the kernel truncates to 15 characters — see
	// procfs.PIDsByExecutable.
	mpsServerExecutable = "nvidia-cuda-mps-server"
	// intentionalRestartDelay is the pause before bringing MPS back after a
	// restart we asked for. It only has to cover socket teardown — unlike a
	// crash, there is no failure to back off from.
	intentionalRestartDelay = time.Second

	DefaultMPSConfigPath = "/etc/nvidia-mps/mps-control.toml" // default path for the MPS config file
	// Fixed MPS feature toggles. memacct and context-share are always on; only
	// the audit-log flag is user-configurable.
	DefaultMemacctEnabled      = true
	DefaultContextShareEnabled = true
	DefaultMemacctAuditLog     = true // can be overridden by environment variable
	// DefaultContextShareDefaultSocket disables implicit default-socket
	// context sharing: fractiond explicitly opts a container into sm-sharing
	// via its annotation, so there is no need for containers to share the
	// default per-node socket unless routed to DefaultSharedServerName.
	DefaultContextShareDefaultSocket = "off"
	// DefaultSharedServerName is the parameterless MPS server fractiond
	// routes sm-sharing containers to (see configuration.SharedMPSSocketPath).
	DefaultSharedServerName = "shared"
	// DefaultSupportSMSharing is the CLI flag default for the sm-sharing
	// installation-time chicken bit (usually overridden by the operator via the
	// SUPPORT_SM_SHARING env var, itself Helm-injected). When false, mpsd
	// renders its MPS config with context-share disabled, exactly as it did
	// before the sm-sharing feature existed.
	DefaultSupportSMSharing = true
)

// SupervisorConfig holds all settings for the MPS daemon supervisor.
type SupervisorConfig struct {
	MPSBinary         string        // path to the nvidia-cuda-mps-control binary
	ControlPort       string        // value for the -p flag; empty omits -p
	ConfigPath        string        // MPS config file for the -a flag; empty omits -a
	ConfigContent     string        // TOML written to ConfigPath at setup; empty skips writing
	Multiuser         bool          // run the daemon multiuser (-m); required by the shared MPS server
	PipeDir           string        // CUDA_MPS_PIPE_DIRECTORY — shared with containers
	LogDir            string        // CUDA_MPS_LOG_DIRECTORY — daemon log output
	Backoff           time.Duration // initial delay before restarting after an unexpected exit
	MaxRetries        int           // max restart attempts before giving up (0 = unlimited)
	StableThreshold   time.Duration // how long the daemon must run to be considered stable (resets retry budget)
	GracefulStopDelay time.Duration // time to wait after "quit" before SIGKILL
	Stdout            io.Writer     // subprocess stdout; nil defaults to os.Stdout
	Stderr            io.Writer     // subprocess stderr; nil defaults to os.Stderr

	// ProcRoot is the procfs mount used to find stray MPS servers during a hard
	// restart. Empty defaults to /proc.
	ProcRoot string
	// KillProcess sends SIGKILL to a pid. Empty defaults to the real syscall;
	// tests substitute a recorder.
	KillProcess func(pid int) error
}

// Supervisor manages the nvidia-cuda-mps-control process lifecycle.
// It starts the daemon as a subprocess and restarts it on unexpected exits
// with a configurable backoff.
type Supervisor struct {
	cfg    SupervisorConfig
	logger *slog.Logger

	// mu guards the running process handle and the pending restart request, both
	// written by Restart (called from the drain server's goroutines) and read by
	// the Run loop.
	mu sync.Mutex
	// current is the MPS process the loop is currently supervising, or nil
	// between runs.
	current *runningMPS
	// restartRequested and restartReason are set by Restart and consumed by the
	// loop, so a restart we asked for is not logged and backed off as if the
	// daemon had crashed. The flag is separate from the reason because an empty
	// reason is still a request: inferring it from a non-empty string would make
	// Restart("", ...) count against the retry budget and eventually take mpsd
	// down for doing what it was told.
	restartRequested bool
	restartReason    string
}

// runningMPS is the handle Restart needs on the supervised process: the
// process itself, and the stdin the control daemon reads commands from.
type runningMPS struct {
	cmd   *exec.Cmd
	stdin io.Writer
	// done is closed when the run ends, so a graceful restart's escalation
	// watchdog stops waiting the moment the daemon actually quits. Without it
	// each recycle leaves a goroutine asleep for the whole graceful stop delay,
	// and mpsd recycles after every drained workload.
	done chan struct{}
}

// NewSupervisor creates a new Supervisor. Nil Stdout/Stderr default to os.Stdout/os.Stderr.
func NewSupervisor(cfg SupervisorConfig, logger *slog.Logger) *Supervisor {
	if cfg.Stdout == nil {
		cfg.Stdout = os.Stdout
	}
	if cfg.Stderr == nil {
		cfg.Stderr = os.Stderr
	}
	if cfg.ProcRoot == "" {
		cfg.ProcRoot = procfs.DefaultRoot
	}
	if cfg.KillProcess == nil {
		cfg.KillProcess = killProcess
	}
	return &Supervisor{cfg: cfg, logger: logger}
}

// Run starts the MPS daemon and supervises it until the context is cancelled.
// If the daemon exits unexpectedly, it retries with exponential backoff
// (capped at 60s). The attempt counter resets after a stable run (one that
// lasted longer than StableThreshold), so a daemon that ran for years and
// then crashes gets a fresh retry budget.
func (s *Supervisor) Run(ctx context.Context) error {
	if err := s.setup(); err != nil {
		return fmt.Errorf("setup: %w", err)
	}

	attempt := 0
	backoff := s.cfg.Backoff

	for {
		attempt++
		startTime := time.Now()
		err := s.runMPS(ctx)

		// ctx.Err() != nil means someone explicitly cancelled our context
		// (e.g. SIGTERM/SIGINT) — this is a legitimate shutdown request,
		// not a daemon crash. Exit without restarting.
		if ctx.Err() != nil {
			s.logger.Info("received shutdown signal, stopping MPS daemon")
			return nil
		}

		uptime := time.Since(startTime)

		// A restart we asked for is not a failure: it must not consume the retry
		// budget, grow the backoff, or be logged as a crash. Recycling MPS after
		// every drained workload is a routine event, and on a busy node it would
		// otherwise exhaust MaxRetries and take mpsd down.
		if reason, requested := s.takeRestartRequest(); requested {
			s.logger.Info("MPS daemon restarted on request", "reason", reason, "uptime", uptime)
			attempt = 0
			backoff = s.cfg.Backoff
			s.removeStaleSocket()
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(intentionalRestartDelay):
			}
			continue
		}

		if err != nil {
			s.logger.Warn("MPS daemon exited with error, restarting",
				"error", err,
				"attempt", attempt,
				"uptime", uptime,
				"retryIn", backoff,
			)
		} else {
			s.logger.Warn("MPS daemon exited cleanly but unexpectedly, restarting",
				"attempt", attempt,
				"uptime", uptime,
				"retryIn", backoff,
			)
		}

		// If the daemon was stable (ran longer than stableThreshold),
		// reset the retry budget and backoff. Done after logging so the
		// log shows the actual attempt count before the reset.
		if uptime >= s.cfg.StableThreshold {
			s.logger.Info("daemon was stable, resetting retry budget", "uptime", uptime)
			attempt = 0
			backoff = s.cfg.Backoff
		}

		if s.cfg.MaxRetries > 0 && attempt >= s.cfg.MaxRetries {
			return fmt.Errorf("MPS daemon failed after %d attempts: %w", attempt, err)
		}

		s.removeStaleSocket()

		select {
		case <-ctx.Done():
			return nil
		case <-time.After(backoff):
		}

		backoff = min(backoff*2, maxBackoff)
	}
}

// setup creates the runtime directories needed by the MPS daemon and removes
// any stale control socket left by a previous instance (e.g. after a pod
// restart). Without this, the first runMPS call would fail to bind the socket
// and waste one attempt before the retry loop cleans it up.
func (s *Supervisor) setup() error {
	for _, dir := range []string{s.cfg.PipeDir, s.cfg.LogDir} {
		if err := os.MkdirAll(dir, dirPerm); err != nil {
			return fmt.Errorf("creating directory %q: %w", dir, err)
		}
	}
	// Generate the MPS control-daemon config on the fly (its content — e.g. the
	// memacct audit-log toggle — is driven by a Helm value the operator injects
	// as an env var). Regenerating each start lets a config change propagate on
	// the next pod rollout without rebuilding the image.
	if err := s.writeConfig(); err != nil {
		return err
	}
	// Remove any stale control socket left by a previous instance (e.g. after a pod
	// restart). Without this, the first runMPS call would fail to bind the socket
	// and waste one attempt before the retry loop cleans it up.
	s.removeStaleSocket()
	return nil
}

// writeConfig writes the rendered MPS config to ConfigPath. It is a no-op when
// either the path or the content is empty (e.g. tests, or -a explicitly disabled).
func (s *Supervisor) writeConfig() error {
	if s.cfg.ConfigPath == "" || s.cfg.ConfigContent == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.cfg.ConfigPath), dirPerm); err != nil {
		return fmt.Errorf("creating MPS config directory: %w", err)
	}
	if err := os.WriteFile(s.cfg.ConfigPath, []byte(s.cfg.ConfigContent), 0o644); err != nil {
		return fmt.Errorf("writing MPS config %q: %w", s.cfg.ConfigPath, err)
	}
	s.logger.Info("wrote MPS config", "path", s.cfg.ConfigPath)
	return nil
}

// removeStaleSocket removes the MPS control socket left by a crashed instance
// so the next restart can bind to the same path.
func (s *Supervisor) removeStaleSocket() {
	socket := filepath.Join(s.cfg.PipeDir, mpsControlSocket)
	if err := os.Remove(socket); err != nil && !os.IsNotExist(err) {
		s.logger.Warn("failed to remove stale MPS control socket", "path", socket, "error", err)
	}
}

// buildMPSArgs builds the nvidia-cuda-mps-control argument list. The daemon is
// always run in the foreground (-f) so we can supervise it. -m (multiuser, so
// containers with differing UIDs can reach one MPS server) is required by the
// shared server and therefore tracks the sm-sharing toggle: disabling the
// feature must restore the pre-feature invocation, not just the pre-feature
// config file. -p (control port) and -a (config file) are included only when
// configured — a blank value acts as an escape hatch to drop the flag without
// rebuilding. Order mirrors the known-good production invocation:
// `-p <port> -m -f -a <config>`.
func buildMPSArgs(controlPort, configPath string, multiuser bool) []string {
	args := make([]string, 0, 5)
	if controlPort != "" {
		args = append(args, "-p", controlPort)
	}
	if multiuser {
		args = append(args, "-m")
	}
	args = append(args, "-f")
	if configPath != "" {
		args = append(args, "-a", configPath)
	}
	return args
}

// runMPS starts the MPS daemon and blocks until it exits or ctx is cancelled.
// Configuration is applied via environment variables (CUDA_MPS_PIPE_DIRECTORY,
// CUDA_MPS_LOG_DIRECTORY) set on the command, plus the -a config file.
func (s *Supervisor) runMPS(ctx context.Context) error {
	args := buildMPSArgs(s.cfg.ControlPort, s.cfg.ConfigPath, s.cfg.Multiuser)
	cmd := exec.CommandContext(ctx, s.cfg.MPSBinary, args...)
	cmd.Env = append(os.Environ(),
		"CUDA_MPS_PIPE_DIRECTORY="+s.cfg.PipeDir,
		"CUDA_MPS_LOG_DIRECTORY="+s.cfg.LogDir,
	)
	cmd.Stdout = s.cfg.Stdout
	cmd.Stderr = s.cfg.Stderr

	// nvidia-cuda-mps-control -f reads commands from stdin; if stdin is
	// nil (Go default) the process gets immediate EOF and exits. Provide
	// a pipe that stays open so the daemon blocks waiting for input.
	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("creating stdin pipe: %w", err)
	}
	defer func() {
		if err := stdinPipe.Close(); err != nil {
			s.logger.Warn("failed to close stdin pipe", "error", err)
		}
	}()

	// ── Graceful shutdown ───────────────────────────────────────────────
	// When the context is cancelled (pod SIGTERM), Go calls cmd.Cancel
	// which writes "quit" to stdin — the NVIDIA-documented way to stop
	// the MPS control daemon. It cleanly drains active GPU clients and
	// shuts down MPS servers. If the daemon doesn't exit within
	// WaitDelay, Go's exec package sends SIGKILL automatically.
	cmd.Cancel = func() error {
		_, err := fmt.Fprintln(stdinPipe, "quit")
		return err
	}
	cmd.WaitDelay = s.cfg.GracefulStopDelay

	s.logger.Info("starting MPS daemon",
		"binary", s.cfg.MPSBinary,
		"args", args,
		"pipeDir", s.cfg.PipeDir,
		"logDir", s.cfg.LogDir,
		"configPath", s.cfg.ConfigPath,
	)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting MPS daemon: %w", err)
	}

	// Publish the handle only once the process exists, and retract it before
	// returning, so Restart can never act on a process that has already exited.
	running := &runningMPS{cmd: cmd, stdin: stdinPipe, done: make(chan struct{})}
	s.setCurrent(running)
	defer func() {
		s.setCurrent(nil)
		close(running.done)
	}()

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("MPS daemon process: %w", err)
	}
	return nil
}

// Restart stops the running MPS control daemon so the supervisor's own loop
// brings a fresh one up. It is the recovery half of the MPS wedge problem: a
// server that lost a client mid-kernel holds a fault that hangs every client
// that connects afterwards, and restarting MPS is the only way to clear it.
//
// graceful asks the daemon to quit through its own command interface, which
// drains and shuts down MPS servers cleanly. Use it when nothing is attached —
// with a client still running a kernel, the quit blocks, which is why it
// escalates to a kill after the configured graceful stop delay.
//
// A non-graceful restart kills the daemon outright and then kills any MPS
// server left behind. The servers are the processes actually holding the wedged
// GPU state, and they can outlive the control daemon, so leaving them running
// would mean "restarting" MPS without fixing anything.
//
// It returns an error only when there is no daemon to restart; the restart
// itself completes asynchronously in the Run loop.
func (s *Supervisor) Restart(reason string, graceful bool) error {
	s.mu.Lock()
	current := s.current
	if current != nil {
		s.restartRequested = true
		s.restartReason = reason
	}
	s.mu.Unlock()

	if current == nil {
		return fmt.Errorf("cannot restart: MPS daemon is not running")
	}

	s.logger.Info("restarting MPS daemon", "reason", reason, "graceful", graceful)

	if !graceful {
		s.hardStop(current)
		return nil
	}

	if _, err := fmt.Fprintln(current.stdin, "quit"); err != nil {
		s.logger.Warn("failed to send quit to MPS daemon, killing instead", "error", err)
		s.hardStop(current)
		return nil
	}

	// Escalate if the quit does not take. Nothing else would: the Run loop is
	// blocked in Wait, and the process-level WaitDelay only applies to a
	// context cancellation, which this is not.
	go func() {
		timer := time.NewTimer(s.cfg.GracefulStopDelay)
		defer timer.Stop()
		select {
		case <-current.done:
			// Quit accepted; nothing to escalate.
			return
		case <-timer.C:
		}
		if s.isCurrent(current) {
			s.logger.Warn("MPS daemon did not quit within the graceful stop delay, killing",
				"reason", reason, "gracefulStopDelay", s.cfg.GracefulStopDelay)
			s.hardStop(current)
		}
	}()

	return nil
}

// hardStop kills the control daemon and every MPS server still running.
func (s *Supervisor) hardStop(current *runningMPS) {
	if current.cmd.Process != nil {
		if err := current.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			s.logger.Warn("failed to kill MPS daemon", "error", err)
		}
	}
	s.killStrayServers()
}

// killStrayServers kills any nvidia-cuda-mps-server process on the node. mpsd
// runs in the host PID namespace, so they are visible in its own /proc.
func (s *Supervisor) killStrayServers() {
	pids, err := procfs.PIDsByExecutable(s.cfg.ProcRoot, mpsServerExecutable)
	if err != nil {
		s.logger.Warn("failed to scan for MPS server processes", "error", err)
		return
	}
	for _, pid := range pids {
		if err := s.cfg.KillProcess(pid); err != nil {
			s.logger.Warn("failed to kill MPS server process", "pid", pid, "error", err)
			continue
		}
		s.logger.Info("killed MPS server process", "pid", pid)
	}
}

func (s *Supervisor) setCurrent(current *runningMPS) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.current = current
}

func (s *Supervisor) isCurrent(current *runningMPS) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current == current
}

// takeRestartRequest reports whether the run that just ended was stopped by
// Restart, clearing the request as it does.
func (s *Supervisor) takeRestartRequest() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	reason, requested := s.restartReason, s.restartRequested
	s.restartReason, s.restartRequested = "", false
	return reason, requested
}

// killProcess is the production KillProcess: SIGKILL, treating an already-dead
// process as success.
func killProcess(pid int) error {
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		if errors.Is(err, syscall.ESRCH) {
			return nil
		}
		return err
	}
	return nil
}
