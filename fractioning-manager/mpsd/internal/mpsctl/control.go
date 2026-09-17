// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package mpsctl drives the nvidia-cuda-mps-control CLI: listing the MPS
// clients currently attached to the node's servers, and terminating one so it
// drains instead of dying mid-kernel.
//
// It shells out rather than speaking the control daemon's socket protocol
// directly. That protocol is not a published interface, whereas the CLI is the
// documented way to administer MPS — and the binary is already in the mpsd
// image, because mpsd supervises it.
package mpsctl

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	// DefaultCommandTimeout bounds a single control-daemon invocation that is
	// expected to answer immediately (e.g. listing clients).
	DefaultCommandTimeout = 10 * time.Second

	// maxPID is the largest value Linux will hand out with the highest
	// permitted /proc/sys/kernel/pid_max. Numbers above it in the CLI output
	// are certainly not PIDs.
	maxPID = 4 * 1024 * 1024
)

// csvFormatArgs asks the control daemon for machine-readable output.
//
// This is not a preference, it is a correctness requirement. The default table
// format truncates the PID column to fit its width: a client with PID 3165259
// prints as "316...", and on a node where PIDs are 7 digits that is every
// client. Parsing the table yields 316, which matches no process, so the drain
// finds nothing to terminate and silently does nothing — the failure mode the
// whole drain exists to prevent, reintroduced by the formatter. Verified
// against nvidia-cuda-mps-control on driver 615: `client list` supports
// --format=<table|csv>[,noheader], and the CSV form prints PIDs in full.
var csvFormatArgs = []string{"--format=csv,noheader"}

// Runner executes a command and returns its combined output. Production uses
// execRunner; tests substitute a fake.
type Runner func(ctx context.Context, name string, args ...string) ([]byte, error)

// Control talks to the node's MPS control daemon.
type Control struct {
	binary  string
	prefix  []string // args that precede every subcommand, e.g. ["-p", "3"]
	pipeDir string
	timeout time.Duration
	run     Runner
	log     *slog.Logger
}

// Options configures a Control.
type Options struct {
	// Binary is the nvidia-cuda-mps-control path.
	Binary string
	// ControlPort is the -p value; empty omits the flag. It must match the
	// port the supervised daemon was started with, or the CLI talks to nothing.
	ControlPort string
	// PipeDir is CUDA_MPS_PIPE_DIRECTORY — how the CLI finds the control socket.
	PipeDir string
	// Timeout bounds a single invocation. Zero uses DefaultCommandTimeout.
	Timeout time.Duration
	// Run overrides command execution (tests).
	Run Runner
	// Log defaults to slog.Default().
	Log *slog.Logger
}

// New builds a Control from opts.
func New(opts Options) *Control {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = DefaultCommandTimeout
	}
	run := opts.Run
	if run == nil {
		run = execRunner
	}
	log := opts.Log
	if log == nil {
		log = slog.Default()
	}

	var prefix []string
	if opts.ControlPort != "" {
		prefix = []string{"-p", opts.ControlPort}
	}

	return &Control{
		binary:  opts.Binary,
		prefix:  prefix,
		pipeDir: opts.PipeDir,
		timeout: timeout,
		run:     run,
		log:     log,
	}
}

// ClientPIDs returns every number the control daemon prints for `client list`,
// as the PIDs of the clients currently attached.
func (c *Control) ClientPIDs(ctx context.Context) ([]int, error) {
	args := append([]string{"client", "list"}, csvFormatArgs...)
	out, err := c.exec(ctx, c.timeout, args...)
	if err != nil {
		return nil, fmt.Errorf("listing MPS clients: %w", err)
	}
	return parsePIDs(out), nil
}

// TerminateClient asks the control daemon to terminate the client with the
// given PID. The command blocks until that client's outstanding GPU work has
// drained, which is the entire point: draining first is what stops a client
// killed mid-kernel from leaving a fault behind that wedges the server for
// every other tenant of the GPU.
//
// Because it blocks, a hung client makes this call hang too. timeout bounds it
// and a ctx deadline exceeded is returned to the caller, which treats it as
// evidence the server is already wedged.
func (c *Control) TerminateClient(ctx context.Context, pid int, timeout time.Duration) error {
	if timeout <= 0 {
		timeout = c.timeout
	}
	if _, err := c.exec(ctx, timeout, "client", "terminate", strconv.Itoa(pid)); err != nil {
		return fmt.Errorf("terminating MPS client %d: %w", pid, err)
	}
	return nil
}

func (c *Control) exec(ctx context.Context, timeout time.Duration, subcommand ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args := append(append([]string{}, c.prefix...), subcommand...)
	out, err := c.run(ctx, c.binary, args...)
	c.log.Debug("ran MPS control command",
		"binary", c.binary,
		"args", args,
		"output", string(bytes.TrimSpace(out)),
		"error", err,
	)
	if err != nil {
		// exec.CommandContext does not report the deadline: it kills the child
		// and returns "signal: killed". Drain distinguishes a wedged server from
		// an ordinary CLI failure by the deadline, so it has to be put back or
		// a timed-out terminate looks like any other error.
		if ctxErr := ctx.Err(); ctxErr != nil {
			return out, fmt.Errorf("%w: %v: %s", ctxErr, err, bytes.TrimSpace(out))
		}
		// The CLI writes its diagnostics to stdout/stderr, so surfacing the
		// output alongside the exit status is what makes a failure readable.
		return out, fmt.Errorf("%w: %s", err, bytes.TrimSpace(out))
	}
	return out, nil
}

// Env returns the environment a control-daemon invocation needs, for callers
// that build their own Runner.
func (c *Control) Env() []string {
	return []string{"CUDA_MPS_PIPE_DIRECTORY=" + c.pipeDir}
}

// PipeDir returns the configured CUDA_MPS_PIPE_DIRECTORY.
func (c *Control) PipeDir() string { return c.pipeDir }

// NewExecRunner returns a Runner that runs commands with pipeDir exported as
// CUDA_MPS_PIPE_DIRECTORY, which is how the CLI locates the control socket.
func NewExecRunner(pipeDir string) Runner {
	return func(ctx context.Context, name string, args ...string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, name, args...)
		cmd.Env = append(cmd.Environ(), "CUDA_MPS_PIPE_DIRECTORY="+pipeDir)
		return cmd.CombinedOutput()
	}
}

// execRunner is the fallback Runner, inheriting the process environment (which
// mpsd already sets CUDA_MPS_PIPE_DIRECTORY in for its own children).
func execRunner(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

// parsePIDs extracts plausible PIDs from control-daemon output, preserving
// order and dropping duplicates.
func parsePIDs(out []byte) []int {
	var pids []int
	seen := map[int]struct{}{}

	// One client per line, PID in the first CSV field. Reading that field
	// rather than scanning the whole line for digits matters: the other
	// columns carry a GPU UUID and a command name, both of which contain
	// digit runs that would otherwise be collected as bogus PIDs.
	for line := range strings.SplitSeq(string(out), "\n") {
		field, _, _ := strings.Cut(strings.TrimSpace(line), ",")
		value, err := strconv.Atoi(field)
		if err != nil || value <= 0 || value > maxPID {
			continue
		}
		if _, duplicate := seen[value]; duplicate {
			continue
		}
		seen[value] = struct{}{}
		pids = append(pids, value)
	}

	return pids
}
