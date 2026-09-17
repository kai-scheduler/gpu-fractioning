// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsctl

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// recorder is a Runner that captures the exact argv (and the deadline the
// caller imposed) for every invocation. The argv is the whole contract with
// nvidia-cuda-mps-control: a missing -p means the CLI silently talks to a
// different control daemon than the one mpsd supervises.
type recorder struct {
	mu        sync.Mutex
	calls     [][]string
	binaries  []string
	deadlines []time.Duration
	out       []byte
	err       error
}

func (r *recorder) run(ctx context.Context, name string, args ...string) ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.binaries = append(r.binaries, name)
	r.calls = append(r.calls, args)
	if deadline, ok := ctx.Deadline(); ok {
		r.deadlines = append(r.deadlines, time.Until(deadline))
	} else {
		r.deadlines = append(r.deadlines, 0)
	}
	return r.out, r.err
}

func (r *recorder) call(t *testing.T, i int) []string {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if i >= len(r.calls) {
		t.Fatalf("wanted call %d, only %d were made", i, len(r.calls))
	}
	return r.calls[i]
}

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestParsePIDs(t *testing.T) {
	tests := []struct {
		name string
		out  string
		want []int
	}{
		{
			name: "empty output has no clients",
			out:  "",
		},
		{
			name: "whitespace only has no clients",
			out:  "\n\n   \t\n",
		},
		{
			name: "typical client list",
			out:  "1234\n5678\n",
			want: []int{1234, 5678},
		},
		{
			// The control daemon prints a banner and a prompt around the list;
			// none of it is a stable interface, so anything numeric in it is
			// collected and filtered later against the container's own PIDs.
			name: "surrounding prose does not hide the PIDs",
			out:  "Available commands:\n1234\n5678\nmps-control>",
			want: []int{1234, 5678},
		},
		{
			// PID 0 is the scheduler; terminating it is not a thing, and a
			// literal 0 in the output (a count, a column of zeroes) must never
			// become a target.
			name: "zero is not a PID",
			out:  "0\n0\n42\n",
			want: []int{42},
		},
		{
			// Reading the first CSV field rather than scanning for digit runs
			// means a sign is part of the field and fails to parse, instead of
			// being silently dropped to yield a positive PID that was never
			// there.
			name: "a negative number is not a PID",
			out:  "-5\n",
		},
		{
			name: "maxPID itself is still a plausible PID",
			out:  strconv.Itoa(maxPID),
			want: []int{maxPID},
		},
		{
			name: "one past maxPID cannot be a PID",
			out:  strconv.Itoa(maxPID + 1),
		},
		{
			// Atoi overflows rather than wrapping; a wrapped value could be a
			// live PID belonging to somebody else.
			name: "a number too large for int is dropped, not wrapped",
			out:  "123456789012345678901234567890\n",
		},
		{
			// Terminating the same client twice makes the second call fail and
			// look like a wedge.
			name: "duplicates are collapsed, first occurrence wins the order",
			out:  "77\n12\n77\n12\n99\n",
			want: []int{77, 12, 99},
		},
		{
			name: "digits embedded in a word are not PIDs",
			out:  "abc123 pid456def 0x1F\n",
		},
		{
			name: "a decimal is not a PID",
			out:  "12.34\n",
		},
		{
			name: "leading zeroes still parse",
			out:  "0007\n",
			want: []int{7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parsePIDs([]byte(tt.out))
			if !slices.Equal(got, tt.want) {
				t.Errorf("parsePIDs(%q) = %v, want %v", tt.out, got, tt.want)
			}
		})
	}
}

func TestParsePIDsHugeOutput(t *testing.T) {
	// A control daemon that goes haywire and prints megabytes must not make the
	// drain path quadratic or allocate per duplicate: the drain runs on the
	// container-stop path, where slowness is a stuck pod.
	var b strings.Builder
	for i := 0; i < 20000; i++ {
		fmt.Fprintf(&b, "%d\n", i%50+1)
	}

	got := parsePIDs([]byte(b.String()))
	if len(got) != 50 {
		t.Fatalf("got %d unique PIDs, want 50", len(got))
	}
	for i, pid := range got {
		if pid != i+1 {
			t.Fatalf("PID %d = %d, want %d (order must follow first appearance)", i, pid, i+1)
		}
	}
}

func TestClientPIDsArgs(t *testing.T) {
	tests := []struct {
		name        string
		controlPort string
		want        []string
	}{
		{
			// The supervised daemon is started with -p 3; without the same flag
			// the CLI would try the default port and report no clients at all,
			// which drain would read as "nothing to drain".
			name:        "control port is prefixed to every subcommand",
			controlPort: "3",
			want:        []string{"-p", "3", "client", "list", "--format=csv,noheader"},
		},
		{
			// A blank port is the documented escape hatch for dropping the flag.
			name:        "empty control port omits -p entirely",
			controlPort: "",
			want:        []string{"client", "list", "--format=csv,noheader"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := &recorder{out: []byte("11\n")}
			c := New(Options{Binary: "/usr/bin/nvidia-cuda-mps-control", ControlPort: tt.controlPort, Run: rec.run, Log: quietLogger()})

			pids, err := c.ClientPIDs(context.Background())
			if err != nil {
				t.Fatalf("ClientPIDs() error = %v", err)
			}
			if !slices.Equal(pids, []int{11}) {
				t.Errorf("ClientPIDs() = %v, want [11]", pids)
			}
			if got := rec.call(t, 0); !slices.Equal(got, tt.want) {
				t.Errorf("argv = %v, want %v", got, tt.want)
			}
			if rec.binaries[0] != "/usr/bin/nvidia-cuda-mps-control" {
				t.Errorf("binary = %q, want the configured path", rec.binaries[0])
			}
		})
	}
}

func TestTerminateClientArgs(t *testing.T) {
	rec := &recorder{}
	c := New(Options{Binary: "mps", ControlPort: "3", Run: rec.run, Log: quietLogger()})

	if err := c.TerminateClient(context.Background(), 4242, time.Minute); err != nil {
		t.Fatalf("TerminateClient() error = %v", err)
	}

	want := []string{"-p", "3", "client", "terminate", "4242"}
	if got := rec.call(t, 0); !slices.Equal(got, want) {
		t.Errorf("argv = %v, want %v", got, want)
	}
}

func TestPrefixIsNotSharedBetweenCalls(t *testing.T) {
	// The prefix slice is reused for every invocation. If it were appended to
	// in place, the second command would inherit the first one's subcommand and
	// terminate an arbitrary client.
	rec := &recorder{out: []byte("7\n")}
	c := New(Options{Binary: "mps", ControlPort: "3", Run: rec.run, Log: quietLogger()})

	if _, err := c.ClientPIDs(context.Background()); err != nil {
		t.Fatalf("ClientPIDs() error = %v", err)
	}
	if err := c.TerminateClient(context.Background(), 7, time.Minute); err != nil {
		t.Fatalf("TerminateClient() error = %v", err)
	}
	if _, err := c.ClientPIDs(context.Background()); err != nil {
		t.Fatalf("second ClientPIDs() error = %v", err)
	}

	wantList := []string{"-p", "3", "client", "list", "--format=csv,noheader"}
	if got := rec.call(t, 0); !slices.Equal(got, wantList) {
		t.Errorf("first argv = %v, want %v", got, wantList)
	}
	if got := rec.call(t, 2); !slices.Equal(got, wantList) {
		t.Errorf("third argv = %v, want %v (prefix was mutated by the second call)", got, wantList)
	}
}

func TestCommandOutputIsSurfacedInTheError(t *testing.T) {
	// The CLI reports its real diagnostics on stdout and exits non-zero with a
	// useless "exit status 1". Dropping the output turns every MPS failure into
	// an unactionable log line.
	rec := &recorder{
		out: []byte("  Cannot connect to the MPS control daemon  \n"),
		err: errors.New("exit status 1"),
	}
	c := New(Options{Binary: "mps", Run: rec.run, Log: quietLogger()})

	_, err := c.ClientPIDs(context.Background())
	if err == nil {
		t.Fatal("ClientPIDs() error = nil, want the runner's failure")
	}
	for _, want := range []string{"listing MPS clients", "exit status 1", "Cannot connect to the MPS control daemon"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}

	termErr := c.TerminateClient(context.Background(), 99, time.Second)
	if termErr == nil {
		t.Fatal("TerminateClient() error = nil, want the runner's failure")
	}
	if !strings.Contains(termErr.Error(), "terminating MPS client 99") {
		t.Errorf("error %q does not name the client that failed", termErr)
	}
}

func TestErrorWithNoOutputDoesNotPanic(t *testing.T) {
	// A runner that fails before producing output returns nil bytes; trimming
	// nil must not be a nil deref on the failure path, which is exactly the
	// path nobody exercises by hand.
	rec := &recorder{out: nil, err: errors.New("fork/exec: no such file or directory")}
	c := New(Options{Binary: "missing", Run: rec.run, Log: quietLogger()})

	if _, err := c.ClientPIDs(context.Background()); err == nil {
		t.Fatal("ClientPIDs() error = nil, want the exec failure")
	}
}

func TestTerminateClientTimeoutIsPlumbedThrough(t *testing.T) {
	tests := []struct {
		name        string
		optsTimeout time.Duration
		argTimeout  time.Duration
		wantAtMost  time.Duration
	}{
		{
			name:        "explicit timeout wins",
			optsTimeout: time.Hour,
			argTimeout:  40 * time.Millisecond,
			wantAtMost:  time.Second,
		},
		{
			// Zero must fall back to the configured command timeout rather than
			// meaning "no deadline": a terminate that never returns is the wedge
			// signature drain relies on detecting.
			name:        "zero falls back to the configured timeout",
			optsTimeout: 40 * time.Millisecond,
			argTimeout:  0,
			wantAtMost:  time.Second,
		},
		{
			name:        "negative falls back to the configured timeout",
			optsTimeout: 40 * time.Millisecond,
			argTimeout:  -time.Hour,
			wantAtMost:  time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// A runner that blocks until its context dies is exactly what a
			// terminate against a wedged server does.
			blocking := func(ctx context.Context, _ string, _ ...string) ([]byte, error) {
				<-ctx.Done()
				return []byte("still draining"), ctx.Err()
			}
			c := New(Options{Binary: "mps", Timeout: tt.optsTimeout, Run: blocking, Log: quietLogger()})

			done := make(chan error, 1)
			start := time.Now()
			go func() { done <- c.TerminateClient(context.Background(), 5, tt.argTimeout) }()

			select {
			case err := <-done:
				if err == nil {
					t.Fatal("TerminateClient() error = nil, want a deadline error")
				}
				// drain treats a timed-out terminate as a wedged server and
				// escalates to a non-graceful MPS restart, so the caller has to
				// be able to recognise the deadline through the wrapping.
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Errorf("error %v is not a context.DeadlineExceeded; drain cannot tell a wedge from a CLI failure", err)
				}
				if elapsed := time.Since(start); elapsed > tt.wantAtMost {
					t.Errorf("took %v to give up, want under %v", elapsed, tt.wantAtMost)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("TerminateClient() never returned: the timeout is not wired to the runner's context")
			}
		})
	}
}

func TestTerminateClientReportsDeadlineEvenWhenTheRunnerHidesIt(t *testing.T) {
	// exec.CommandContext does not return ctx.Err(): it kills the child and
	// reports "signal: killed". If that is passed straight through, drain sees
	// an ordinary CLI failure instead of the wedge it must escalate on.
	hidesDeadline := func(ctx context.Context, _ string, _ ...string) ([]byte, error) {
		<-ctx.Done()
		return []byte("terminating client 5"), errors.New("signal: killed")
	}
	c := New(Options{Binary: "mps", Run: hidesDeadline, Log: quietLogger()})

	err := c.TerminateClient(context.Background(), 5, 30*time.Millisecond)
	if err == nil {
		t.Fatal("TerminateClient() error = nil, want a deadline error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error %v is not a context.DeadlineExceeded", err)
	}
	if !strings.Contains(err.Error(), "signal: killed") {
		t.Errorf("error %v dropped the underlying failure", err)
	}
}

func TestParentContextCancellationPropagates(t *testing.T) {
	// mpsd's shutdown cancels the drain server's context. A terminate that
	// ignored it would hold the process open for the full drain timeout.
	started := make(chan struct{})
	blocking := func(ctx context.Context, _ string, _ ...string) ([]byte, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	c := New(Options{Binary: "mps", Timeout: time.Hour, Run: blocking, Log: quietLogger()})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- c.TerminateClient(ctx, 5, time.Hour) }()

	<-started
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("error = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("TerminateClient() ignored the parent context")
	}
}

func TestDefaultTimeoutApplies(t *testing.T) {
	rec := &recorder{}
	c := New(Options{Binary: "mps", Run: rec.run, Log: quietLogger()})

	if _, err := c.ClientPIDs(context.Background()); err != nil {
		t.Fatalf("ClientPIDs() error = %v", err)
	}
	// Every invocation must carry a deadline; an unbounded one blocks the
	// container-stop path forever.
	if rec.deadlines[0] <= 0 || rec.deadlines[0] > DefaultCommandTimeout {
		t.Errorf("deadline = %v, want (0, %v]", rec.deadlines[0], DefaultCommandTimeout)
	}
}

func TestEnvAndPipeDir(t *testing.T) {
	// Callers that build their own Runner rely on Env() to find the control
	// socket; a wrong or empty value silently talks to the wrong daemon.
	c := New(Options{Binary: "mps", PipeDir: "/tmp/pipe", Log: quietLogger()})

	if got := c.PipeDir(); got != "/tmp/pipe" {
		t.Errorf("PipeDir() = %q, want /tmp/pipe", got)
	}
	want := []string{"CUDA_MPS_PIPE_DIRECTORY=/tmp/pipe"}
	if got := c.Env(); !slices.Equal(got, want) {
		t.Errorf("Env() = %v, want %v", got, want)
	}
}

func TestNewExecRunnerExportsPipeDirectory(t *testing.T) {
	// The CLI locates the control socket only through this variable, and the
	// runner is the single place it is set for a self-built Runner.
	dir := t.TempDir()
	out, err := NewExecRunner(dir)(context.Background(), "/bin/sh", "-c", `printf %s "$CUDA_MPS_PIPE_DIRECTORY"`)
	if err != nil {
		t.Fatalf("runner error = %v (output %q)", err, out)
	}
	if string(out) != dir {
		t.Errorf("CUDA_MPS_PIPE_DIRECTORY = %q, want %q", out, dir)
	}
}

func TestDefaultRunnerIsWiredWhenRunIsNil(t *testing.T) {
	// Production leaves Options.Run unset; a nil Runner would be a nil call.
	c := New(Options{Binary: "/bin/echo", Log: quietLogger()})

	pids, err := c.ClientPIDs(context.Background())
	if err != nil {
		t.Fatalf("ClientPIDs() error = %v", err)
	}
	if len(pids) != 0 {
		t.Errorf("ClientPIDs() = %v, want none from `echo client list`", pids)
	}
}

func TestNilLogDoesNotPanic(t *testing.T) {
	// Options.Log is optional and the debug line runs on every command.
	c := New(Options{Binary: "mps", Run: func(context.Context, string, ...string) ([]byte, error) {
		return []byte("1\n"), nil
	}})
	if _, err := c.ClientPIDs(context.Background()); err != nil {
		t.Fatalf("ClientPIDs() error = %v", err)
	}
}

// The bug this whole CSV format exists to avoid, pinned with output captured
// verbatim from nvidia-cuda-mps-control on driver 615.
//
// `client list` in its default table format truncates the PID column to fit:
// a client whose real PID is 3165259 prints as "316...". Parsed, that yields
// 316 — a PID that belongs to no container, so the drain's intersection finds
// no targets and terminates nothing. The drain reports success and silently
// does not drain, which is exactly the orphaned-client wedge it was written to
// prevent. Nothing about it is visible in a log.
func TestParsePIDsRejectsTheTruncatedTableFormat(t *testing.T) {
	table := "PID     SERVER        DEVICE                                         MAWS NS                   CMD\n" +
		"316...  srv           GPU-cc97390c-c189-71f5-1bb7-5b0af481fa0d       srv/default (0)           VLLM::EngineCore\n" +
		"316...  srv           GPU-cc97390c-c189-71f5-1bb7-5b0af481fa0d       srv/default (0)           VLLM::EngineCore\n"

	if got := parsePIDs([]byte(table)); len(got) != 0 {
		t.Errorf("parsePIDs(table output) = %v, want none: a truncated PID must never be mistaken for a real one", got)
	}

	// The CSV form of the same two clients, also captured verbatim.
	csv := "3167082,srv,GPU-cc97390c-c189-71f5-1bb7-5b0af481fa0d,srv/default (0),VLLM::EngineCore\n" +
		"3165259,srv,GPU-cc97390c-c189-71f5-1bb7-5b0af481fa0d,srv/default (0),VLLM::EngineCore\n"

	want := []int{3167082, 3165259}
	if got := parsePIDs([]byte(csv)); !slices.Equal(got, want) {
		t.Errorf("parsePIDs(csv output) = %v, want %v", got, want)
	}
}
