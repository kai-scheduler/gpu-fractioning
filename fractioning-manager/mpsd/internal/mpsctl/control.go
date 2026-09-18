// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package mpsctl drives the nvidia-cuda-mps-control CLI: listing the MPS
// clients currently attached to the node's servers, terminating one so it
// drains instead of dying mid-kernel, and administering the per-container MPS
// namespaces that carry each fractional container's compute cap.
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
	"path/filepath"
	"regexp"
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

// ── MPS namespaces ──────────────────────────────────────────────────────────
//
// A namespace is the unit the control daemon enforces a compute cap on. Per
// namespace, `--active-thread-percentage` is an authoritative ceiling: a client
// in that namespace that exports CUDA_MPS_ACTIVE_THREAD_PERCENTAGE can only
// lower its own share, never raise it past the namespace's. That is the entire
// reason this file grew a namespace API — the env var alone is advisory, and
// any process in a container can re-export it before cuInit.
//
// The verbs below mirror `<module> <verb> --help` on the protocol-3 control
// daemon (driver 615). Their exact argument shapes are recorded in one place
// each, so a correction is a one-line change rather than a hunt.

const (
	// namespaceModule is the control-daemon module that owns namespaces.
	namespaceModule = "namespace"

	// pipeDirectoryField is the `namespace get` field holding the directory the
	// namespace's control socket lives in — the directory a container must be
	// handed as CUDA_MPS_PIPE_DIRECTORY to be inside the namespace (and thus
	// under its cap).
	//
	// It is READ rather than derived. The layout happens to be
	// <pipeDir>/<server>/<namespace> today, but a container handed a path that
	// is wrong in a way the daemon tolerates — the server directory instead of
	// the namespace directory, say — silently lands in the uncapped `default`
	// namespace. A guess that is wrong fails loudly here; a guess that is
	// wrong-but-plausible fails silently on the GPU.
	pipeDirectoryField = "pipe_directory"
)

// namespaceNamePattern is the only shape the control daemon accepts for a
// namespace name. It is enforced here, at the boundary, because a name outside
// it is rejected by the daemon with a message that says nothing about which of
// the caller's inputs produced it.
var namespaceNamePattern = regexp.MustCompile(`^[a-z0-9_]+$`)

// ValidNamespaceName reports whether name is one the control daemon will
// accept: lowercase letters, digits and underscores only. No uppercase and no
// hyphens, which rules out most Kubernetes object names verbatim.
func ValidNamespaceName(name string) bool { return namespaceNamePattern.MatchString(name) }

// CreateNamespace creates a namespace on the given server.
//
// Creating a namespace that already exists is an error from the daemon, so
// callers that must be idempotent should treat a create failure as inconclusive
// and settle it with NamespacePipeDirectory, which answers the only question
// that matters: is the namespace there and usable.
func (c *Control) CreateNamespace(ctx context.Context, server, name string) error {
	if err := checkNamespaceArgs(server, name); err != nil {
		return err
	}
	if _, err := c.exec(ctx, c.timeout, namespaceModule, "create", name, "--server="+server); err != nil {
		return fmt.Errorf("creating MPS namespace %q on server %q: %w", name, server, err)
	}
	return nil
}

// SetNamespaceActiveThreadPercentage sets the namespace's SM ceiling.
//
// It FAILS while the namespace has an active client, which is not a bug to work
// around but the property the whole design rests on: a cap that could be
// changed under a running client would be a cap the client's own lifecycle
// could race. Every call therefore has to happen before the container's CUDA
// process starts — which is why provisioning runs from fractiond's NRI
// CreateContainer hook and not from anywhere later.
func (c *Control) SetNamespaceActiveThreadPercentage(ctx context.Context, server, name string, percent int) error {
	if err := checkNamespaceArgs(server, name); err != nil {
		return err
	}
	if percent < 1 || percent > 100 {
		return fmt.Errorf("active thread percentage %d for MPS namespace %q is out of range, expected 1..100", percent, name)
	}
	_, err := c.exec(ctx, c.timeout, namespaceModule, "set", name,
		"--server="+server,
		"--active-thread-percentage="+strconv.Itoa(percent))
	if err != nil {
		return fmt.Errorf("setting active thread percentage %d%% on MPS namespace %q (server %q): %w", percent, name, server, err)
	}
	return nil
}

// DeleteNamespace removes a namespace from the server. Like `namespace set`, it
// fails while a client is still attached, so callers delete on container
// teardown and retry rather than assuming a single attempt succeeds.
func (c *Control) DeleteNamespace(ctx context.Context, server, name string) error {
	if err := checkNamespaceArgs(server, name); err != nil {
		return err
	}
	if _, err := c.exec(ctx, c.timeout, namespaceModule, "delete", name, "--server="+server); err != nil {
		return fmt.Errorf("deleting MPS namespace %q on server %q: %w", name, server, err)
	}
	return nil
}

// NamespacePipeDirectory returns the directory holding the namespace's control
// socket — the value a container gets as CUDA_MPS_PIPE_DIRECTORY, and the
// single bind-mount source that puts it inside the namespace.
//
// It doubles as the existence check: a namespace that is not there has no pipe
// directory to report, so an error here means "not usable", which is exactly
// what an idempotent provision needs to know.
func (c *Control) NamespacePipeDirectory(ctx context.Context, server, name string) (string, error) {
	if err := checkNamespaceArgs(server, name); err != nil {
		return "", err
	}
	out, err := c.exec(ctx, c.timeout, namespaceModule, "get", name, server, pipeDirectoryField)
	if err != nil {
		return "", fmt.Errorf("reading %s of MPS namespace %q on server %q: %w", pipeDirectoryField, name, server, err)
	}
	dir := parseAbsolutePath(out)
	if dir == "" {
		return "", fmt.Errorf("reading %s of MPS namespace %q on server %q: no absolute path in %q",
			pipeDirectoryField, name, server, bytes.TrimSpace(out))
	}
	return dir, nil
}

// NamespaceNames lists the namespaces the server currently has.
//
// The parse is deliberately permissive — it collects every token on every line
// that is a syntactically valid namespace name — because the output format is
// not a published interface and a banner, a prompt or a header column must not
// be able to hide a real namespace. Permissiveness is safe here only because of
// how the caller uses the result: reconciliation acts on names carrying its own
// generated prefix and ignores everything else, so a stray token picked up from
// a header can never become a namespace somebody's container is using and we
// delete.
func (c *Control) NamespaceNames(ctx context.Context, server string) ([]string, error) {
	if server == "" {
		return nil, fmt.Errorf("listing MPS namespaces: server name is empty")
	}
	out, err := c.exec(ctx, c.timeout, namespaceModule, "list", "--server="+server)
	if err != nil {
		return nil, fmt.Errorf("listing MPS namespaces on server %q: %w", server, err)
	}
	return parseNamespaceNames(out), nil
}

// checkNamespaceArgs rejects the inputs the daemon would reject, with an error
// that names which one was wrong.
func checkNamespaceArgs(server, name string) error {
	if server == "" {
		return fmt.Errorf("MPS namespace operation needs a server name")
	}
	if !ValidNamespaceName(name) {
		return fmt.Errorf("invalid MPS namespace name %q: only lowercase letters, digits and underscores are accepted", name)
	}
	return nil
}

// parseAbsolutePath pulls the first absolute path out of command output,
// tolerating the shapes the CLI might print it in ("value", "key: value",
// "key=value"). Anything that is not an absolute path is not a pipe directory,
// and returning "" rather than a best guess keeps a misread out of a container's
// CUDA_MPS_PIPE_DIRECTORY.
func parseAbsolutePath(out []byte) string {
	for line := range strings.SplitSeq(string(out), "\n") {
		for field := range strings.FieldsSeq(strings.ReplaceAll(strings.ReplaceAll(line, "=", " "), ",", " ")) {
			field = strings.Trim(field, `"'`)
			if strings.HasPrefix(field, "/") {
				return filepath.Clean(field)
			}
		}
	}
	return ""
}

// parseNamespaceNames collects every valid namespace name in the output,
// preserving order and dropping duplicates.
func parseNamespaceNames(out []byte) []string {
	var names []string
	seen := map[string]struct{}{}

	for line := range strings.SplitSeq(string(out), "\n") {
		for field := range strings.FieldsSeq(strings.ReplaceAll(line, ",", " ")) {
			field = strings.Trim(field, `"'`)
			if !ValidNamespaceName(field) {
				continue
			}
			if _, duplicate := seen[field]; duplicate {
				continue
			}
			seen[field] = struct{}{}
			names = append(names, field)
		}
	}
	return names
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
