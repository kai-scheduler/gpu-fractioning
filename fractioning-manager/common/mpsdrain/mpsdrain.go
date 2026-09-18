// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package mpsdrain is the wire contract of the unix-socket channel from
// fractiond to mpsd, plus the client half of it.
//
// mpsd serves it (it owns the MPS control daemon and runs in the host PID
// namespace); fractiond calls it from its NRI hooks, because those are the only
// places that know what is happening to a container while there is still time
// to act. The two are separate binaries in separate images, so the request and
// response shapes live here, in the package they both already depend on, rather
// than being duplicated on each side.
//
// Three operations share the one socket:
//
//   - drain (Path), from StopContainer: terminate the container's MPS clients
//     through the control daemon before the runtime kills them.
//   - provision (PathProvision), from CreateContainer: get the container an MPS
//     namespace capped at its compute portion, and the pipe directory that puts
//     it inside that namespace. It runs in CreateContainer because the cap has
//     to be set before the container's first CUDA process exists — `namespace
//     set` fails once a client is attached.
//   - release (PathRelease), from RemoveContainer: give the namespace back.
//
// The name is now narrower than the package, which is kept deliberately: this
// is one channel between two daemons and splitting it would mean a second
// socket, a second mount, and a second way for the operator's wiring to be
// silently wrong.
package mpsdrain

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/daemonpaths"
)

const (
	// Path is the HTTP path the drain endpoint is served on.
	Path = "/drain"

	// PathProvision is where a container's MPS namespace is asked for, and
	// PathRelease is where it is given back.
	PathProvision = "/namespace/provision"
	PathRelease   = "/namespace/release"

	// DefaultSocketPath is the unix socket mpsd listens on and fractiond dials,
	// and the path the operator mounts into both pods. All three have to name
	// the same file and a mismatch between them has no symptom, so the value
	// lives in one place both modules import rather than being repeated here.
	DefaultSocketPath = daemonpaths.MPSDrainSocket

	// DefaultCallTimeout bounds a drain call from the caller's side. It is far
	// shorter than the server's per-client drain timeout, because the caller is
	// an NRI StopContainer hook and the runtime puts a hard deadline on it:
	// containerd wraps every plugin request in its plugin_request_timeout (2s by
	// default) and treats the resulting context.DeadlineExceeded as *fatal* —
	// it closes the plugin connection outright. A drain that overran it would
	// therefore not merely delay one container stop, it would disconnect
	// fractiond from NRI, and while disconnected newly created fractional
	// containers get no limits injected at all. Giving up well inside that
	// window costs nothing: mpsd finishes the drain in the background either
	// way, and the caller's only job is to not be the reason enforcement drops.
	//
	// Callers that can see the timeout the runtime actually negotiated should
	// clamp against that instead of relying on this default.
	DefaultCallTimeout = time.Second

	// maxResponseBytes caps how much of a response is read. The body is a small
	// fixed struct; anything larger means something other than mpsd is on the
	// socket.
	maxResponseBytes = 1 << 16
)

// Request is the body posted to Path.
type Request struct {
	// ContainerID is the runtime's container id, which is how the server works
	// out which MPS clients belong to the container being stopped.
	ContainerID string `json:"containerId"`
	// Pod and Namespace are carried for logging only.
	Pod       string `json:"pod,omitempty"`
	Namespace string `json:"namespace,omitempty"`
}

// Response is the body returned from Path.
type Response struct {
	// Drained lists the client PIDs that were terminated through MPS.
	Drained []int `json:"drained,omitempty"`
	// Wedged reports that a terminate did not complete in time, meaning the MPS
	// server was already wedged before this drain started.
	Wedged bool `json:"wedged,omitempty"`
	// Restarted reports that MPS was restarted, either to clear a wedge or
	// because this drain left the node with no MPS clients.
	Restarted bool `json:"restarted,omitempty"`
	// Message carries a human-readable summary, including why a drain could not
	// be completed.
	Message string `json:"message,omitempty"`
}

// ProvisionRequest asks mpsd for the container's capped MPS namespace.
type ProvisionRequest struct {
	// ContainerID is the runtime's container id. It is the whole identity of
	// the namespace: mpsd derives the namespace name from it, and it is what
	// ReleaseRequest gives back.
	ContainerID string `json:"containerId"`
	// ActiveThreadPercent is the ceiling to put on the namespace, 1..100.
	//
	// A whole percent rather than the scheduler's (0, 1] portion, because the
	// portion→percent conversion is where the rounding rule lives (floor, with
	// a floor of 1, so N tenants of one GPU are never promised more than 100%
	// of it) and that rule must exist in exactly one place. fractiond has
	// already applied it — see annotations.ComputePercent — and sending the
	// raw portion would mean implementing it a second time in mpsd, where a
	// divergence between the two would be invisible.
	ActiveThreadPercent int `json:"activeThreadPercent"`
	// GPUUUIDs are the GPUs the scheduler assigned the container. mpsd records
	// them against the namespace for diagnostics; an MPS namespace belongs to a
	// server rather than to a device, so they do not scope the cap.
	GPUUUIDs []string `json:"gpuUuids,omitempty"`
	// Pod and Namespace are carried for logging only.
	Pod       string `json:"pod,omitempty"`
	Namespace string `json:"namespace,omitempty"`
}

// ProvisionResponse is what the container has to be pointed at.
type ProvisionResponse struct {
	// PipeDirectory is the host directory holding the namespace's control
	// socket — the ONLY path that may be bind-mounted into the container. It is
	// read back from the control daemon rather than derived from a layout, and
	// a container handed the server's directory instead of this one silently
	// lands in the uncapped `default` namespace.
	PipeDirectory string `json:"pipeDirectory"`
	// Namespace and Server name the namespace that was provisioned.
	Namespace string `json:"namespace"`
	Server    string `json:"server"`
	// ActiveThreadPercent is the ceiling that was applied.
	ActiveThreadPercent int `json:"activeThreadPercent"`
}

// ReleaseRequest gives a container's namespace back.
type ReleaseRequest struct {
	// ContainerID is the runtime's container id.
	ContainerID string `json:"containerId"`
	// Pod and Namespace are carried for logging only.
	Pod       string `json:"pod,omitempty"`
	Namespace string `json:"namespace,omitempty"`
}

// ReleaseResponse reports what happened to the namespace.
type ReleaseResponse struct {
	// Deleted is true when the namespace is gone. False is normal rather than a
	// failure: a namespace with a client still attached cannot be deleted, and
	// mpsd retries in the background until it can.
	Deleted bool `json:"deleted,omitempty"`
	// Namespace names what was released, when there was anything to release.
	Namespace string `json:"namespace,omitempty"`
	// Message carries a human-readable summary.
	Message string `json:"message,omitempty"`
}

// Client calls the mpsd endpoints over their unix socket.
type Client struct {
	socketPath string
	timeout    time.Duration
	httpClient *http.Client
	log        *slog.Logger
}

// NewClient returns a Client for the endpoint at socketPath. A non-positive
// timeout uses DefaultCallTimeout; a nil log uses slog.Default().
func NewClient(socketPath string, timeout time.Duration, log *slog.Logger) *Client {
	if socketPath == "" {
		socketPath = DefaultSocketPath
	}
	if timeout <= 0 {
		timeout = DefaultCallTimeout
	}
	if log == nil {
		log = slog.Default()
	}

	return &Client{
		socketPath: socketPath,
		timeout:    timeout,
		log:        log,
		httpClient: &http.Client{
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "unix", socketPath)
				},
			},
		},
	}
}

// SocketPath returns the endpoint socket this client dials.
func (c *Client) SocketPath() string { return c.socketPath }

// Drain asks mpsd to drain the container's MPS clients, bounded by the client's
// timeout.
//
// The error is informational: every caller is on a path that is about to stop a
// container anyway, so a drain that fails must be logged and stepped over, not
// retried or escalated. Failing to drain costs the GPU its protection against a
// wedge; refusing to stop the container would cost the cluster a stuck pod.
func (c *Client) Drain(ctx context.Context, req Request) (Response, error) {
	return post[Request, Response](c, ctx, Path, "drain", req)
}

// Provision asks mpsd for the container's capped MPS namespace and returns the
// pipe directory that puts the container inside it.
//
// Unlike Drain, the error here is NOT informational: a container whose
// namespace could not be provisioned has no enforceable compute cap, and
// starting it anyway would hand it the whole card. The caller is expected to
// fail the container's creation.
func (c *Client) Provision(ctx context.Context, req ProvisionRequest) (ProvisionResponse, error) {
	return post[ProvisionRequest, ProvisionResponse](c, ctx, PathProvision, "namespace provision", req)
}

// Release gives the container's namespace back.
//
// Like Drain, a failure is informational: the container is going away either
// way, and mpsd reclaims a namespace whose container is gone during its own
// reconciliation, so a lost release costs a delay rather than a leak.
func (c *Client) Release(ctx context.Context, req ReleaseRequest) (ReleaseResponse, error) {
	return post[ReleaseRequest, ReleaseResponse](c, ctx, PathRelease, "namespace release", req)
}

// post is the one request path all three operations share: encode, POST over
// the unix socket under the client's timeout, and decode. A free function
// rather than a method because Go methods cannot take type parameters.
func post[Req, Resp any](c *Client, ctx context.Context, path, what string, req Req) (Resp, error) {
	var resp Resp

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	body, err := json.Marshal(req)
	if err != nil {
		return resp, fmt.Errorf("encoding %s request: %w", what, err)
	}

	// The host is ignored by the unix-socket dialer but has to be syntactically
	// valid for net/http to build the request.
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://mpsd"+path, bytes.NewReader(body))
	if err != nil {
		return resp, fmt.Errorf("building %s request: %w", what, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return resp, fmt.Errorf("calling mpsd %s endpoint %q: %w", what, c.socketPath, err)
	}
	defer func() {
		if err := httpResp.Body.Close(); err != nil {
			c.log.Debug("closing mpsd response body", "path", path, "error", err)
		}
	}()

	payload, err := io.ReadAll(io.LimitReader(httpResp.Body, maxResponseBytes))
	if err != nil {
		return resp, fmt.Errorf("reading %s response: %w", what, err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return resp, fmt.Errorf("mpsd %s endpoint returned %s: %s", what, httpResp.Status, bytes.TrimSpace(payload))
	}

	if err := json.Unmarshal(payload, &resp); err != nil {
		return resp, fmt.Errorf("decoding %s response: %w", what, err)
	}
	return resp, nil
}
