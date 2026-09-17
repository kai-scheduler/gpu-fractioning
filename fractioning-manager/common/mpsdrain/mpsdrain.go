// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package mpsdrain is the wire contract of the MPS client-drain endpoint, plus
// the client half of it.
//
// mpsd serves the endpoint (it owns the MPS control daemon and runs in the host
// PID namespace); fractiond calls it from its NRI StopContainer hook, because
// that is the only place that knows a container is about to be stopped while
// there is still time to act. The two are separate binaries in separate images,
// so the request and response shapes live here, in the package they both
// already depend on, rather than being duplicated on each side.
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
	// Path is the HTTP path the endpoint is served on.
	Path = "/drain"

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

// Client calls the drain endpoint over its unix socket.
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
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	body, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("encoding drain request: %w", err)
	}

	// The host is ignored by the unix-socket dialer but has to be syntactically
	// valid for net/http to build the request.
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://mpsd"+Path, bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("building drain request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("calling MPS drain endpoint %q: %w", c.socketPath, err)
	}
	defer func() {
		if err := httpResp.Body.Close(); err != nil {
			c.log.Debug("closing drain response body", "error", err)
		}
	}()

	payload, err := io.ReadAll(io.LimitReader(httpResp.Body, maxResponseBytes))
	if err != nil {
		return Response{}, fmt.Errorf("reading drain response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return Response{}, fmt.Errorf("MPS drain endpoint returned %s: %s", httpResp.Status, bytes.TrimSpace(payload))
	}

	var resp Response
	if err := json.Unmarshal(payload, &resp); err != nil {
		return Response{}, fmt.Errorf("decoding drain response: %w", err)
	}
	return resp, nil
}
