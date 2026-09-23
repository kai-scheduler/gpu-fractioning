// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package portforward opens SPDY port-forwards to pods, the programmatic
// equivalent of `kubectl port-forward`.
package portforward

import (
	"fmt"
	"net/http"

	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
)

// Forwarder holds an active port-forward to a pod. Close must be called to
// release it.
type Forwarder struct {
	LocalPort int
	stopCh    chan struct{}
}

// Close tears down the port-forward.
func (f *Forwarder) Close() {
	close(f.stopCh)
}

// ToPod opens a SPDY port-forward to podPort on the named pod, equivalent to
// `kubectl port-forward pod/<name> <local>:<podPort>`. The returned forwarder
// is ready to use once ToPod returns.
func ToPod(c *cluster.Client, namespace, name string, podPort int) (*Forwarder, error) {
	url := c.RESTClient().Post().
		Resource("pods").
		Namespace(namespace).
		Name(name).
		SubResource("portforward").
		URL()

	transport, upgrader, err := spdy.RoundTripperFor(c.REST)
	if err != nil {
		return nil, fmt.Errorf("build spdy round tripper: %w", err)
	}
	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, "POST", url)

	stopCh := make(chan struct{})
	readyCh := make(chan struct{})
	errCh := make(chan error, 1)

	fw, err := portforward.New(dialer, []string{fmt.Sprintf("0:%d", podPort)}, stopCh, readyCh, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create port-forward to %s/%s: %w", namespace, name, err)
	}

	go func() {
		errCh <- fw.ForwardPorts()
	}()

	select {
	case <-readyCh:
	case err := <-errCh:
		return nil, fmt.Errorf("port-forward to %s/%s failed: %w", namespace, name, err)
	}

	ports, err := fw.GetPorts()
	if err != nil {
		close(stopCh)
		return nil, fmt.Errorf("get forwarded ports for %s/%s: %w", namespace, name, err)
	}
	if len(ports) == 0 {
		close(stopCh)
		return nil, fmt.Errorf("no forwarded ports returned for %s/%s", namespace, name)
	}

	return &Forwarder{
		LocalPort: int(ports[0].Local),
		stopCh:    stopCh,
	}, nil
}
