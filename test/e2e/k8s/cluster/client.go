// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package cluster connects to an existing Kubernetes cluster (whatever the
// kubeconfig points at). It never creates or destroys clusters.
package cluster

import (
	"fmt"

	"k8s.io/client-go/kubernetes/scheme"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/config"
)

// Client bundles the pieces of the cluster connection tests need: a
// scheme-aware controller-runtime client for CRUD and a rest.Config for
// building SPDY port-forwards. It is the common handle threaded through the
// nodes/plugin/pods/portforward packages. Matches the client type used by
// the operator's own reconcilers (operator/internal/controller).
type Client struct {
	Config *config.Config
	REST   *rest.Config
	Ctrl   ctrlclient.Client

	// core is a narrow raw REST client used only by portforward.ToPod to
	// build the pods/portforward subresource URL — controller-runtime's
	// client.Client has no equivalent, since it isn't a REST client.
	core corev1client.CoreV1Interface
}

// NewClient connects to the cluster pointed at by cfg.Kubeconfig.
func NewClient(cfg config.Config) (*Client, error) {
	restCfg, err := clientcmd.BuildConfigFromFlags("", cfg.Kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("load kubeconfig %q: %w", cfg.Kubeconfig, err)
	}

	ctrl, err := ctrlclient.New(restCfg, ctrlclient.Options{Scheme: scheme.Scheme})
	if err != nil {
		return nil, fmt.Errorf("build controller-runtime client: %w", err)
	}

	core, err := corev1client.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("build core/v1 REST client: %w", err)
	}

	return &Client{Config: &cfg, REST: restCfg, Ctrl: ctrl, core: core}, nil
}

// RESTClient exposes the raw core/v1 REST client for building subresource
// requests (e.g. portforward) that controller-runtime's client.Client can't.
func (c *Client) RESTClient() rest.Interface {
	return c.core.RESTClient()
}
