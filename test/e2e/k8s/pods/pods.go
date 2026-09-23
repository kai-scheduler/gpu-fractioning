// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package pods provides list and exec helpers for pods running in the
// cluster.
package pods

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/remotecommand"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
)

// ListByLabel lists pods in namespace matching labelSelector.
func ListByLabel(ctx context.Context, c *cluster.Client, namespace, labelSelector string) ([]corev1.Pod, error) {
	sel, err := labels.Parse(labelSelector)
	if err != nil {
		return nil, fmt.Errorf("parse label selector %q: %w", labelSelector, err)
	}

	var list corev1.PodList
	if err := c.Ctrl.List(ctx, &list, ctrlclient.InNamespace(namespace), ctrlclient.MatchingLabelsSelector{Selector: sel}); err != nil {
		return nil, fmt.Errorf("list pods matching %q in namespace %s: %w", labelSelector, namespace, err)
	}
	return list.Items, nil
}

// FirstByLabel returns one pod in namespace matching labelSelector, capping the
// server-side list to a single item (for callers that just need any one pod, not
// the full set). Returns (nil, nil) when none match.
func FirstByLabel(ctx context.Context, c *cluster.Client, namespace, labelSelector string) (*corev1.Pod, error) {
	sel, err := labels.Parse(labelSelector)
	if err != nil {
		return nil, fmt.Errorf("parse label selector %q: %w", labelSelector, err)
	}

	var list corev1.PodList
	if err := c.Ctrl.List(ctx, &list, ctrlclient.InNamespace(namespace), ctrlclient.MatchingLabelsSelector{Selector: sel}, ctrlclient.Limit(1)); err != nil {
		return nil, fmt.Errorf("list pods matching %q in namespace %s: %w", labelSelector, namespace, err)
	}
	if len(list.Items) == 0 {
		return nil, nil
	}
	return &list.Items[0], nil
}

// Exec runs command inside container of the named pod and returns its
// stdout, equivalent to `kubectl exec <pod> -c <container> -- <command>`.
// Used by attribution tests to read env vars (e.g.
// MOCK_NVIDIA_VISIBLE_DEVICES) the device plugin injected into a running
// container — information not visible on the stored Pod spec.
func Exec(ctx context.Context, c *cluster.Client, namespace, name, container string, command []string) (string, error) {
	return ExecStdin(ctx, c, namespace, name, container, command, "")
}

// ExecStdin is Exec with stdin piped into the command, equivalent to
// `printf %s <stdin> | kubectl exec -i <pod> -c <container> -- <command>`.
// Passing data on stdin (rather than as a command argument) keeps it off the
// process's own command line — nvmlmock.HostPID relies on this so the marker
// it searches for never appears in the search process's /proc/<pid>/cmdline.
func ExecStdin(ctx context.Context, c *cluster.Client, namespace, name, container string, command []string, stdin string) (string, error) {
	opts := &corev1.PodExecOptions{
		Container: container,
		Command:   command,
		Stdout:    true,
		Stderr:    true,
	}
	var stdinReader *strings.Reader
	if stdin != "" {
		stdinReader = strings.NewReader(stdin)
		opts.Stdin = true
	}

	req := c.RESTClient().Post().
		Resource("pods").
		Namespace(namespace).
		Name(name).
		SubResource("exec").
		VersionedParams(opts, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(c.REST, "POST", req.URL())
	if err != nil {
		return "", fmt.Errorf("build executor for %s/%s: %w", namespace, name, err)
	}

	streamOpts := remotecommand.StreamOptions{}
	var stdout, stderr bytes.Buffer
	streamOpts.Stdout = &stdout
	streamOpts.Stderr = &stderr
	if stdinReader != nil {
		streamOpts.Stdin = stdinReader
	}
	if err := exec.StreamWithContext(ctx, streamOpts); err != nil {
		return "", fmt.Errorf("exec %v in %s/%s: %w (stderr: %s)", command, namespace, name, err, stderr.String())
	}
	return stdout.String(), nil
}
