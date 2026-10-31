// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package driverlabel

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strings"

	"github.com/kai-scheduler/gpu-fractioning/pkg/env"
	"github.com/kai-scheduler/gpu-fractioning/pkg/toolkitinfo"
	"golang.org/x/mod/semver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
)

// LabelToolkitVersion checks the host installation only when the operator
// requests it for an externally managed toolkit. mpsd runs with HostPID, so
// /proc/1/root reaches the host filesystem without an additional host mount.
func LabelToolkitVersion(ctx context.Context) error {
	if !env.Bool("CHECK_HOST_TOOLKIT_VERSION", false) {
		return nil
	}
	if env.Bool("SKIP_GPU_STACK_VERSION_CHECKS", false) {
		slog.Warn("host NVIDIA Container Toolkit validation is disabled by SKIP_GPU_STACK_VERSION_CHECKS")
		return nil
	}
	nodeName := os.Getenv(envNodeName)
	if nodeName == "" {
		return fmt.Errorf("%s is required", envNodeName)
	}
	client, err := inClusterNodeInterface()
	if err != nil {
		return err
	}
	return labelToolkitVersion(ctx, client, nodeName, hostToolkitVersion)
}

func hostToolkitVersion(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, defaultRequestTimeout)
	defer cancel()
	// chroot uses the host's dynamic loader/libraries as well as its binary.
	// The first PATH entry covers the GPU Operator's default toolkit install.
	cmd := exec.CommandContext(ctx, "chroot", "/proc/1/root", "/bin/sh", "-c", "exec nvidia-ctk --version")
	cmd.Env = []string{"PATH=/usr/local/nvidia/toolkit:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("read host NVIDIA Container Toolkit version: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return parseToolkitVersion(string(output))
}

func parseToolkitVersion(output string) (string, error) {
	for _, line := range strings.Split(output, "\n") {
		if version, ok := strings.CutPrefix(strings.TrimSpace(line), "NVIDIA Container Toolkit CLI version "); ok {
			version = "v" + strings.TrimPrefix(strings.TrimSpace(version), "v")
			if semver.IsValid(version) {
				return strings.TrimPrefix(semver.Canonical(version), "v"), nil
			}
		}
	}
	return "", fmt.Errorf("unrecognized host NVIDIA Container Toolkit version output: %q", output)
}

func labelToolkitVersion(ctx context.Context, client corev1client.NodeInterface, nodeName string, readVersion func(context.Context) (string, error)) error {
	version, probeErr := readVersion(ctx)
	// Clear stale successful detection when the host binary cannot be read.
	var value any
	if probeErr == nil {
		value = version
	}
	body, err := json.Marshal(map[string]any{"metadata": map[string]any{
		"labels": map[string]any{toolkitinfo.VersionLabel: value},
	}})
	if err != nil {
		return err
	}
	if _, err := client.Patch(ctx, nodeName, types.MergePatchType, body, metav1.PatchOptions{}); err != nil {
		return fmt.Errorf("patch node %s toolkit version: %w", nodeName, err)
	}
	if probeErr != nil {
		return probeErr
	}
	if semver.Compare("v"+version, toolkitinfo.MinimumVersion) < 0 {
		return fmt.Errorf("host NVIDIA Container Toolkit %s is below minimum %s", version, toolkitinfo.MinimumVersion)
	}
	return nil
}
