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
	"path/filepath"
	"strings"

	"github.com/kai-scheduler/gpu-fractioning/pkg/env"
	"github.com/kai-scheduler/gpu-fractioning/pkg/toolkitinfo"
	"golang.org/x/mod/semver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
)

// ValidateHostToolkit checks and labels the installed toolkit before MPS starts.
// The operator selects its location and whether to bypass validation. HostPID lets
// /proc/1/root reach the host filesystem without an additional host mount.
func ValidateHostToolkit(ctx context.Context) error {
	if env.Bool("SKIP_GPU_STACK_VERSION_CHECKS", false) {
		slog.Info("host NVIDIA Container Toolkit validation is skipped", "reason", "SKIP_GPU_STACK_VERSION_CHECKS=true")
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
	return validateAndLabelHostToolkit(ctx, client, nodeName)
}

func hostToolkitVersion(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, defaultRequestTimeout)
	defer cancel()
	binary := os.Getenv("HOST_TOOLKIT_BINARY")
	searchPath := "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
	if binary == "" {
		// With no ClusterPolicy, allow both package and default operator installs.
		searchPath += ":/usr/local/nvidia/toolkit"
		binary = "nvidia-ctk"
	} else if binary != "nvidia-ctk" && !filepath.IsAbs(binary) {
		return "", fmt.Errorf("host NVIDIA Container Toolkit binary must be an absolute path: %q", binary)
	}
	// chroot resolves the binary and its libraries in the host filesystem.
	// Pass the selected binary as an argument, without shell interpolation.
	slog.Info("checking host NVIDIA Container Toolkit", "binary", binary, "searchPath", searchPath)
	cmd := exec.CommandContext(ctx, "chroot", "/proc/1/root", binary, "--version")
	cmd.Env = []string{"PATH=" + searchPath}
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("read host NVIDIA Container Toolkit version using %q: %w: %s", binary, err, strings.TrimSpace(string(output)))
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

func validateAndLabelHostToolkit(ctx context.Context, client corev1client.NodeInterface, nodeName string) error {
	version, probeErr := hostToolkitVersion(ctx)
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
	slog.Info("validated host NVIDIA Container Toolkit", "version", version, "node", nodeName)
	return nil
}
