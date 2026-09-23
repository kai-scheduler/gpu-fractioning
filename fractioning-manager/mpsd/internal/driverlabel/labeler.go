// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package driverlabel

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/kai-scheduler/gpu-fractioning/pkg/driverinfo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
)

const (
	envNodeName           = "NODE_NAME"
	defaultRequestTimeout = 10 * time.Second
)

// LabelCurrentNode reads the local NVIDIA driver version via NVML and labels
// the current node with the parsed driver major branch.
func LabelCurrentNode(ctx context.Context, logger *slog.Logger) error {
	if logger == nil {
		logger = slog.Default()
	}

	nodeName := os.Getenv(envNodeName)
	if nodeName == "" {
		return fmt.Errorf("%s is required", envNodeName)
	}

	driverVersion, err := driverVersionFromNVML(logger)
	if err != nil {
		return err
	}

	major, err := driverinfo.ParseDriverVersionMajor(driverVersion)
	if err != nil {
		return fmt.Errorf("parse NVIDIA driver version %q: %w", driverVersion, err)
	}

	if err := patchNodeDriverMajor(ctx, nodeName, major); err != nil {
		return err
	}

	logger.Info("labeled node with NVIDIA driver major version",
		"driverVersion", driverVersion,
		"driverMajor", major,
	)
	return nil
}

func driverVersionFromNVML(logger *slog.Logger) (string, error) {
	ret := nvml.Init()
	if ret != nvml.SUCCESS && ret != nvml.ERROR_ALREADY_INITIALIZED {
		return "", fmt.Errorf("initialize NVML: %w", ret)
	}
	defer func() {
		ret := nvml.Shutdown()
		if ret != nvml.SUCCESS && ret != nvml.ERROR_UNINITIALIZED {
			logger.Warn("shutdown NVML", "error", ret.Error())
		}
	}()

	version, ret := nvml.SystemGetDriverVersion()
	if ret != nvml.SUCCESS {
		return "", fmt.Errorf("read NVIDIA driver version from NVML: %w", ret)
	}
	version = strings.TrimSpace(version)
	if version == "" {
		return "", fmt.Errorf("NVML returned an empty NVIDIA driver version")
	}
	return version, nil
}

func patchNodeDriverMajor(ctx context.Context, nodeName string, major int) error {
	client, err := inClusterNodeInterface()
	if err != nil {
		return err
	}
	return patchNodeDriverMajorWithClient(ctx, client, nodeName, major)
}

func patchNodeDriverMajorWithClient(ctx context.Context, client corev1client.NodeInterface, nodeName string, major int) error {
	body, err := driverMajorLabelPatch(major)
	if err != nil {
		return err
	}

	if _, err := client.Patch(ctx, nodeName, types.MergePatchType, body, metav1.PatchOptions{}); err != nil {
		return fmt.Errorf("patch node %s label %q: %w", nodeName, driverinfo.NVIDIADriverMajorLabel, err)
	}
	return nil
}

func driverMajorLabelPatch(major int) ([]byte, error) {
	if major <= 0 {
		return nil, fmt.Errorf("driver major must be positive")
	}

	patch := map[string]any{
		"metadata": map[string]any{
			"labels": map[string]string{
				driverinfo.NVIDIADriverMajorLabel: strconv.Itoa(major),
			},
		},
	}
	return json.Marshal(patch)
}

func inClusterNodeInterface() (corev1client.NodeInterface, error) {
	restCfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("load in-cluster Kubernetes config: %w", err)
	}
	restCfg.Timeout = defaultRequestTimeout

	coreClient, err := corev1client.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("create Kubernetes CoreV1 client: %w", err)
	}
	return coreClient.Nodes(), nil
}
