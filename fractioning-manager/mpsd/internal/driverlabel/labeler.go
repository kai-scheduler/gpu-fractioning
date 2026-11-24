// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package driverlabel records the node-local NVIDIA driver major version, read
// through NVML, on the mpsd pod. It used to label the Node, which needed a
// cluster-wide nodes/patch grant for a privileged, host-PID pod; annotating the
// pod it runs in needs only a namespaced pods/patch, and the operator reads the
// annotation from the mpsd pod of a node where it needs it.
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
	envPodName            = "POD_NAME"
	envPodNamespace       = "POD_NAMESPACE"
	defaultRequestTimeout = 10 * time.Second
)

// RecordDriverMajor reads the local NVIDIA driver version via NVML and
// annotates the current pod with the parsed driver major branch. It returns the
// major it found so the caller can refuse to start MPS on a branch that does
// not support the flags mpsd uses.
func RecordDriverMajor(ctx context.Context, logger *slog.Logger) (int, error) {
	if logger == nil {
		logger = slog.Default()
	}

	podName := os.Getenv(envPodName)
	podNamespace := os.Getenv(envPodNamespace)
	if podName == "" || podNamespace == "" {
		return 0, fmt.Errorf("%s and %s are required", envPodName, envPodNamespace)
	}

	driverVersion, err := driverVersionFromNVML(logger)
	if err != nil {
		return 0, err
	}

	major, err := driverinfo.ParseDriverVersionMajor(driverVersion)
	if err != nil {
		return 0, fmt.Errorf("parse NVIDIA driver version %q: %w", driverVersion, err)
	}

	if err := patchPodDriverMajor(ctx, podNamespace, podName, major); err != nil {
		return 0, err
	}

	logger.Info("recorded NVIDIA driver major version on the pod",
		"driverVersion", driverVersion,
		"driverMajor", major,
	)
	return major, nil
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

func patchPodDriverMajor(ctx context.Context, namespace, name string, major int) error {
	client, err := inClusterPodInterface(namespace)
	if err != nil {
		return err
	}
	return patchPodDriverMajorWithClient(ctx, client, name, major)
}

func patchPodDriverMajorWithClient(ctx context.Context, client corev1client.PodInterface, name string, major int) error {
	body, err := driverMajorAnnotationPatch(major)
	if err != nil {
		return err
	}

	if _, err := client.Patch(ctx, name, types.MergePatchType, body, metav1.PatchOptions{}); err != nil {
		return fmt.Errorf("patch pod %s annotation %q: %w", name, driverinfo.NVIDIADriverMajorAnnotation, err)
	}
	return nil
}

func driverMajorAnnotationPatch(major int) ([]byte, error) {
	if major <= 0 {
		return nil, fmt.Errorf("driver major must be positive")
	}

	patch := map[string]any{
		"metadata": map[string]any{
			"annotations": map[string]string{
				driverinfo.NVIDIADriverMajorAnnotation: strconv.Itoa(major),
			},
		},
	}
	return json.Marshal(patch)
}

func inClusterPodInterface(namespace string) (corev1client.PodInterface, error) {
	restCfg, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("load in-cluster Kubernetes config: %w", err)
	}
	restCfg.Timeout = defaultRequestTimeout

	coreClient, err := corev1client.NewForConfig(restCfg)
	if err != nil {
		return nil, fmt.Errorf("create Kubernetes CoreV1 client: %w", err)
	}
	return coreClient.Pods(namespace), nil
}
