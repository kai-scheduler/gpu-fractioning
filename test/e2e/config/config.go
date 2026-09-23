// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package config loads e2e run configuration entirely from the environment so
// the suite can point at whatever cluster the caller provides (no config file,
// no assumption about how the cluster was created). The suite only connects to
// an existing cluster — it never provisions one.
package config

import (
	"os"
	"path/filepath"
	"time"

	"github.com/kai-scheduler/gpu-fractioning/pkg/env"
)

// Config is read entirely from the environment.
type Config struct {
	// Kubeconfig is the path to the kubeconfig used to reach the cluster.
	Kubeconfig string

	// OperatorNamespace is where the gpu-fractioning is installed and,
	// therefore, where the operator-managed DaemonSets (fractiond — which hosts
	// the metricsd sidecar — and mpsd) are created. The suite scrapes the
	// metricsd sidecar in the fractiond pods here.
	OperatorNamespace string

	// GPUNodeSelector selects nodes expected to advertise GPUs.
	GPUNodeSelector string

	// GPUNodeCount, if > 0, asserts the exact count of nodes matching
	// GPUNodeSelector; 0 skips the assertion (and the node listing) entirely.
	// Set it to the number of GPU nodes the target cluster is expected to have,
	// to catch a misconfigured/partially-up cluster before any test runs.
	GPUNodeCount int

	// PodReadyTimeout bounds how long workload.Apply waits for a test pod to
	// reach Running.
	PodReadyTimeout time.Duration

	// PollInterval is the poll cadence shared by the workload/nvmlmock waiters.
	PollInterval time.Duration

	// DaemonSetReadyTimeout bounds how long nvmlmock waits for a DaemonSet
	// rollout (nvml-mock or fractiond) to complete after a config change.
	DaemonSetReadyTimeout time.Duration

	// GPUMemoryMiB is the total GPU memory per device in MiB, used by
	// attribution tests that derive fractional memory requests (e.g. half the
	// device memory for two co-located pods). Matches the memory of the GPU
	// profile installed in the cluster — for the default nvml-mock A100 profile
	// this is 40 GiB (40960 MiB). Override via E2E_GPU_MEMORY_MIB when
	// targeting a cluster with a different GPU model.
	GPUMemoryMiB int

	// GPUCountPerNode is the number of GPUs available per GPU node. Tests that
	// require multiple physical devices on one node skip when this is
	// less than 2. The default is 1 (conservative); set E2E_GPU_COUNT_PER_NODE=2
	// for nvml-mock clusters, which always expose two devices per node.
	GPUCountPerNode int

	// NVMLMock signals that the cluster is running the nvml-mock DaemonSet
	// instead of a real NVIDIA driver. Tests that rely on per-process SM
	// utilization injection require nvml-mock and skip when this is
	// false. Set E2E_NVML_MOCK=1 for nvml-mock clusters.
	NVMLMock bool

	// RolloutTimeout bounds DaemonSet-rollout / CR-Ready waits in the operator
	// suite (a spec change re-rolls a DaemonSet). CondTimeout bounds the shorter
	// condition-transition waits (fault injection flipping a node/CR condition).
	RolloutTimeout time.Duration
	CondTimeout    time.Duration
}

const (
	envKubeconfig            = "E2E_KUBECONFIG"
	envOperatorNamespace     = "E2E_OPERATOR_NAMESPACE"
	envGPUNodeSelector       = "E2E_GPU_NODE_SELECTOR"
	envGPUNodeCount          = "E2E_GPU_NODE_COUNT"
	envGPUMemoryMiB          = "E2E_GPU_MEMORY_MIB"
	envGPUCountPerNode       = "E2E_GPU_COUNT_PER_NODE"
	envNVMLMock              = "E2E_NVML_MOCK"
	envPodReadyTimeout       = "E2E_POD_READY_TIMEOUT"
	envPollInterval          = "E2E_POLL_INTERVAL"
	envDaemonSetReadyTimeout = "E2E_DAEMONSET_READY_TIMEOUT"
	envRolloutTimeout        = "E2E_ROLLOUT_TIMEOUT"
	envCondTimeout           = "E2E_COND_TIMEOUT"

	defaultOperatorNamespace     = "gpu-fractioning"
	defaultGPUNodeSelector       = "nvidia.com/gpu.present=true"
	defaultGPUNodeCount          = 0
	defaultGPUMemoryMiB          = 40960
	defaultGPUCountPerNode       = 1
	defaultPodReadyTimeout       = 2 * time.Minute
	defaultPollInterval          = 2 * time.Second
	defaultDaemonSetReadyTimeout = 3 * time.Minute
	defaultRolloutTimeout        = 3 * time.Minute
	defaultCondTimeout           = 2 * time.Minute
)

// Load builds a Config from environment variables.
func Load() Config {
	return Config{
		Kubeconfig:            env.String(envKubeconfig, defaultKubeconfig()),
		OperatorNamespace:     env.String(envOperatorNamespace, defaultOperatorNamespace),
		GPUNodeSelector:       env.String(envGPUNodeSelector, defaultGPUNodeSelector),
		GPUNodeCount:          env.Int(envGPUNodeCount, defaultGPUNodeCount),
		GPUMemoryMiB:          env.Int(envGPUMemoryMiB, defaultGPUMemoryMiB),
		GPUCountPerNode:       env.Int(envGPUCountPerNode, defaultGPUCountPerNode),
		NVMLMock:              env.Bool(envNVMLMock, false),
		PodReadyTimeout:       env.Duration(envPodReadyTimeout, defaultPodReadyTimeout),
		PollInterval:          env.Duration(envPollInterval, defaultPollInterval),
		DaemonSetReadyTimeout: env.Duration(envDaemonSetReadyTimeout, defaultDaemonSetReadyTimeout),
		RolloutTimeout:        env.Duration(envRolloutTimeout, defaultRolloutTimeout),
		CondTimeout:           env.Duration(envCondTimeout, defaultCondTimeout),
	}
}

func defaultKubeconfig() string {
	if v := os.Getenv("KUBECONFIG"); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".kube", "config")
}
