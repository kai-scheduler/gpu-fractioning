// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nvmlmock drives the nvml-mock ConfigMap so per-pod GPU metrics are
// deterministic in the e2e suite.
//
// The metricsd e2e image (Dockerfile.e2e) bakes in the nvml-mock shared
// library (libnvidia-ml.so.1). The production NVML collector is used
// unchanged; the library reads MOCK_NVML_CONFIG at nvmlInit() time and
// returns the configured processes on each NVML API call. To make NVML
// attribute metrics to a real pod this package:
//
//  1. resolves the workload container's host-namespace PID (HostPID), and
//  2. rewrites the gpu-fractioning/nvml-mock-config ConfigMap so that
//     PID appears as a GPU process on a chosen device UUID (SetProcesses).
//
// Changing the ConfigMap requires metricsd pod restarts to take effect:
// the library reads MOCK_NVML_CONFIG once at nvmlInit() time, so a new pod
// is required to pick up the updated process list. SetProcesses calls
// restartMetricsd automatically.
package nvmlmock

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"strconv"
	"strings"
	"text/template"
	"time"

	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/pods"
)

//go:embed config.yaml.tmpl
var configYAMLTemplate string

var configTmpl = template.Must(template.New("nvml-mock-config").Parse(configYAMLTemplate))

const (
	// MockNamespace / MockConfigMapName identify the nvml-mock DaemonSet objects
	// in the gpu-operator namespace — used only for HostPID resolution.
	MockNamespace     = "gpu-operator"
	MockConfigMapName = "nvml-mock-config"

	// Container is the nvml-mock pod's container name; it runs hostPID: true and
	// privileged, so an exec into it sees the whole node's process table.
	Container = "nvml-mock"

	// MetricsdNamespace / MetricsdConfigMapName identify the ConfigMap that the
	// metricsd sidecar mounts as MOCK_NVML_CONFIG. SetProcesses updates this
	// ConfigMap and restarts the metricsd pod so the production NVML collector
	// reads the new process list at nvmlInit time.
	MetricsdNamespace     = "gpu-fractioning"
	MetricsdConfigMapName = "nvml-mock-config"

	// metricsdComponent is the app.kubernetes.io/component label value on fractiond pods.
	metricsdComponent = "fractiond"
	// metricsdContainer is the container name of the metricsd sidecar.
	metricsdContainer = "metricsd"
)

// Device UUIDs baked into the nvml-mock config. Callers pin a Proc to one of
// these; the exported gpu_uuid label equals the chosen value.
const (
	Device0UUID = "2881be89-519e-4b30-8d25-b9032893e6f9"
	Device1UUID = "1a1fb9fb-a857-4986-8b0f-d0d664ba0364"
)

// Proc is one mocked GPU process pinned to a real host PID.
type Proc struct {
	UUID          string // must equal a device uuid below (Device0UUID/Device1UUID)
	PID           uint32 // real host-namespace PID (from HostPID)
	UsedMemoryMiB uint64 // MiB; nvml-mock stores used_memory_mib (not bytes) in its YAML config
	// SMUtil is the per-process SM utilization percent (0–100) reported via
	// GetProcessUtilization. Non-zero values produce a non-zero
	// gpu_fractioning_gpu_sm_utilization_percent metric.
	SMUtil uint32
}

// HostPID resolves the host-namespace PID of the workload process identified by
// marker (the token workload.Apply embeds in the container command line). It
// scans /proc/<pid>/cmdline inside the nvml-mock pod on the workload's node,
// which — because that pod runs hostPID: true — sees the workload's process.
//
// The nvml-mock image (Debian, ghcr.io/nvidia/nvml-mock) ships no procps, so
// pgrep/ps are unavailable; grep over /proc is the portable alternative. The
// marker is fed on stdin (grep -f /dev/stdin), not as a command argument, so it
// never lands in the search command's own /proc/<pid>/cmdline — otherwise grep
// and its parent shell would match themselves and the marker would resolve to
// more than one PID.
func HostPID(ctx context.Context, c *cluster.Client, nodeName, marker string) (uint32, error) {
	mockPod, err := podOnNode(ctx, c, nodeName)
	if err != nil {
		return 0, err
	}

	// -l: list matching files, -a: treat NUL-separated cmdline as text,
	// -F: fixed string (the marker), -f /dev/stdin: read the pattern from stdin.
	// `|| true`: a short-lived process can exit mid-scan, leaving grep unable to
	// read its /proc/<pid>/cmdline; grep then exits 2 (error) even though it
	// printed the real match. Swallow that so the exec succeeds — the match-count
	// check below is the real validation.
	out, err := pods.ExecStdin(ctx, c, MockNamespace, mockPod, Container,
		[]string{"sh", "-c", "grep -laF -f /dev/stdin /proc/[0-9]*/cmdline 2>/dev/null || true"}, marker)
	if err != nil {
		return 0, fmt.Errorf("scan /proc for marker %q in %s/%s: %w", marker, MockNamespace, mockPod, err)
	}

	lines := strings.Fields(strings.TrimSpace(out))
	if len(lines) != 1 {
		return 0, fmt.Errorf("want exactly one host PID for marker %q, matched %v", marker, lines)
	}
	// lines[0] is "/proc/<pid>/cmdline"; extract <pid>.
	pidStr := strings.TrimSuffix(strings.TrimPrefix(lines[0], "/proc/"), "/cmdline")
	pid, err := strconv.ParseUint(pidStr, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("parse host PID from %q: %w", lines[0], err)
	}
	return uint32(pid), nil
}

// SetProcesses rewrites the metricsd nvml-mock ConfigMap (gpu-fractioning
// namespace) for GPU model gpu (a Profiles key, e.g. "a100"; "" selects
// DefaultGPU) so its devices carry procs, then restarts the metricsd pod on
// each node so the production NVML collector re-reads the config at nvmlInit.
// Passing an empty procs slice resets the mock to idle.
func SetProcesses(ctx context.Context, c *cluster.Client, gpu string, procs []Proc) error {
	if gpu == "" {
		gpu = DefaultGPU
	}
	profile, ok := Profiles[gpu]
	if !ok {
		return fmt.Errorf("unknown GPU profile %q (known: %s)", gpu, knownProfiles())
	}

	var cm corev1.ConfigMap
	if err := c.Ctrl.Get(ctx, ctrlclient.ObjectKey{Namespace: MetricsdNamespace, Name: MetricsdConfigMapName}, &cm); err != nil {
		return fmt.Errorf("get configmap %s/%s: %w", MetricsdNamespace, MetricsdConfigMapName, err)
	}
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data["config.yaml"] = renderConfig(profile, procs)
	if err := c.Ctrl.Update(ctx, &cm); err != nil {
		return fmt.Errorf("update configmap %s/%s: %w", MetricsdNamespace, MetricsdConfigMapName, err)
	}

	return restartMetricsd(ctx, c)
}

// restartMetricsd deletes all fractiond pods (which contain the metricsd sidecar)
// so the DaemonSet controller recreates them with the latest ConfigMap data.
// The new pods call nvmlInit and read MOCK_NVML_CONFIG fresh, picking up the
// updated process list.
func restartMetricsd(ctx context.Context, c *cluster.Client) error {
	var podList corev1.PodList
	if err := c.Ctrl.List(ctx, &podList,
		ctrlclient.InNamespace(MetricsdNamespace),
		ctrlclient.MatchingLabels{"app.kubernetes.io/component": metricsdComponent}); err != nil {
		return fmt.Errorf("list metricsd pods: %w", err)
	}

	// Track which pods we deleted so the wait loop ignores terminating copies
	// of the same pods — terminating pods can briefly still show cs.Ready=true
	// during their grace period, causing a premature return before replacement
	// pods are created.
	deleted := make(map[string]struct{}, len(podList.Items))
	for i := range podList.Items {
		deleted[podList.Items[i].Name] = struct{}{}
		if err := c.Ctrl.Delete(ctx, &podList.Items[i]); err != nil {
			return fmt.Errorf("delete pod %s: %w", podList.Items[i].Name, err)
		}
	}

	// Wait until ALL replacement pods (one per deleted pod) have their metricsd
	// container Running (cs.Ready=true). metricsd has no readiness probe, so
	// cs.Ready flips as soon as the process starts — before nvmlInit() reads the
	// ConfigMap and before the first collect cycle. Callers then poll for metric
	// series, which naturally waits for NVML initialisation and the first collect.
	// Waiting for only one pod is insufficient: NRI Synchronize fires after a pod
	// is running, so it may not have attributed pre-existing containers yet.
	// Waiting for all N pods ensures every node's metricsd is at least running.
	want := len(deleted)
	deadline := time.Now().Add(c.Config.DaemonSetReadyTimeout)
	for time.Now().Before(deadline) {
		time.Sleep(2 * time.Second)
		var updated corev1.PodList
		if err := c.Ctrl.List(ctx, &updated,
			ctrlclient.InNamespace(MetricsdNamespace),
			ctrlclient.MatchingLabels{"app.kubernetes.io/component": metricsdComponent}); err != nil {
			continue
		}
		ready := 0
		for _, pod := range updated.Items {
			if _, wasDeleted := deleted[pod.Name]; wasDeleted {
				continue // skip old (possibly still-terminating) pods
			}
			if pod.DeletionTimestamp != nil {
				continue // skip any other terminating pods
			}
			for _, cs := range pod.Status.ContainerStatuses {
				if cs.Name == metricsdContainer && cs.Ready {
					ready++
					break
				}
			}
		}
		if ready >= want {
			return nil
		}
	}
	return fmt.Errorf("timed out waiting for %d metricsd pod(s) to become ready after restart", want)
}

func podOnNode(ctx context.Context, c *cluster.Client, nodeName string) (string, error) {
	mockPods, err := pods.ListByLabel(ctx, c, MockNamespace, "app=nvml-mock")
	if err != nil {
		return "", fmt.Errorf("list nvml-mock pods: %w", err)
	}
	for i := range mockPods {
		if mockPods[i].Spec.NodeName == nodeName {
			return mockPods[i].Name, nil
		}
	}
	return "", fmt.Errorf("no nvml-mock pod scheduled on node %q", nodeName)
}

type deviceEntry struct {
	Index int
	UUID  string
	Bus   string
	Procs []Proc
}

// renderConfig renders the nvml-mock ConfigMap YAML for the given GPU profile
// with two devices, attaching each Proc to the device whose UUID it names.
func renderConfig(profile GPUProfile, procs []Proc) string {
	byUUID := map[string][]Proc{}
	for _, p := range procs {
		byUUID[p.UUID] = append(byUUID[p.UUID], p)
	}

	data := struct {
		Profile GPUProfile
		Devices []deviceEntry
	}{
		Profile: profile,
		Devices: []deviceEntry{
			{0, Device0UUID, "0000:07:00.0", byUUID[Device0UUID]},
			{1, Device1UUID, "0000:0F:00.0", byUUID[Device1UUID]},
		},
	}

	var b bytes.Buffer
	if err := configTmpl.Execute(&b, data); err != nil {
		panic(fmt.Sprintf("renderConfig: %v", err))
	}
	return b.String()
}
