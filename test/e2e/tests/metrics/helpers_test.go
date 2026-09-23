// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package metrics

import (
	"context"
	"fmt"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/cluster"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/nodes"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/pods"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/metrics"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/nvmlmock"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/plugin"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/workload"
)

// attributionTestNamespace hosts every workload pod these tests create —
// kept separate from any scheduler project namespace so the tests don't depend
// on scaffolding this e2e cluster may not have.
const attributionTestNamespace = "metricsd-e2e"

const skipNoNVMLMock = "requires nvml-mock (set E2E_NVML_MOCK=1)"

// Per-test context deadlines. Each constant is sized for its worst-case
// path: one nvml-mock SetProcesses cycle takes ~90 s (pod restart + ready
// wait), and each metric-series poll window adds up to c.Config.PodReadyTimeout.
const (
	// testTimeout covers a single SetProcesses cycle plus one poll window.
	// Used by most nvml-mock tests (clamping, compute, isolation, lifecycle,
	// normalized, fractioning).
	testTimeout = 10 * time.Minute

	// aggregationTestTimeout allows for two pods with two SetProcesses calls
	// and two independent poll windows.
	aggregationTestTimeout = 12 * time.Minute

	// attributionTestTimeout covers a single fractional-pod attribution
	// cycle (one SetProcesses + one poll window).
	attributionTestTimeout = 8 * time.Minute

	// podUIDTestTimeout adds headroom for pod deletion and the metric-pruning
	// wait on top of the standard attribution cycle.
	podUIDTestTimeout = 15 * time.Minute

	// exclusionTestTimeout covers negative assertNeverAppears checks;
	// no nvml-mock is involved so the full 2 min is generous.
	exclusionTestTimeout = 2 * time.Minute

	// metricsPresenceTimeout is a short deadline for basic metric-presence
	// checks that do not require nvml-mock or pod restarts.
	metricsPresenceTimeout = 30 * time.Second

	// fractiondRestartTestTimeout covers two DaemonSet restart cycles (one
	// SetProcesses + one explicit fractiond restart) plus two metric-series
	// poll windows.
	fractiondRestartTestTimeout = 15 * time.Minute
)

const (
	memMetricName  = "gpu_fractioning_gpu_memory_used_bytes"
	smMetricName   = "gpu_fractioning_gpu_sm_utilization_percent"
	normMetricName = "gpu_fractioning_gpu_sm_utilization_percent_normalized"
)

// allMetricNames is the full set of per-pod series every test must cover —
// mem, raw SM, and normalized SM. All three share the same label set and
// lifecycle: they are emitted and pruned together by the exporter.
var allMetricNames = []string{memMetricName, smMetricName, normMetricName}

// waitForSeries polls every fractiond pod's /metrics until a series
// for metricName matches every label in match, or times out.
func waitForSeries(ctx context.Context, c *cluster.Client, metricName string, match map[string]string) (*dto.Metric, error) {
	var found *dto.Metric

	err := waiter.PollUntil(ctx, c.Config.PodReadyTimeout, c.Config.PollInterval,
		fmt.Sprintf("metric series %s matching %v", metricName, match),
		func(ctx context.Context) (bool, error) {
			series, err := findSeriesAcrossPluginPods(ctx, c, metricName, match)
			if err != nil {
				return false, err
			}
			if len(series) > 0 {
				found = series[0]
				return true, nil
			}
			return false, nil
		})
	if err != nil {
		return nil, err
	}
	return found, nil
}

// waitForAbsence polls until no fractiond pod reports a series for
// metricName matching every label in match, or times out. The inverse of
// waitForSeries — used to confirm a deleted pod's series is eventually
// pruned (exporter.go's pruneDeletedPods).
func waitForAbsence(ctx context.Context, c *cluster.Client, metricName string, match map[string]string) error {
	return waiter.PollUntil(ctx, c.Config.PodReadyTimeout, c.Config.PollInterval,
		fmt.Sprintf("metric series %s matching %v to disappear", metricName, match),
		func(ctx context.Context) (bool, error) {
			series, err := findSeriesAcrossPluginPods(ctx, c, metricName, match)
			if err != nil {
				return false, err
			}
			return len(series) == 0, nil
		})
}

// assertNeverAppears scrapes repeatedly over window and fails the test the
// instant a series for metricName matches every label in match. Used for
// negative cases (full-GPU and malformed-annotation exclusion) where the expected state is deterministic and
// immediate (the adapter rejects/accepts a container synchronously at
// create time — there's no eventual-consistency to wait out), so this
// confirms the negative holds rather than just checking once, which could
// pass by scraping before the plugin has processed the container event at
// all.
func assertNeverAppears(ctx context.Context, t *testing.T, c *cluster.Client, metricName string, match map[string]string, window, interval time.Duration) {
	t.Helper()

	deadline := time.Now().Add(window)
	for {
		series, err := findSeriesAcrossPluginPods(ctx, c, metricName, match)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			t.Fatalf("scrape plugin pods: %v", err)
		}
		if len(series) > 0 {
			t.Fatalf("%s: found unexpected series matching %v: %v", metricName, match, series[0])
		}
		if time.Now().After(deadline) {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(interval):
		}
	}
}

// scrapeFreshFamilies scrapes /metrics from the cached fractiond pods. If all
// cached pods are unreachable (empty result — e.g. pods were replaced by a
// nvmlmock.SetProcesses restart and s.PluginPods is stale), it re-lists pods
// once, updates s.PluginPods for subsequent polls, and retries. This keeps
// the happy-path poll-loop free of ListByLabel calls while self-healing after
// pod restarts without overloading the API server with per-poll list requests.
func scrapeFreshFamilies(ctx context.Context, c *cluster.Client) (map[string]map[string]*dto.MetricFamily, error) {
	allFamilies, err := plugin.ScrapeFrom(ctx, c, s.PluginPods)
	if err != nil {
		return nil, err
	}
	// len == 0 means every cached pod's port-forward failed: the list is stale.
	// Re-list once and update the cache so subsequent polls skip this branch.
	if len(allFamilies) == 0 && len(s.PluginPods) > 0 {
		freshPods, listErr := pods.ListByLabel(ctx, c, c.Config.OperatorNamespace, plugin.LabelSelector)
		if listErr == nil && len(freshPods) > 0 {
			s.PluginPods = freshPods
			allFamilies, err = plugin.ScrapeFrom(ctx, c, freshPods)
			if err != nil {
				return nil, err
			}
		}
	}
	return allFamilies, nil
}

func findSeriesAcrossPluginPods(ctx context.Context, c *cluster.Client, metricName string, match map[string]string) ([]*dto.Metric, error) {
	allFamilies, err := scrapeFreshFamilies(ctx, c)
	if err != nil {
		return nil, err
	}
	var out []*dto.Metric
	for _, families := range allFamilies {
		out = append(out, metrics.FindSeries(families, metricName, match)...)
	}
	return out, nil
}

// pluginPodRestartCounts returns each fractiond pod's total
// container restart count, keyed by pod name. A crash (e.g. a panic from a
// malformed annotation or a delete-during-scrape race) shows up as an
// increase here — a more reliable resilience signal than "the test didn't
// time out," since a crashed pod still gets recreated and can pass a
// later scrape once it's back up.
func pluginPodRestartCounts(ctx context.Context, c *cluster.Client) (map[string]int32, error) {
	pluginPods, err := pods.ListByLabel(ctx, c, c.Config.OperatorNamespace, plugin.LabelSelector)
	if err != nil {
		return nil, fmt.Errorf("list fractiond pods: %w", err)
	}

	counts := make(map[string]int32, len(pluginPods))
	for _, pod := range pluginPods {
		var total int32
		for _, cs := range pod.Status.ContainerStatuses {
			total += cs.RestartCount
		}
		counts[pod.Name] = total
	}
	return counts, nil
}

// setProcesses wraps nvmlmock.SetProcesses and immediately refreshes
// s.PluginPods to the current fractiond pods after the restart. Without this,
// s.PluginPods still points to the terminating pre-restart pods. In CI those
// pods stay reachable long enough that scrapeFromPods returns non-empty results
// (bypassing the automatic re-list) and every poll sees 0 gpu_fractioning_ series
// until the old pods finally terminate. Cleanup calls that use
// context.Background() should call nvmlmock.SetProcesses directly — they run
// after assertions and do not need a fresh pod cache.
func setProcesses(ctx context.Context, c *cluster.Client, gpu string, procs []nvmlmock.Proc) error {
	if err := nvmlmock.SetProcesses(ctx, c, gpu, procs); err != nil {
		return err
	}
	freshPods, err := pods.ListByLabel(ctx, c, c.Config.OperatorNamespace, plugin.LabelSelector)
	if err != nil {
		return fmt.Errorf("list fractiond pods after restart: %w", err)
	}
	s.PluginPods = freshPods
	return nil
}

// restartFractiondPods deletes every fractiond pod and waits for their
// replacements to be ready, then refreshes s.PluginPods. Use this to simulate
// an NRI plugin reconnect without changing the nvml-mock ConfigMap.
func restartFractiondPods(ctx context.Context, t *testing.T, c *cluster.Client) {
	t.Helper()

	existing, err := pods.ListByLabel(ctx, c, c.Config.OperatorNamespace, plugin.LabelSelector)
	if err != nil {
		t.Fatalf("list fractiond pods before restart: %v", err)
	}

	deleted := make(map[string]struct{}, len(existing))
	for i := range existing {
		deleted[existing[i].Name] = struct{}{}
		if err := c.Ctrl.Delete(ctx, &existing[i]); err != nil {
			t.Fatalf("delete fractiond pod %s: %v", existing[i].Name, err)
		}
	}

	want := len(deleted)
	deadline := time.Now().Add(c.Config.DaemonSetReadyTimeout)
	for time.Now().Before(deadline) {
		time.Sleep(2 * time.Second)
		var updated corev1.PodList
		if err := c.Ctrl.List(ctx, &updated,
			ctrlclient.InNamespace(c.Config.OperatorNamespace),
			ctrlclient.MatchingLabels{"app.kubernetes.io/component": "fractiond", "app.kubernetes.io/managed-by": "gpu-fractioning"}); err != nil {
			continue
		}
		ready := 0
		for _, pod := range updated.Items {
			if _, wasDeleted := deleted[pod.Name]; wasDeleted {
				continue
			}
			if pod.DeletionTimestamp != nil {
				continue
			}
			for _, cs := range pod.Status.ContainerStatuses {
				if cs.Name == "metricsd" && cs.Ready {
					ready++
					break
				}
			}
		}
		if ready >= want {
			freshPods, err := pods.ListByLabel(ctx, c, c.Config.OperatorNamespace, plugin.LabelSelector)
			if err != nil {
				t.Fatalf("list fractiond pods after restart: %v", err)
			}
			s.PluginPods = freshPods
			return
		}
	}
	t.Fatalf("fractiond pods not ready within %v", c.Config.DaemonSetReadyTimeout)
}

// firstGPUNode returns the name of the first GPU node in the cluster.
// The test is failed immediately if listing fails or the cluster has no GPU nodes.
func firstGPUNode(t *testing.T, ctx context.Context, c *cluster.Client) string {
	t.Helper()
	gpuNodes, err := nodes.ListGPUNodes(ctx, c)
	if err != nil {
		t.Fatalf("list GPU nodes: %v", err)
	}
	if len(gpuNodes) == 0 {
		t.Fatalf("no GPU nodes found matching selector %q", c.Config.GPUNodeSelector)
	}
	return gpuNodes[0].Name
}

// resetNVMLMockOnCleanup registers a t.Cleanup that resets the nvml-mock
// process table to idle. Every test that calls setProcesses must also call
// this; a missed cleanup leaks stale process entries into later tests.
func resetNVMLMockOnCleanup(t *testing.T, c *cluster.Client) {
	t.Helper()
	t.Cleanup(func() {
		if err := nvmlmock.SetProcesses(context.Background(), c, nvmlmock.A100, nil); err != nil {
			t.Errorf("reset nvml-mock to idle: %v", err)
		}
	})
}

// multiContainerPodSpec describes a fractional pod with several GPU-tracked containers.
type multiContainerPodSpec struct {
	Namespace    string
	Name         string
	Containers   []string // container names; each gets its own GPU annotation + unique marker
	MemoryMiB    string   // GPU memory per container (same for all)
	NodeSelector map[string]string
}

// multiContainerMarker returns the unique cmdline token embedded in each
// container's shell argv so HostPID can resolve exactly one PID per container.
// The namespace+pod+container triple guarantees uniqueness across concurrent tests.
func multiContainerMarker(namespace, podName, containerName string) string {
	return fmt.Sprintf("gpumock-%s-%s-%s", namespace, podName, containerName)
}

// applyMultiContainerFractionalPod creates a pod whose containers each carry a
// GPU-memory annotation, making fractiond track each container independently.
// The pod runs on the node given by spec.NodeSelector. Callers must Delete the
// pod when done; use workload.Delete since the pod name/namespace is the key.
func applyMultiContainerFractionalPod(ctx context.Context, c *cluster.Client, spec multiContainerPodSpec) (*corev1.Pod, error) {
	if err := workload.EnsureNamespace(ctx, c, spec.Namespace); err != nil {
		return nil, fmt.Errorf("ensure namespace %s: %w", spec.Namespace, err)
	}
	if err := workload.Delete(ctx, c, spec.Namespace, spec.Name); err != nil {
		return nil, fmt.Errorf("pre-create cleanup of %s/%s: %w", spec.Namespace, spec.Name, err)
	}

	// One annotation pair per container so fractiond tracks every container.
	annotations := make(map[string]string, len(spec.Containers)*2)
	for _, name := range spec.Containers {
		annotations[fmt.Sprintf("nvidia.com/container.%s.gpu-memory.limit", name)] = spec.MemoryMiB + "Mi"
		annotations[fmt.Sprintf("nvidia.com/container.%s.gpu-memory.request", name)] = spec.MemoryMiB + "Mi"
	}

	containers := make([]corev1.Container, 0, len(spec.Containers))
	for _, name := range spec.Containers {
		marker := multiContainerMarker(spec.Namespace, spec.Name, name)
		containers = append(containers, corev1.Container{
			Name:  name,
			Image: workload.DefaultImage,
			// Marker rides in the shell argv (same trick as workload.Apply).
			// The backgrounded sleep keeps the shell alive with the marker in
			// its /proc/<pid>/cmdline so HostPID finds exactly one match.
			Command: []string{"sh", "-c", fmt.Sprintf("sleep 86400 & wait # %s", marker)},
		})
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        spec.Name,
			Namespace:   spec.Namespace,
			Annotations: annotations,
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			NodeSelector:  spec.NodeSelector,
			Containers:    containers,
		},
	}
	if err := c.Ctrl.Create(ctx, pod); err != nil {
		return nil, fmt.Errorf("create pod %s/%s: %w", spec.Namespace, spec.Name, err)
	}

	return workload.WaitRunning(ctx, c, spec.Namespace, spec.Name, c.Config.PodReadyTimeout, c.Config.PollInterval)
}
