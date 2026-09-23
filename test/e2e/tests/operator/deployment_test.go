// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package operator

import (
	"context"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/harness"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/daemonset"
	gsc "github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/gpufractioningconfig"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/nodes"
)

// stabilityWindow is how long the "stable" cases (ReconcileIdempotent,
// ReadyTransitionTimeStable, NodeConditionStable) observe for churn.
const stabilityWindow = 15 * time.Second

// RollsOutToReady — full rollout from FX-CLEAN to FX-STEADY. Owns the CR:
// deletes it (both DaemonSets GC'd), re-creates the shipped default, and asserts
// the operator drives everything back to a healthy steady state — every CR Ready
// condition True with its healthy reason code. This subsumes the old standalone
// HealthySteadyState case: after a successful rollout the resting state is exactly
// what that case checked.
func caseRollsOutToReady(ctx context.Context, t *testing.T) {
	h.CleanCR(ctx, t)
	h.RestoreSteady(ctx, t) // re-creates + asserts FX-STEADY

	obj, err := gsc.Get(ctx, h.Client())
	if err != nil {
		t.Fatalf("get CR after rollout: %v", err)
	}
	if cond, ok := gsc.Condition(obj, harness.CondReady); !ok || cond.Status != metav1.ConditionTrue || cond.Reason != reasonAllComponentsReady {
		t.Errorf("Ready = %s/%q, want True/%s", harness.CondStatus(ok, cond), harness.CondReasonOf(ok, cond), reasonAllComponentsReady)
	}
	for _, ct := range []string{harness.CondFractiondReady, harness.CondMpsdReady} {
		if cond, ok := gsc.Condition(obj, ct); !ok || cond.Status != metav1.ConditionTrue || cond.Reason != reasonAllPodsReady {
			t.Errorf("%s = %s/%q, want True/%s", ct, harness.CondStatus(ok, cond), harness.CondReasonOf(ok, cond), reasonAllPodsReady)
		}
	}
}

// BothDaemonSetsReady — both DaemonSets exist and are fully ready over
// every matching node.
func verifyBothDaemonSetsReady(ctx context.Context, t *testing.T) {
	want := int32(len(h.GPUNodeNames(ctx, t)))
	if want == 0 {
		t.Skip("no matching GPU nodes")
	}
	for _, comp := range []string{harness.ComponentFractiond, harness.ComponentMpsd} {
		ds := h.GetDaemonSet(ctx, t, comp)
		if ds.Status.DesiredNumberScheduled != want || ds.Status.NumberReady != want {
			t.Errorf("%s: desired=%d ready=%d, want both %d", ds.Name, ds.Status.DesiredNumberScheduled, ds.Status.NumberReady, want)
		}
	}
	// fractiond hosts the metricsd sidecar; mpsd is single-container.
	if got := daemonset.ContainerNames(h.GetDaemonSet(ctx, t, harness.ComponentFractiond)); !hasAll(got, containerFractiond, containerMetricsd) {
		t.Errorf("fractiond DaemonSet containers = %v, want to include %s and %s", got, containerFractiond, containerMetricsd)
	}
	if got := daemonset.ContainerNames(h.GetDaemonSet(ctx, t, harness.ComponentMpsd)); !hasAll(got, containerMpsd) {
		t.Errorf("mpsd DaemonSet containers = %v, want to include %s", got, containerMpsd)
	}
}

// NamingAndLabels — DaemonSet naming and management labels.
func verifyNamingAndLabels(ctx context.Context, t *testing.T) {
	for _, comp := range []string{harness.ComponentFractiond, harness.ComponentMpsd} {
		ds := h.GetDaemonSet(ctx, t, comp)
		if ds.Labels[harness.LabelManagedBy] != harness.ManagedByValue {
			t.Errorf("%s: %s=%q, want %q", ds.Name, harness.LabelManagedBy, ds.Labels[harness.LabelManagedBy], harness.ManagedByValue)
		}
		if ds.Labels[harness.LabelComponent] != comp {
			t.Errorf("%s: %s=%q, want %q", ds.Name, harness.LabelComponent, ds.Labels[harness.LabelComponent], comp)
		}
		tmpl := ds.Spec.Template.Labels
		if tmpl[harness.LabelManagedBy] != harness.ManagedByValue || tmpl[harness.LabelComponent] != comp {
			t.Errorf("%s: pod template labels = %v, want managed-by/component set", ds.Name, tmpl)
		}
	}
}

// OwnerReferences — each DaemonSet is owned (controller ref) by the
// default GpuFractioningConfig, so CR deletion garbage-collects them.
func verifyOwnerReferences(ctx context.Context, t *testing.T) {
	for _, comp := range []string{harness.ComponentFractiond, harness.ComponentMpsd} {
		ds := h.GetDaemonSet(ctx, t, comp)
		var owned bool
		for _, ref := range ds.OwnerReferences {
			if ref.Kind == "GpuFractioningConfig" && ref.Name == gsc.DefaultName && ref.Controller != nil && *ref.Controller {
				owned = true
			}
		}
		if !owned {
			t.Errorf("%s: missing controller ownerReference to GpuFractioningConfig/%s (refs=%v)", ds.Name, gsc.DefaultName, ds.OwnerReferences)
		}
	}
}

// SchedulingScope — the CR's nodeSelector fully scopes the operator's footprint,
// verified from real cluster state (not DaemonSet/CR status; readiness and rollout
// are covered by BothDaemonSetsReady / RollsOutToReady):
//   - matching nodes: exactly one managed pod each;
//   - non-matching nodes: no managed pod and no gpu-fractioning node condition.
//
// Each half is guarded on the topology it needs, so a cluster missing one class of
// node still exercises the other. The "exactly one per matching node" check needs
// ≥2 matching nodes to be meaningful (with a single matching node "scoped
// correctly" is indistinguishable from "happens to be on the one node").
func verifySchedulingScope(ctx context.Context, t *testing.T) {
	gpuNodes := h.GPUNodeNames(ctx, t)
	nonGPU := h.NonGPUNodeNames(ctx, t)
	if len(gpuNodes) < 2 && len(nonGPU) == 0 {
		t.Skipf("selector scoping needs ≥2 matching nodes or ≥1 non-matching node; have %d matching, %d non-matching", len(gpuNodes), len(nonGPU))
	}
	gpuSet := toSet(gpuNodes)

	// One pod scan per component drives both the "none on non-matching nodes" and
	// the "exactly one per matching node" assertions.
	for _, comp := range []string{harness.ComponentFractiond, harness.ComponentMpsd} {
		perNode := map[string]int{}
		for _, p := range h.ListComponentPods(ctx, t, comp) {
			node := p.Spec.NodeName
			if node == "" {
				continue
			}
			perNode[node]++
			if !gpuSet[node] {
				t.Errorf("%s pod %s scheduled on non-matching node %s", comp, p.Name, node)
			}
		}
		if len(gpuNodes) >= 2 {
			for _, node := range gpuNodes {
				if perNode[node] != 1 {
					t.Errorf("%s: node %s hosts %d pods, want exactly 1", comp, node, perNode[node])
				}
			}
		}
	}

	// Non-matching nodes are never annotated with the gpu-fractioning condition.
	for _, name := range nonGPU {
		n, err := nodes.Get(ctx, h.Client(), name)
		if err != nil {
			t.Errorf("get node %s: %v", name, err)
			continue
		}
		if _, ok := nodes.Condition(n, harness.NodeConditionType); ok {
			t.Errorf("non-matching node %s unexpectedly carries the gpu-fractioning condition", name)
		}
	}
}

// FractiondPodSpec — fractiond pod spec: hostPID, privileged main
// container, the metricsd sidecar, and the NRI-socket + MPS-pipe hostPath mounts.
func verifyFractiondPodSpec(ctx context.Context, t *testing.T) {
	ds := h.GetDaemonSet(ctx, t, harness.ComponentFractiond)
	if !ds.Spec.Template.Spec.HostPID {
		t.Error("fractiond pod: hostPID = false, want true")
	}
	if ds.Spec.Template.Spec.RuntimeClassName != nil {
		t.Errorf("fractiond pod: runtimeClassName = %q, want unset", *ds.Spec.Template.Spec.RuntimeClassName)
	}
	sh, ok := daemonset.Container(ds, containerFractiond)
	if !ok {
		t.Fatal("fractiond container not found")
	}
	if sh.SecurityContext == nil || sh.SecurityContext.Privileged == nil || !*sh.SecurityContext.Privileged {
		t.Error("fractiond container is not privileged")
	}
	if _, ok := daemonset.Container(ds, containerMetricsd); !ok {
		t.Error("metricsd sidecar not present in fractiond DaemonSet")
	}
	mounts := daemonset.HostPathMounts(ds, containerFractiond)
	if !mounts[harness.MPSPipeDir] {
		t.Errorf("fractiond: missing MPS pipe hostPath mount %s (mounts=%v)", harness.MPSPipeDir, mounts)
	}
	if !mounts["/var/run/nri"] {
		t.Errorf("fractiond: missing NRI socket hostPath mount /var/run/nri (mounts=%v)", mounts)
	}
}

// MpsdPodSpec — mpsd pod spec: hostPID, privileged, service-accounted for node
// labeling, MPS pipe + log hostPath mounts, and a control-socket readiness probe.
func verifyMpsdPodSpec(ctx context.Context, t *testing.T) {
	ds := h.GetDaemonSet(ctx, t, harness.ComponentMpsd)
	// Without hostPID the MPS control daemon resolves every client's PID as 0,
	// memacct registration fails, and GPU memory limits go unenforced.
	if !ds.Spec.Template.Spec.HostPID {
		t.Error("mpsd pod: hostPID = false, want true")
	}
	if ds.Spec.Template.Spec.ServiceAccountName == "" {
		t.Error("mpsd pod: serviceAccountName is empty")
	}
	if ds.Spec.Template.Spec.RuntimeClassName != nil {
		t.Errorf("mpsd pod: runtimeClassName = %q, want unset", *ds.Spec.Template.Spec.RuntimeClassName)
	}
	mp, ok := daemonset.Container(ds, containerMpsd)
	if !ok {
		t.Fatal("mpsd container not found")
	}
	if mp.SecurityContext == nil || mp.SecurityContext.Privileged == nil || !*mp.SecurityContext.Privileged {
		t.Error("mpsd container is not privileged")
	}
	if got := envFieldRef(mp.Env, "NODE_NAME"); got != "spec.nodeName" {
		t.Errorf("mpsd NODE_NAME fieldRef = %q, want spec.nodeName", got)
	}
	if got := envValue(mp.Env, "NVIDIA_VISIBLE_DEVICES"); got != "all" {
		t.Errorf("mpsd NVIDIA_VISIBLE_DEVICES = %q, want all", got)
	}
	if got := envValue(mp.Env, "NVIDIA_DRIVER_CAPABILITIES"); got != "compute,utility" {
		t.Errorf("mpsd NVIDIA_DRIVER_CAPABILITIES = %q, want compute,utility", got)
	}
	mounts := daemonset.HostPathMounts(ds, containerMpsd)
	if !mounts[harness.MPSPipeDir] {
		t.Errorf("mpsd: missing MPS pipe hostPath mount %s (mounts=%v)", harness.MPSPipeDir, mounts)
	}
	if mp.ReadinessProbe == nil || mp.ReadinessProbe.Exec == nil ||
		!containsArg(mp.ReadinessProbe.Exec.Command, mpsControlSocket) {
		t.Errorf("mpsd: readiness probe is not the control-socket check (probe=%+v)", mp.ReadinessProbe)
	}
}

func envValue(env []corev1.EnvVar, name string) string {
	for _, e := range env {
		if e.Name == name {
			return e.Value
		}
	}
	return ""
}

func envFieldRef(env []corev1.EnvVar, name string) string {
	for _, e := range env {
		if e.Name == name && e.ValueFrom != nil && e.ValueFrom.FieldRef != nil {
			return e.ValueFrom.FieldRef.FieldPath
		}
	}
	return ""
}

// ReconcileIdempotent — steady-state idempotency: no DaemonSet/CR
// generation churn while nothing changes (the controller must not fight itself).
func verifyReconcileIdempotent(ctx context.Context, t *testing.T) {
	shGen := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation
	mpGen := h.GetDaemonSet(ctx, t, harness.ComponentMpsd).Generation
	crObj, err := gsc.Get(ctx, h.Client())
	if err != nil {
		t.Fatalf("get CR: %v", err)
	}
	crGen := crObj.Generation

	time.Sleep(stabilityWindow)

	if g := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation; g != shGen {
		t.Errorf("fractiond DaemonSet generation churned: %d → %d", shGen, g)
	}
	if g := h.GetDaemonSet(ctx, t, harness.ComponentMpsd).Generation; g != mpGen {
		t.Errorf("mpsd DaemonSet generation churned: %d → %d", mpGen, g)
	}
	if crObj2, err := gsc.Get(ctx, h.Client()); err != nil {
		t.Errorf("re-get CR: %v", err)
	} else if crObj2.Generation != crGen {
		t.Errorf("CR generation churned: %d → %d", crGen, crObj2.Generation)
	}
}

// ReadyTransitionTimeStable — the aggregate Ready condition's
// lastTransitionTime is stable while the status stays True (SetCondition
// preserves it), so consumers can trust "how long has it been Ready".
func verifyReadyTransitionTimeStable(ctx context.Context, t *testing.T) {
	obj, err := gsc.Get(ctx, h.Client())
	if err != nil {
		t.Fatalf("get CR: %v", err)
	}
	before, ok := gsc.Condition(obj, harness.CondReady)
	if !ok {
		t.Fatal("Ready condition absent")
	}
	time.Sleep(stabilityWindow)
	obj2, err := gsc.Get(ctx, h.Client())
	if err != nil {
		t.Fatalf("re-get CR: %v", err)
	}
	after, _ := gsc.Condition(obj2, harness.CondReady)
	if !after.LastTransitionTime.Equal(&before.LastTransitionTime) {
		t.Errorf("Ready lastTransitionTime moved while status stayed True: %v → %v", before.LastTransitionTime, after.LastTransitionTime)
	}
}

// NodeConditionReady — every matching node carries the gpu-fractioning
// Ready condition True with reason AllDaemonsReady.
func verifyNodeConditionReady(ctx context.Context, t *testing.T) {
	list, err := nodes.ListGPUNodes(ctx, h.Client())
	if err != nil {
		t.Fatalf("list GPU nodes: %v", err)
	}
	if len(list) == 0 {
		t.Skip("no matching GPU nodes")
	}
	for i := range list {
		cond, ok := nodes.Condition(&list[i], harness.NodeConditionType)
		if !ok || cond.Status != corev1.ConditionTrue || cond.Reason != reasonAllDaemonsReady {
			t.Errorf("node %s condition = %s/%q, want True/%s", list[i].Name, harness.NodeCondStatus(ok, cond), harness.NodeCondReason(ok, cond), reasonAllDaemonsReady)
		}
	}
}

// NodeConditionStable — the node condition's lastTransitionTime is
// stable while daemons stay healthy (the controller must not re-flip a steady
// condition).
func verifyNodeConditionStable(ctx context.Context, t *testing.T) {
	node := h.GPUNodeNames(ctx, t)
	if len(node) == 0 {
		t.Skip("no matching GPU nodes")
	}
	n0, err := nodes.Get(ctx, h.Client(), node[0])
	if err != nil {
		t.Fatalf("get node: %v", err)
	}
	before, ok := nodes.Condition(n0, harness.NodeConditionType)
	if !ok {
		t.Fatalf("node %s missing gpu-fractioning condition", node[0])
	}
	time.Sleep(stabilityWindow)
	n1, err := nodes.Get(ctx, h.Client(), node[0])
	if err != nil {
		t.Fatalf("re-get node: %v", err)
	}
	after, _ := nodes.Condition(n1, harness.NodeConditionType)
	if after.Status != corev1.ConditionTrue {
		t.Errorf("node condition flipped away from True: %s", after.Status)
	}
	if !after.LastTransitionTime.Equal(&before.LastTransitionTime) {
		t.Errorf("node condition lastTransitionTime moved while healthy: %v → %v", before.LastTransitionTime, after.LastTransitionTime)
	}
}

// MpsdReadinessSocket — mpsd reaches readiness via its control-socket
// probe: the socket exists in the pod and the container is Ready. This is what
// the fake-mps image (or a real MPS daemon) must deliver for FX-STEADY to hold.
func verifyMpsdReadinessSocket(ctx context.Context, t *testing.T) {
	pod := h.FirstComponentPod(ctx, t, harness.ComponentMpsd)
	if !harness.ContainerReady(pod, containerMpsd) {
		t.Fatalf("mpsd container in %s is not Ready", pod.Name)
	}
	if !h.SocketExists(ctx, t, pod, containerMpsd, mpsControlSocket) {
		t.Errorf("mpsd control socket %s missing in pod %s", mpsControlSocket, pod.Name)
	}
}

// ── small local helpers ──────────────────────────────────────────────────────

func hasAll(have []string, want ...string) bool {
	set := toSet(have)
	for _, w := range want {
		if !set[w] {
			return false
		}
	}
	return true
}

func toSet(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

func containsArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}
