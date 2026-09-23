// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build e2e

package operator

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kai-scheduler/gpu-fractioning/test/e2e/harness"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/daemonset"
	gsc "github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/gpufractioningconfig"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/nodes"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/k8s/pods"
	"github.com/kai-scheduler/gpu-fractioning/test/e2e/waiter"
)

// MpsdFaultAndRecovery — mpsd fault, AND-gated status, node isolation, and
// recovery. Holding the mpsd control socket away makes the mpsd readiness probe
// fail, and asserts:
//   - the faulted node's gpu-fractioning condition flips False;
//   - CR MpsdReady=False while FractiondReady stays True, so aggregate Ready=False;
//   - another matching node stays True (daemon health is scoped per node);
//   - on release, everything returns to Ready (assertSteady).
func caseMpsdFaultAndRecovery(ctx context.Context, t *testing.T) {
	gpuNodes := h.GPUNodeNames(ctx, t)
	if len(gpuNodes) == 0 {
		t.Skip("no matching GPU nodes")
	}
	node := gpuNodes[0]
	if _, ok := h.PodOnNode(ctx, t, harness.ComponentMpsd, node); !ok {
		t.Skipf("no mpsd pod on node %s", node)
	}

	// startMpsdSocketRemoval removes the control socket from the mpsd pod on the node.
	// to revert the fault, call the stop function.
	stop := startMpsdSocketRemoval(node)
	var once sync.Once
	release := func() { once.Do(stop) }
	defer release()

	// node condition flips False on the faulted node.
	if _, err := nodes.WaitCondition(ctx, h.Client(), node, harness.NodeConditionType, corev1.ConditionFalse, h.CondTimeout(), h.PollInterval()); err != nil {
		t.Fatalf("node %s condition did not flip False under mpsd fault: %v", node, err)
	}

	// AND-gate: MpsdReady False, FractiondReady still True, Ready False.
	obj, err := gsc.WaitCondition(ctx, h.Client(), gsc.DefaultName, harness.CondMpsdReady, metav1.ConditionFalse, h.CondTimeout(), h.PollInterval())
	if err != nil {
		t.Fatalf("CR MpsdReady did not go False: %v", err)
	}
	if cond, ok := gsc.Condition(obj, harness.CondFractiondReady); !ok || cond.Status != metav1.ConditionTrue {
		t.Errorf("FractiondReady = %s, want True (fractiond must be unaffected by an mpsd fault)", harness.CondStatus(ok, cond))
	}
	// Aggregate Ready must be False (a component is down); we assert only the
	// status, not the reason. The per-daemon conditions above already pin down
	// *why* (MpsdReady False, FractiondReady True). The aggregate reason is an
	// operator implementation detail that varies by environment: when Ready is
	// False the operator consults its GPU-operator dependency checker, so a real
	// GPU cluster reports ComponentNotReady while a fake-gpu-operator cluster
	// (no ClusterPolicy/CSV to verify) reports GPUOperatorNotReady. Asserting the
	// reason here would couple this case to the cluster's GPU-operator setup.
	if cond, ok := gsc.Condition(obj, harness.CondReady); !ok || cond.Status != metav1.ConditionFalse {
		t.Errorf("Ready = %s, want False", harness.CondStatus(ok, cond))
	}

	// Isolation — another matching node stays True (needs ≥2 matching nodes).
	if len(gpuNodes) >= 2 {
		other := gpuNodes[1]
		n, err := nodes.Get(ctx, h.Client(), other)
		if err != nil {
			t.Errorf("get isolation node %s: %v", other, err)
		} else if cond, ok := nodes.Condition(n, harness.NodeConditionType); !ok || cond.Status != corev1.ConditionTrue {
			t.Errorf("isolation node %s condition = %s, want True", other, harness.NodeCondStatus(ok, cond))
		}
	}

	// release the fault; everything recovers.
	release()
	h.AssertSteady(ctx, t)
}

// FractiondSocketFaultIsolation — an unreachable NRI socket path breaks fractiond
// only, and the breakage shows up at every layer while mpsd stays healthy:
//   - CR: FractiondReady=False, MpsdReady=True, aggregate Ready=False;
//   - DaemonSet: fractiond DS degraded (NumberReady < desired), mpsd DS unaffected;
//   - node: at least one targeted node's gpu-fractioning condition flips False
//     (the condition is AND-gated on both daemons per node).
//
// Reverting recovers — withCRConfig's deferred revert + assertSteady re-checks CR,
// DaemonSets, and node conditions back to True.
func caseFractiondSocketFaultIsolation(ctx context.Context, t *testing.T) {
	gpuNodes := h.GPUNodeNames(ctx, t)
	if len(gpuNodes) == 0 {
		// With no target nodes the fractiond DS has desired=0 and no pods, so a bad
		// socket path can't break fractiond (FractiondReady stays True/NoTargetNodes).
		t.Skip("no matching GPU nodes; a fractiond fault has no target pods to break")
	}
	patch := []byte(`{"spec":{"fractioningAgent":{"nriSocketPath":"/var/run/nri-e2e-bogus/nri.sock"}}}`)
	revert := []byte(`{"spec":{"fractioningAgent":{"nriSocketPath":null}}}`)
	h.WithCRConfig(ctx, t, patch, revert, func() {
		// CR conditions: fractiond down, mpsd isolated, aggregate down.
		obj, err := gsc.WaitCondition(ctx, h.Client(), gsc.DefaultName, harness.CondFractiondReady, metav1.ConditionFalse, h.RolloutTimeout(), h.PollInterval())
		if err != nil {
			t.Fatalf("FractiondReady did not go False with a bad NRI socket path: %v", err)
		}
		if cond, ok := gsc.Condition(obj, harness.CondMpsdReady); !ok || cond.Status != metav1.ConditionTrue {
			t.Errorf("MpsdReady = %s, want True (mpsd must be unaffected by a fractiond fault)", harness.CondStatus(ok, cond))
		}
		if cond, ok := gsc.Condition(obj, harness.CondReady); !ok || cond.Status != metav1.ConditionFalse {
			t.Errorf("Ready = %s, want False", harness.CondStatus(ok, cond))
		}

		// DaemonSet: fractiond is degraded (FractiondReady=False derives from its
		// pods failing readiness), while mpsd stays fully rolled out.
		if sh := h.GetDaemonSet(ctx, t, harness.ComponentFractiond); sh.Status.NumberReady >= sh.Status.DesiredNumberScheduled {
			t.Errorf("fractiond DS NumberReady=%d desired=%d, want NumberReady < desired under the fault",
				sh.Status.NumberReady, sh.Status.DesiredNumberScheduled)
		}
		if mp := h.GetDaemonSet(ctx, t, harness.ComponentMpsd); !daemonset.RolledOut(mp) {
			t.Errorf("mpsd DS not fully rolled out (ready=%d desired=%d); it must be unaffected by a fractiond fault",
				mp.Status.NumberReady, mp.Status.DesiredNumberScheduled)
		}

		// Node: the per-node condition is AND-gated on both daemons, so at least
		// one targeted node flips False while fractiond is broken there.
		if err := waiter.PollUntil(ctx, h.CondTimeout(), h.PollInterval(), "a targeted node condition to flip False",
			func(ctx context.Context) (bool, error) {
				for _, name := range gpuNodes {
					n, err := nodes.Get(ctx, h.Client(), name)
					if err != nil {
						return false, err
					}
					if cond, ok := nodes.Condition(n, harness.NodeConditionType); ok && cond.Status == corev1.ConditionFalse {
						return true, nil
					}
				}
				return false, nil
			}); err != nil {
			t.Errorf("no targeted node flipped the gpu-fractioning condition False under the fractiond fault: %v", err)
		}
	})
}

// SupervisorBackoffNoRestart — supervisor backoff, not a pod kill:
// killing the MPS child once makes the supervisor restart it after backoff;
// kubelet must NOT restart the mpsd container (generous liveness), and the
// control socket recovers.
func caseSupervisorBackoffNoRestart(ctx context.Context, t *testing.T) {
	gpuNodes := h.GPUNodeNames(ctx, t)
	if len(gpuNodes) == 0 {
		t.Skip("no matching GPU nodes")
	}
	node := gpuNodes[0]
	pod, ok := h.PodOnNode(ctx, t, harness.ComponentMpsd, node)
	if !ok {
		t.Skipf("no mpsd pod on node %s", node)
	}
	before := harness.ContainerRestarts(pod, containerMpsd)

	killMpsChild(ctx, t, pod)

	// Socket comes back once the supervisor restarts the child after backoff.
	if err := waiter.PollUntil(ctx, h.CondTimeout(), h.PollInterval(), "mpsd control socket to recover",
		func(ctx context.Context) (bool, error) {
			p, ok := h.PodOnNode(ctx, t, harness.ComponentMpsd, node)
			if !ok {
				return false, nil
			}
			return h.SocketExists(ctx, t, p, containerMpsd, mpsControlSocket), nil
		}); err != nil {
		t.Fatalf("mpsd socket did not recover after child kill: %v", err)
	}

	after := h.GetPod(ctx, t, pod.Namespace, pod.Name)
	if got := harness.ContainerRestarts(after, containerMpsd); got != before {
		t.Errorf("mpsd container was restarted by kubelet (%d → %d); the supervisor should have absorbed the child exit", before, got)
	}
	h.AssertSteady(ctx, t)
}

// RetryBudgetExhaustedRestart — exhausted retry budget: with a tiny
// maxRetries and repeated child kills, the supervisor gives up and exits, so
// kubelet restarts the mpsd container (restartCount increments). Reverting
// maxRetries restores steady state.
func caseRetryBudgetExhaustedRestart(ctx context.Context, t *testing.T) {
	gpuNodes := h.GPUNodeNames(ctx, t)
	if len(gpuNodes) == 0 {
		t.Skip("no matching GPU nodes")
	}
	node := gpuNodes[0]

	// Wait for the DaemonSet to roll out PAST its current generation, so the kills
	// below hit a pod running the new maxRetries=2 config — not the pre-patch pod
	// (default maxRetries=0/unlimited), against which the supervisor never gives up.
	beforeGen := h.GetDaemonSet(ctx, t, harness.ComponentMpsd).Generation
	patch := []byte(`{"spec":{"mpsDaemon":{"maxRetries":2,"backoff":"1s"}}}`)
	revert := []byte(`{"spec":{"mpsDaemon":{"maxRetries":null,"backoff":null}}}`)
	// withCRConfig patches the CR config and runs the given function, and always reverts the patch.
	h.WithCRConfig(ctx, t, patch, revert, func() {
		if _, err := daemonset.WaitRolledOutAfter(ctx, h.Client(), h.NS(), harness.DSName(harness.ComponentMpsd), beforeGen, h.RolloutTimeout(), h.PollInterval()); err != nil {
			t.Fatalf("mpsd rollout after maxRetries change: %v", err)
		}
		pod, ok := h.PodOnNode(ctx, t, harness.ComponentMpsd, node)
		if !ok {
			t.Skipf("no mpsd pod on node %s", node)
		}
		before := harness.ContainerRestarts(pod, containerMpsd)

		// Kill the child faster than it can be considered "stable", enough
		// times to exhaust the (2) retry budget.
		for i := 0; i < 4; i++ {
			if p, ok := h.PodOnNode(ctx, t, harness.ComponentMpsd, node); ok {
				killMpsChildBestEffort(ctx, p)
			}
			time.Sleep(2 * time.Second)
		}

		if err := waiter.PollUntil(ctx, h.CondTimeout(), h.PollInterval(), "mpsd container to be restarted by kubelet",
			func(ctx context.Context) (bool, error) {
				p, ok := h.PodOnNode(ctx, t, harness.ComponentMpsd, node)
				if !ok {
					return false, nil
				}
				return harness.ContainerRestarts(p, containerMpsd) > before, nil
			}); err != nil {
			t.Errorf("mpsd container was not restarted after exhausting maxRetries: %v", err)
		}
	})
}

// SpecDriftReverted — spec drift on a managed DaemonSet is reverted by
// the controller.
func caseSpecDriftReverted(ctx context.Context, t *testing.T) {
	name := harness.DSName(harness.ComponentFractiond)
	const marker = "e2e.drift/marker"
	// Inject a pod-template annotation the operator never sets.
	patch := []byte(fmt.Sprintf(`{"spec":{"template":{"metadata":{"annotations":{%q:"1"}}}}}`, marker))
	ds := &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Namespace: h.NS(), Name: name}}
	if err := h.Client().Ctrl.Patch(ctx, ds, ctrlclient.RawPatch(types.MergePatchType, patch)); err != nil {
		t.Fatalf("inject drift into %s: %v", name, err)
	}
	// The controller's create-or-update reconcile should strip it.
	if err := waiter.PollUntil(ctx, h.CondTimeout(), h.PollInterval(), "controller to revert DaemonSet drift",
		func(ctx context.Context) (bool, error) {
			cur, err := daemonset.Get(ctx, h.Client(), h.NS(), name)
			if err != nil {
				return false, err
			}
			_, present := cur.Spec.Template.Annotations[marker]
			return !present, nil
		}); err != nil {
		t.Errorf("drift annotation %s was not reverted: %v", marker, err)
	}
	h.AssertSteady(ctx, t)
}

// DeletedDaemonSetRecreated — a deleted managed DaemonSet is recreated
// by the controller.
func caseDeletedDaemonSetRecreated(ctx context.Context, t *testing.T) {
	name := harness.DSName(harness.ComponentMpsd)
	ds := &appsv1.DaemonSet{ObjectMeta: metav1.ObjectMeta{Namespace: h.NS(), Name: name}}
	if err := h.Client().Ctrl.Delete(ctx, ds); err != nil {
		t.Fatalf("delete %s: %v", name, err)
	}
	if _, err := daemonset.WaitRolledOut(ctx, h.Client(), h.NS(), name, h.RolloutTimeout(), h.PollInterval()); err != nil {
		t.Errorf("controller did not recreate+roll out %s: %v", name, err)
	}
	h.AssertSteady(ctx, t)
}

// ControllerRestartTransparent — a controller restart is transparent:
// the operator comes back Available, FX-STEADY holds, and the managed DaemonSets
// are not re-rolled (no generation churn from a fresh reconcile).
func caseControllerRestartTransparent(ctx context.Context, t *testing.T) {
	shGen := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation
	mpGen := h.GetDaemonSet(ctx, t, harness.ComponentMpsd).Generation

	dep := h.OperatorDeployment(ctx, t)
	restartDeployment(ctx, t, dep)

	if err := waiter.PollUntil(ctx, h.RolloutTimeout(), h.PollInterval(), "operator Deployment to become Available after restart",
		func(ctx context.Context) (bool, error) {
			d, err := getDeployment(ctx, dep.Namespace, dep.Name)
			if err != nil {
				return false, err
			}
			return d.Status.ObservedGeneration >= d.Generation &&
				d.Status.UpdatedReplicas == d.Status.Replicas &&
				d.Status.AvailableReplicas >= 1, nil
		}); err != nil {
		t.Fatalf("operator did not recover after restart: %v", err)
	}

	h.AssertSteady(ctx, t)

	if g := h.GetDaemonSet(ctx, t, harness.ComponentFractiond).Generation; g != shGen {
		t.Errorf("fractiond DaemonSet re-rolled across controller restart: gen %d → %d", shGen, g)
	}
	if g := h.GetDaemonSet(ctx, t, harness.ComponentMpsd).Generation; g != mpGen {
		t.Errorf("mpsd DaemonSet re-rolled across controller restart: gen %d → %d", mpGen, g)
	}
}

// ── fault primitives ─────────────────────────────────────────────────────────

// startMpsdSocketRemoval loops `rm -f <control socket>` in the mpsd pod on node
// until the returned stop func is called, holding the socket away so the mpsd
// readiness probe keeps failing. It re-resolves the pod each iteration (the pod
// may be replaced) and never calls t.* (it runs on a background goroutine).
func startMpsdSocketRemoval(node string) func() {
	ctx, cancel := harness.BackgroundCtx()
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			list, err := pods.ListByLabel(ctx, h.Client(), h.NS(), harness.ComponentSelector(harness.ComponentMpsd))
			if err != nil {
				continue
			}
			for i := range list {
				if list[i].Spec.NodeName == node {
					_, _ = pods.Exec(ctx, h.Client(), list[i].Namespace, list[i].Name, containerMpsd, []string{"rm", "-f", mpsControlSocket})
				}
			}
		}
	}()
	return func() {
		cancel()
		<-done
	}
}

// killMpsChild kills the supervised MPS control process inside an mpsd pod
// (fatal on exec failure).
func killMpsChild(ctx context.Context, t *testing.T, pod *corev1.Pod) {
	t.Helper()
	if _, err := h.Exec(ctx, pod, containerMpsd, "sh", "-c", "kill $(pidof nvidia-cuda-mps-control) 2>/dev/null || true"); err != nil {
		t.Fatalf("kill mps child in %s: %v", pod.Name, err)
	}
}

// killMpsChildBestEffort is killMpsChild without failing the test (used in a
// loop where a transient exec error is acceptable).
func killMpsChildBestEffort(ctx context.Context, pod *corev1.Pod) {
	_, _ = h.Exec(ctx, pod, containerMpsd, "sh", "-c", "kill $(pidof nvidia-cuda-mps-control) 2>/dev/null || true")
}

// ── deployment helpers ───────────────────────────────────────────────────────

func getDeployment(ctx context.Context, namespace, name string) (*appsv1.Deployment, error) {
	var d appsv1.Deployment
	if err := h.Client().Ctrl.Get(ctx, ctrlclient.ObjectKey{Namespace: namespace, Name: name}, &d); err != nil {
		return nil, err
	}
	return &d, nil
}

// restartDeployment triggers a rollout restart by stamping the standard
// restartedAt annotation on the pod template.
func restartDeployment(ctx context.Context, t *testing.T, dep *appsv1.Deployment) {
	t.Helper()
	patch := []byte(fmt.Sprintf(
		`{"spec":{"template":{"metadata":{"annotations":{"kubectl.kubernetes.io/restartedAt":%q}}}}}`,
		time.Now().Format(time.RFC3339Nano)))
	obj := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Namespace: dep.Namespace, Name: dep.Name}}
	if err := h.Client().Ctrl.Patch(ctx, obj, ctrlclient.RawPatch(types.MergePatchType, patch)); err != nil {
		t.Fatalf("restart operator Deployment %s: %v", dep.Name, err)
	}
}
