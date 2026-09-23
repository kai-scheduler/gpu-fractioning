// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/common/daemonmgr"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/fractioningmanager/components/fractiond"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/fractioningmanager/components/mpsd"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/metrics"
	"github.com/kai-scheduler/gpu-fractioning/pkg/driverinfo"
	"github.com/kai-scheduler/gpu-fractioning/pkg/env"
)

const (
	// defaultGpuFractioningConfigName is the CRD-enforced singleton name
	// (+kubebuilder validation requires metadata.name == "default"), so
	// dependency watch events can enqueue this one cluster-scoped object.
	defaultGpuFractioningConfigName = "default"
	requeueInterval                 = 30 * time.Second
	podListPageSize                 = 500
	nodeListPageSize                = 500

	// NodeConditionCleanupFinalizer blocks GpuFractioningConfig deletion until the
	// gpu-fractioning-owned node state is removed. Without it, DaemonSets are
	// garbage-collected via owner references but nodes keep advertising stale
	// Ready conditions and driver-major labels.
	NodeConditionCleanupFinalizer = "gpu-fractioning.nvidia.com/cleanup-node-conditions"
)

// GpuFractioningConfigReconciler reconciles a GpuFractioningConfig object
type GpuFractioningConfigReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder

	// APIReader is the manager's uncached reader (direct API server reads).
	// Pods and Nodes are read through it — never through the cached Client —
	// so the operator never maintains cluster-scale Pod/Node informers. See
	// patchNodeConditions for the rationale.
	APIReader client.Reader

	// Namespace is the namespace where DaemonSets and their pods are created.
	// Since GpuFractioningConfig is cluster-scoped, the DaemonSets use the
	// operator's own namespace (injected via POD_NAMESPACE).
	Namespace string

	// DefaultImages holds the Helm-injected default image for each daemon,
	// keyed by daemon name (e.g. "fractiond").
	DefaultImages map[string]daemonmgr.ImageSpec

	// DaemonServiceAccountName is the ServiceAccount assigned to daemon pods that
	// need limited Kubernetes API access, such as mpsd patching its own node label
	// during startup.
	DaemonServiceAccountName string

	// MpsdAuditLog is the Helm-injected toggle for the mpsd MPS memacct audit
	// log, forwarded to the mpsd container via env.
	MpsdAuditLog bool

	// SupportSMSharing is the Helm-injected installation-time toggle for the
	// sm-sharing compute mode, forwarded to both the mpsd and fractiond
	// containers. See buildDaemons.
	SupportSMSharing bool

	// FIPSOnly is the Helm-injected installation-time toggle for FIPS-only
	// enforcement, forwarded to every daemon container via BuildOptions.
	// See daemonmgr.FIPSOnlyEnv.
	FIPSOnly bool

	// SkipGpuStackVersionChecks is the Helm-injected installation-time bypass
	// for the GPU stack version gates, forwarded to GpuOperatorChecker. It
	// disables verification only — it does not make an unsupported GPU stack
	// work. See GpuOperatorDependencyChecker.
	SkipGpuStackVersionChecks bool

	// GpuOperatorChecker checks NVIDIA GPU Operator dependency failures against
	// the aggregate Ready condition. Version validation runs even when managed
	// daemons are healthy so a GPU Operator downgrade is reflected in status as
	// soon as the dependency watch fires; transient GPU Operator readiness is used
	// only to refine an already-unhealthy status.
	GpuOperatorChecker GpuOperatorDependencyChecker

	// GpuDriverChecker refines already-false node Ready conditions with CUDA
	// driver dependency failures.
	GpuDriverChecker GpuDriverDependencyChecker
}

// NewGpuFractioningConfigReconciler creates a reconciler with safe defaults.
func NewGpuFractioningConfigReconciler(
	c client.Client,
	apiReader client.Reader,
	scheme *runtime.Scheme,
	recorder record.EventRecorder,
	namespace string,
	defaultImages map[string]daemonmgr.ImageSpec,
	daemonServiceAccountName string,
	mpsdAuditLog bool,
	supportSMSharing bool,
	fipsOnly bool,
	skipGpuStackVersionChecks bool,
) *GpuFractioningConfigReconciler {
	return &GpuFractioningConfigReconciler{
		Client:                    c,
		APIReader:                 apiReader,
		Scheme:                    scheme,
		Recorder:                  recorder,
		Namespace:                 namespace,
		DefaultImages:             defaultImages,
		DaemonServiceAccountName:  daemonServiceAccountName,
		MpsdAuditLog:              mpsdAuditLog,
		SupportSMSharing:          supportSMSharing,
		FIPSOnly:                  fipsOnly,
		SkipGpuStackVersionChecks: skipGpuStackVersionChecks,
		GpuOperatorChecker:        NewGpuOperatorDependencyChecker(apiReader, skipGpuStackVersionChecks),
		GpuDriverChecker:          NewGpuDriverDependencyChecker(apiReader),
	}
}

// buildDaemons constructs the managed daemons from the current CRD spec,
// so each reconcile sees the latest configuration.
func buildDaemons(spec *v1alpha1.GpuFractioningConfigSpec, mpsdAuditLog, supportSMSharing bool) []daemonmgr.ManagedDaemon {
	return []daemonmgr.ManagedDaemon{
		fractiond.NewFractiondDaemon(spec.FractioningAgent, spec.MetricsAgent, supportSMSharing),
		mpsd.NewMpsdDaemon(spec.MpsDaemon, mpsdAuditLog, supportSMSharing),
	}
}

func (r *GpuFractioningConfigReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	var config v1alpha1.GpuFractioningConfig
	if err := r.Get(ctx, req.NamespacedName, &config); err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("GpuFractioningConfig resource deleted")
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, fmt.Errorf("unable to fetch GpuFractioningConfig: %w", err)
	}

	if !config.DeletionTimestamp.IsZero() {
		return r.reconcileDelete(ctx, &config)
	}

	// Register the cleanup finalizer before any DaemonSets are created or node
	// conditions patched, so deletion can never race past the cleanup logic.
	if controllerutil.AddFinalizer(&config, NodeConditionCleanupFinalizer) {
		if err := r.Update(ctx, &config); err != nil {
			return ctrl.Result{}, fmt.Errorf("adding finalizer: %w", err)
		}
	}

	log.Info("reconciling GpuFractioningConfig", "generation", config.Generation)

	// Derive previous health from the persisted Ready condition so the
	// controller stays stateless across restarts and leader elections.
	wasUnhealthy := !isConditionTrue(config.Status.Conditions, daemonmgr.ConditionReady)

	// Reconcile each managed daemon and collect health + conditions.
	var needsRequeue bool
	daemons := buildDaemons(&config.Spec, r.MpsdAuditLog, r.SupportSMSharing)

	opts := r.buildOptions(&config)
	for _, daemon := range daemons {

		health, reconcileErr := daemonmgr.ReconcileDaemon(ctx, r.Client, r.Scheme, &config, daemon, opts)
		if reconcileErr != nil {
			r.Recorder.Eventf(&config, corev1.EventTypeWarning, "ReconcileError", "%s: %v", daemon.Name(), reconcileErr)
			health = &daemonmgr.DaemonHealth{}
			needsRequeue = true
		}

		cond := daemonmgr.DaemonHealthToCondition(daemon.Name(), health, config.Generation, reconcileErr)
		daemonmgr.SetCondition(&config.Status, cond)
		metrics.SetDaemonHealth(daemon.Name(), health.ReadyNodes, health.DesiredNodes)

		if health.ReadyNodes < health.DesiredNodes {
			needsRequeue = true
		}
	}

	// Aggregate Ready condition.
	readyCond := daemonmgr.AggregateReadyCondition(config.Status.Conditions, config.Generation)
	readyCond, err := r.GpuOperatorChecker.Check(ctx, &config, readyCond)
	if err != nil {
		r.Recorder.Eventf(&config, corev1.EventTypeWarning, "DependencyCheckError", "%v", err)
		log.Error(err, "failed to check GPU stack dependencies")
		needsRequeue = true
	}
	daemonmgr.SetCondition(&config.Status, readyCond)

	// Reflect GPU driver-upgrade state on the CR. The nodeAffinity on the managed
	// DaemonSets already drains daemons from upgrading nodes (see
	// daemonmgr.driverUpgradeNodeAffinity); this surfaces that as a condition.
	// The reconcile is retriggered by the owned DaemonSets' status changing as
	// pods drain/return, so no Node watch is required.
	upgrading, upErr := r.evaluateDriverUpgrade(ctx, &config)
	if upErr != nil {
		log.Error(upErr, "failed to evaluate driver-upgrade state, will requeue")
		needsRequeue = true
	} else {
		daemonmgr.SetCondition(&config.Status, daemonmgr.DriverUpgradeCondition(upgrading, config.Generation))
	}

	// Patch gpu-fractioning.nvidia.com/Ready on each node. We only list pods
	// when something is unhealthy or when transitioning back to healthy
	// (to flip nodes from False→True). In steady state this is skipped entirely.
	if needsRequeue || wasUnhealthy {
		nodesUnhealthy, err := r.patchNodeConditions(ctx, r.Namespace)
		if err != nil {
			// In case it that prev state was unhealthy, but we failed to patch nodes, we want to requeue
			log.Error(err, "failed to patch node conditions, will requeue")
			needsRequeue = true
		} else {
			needsRequeue = needsRequeue || nodesUnhealthy
		}
	}

	// Update CR status.
	config.Status.ObservedGeneration = config.Generation
	if err := r.Status().Update(ctx, &config); err != nil {
		return ctrl.Result{}, fmt.Errorf("unable to update status: %w", err)
	}

	if needsRequeue {
		return ctrl.Result{RequeueAfter: requeueInterval}, nil
	}
	return ctrl.Result{}, nil
}

// evaluateDriverUpgrade reports whether any GPU node targeted by the CR is
// actively undergoing a driver upgrade, and clears the stale gpu-fractioning Ready
// condition from those nodes. The filter is entirely server-side: nodes matching
// the CR's nodeSelector whose gpu-operator upgrade-state label is set to
// something other than the terminal "upgrade-done" — i.e. actively upgrading.
// Excluding "upgrade-done" at the server matters because that label persists on
// nodes after an upgrade, so a plain "has the label" list would return every
// targeted GPU node in a mature cluster; the surviving set is just the nodes
// mid-upgrade right now (bounded).
//
// Draining a node for an upgrade removes its daemon pods, and patchNodeConditions
// is pod-driven, so it never revisits the node and would leave a stale
// Ready=True there. We remove the condition here so the node stops advertising a
// readiness it no longer has; it is re-added (Ready=True) once the daemons
// reschedule after the upgrade. Reads go through the uncached APIReader for the
// same reason as patchNodeConditions: no cluster-scale Node informer.
func (r *GpuFractioningConfigReconciler) evaluateDriverUpgrade(ctx context.Context, config *v1alpha1.GpuFractioningConfig) (bool, error) {
	// Exists is required alongside NotIn: set-based NotIn also matches nodes that
	// lack the label, so on its own it would count non-upgrading nodes.
	exists, err := labels.NewRequirement(daemonmgr.DriverUpgradeStateLabel, selection.Exists, nil)
	if err != nil {
		return false, err
	}
	notDone, err := labels.NewRequirement(daemonmgr.DriverUpgradeStateLabel, selection.NotIn, []string{daemonmgr.DriverUpgradeStateDone})
	if err != nil {
		return false, err
	}
	selector := labels.SelectorFromSet(config.Spec.NodeSelector).Add(*exists, *notDone)

	var nodeList corev1.NodeList
	if err := r.APIReader.List(ctx, &nodeList, client.MatchingLabelsSelector{Selector: selector}); err != nil {
		return false, fmt.Errorf("listing driver-upgrade nodes: %w", err)
	}

	var errs []error
	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		// Only patch when there is actually a condition to remove, so a node stays
		// untouched across reconciles once cleared.
		if _, found := daemonmgr.FindNodeCondition(node); found {
			if err := daemonmgr.RemoveNodeCondition(ctx, r.Client, node.Name); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return len(nodeList.Items) > 0, errors.Join(errs...)
}

// buildOptions resolves the BuildOptions shared by all daemons. DefaultImages
// contains the Helm-injected defaults; each daemon's BuildDaemonSet merges its
// own CRD overrides internally.
func (r *GpuFractioningConfigReconciler) buildOptions(config *v1alpha1.GpuFractioningConfig) daemonmgr.BuildOptions {
	return daemonmgr.BuildOptions{
		Namespace:          r.Namespace,
		NodeSelector:       config.Spec.NodeSelector,
		ServiceAccountName: r.DaemonServiceAccountName,
		RuntimeClassName:   daemonmgr.ResolveRuntimeClassName(config.Spec.RuntimeClassName),
		DefaultImages:      r.DefaultImages,
		FIPSOnly:           r.FIPSOnly,
	}
}

// reconcileDelete handles a GpuFractioningConfig with a non-zero deletionTimestamp:
// it removes the gpu-fractioning-owned condition and driver-major label from the
// nodes the CR targets, then releases the finalizer so deletion can complete. Any
// cleanup error is returned with the finalizer still in place, so
// controller-runtime requeues with backoff until cleanup converges — deletion is
// never unblocked prematurely, but also never blocked forever by a transient
// failure.
func (r *GpuFractioningConfigReconciler) reconcileDelete(ctx context.Context, config *v1alpha1.GpuFractioningConfig) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	if !controllerutil.ContainsFinalizer(config, NodeConditionCleanupFinalizer) {
		return ctrl.Result{}, nil
	}

	if err := daemonmgr.RemoveNodeConditions(ctx, r.APIReader, r.Client, config.Spec.NodeSelector); err != nil {
		return ctrl.Result{}, fmt.Errorf("cleaning up node conditions: %w", err)
	}
	if err := r.removeDriverMajorLabels(ctx, config.Spec.NodeSelector); err != nil {
		return ctrl.Result{}, fmt.Errorf("cleaning up driver labels: %w", err)
	}

	controllerutil.RemoveFinalizer(config, NodeConditionCleanupFinalizer)
	if err := r.Update(ctx, config); err != nil {
		return ctrl.Result{}, fmt.Errorf("removing finalizer: %w", err)
	}

	log.Info("removed node conditions, driver labels and finalizer; deletion can complete")
	return ctrl.Result{}, nil
}

func (r *GpuFractioningConfigReconciler) removeDriverMajorLabels(ctx context.Context, nodeSelector map[string]string) error {
	log := logf.FromContext(ctx)

	patchBytes, err := removeDriverMajorLabelPatch()
	if err != nil {
		return err
	}

	listOpts := []client.ListOption{
		client.MatchingLabels(nodeSelector),
		client.Limit(nodeListPageSize),
	}

	var errs []error
	for {
		var nodeList corev1.NodeList
		if err := r.APIReader.List(ctx, &nodeList, listOpts...); err != nil {
			return fmt.Errorf("listing nodes: %w", err)
		}

		for i := range nodeList.Items {
			node := &nodeList.Items[i]
			if node.Labels[driverinfo.NVIDIADriverMajorLabel] == "" {
				continue
			}

			patchNode := &corev1.Node{}
			patchNode.Name = node.Name
			if err := r.Patch(ctx, patchNode, client.RawPatch(types.MergePatchType, patchBytes)); err != nil {
				log.Error(err, "failed to remove driver label from node", "node", node.Name)
				errs = append(errs, fmt.Errorf("removing driver label from node %s: %w", node.Name, err))
			}
		}

		if nodeList.Continue == "" {
			break
		}
		listOpts = []client.ListOption{
			client.MatchingLabels(nodeSelector),
			client.Limit(nodeListPageSize),
			client.Continue(nodeList.Continue),
		}
	}

	return errors.Join(errs...)
}

func removeDriverMajorLabelPatch() ([]byte, error) {
	patch := map[string]any{
		"metadata": map[string]any{
			"labels": map[string]any{
				driverinfo.NVIDIADriverMajorLabel: nil,
			},
		},
	}
	return json.Marshal(patch)
}

// patchNodeConditions lists all managed daemon pods (across all daemons) and
// patches the gpu-fractioning.nvidia.com/Ready condition on each node.
// A node is marked Ready only when every daemon pod on it is ready.
//
// Pods are listed via the uncached API reader (r.APIReader), NOT the cached
// client, and are intentionally never cached. Rationale: this function only
// runs in the rare unhealthy/recovery path (see Reconcile's
// `needsRequeue || wasUnhealthy` guard), and reconciles are driven by DaemonSet
// status via Owns(DaemonSet) — not by pod events. Maintaining a cluster-scale
// Pod informer (O(nodes*daemons) objects resident, plus watch traffic) purely
// to serve a ~0.1%-of-the-time read is wasteful at large node counts, so we
// read on demand instead. Because the API reader supports server-side
// pagination, we page the list to bound peak memory during an incident.
//
// To avoid holding the full result in memory, nodes are patched inline as pods
// are streamed in pages. Only nodes with at least one unhealthy pod are tracked
// (the "unhealthy" set); in steady state this set is empty. If a node was
// already patched True and a later pod reveals it unhealthy, the condition is
// overwritten to False — correctness over patch count.
func (r *GpuFractioningConfigReconciler) patchNodeConditions(ctx context.Context, namespace string) (bool, error) {
	log := logf.FromContext(ctx)

	managedByLabels := map[string]string{
		daemonmgr.LabelManagedBy: daemonmgr.ManagedByValue,
	}

	listOpts := []client.ListOption{
		client.InNamespace(namespace),
		client.MatchingLabels(managedByLabels),
		client.Limit(podListPageSize),
	}

	// Tracks nodes already marked unhealthy so we don't overwrite a False
	// with True when a later ready pod on the same node is encountered.
	// In steady state (all healthy) this set stays empty.
	unhealthyNodes := make(map[string]struct{})
	// All managed nodes seen this pass, so we can emit ready/degraded gauges.
	seenNodes := make(map[string]struct{})

	// Per-node patch failures are collected and joined so the caller requeues
	// and retries them; one bad node does not abort the rest of the pass.
	var errs []error

	var podList corev1.PodList
	for {
		if err := r.APIReader.List(ctx, &podList, listOpts...); err != nil {
			errs = append(errs, fmt.Errorf("listing pods for node condition patching: %w", err))
			return len(unhealthyNodes) > 0, errors.Join(errs...)
		}

		for i := range podList.Items {
			pod := &podList.Items[i]
			if pod.Spec.NodeName == "" {
				continue
			}
			seenNodes[pod.Spec.NodeName] = struct{}{}

			// Determine per-node health: if any daemon pod on this node is
			// not ready, the node is marked unhealthy. The GPU driver checker may
			// refine the message with a GPU driver version issue if the node's
			// labels explain why the daemon cannot start.
			// Once a node is marked unhealthy, later healthy pods on the
			// same node are skipped to avoid overwriting the False condition.
			ready := isPodReady(pod)
			if !ready {
				unhealthyNodes[pod.Spec.NodeName] = struct{}{}
			} else if _, poisoned := unhealthyNodes[pod.Spec.NodeName]; poisoned {
				continue
			}

			condition := nodeConditionForPod(ready, pod)
			condition, err := r.GpuDriverChecker.Check(ctx, pod.Spec.NodeName, condition)
			if err != nil {
				log.Error(err, "failed to check GPU driver dependency", "node", pod.Spec.NodeName)
				errs = append(errs, err)
			}
			if err := daemonmgr.PatchNodeCondition(ctx, r.APIReader, r.Client, pod.Spec.NodeName, condition); err != nil {
				log.Error(err, "failed to patch node condition", "node", pod.Spec.NodeName)
				errs = append(errs, err)
			}
		}

		if podList.Continue == "" {
			break
		}
		listOpts = []client.ListOption{
			client.InNamespace(namespace),
			client.MatchingLabels(managedByLabels),
			client.Limit(podListPageSize),
			client.Continue(podList.Continue),
		}
	}

	metrics.SetNodeHealth(len(seenNodes)-len(unhealthyNodes), len(unhealthyNodes))
	return len(unhealthyNodes) > 0, errors.Join(errs...)
}

// isPodReady returns true if the pod has condition Ready=True.
// There is a utility in k8s.io/component-helpers, but importing that module
// pulls in the entire k8s.io/kubernetes dependency tree, so we keep this
// trivial 4-line helper inline.
func isPodReady(pod *corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func nodeConditionForPod(ready bool, pod *corev1.Pod) corev1.NodeCondition {
	if ready {
		return corev1.NodeCondition{
			Type:    corev1.NodeConditionType(daemonmgr.NodeConditionType),
			Status:  corev1.ConditionTrue,
			Reason:  daemonmgr.ReasonAllDaemonsReady,
			Message: daemonmgr.MessageAllDaemonsReady,
		}
	}
	reason := podFailureReason(pod)
	daemonName := pod.Labels[daemonmgr.LabelComponent]
	return corev1.NodeCondition{
		Type:    corev1.NodeConditionType(daemonmgr.NodeConditionType),
		Status:  corev1.ConditionFalse,
		Reason:  reason,
		Message: fmt.Sprintf("%s pod is not ready: %s", daemonName, reason),
	}
}

func podFailureReason(pod *corev1.Pod) string {
	// Init containers gate the regular containers, so a stuck init container
	// (e.g. Init:CrashLoopBackOff) is the primary failure signal. Waiting.Reason
	// is optional; an empty one must not leak into the node condition's Reason,
	// which the Kubernetes API requires to be a non-empty identifier.
	for _, statuses := range [][]corev1.ContainerStatus{pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses} {
		for _, cs := range statuses {
			if cs.State.Waiting != nil && cs.State.Waiting.Reason != "" {
				return cs.State.Waiting.Reason
			}
		}
	}

	switch pod.Status.Phase {
	case corev1.PodPending:
		return "Pending"
	case corev1.PodFailed:
		return "Failed"
	default:
		return "NotReady"
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *GpuFractioningConfigReconciler) SetupWithManager(mgr ctrl.Manager) error {
	b := ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.GpuFractioningConfig{}).
		Owns(&appsv1.DaemonSet{})

	var err error
	for _, gvk := range []schema.GroupVersionKind{clusterPolicyGVK, clusterServiceVersionGVK} {
		b, err = watchGpuOperatorDependencyIfAvailable(b, mgr, gvk)
		if err != nil {
			return err
		}
	}

	return b.Named("gpufractioningconfig").Complete(r)
}

func watchGpuOperatorDependencyIfAvailable(b *builder.Builder, mgr ctrl.Manager, gvk schema.GroupVersionKind) (*builder.Builder, error) {
	available, err := gpuOperatorDependencyGVKExists(mgr.GetRESTMapper(), gvk)
	if err != nil {
		return nil, fmt.Errorf("checking optional GPU Operator dependency API %s: %w", gvk.String(), err)
	}
	if !available {
		// These dependency APIs are optional across supported environments. The
		// checker still tolerates their absence on every reconcile; this only avoids
		// starting an informer for a GVK the API server does not currently serve.
		ctrl.Log.WithName("setup").Info("skipping optional GPU Operator dependency watch because API is not installed", "gvk", gvk.String())
		return b, nil
	}
	return b.WatchesRawSource(gpuOperatorDependencySource(mgr, gvk)), nil
}

func gpuOperatorDependencyGVKExists(mapper meta.RESTMapper, gvk schema.GroupVersionKind) (bool, error) {
	_, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err == nil {
		return true, nil
	}
	if meta.IsNoMatchError(err) {
		return false, nil
	}
	return false, err
}

type nonBlockingSyncingSource struct {
	src source.SyncingSource
}

// Start delegates to the underlying kind source but intentionally does not
// expose WaitForSync. The source is registered only when discovery reports the
// optional API during setup; if that API disappears before source startup,
// source.Kind can keep retrying without blocking the primary CR/DaemonSet
// controller.
func (s nonBlockingSyncingSource) Start(ctx context.Context, queue workqueue.TypedRateLimitingInterface[reconcile.Request]) error {
	return s.src.Start(ctx, queue)
}

func (s nonBlockingSyncingSource) String() string {
	if stringer, ok := s.src.(fmt.Stringer); ok {
		return stringer.String()
	}
	return fmt.Sprintf("%T", s.src)
}

func gpuOperatorDependencySource(mgr ctrl.Manager, gvk schema.GroupVersionKind) source.Source {
	obj := &metav1.PartialObjectMetadata{}
	obj.SetGroupVersionKind(gvk)

	return nonBlockingSyncingSource{
		src: source.Kind(mgr.GetCache(), obj, handler.TypedEnqueueRequestsFromMapFunc(func(context.Context, *metav1.PartialObjectMetadata) []reconcile.Request {
			return []reconcile.Request{{NamespacedName: client.ObjectKey{Name: defaultGpuFractioningConfigName}}}
		})),
	}
}

// ReadImageFromEnv reads the <PREFIX>_REPOSITORY, <PREFIX>_TAG, and
// <PREFIX>_PULL_POLICY env vars and returns an ImageSpec.
func ReadImageFromEnv(prefix string) daemonmgr.ImageSpec {
	return daemonmgr.ImageSpec{
		Repository:      env.String(prefix+"_REPOSITORY", ""),
		Tag:             env.String(prefix+"_TAG", ""),
		ImagePullPolicy: env.String(prefix+"_PULL_POLICY", ""),
	}
}

func isConditionTrue(conditions []metav1.Condition, condType string) bool {
	for _, c := range conditions {
		if c.Type == condType {
			return c.Status == metav1.ConditionTrue
		}
	}
	return false
}
