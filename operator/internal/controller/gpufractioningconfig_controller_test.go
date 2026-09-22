// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	gpufractioningv1alpha1 "github.com/kai-scheduler/kai-gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/common/daemonmgr"
	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/driverinfo"
)

func TestPodFailureReason(t *testing.T) {
	tests := []struct {
		name     string
		pod      corev1.Pod
		expected string
	}{
		{
			name: "regular container waiting reason",
			pod: corev1.Pod{Status: corev1.PodStatus{
				ContainerStatuses: []corev1.ContainerStatus{waitingStatus("CrashLoopBackOff")},
			}},
			expected: "CrashLoopBackOff",
		},
		{
			name: "stuck init container takes precedence",
			pod: corev1.Pod{Status: corev1.PodStatus{
				InitContainerStatuses: []corev1.ContainerStatus{waitingStatus("ImagePullBackOff")},
				ContainerStatuses:     []corev1.ContainerStatus{waitingStatus("PodInitializing")},
			}},
			expected: "ImagePullBackOff",
		},
		{
			name: "empty waiting reason falls through to phase",
			pod: corev1.Pod{Status: corev1.PodStatus{
				ContainerStatuses: []corev1.ContainerStatus{waitingStatus("")},
				Phase:             corev1.PodPending,
			}},
			expected: "Pending",
		},
		{
			name:     "failed phase",
			pod:      corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodFailed}},
			expected: "Failed",
		},
		{
			name:     "running but not ready",
			pod:      corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning}},
			expected: "NotReady",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := podFailureReason(&tt.pod); got != tt.expected {
				t.Errorf("podFailureReason() = %q, expected %q", got, tt.expected)
			}
		})
	}
}

func waitingStatus(reason string) corev1.ContainerStatus {
	return corev1.ContainerStatus{
		State: corev1.ContainerState{
			Waiting: &corev1.ContainerStateWaiting{Reason: reason},
		},
	}
}

func TestGpuOperatorDependencyGVKExists(t *testing.T) {
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{clusterPolicyGVK.GroupVersion()})
	mapper.Add(clusterPolicyGVK, meta.RESTScopeRoot)

	tests := []struct {
		name     string
		gvk      schema.GroupVersionKind
		expected bool
	}{
		{
			name:     "installed API is watchable",
			gvk:      clusterPolicyGVK,
			expected: true,
		},
		{
			name:     "absent optional API is skipped",
			gvk:      clusterServiceVersionGVK,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := gpuOperatorDependencyGVKExists(mapper, tt.gvk)
			if err != nil {
				t.Fatalf("gpuOperatorDependencyGVKExists returned error: %v", err)
			}
			if got != tt.expected {
				t.Fatalf("gpuOperatorDependencyGVKExists(%s) = %t, expected %t", tt.gvk, got, tt.expected)
			}
		})
	}
}

func gpuOperatorCheckerWithClusterPolicyVersion(version string) GpuOperatorDependencyChecker {
	return newGpuOperatorChecker(version, false)
}

func newGpuOperatorChecker(version string, skipVersionChecks bool) GpuOperatorDependencyChecker {
	return NewGpuOperatorDependencyChecker(fake.NewClientBuilder().
		WithScheme(newDependencyScheme()).
		WithObjects(clusterPolicyObject(
			map[string]string{clusterPolicyVersionLabel: version},
			clusterPolicyStatus("ready", "True", "False", ""),
		)).
		Build(), skipVersionChecks)
}

// Fixed args passed to every NewGpuFractioningConfigReconciler call below;
// named here purely so call sites read as words instead of bare "true, true".
const (
	testMpsdAuditLogTrue     = true
	testSupportSMSharingTrue = true
	testFIPSOnlyDisabled     = false

	// testSkipGpuStackVersionChecksDisabled keeps the GPU stack version gates on,
	// which is the default and what every reconciler test below exercises.
	testSkipGpuStackVersionChecksDisabled = false
)

var _ = Describe("GpuFractioningConfig Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "default"

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name: resourceName,
		}

		defaultImages := map[string]daemonmgr.ImageSpec{
			"fractiond": {Repository: "example.com/fractiond", Tag: "test"},
			"mpsd":      {Repository: "example.com/mpsd", Tag: "test"},
		}

		BeforeEach(func() {
			By("creating the custom resource for the Kind GpuFractioningConfig")
			resource := &gpufractioningv1alpha1.GpuFractioningConfig{
				ObjectMeta: metav1.ObjectMeta{
					Name: resourceName,
				},
				Spec: gpufractioningv1alpha1.GpuFractioningConfigSpec{
					NodeSelector: map[string]string{"nvidia.com/gpu.present": "true"},
				},
			}
			Expect(k8sClient.Create(ctx, resource)).To(Succeed())
		})

		AfterEach(func() {
			resource := &gpufractioningv1alpha1.GpuFractioningConfig{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance GpuFractioningConfig")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())

			// Reconciling registers the node-condition cleanup finalizer, so
			// deletion only completes after another reconcile releases it.
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)
			_, err = controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() bool {
				return apierrors.IsNotFound(k8sClient.Get(ctx, typeNamespacedName, &gpufractioningv1alpha1.GpuFractioningConfig{}))
			}).Should(BeTrue())
		})

		It("should reject changing nodeSelector on a live resource (immutable)", func() {
			resource := &gpufractioningv1alpha1.GpuFractioningConfig{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, resource)).To(Succeed())

			resource.Spec.NodeSelector = map[string]string{"nvidia.com/gpu.present": "false"}
			err := k8sClient.Update(ctx, resource)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("nodeSelector is immutable"))
		})

		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
		})

		It("should check dependencies and not requeue after a healthy reconcile with a supported GPU Operator", func() {
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)
			controllerReconciler.GpuOperatorChecker = gpuOperatorCheckerWithClusterPolicyVersion("v26.7.1")

			result, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(reconcile.Result{}))

			var updated gpufractioningv1alpha1.GpuFractioningConfig
			Expect(k8sClient.Get(ctx, typeNamespacedName, &updated)).To(Succeed())
			ready := meta.FindStatusCondition(updated.Status.Conditions, daemonmgr.ConditionReady)
			Expect(ready).NotTo(BeNil())
			Expect(ready.Status).To(Equal(metav1.ConditionTrue))
			Expect(ready.Reason).To(Equal(daemonmgr.ReasonAllComponentsReady))
		})

		It("should mark Ready false when the GPU Operator is downgraded while daemons stay healthy", func() {
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)
			controllerReconciler.GpuOperatorChecker = gpuOperatorCheckerWithClusterPolicyVersion("v26.7.1")

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			controllerReconciler.GpuOperatorChecker = gpuOperatorCheckerWithClusterPolicyVersion("v26.7.0")
			result, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(reconcile.Result{}))

			var updated gpufractioningv1alpha1.GpuFractioningConfig
			Expect(k8sClient.Get(ctx, typeNamespacedName, &updated)).To(Succeed())
			ready := meta.FindStatusCondition(updated.Status.Conditions, daemonmgr.ConditionReady)
			Expect(ready).NotTo(BeNil())
			Expect(ready.Status).To(Equal(metav1.ConditionFalse))
			Expect(ready.Reason).To(Equal(daemonmgr.ReasonGPUOperatorVersionUnsupported))
			Expect(ready.Message).To(ContainSubstring("v26.7.0"))
		})

		It("should keep Ready true on an unsupported GPU Operator version when the version checks are bypassed", func() {
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				true,
			)
			// The bypass reaches the checker the reconciler builds for itself,
			// not just the one this test substitutes below.
			Expect(controllerReconciler.GpuOperatorChecker.skipVersionChecks).To(BeTrue())
			controllerReconciler.GpuOperatorChecker = newGpuOperatorChecker("v26.7.0", true)

			result, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(reconcile.Result{}))

			var updated gpufractioningv1alpha1.GpuFractioningConfig
			Expect(k8sClient.Get(ctx, typeNamespacedName, &updated)).To(Succeed())
			ready := meta.FindStatusCondition(updated.Status.Conditions, daemonmgr.ConditionReady)
			Expect(ready).NotTo(BeNil())
			Expect(ready.Status).To(Equal(metav1.ConditionTrue))
			Expect(ready.Reason).To(Equal(daemonmgr.ReasonAllComponentsReady))
		})

		It("should tolerate missing GPU Operator sources and retry sooner when daemon readiness is false", func() {
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", nil, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)

			result, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(result.RequeueAfter).To(Equal(requeueInterval))

			var updated gpufractioningv1alpha1.GpuFractioningConfig
			Expect(k8sClient.Get(ctx, typeNamespacedName, &updated)).To(Succeed())
			ready := meta.FindStatusCondition(updated.Status.Conditions, daemonmgr.ConditionReady)
			Expect(ready).NotTo(BeNil())
			Expect(ready.Status).To(Equal(metav1.ConditionFalse))
			Expect(ready.Reason).To(Equal(daemonmgr.ReasonComponentNotReady))
			Expect(ready.Message).To(ContainSubstring("FractiondReady"))
		})

		It("should update observedGeneration on reconcile", func() {
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			var updated gpufractioningv1alpha1.GpuFractioningConfig
			Expect(k8sClient.Get(ctx, typeNamespacedName, &updated)).To(Succeed())
			Expect(updated.Status.ObservedGeneration).To(Equal(updated.Generation))
		})

		It("should handle not-found resources gracefully", func() {
			controllerReconciler := NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(10),
				"default", defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      "nonexistent",
					Namespace: "default",
				},
			})
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("When deleting a resource", func() {
		// The CRD's CEL rule enforces a singleton CR named "default".
		const (
			resourceName = "default"
			namespace    = "gpu-fractioning-delete-test"
		)

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{Name: resourceName}
		selectorLabels := map[string]string{"gpu-fractioning-test/target": "true"}

		var (
			reconciler   *GpuFractioningConfigReconciler
			createdNodes []string
		)

		// gpuFractioningConditionOf returns the node's gpu-fractioning.nvidia.com/Ready
		// condition, or nil if the node does not carry it.
		gpuFractioningConditionOf := func(nodeName string) *corev1.NodeCondition {
			node := &corev1.Node{}
			ExpectWithOffset(1, k8sClient.Get(ctx, types.NamespacedName{Name: nodeName}, node)).To(Succeed())
			for i := range node.Status.Conditions {
				if string(node.Status.Conditions[i].Type) == daemonmgr.NodeConditionType {
					return &node.Status.Conditions[i]
				}
			}
			return nil
		}

		createNode := func(name string, labels map[string]string, withCondition bool) {
			nodeLabels := make(map[string]string, len(labels))
			for key, value := range labels {
				nodeLabels[key] = value
			}
			node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: nodeLabels}}
			ExpectWithOffset(1, k8sClient.Create(ctx, node)).To(Succeed())
			createdNodes = append(createdNodes, name)
			if withCondition {
				node.Status.Conditions = append(node.Status.Conditions,
					corev1.NodeCondition{
						Type:   corev1.NodeConditionType(daemonmgr.NodeConditionType),
						Status: corev1.ConditionTrue,
						Reason: "AllDaemonsReady",
					},
					corev1.NodeCondition{
						Type:   corev1.NodeReady,
						Status: corev1.ConditionTrue,
						Reason: "KubeletReady",
					},
				)
				ExpectWithOffset(1, k8sClient.Status().Update(ctx, node)).To(Succeed())
			}
		}

		createConfig := func(name string, selector map[string]string) {
			resource := &gpufractioningv1alpha1.GpuFractioningConfig{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec:       gpufractioningv1alpha1.GpuFractioningConfigSpec{NodeSelector: selector},
			}
			ExpectWithOffset(1, k8sClient.Create(ctx, resource)).To(Succeed())
		}

		reconcileName := func(name string) error {
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{Name: name},
			})
			return err
		}

		// deleteConfigAndFinalize deletes the CR and reconciles once so the
		// finalizer cleanup runs, then waits for the CR to be fully gone.
		deleteConfigAndFinalize := func(name string) {
			key := types.NamespacedName{Name: name}
			resource := &gpufractioningv1alpha1.GpuFractioningConfig{}
			ExpectWithOffset(1, k8sClient.Get(ctx, key, resource)).To(Succeed())
			ExpectWithOffset(1, k8sClient.Delete(ctx, resource)).To(Succeed())
			ExpectWithOffset(1, reconcileName(name)).To(Succeed())
			EventuallyWithOffset(1, func() bool {
				return apierrors.IsNotFound(k8sClient.Get(ctx, key, &gpufractioningv1alpha1.GpuFractioningConfig{}))
			}).Should(BeTrue())
		}

		BeforeEach(func() {
			// Namespaces cannot be deleted in envtest (they hang in
			// Terminating), so create it once and tolerate AlreadyExists.
			ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
			if err := k8sClient.Create(ctx, ns); err != nil {
				Expect(apierrors.IsAlreadyExists(err)).To(BeTrue())
			}

			// Default images are required for the DaemonSets to pass API
			// validation (an empty image is rejected).
			defaultImages := map[string]daemonmgr.ImageSpec{
				"fractiond": {Repository: "example.com/fractiond", Tag: "test"},
				"mpsd":      {Repository: "example.com/mpsd", Tag: "test"},
			}
			reconciler = NewGpuFractioningConfigReconciler(
				k8sClient, k8sClient, k8sClient.Scheme(), record.NewFakeRecorder(20),
				namespace, defaultImages, "gpu-fractioning-daemon", testMpsdAuditLogTrue, testSupportSMSharingTrue, testFIPSOnlyDisabled,
				testSkipGpuStackVersionChecksDisabled,
			)
			createdNodes = nil
		})

		AfterEach(func() {
			var configs gpufractioningv1alpha1.GpuFractioningConfigList
			Expect(k8sClient.List(ctx, &configs)).To(Succeed())
			for i := range configs.Items {
				Expect(client.IgnoreNotFound(k8sClient.Delete(ctx, &configs.Items[i]))).To(Succeed())
				Expect(reconcileName(configs.Items[i].Name)).To(Succeed())
			}
			Eventually(func() int {
				Expect(k8sClient.List(ctx, &configs)).To(Succeed())
				return len(configs.Items)
			}).Should(BeZero())

			for _, name := range createdNodes {
				node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}
				Expect(client.IgnoreNotFound(k8sClient.Delete(ctx, node))).To(Succeed())
			}

			// envtest has no garbage collector, so owned DaemonSets must be
			// removed by hand to keep tests independent.
			Expect(k8sClient.DeleteAllOf(ctx, &appsv1.DaemonSet{}, client.InNamespace(namespace))).To(Succeed())
		})

		It("removes node conditions and completes deletion", func() {
			createNode("g4-node-a", map[string]string{
				"gpu-fractioning-test/target":     "true",
				driverinfo.NVIDIADriverMajorLabel: "615",
			}, true)
			createNode("g4-node-b", map[string]string{
				"gpu-fractioning-test/target":     "true",
				driverinfo.NVIDIADriverMajorLabel: "615",
			}, true)
			// Outside the selector: cleanup is selector-scoped, so this node
			// must not be touched even though it carries the condition.
			createNode("g4-node-c", map[string]string{
				"gpu-fractioning-test/other":      "true",
				driverinfo.NVIDIADriverMajorLabel: "615",
			}, true)

			createConfig(resourceName, selectorLabels)
			Expect(reconcileName(resourceName)).To(Succeed())

			By("registering the cleanup finalizer on first reconcile")
			resource := &gpufractioningv1alpha1.GpuFractioningConfig{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, resource)).To(Succeed())
			Expect(resource.Finalizers).To(ContainElement(NodeConditionCleanupFinalizer))

			By("owning the DaemonSets so garbage collection still applies")
			var dsList appsv1.DaemonSetList
			Expect(k8sClient.List(ctx, &dsList, client.InNamespace(namespace))).To(Succeed())
			Expect(dsList.Items).NotTo(BeEmpty())
			for _, ds := range dsList.Items {
				Expect(ds.OwnerReferences).To(HaveLen(1))
				Expect(ds.OwnerReferences[0].UID).To(Equal(resource.UID))
			}

			By("cleaning targeted nodes when the CR is deleted")
			deleteConfigAndFinalize(resourceName)
			Expect(gpuFractioningConditionOf("g4-node-a")).To(BeNil())
			Expect(gpuFractioningConditionOf("g4-node-b")).To(BeNil())
			Expect(gpuFractioningConditionOf("g4-node-c")).NotTo(BeNil())

			By("removing targeted driver labels")
			node := &corev1.Node{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "g4-node-a"}, node)).To(Succeed())
			Expect(node.Labels).NotTo(HaveKey(driverinfo.NVIDIADriverMajorLabel))
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "g4-node-b"}, node)).To(Succeed())
			Expect(node.Labels).NotTo(HaveKey(driverinfo.NVIDIADriverMajorLabel))
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "g4-node-c"}, node)).To(Succeed())
			Expect(node.Labels).To(HaveKeyWithValue(driverinfo.NVIDIADriverMajorLabel, "615"))

			By("preserving unrelated node conditions")
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "g4-node-a"}, node)).To(Succeed())
			Expect(node.Status.Conditions).To(HaveLen(1))
			Expect(node.Status.Conditions[0].Type).To(Equal(corev1.NodeReady))
		})

		It("leaves no stale condition on delete + recreate with a different nodeSelector", func() {
			oldSelector := map[string]string{"gpu-fractioning-test/pool": "a"}
			newSelector := map[string]string{"gpu-fractioning-test/pool": "b"}
			createNode("g9-node-a", oldSelector, true)
			createNode("g9-node-b", newSelector, false)

			createConfig(resourceName, oldSelector)
			Expect(reconcileName(resourceName)).To(Succeed())
			deleteConfigAndFinalize(resourceName)

			// Simulate the garbage collection envtest lacks before recreating.
			Expect(k8sClient.DeleteAllOf(ctx, &appsv1.DaemonSet{}, client.InNamespace(namespace))).To(Succeed())

			createConfig(resourceName, newSelector)
			Expect(reconcileName(resourceName)).To(Succeed())

			Expect(gpuFractioningConditionOf("g9-node-a")).To(BeNil())
			Expect(gpuFractioningConditionOf("g9-node-b")).To(BeNil())
		})

		It("completes deletion when no node carries the condition", func() {
			createNode("idem-node", selectorLabels, false)

			createConfig(resourceName, selectorLabels)
			Expect(reconcileName(resourceName)).To(Succeed())
			deleteConfigAndFinalize(resourceName)

			// A further reconcile after full deletion is a no-op.
			Expect(reconcileName(resourceName)).To(Succeed())
		})
	})
})
