// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	v1alpha1 "github.com/kai-scheduler/kai-gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/common/daemonmgr"
	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/driverinfo"
)

func TestGpuOperatorDependencyChecker(t *testing.T) {
	trueReady := metav1.Condition{
		Type:               daemonmgr.ConditionReady,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: 7,
		Reason:             daemonmgr.ReasonAllComponentsReady,
		Message:            daemonmgr.MessageAllComponentsReady,
	}
	falseReady := metav1.Condition{
		Type:               daemonmgr.ConditionReady,
		Status:             metav1.ConditionFalse,
		ObservedGeneration: 7,
		Reason:             daemonmgr.ReasonComponentNotReady,
		Message:            "not ready: FractiondReady",
	}
	config := &v1alpha1.GpuFractioningConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "default",
			Generation: 7,
		},
	}

	tests := []struct {
		name            string
		input           metav1.Condition
		objects         []client.Object
		expectedStatus  metav1.ConditionStatus
		expectedReason  string
		expectedMessage string
	}{
		{
			name:  "true Ready condition stays true when GPU Operator version is supported",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.7.1"}, clusterPolicyStatus("ready", "True", "False", "")),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			name:  "true Ready condition is blocked when GPU Operator version is unsupported",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.3.3"}, clusterPolicyStatus("ready", "True", "False", "")),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "v26.3.3",
		},
		{
			name:  "unready ClusterPolicy with supported version leaves true Ready unchanged",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.7.1"}, clusterPolicyStatus("notReady", "False", "False", "operator upgrade in progress")),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			name:  "ready ClusterPolicy with supported version leaves false Ready unchanged",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.7.1"}, clusterPolicyStatus("ready", "True", "False", "")),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			name:  "ready ClusterPolicy with unsupported version blocks Ready",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.3.3"}, clusterPolicyStatus("ready", "True", "False", "")),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "v26.3.3",
		},
		{
			// v26.7.0 is the immediately preceding patch release; the supported
			// floor is v26.7.1, so it must be rejected.
			name:  "ready ClusterPolicy one patch below the minimum blocks Ready",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.7.0"}, clusterPolicyStatus("ready", "True", "False", "")),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "v26.7.0",
		},
		{
			name:  "ClusterPolicy Error condition blocks Ready with its message",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.7.1"}, clusterPolicyStatus("notReady", "False", "True", "operand failed")),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorNotReady,
			expectedMessage: "operand failed",
		},
		{
			name:            "true Ready condition stays true when no GPU Operator source is present",
			input:           trueReady,
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			name:            "missing GPU Operator sources leave false Ready unchanged",
			input:           falseReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			name:  "supported OpenShift ClusterServiceVersion leaves false Ready unchanged",
			input: falseReady,
			objects: []client.Object{
				clusterServiceVersionObject("gpu-operator-certified.v26.7.1", "26.7.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			name:  "unsupported OpenShift ClusterServiceVersion blocks Ready",
			input: falseReady,
			objects: []client.Object{
				clusterServiceVersionObject("gpu-operator-certified.v26.3.3", "26.3.3"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "26.3.3",
		},
		{
			// Same boundary as the ClusterPolicy case above, over the OpenShift
			// discovery path: 26.7.0 is one patch below the supported floor.
			name:  "OpenShift ClusterServiceVersion one patch below the minimum blocks Ready",
			input: falseReady,
			objects: []client.Object{
				clusterServiceVersionObject("gpu-operator-certified.v26.7.0", "26.7.0"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "26.7.0",
		},
		{
			name:  "OpenShift ClusterServiceVersion without a version blocks Ready",
			input: falseReady,
			objects: []client.Object{
				clusterServiceVersionObject("gpu-operator-certified.v26.7.1", ""),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "is missing",
		},
		{
			// Only CSVs whose name carries the gpu-operator prefix are considered,
			// so an unrelated operator's CSV must not satisfy or fail the dependency.
			name:  "non-gpu-operator ClusterServiceVersion is ignored",
			input: falseReady,
			objects: []client.Object{
				clusterServiceVersionObject("some-other-operator.v26.7.1", "26.7.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			name:  "ClusterPolicy takes precedence over OpenShift ClusterServiceVersion",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.3.3"}, clusterPolicyStatus("ready", "True", "False", "")),
				clusterServiceVersionObject("gpu-operator-certified.v26.7.1", "26.7.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "v26.3.3",
		},
		{
			name:  "missing version label blocks Ready",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(nil, clusterPolicyStatus("ready", "True", "False", "")),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: clusterPolicyVersionLabel,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := fake.NewClientBuilder().
				WithScheme(clusterPolicyScheme(t)).
				WithObjects(tt.objects...).
				Build()
			checker := NewGpuOperatorDependencyChecker(reader, DefaultMinimumGPUOperatorVersion)

			got, err := checker.Check(context.Background(), config, tt.input)
			if err != nil {
				t.Fatalf("Check returned error: %v", err)
			}
			if got.Status != tt.expectedStatus {
				t.Fatalf("Status = %s, expected %s", got.Status, tt.expectedStatus)
			}
			if got.Reason != tt.expectedReason {
				t.Fatalf("Reason = %q, expected %q", got.Reason, tt.expectedReason)
			}
			if !strings.Contains(got.Message, tt.expectedMessage) {
				t.Fatalf("Message = %q, expected to contain %q", got.Message, tt.expectedMessage)
			}
		})
	}
}

func TestGpuOperatorDependencyChecker_ToleratesMissingDependencyAPIs(t *testing.T) {
	ready := metav1.Condition{
		Type:               daemonmgr.ConditionReady,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: 7,
		Reason:             daemonmgr.ReasonAllComponentsReady,
		Message:            daemonmgr.MessageAllComponentsReady,
	}
	config := &v1alpha1.GpuFractioningConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Generation: 7},
	}

	got, err := NewGpuOperatorDependencyChecker(missingDependencyAPIReader{}, DefaultMinimumGPUOperatorVersion).Check(context.Background(), config, ready)
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}
	if got.Status != metav1.ConditionTrue {
		t.Fatalf("Status = %s, expected True", got.Status)
	}
	if got.Reason != daemonmgr.ReasonAllComponentsReady {
		t.Fatalf("Reason = %q, expected %q", got.Reason, daemonmgr.ReasonAllComponentsReady)
	}
}

type missingDependencyAPIReader struct{}

func (missingDependencyAPIReader) Get(context.Context, types.NamespacedName, client.Object, ...client.GetOption) error {
	return apierrors.NewNotFound(schema.GroupResource{Group: "test.kai.scheduler", Resource: "missing"}, "")
}

func (missingDependencyAPIReader) List(context.Context, client.ObjectList, ...client.ListOption) error {
	return apierrors.NewNotFound(schema.GroupResource{Group: "test.kai.scheduler", Resource: "missing"}, "")
}

func TestGpuDriverDependencyChecker(t *testing.T) {
	tests := []struct {
		name           string
		status         corev1.ConditionStatus
		labels         map[string]string
		missingNode    bool
		initialReason  string
		expectedStatus corev1.ConditionStatus
		expectedReason string
	}{
		{
			name:           "ready condition is not changed even with missing driver label",
			status:         corev1.ConditionTrue,
			labels:         nil,
			initialReason:  daemonmgr.ReasonAllDaemonsReady,
			expectedStatus: corev1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllDaemonsReady,
		},
		{
			name:           "not ready condition is enriched when driver label is missing",
			status:         corev1.ConditionFalse,
			labels:         nil,
			initialReason:  "CrashLoopBackOff",
			expectedStatus: corev1.ConditionFalse,
			expectedReason: daemonmgr.ReasonGPUDriverVersionMissing,
		},
		{
			name:           "not ready condition is enriched when driver version is too old",
			status:         corev1.ConditionFalse,
			labels:         map[string]string{driverinfo.NVIDIADriverMajorLabel: "614"},
			initialReason:  "CrashLoopBackOff",
			expectedStatus: corev1.ConditionFalse,
			expectedReason: daemonmgr.ReasonGPUDriverVersionUnsupported,
		},
		{
			name:           "not ready condition keeps pod failure when driver version is supported",
			status:         corev1.ConditionFalse,
			labels:         map[string]string{driverinfo.NVIDIADriverMajorLabel: "615"},
			initialReason:  "CrashLoopBackOff",
			expectedStatus: corev1.ConditionFalse,
			expectedReason: "CrashLoopBackOff",
		},
		{
			name:           "not ready condition keeps pod failure when node cannot be read",
			status:         corev1.ConditionFalse,
			missingNode:    true,
			initialReason:  "CrashLoopBackOff",
			expectedStatus: corev1.ConditionFalse,
			expectedReason: "CrashLoopBackOff",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			objects := []client.Object{}
			if !tt.missingNode {
				objects = append(objects, &corev1.Node{ObjectMeta: metav1.ObjectMeta{
					Name:   "node-a",
					Labels: tt.labels,
				}})
			}
			checker := NewGpuDriverDependencyChecker(fake.NewClientBuilder().WithObjects(objects...).Build())

			got, err := checker.Check(context.Background(), "node-a", corev1.NodeCondition{
				Type:    corev1.NodeConditionType(daemonmgr.NodeConditionType),
				Status:  tt.status,
				Reason:  tt.initialReason,
				Message: "message",
			})
			if err != nil {
				t.Fatalf("Check returned error: %v", err)
			}
			if got.Status != tt.expectedStatus {
				t.Fatalf("status = %s, expected %s", got.Status, tt.expectedStatus)
			}
			if got.Reason != tt.expectedReason {
				t.Fatalf("reason = %q, expected %q", got.Reason, tt.expectedReason)
			}
		})
	}
}

func TestGpuDriverFailureReason(t *testing.T) {
	tests := []struct {
		name           string
		labels         map[string]string
		expectedFound  bool
		expectedReason string
	}{
		{
			name:           "missing label",
			expectedFound:  true,
			expectedReason: daemonmgr.ReasonGPUDriverVersionMissing,
		},
		{
			name:           "invalid label",
			labels:         map[string]string{driverinfo.NVIDIADriverMajorLabel: "615.34"},
			expectedFound:  true,
			expectedReason: daemonmgr.ReasonGPUDriverVersionInvalid,
		},
		{
			name:           "too old",
			labels:         map[string]string{driverinfo.NVIDIADriverMajorLabel: "614"},
			expectedFound:  true,
			expectedReason: daemonmgr.ReasonGPUDriverVersionUnsupported,
		},
		{
			name:          "minimum supported",
			labels:        map[string]string{driverinfo.NVIDIADriverMajorLabel: "615"},
			expectedFound: false,
		},
		{
			name:          "newer supported",
			labels:        map[string]string{driverinfo.NVIDIADriverMajorLabel: "620"},
			expectedFound: false,
		},
		{
			name:           "GPU Operator label is ignored",
			labels:         map[string]string{"nvidia.com/cuda.driver-version.major": "615"},
			expectedFound:  true,
			expectedReason: daemonmgr.ReasonGPUDriverVersionMissing,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: tt.labels}}
			gotReason, _, gotFound := gpuDriverFailureReason(node)
			if gotFound != tt.expectedFound {
				t.Fatalf("found = %t, expected %t", gotFound, tt.expectedFound)
			}
			if gotReason != tt.expectedReason {
				t.Fatalf("reason = %q, expected %q", gotReason, tt.expectedReason)
			}
		})
	}
}

func TestNormalizeGPUOperatorVersion(t *testing.T) {
	tests := []struct {
		name       string
		version    string
		expected   string
		expectedOK bool
	}{
		{name: "full version", version: "v26.7.1", expected: "v26.7.1", expectedOK: true},
		{name: "without v prefix", version: "26.7.1", expected: "v26.7.1", expectedOK: true},
		{name: "major minor", version: "26.7", expected: "v26.7.0", expectedOK: true},
		{name: "major minor prerelease", version: "v26.7-rc.1", expected: "v26.7.0-rc.1", expectedOK: true},
		{name: "invalid", version: "not-a-version", expectedOK: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := normalizeGPUOperatorVersion(tt.version)
			if ok != tt.expectedOK {
				t.Fatalf("ok = %t, expected %t", ok, tt.expectedOK)
			}
			if got != tt.expected {
				t.Fatalf("version = %q, expected %q", got, tt.expected)
			}
		})
	}
}

func clusterPolicyScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	return newDependencyScheme()
}

func newDependencyScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(clusterPolicyGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(clusterPolicyGVK.GroupVersion().WithKind(clusterPolicyGVK.Kind+"List"), &unstructured.UnstructuredList{})
	scheme.AddKnownTypeWithName(clusterServiceVersionGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(clusterServiceVersionGVK.GroupVersion().WithKind(clusterServiceVersionGVK.Kind+"List"), &unstructured.UnstructuredList{})
	return scheme
}

func clusterPolicyObject(labels map[string]string, status map[string]any) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{Object: map[string]any{}}
	obj.SetGroupVersionKind(clusterPolicyGVK)
	obj.SetName("cluster-policy")
	obj.SetLabels(labels)
	if status != nil {
		obj.Object["status"] = status
	}
	return obj
}

func clusterServiceVersionObject(name, version string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{Object: map[string]any{
		"spec": map[string]any{
			"version": version,
		},
	}}
	obj.SetGroupVersionKind(clusterServiceVersionGVK)
	obj.SetName(name)
	obj.SetNamespace("openshift-operators")
	return obj
}

func clusterPolicyStatus(state, readyStatus, errorStatus, message string) map[string]any {
	return map[string]any{
		"state": state,
		"conditions": []any{
			map[string]any{
				"type":    "Ready",
				"status":  readyStatus,
				"message": message,
			},
			map[string]any{
				"type":    "Error",
				"status":  errorStatus,
				"message": message,
			},
		},
	}
}

// The minimum GPU Operator version is a Helm value, not a compiled-in constant,
// so clusters stuck on an older operator can run with MPS-only enforcement. The
// constructor is the only place that value is interpreted, and the one outcome
// that must never happen is a typo turning the gate off.
func TestNewGpuOperatorDependencyChecker_NormalizesMinimumVersion(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{name: "empty disables the gate", input: "", expected: ""},
		{name: "whitespace only disables the gate", input: "   ", expected: ""},
		{name: "canonical version is kept", input: "v26.7.0", expected: "v26.7.0"},
		{name: "missing v prefix is added", input: "26.7.0", expected: "v26.7.0"},
		{name: "two-part version gains a patch", input: "26.7", expected: "v26.7.0"},
		{name: "surrounding whitespace is trimmed", input: " v26.7.0 ", expected: "v26.7.0"},
		{name: "prerelease is preserved", input: "26.8-rc.1", expected: "v26.8.0-rc.1"},
		// The dangerous case: an unparseable value must NOT collapse to "",
		// which would silently disable the gate on every cluster that has the
		// typo — the opposite of the conservative default.
		{name: "garbage falls back to the default", input: "latest", expected: DefaultMinimumGPUOperatorVersion},
		{name: "empty-ish garbage falls back to the default", input: "v", expected: DefaultMinimumGPUOperatorVersion},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewGpuOperatorDependencyChecker(nil, tt.input).minimumVersion
			if got != tt.expected {
				t.Fatalf("minimumVersion = %q, expected %q", got, tt.expected)
			}
		})
	}
}

func TestGpuOperatorDependencyChecker_ConfigurableMinimumVersion(t *testing.T) {
	trueReady := metav1.Condition{
		Type:               daemonmgr.ConditionReady,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: 7,
		Reason:             daemonmgr.ReasonAllComponentsReady,
		Message:            daemonmgr.MessageAllComponentsReady,
	}
	falseReady := metav1.Condition{
		Type:               daemonmgr.ConditionReady,
		Status:             metav1.ConditionFalse,
		ObservedGeneration: 7,
		Reason:             daemonmgr.ReasonComponentNotReady,
		Message:            "not ready: FractiondReady",
	}
	config := &v1alpha1.GpuFractioningConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Generation: 7},
	}
	readyPolicy := clusterPolicyStatus("ready", "True", "False", "")

	tests := []struct {
		name           string
		minimumVersion string
		labels         map[string]string
		policyStatus   map[string]any
		csv            *unstructured.Unstructured
		input          metav1.Condition
		expectedStatus metav1.ConditionStatus
		expectedReason string
		// messageContains / messageExcludes are checked as substrings of the
		// resulting condition message.
		messageContains []string
		messageExcludes []string
	}{
		// --- gate disabled --------------------------------------------------
		{
			// A cluster far below every supported floor must still pass once the
			// gate is off; otherwise "" would not actually disable anything.
			name:           "empty minimum accepts an ancient version",
			minimumVersion: "",
			labels:         map[string]string{clusterPolicyVersionLabel: "v1.0.0"},
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			name:           "empty minimum accepts a missing version label",
			minimumVersion: "",
			labels:         nil,
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			name:           "empty minimum accepts an unparseable version label",
			minimumVersion: "",
			labels:         map[string]string{clusterPolicyVersionLabel: "nightly-2026-09-17"},
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			// Turning the version gate off must not also turn off the readiness
			// checks: an operand that is actually broken still has to surface.
			name:            "empty minimum still reports a ClusterPolicy error",
			minimumVersion:  "",
			labels:          map[string]string{clusterPolicyVersionLabel: "nightly-2026-09-17"},
			policyStatus:    clusterPolicyStatus("notReady", "False", "True", "operand failed"),
			input:           falseReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorNotReady,
			messageContains: []string{"operand failed"},
		},
		{
			name:            "empty minimum still reports an unready ClusterPolicy state",
			minimumVersion:  "",
			labels:          nil,
			policyStatus:    clusterPolicyStatus("notReady", "False", "False", ""),
			input:           falseReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorNotReady,
			messageContains: []string{"Ready condition is False"},
		},
		{
			name:           "empty minimum accepts a garbage OpenShift CSV version",
			minimumVersion: "",
			csv:            clusterServiceVersionObject("gpu-operator-certified.vmain", "main"),
			input:          falseReady,
			expectedStatus: metav1.ConditionFalse,
			expectedReason: daemonmgr.ReasonComponentNotReady,
		},

		// --- lowered gate ---------------------------------------------------
		{
			name:           "configured minimum accepts its own version",
			minimumVersion: "v26.7.0",
			labels:         map[string]string{clusterPolicyVersionLabel: "v26.7.0"},
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			// The message has to name the configured minimum. Printing the
			// compiled-in default instead sends an operator chasing a version
			// their cluster was never gated on.
			name:            "configured minimum rejects one patch below and names itself",
			minimumVersion:  "v26.7.0",
			labels:          map[string]string{clusterPolicyVersionLabel: "v26.6.9"},
			policyStatus:    readyPolicy,
			input:           trueReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			messageContains: []string{"v26.6.9", "v26.7.0"},
			messageExcludes: []string{DefaultMinimumGPUOperatorVersion},
		},
		{
			name:           "bare minimum without a v prefix accepts its own version",
			minimumVersion: "26.7.0",
			labels:         map[string]string{clusterPolicyVersionLabel: "v26.7.0"},
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			name:            "bare minimum without a v prefix still rejects below it",
			minimumVersion:  "26.7.0",
			labels:          map[string]string{clusterPolicyVersionLabel: "v26.6.9"},
			policyStatus:    readyPolicy,
			input:           trueReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			messageContains: []string{"v26.7.0"},
		},
		{
			// "26.7" is how a chart value is usually written. It must mean
			// v26.7.0, not v26.7.1 (the default) and not be rejected outright.
			name:           "two-part minimum accepts the .0 patch",
			minimumVersion: "26.7",
			labels:         map[string]string{clusterPolicyVersionLabel: "v26.7.0"},
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			name:            "two-part minimum rejects the previous minor",
			minimumVersion:  "26.7",
			labels:          map[string]string{clusterPolicyVersionLabel: "v26.6.9"},
			policyStatus:    readyPolicy,
			input:           trueReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			messageContains: []string{"v26.7.0"},
		},
		{
			name:           "whitespace-padded minimum is trimmed, not treated as garbage",
			minimumVersion: " v26.7.0 ",
			labels:         map[string]string{clusterPolicyVersionLabel: "v26.7.0"},
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			name:           "lowered minimum also applies on the OpenShift CSV path",
			minimumVersion: "26.3",
			csv:            clusterServiceVersionObject("gpu-operator-certified.v26.3.3", "26.3.3"),
			input:          falseReady,
			expectedStatus: metav1.ConditionFalse,
			expectedReason: daemonmgr.ReasonComponentNotReady,
		},

		// --- garbage minimum falls back, it does not disable -----------------
		{
			// The most dangerous regression: a mistyped chart value silently
			// disabling the gate would let an unsupported GPU Operator through
			// with no enforcement of the CUDA memory limit at all.
			name:            "garbage minimum falls back to the default and still rejects",
			minimumVersion:  "latest",
			labels:          map[string]string{clusterPolicyVersionLabel: "v26.7.0"},
			policyStatus:    readyPolicy,
			input:           trueReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			messageContains: []string{DefaultMinimumGPUOperatorVersion},
		},
		{
			name:           "garbage minimum falls back to the default and accepts it",
			minimumVersion: "latest",
			labels:         map[string]string{clusterPolicyVersionLabel: DefaultMinimumGPUOperatorVersion},
			policyStatus:   readyPolicy,
			input:          trueReady,
			expectedStatus: metav1.ConditionTrue,
			expectedReason: daemonmgr.ReasonAllComponentsReady,
		},
		{
			name:            "garbage minimum still rejects a missing version label",
			minimumVersion:  "latest",
			labels:          nil,
			policyStatus:    readyPolicy,
			input:           trueReady,
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			messageContains: []string{"is missing", DefaultMinimumGPUOperatorVersion},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objects []client.Object
			if tt.policyStatus != nil || tt.labels != nil {
				objects = append(objects, clusterPolicyObject(tt.labels, tt.policyStatus))
			}
			if tt.csv != nil {
				objects = append(objects, tt.csv)
			}

			reader := fake.NewClientBuilder().
				WithScheme(clusterPolicyScheme(t)).
				WithObjects(objects...).
				Build()

			got, err := NewGpuOperatorDependencyChecker(reader, tt.minimumVersion).
				Check(context.Background(), config, tt.input)
			if err != nil {
				t.Fatalf("Check returned error: %v", err)
			}
			if got.Status != tt.expectedStatus {
				t.Fatalf("Status = %s, expected %s (message %q)", got.Status, tt.expectedStatus, got.Message)
			}
			if got.Reason != tt.expectedReason {
				t.Fatalf("Reason = %q, expected %q (message %q)", got.Reason, tt.expectedReason, got.Message)
			}
			for _, want := range tt.messageContains {
				if !strings.Contains(got.Message, want) {
					t.Errorf("Message = %q, expected to contain %q", got.Message, want)
				}
			}
			for _, unwanted := range tt.messageExcludes {
				if strings.Contains(got.Message, unwanted) {
					t.Errorf("Message = %q, expected NOT to contain %q; the failure must name the configured minimum, not the compiled-in default", got.Message, unwanted)
				}
			}
		})
	}
}
