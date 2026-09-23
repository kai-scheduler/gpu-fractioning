// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"strings"
	"testing"

	"golang.org/x/mod/semver"
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
		name              string
		input             metav1.Condition
		objects           []client.Object
		skipVersionChecks bool
		expectedStatus    metav1.ConditionStatus
		expectedReason    string
		expectedMessage   string
	}{
		{
			// The GPU Operator version is not gated where a ClusterPolicy
			// exists. This is issue #120's first false rejection: an operator
			// below the old floor whose operands were overridden onto supported
			// versions is a working install and must roll out.
			name:  "old GPU Operator version with supported operands leaves Ready unchanged",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.3.3", "v1.20.1-ubuntu20.04", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			// A ClusterPolicy labelled only "26.7" normalizes to v26.7.0 and
			// used to be rejected on that alone. With both operands pinned and
			// supported the label is never read, so its shape cannot matter.
			name:  "unparseable GPU Operator version label with pinned operands leaves Ready unchanged",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("26.7", "v1.20.1-ubuntu20.04", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			name:  "missing GPU Operator version label with pinned operands leaves Ready unchanged",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyOperandObject("", "v1.20.1-ubuntu20.04", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			name:  "unready ClusterPolicy leaves true Ready unchanged",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.7.1"}, clusterPolicyStatus("notReady", "False", "False", "operator upgrade in progress")),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			name:  "ready ClusterPolicy leaves false Ready unchanged",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.7.1"}, clusterPolicyStatus("ready", "True", "False", "")),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			// The distro suffix on a toolkit tag is not a semver prerelease:
			// comparing it as one puts v1.20.1-ubuntu20.04 below the v1.20.1
			// floor and rejects a supported install.
			name:  "supported operands with a distro-suffixed toolkit tag leave Ready unchanged",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.1", "v1.20.1-ubuntu20.04", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			// The case the GPU Operator version alone cannot catch: a supported
			// operator running an operand pinned below its floor.
			name:  "supported GPU Operator with an old container toolkit blocks Ready",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.1", "v1.20.0-ubuntu20.04", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperandVersionUnsupported,
			expectedMessage: "container toolkit version v1.20.0-ubuntu20.04",
		},
		{
			name:  "old device plugin blocks Ready",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.1", "v1.20.1-ubuntu20.04", "v0.20.0"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperandVersionUnsupported,
			expectedMessage: "device plugin version v0.20.0",
		},
		{
			// The converse case, and the one that motivated this check:
			// operands overridden onto an operator below the old floor.
			name:  "supported operands on the GPU Operator floor leave Ready unchanged",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.0", "v1.21.0-ubuntu20.04", "v0.21.0"),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			// A digest names an image the GPU Operator version says nothing
			// about, so it is neither compared nor resolved through the floor:
			// an unreadable version is not evidence of an unsupported one. The
			// operator here is below the floor to prove no fallback happens.
			name:  "digest-pinned operand version leaves Ready unchanged on an old GPU Operator",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.0", "sha256-4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			// An operand left unset runs the GPU Operator's own default, so the
			// operator version determines it. On the floor, that default is
			// supported.
			name:  "unset operand version on a supported GPU Operator leaves Ready unchanged",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.1", "", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionTrue,
			expectedReason:  daemonmgr.ReasonAllComponentsReady,
			expectedMessage: daemonmgr.MessageAllComponentsReady,
		},
		{
			// The same default below the floor ships a toolkit that cannot apply
			// the GPU-memory limits. Nothing pinned it, so the operator version
			// is the only evidence there is — and it is evidence.
			name:  "unset operand version on an old GPU Operator blocks Ready",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.0", "", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "default container toolkit version that the ClusterPolicy leaves unset",
		},
		{
			// Both unset: the message names both operands so the reader knows
			// what the operator version is standing in for.
			name:  "all operand versions unset on an old GPU Operator names both",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.0", "", ""),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "container toolkit and device plugin version",
		},
		{
			// A pinned operand below its floor is reported as itself, not
			// deferred to the operator version, even with the other unset.
			name:  "pinned operand below its floor wins over an unset one",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.1", "v1.20.0-ubuntu20.04", ""),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperandVersionUnsupported,
			expectedMessage: "container toolkit version v1.20.0-ubuntu20.04",
		},
		{
			// An unset operand with no operator version to fall back on leaves
			// the effective version unknowable, so it blocks rather than admit a
			// stack that may not enforce GPU memory limits.
			name:  "unset operand version with no GPU Operator version label blocks Ready",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("", "", "v0.20.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonGPUOperatorVersionUnsupported,
			expectedMessage: "is missing",
		},
		{
			// The bypass returns before any version is read, so it covers the
			// operator-floor fallback for an unset operand too, not just the
			// operands compared directly.
			name:  "bypass leaves an unset operand on an old GPU Operator alone",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.0", "", "v0.20.1"),
			},
			skipVersionChecks: true,
			expectedStatus:    metav1.ConditionTrue,
			expectedReason:    daemonmgr.ReasonAllComponentsReady,
			expectedMessage:   daemonmgr.MessageAllComponentsReady,
		},
		{
			name:  "bypass leaves an old container toolkit alone",
			input: trueReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.1", "v1.19.0-ubuntu20.04", "v0.19.0"),
			},
			skipVersionChecks: true,
			expectedStatus:    metav1.ConditionTrue,
			expectedReason:    daemonmgr.ReasonAllComponentsReady,
			expectedMessage:   daemonmgr.MessageAllComponentsReady,
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
			// A CSV carries no operand versions, so the coarse floor is the only
			// gate on this path — and it is the first release whose defaults
			// satisfy both operands, since defaults are all that can be inferred.
			name:  "OpenShift ClusterServiceVersion at the GPU Operator floor leaves Ready unchanged",
			input: falseReady,
			objects: []client.Object{
				clusterServiceVersionObject("gpu-operator-certified.v26.7.1", "26.7.1"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			// The release below the floor ships defaults that do not satisfy the
			// operands. With no ClusterPolicy there is no way to see whether they
			// were overridden, so it is rejected rather than silently admitted.
			name:  "OpenShift ClusterServiceVersion below the GPU Operator floor blocks Ready",
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
			// A ClusterPolicy short-circuits the ClusterServiceVersion lookup
			// entirely, so a CSV below the floor alongside one must not block:
			// where operand versions are readable they are the only gate.
			name:  "ClusterPolicy takes precedence over OpenShift ClusterServiceVersion",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyOperandObject("v26.7.1", "v1.20.1-ubuntu20.04", "v0.20.1"),
				clusterServiceVersionObject("gpu-operator-certified.v26.3.3", "26.3.3"),
			},
			expectedStatus:  metav1.ConditionFalse,
			expectedReason:  daemonmgr.ReasonComponentNotReady,
			expectedMessage: "not ready: FractiondReady",
		},
		{
			// The bypass disables the version gates only; a ClusterPolicy that
			// reports a failure still refines an already-unhealthy Ready.
			name:  "bypass still reports ClusterPolicy readiness failures",
			input: falseReady,
			objects: []client.Object{
				clusterPolicyObject(map[string]string{clusterPolicyVersionLabel: "v26.3.3"}, clusterPolicyStatus("notReady", "False", "True", "operand failed")),
			},
			skipVersionChecks: true,
			expectedStatus:    metav1.ConditionFalse,
			expectedReason:    daemonmgr.ReasonGPUOperatorNotReady,
			expectedMessage:   "operand failed",
		},
		{
			name:  "bypass leaves false Ready unchanged on an unsupported OpenShift ClusterServiceVersion",
			input: falseReady,
			objects: []client.Object{
				clusterServiceVersionObject("gpu-operator-certified.v26.3.3", "26.3.3"),
			},
			skipVersionChecks: true,
			expectedStatus:    metav1.ConditionFalse,
			expectedReason:    daemonmgr.ReasonComponentNotReady,
			expectedMessage:   "not ready: FractiondReady",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := fake.NewClientBuilder().
				WithScheme(clusterPolicyScheme(t)).
				WithObjects(tt.objects...).
				Build()
			checker := NewGpuOperatorDependencyChecker(reader, tt.skipVersionChecks)

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

	got, err := NewGpuOperatorDependencyChecker(missingDependencyAPIReader{}, false).Check(context.Background(), config, ready)
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

// The ClusterServiceVersion is read only to discover a version, so the bypass
// must stop the operator from reading it at all. Otherwise a cluster whose CSV
// list fails for a reason the checker cannot dismiss — RBAC, an unavailable API
// server — still has Ready blocked over a version nothing is going to check,
// which is exactly what the bypass exists to clear.
func TestGpuOperatorDependencyChecker_ClusterServiceVersionListError(t *testing.T) {
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

	tests := []struct {
		name              string
		skipVersionChecks bool
		expectedStatus    metav1.ConditionStatus
		expectedReason    string
	}{
		{
			name:              "version checks on",
			skipVersionChecks: false,
			expectedStatus:    metav1.ConditionFalse,
			expectedReason:    daemonmgr.ReasonGPUOperatorNotReady,
		},
		{
			name:              "version checks bypassed",
			skipVersionChecks: true,
			expectedStatus:    metav1.ConditionTrue,
			expectedReason:    daemonmgr.ReasonAllComponentsReady,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := clusterServiceVersionErrorReader{}
			got, err := NewGpuOperatorDependencyChecker(reader, tt.skipVersionChecks).Check(context.Background(), config, ready)
			if err != nil {
				t.Fatalf("Check returned error: %v", err)
			}
			if got.Status != tt.expectedStatus {
				t.Fatalf("Status = %s, expected %s", got.Status, tt.expectedStatus)
			}
			if got.Reason != tt.expectedReason {
				t.Fatalf("Reason = %q, expected %q", got.Reason, tt.expectedReason)
			}
		})
	}
}

// clusterServiceVersionErrorReader reports no ClusterPolicy and fails the
// ClusterServiceVersion list with an error the checker cannot dismiss.
type clusterServiceVersionErrorReader struct{}

func (clusterServiceVersionErrorReader) Get(context.Context, types.NamespacedName, client.Object, ...client.GetOption) error {
	return apierrors.NewNotFound(schema.GroupResource{Group: "test.kai.scheduler", Resource: "missing"}, "")
}

func (clusterServiceVersionErrorReader) List(_ context.Context, list client.ObjectList, _ ...client.ListOption) error {
	if list.GetObjectKind().GroupVersionKind().Kind == clusterServiceVersionGVK.Kind+"List" {
		return apierrors.NewForbidden(
			schema.GroupResource{Group: clusterServiceVersionGVK.Group, Resource: "clusterserviceversions"},
			"", errors.New("not authorized"))
	}
	return nil
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

func TestNormalizeOperandVersion(t *testing.T) {
	tests := []struct {
		name       string
		version    string
		expected   string
		expectedOK bool
	}{
		{name: "plain version", version: "v1.20.1", expected: "v1.20.1", expectedOK: true},
		{name: "without v prefix", version: "1.20.1", expected: "v1.20.1", expectedOK: true},
		{name: "ubuntu distro suffix", version: "v1.20.1-ubuntu20.04", expected: "v1.20.1", expectedOK: true},
		{name: "ubi distro suffix", version: "v1.20.1-ubi8", expected: "v1.20.1", expectedOK: true},
		{name: "major minor", version: "v1.20", expected: "v1.20.0", expectedOK: true},
		{name: "digest tag", version: "sha256-4f53cda18c2baa0c035", expectedOK: false},
		{name: "floating tag", version: "latest", expectedOK: false},
		{name: "empty", version: "", expectedOK: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := normalizeOperandVersion(tt.version)
			if ok != tt.expectedOK {
				t.Fatalf("ok = %t, expected %t", ok, tt.expectedOK)
			}
			if got != tt.expected {
				t.Fatalf("version = %q, expected %q", got, tt.expected)
			}
		})
	}
}

// Each operand floor must be met by its own distro-suffixed tag. Compared as a
// semver prerelease a suffix orders below the release it qualifies, so the
// minimum supported tag would read as older than the minimum itself.
func TestOperandFloorAcceptsItsOwnDistroSuffixedTag(t *testing.T) {
	for _, operand := range gpuOperandRequirements {
		t.Run(operand.name, func(t *testing.T) {
			tag := operand.minVersion + "-ubuntu20.04"
			version, ok := normalizeOperandVersion(tag)
			if !ok {
				t.Fatalf("normalizeOperandVersion(%q) failed to parse", tag)
			}
			if semver.Compare(version, operand.minVersion) < 0 {
				t.Fatalf("version %q compares below the minimum %q", version, operand.minVersion)
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

// clusterPolicyOperandObject builds a ready ClusterPolicy that also carries the
// operand versions, as a Helm-installed GPU Operator records them. An empty
// operand version is omitted, matching a ClusterPolicy that leaves it to the
// GPU Operator's own default.
func clusterPolicyOperandObject(operatorVersion, toolkitVersion, devicePluginVersion string) *unstructured.Unstructured {
	obj := clusterPolicyObject(
		map[string]string{clusterPolicyVersionLabel: operatorVersion},
		clusterPolicyStatus("ready", "True", "False", ""),
	)

	spec := map[string]any{}
	if toolkitVersion != "" {
		spec["toolkit"] = map[string]any{"version": toolkitVersion}
	}
	if devicePluginVersion != "" {
		spec["devicePlugin"] = map[string]any{"version": devicePluginVersion}
	}
	obj.Object["spec"] = spec

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
