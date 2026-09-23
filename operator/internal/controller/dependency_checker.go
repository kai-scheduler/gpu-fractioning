// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"strings"
	"time"

	"golang.org/x/mod/semver"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/common/daemonmgr"
	"github.com/kai-scheduler/gpu-fractioning/pkg/driverinfo"
)

const (
	clusterPolicyVersionLabel = "app.kubernetes.io/version"
	// minimumGPUOperatorVersion is the floor for the GPU Operator's own version,
	// used only where the operand versions below cannot be verified directly: no
	// ClusterPolicy, or one that leaves an operand unset. It is the first release
	// whose defaults satisfy both operands.
	minimumGPUOperatorVersion   = "v26.7.1"
	clusterPolicyReadyState     = "ready"
	clusterPolicyReadyCondition = "Ready"
	clusterPolicyErrorCondition = "Error"
	clusterServiceVersionPrefix = "gpu-operator"

	// minGPUDriverMajor is the minimum NVIDIA driver major version that supports
	// the MPS memory and compute limit behavior gpu-fractioning depends on.
	minGPUDriverMajor = 615
)

// gpuOperandRequirement is a GPU Operator operand whose version
// gpu-fractioning depends on, and where to read it from the ClusterPolicy.
type gpuOperandRequirement struct {
	name       string
	specPath   []string
	minVersion string
}

var gpuOperandRequirements = []gpuOperandRequirement{
	{
		// The container toolkit carries the apply-cuda-memory-limits CDI hook
		// that applies the NVIDIA_GPU_MEMORY_REQUEST/NVIDIA_GPU_MEMORY_LIMIT
		// values fractiond injects. On an older toolkit the limits are injected
		// and nothing acts on them, so GPU memory goes unfenced silently.
		name:       "container toolkit",
		specPath:   []string{"spec", "toolkit", "version"},
		minVersion: "v1.20.1",
	},
	{
		name:       "device plugin",
		specPath:   []string{"spec", "devicePlugin", "version"},
		minVersion: "v0.20.1",
	},
}

var clusterPolicyGVK = schema.GroupVersionKind{
	Group:   "nvidia.com",
	Version: "v1",
	Kind:    "ClusterPolicy",
}

var clusterServiceVersionGVK = schema.GroupVersionKind{
	Group:   "operators.coreos.com",
	Version: "v1alpha1",
	Kind:    "ClusterServiceVersion",
}

// GpuOperatorDependencyChecker checks the NVIDIA GPU Operator ClusterPolicy.
type GpuOperatorDependencyChecker struct {
	reader client.Reader

	// skipVersionChecks is the installation-time bypass (Helm value
	// skipGpuStackVersionChecks -> SKIP_GPU_STACK_VERSION_CHECKS). When true,
	// the GPU stack version gates are not evaluated at all, so an install whose
	// versions the operator reads wrongly can be unblocked without a code
	// rollback — the same revert-lever pattern as supportSmSharing. It disables
	// verification only: it does not make an unsupported GPU stack work.
	// ClusterPolicy readiness is still checked.
	skipVersionChecks bool
}

func NewGpuOperatorDependencyChecker(reader client.Reader, skipVersionChecks bool) GpuOperatorDependencyChecker {
	return GpuOperatorDependencyChecker{reader: reader, skipVersionChecks: skipVersionChecks}
}

func (c GpuOperatorDependencyChecker) Check(ctx context.Context, config *v1alpha1.GpuFractioningConfig, ready metav1.Condition) (metav1.Condition, error) {
	clusterPolicy, err := c.clusterPolicy(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return ready, err
		}
		return gpuOperatorReadyCondition(config.Generation, daemonmgr.ReasonGPUOperatorNotReady,
			fmt.Sprintf("unable to read NVIDIA GPU Operator ClusterPolicy: %v", err)), nil
	}
	reason, message, err := c.versionFailure(ctx, clusterPolicy)
	if err != nil {
		return ready, err
	}
	if reason != "" {
		return gpuOperatorReadyCondition(config.Generation, reason, message), nil
	}

	if clusterPolicy == nil {
		// The OLM ClusterServiceVersion carries no operand or readiness state,
		// so the version gate above is all this path can check. Treat the GPU
		// Operator dependency as not the cause of the current
		// GpuFractioningConfig failure and leave Ready unchanged.
		return ready, nil
	}

	if ready.Status == metav1.ConditionFalse {
		if msg := clusterPolicyReadinessFailureMessage(clusterPolicy); msg != "" {
			return gpuOperatorReadyCondition(config.Generation, daemonmgr.ReasonGPUOperatorNotReady, msg), nil
		}
	}

	return ready, nil
}

// versionFailure evaluates the GPU stack version gates and returns the Ready
// condition reason and message for a version that does not meet its minimum, or
// empty strings when they do, when no version source is present, or when the
// gates are bypassed.
//
// Where a ClusterPolicy exists the pinned operand versions are gated directly:
// they are the dependency gpu-fractioning actually has. The GPU Operator version
// is deliberately not checked against them — it is wrong in both directions,
// rejecting an older operator whose operands were overridden onto supported
// versions, and admitting a supported operator whose operands are pinned below
// their floor. It is consulted only for operands the ClusterPolicy leaves unset,
// whose effective version is the operator's own build default.
//
// This is the one place skipVersionChecks is honoured. Reading a version is all
// this function does, so bypassing the gate here means no version is read at
// all — including the ClusterServiceVersion list, whose failure would otherwise
// block Ready over a version nothing is going to check.
//
// clusterPolicy may be nil: OpenShift installations may not expose the NVIDIA
// ClusterPolicy CR, and the OLM ClusterServiceVersion is then the only source
// of the installed GPU Operator version.
func (c GpuOperatorDependencyChecker) versionFailure(ctx context.Context, clusterPolicy *unstructured.Unstructured) (reason, message string, err error) {
	if c.skipVersionChecks {
		return "", "", nil
	}

	if clusterPolicy != nil {
		msg, defaulted := gpuOperandVersionFailureMessage(clusterPolicy)
		if msg != "" {
			return daemonmgr.ReasonGPUOperandVersionUnsupported, msg, nil
		}
		// An operand left off the ClusterPolicy runs the GPU Operator's own
		// default, so the operator version is what determines it — the one case
		// on this path where that version is evidence of anything.
		if len(defaulted) > 0 {
			version := gpuOperatorVersionFromClusterPolicy(clusterPolicy)
			if msg := gpuOperatorVersionFailureMessage(version, fmt.Sprintf("ClusterPolicy label %q", clusterPolicyVersionLabel)); msg != "" {
				return daemonmgr.ReasonGPUOperatorVersionUnsupported,
					fmt.Sprintf("%s, and it supplies the default %s version that the ClusterPolicy leaves unset",
						msg, strings.Join(defaulted, " and ")), nil
			}
		}
		return "", "", nil
	}

	version, found, err := c.clusterServiceVersionGPUOperatorVersion(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return "", "", err
		}
		return daemonmgr.ReasonGPUOperatorNotReady,
			fmt.Sprintf("unable to discover NVIDIA GPU Operator version from ClusterPolicy or ClusterServiceVersion: %v", err), nil
	}
	if !found {
		return "", "", nil
	}
	if msg := gpuOperatorVersionFailureMessage(version, "ClusterServiceVersion"); msg != "" {
		return daemonmgr.ReasonGPUOperatorVersionUnsupported, msg, nil
	}
	return "", "", nil
}

// GpuDriverDependencyChecker checks the gpu-fractioning-owned NVIDIA driver
// label on a node. It is used only to refine already-unhealthy node conditions
// with a clearer reason.
type GpuDriverDependencyChecker struct {
	reader client.Reader
}

func NewGpuDriverDependencyChecker(reader client.Reader) GpuDriverDependencyChecker {
	return GpuDriverDependencyChecker{reader: reader}
}

func (c GpuDriverDependencyChecker) Check(ctx context.Context, nodeName string, condition corev1.NodeCondition) (corev1.NodeCondition, error) {
	if condition.Status != corev1.ConditionFalse {
		return condition, nil
	}

	node := &corev1.Node{}
	if err := c.reader.Get(ctx, types.NamespacedName{Name: nodeName}, node); err != nil {
		if ctx.Err() != nil {
			return condition, err
		}
		return condition, nil
	}

	if driverReason, driverMessage, found := gpuDriverFailureReason(node); found {
		condition.Reason = driverReason
		condition.Message = driverMessage
	}
	return condition, nil
}

func (c GpuOperatorDependencyChecker) clusterPolicy(ctx context.Context) (*unstructured.Unstructured, error) {
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(clusterPolicyGVK.GroupVersion().WithKind(clusterPolicyGVK.Kind + "List"))

	if err := c.reader.List(ctx, list); err != nil {
		if meta.IsNoMatchError(err) || apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}

	if len(list.Items) == 0 {
		return nil, nil
	}
	return &list.Items[0], nil
}

func (c GpuOperatorDependencyChecker) clusterServiceVersionGPUOperatorVersion(ctx context.Context) (version string, found bool, err error) {
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(clusterServiceVersionGVK.GroupVersion().WithKind(clusterServiceVersionGVK.Kind + "List"))

	if err := c.reader.List(ctx, list); err != nil {
		if meta.IsNoMatchError(err) || apierrors.IsNotFound(err) {
			return "", false, nil
		}
		return "", false, fmt.Errorf("failed to list ClusterServiceVersions: %w", err)
	}

	for _, csv := range list.Items {
		if strings.HasPrefix(csv.GetName(), clusterServiceVersionPrefix) {
			version, _, _ := unstructured.NestedString(csv.Object, "spec", "version")
			return strings.TrimSpace(version), true, nil
		}
	}

	return "", false, nil
}

func clusterPolicyReadinessFailureMessage(clusterPolicy *unstructured.Unstructured) string {
	if cond, found := clusterPolicyCondition(clusterPolicy, clusterPolicyErrorCondition); found && conditionStatusIsTrue(cond.Status) {
		if cond.Message != "" {
			return cond.Message
		}
		return "NVIDIA GPU Operator ClusterPolicy reports Error=True"
	}

	if cond, found := clusterPolicyCondition(clusterPolicy, clusterPolicyReadyCondition); found && !conditionStatusIsTrue(cond.Status) {
		if cond.Message != "" {
			return cond.Message
		}
		return fmt.Sprintf("NVIDIA GPU Operator ClusterPolicy Ready condition is %s", conditionStatusText(cond.Status))
	}

	state, found, _ := unstructured.NestedString(clusterPolicy.Object, "status", "state")
	if found && !strings.EqualFold(strings.TrimSpace(state), clusterPolicyReadyState) {
		return fmt.Sprintf("NVIDIA GPU Operator ClusterPolicy state is %q, expected %q", state, clusterPolicyReadyState)
	}

	if !found {
		if _, hasReady := clusterPolicyCondition(clusterPolicy, clusterPolicyReadyCondition); !hasReady {
			return "NVIDIA GPU Operator ClusterPolicy status does not report readiness"
		}
	}

	return ""
}

// gpuOperatorVersionFromClusterPolicy reads the GPU Operator's own version
// label. It is consulted only to resolve an operand the ClusterPolicy leaves
// unset, where the operator's build default is the effective version; a pinned
// operand is compared directly and this label is not read.
func gpuOperatorVersionFromClusterPolicy(clusterPolicy *unstructured.Unstructured) string {
	return strings.TrimSpace(clusterPolicy.GetLabels()[clusterPolicyVersionLabel])
}

func gpuOperatorVersionFailureMessage(rawVersion, source string) string {
	rawVersion = strings.TrimSpace(rawVersion)
	if rawVersion == "" {
		return fmt.Sprintf("NVIDIA GPU Operator version from %s is missing; minimum supported version is %s",
			source, minimumGPUOperatorVersion)
	}

	version, ok := normalizeGPUOperatorVersion(rawVersion)
	if !ok {
		return fmt.Sprintf("NVIDIA GPU Operator version %q from %s is invalid; minimum supported version is %s",
			rawVersion, source, minimumGPUOperatorVersion)
	}

	// A prerelease from the minimum supported patch train provides the same
	// GPU Operator API baseline. Compare the release cores so, for example,
	// v26.7.1-rc.1 satisfies a v26.7.1 floor while v26.7.0-rc.1 does not.
	versionCore, _ := splitSemverSuffix(version)
	if semver.Compare(versionCore, minimumGPUOperatorVersion) < 0 {
		return fmt.Sprintf("NVIDIA GPU Operator version %s is below minimum supported version %s",
			rawVersion, minimumGPUOperatorVersion)
	}

	return ""
}

// gpuOperandVersionFailureMessage checks the operand versions pinned on the
// ClusterPolicy, which is the dependency gpu-fractioning actually has, rather
// than inferring them from the GPU Operator version.
//
// It also returns the operands whose version is absent. Those are not unchecked:
// an omitted version means the GPU Operator supplies its own build's default, so
// the operator version determines it and the caller resolves them through the
// operator floor instead. That inference is invalid for a pinned operand — the
// whole reason this check exists — and exactly right for a defaulted one.
//
// A version that is pinned but unparseable is neither. A digest names an image
// the operator version says nothing about, and an unreadable version is not
// evidence of an unsupported one, so it is skipped rather than blocked.
func gpuOperandVersionFailureMessage(clusterPolicy *unstructured.Unstructured) (message string, defaulted []string) {
	for _, operand := range gpuOperandRequirements {
		rawVersion, _, _ := unstructured.NestedString(clusterPolicy.Object, operand.specPath...)
		rawVersion = strings.TrimSpace(rawVersion)
		if rawVersion == "" {
			defaulted = append(defaulted, operand.name)
			continue
		}

		version, ok := normalizeOperandVersion(rawVersion)
		if !ok {
			continue
		}

		if semver.Compare(version, operand.minVersion) < 0 {
			return fmt.Sprintf("NVIDIA %s version %s from ClusterPolicy %s is below minimum supported version %s",
				operand.name, rawVersion, strings.Join(operand.specPath, "."), operand.minVersion), defaulted
		}
	}

	return "", defaulted
}

func gpuDriverFailureReason(node *corev1.Node) (reason, message string, found bool) {
	// Use the gpu-fractioning-owned label written by mpsd during startup,
	// not nvidia.com/cuda.driver-version.major. The GPU Operator label can be
	// missing, stale, or absent entirely when the driver is installed by a managed
	// cloud image or other non-GPU-Operator mechanism.
	rawMajor := strings.TrimSpace(node.Labels[driverinfo.NVIDIADriverMajorLabel])
	if rawMajor == "" {
		return daemonmgr.ReasonGPUDriverVersionMissing,
			fmt.Sprintf("NVIDIA driver major version label %q is missing; minimum supported major version is %d", driverinfo.NVIDIADriverMajorLabel, minGPUDriverMajor),
			true
	}

	major, err := driverinfo.ParseDriverMajorLabel(rawMajor)
	if err != nil {
		return daemonmgr.ReasonGPUDriverVersionInvalid,
			fmt.Sprintf("NVIDIA driver major version %q from label %q is invalid; minimum supported major version is %d", rawMajor, driverinfo.NVIDIADriverMajorLabel, minGPUDriverMajor),
			true
	}

	if major < minGPUDriverMajor {
		return daemonmgr.ReasonGPUDriverVersionUnsupported,
			fmt.Sprintf("NVIDIA driver major version %d is below minimum supported major version %d", major, minGPUDriverMajor),
			true
	}

	return "", "", false
}

// normalizeOperandVersion parses a GPU Operator operand version into a
// comparable semver version, keeping only the major.minor.patch release core
// and discarding any suffix.
//
// Operand versions are container image tags, and the container toolkit's carry
// a distro suffix: "v1.20.1-ubuntu20.04". Those cannot be compared as semver
// prereleases for two reasons — "04" is a numeric identifier with a leading
// zero, which makes the whole tag invalid semver, and even where a suffix does
// parse, semver orders a prerelease below the release it qualifies, so
// v1.20.1-ubuntu20.04 would read as older than the v1.20.1 floor it meets.
func normalizeOperandVersion(version string) (string, bool) {
	core, _ := splitSemverSuffix(strings.TrimSpace(version))
	if core == "" {
		return "", false
	}
	if !strings.HasPrefix(core, "v") {
		core = "v" + core
	}

	if !semver.IsValid(core) {
		return "", false
	}
	return semver.Canonical(core), true
}

func normalizeGPUOperatorVersion(version string) (string, bool) {
	normalized := strings.TrimSpace(version)
	if normalized == "" {
		return "", false
	}
	if !strings.HasPrefix(normalized, "v") {
		normalized = "v" + normalized
	}

	if semver.IsValid(normalized) {
		return semver.Canonical(normalized), true
	}

	main, suffix := splitSemverSuffix(normalized)
	if strings.Count(main, ".") == 1 {
		withPatch := main + ".0" + suffix
		if semver.IsValid(withPatch) {
			return semver.Canonical(withPatch), true
		}
	}

	return "", false
}

func splitSemverSuffix(version string) (main, suffix string) {
	idx := strings.IndexAny(version, "-+")
	if idx == -1 {
		return version, ""
	}
	return version[:idx], version[idx:]
}

type clusterPolicyStatusCondition struct {
	Status  string
	Message string
}

func clusterPolicyCondition(clusterPolicy *unstructured.Unstructured, condType string) (clusterPolicyStatusCondition, bool) {
	conditions, found, err := unstructured.NestedSlice(clusterPolicy.Object, "status", "conditions")
	if err != nil || !found {
		return clusterPolicyStatusCondition{}, false
	}

	for _, condition := range conditions {
		conditionMap, ok := condition.(map[string]any)
		if !ok {
			continue
		}
		currentType, _, _ := unstructured.NestedString(conditionMap, "type")
		if currentType != condType {
			continue
		}
		status, _, _ := unstructured.NestedString(conditionMap, "status")
		message, _, _ := unstructured.NestedString(conditionMap, "message")
		return clusterPolicyStatusCondition{Status: status, Message: message}, true
	}

	return clusterPolicyStatusCondition{}, false
}

func conditionStatusIsTrue(status string) bool {
	return strings.EqualFold(strings.TrimSpace(status), string(metav1.ConditionTrue))
}

func conditionStatusText(status string) string {
	if strings.TrimSpace(status) == "" {
		return "Unknown"
	}
	return status
}

func gpuOperatorReadyCondition(generation int64, reason, message string) metav1.Condition {
	return metav1.Condition{
		Type:               daemonmgr.ConditionReady,
		Status:             metav1.ConditionFalse,
		ObservedGeneration: generation,
		LastTransitionTime: metav1.NewTime(time.Now()),
		Reason:             reason,
		Message:            message,
	}
}
