// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package daemonmgr

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func nodeWithConditions(name string, conditions ...corev1.NodeCondition) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status:     corev1.NodeStatus{Conditions: conditions},
	}
}

func gpuFractioningCondition(status corev1.ConditionStatus) corev1.NodeCondition {
	return corev1.NodeCondition{
		Type:   corev1.NodeConditionType(NodeConditionType),
		Status: status,
		Reason: "AllDaemonsReady",
	}
}

func TestRemoveNodeCondition(t *testing.T) {
	readyCondition := corev1.NodeCondition{
		Type:   corev1.NodeReady,
		Status: corev1.ConditionTrue,
	}

	tests := []struct {
		name               string
		node               *corev1.Node
		expectedConditions []corev1.NodeConditionType
	}{
		{
			name:               "removes the gpu-fractioning condition and preserves others",
			node:               nodeWithConditions("node-a", readyCondition, gpuFractioningCondition(corev1.ConditionTrue)),
			expectedConditions: []corev1.NodeConditionType{corev1.NodeReady},
		},
		{
			name:               "no-op when the condition is absent",
			node:               nodeWithConditions("node-b", readyCondition),
			expectedConditions: []corev1.NodeConditionType{corev1.NodeReady},
		},
		{
			name:               "no-op when the node has no conditions",
			node:               nodeWithConditions("node-c"),
			expectedConditions: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := fake.NewClientBuilder().WithObjects(tt.node).Build()

			if err := RemoveNodeCondition(context.Background(), c, tt.node.Name); err != nil {
				t.Fatalf("RemoveNodeCondition() error = %v", err)
			}

			var got corev1.Node
			if err := c.Get(context.Background(), client.ObjectKeyFromObject(tt.node), &got); err != nil {
				t.Fatalf("getting node: %v", err)
			}

			var gotTypes []corev1.NodeConditionType
			for _, cond := range got.Status.Conditions {
				gotTypes = append(gotTypes, cond.Type)
			}
			if len(gotTypes) != len(tt.expectedConditions) {
				t.Fatalf("conditions = %v, expected %v", gotTypes, tt.expectedConditions)
			}
			for i := range gotTypes {
				if gotTypes[i] != tt.expectedConditions[i] {
					t.Errorf("conditions = %v, expected %v", gotTypes, tt.expectedConditions)
				}
			}
		})
	}
}

func TestRemoveNodeCondition_MissingNode(t *testing.T) {
	c := fake.NewClientBuilder().Build()

	if err := RemoveNodeCondition(context.Background(), c, "no-such-node"); err != nil {
		t.Fatalf("RemoveNodeCondition() on missing node error = %v, expected nil", err)
	}
}

func TestPatchNodeCondition_RejectsUnexpectedConditionType(t *testing.T) {
	node := nodeWithConditions("node-a")
	c := fake.NewClientBuilder().WithObjects(node).Build()

	err := PatchNodeCondition(context.Background(), c, c, node.Name, corev1.NodeCondition{
		Type:    corev1.NodeReady,
		Status:  corev1.ConditionFalse,
		Reason:  "BadType",
		Message: "bad type",
	})
	if err == nil {
		t.Fatal("PatchNodeCondition() error = nil, expected error")
	}
	if !strings.Contains(err.Error(), "unexpected node condition type") {
		t.Fatalf("PatchNodeCondition() error = %q, expected unexpected type message", err)
	}
}

// unknownCondition is the condition the shutdown sweep stamps; it is the only
// caller that routinely patches nodes that may already be gone.
func unknownCondition() corev1.NodeCondition {
	return corev1.NodeCondition{
		Type:    corev1.NodeConditionType(NodeConditionType),
		Status:  corev1.ConditionUnknown,
		Reason:  ReasonControllerUnavailable,
		Message: MessageControllerUnavailable,
	}
}

// A node deleted between the list that selected it and the patch that marks it
// has nothing left to assert, so the patch's NotFound is success — the same
// contract RemoveNodeCondition already has. Treating it as a failure makes a
// shutdown that overlaps a node rotation (the common case: both are triggered by
// a cluster upgrade) report per-node errors that are pure noise, and that noise
// is what hides a real failure in the joined error MarkNodeConditionsUnknown
// returns.
func TestPatchNodeCondition_MissingNodeIsSuccess(t *testing.T) {
	// Node absent from the store: the end-to-end shape of the race.
	t.Run("node absent from the store", func(t *testing.T) {
		c := fake.NewClientBuilder().Build()

		if err := PatchNodeCondition(context.Background(), c, c, "no-such-node", unknownCondition()); err != nil {
			t.Fatalf("PatchNodeCondition() on a missing node error = %v, want nil", err)
		}
	})

	// The node exists at read time and is deleted before the patch lands, so
	// only the patch fails. The NotFound is injected rather than raced for, so
	// the branch is exercised deterministically.
	t.Run("node deleted between the read and the patch", func(t *testing.T) {
		base := fake.NewClientBuilder().
			WithObjects(nodeWithConditions("node-a", gpuFractioningCondition(corev1.ConditionTrue))).
			Build()
		writer := failingStatusClient{
			Client:   base,
			failures: map[string]error{"node-a": apierrors.NewNotFound(corev1.Resource("nodes"), "node-a")},
		}

		if err := PatchNodeCondition(context.Background(), base, writer, "node-a", unknownCondition()); err != nil {
			t.Fatalf("PatchNodeCondition() error = %v, want nil when the patch reports NotFound", err)
		}
	})
}

// The NotFound tolerance must be exactly that. A branch that swallowed every
// patch failure would make the shutdown sweep report a clean pass while leaving
// every node still advertising Ready — the precise state the sweep exists to
// prevent, now with nothing in the logs to say so.
func TestPatchNodeCondition_SurfacesNonNotFoundPatchErrors(t *testing.T) {
	for _, tt := range []struct {
		name string
		err  error
	}{
		{name: "forbidden", err: apierrors.NewForbidden(corev1.Resource("nodes"), "node-a", errors.New("RBAC denied"))},
		{name: "conflict", err: apierrors.NewConflict(corev1.Resource("nodes"), "node-a", errors.New("object was modified"))},
		{name: "server timeout", err: apierrors.NewTimeoutError("apiserver is shutting down", 1)},
		{name: "opaque non-status error", err: errors.New("connection refused")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			base := fake.NewClientBuilder().
				WithObjects(nodeWithConditions("node-a", gpuFractioningCondition(corev1.ConditionTrue))).
				Build()
			writer := failingStatusClient{Client: base, failures: map[string]error{"node-a": tt.err}}

			err := PatchNodeCondition(context.Background(), base, writer, "node-a", unknownCondition())
			if err == nil {
				t.Fatalf("PatchNodeCondition() error = nil, want the %s failure surfaced", tt.name)
			}
			// Named node and wrapped cause: this error is what a stuck
			// shutdown has to be diagnosed from.
			if !strings.Contains(err.Error(), "node-a") {
				t.Errorf("error = %q, want it to name the node", err)
			}
			if !errors.Is(err, tt.err) {
				t.Errorf("error = %q, want it to wrap the underlying failure", err)
			}
		})
	}
}

// labelledNode builds a node carrying the gpu-fractioning target label, with or
// without the Ready condition the shutdown sweep looks for.
func labelledNode(name string, labels map[string]string, conditions ...corev1.NodeCondition) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels},
		Status:     corev1.NodeStatus{Conditions: conditions},
	}
}

func getNodeCondition(t *testing.T, c client.Client, name string) (corev1.NodeCondition, bool) {
	t.Helper()
	var node corev1.Node
	if err := c.Get(context.Background(), types.NamespacedName{Name: name}, &node); err != nil {
		t.Fatalf("getting node %s: %v", name, err)
	}
	return FindNodeCondition(&node)
}

var gpuNodeSelector = map[string]string{"nvidia.com/gpu.present": "true"}

// The sweep is scoped by the CR's nodeSelector. Touching a node outside it would
// mean the operator stamping a condition it has no daemons on — and on a cluster
// where a second gpu-fractioning install owns those nodes, overwriting a
// condition that is still being maintained.
func TestMarkNodeConditionsUnknown_OnlyMatchingNodes(t *testing.T) {
	targeted := labelledNode("gpu-node", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue))
	untargeted := labelledNode("cpu-node", map[string]string{"nvidia.com/gpu.present": "false"}, gpuFractioningCondition(corev1.ConditionTrue))
	c := fake.NewClientBuilder().WithObjects(targeted, untargeted).Build()

	if err := MarkNodeConditionsUnknown(context.Background(), c, c, gpuNodeSelector); err != nil {
		t.Fatalf("MarkNodeConditionsUnknown() error = %v", err)
	}

	got, found := getNodeCondition(t, c, "gpu-node")
	if !found {
		t.Fatal("targeted node lost its gpu-fractioning condition")
	}
	if got.Status != corev1.ConditionUnknown {
		t.Errorf("targeted node status = %s, want %s", got.Status, corev1.ConditionUnknown)
	}

	other, found := getNodeCondition(t, c, "cpu-node")
	if !found {
		t.Fatal("untargeted node lost its gpu-fractioning condition")
	}
	if other.Status != corev1.ConditionTrue {
		t.Errorf("untargeted node status = %s, want it left at %s", other.Status, corev1.ConditionTrue)
	}
}

// A node the selector matches but that never carried the condition is not a node
// this operator has said anything about — most likely one the DaemonSet has not
// landed on yet. Introducing Unknown there invents a readiness statement out of
// nothing, and a scheduler gating on the condition would start avoiding a node
// that was previously simply unannotated.
func TestMarkNodeConditionsUnknown_SkipsNodesWithoutTheCondition(t *testing.T) {
	bare := labelledNode("fresh-node", gpuNodeSelector)
	withOtherConditions := labelledNode("kubelet-node", gpuNodeSelector, corev1.NodeCondition{
		Type:   corev1.NodeReady,
		Status: corev1.ConditionTrue,
	})
	c := fake.NewClientBuilder().WithObjects(bare, withOtherConditions).Build()

	if err := MarkNodeConditionsUnknown(context.Background(), c, c, gpuNodeSelector); err != nil {
		t.Fatalf("MarkNodeConditionsUnknown() error = %v", err)
	}

	for _, name := range []string{"fresh-node", "kubelet-node"} {
		if _, found := getNodeCondition(t, c, name); found {
			t.Errorf("node %s gained the %s condition; the sweep must only flip conditions it already owns", name, NodeConditionType)
		}
	}

	// Unrelated conditions on a skipped node are untouched.
	var node corev1.Node
	if err := c.Get(context.Background(), types.NamespacedName{Name: "kubelet-node"}, &node); err != nil {
		t.Fatalf("getting node: %v", err)
	}
	if len(node.Status.Conditions) != 1 || node.Status.Conditions[0].Type != corev1.NodeReady {
		t.Errorf("conditions = %v, want only the kubelet Ready condition", node.Status.Conditions)
	}
}

// Reason and message are what an operator reads in `kubectl describe node` when
// the cluster stalls. A generic Unknown with the previous reason still attached
// would look like a transient blip rather than "nobody is watching this node".
func TestMarkNodeConditionsUnknown_SetsUnknownWithControllerUnavailable(t *testing.T) {
	kubeletReady := corev1.NodeCondition{Type: corev1.NodeReady, Status: corev1.ConditionTrue}

	for _, tt := range []struct {
		name    string
		initial corev1.NodeCondition
	}{
		{name: "from true", initial: gpuFractioningCondition(corev1.ConditionTrue)},
		{name: "from false", initial: gpuFractioningCondition(corev1.ConditionFalse)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			node := labelledNode("gpu-node", gpuNodeSelector, kubeletReady, tt.initial)
			c := fake.NewClientBuilder().WithObjects(node).Build()

			if err := MarkNodeConditionsUnknown(context.Background(), c, c, gpuNodeSelector); err != nil {
				t.Fatalf("MarkNodeConditionsUnknown() error = %v", err)
			}

			got, found := getNodeCondition(t, c, "gpu-node")
			if !found {
				t.Fatal("condition disappeared")
			}
			if got.Status != corev1.ConditionUnknown {
				t.Errorf("status = %s, want %s", got.Status, corev1.ConditionUnknown)
			}
			if got.Reason != ReasonControllerUnavailable {
				t.Errorf("reason = %q, want %q", got.Reason, ReasonControllerUnavailable)
			}
			if got.Message != MessageControllerUnavailable {
				t.Errorf("message = %q, want %q", got.Message, MessageControllerUnavailable)
			}

			// The kubelet's own Ready condition must survive the merge patch:
			// clobbering it would make the node unschedulable for everything.
			var updated corev1.Node
			if err := c.Get(context.Background(), types.NamespacedName{Name: "gpu-node"}, &updated); err != nil {
				t.Fatalf("getting node: %v", err)
			}
			var foundKubelet bool
			for _, cond := range updated.Status.Conditions {
				if cond.Type == corev1.NodeReady && cond.Status == corev1.ConditionTrue {
					foundKubelet = true
				}
			}
			if !foundKubelet {
				t.Errorf("conditions = %v, want the kubelet Ready condition preserved", updated.Status.Conditions)
			}
		})
	}
}

// A CR with no nodeSelector targets every node, so the sweep must too — but
// still only for nodes that already carry the condition.
func TestMarkNodeConditionsUnknown_NilSelectorMatchesEveryNode(t *testing.T) {
	a := labelledNode("node-a", nil, gpuFractioningCondition(corev1.ConditionTrue))
	b := labelledNode("node-b", map[string]string{"any": "label"}, gpuFractioningCondition(corev1.ConditionTrue))
	bare := labelledNode("node-c", nil)
	c := fake.NewClientBuilder().WithObjects(a, b, bare).Build()

	if err := MarkNodeConditionsUnknown(context.Background(), c, c, nil); err != nil {
		t.Fatalf("MarkNodeConditionsUnknown() error = %v", err)
	}

	for _, name := range []string{"node-a", "node-b"} {
		got, found := getNodeCondition(t, c, name)
		if !found || got.Status != corev1.ConditionUnknown {
			t.Errorf("node %s condition = %+v, want Unknown", name, got)
		}
	}
	if _, found := getNodeCondition(t, c, "node-c"); found {
		t.Error("node-c gained a condition it never had")
	}
}

// The sweep runs inside a shutdown window. One node that refuses the patch must
// not cost every node behind it in the list its Unknown marking, and the failure
// must still be reported so it is not lost.
func TestMarkNodeConditionsUnknown_JoinsPerNodeFailures(t *testing.T) {
	objs := []client.Object{
		labelledNode("node-a", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
		labelledNode("node-b", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
		labelledNode("node-c", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
	}
	base := fake.NewClientBuilder().WithObjects(objs...).Build()
	writer := failingStatusClient{
		Client: base,
		failures: map[string]error{
			"node-a": errors.New("boom-a"),
			"node-b": errors.New("boom-b"),
		},
	}

	err := MarkNodeConditionsUnknown(context.Background(), base, writer, gpuNodeSelector)
	if err == nil {
		t.Fatal("MarkNodeConditionsUnknown() error = nil, want the per-node failures reported")
	}
	// Both failures, not just the first: errors.Join keeps every node that
	// needs manual attention visible in the shutdown log.
	for _, want := range []string{"node-a", "boom-a", "node-b", "boom-b"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %q, want it to mention %q", err, want)
		}
	}

	// node-c is listed after both failures and must still have been swept.
	got, found := getNodeCondition(t, base, "node-c")
	if !found || got.Status != corev1.ConditionUnknown {
		t.Errorf("node-c condition = %+v (found=%t), want Unknown; the sweep aborted on the first failure", got, found)
	}
}

// Nodes disappear during a rolling cluster upgrade, which is exactly when the
// operator is also being restarted. A node that vanished between the list and
// the patch must not cost the remaining nodes their marking.
func TestMarkNodeConditionsUnknown_NodeDeletedMidSweep(t *testing.T) {
	present := []client.Object{
		labelledNode("node-a", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
		labelledNode("node-c", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
	}
	store := fake.NewClientBuilder().WithObjects(present...).Build()

	// The list still reports node-b; it is gone from the store by the time the
	// patch lands.
	reader := &pagedNodeReader{
		inner: store,
		pages: [][]corev1.Node{{
			*labelledNode("node-a", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
			*labelledNode("node-b", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
			*labelledNode("node-c", gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue)),
		}},
	}

	// A vanished node is not a failure: PatchNodeCondition treats the patch's
	// NotFound as success, so the sweep reports a clean pass. Asserted
	// unconditionally — an `if err != nil` guard here would make this test pass
	// whether or not that tolerance exists.
	if err := MarkNodeConditionsUnknown(context.Background(), reader, store, gpuNodeSelector); err != nil {
		t.Fatalf("MarkNodeConditionsUnknown() error = %v, want nil; a node deleted mid-sweep is not a shutdown failure", err)
	}

	for _, name := range []string{"node-a", "node-c"} {
		got, found := getNodeCondition(t, store, name)
		if !found || got.Status != corev1.ConditionUnknown {
			t.Errorf("node %s condition = %+v (found=%t), want Unknown; a deleted node aborted the sweep", name, got, found)
		}
	}
}

// A list failure is not a per-node problem, so it aborts rather than silently
// sweeping a partial cluster and reporting success.
func TestMarkNodeConditionsUnknown_ListErrorAborts(t *testing.T) {
	reader := &pagedNodeReader{listErr: errors.New("apiserver is going away")}

	err := MarkNodeConditionsUnknown(context.Background(), reader, fake.NewClientBuilder().Build(), gpuNodeSelector)
	if err == nil {
		t.Fatal("MarkNodeConditionsUnknown() error = nil, want the list failure surfaced")
	}
	if !strings.Contains(err.Error(), "apiserver is going away") {
		t.Errorf("error = %q, want it to wrap the list failure", err)
	}
}

// Large GPU fleets exceed a single list page. The fake client ignores Limit and
// never returns a continue token, so pagination is driven through a reader that
// implements the real server contract: every page must be visited, and every
// follow-up request must still carry the same label selector — dropping it on
// page two would sweep the whole cluster instead of the CR's nodes.
func TestMarkNodeConditionsUnknown_Paginates(t *testing.T) {
	var (
		objs  []client.Object
		pages [][]corev1.Node
	)
	const (
		pageCount    = 3
		perPageNodes = 4
	)
	for p := range pageCount {
		var page []corev1.Node
		for i := range perPageNodes {
			node := labelledNode(fmt.Sprintf("node-%d-%d", p, i), gpuNodeSelector, gpuFractioningCondition(corev1.ConditionTrue))
			objs = append(objs, node)
			page = append(page, *node.DeepCopy())
		}
		pages = append(pages, page)
	}

	store := fake.NewClientBuilder().WithObjects(objs...).Build()
	reader := &pagedNodeReader{inner: store, pages: pages}

	if err := MarkNodeConditionsUnknown(context.Background(), reader, store, gpuNodeSelector); err != nil {
		t.Fatalf("MarkNodeConditionsUnknown() error = %v", err)
	}

	if len(reader.calls) != pageCount {
		t.Fatalf("list calls = %d, want %d (pages were skipped or re-requested)", len(reader.calls), pageCount)
	}
	for i, call := range reader.calls {
		if call.selector != "nvidia.com/gpu.present=true" {
			t.Errorf("list call %d selector = %q, want the CR's node selector; a later page sweeping the whole cluster would mark nodes this CR does not own", i, call.selector)
		}
		if call.limit != nodeListPageSize {
			t.Errorf("list call %d limit = %d, want %d", i, call.limit, nodeListPageSize)
		}
	}

	for _, o := range objs {
		got, found := getNodeCondition(t, store, o.GetName())
		if !found || got.Status != corev1.ConditionUnknown {
			t.Errorf("node %s condition = %+v (found=%t), want Unknown", o.GetName(), got, found)
		}
	}
}

// failingStatusClient makes the status patch of named nodes fail, so the sweep's
// per-node error handling can be exercised without a real API server.
type failingStatusClient struct {
	client.Client
	failures map[string]error
}

func (c failingStatusClient) Status() client.SubResourceWriter {
	return failingStatusWriter{SubResourceWriter: c.Client.Status(), failures: c.failures}
}

type failingStatusWriter struct {
	client.SubResourceWriter
	failures map[string]error
}

func (w failingStatusWriter) Patch(ctx context.Context, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
	if err, ok := w.failures[obj.GetName()]; ok {
		return err
	}
	return w.SubResourceWriter.Patch(ctx, obj, patch, opts...)
}

type recordedListCall struct {
	selector string
	limit    int64
	token    string
}

// pagedNodeReader serves node lists the way the API server does — one Limit-sized
// page at a time with a continue token — which the fake client does not do.
type pagedNodeReader struct {
	inner   client.Reader
	pages   [][]corev1.Node
	listErr error
	calls   []recordedListCall
}

func (r *pagedNodeReader) Get(ctx context.Context, key types.NamespacedName, obj client.Object, opts ...client.GetOption) error {
	return r.inner.Get(ctx, key, obj, opts...)
}

func (r *pagedNodeReader) List(_ context.Context, list client.ObjectList, opts ...client.ListOption) error {
	if r.listErr != nil {
		return r.listErr
	}

	nodeList, ok := list.(*corev1.NodeList)
	if !ok {
		return fmt.Errorf("pagedNodeReader only serves NodeList, got %T", list)
	}

	options := &client.ListOptions{}
	options.ApplyOptions(opts)

	call := recordedListCall{limit: options.Limit, token: options.Continue}
	if options.LabelSelector != nil {
		call.selector = options.LabelSelector.String()
	}
	r.calls = append(r.calls, call)

	// Guard against a loop that never advances its continue token.
	if len(r.calls) > len(r.pages)+1 {
		return fmt.Errorf("pagination did not terminate after %d list calls", len(r.calls))
	}

	page := 0
	if options.Continue != "" {
		var err error
		if page, err = strconv.Atoi(options.Continue); err != nil {
			return fmt.Errorf("unexpected continue token %q: %w", options.Continue, err)
		}
	}
	if page >= len(r.pages) {
		return fmt.Errorf("continue token %q points past the last page", options.Continue)
	}

	// Assigned, not appended: the real decoder replaces the list contents on
	// every page.
	nodeList.Items = append([]corev1.Node(nil), r.pages[page]...)
	if page+1 < len(r.pages) {
		nodeList.Continue = strconv.Itoa(page + 1)
	} else {
		nodeList.Continue = ""
	}
	return nil
}
