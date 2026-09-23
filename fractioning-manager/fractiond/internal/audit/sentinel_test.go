// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package audit

import (
	"context"
	"testing"

	"github.com/containerd/nri/pkg/api"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/configuration"
)

func newSentinel(stopper ContainerStopper) *Sentinel {
	return NewSentinel(Config{
		AnnotationPrefix: configuration.DefaultAnnotationPrefix,
		MPSPipeDirectory: configuration.DefaultMPSPipeDirectory,
		SMSharingEnabled: true,
		FailOpen:         false,
	}, stopper, nil)
}

func TestAuditStopsOnlyViolations(t *testing.T) {
	fake := &fakeStopper{}
	e := newSentinel(fake)

	pods := []*api.PodSandbox{
		fractioningPod("p1", "pod1", "trainer", "4Gi", ""),
		fractioningPod("p2", "pod2", "trainer", "4Gi", ""),
	}
	containers := []*api.Container{
		{ // injected → ok
			Id: "c1", Name: "trainer", PodSandboxId: "p1",
			State: api.ContainerState_CONTAINER_RUNNING, Env: injectedEqualEnv(), Mounts: []*api.Mount{mpsMount()},
		},
		{ // uninjected → violator
			Id: "c2", Name: "trainer", PodSandboxId: "p2",
			State: api.ContainerState_CONTAINER_RUNNING,
		},
	}

	e.Audit(context.Background(), pods, containers)
	e.Wait()

	assertSameSet(t, fake.stoppedIDs(), []string{"c2"})
}

func TestAuditNoViolationsNoStops(t *testing.T) {
	fake := &fakeStopper{}
	e := newSentinel(fake)

	pods := []*api.PodSandbox{fractioningPod("p1", "pod1", "trainer", "4Gi", "")}
	containers := []*api.Container{{
		Id: "c1", Name: "trainer", PodSandboxId: "p1",
		State: api.ContainerState_CONTAINER_RUNNING, Env: injectedEqualEnv(), Mounts: []*api.Mount{mpsMount()},
	}}

	e.Audit(context.Background(), pods, containers)
	e.Wait()

	if got := len(fake.stoppedIDs()); got != 0 {
		t.Fatalf("expected no stops, got %d", got)
	}
}
