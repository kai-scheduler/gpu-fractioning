# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------
# E2E. Requires kind, kubectl, helm, docker, python3
# (see test/e2e/hack/requirements.txt).
#
#   make e2e                          # cluster up + load images + helm deploy + run tests
#   make e2e-cluster-down             # tear down the kind cluster
#
# The gpu-fractioning stack is built and installed the way it ships, by Skaffold
# (skaffold.yaml): it builds all four component images (operator, fractiond,
# metricsd, mpsd), loads them into the cluster, and `helm install`s the operator
# chart (operator/charts). The operator then creates the fractiond DaemonSet
# (which hosts the metricsd sidecar under test) and the mpsd DaemonSet. The Go
# suite only connects to the deployed cluster and asserts — it deploys nothing.
#
# All knobs are E2E_* env vars — see test/e2e/README.md and
# test/e2e/hack/create-cluster.py for the full list (node count, GPU Operator + Mokka
# version, GPUs per node, etc).
#
# Included from the root Makefile, so all paths here are relative to the repo
# root (make's `include` does not change the working directory).
# -----------------------------------------------------------

E2E_CLUSTER_NAME              ?= gpu-fractioning-e2e
E2E_GPU_WORKER_NODES          ?= 2
E2E_NON_GPU_WORKER_NODES      ?= 1
E2E_GPUS_PER_NODE             ?= 2
E2E_GPU_OPERATOR_VERSION      ?= v26.7.2
PYTHON                        ?= python3

# Port of the local image registry create-cluster.py stands up.
# Skaffold pushes the component images to localhost:$(E2E_REGISTRY_PORT) (see
# --default-repo in E2E_SKAFFOLD_FLAGS); nodes pull them back via the cluster's
# containerd hosts.toml mirror. Must match create-cluster.py's E2E_REGISTRY_PORT.
E2E_REGISTRY_PORT             ?= 5001

# Namespace the operator chart is installed into (must match skaffold.yaml's
# release namespace). The Go suite reads this via E2E_OPERATOR_NAMESPACE.
E2E_OPERATOR_NAMESPACE        ?= gpu-fractioning

# Dedicated kubeconfig for the e2e cluster. create-cluster.py writes it here
# and the deploy/test targets below point kubectl/helm/skaffold/go-test at it. This keeps
# the whole flow off your default context (e.g. a remote cluster) entirely.
E2E_KUBECONFIG                ?= $(HOME)/.kube/$(E2E_CLUSTER_NAME).yaml

# Your default kubeconfig — only touched by the opt-in e2e-kubeconfig-merge/
# -unmerge targets (never by the core flow).
KUBECONFIG_DEFAULT            ?= $(HOME)/.kube/config
E2E_KUBE_CONTEXT              := kind-$(E2E_CLUSTER_NAME)

# Host arch, passed to Skaffold for metricsd's TARGETARCH build-arg (its
# Dockerfile hardcodes ARG TARGETARCH=amd64, which otherwise wins over the build
# platform). amd64 CI runners get amd64; Apple Silicon gets arm64.
E2E_ARCH                      ?= $(shell uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')

export E2E_CLUSTER_NAME
export E2E_GPU_WORKER_NODES
export E2E_NON_GPU_WORKER_NODES
export E2E_GPUS_PER_NODE
export E2E_GPU_OPERATOR_VERSION
export E2E_ARCH
export E2E_REGISTRY_PORT
# create-cluster.py writes only the dedicated E2E_KUBECONFIG;
# the Go suite (config.go) reads it to connect. skaffold/helm/kubectl instead read
# KUBECONFIG, so the deploy/undeploy recipes set it inline from E2E_KUBECONFIG.
export E2E_KUBECONFIG

.PHONY: e2e e2e-cluster-up e2e-cluster-down e2e-cluster-deps \
	e2e-deploy e2e-undeploy e2e-kubeconfig-merge e2e-kubeconfig-unmerge \
	test-e2e test-e2e-metrics test-e2e-operator test-e2e-fractiond run-e2e

e2e: e2e-cluster-up e2e-deploy test-e2e

# E2E_GO_TEST wraps a suite's `go test` invocation. Each suite lives in its own
# tests/<suite> package so a CI job can run exactly one suite.
#
# -timeout 20m (< the job timeout): the metrics suite legitimately runs
# ~10-11 min — each nvml-mock SetProcesses cycle restarts metricsd and waits for
# the mounted ConfigMap to propagate (~90 s), across ~11 tests — so the timeout
# must clear that with margin. Keeping it below the job timeout means a genuine
# hang still dumps every goroutine's stack here instead of being silently killed
# when the job is cancelled at its own limit.
E2E_GO_TEST = cd test/e2e && E2E_OPERATOR_NAMESPACE=$(E2E_OPERATOR_NAMESPACE) E2E_GPU_NODE_COUNT=$(E2E_GPU_WORKER_NODES) E2E_NVML_MOCK=1 E2E_GPU_COUNT_PER_NODE=$(E2E_GPUS_PER_NODE) go test -tags e2e -v -timeout 20m

e2e-cluster-deps:
	$(PYTHON) -m pip install -q -r test/e2e/hack/requirements.txt

e2e-cluster-up: e2e-cluster-deps
	$(PYTHON) test/e2e/hack/create-cluster.py

e2e-cluster-down: e2e-cluster-deps e2e-kubeconfig-unmerge
	$(PYTHON) test/e2e/hack/create-cluster.py --delete

# Opt-in convenience: add the e2e cluster to your DEFAULT kubeconfig
# ($(KUBECONFIG_DEFAULT)) as the named context "$(E2E_KUBE_CONTEXT)", so you can
# `kubectl config use-context $(E2E_KUBE_CONTEXT)` without setting KUBECONFIG.
# Your current-context is preserved and the previous file is backed up to
# <config>.bak. The core flow never calls this — it uses the standalone
# E2E_KUBECONFIG file. e2e-cluster-down runs -unmerge to clean up.
e2e-kubeconfig-merge:
	@test -f "$(E2E_KUBECONFIG)" || { echo "no $(E2E_KUBECONFIG) — run 'make e2e-cluster-up' first"; exit 1; }
	@mkdir -p "$(dir $(KUBECONFIG_DEFAULT))"
	@cur=$$(KUBECONFIG="$(KUBECONFIG_DEFAULT)" kubectl config current-context 2>/dev/null); \
	[ -f "$(KUBECONFIG_DEFAULT)" ] && cp -f "$(KUBECONFIG_DEFAULT)" "$(KUBECONFIG_DEFAULT).bak"; \
	KUBECONFIG="$(KUBECONFIG_DEFAULT):$(E2E_KUBECONFIG)" kubectl config view --flatten > "$(KUBECONFIG_DEFAULT).tmp"; \
	mv "$(KUBECONFIG_DEFAULT).tmp" "$(KUBECONFIG_DEFAULT)"; \
	if [ -n "$$cur" ]; then KUBECONFIG="$(KUBECONFIG_DEFAULT)" kubectl config use-context "$$cur" >/dev/null; \
	else KUBECONFIG="$(KUBECONFIG_DEFAULT)" kubectl config unset current-context >/dev/null; fi; \
	echo "Merged context '$(E2E_KUBE_CONTEXT)' into $(KUBECONFIG_DEFAULT) (backup: $(KUBECONFIG_DEFAULT).bak; current-context unchanged)"

# Remove the e2e cluster's entries from your default kubeconfig (no-op if absent).
e2e-kubeconfig-unmerge:
	@KUBECONFIG="$(KUBECONFIG_DEFAULT)" kubectl config delete-context "$(E2E_KUBE_CONTEXT)" >/dev/null 2>&1 || true
	@KUBECONFIG="$(KUBECONFIG_DEFAULT)" kubectl config delete-cluster "$(E2E_KUBE_CONTEXT)" >/dev/null 2>&1 || true
	@KUBECONFIG="$(KUBECONFIG_DEFAULT)" kubectl config unset "users.$(E2E_KUBE_CONTEXT)" >/dev/null 2>&1 || true

# e2e-deploy builds all four images, pushes them to the kind local registry, and
# helm installs the operator chart — all via `skaffold run -p e2e`
# (skaffold.yaml). The current kube-context must point at the e2e cluster.
#
# Flags:
#   --platform pins the build to the host arch instead of letting Skaffold probe
#     the cluster for its node platform: that probe fails (and passes an empty
#     --platform to metricsd's `FROM --platform=$TARGETPLATFORM`) whenever the
#     current kube-context is unreachable or non-kind.
#   --default-repo=localhost:$(E2E_REGISTRY_PORT) makes Skaffold tag/push the
#     images into the kind local registry (create-cluster.py). Combined with
#     push:true + pullPolicy=IfNotPresent (skaffold.yaml e2e profile), a node
#     that evicts an image under disk pressure re-pulls it from the registry
#     instead of wedging at ErrImageNeverPull.
#   --cache-artifacts=false skips Skaffold's registry-backed cache check, which
#     otherwise shells out to a docker credential helper (e.g. gcloud) to look up
#     the chart's default ghcr.io repos.
#   --verbosity=error silences Skaffold's "image [fractiond|mpsd|metricsd] is not
#     used" warnings: those images reach the pods indirectly (the chart passes
#     their repo/tag to the operator, which creates the DaemonSets), so Skaffold's
#     rendered-manifest scan can't see them. False positives, not real problems.
#
# It then waits on the fractiond DaemonSet specifically (which hosts the metricsd
# sidecar under test), NOT the operator's aggregate CR Ready condition. The chart
# also creates mpsd. Daemons use the real nvidia RuntimeClass. Fake MPS and the
# test-controlled metrics NVML input remain explicit test dependencies.
E2E_SKAFFOLD_FLAGS = --platform=linux/$(E2E_ARCH) --default-repo=localhost:$(E2E_REGISTRY_PORT) --cache-artifacts=false --verbosity=error

e2e-deploy:
	KUBECONFIG=$(E2E_KUBECONFIG) skaffold run -p e2e $(E2E_SKAFFOLD_FLAGS)
	@# Pods pull Skaffold's images through the containerd registry mirror.
	@echo "Waiting for the operator to create the fractiond DaemonSet..."
	@for i in $$(seq 1 60); do KUBECONFIG=$(E2E_KUBECONFIG) kubectl -n $(E2E_OPERATOR_NAMESPACE) get ds gpu-fractioning-fractiond >/dev/null 2>&1 && break; sleep 2; done
	KUBECONFIG=$(E2E_KUBECONFIG) kubectl -n $(E2E_OPERATOR_NAMESPACE) rollout status ds/gpu-fractioning-fractiond --timeout=180s

e2e-undeploy:
	KUBECONFIG=$(E2E_KUBECONFIG) skaffold delete -p e2e --verbosity=error

# test-e2e runs every suite. CI jobs should call a specific suite target
# (e.g. test-e2e-metrics) so each job runs only its own suite.
test-e2e:
	$(E2E_GO_TEST) ./tests/...

test-e2e-metrics:
	$(E2E_GO_TEST) ./tests/metrics/...

# The operator suite drives CR lifecycle, rollouts, fault injection and config
# propagation, so it needs a longer timeout than the metrics suite (several
# delete→recreate rollouts + fault-recovery windows). On fake clusters, mpsd
# reads the baked-in nvml-mock library before it starts MPS. E2E_GPU_NODE_COUNT
# primes the preflight.
test-e2e-operator:
	cd test/e2e && E2E_OPERATOR_NAMESPACE=$(E2E_OPERATOR_NAMESPACE) E2E_GPU_NODE_COUNT=$(E2E_GPU_WORKER_NODES) E2E_GPU_COUNT_PER_NODE=$(E2E_GPUS_PER_NODE) go test -tags e2e -v -timeout 40m ./tests/operator/...

# The fractiond suite exercises the fractiond NRI data plane (env/mount injection
# on annotated workloads). It asserts daemon behavior, not GPU metrics. On fake
# clusters, mpsd reads the baked-in nvml-mock library before it starts MPS. The
# suite creates a handful of workload pods and does one CR re-roll (fail-open),
# so it's much shorter than the operator suite — a 20m timeout clears it with
# margin, under the job limit.
test-e2e-fractiond:
	cd test/e2e && E2E_OPERATOR_NAMESPACE=$(E2E_OPERATOR_NAMESPACE) E2E_GPU_NODE_COUNT=$(E2E_GPU_WORKER_NODES) E2E_GPU_COUNT_PER_NODE=$(E2E_GPUS_PER_NODE) go test -tags e2e -v -timeout 20m ./tests/fractiond/...

# run-e2e runs the suite against whatever cluster the caller provides
# (set E2E_KUBECONFIG=...). The suite never provisions a cluster, so this works
# against any cluster regardless of how it was created — not just kind.
run-e2e: test-e2e
