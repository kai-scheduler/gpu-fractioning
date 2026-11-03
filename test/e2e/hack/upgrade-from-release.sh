#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Upgrade check for the chart. Installs the last published chart, which created
# the default GpuFractioningConfig as a post-install hook, upgrades it to this
# checkout through the same skaffold + helm flow the other suites use, and checks
# the CR and the DaemonSets it owns survived in place. Then uninstalls and checks
# they are gone. Run through `make test-e2e-upgrade` against an e2e cluster.
set -euo pipefail

: "${E2E_KUBECONFIG:?set E2E_KUBECONFIG (make e2e-cluster-up writes it)}"
ns=${E2E_OPERATOR_NAMESPACE:-gpu-fractioning}
release=${E2E_HELM_RELEASE:-gso}
previous=${E2E_PREVIOUS_CHART_VERSION:-0.1.8}
chart=${E2E_PREVIOUS_CHART:-oci://ghcr.io/kai-scheduler/gpu-fractioning}
export KUBECONFIG="$E2E_KUBECONFIG"

log() { echo "[upgrade-e2e] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# wait_for <description> <seconds> <command...>: polls every 5 s.
wait_for() {
  local desc=$1 timeout=$2 elapsed=0
  shift 2
  until "$@"; do
    if ((elapsed >= timeout)); then fail "timed out after ${timeout}s waiting for $desc"; fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
}

cr_uid() { kubectl get gpufractioningconfig default -o jsonpath='{.metadata.uid}'; }
cr_annotation() { kubectl get gpufractioningconfig default -o jsonpath="{.metadata.annotations.$1}"; }
daemonset_uids() { kubectl -n "$ns" get daemonsets -o jsonpath='{range .items[*]}{.metadata.name}={.metadata.uid}{"\n"}{end}' | sort; }
daemonset_count() { kubectl -n "$ns" get daemonsets -o name | wc -l | tr -d ' '; }
cr_absent() { ! kubectl get gpufractioningconfig default >/dev/null 2>&1; }
daemonsets_absent() { [ "$(daemonset_count)" -eq 0 ]; }
daemonsets_present() { [ "$(daemonset_count)" -ge 2 ]; }

log "installing chart $previous as release $release in namespace $ns"
# The published images cannot run on a fake-GPU cluster; this check only needs the
# operator to reconcile the CR into DaemonSets, which it does regardless, and the
# version gate skipped so an older operator rolls them out on the fake stack.
helm install "$release" "$chart" --version "$previous" -n "$ns" --create-namespace \
  --set skipGpuStackVersionChecks=true --wait --timeout 5m
wait_for "the operator to create the DaemonSets" 180 daemonsets_present

uid_before=$(cr_uid)
daemonsets_before=$(daemonset_uids)
log "before upgrade: CR uid $uid_before, hook annotation '$(cr_annotation 'helm\.sh/hook')'"
echo "$daemonsets_before"

log "upgrading release $release to this checkout"
"${MAKE:-make}" e2e-deploy

uid_after=$(cr_uid)
daemonsets_after=$(daemonset_uids)
log "after upgrade: CR uid $uid_after"
echo "$daemonsets_after"
[ "$uid_before" = "$uid_after" ] || fail "the GpuFractioningConfig was recreated ($uid_before -> $uid_after)"
[ "$daemonsets_before" = "$daemonsets_after" ] || fail "the DaemonSets were recreated"
[ "$(cr_annotation 'meta\.helm\.sh/release-name')" = "$release" ] || fail "the GpuFractioningConfig is not owned by release $release"
[ -z "$(cr_annotation 'helm\.sh/hook')" ] || fail "the GpuFractioningConfig still carries the hook annotation"
helm get manifest "$release" -n "$ns" | grep -q '^kind: GpuFractioningConfig' || fail "the GpuFractioningConfig is not in the release manifest"

log "uninstalling release $release"
helm uninstall "$release" -n "$ns" --wait --timeout 5m
wait_for "the GpuFractioningConfig to be deleted" 180 cr_absent
wait_for "the DaemonSets to be garbage collected" 180 daemonsets_absent
log "OK: the CR and its DaemonSets survived the upgrade in place and were removed by the uninstall"
