{{/*
Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "gpu-fractioning.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name.
*/}}
{{- define "gpu-fractioning.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "gpu-fractioning.labels" -}}
helm.sh/chart: {{ include "gpu-fractioning.chart" . }}
{{ include "gpu-fractioning.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "gpu-fractioning.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gpu-fractioning.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Chart name and version.
*/}}
{{- define "gpu-fractioning.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Service account name.
*/}}
{{- define "gpu-fractioning.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "gpu-fractioning.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
The validated global.fipsMode. Resolved in one place so an unrecognised value
fails the render with a clear message, rather than silently falling through to
the regular images — a FIPS install quietly not being FIPS is the worst outcome
here. Accepts "off", "on" and "only"; see values.yaml for what each selects.
*/}}
{{- define "gpu-fractioning.fipsMode" -}}
{{- $mode := default "off" ((.Values.global).fipsMode) -}}
{{- if not (has $mode (list "off" "on" "only")) -}}
{{- fail (printf "global.fipsMode must be \"off\", \"on\" or \"only\", got %q" $mode) -}}
{{- end -}}
{{- $mode -}}
{{- end }}

{{/*
The GODEBUG entry that puts a container's Go runtime into FIPS-only mode, or
nothing at all in the other modes. Emitted as a full env list entry so callers
can include it inline without an enclosing conditional.

tlsmlkem=0 travels with it because crypto/tls prefers the X25519MLKEM768 hybrid
key exchange, whose implementation calls the unapproved X25519 primitive; under
fips140=only that fails every outbound TLS handshake. See golang/go#78298.
*/}}
{{- define "gpu-fractioning.fipsOnlyEnv" -}}
{{- if eq (include "gpu-fractioning.fipsMode" .) "only" -}}
- name: GODEBUG
  value: fips140=only,tlsmlkem=0
{{- end -}}
{{- end }}

{{/*
Resolves a component's image tag: the explicit per-image tag if set, otherwise
the chart's appVersion. When fipsMode is not "off", appends "-fips" to whatever
resolved, so selecting FIPS never conflicts with pinning a version.
Usage:
  {{ include "gpu-fractioning.imageTag" (dict "root" $ "tag" .Values.images.mpsd.tag) }}
*/}}
{{- define "gpu-fractioning.imageTag" -}}
{{- $tag := .tag | default .root.Chart.AppVersion -}}
{{- if ne (include "gpu-fractioning.fipsMode" .root) "off" -}}
{{- $tag = printf "%s-fips" $tag -}}
{{- end -}}
{{- $tag -}}
{{- end }}

{{/*
Name shared by the crd-upgrader hook Job and the ServiceAccount, ClusterRole and
ClusterRoleBinding it runs under. The ClusterRole and binding are cluster-scoped,
so this must not collide between releases in different namespaces.

The base is truncated to 50 before the 13-character suffix is appended, not
after. Appending first and truncating to 63 silently drops the suffix once
fullname reaches 50 characters, returning fullname itself — which collides with
the operator's own ServiceAccount and RBAC, and the hook's delete policies would
then delete them out from under the running operator.
*/}}
{{- define "gpu-fractioning.crdUpgraderName" -}}
{{- printf "%s-crd-upgrader" (include "gpu-fractioning.fullname" . | trunc 50 | trimSuffix "-") }}
{{- end }}

{{/*
Whether to render the OpenShift-specific resources, as "true" or "" so callers
can use it directly in an if.

Resolved in one place because three things key off it (the SCC, its RBAC and the
CRI socket default) and they must not disagree: an install that creates the SCC
but keeps the containerd socket path, or vice versa, is worse than either
consistent outcome.

"auto" detects OpenShift by the presence of the security.openshift.io/v1 API
group, which is what serves SecurityContextConstraints — the capability actually
being used, rather than a proxy for it like a cluster version or node label.

The explicit override exists because detection is only as good as .Capabilities,
which `helm template` populates from the local Helm build rather than a cluster.
Rendering manifests for OpenShift from a non-OpenShift machine (GitOps, an
air-gapped bundle, `helm template | kubectl apply`) therefore needs
openshift.enabled=true, and that is the common case for both, not an edge case.
*/}}
{{- define "gpu-fractioning.isOpenShift" -}}
{{- $raw := (.Values.openshift).enabled -}}
{{- /* Not `default "auto" $raw`: Helm's default treats false as empty, which
       would silently turn openshift.enabled=false back into auto-detection. */ -}}
{{- $enabled := "auto" -}}
{{- if not (kindIs "invalid" $raw) -}}
{{- $enabled = printf "%v" $raw -}}
{{- end -}}
{{- if not (has $enabled (list "auto" "true" "false")) -}}
{{- fail (printf "openshift.enabled must be \"auto\", true or false, got %q" $enabled) -}}
{{- end -}}
{{- if eq $enabled "auto" -}}
{{- if .Capabilities.APIVersions.Has "security.openshift.io/v1" -}}true{{- end -}}
{{- else if eq $enabled "true" -}}
true
{{- end -}}
{{- end }}

{{/*
The CRI socket path for the generated CR, or nothing to leave the field unset
and let the API server default it to containerd's socket.

An explicit value always wins. Otherwise OpenShift gets CRI-O's socket, since
that is the runtime it ships and the containerd default is simply wrong there.
*/}}
{{- define "gpu-fractioning.criSocketPath" -}}
{{- $path := (.Values.fractioningAgent).criSocketPath -}}
{{- if $path -}}
{{- $path -}}
{{- else if include "gpu-fractioning.isOpenShift" . -}}
/run/crio/crio.sock
{{- end -}}
{{- end }}

{{/*
Name of the SecurityContextConstraints and the Role/RoleBinding that grants use
of it. SCCs are cluster-scoped, so this must not collide between releases in
different namespaces.
*/}}
{{- define "gpu-fractioning.sccName" -}}
{{- printf "%s-daemon" (include "gpu-fractioning.fullname" . | trunc 56 | trimSuffix "-") }}
{{- end }}

{{/*
Daemon service account name.
*/}}
{{- define "gpu-fractioning.daemonServiceAccountName" -}}
{{- if .Values.daemonServiceAccount.create }}
{{- default (printf "%s-daemon" (include "gpu-fractioning.fullname" .)) .Values.daemonServiceAccount.name }}
{{- else }}
{{- default "default" .Values.daemonServiceAccount.name }}
{{- end }}
{{- end }}
