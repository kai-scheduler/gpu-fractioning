// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package daemonpaths holds host paths that more than one Go module has to
// agree on exactly.
//
// The operator and the node daemons are separate modules, so a shared path is
// otherwise a literal repeated in two places with a "keep in sync" comment —
// the pattern used for the MPS pipe directory. That works only because a
// mismatch there is loud: the mount is missing and containers fail to start.
// The paths here are ones where a mismatch is SILENT, so they get a single
// definition that both modules import instead.
package daemonpaths

// MPSDrainSocket is the unix socket mpsd serves its MPS client-drain endpoint
// on and fractiond dials, and the path the operator mounts into both pods.
//
// A mismatch between the three has no symptom at all: fractiond's
// drain-before-stop call quietly fails to connect on every container stop, the
// container is stopped anyway (by design — a drain problem must never become a
// stuck pod), and the node silently loses its protection against a killed MPS
// client wedging the GPU for every other tenant. Nothing logs an error loud
// enough to notice, and the damage only shows up as an unexplained hang much
// later. Hence one constant, not three.
const MPSDrainSocket = "/var/run/gpu-fractioning/drain/mpsd.sock"
