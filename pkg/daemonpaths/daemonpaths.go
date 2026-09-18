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

// NVMLShimDir is the host directory fractiond stages its shadow
// libnvidia-ml.so.1 into, the directory the operator hostPath-mounts into the
// fractiond pod so that staging lands on the node, and the bind-mount source
// fractiond then hands to every GPU-fractioning container.
//
// A mismatch between those three is silent in the same way the drain socket is.
// fractiond writes the library to whatever path it was told, sees no error, and
// keeps injecting a mount that points at it; the containers come up, the caps
// are still enforced by MPS, and the only symptom is that nvidia-smi inside the
// pod goes on reporting the whole physical GPU — which is exactly the state
// this feature exists to fix, and is indistinguishable from the feature simply
// not being deployed yet.
//
// Deliberately under /var/lib rather than /var/run: /var/run is a tmpfs that is
// wiped on reboot, and this holds a real file copied out of the fractiond image
// rather than a runtime socket. Staging re-runs on every fractiond start, so a
// wipe would be survivable, but a library is not runtime state and does not
// belong in a runtime-state directory.
const NVMLShimDir = "/var/lib/gpu-fractioning/lib"
