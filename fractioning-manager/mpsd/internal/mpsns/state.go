// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mpsns

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

const (
	statePerm    = 0o600
	stateDirPerm = 0o755
)

// Lease is one container's MPS namespace: what was provisioned, where the
// container has to be pointed to be inside it, and what it was capped at.
type Lease struct {
	// ContainerID is the runtime's container id — the key everything else is
	// found by, and the identity fractiond releases with.
	ContainerID string `json:"containerId"`
	// Namespace is the control-daemon namespace name derived from ContainerID.
	Namespace string `json:"namespace"`
	// Server is the MPS server the namespace lives on.
	Server string `json:"server"`
	// PipeDirectory is the directory holding the namespace's control socket, as
	// READ back from the daemon rather than derived. It is the only path a
	// container may be given: handed the server's directory instead, the
	// container lands in the uncapped `default` namespace.
	PipeDirectory string `json:"pipeDirectory"`
	// ActiveThreadPercent is the namespace's SM ceiling, the value the
	// container cannot raise from inside itself.
	ActiveThreadPercent int `json:"activeThreadPercent"`
	// GPUUUIDs records the devices the scheduler assigned the container.
	//
	// It is diagnostic only. A namespace belongs to a server, not to a device,
	// so the cap is not scoped per GPU by this field and nothing here pretends
	// otherwise; it is carried so a namespace in `namespace list` can be traced
	// back to the GPUs its container was placed on.
	GPUUUIDs []string `json:"gpuUuids,omitempty"`
	// CreatedAt is when the lease was first taken. Reconciliation uses it to
	// avoid pruning a namespace provisioned moments before mpsd restarted, for
	// a container whose first process has not started yet.
	CreatedAt time.Time `json:"createdAt"`
	// ReleasedAt is set when a release was asked for but the daemon would not
	// delete the namespace yet (an attached client blocks the delete). The
	// sweep retries these; it is nil for a live lease.
	ReleasedAt *time.Time `json:"releasedAt,omitempty"`
}

// released reports whether the lease is waiting to be deleted.
func (l Lease) released() bool { return l.ReleasedAt != nil }

// state is the on-disk lease set.
//
// It is persisted because mpsd restarting must not strand namespaces, and
// because the namespaces themselves do not survive an MPS restart: after mpsd
// bounces the control daemon (which it does to clear a wedge, and after the
// last client leaves when recycling is on), every namespace is gone while the
// containers that were using them are still running. Reconciliation rebuilds
// them at the same names — and therefore the same pipe directories the
// containers are already bind-mounted to — from this file. Without it, a
// recycle would quietly leave every running fractional container pointing at a
// directory the daemon no longer owns.
//
// It deliberately lives on a host path that does NOT survive a reboot: the
// namespaces it describes do not either.
type state struct {
	Leases []Lease `json:"leases"`
}

// store persists the lease set to a single JSON file.
type store struct {
	path string
}

// load reads the lease set. A missing file is an empty set, not an error: the
// first start on a node, and the first start after a reboot wiped the runtime
// directory, are both normal.
func (s store) load() ([]Lease, error) {
	if s.path == "" {
		return nil, nil
	}
	content, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading MPS namespace state %q: %w", s.path, err)
	}
	if len(strings.TrimSpace(string(content))) == 0 {
		return nil, nil
	}

	var decoded state
	if err := json.Unmarshal(content, &decoded); err != nil {
		// A corrupt file must not wedge mpsd at startup. Reporting it lets the
		// caller decide, and the caller starts from an empty set and lets
		// reconciliation delete whatever namespaces it finds — safe, because
		// reconciliation only ever touches names this package generated.
		return nil, fmt.Errorf("decoding MPS namespace state %q: %w", s.path, err)
	}
	return decoded.Leases, nil
}

// save writes the lease set atomically, so an mpsd killed mid-write leaves the
// previous state rather than a truncated file.
func (s store) save(leases []Lease) error {
	if s.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.path), stateDirPerm); err != nil {
		return fmt.Errorf("creating MPS namespace state directory: %w", err)
	}

	sorted := slices.Clone(leases)
	slices.SortFunc(sorted, func(a, b Lease) int { return strings.Compare(a.ContainerID, b.ContainerID) })

	content, err := json.MarshalIndent(state{Leases: sorted}, "", "  ")
	if err != nil {
		return fmt.Errorf("encoding MPS namespace state: %w", err)
	}
	content = append(content, '\n')

	temp, err := os.CreateTemp(filepath.Dir(s.path), filepath.Base(s.path)+".*")
	if err != nil {
		return fmt.Errorf("creating MPS namespace state temp file: %w", err)
	}
	tempName := temp.Name()
	defer func() { _ = os.Remove(tempName) }()

	if _, err := temp.Write(content); err != nil {
		_ = temp.Close()
		return fmt.Errorf("writing MPS namespace state: %w", err)
	}
	if err := temp.Chmod(statePerm); err != nil {
		_ = temp.Close()
		return fmt.Errorf("setting MPS namespace state permissions: %w", err)
	}
	if err := temp.Close(); err != nil {
		return fmt.Errorf("closing MPS namespace state temp file: %w", err)
	}
	if err := os.Rename(tempName, s.path); err != nil {
		return fmt.Errorf("replacing MPS namespace state %q: %w", s.path, err)
	}
	return nil
}
