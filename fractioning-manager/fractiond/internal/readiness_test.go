// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"context"
	"io"
	"log/slog"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/fractiond/internal/readiness"
)

// The plugin depends on the ReadinessSetter interface; readiness.State is the
// production implementation, so keep the two in sync at compile time.
var _ ReadinessSetter = (*readiness.State)(nil)

func TestReadinessFollowsNRILifecycle(t *testing.T) {
	state := readiness.NewState()
	p, err := NewPlugin(Config{
		MapDir:    t.TempDir(),
		Log:       slog.New(slog.NewTextHandler(io.Discard, nil)),
		Readiness: state,
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}

	if state.Ready() {
		t.Fatal("plugin should start not-ready")
	}

	// Synchronize is only invoked after successful NRI registration, so it is
	// the ready-flip point.
	if _, err := p.Synchronize(context.Background(), nil, nil); err != nil {
		t.Fatalf("Synchronize returned error: %v", err)
	}
	if !state.Ready() {
		t.Error("expected ready after Synchronize")
	}

	// Shutdown means the runtime is disconnecting; readiness must drop.
	p.Shutdown(context.Background())
	if state.Ready() {
		t.Error("expected not-ready after Shutdown")
	}
}

func TestReadinessNilStateIsSafe(t *testing.T) {
	p, err := NewPlugin(Config{
		MapDir: t.TempDir(),
		Log:    slog.New(slog.NewTextHandler(io.Discard, nil)),
	}, nil)
	if err != nil {
		t.Fatalf("NewPlugin: %v", err)
	}

	// Without a configured readiness state the callbacks must not panic.
	if _, err := p.Synchronize(context.Background(), nil, nil); err != nil {
		t.Fatalf("Synchronize returned error: %v", err)
	}
	p.Shutdown(context.Background())
}
