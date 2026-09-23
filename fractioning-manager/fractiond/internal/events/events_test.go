// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package events

import (
	"io"
	"log/slog"
	"sync"
	"testing"

	"github.com/kai-scheduler/gpu-fractioning/fractioning-manager/common/mapping/store"
)

type fakeWriter struct {
	mu       sync.Mutex
	upserts  []store.ContainerInfo
	deletes  []string
	replaces [][]store.ContainerInfo
}

func (w *fakeWriter) Upsert(info store.ContainerInfo) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.upserts = append(w.upserts, info)
}

func (w *fakeWriter) Delete(id string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.deletes = append(w.deletes, id)
}

func (w *fakeWriter) Replace(infos []store.ContainerInfo) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.replaces = append(w.replaces, infos)
}

func newTestProcessor(w store.Writer, opts Options) *Processor {
	return NewProcessor(w, slog.New(slog.NewTextHandler(io.Discard, nil)), opts)
}

// upsertOf returns an adapter that yields info unconditionally.
func upsertOf(info store.ContainerInfo) adapter {
	return func() (store.ContainerInfo, bool) { return info, true }
}

func TestProcessorAppliesEventsInOrder(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "a"}))
	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "b"}))
	p.Delete("a")
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.upserts) != 2 || w.upserts[0].ContainerID != "a" || w.upserts[1].ContainerID != "b" {
		t.Fatalf("unexpected upserts: %#v", w.upserts)
	}
	if len(w.deletes) != 1 || w.deletes[0] != "a" {
		t.Fatalf("unexpected deletes: %#v", w.deletes)
	}
}

func TestProcessorDropsUpsertWhenAdapterReturnsNotOK(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	p.Upsert(func() (store.ContainerInfo, bool) { return store.ContainerInfo{ContainerID: "skip"}, false })
	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "keep"}))
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.upserts) != 1 || w.upserts[0].ContainerID != "keep" {
		t.Fatalf("expected only the ok adapter to be written, got %#v", w.upserts)
	}
}

func TestProcessorReplace(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	p.Synchronize(func() []store.ContainerInfo {
		return []store.ContainerInfo{{ContainerID: "x"}, {ContainerID: "y"}}
	})
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.replaces) != 1 || len(w.replaces[0]) != 2 {
		t.Fatalf("unexpected replaces: %#v", w.replaces)
	}
}

func TestProcessorFlushIsBarrier(t *testing.T) {
	w := &fakeWriter{}
	p := newTestProcessor(w, Options{})

	for range 50 {
		p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "c"}))
	}
	p.Flush()

	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.upserts) != 50 {
		t.Fatalf("expected 50 upserts applied before Flush returned, got %d", len(w.upserts))
	}
}

func TestProcessorConsultsLogPredicate(t *testing.T) {
	w := &fakeWriter{}
	var mu sync.Mutex
	calls := 0
	p := newTestProcessor(w, Options{LogEvents: func() bool {
		mu.Lock()
		defer mu.Unlock()
		calls++
		return false
	}})

	p.Upsert(upsertOf(store.ContainerInfo{ContainerID: "a"}))
	p.Flush()

	mu.Lock()
	defer mu.Unlock()
	if calls == 0 {
		t.Fatal("expected the LogEvents predicate to be consulted on apply")
	}
}
