# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the shadow libnvidia-ml.so.1.
#
# Nothing in this build needs a driver or the CUDA toolkit: the entry-point list
# comes from the committed symbols.txt and the NVML structs are restated in
# nvmlshim.h, so the library builds in a container stage on a machine with no
# GPU. gen-trampolines.sh does check the committed list against a driver when it
# finds one, which is why building on a GPU node also catches driver drift.

CC ?= gcc
BUILD_DIR ?= build
SONAME := libnvidia-ml.so.1
LIB := $(BUILD_DIR)/$(SONAME)
GENERATED_ASM := generated/trampolines.S

# The target CPU the compiler will actually emit for, which is what matters
# under cross-compilation; uname only describes the builder.
TARGET_CPU := $(shell $(CC) -dumpmachine 2>/dev/null | cut -d- -f1)
ifeq ($(TARGET_CPU),)
TARGET_CPU := $(shell uname -m)
endif

CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -fPIC -fvisibility=hidden \
          -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes -Wmissing-prototypes
ASFLAGS := -fPIC -g
# --no-undefined so a typo in a libc call fails the link here rather than
# becoming an unresolved symbol inside somebody's GPU workload.
LDFLAGS += -shared -Wl,-soname,$(SONAME) -Wl,--no-undefined -Wl,-z,relro -Wl,-z,now
LDLIBS += -ldl

.PHONY: all test clean verify-symbols regen-symbols

ifeq ($(filter x86_64 amd64,$(TARGET_CPU)),)

# ---------------------------------------------------------------------------
# Unsupported architecture.
#
# Building nothing is the safe outcome, and it is deliberately not an error: the
# image build that carries this library is multi-arch, and failing here would
# break the arm64 image rather than merely leave it without a shim. With no
# shadow library on the search path the loader finds the driver's own
# libnvidia-ml.so.1 and the container sees exactly what it sees today.
# ---------------------------------------------------------------------------

all:
	@echo "nvmlshim: $(TARGET_CPU) is not supported, building nothing."
	@echo "nvmlshim: the forwarders are hand-written x86-64 trampolines; see README.md."
	@echo "nvmlshim: GPU memory reporting on this architecture stays uncapped."

test: all
	@echo "nvmlshim: no tests to run on $(TARGET_CPU)."

verify-symbols regen-symbols:
	@echo "nvmlshim: nothing to do on $(TARGET_CPU)."

clean:
	rm -rf $(BUILD_DIR) generated

else

# ---------------------------------------------------------------------------
# x86-64
# ---------------------------------------------------------------------------

OBJS := $(BUILD_DIR)/nvmlshim.o $(BUILD_DIR)/trampolines.o

all: $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(BUILD_DIR)/nvmlshim.o: nvmlshim.c nvmlshim.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/trampolines.o: $(GENERATED_ASM) | $(BUILD_DIR)
	$(CC) $(ASFLAGS) -c $< -o $@

$(GENERATED_ASM): symbols.txt overrides.txt gen-trampolines.sh
	./gen-trampolines.sh --out $@

$(BUILD_DIR):
	mkdir -p $@

TEST_CFLAGS := -O2 -g -std=c11 -Wall -Wextra
TEST_BINARIES := $(BUILD_DIR)/memquery $(BUILD_DIR)/shimcheck $(BUILD_DIR)/libfakenvml.so

test: $(LIB) $(TEST_BINARIES)
	./test/run-tests.sh

# Ordinary programs, not linked against NVML: they dlopen the SONAME the way
# nvidia-smi does, so LD_LIBRARY_PATH decides which library answers.
$(BUILD_DIR)/memquery: test/memquery.c | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -o $@ $< -ldl

$(BUILD_DIR)/shimcheck: test/shimcheck.c | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -o $@ $< -ldl

# Named libfakenvml.so, not libnvidia-ml.so.*, so it cannot be picked up by the
# shim's own search globs -- the test reaches it only through the explicit
# GPU_FRACTIONING_REAL_NVML override.
$(BUILD_DIR)/libfakenvml.so: test/fakenvml.c | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -fPIC -shared -o $@ $<

# Fails if the driver installed here exports a different set of entry points
# than symbols.txt records. Implied by a normal build on a GPU node; useful on
# its own when qualifying a driver upgrade.
verify-symbols:
	./gen-trampolines.sh --out $(GENERATED_ASM)

regen-symbols:
	./gen-trampolines.sh --update --out $(GENERATED_ASM)

clean:
	rm -rf $(BUILD_DIR) generated

endif
