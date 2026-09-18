/* Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * ABI surface the shim needs from NVML, plus the interface to the generated
 * trampoline table.
 *
 * The NVML types are restated here rather than pulled from the CUDA toolkit's
 * nvml.h because this library is built in a container stage that has neither
 * the toolkit nor a driver, and adding that dependency to get four struct
 * definitions would be a poor trade. The restatement is safe in the way that
 * matters: these layouts are the driver's frozen public ABI, and the
 * _Static_asserts below fail the build if a toolchain ever lays them out
 * differently from what the driver was measured to expect.
 */

#ifndef NVMLSHIM_H
#define NVMLSHIM_H

#include <stddef.h>

#if !defined(__x86_64__)
#error "nvmlshim forwards through hand-written x86-64 trampolines; see README.md for the arch contract"
#endif

#define NVMLSHIM_EXPORT __attribute__((visibility("default")))
#define NVMLSHIM_HIDDEN __attribute__((visibility("hidden")))

/*
 * nvmlReturn_t is an unadorned enum in nvml.h, so it is passed and returned as
 * an int. Only the codes the shim itself produces or tests are named.
 */
typedef int nvmlReturn_t;

#define NVML_SUCCESS                  0
#define NVML_ERROR_INSUFFICIENT_SIZE  7
#define NVML_ERROR_FUNCTION_NOT_FOUND 13

/* An opaque driver-owned handle; the shim only ever passes it back through. */
typedef void *nvmlDevice_t;

/*
 * NVML_VALUE_NOT_AVAILABLE is (-1) in nvml.h, which for the unsigned long long
 * memory fields means every bit set. Entries carrying it are not zero and must
 * not be summed.
 */
#define NVML_VALUE_NOT_AVAILABLE_ULL 0xFFFFFFFFFFFFFFFFULL

typedef struct {
	unsigned long long total;
	unsigned long long free;
	unsigned long long used;
} nvmlMemory_t;

typedef struct {
	unsigned int version;
	unsigned long long total;
	unsigned long long reserved;
	unsigned long long free;
	unsigned long long used;
} nvmlMemory_v2_t;

typedef struct {
	unsigned int pid;
	unsigned long long usedGpuMemory;
} nvmlProcessInfo_v1_t;

/* nvml.h declares nvmlProcessInfo_v2_t and nvmlProcessInfo_t as one typedef
 * pair, so _v2 and _v3 of the process queries share this layout. */
typedef struct {
	unsigned int pid;
	unsigned long long usedGpuMemory;
	unsigned int gpuInstanceId;
	unsigned int computeInstanceId;
} nvmlProcessInfo_v2_t;

/*
 * The driver validates nvmlMemory_v2_t by the version tag the caller stamps
 * into it, and that tag encodes sizeof the struct: NVML_STRUCT_VERSION(Memory,
 * 2) is sizeof(nvmlMemory_v2_t) | (2 << 24), measured as 0x2000028 against
 * driver 615.71.09. A layout change here would therefore not be caught by the
 * driver -- it would be caught by the caller's tag no longer matching, which
 * surfaces as every memory query failing. Catch it at build time instead.
 */
_Static_assert(sizeof(nvmlMemory_t) == 24, "nvmlMemory_t must match the driver ABI");
_Static_assert(sizeof(nvmlMemory_v2_t) == 40, "nvmlMemory_v2_t must match the driver ABI");
_Static_assert(sizeof(nvmlProcessInfo_v1_t) == 16, "nvmlProcessInfo_v1_t must match the driver ABI");
_Static_assert(sizeof(nvmlProcessInfo_v2_t) == 24, "nvmlProcessInfo_v2_t must match the driver ABI");

/*
 * The process-list reader walks v1 and v2 entries with a single field offset
 * and a per-version stride. That is only legitimate while both layouts agree
 * on where usedGpuMemory sits.
 */
_Static_assert(offsetof(nvmlProcessInfo_v1_t, usedGpuMemory) ==
		       offsetof(nvmlProcessInfo_v2_t, usedGpuMemory),
	       "process-list reader assumes one usedGpuMemory offset for both layouts");

/*
 * Emitted by gen-trampolines.sh into generated/trampolines.S.
 *
 * nvmlshim_slots is indexed in lockstep with the NUL-separated names packed
 * into nvmlshim_symbol_names; the constructor is what ties the two together.
 */
extern void *nvmlshim_slots[] NVMLSHIM_HIDDEN;
extern const char nvmlshim_symbol_names[] NVMLSHIM_HIDDEN;
extern const unsigned long nvmlshim_symbol_count NVMLSHIM_HIDDEN;

#endif /* NVMLSHIM_H */
