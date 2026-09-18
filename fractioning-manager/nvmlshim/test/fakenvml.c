/* Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * A stand-in driver for the self-test, pointed at with GPU_FRACTIONING_REAL_NVML.
 *
 * It exists to reach the paths a real GPU cannot be made to produce on demand:
 * a driver that is missing most of its entry points, a process list that fails,
 * one that reports NVML_VALUE_NOT_AVAILABLE, and one that claims more memory
 * than the limit allows. Those are the branches where a bug would either crash
 * a GPU workload or silently report a wrong number, and on a healthy GPU host
 * none of them are reachable.
 *
 * It also makes the capping arithmetic checkable against exact figures rather
 * than against whatever happens to be resident on the test machine.
 *
 * Deliberately exports only the handful of entry points the shim itself calls.
 * Everything else the shim forwards resolves to nothing, which is exactly the
 * unresolved-slot case the trampolines have to survive.
 */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef void *nvmlDevice_t;

#define NVML_SUCCESS                  0
#define NVML_ERROR_NOT_SUPPORTED      3
#define NVML_ERROR_INSUFFICIENT_SIZE  7

/* Round numbers so the shell can assert on them without arithmetic. */
#define FAKE_TOTAL    (100ULL * 1024 * 1024 * 1024)
#define FAKE_RESERVED (1ULL * 1024 * 1024 * 1024)
#define FAKE_USED     (80ULL * 1024 * 1024 * 1024)

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
	unsigned int gpuInstanceId;
	unsigned int computeInstanceId;
} nvmlProcessInfo_v2_t;

static unsigned long long env_ull(const char *name, unsigned long long fallback)
{
	const char *value = getenv(name);

	return value != NULL && *value != '\0' ? strtoull(value, NULL, 0) : fallback;
}

nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index);
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory);
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *count,
						    nvmlProcessInfo_v2_t *infos);

nvmlReturn_t nvmlInit_v2(void)
{
	return NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown(void)
{
	return NVML_SUCCESS;
}

/* Handles are index + 1 so that a NULL handle is never a valid device. */
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
	if (device == NULL)
		return NVML_ERROR_NOT_SUPPORTED;
	*device = (nvmlDevice_t)(uintptr_t)(index + 1);
	return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index)
{
	if (index == NULL || device == NULL)
		return NVML_ERROR_NOT_SUPPORTED;

	/* Lets the test drive the shim's "cannot resolve an index" branch. */
	if (env_ull("FAKE_NVML_NO_INDEX", 0) != 0)
		return NVML_ERROR_NOT_SUPPORTED;

	*index = (unsigned int)((uintptr_t)device - 1);
	return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
	(void)device;
	if (memory == NULL)
		return NVML_ERROR_NOT_SUPPORTED;

	memory->total = FAKE_TOTAL;
	memory->used = FAKE_USED;
	memory->free = FAKE_TOTAL - FAKE_USED;
	return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory)
{
	(void)device;
	if (memory == NULL)
		return NVML_ERROR_NOT_SUPPORTED;

	memory->total = FAKE_TOTAL;
	memory->reserved = FAKE_RESERVED;
	memory->used = FAKE_USED;
	memory->free = FAKE_TOTAL - FAKE_USED;
	return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *count,
						    nvmlProcessInfo_v2_t *infos)
{
	unsigned int wanted = (unsigned int)env_ull("FAKE_NVML_PROC_COUNT", 1);
	unsigned long long per_process = env_ull("FAKE_NVML_PROC_BYTES", 1024ULL * 1024 * 1024);
	unsigned int i;

	(void)device;
	if (count == NULL)
		return NVML_ERROR_NOT_SUPPORTED;

	if (env_ull("FAKE_NVML_PROC_FAIL", 0) != 0)
		return NVML_ERROR_NOT_SUPPORTED;

	if (wanted == 0) {
		*count = 0;
		return NVML_SUCCESS;
	}

	if (infos == NULL || *count < wanted) {
		*count = wanted;
		return NVML_ERROR_INSUFFICIENT_SIZE;
	}

	for (i = 0; i < wanted; i++) {
		infos[i].pid = 1000 + i;
		infos[i].usedGpuMemory = per_process;
		infos[i].gpuInstanceId = 0xFFFFFFFFu;
		infos[i].computeInstanceId = 0xFFFFFFFFu;
	}

	/* One entry with every bit set, which the shim must skip rather than
	 * add; adding it would overstate usage by 16 exabytes. */
	if (env_ull("FAKE_NVML_PROC_NOT_AVAILABLE", 0) != 0)
		infos[0].usedGpuMemory = 0xFFFFFFFFFFFFFFFFULL;

	*count = wanted;
	return NVML_SUCCESS;
}
