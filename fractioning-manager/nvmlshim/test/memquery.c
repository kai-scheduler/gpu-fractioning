/* Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * Prints both memory-info entry points in bytes, for the self-test.
 *
 * nvidia-smi on driver 615.71.09 only ever calls nvmlDeviceGetMemoryInfo_v2, so
 * testing through nvidia-smi alone would leave the unversioned entry point --
 * the one pynvml and older DCGM builds call -- entirely unexercised. An
 * uncapped v1 is a silent leak of the real device size, which is the whole
 * failure this component exists to prevent.
 *
 * It resolves NVML the way nvidia-smi does, by dlopening the SONAME and reading
 * symbols off the handle, so LD_LIBRARY_PATH decides whether the shim or the
 * driver answers. Reporting in bytes rather than MiB keeps the comparison exact
 * and independent of nvidia-smi's rounding.
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef void *nvmlDevice_t;

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

int main(void)
{
	void *handle;
	nvmlReturn_t (*nvml_init)(void);
	nvmlReturn_t (*get_handle)(unsigned int, nvmlDevice_t *);
	nvmlReturn_t (*mem_v1)(nvmlDevice_t, nvmlMemory_t *);
	nvmlReturn_t (*mem_v2)(nvmlDevice_t, nvmlMemory_v2_t *);
	nvmlDevice_t device = NULL;
	nvmlMemory_t v1;
	nvmlMemory_v2_t v2;
	nvmlReturn_t rc;

	handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
	if (handle == NULL) {
		fprintf(stderr, "memquery: dlopen failed: %s\n", dlerror());
		return 1;
	}

	*(void **)&nvml_init = dlsym(handle, "nvmlInit_v2");
	*(void **)&get_handle = dlsym(handle, "nvmlDeviceGetHandleByIndex_v2");
	*(void **)&mem_v1 = dlsym(handle, "nvmlDeviceGetMemoryInfo");
	*(void **)&mem_v2 = dlsym(handle, "nvmlDeviceGetMemoryInfo_v2");

	if (nvml_init == NULL || get_handle == NULL || mem_v1 == NULL || mem_v2 == NULL) {
		fprintf(stderr, "memquery: NVML is missing an entry point this test needs\n");
		return 1;
	}

	rc = nvml_init();
	if (rc != 0) {
		fprintf(stderr, "memquery: nvmlInit_v2 returned %d\n", rc);
		return 1;
	}

	rc = get_handle(0, &device);
	if (rc != 0) {
		fprintf(stderr, "memquery: nvmlDeviceGetHandleByIndex_v2 returned %d\n", rc);
		return 1;
	}

	memset(&v1, 0, sizeof v1);
	rc = mem_v1(device, &v1);
	if (rc != 0) {
		fprintf(stderr, "memquery: nvmlDeviceGetMemoryInfo returned %d\n", rc);
		return 1;
	}
	printf("v1 total=%llu free=%llu used=%llu\n", v1.total, v1.free, v1.used);

	memset(&v2, 0, sizeof v2);
	/* NVML_STRUCT_VERSION(Memory, 2): the struct size in the low bits, the
	 * version in bits 24 and up. The driver rejects the call outright if
	 * this does not match what it expects. */
	v2.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2u << 24));
	rc = mem_v2(device, &v2);
	if (rc != 0) {
		fprintf(stderr, "memquery: nvmlDeviceGetMemoryInfo_v2 returned %d\n", rc);
		return 1;
	}
	printf("v2 total=%llu reserved=%llu free=%llu used=%llu\n",
	       v2.total, v2.reserved, v2.free, v2.used);

	return 0;
}
