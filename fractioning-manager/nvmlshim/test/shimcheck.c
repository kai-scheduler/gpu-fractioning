/* Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * Reports what the shim returns for both memory entry points and for entry
 * points whose forwarding slot could not be resolved.
 *
 * The unresolved calls are the point of this program. A trampoline whose slot
 * is NULL must return NVML_ERROR_FUNCTION_NOT_FOUND; if it jumped through the
 * NULL instead, this process would die of SIGSEGV, so "exits at all" is itself
 * part of the assertion.
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
	nvmlReturn_t (*get_name)(nvmlDevice_t, char *, unsigned int);
	nvmlReturn_t (*get_driver)(char *, unsigned int);
	nvmlDevice_t device = NULL;
	nvmlMemory_t v1;
	nvmlMemory_v2_t v2;
	char buffer[128];

	handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
	if (handle == NULL) {
		fprintf(stderr, "shimcheck: dlopen failed: %s\n", dlerror());
		return 1;
	}

	*(void **)&nvml_init = dlsym(handle, "nvmlInit_v2");
	*(void **)&get_handle = dlsym(handle, "nvmlDeviceGetHandleByIndex_v2");
	*(void **)&mem_v1 = dlsym(handle, "nvmlDeviceGetMemoryInfo");
	*(void **)&mem_v2 = dlsym(handle, "nvmlDeviceGetMemoryInfo_v2");
	*(void **)&get_name = dlsym(handle, "nvmlDeviceGetName");
	*(void **)&get_driver = dlsym(handle, "nvmlSystemGetDriverVersion");

	if (nvml_init == NULL || get_handle == NULL || mem_v1 == NULL || mem_v2 == NULL) {
		fprintf(stderr, "shimcheck: the shim is missing an entry point it must define\n");
		return 1;
	}

	/* These are forwarded, never implemented here, so the shim must export
	 * them whether or not the library behind it does. */
	if (get_name == NULL || get_driver == NULL) {
		fprintf(stderr, "shimcheck: the shim did not export a forwarded entry point\n");
		return 1;
	}

	if (nvml_init() != 0 || get_handle(0, &device) != 0) {
		fprintf(stderr, "shimcheck: initialisation failed\n");
		return 1;
	}

	memset(&v1, 0, sizeof v1);
	printf("v1 rc=%d", mem_v1(device, &v1));
	printf(" total=%llu free=%llu used=%llu\n", v1.total, v1.free, v1.used);

	memset(&v2, 0, sizeof v2);
	v2.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2u << 24));
	printf("v2 rc=%d", mem_v2(device, &v2));
	printf(" total=%llu reserved=%llu free=%llu used=%llu\n",
	       v2.total, v2.reserved, v2.free, v2.used);

	memset(buffer, 0, sizeof buffer);
	printf("unresolved nvmlDeviceGetName rc=%d\n",
	       get_name(device, buffer, (unsigned int)sizeof buffer));
	printf("unresolved nvmlSystemGetDriverVersion rc=%d\n",
	       get_driver(buffer, (unsigned int)sizeof buffer));

	return 0;
}
