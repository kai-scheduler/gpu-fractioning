/* Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * A shadow libnvidia-ml.so.1 that gives a container a capped, container-scoped
 * view of GPU memory, and forwards everything else to the driver untouched.
 *
 * MPS enforces CUDA_MPS_PINNED_DEVICE_MEM_LIMIT at the CUDA layer, but NVML
 * does not go through MPS: nvidia-smi, pynvml and DCGM all read the raw
 * device. Measured in a pod with a 48431 MiB limit on a 97887 MiB card sharing
 * the GPU with one other pod, nvidia-smi reported total=97887 (the card, not
 * the limit) and used=82662 (both tenants, so the neighbour's footprint leaked
 * across the pod boundary). Capping has to happen in NVML because that is
 * where the wrong numbers are produced.
 *
 * This is loaded by being first on the library search path under the driver's
 * own SONAME, not by LD_PRELOAD. LD_PRELOAD cannot work: nvidia-smi does not
 * link libnvidia-ml.so at all -- readelf -d shows only libpthread, libm, libdl,
 * libc and librt -- it dlopens NVML and resolves each entry point with dlsym on
 * the resulting handle, and dlsym on a handle searches that object, never the
 * preload list. Hooks installed by LD_PRELOAD are simply not consulted.
 *
 * Because the shim is installed under the driver's SONAME it cannot declare a
 * DT_NEEDED dependency on the driver -- that would resolve straight back to
 * itself. It dlopens the driver by absolute path instead, and checks at every
 * step that the file it is about to open is not this library.
 *
 * Fail-open is the contract, not a nicety. This library sits in front of every
 * GPU process in the container, so every failure path here -- driver not found,
 * environment absent or malformed, symbol missing, process query failing --
 * degrades to reporting the real device rather than to an error or a crash. A
 * bug in a reporting shim must not be able to take down a GPU workload.
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <glob.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "nvmlshim.h"

#define ENV_ENABLE    "GPU_FRACTIONING_NVML_SHIM"
#define ENV_DEBUG     "GPU_FRACTIONING_NVML_SHIM_DEBUG"
#define ENV_REAL_NVML "GPU_FRACTIONING_REAL_NVML"
#define ENV_MEM_LIMIT "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"

#define NVML_SONAME "libnvidia-ml.so.1"

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* More devices than any single container is given a distinct MPS limit for.
 * The cap is on the parse, so an absurd environment is rejected rather than
 * truncated to something that would silently apply the wrong limit. */
#define MAX_DEVICE_CAPS 32

/* An upper bound on the compute processes the shim will read back. Past this
 * the list is not plausibly one container's, and allocating for it is a worse
 * outcome than reporting the real device. */
#define MAX_PROCESS_ENTRIES 4096

/*
 * ISO C provides no conversion between object and function pointers; POSIX
 * requires dlsym's result to be usable as one, and documents the assignment
 * through a void ** below as the way to spell it without a diagnostic.
 */
#define RESOLVE_FN(slot, handle, name) (*(void **)&(slot) = dlsym((handle), (name)))

struct file_identity {
	dev_t dev;
	ino_t ino;
	int valid;
};

typedef nvmlReturn_t (*proc_query_fn)(nvmlDevice_t, unsigned int *, void *);

struct proc_query {
	proc_query_fn query;
	size_t stride;
	const char *name;
};

struct device_cap {
	/* The device ordinal the limit is keyed by, or -1 when the key was not
	 * an ordinal (MPS also accepts a UUID). */
	long index;
	unsigned long long bytes;
};

/*
 * All of this is written once, by the constructor, before any caller can have
 * reached a symbol of ours: the loader runs constructors to completion before
 * dlopen returns the handle the caller then dlsyms from. Nothing here is
 * mutated afterwards, so the overrides need no locking.
 */
static int debug_enabled;
static int caps_active;
static void *real_handle;
static const void *self_base;
static struct file_identity self_identity;

static nvmlReturn_t (*real_memory_info)(nvmlDevice_t, nvmlMemory_t *);
static nvmlReturn_t (*real_memory_info_v2)(nvmlDevice_t, nvmlMemory_v2_t *);
static nvmlReturn_t (*real_device_get_index)(nvmlDevice_t, unsigned int *);

static struct proc_query proc_queries[3];
static unsigned int proc_query_count;

static struct device_cap device_caps[MAX_DEVICE_CAPS];
static unsigned int device_cap_count;

static void nvmlshim_init(void) __attribute__((constructor));

/* ------------------------------------------------------------------------- */
/* Diagnostics                                                               */
/* ------------------------------------------------------------------------- */

static void shim_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/*
 * Formats into one buffer and emits it with a single fprintf so a line from one
 * process cannot end up interleaved into a line from another; this is on the
 * path of every GPU process in the container and garbled output would be worse
 * than none. Silent unless ENV_DEBUG is set, for the same reason.
 */
static void shim_log(const char *fmt, ...)
{
	char message[512];
	va_list ap;

	if (!debug_enabled)
		return;

	va_start(ap, fmt);
	(void)vsnprintf(message, sizeof message, fmt, ap);
	va_end(ap);

	(void)fprintf(stderr, "nvml-shim[%ld]: %s\n", (long)getpid(), message);
}

static int env_equals(const char *name, const char *want)
{
	const char *value = getenv(name);

	return value != NULL && strcmp(value, want) == 0;
}

/* ------------------------------------------------------------------------- */
/* Finding the real driver without finding ourselves                         */
/* ------------------------------------------------------------------------- */

/*
 * Directories a driver may be installed in, most specific first.
 *
 * /usr/lib/x86_64-linux-gnu is the one that matters on the Debian- and
 * Ubuntu-based CUDA images this ships into, and it is notably absent from
 * nvidia-smi's own hardcoded list. /usr/local/nvidia/{lib,lib64} are here
 * because the NVIDIA container runtime puts them on LD_LIBRARY_PATH inside GPU
 * containers, so some images stage the driver there instead.
 *
 * The patterns deliberately ask for a version-suffixed file. They do also match
 * the bare SONAME (the '*' can match nothing), which is exactly the name this
 * shim is installed under -- see the two-pass search in open_real_nvml.
 */
static const char *const nvml_search_globs[] = {
	"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*[0-9]",
	"/usr/lib64/libnvidia-ml.so.*[0-9]",
	"/usr/lib64/nvidia/libnvidia-ml.so.*[0-9]",
	"/usr/lib/libnvidia-ml.so.*[0-9]",
	"/usr/lib/nvidia/libnvidia-ml.so.*[0-9]",
	"/usr/local/nvidia/lib64/libnvidia-ml.so.*[0-9]",
	"/usr/local/nvidia/lib/libnvidia-ml.so.*[0-9]",
};

static int stat_identity(const char *path, struct file_identity *out)
{
	struct stat st;

	/* stat, not lstat: the driver is normally reached through a symlink and
	 * it is the target's identity that decides whether it is us. */
	if (stat(path, &st) != 0)
		return 0;

	out->dev = st.st_dev;
	out->ino = st.st_ino;
	out->valid = 1;
	return 1;
}

static void locate_self(void)
{
	Dl_info info;

	/*
	 * Taken from a static function's address on purpose. An exported nvml*
	 * symbol would be resolved through the GOT and is subject to
	 * interposition, so it can name a different object than this one;
	 * nvmlshim_init has internal linkage and can only be in this mapping.
	 */
	if (dladdr((const void *)(uintptr_t)&nvmlshim_init, &info) == 0) {
		shim_log("dladdr could not locate this library; relying on the post-dlopen identity check");
		return;
	}

	self_base = info.dli_fbase;
	if (info.dli_fname != NULL && stat_identity(info.dli_fname, &self_identity))
		shim_log("this shim is %s", info.dli_fname);
}

static int is_self(const char *path)
{
	struct file_identity id;

	memset(&id, 0, sizeof id);
	if (!self_identity.valid || !stat_identity(path, &id))
		return 0;

	return id.dev == self_identity.dev && id.ino == self_identity.ino;
}

static int is_soname_alias(const char *path)
{
	const char *base = strrchr(path, '/');

	return strcmp(base != NULL ? base + 1 : path, NVML_SONAME) == 0;
}

/*
 * Opens path if it is a real NVML and is not this library, else returns NULL.
 */
static void *try_real_nvml(const char *path)
{
	void *handle;
	void *probe;
	Dl_info info;

	if (is_self(path)) {
		shim_log("skipping %s: it is this shim", path);
		return NULL;
	}

	handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
	if (handle == NULL) {
		shim_log("dlopen %s failed: %s", path, dlerror());
		return NULL;
	}

	/* Matching the glob does not make a file NVML. Require the entry point
	 * the shim cannot function without before accepting the handle. */
	probe = dlsym(handle, "nvmlDeviceGetMemoryInfo");
	if (probe == NULL) {
		shim_log("rejecting %s: exports no nvmlDeviceGetMemoryInfo", path);
		(void)dlclose(handle);
		return NULL;
	}

	/*
	 * The stat comparison above is the cheap check and it can come up empty
	 * -- a path we lack permission to stat, or one dladdr never gave us --
	 * in which case is_self conservatively answers no. dlopen would then
	 * hand back this library's own already-loaded mapping, and the base
	 * address is how that is caught. Recursing into our own overrides would
	 * not merely double-cap, it would never reach a driver at all.
	 */
	if (self_base != NULL && dladdr(probe, &info) != 0 && info.dli_fbase == self_base) {
		shim_log("rejecting %s: resolves back into this shim", path);
		(void)dlclose(handle);
		return NULL;
	}

	shim_log("real NVML resolved to %s", path);
	return handle;
}

/* Tries each colon-separated element of an explicit override in turn. */
static void *try_path_list(const char *spec)
{
	const char *cursor = spec;
	char path[PATH_MAX];

	while (*cursor != '\0') {
		const char *separator = strchr(cursor, ':');
		size_t length = separator != NULL ? (size_t)(separator - cursor) : strlen(cursor);

		if (length > 0 && length < sizeof path) {
			void *handle;

			memcpy(path, cursor, length);
			path[length] = '\0';

			handle = try_real_nvml(path);
			if (handle != NULL)
				return handle;
		} else if (length >= sizeof path) {
			shim_log("ignoring an over-long path in %s", ENV_REAL_NVML);
		}

		if (separator == NULL)
			break;
		cursor = separator + 1;
	}

	return NULL;
}

static void *open_real_nvml(void)
{
	const char *override = getenv(ENV_REAL_NVML);
	glob_t matches;
	void *handle = NULL;
	int flags = 0;
	int matched = 0;
	size_t i;
	int pass;

	if (override != NULL && *override != '\0') {
		handle = try_path_list(override);
		if (handle != NULL)
			return handle;
		shim_log("%s yielded no usable NVML; falling back to the search path", ENV_REAL_NVML);
	}

	memset(&matches, 0, sizeof matches);
	for (i = 0; i < ARRAY_LEN(nvml_search_globs); i++) {
		int rc = glob(nvml_search_globs[i], flags, NULL, &matches);

		/* GLOB_APPEND is only defined after a call that found something,
		 * so it is set from the first success onwards rather than up
		 * front. A pattern that matches nothing leaves the accumulated
		 * results alone. */
		if (rc == 0) {
			flags |= GLOB_APPEND;
			matched = 1;
		} else if (rc != GLOB_NOMATCH) {
			shim_log("glob %s failed (%d); continuing", nvml_search_globs[i], rc);
		}
	}

	if (!matched)
		shim_log("no libnvidia-ml found on the search path");

	/*
	 * Two passes, version-suffixed names first. This shim is installed as
	 * the bare SONAME, and on Debian-style installs the driver's own
	 * libnvidia-ml.so.1 is a symlink to the versioned file -- so a match on
	 * "libnvidia-ml.so.1" is at least as likely to be this library as the
	 * driver, while "libnvidia-ml.so.615.71.09" can only be the driver.
	 * The alias is still tried, last, because an image that ships only the
	 * SONAME is possible and the identity checks make trying it safe.
	 */
	for (pass = 0; pass < 2 && handle == NULL; pass++) {
		const int want_alias = (pass == 1);

		for (i = 0; i < matches.gl_pathc && handle == NULL; i++) {
			const char *path = matches.gl_pathv[i];

			if (is_soname_alias(path) != want_alias)
				continue;
			handle = try_real_nvml(path);
		}
	}

	globfree(&matches);
	return handle;
}

/* ------------------------------------------------------------------------- */
/* CUDA_MPS_PINNED_DEVICE_MEM_LIMIT                                          */
/* ------------------------------------------------------------------------- */

static const char *skip_spaces(const char *p)
{
	while (*p == ' ' || *p == '\t')
		p++;
	return p;
}

static const char *trim_trailing(const char *begin, const char *end)
{
	while (end > begin && (end[-1] == ' ' || end[-1] == '\t'))
		end--;
	return end;
}

static int parse_ull(const char *begin, const char *end, unsigned long long *out)
{
	unsigned long long value = 0;

	if (begin == end)
		return 0;

	for (; begin < end; begin++) {
		unsigned int digit;

		if (*begin < '0' || *begin > '9')
			return 0;

		digit = (unsigned int)(*begin - '0');
		if (value > (ULLONG_MAX - digit) / 10)
			return 0;
		value = value * 10 + digit;
	}

	*out = value;
	return 1;
}

/* Parses an MPS size such as 48431M, 2G, 1048576 or 512MB. */
static int parse_size(const char *begin, const char *end, unsigned long long *out)
{
	unsigned long long value;
	unsigned int shift = 0;

	if (end > begin && (end[-1] == 'B' || end[-1] == 'b'))
		end--;

	if (end > begin) {
		switch (end[-1]) {
		case 'K': case 'k': shift = 10; end--; break;
		case 'M': case 'm': shift = 20; end--; break;
		case 'G': case 'g': shift = 30; end--; break;
		default: break;
		}
	}

	if (!parse_ull(begin, end, &value))
		return 0;

	/*
	 * A zero limit is treated as malformed rather than honoured. Reporting a
	 * GPU with zero total memory would read to a scheduler or an autoscaler
	 * as a broken device, which is a louder and more damaging lie than the
	 * uncapped number this rejection falls back to.
	 */
	if (value == 0)
		return 0;

	if (value > (ULLONG_MAX >> shift))
		return 0;

	*out = value << shift;
	return 1;
}

/*
 * Parses the whole comma-separated <device>=<size> list. All or nothing: a
 * single malformed entry rejects the lot, because a partial parse would apply
 * a limit to some devices and not others with nothing to show for it.
 */
static int parse_cap_spec(const char *spec)
{
	const char *cursor = spec;

	device_cap_count = 0;

	while (*cursor != '\0') {
		const char *key;
		const char *key_end;
		const char *value;
		const char *value_end;
		struct device_cap cap;
		unsigned long long ordinal;

		key = skip_spaces(cursor);
		cursor = key;
		while (*cursor != '\0' && *cursor != '=' && *cursor != ',')
			cursor++;
		if (*cursor != '=')
			return 0;
		key_end = trim_trailing(key, cursor);
		cursor++;

		value = skip_spaces(cursor);
		cursor = value;
		while (*cursor != '\0' && *cursor != ',')
			cursor++;
		value_end = trim_trailing(value, cursor);
		if (*cursor == ',')
			cursor++;

		if (!parse_size(value, value_end, &cap.bytes))
			return 0;

		/* MPS keys a limit by device ordinal or by UUID. A UUID is kept
		 * with an index of -1 rather than rejected: it cannot be matched
		 * to a handle, but the single-entry fallback in
		 * lookup_device_cap still makes it usable on the one-GPU
		 * containers this is deployed to. */
		cap.index = -1;
		if (parse_ull(key, key_end, &ordinal) && ordinal <= LONG_MAX)
			cap.index = (long)ordinal;

		if (device_cap_count >= MAX_DEVICE_CAPS)
			return 0;
		device_caps[device_cap_count++] = cap;
	}

	return device_cap_count > 0;
}

static int lookup_device_cap(nvmlDevice_t device, unsigned long long *out)
{
	unsigned int index;
	unsigned int i;

	if (real_device_get_index != NULL &&
	    real_device_get_index(device, &index) == NVML_SUCCESS) {
		for (i = 0; i < device_cap_count; i++) {
			if (device_caps[i].index >= 0 &&
			    (unsigned long long)device_caps[i].index == index) {
				*out = device_caps[i].bytes;
				return 1;
			}
		}
		shim_log("no limit configured for device index %u; reporting the real device", index);
		return 0;
	}

	/*
	 * Without an index there is nothing to match a keyed entry against, so
	 * only the unambiguous case can be honoured. This also absorbs the
	 * keyed-by-UUID case, and the one worth knowing about: MPS numbers
	 * devices as the control daemon sees them, which need not be how NVML
	 * numbers them inside the container.
	 */
	if (device_cap_count == 1) {
		*out = device_caps[0].bytes;
		return 1;
	}

	shim_log("no device index available and %u limits configured; reporting the real device",
		 device_cap_count);
	return 0;
}

/* ------------------------------------------------------------------------- */
/* This container's own usage                                                */
/* ------------------------------------------------------------------------- */

/*
 * Sums usedGpuMemory over one version of the running-compute-process list.
 * Returns 1 and the total on success, 0 if the caller should not trust it.
 */
static int sum_with_query(const struct proc_query *variant, nvmlDevice_t device,
			  unsigned long long *out)
{
	unsigned int count = 0;
	nvmlReturn_t rc;
	int attempt;

	/* The NULL probe asks how many entries there are. SUCCESS here means
	 * there are none, which is a real answer of zero, not a failure. */
	rc = variant->query(device, &count, NULL);
	if (rc == NVML_SUCCESS) {
		*out = 0;
		return 1;
	}
	if (rc != NVML_ERROR_INSUFFICIENT_SIZE || count == 0)
		return 0;

	/* A process can start between the sizing call and the fetch, and the
	 * driver answers that with INSUFFICIENT_SIZE and a larger count.
	 * Bounded retries, because an unbounded loop here would hang every
	 * nvidia-smi on the node. */
	for (attempt = 0; attempt < 3; attempt++) {
		unsigned char *entries;
		unsigned int capacity = count;
		unsigned long long total = 0;
		unsigned int i;

		if (count > MAX_PROCESS_ENTRIES) {
			shim_log("%s reports %u processes, more than this shim will read",
				 variant->name, count);
			return 0;
		}

		entries = calloc(capacity, variant->stride);
		if (entries == NULL)
			return 0;

		rc = variant->query(device, &capacity, entries);
		if (rc != NVML_SUCCESS) {
			free(entries);
			if (rc != NVML_ERROR_INSUFFICIENT_SIZE || capacity <= count)
				return 0;
			count = capacity;
			continue;
		}

		for (i = 0; i < capacity; i++) {
			unsigned long long used;

			memcpy(&used,
			       entries + (size_t)i * variant->stride +
				       offsetof(nvmlProcessInfo_v1_t, usedGpuMemory),
			       sizeof used);

			/* Not zero: NVML_VALUE_NOT_AVAILABLE is every bit set,
			 * so adding it would overstate usage by 16 exabytes. */
			if (used == NVML_VALUE_NOT_AVAILABLE_ULL)
				continue;
			if (total > ULLONG_MAX - used) {
				free(entries);
				return 0;
			}
			total += used;
		}

		free(entries);
		*out = total;
		shim_log("%s: %u process(es), %llu bytes", variant->name, capacity, total);
		return 1;
	}

	return 0;
}

/*
 * The container's own GPU memory footprint.
 *
 * Inside the container this list is already scoped to the container's own
 * processes -- verified in a pod, where it returned exactly one entry, our own
 * process, with a namespaced PID. The PID namespace does the scoping for us, so
 * summing it needs no access to the MPS control socket.
 */
static int container_used_bytes(nvmlDevice_t device, unsigned long long *out)
{
	unsigned int i;

	for (i = 0; i < proc_query_count; i++) {
		if (sum_with_query(&proc_queries[i], device, out))
			return 1;
	}

	return 0;
}

/* ------------------------------------------------------------------------- */
/* Capping                                                                   */
/* ------------------------------------------------------------------------- */

/*
 * Rewrites a memory triple in place. Returns 1 if a limit was applied and 0 if
 * the caller should report what the driver said.
 */
static int apply_cap(nvmlDevice_t device, unsigned long long *total,
		     unsigned long long *used, unsigned long long *available)
{
	unsigned long long cap;
	unsigned long long owned;

	if (!caps_active)
		return 0;
	if (!lookup_device_cap(device, &cap))
		return 0;

	if (container_used_bytes(device, &owned)) {
		if (owned > cap) {
			/*
			 * MPS refuses allocations past the limit, so a container
			 * cannot really own more than its cap; a larger sum
			 * means the list was not container-scoped after all
			 * (running on the host, or a pod sharing the host PID
			 * namespace). Clamping keeps the triple self-consistent:
			 * every consumer that computes total - used would
			 * otherwise underflow into an enormous free.
			 */
			shim_log("summed usage %llu exceeds the %llu limit, so the process list "
				 "was not container-scoped; clamping",
				 owned, cap);
			owned = cap;
		}
	} else {
		/* Fail open on the part that failed, not on the whole answer:
		 * the driver's used is a real number, just a device-wide one. */
		shim_log("no usable process list; reporting the driver's used value under the limit");
		owned = *used > cap ? cap : *used;
	}

	*total = cap;
	*used = owned;
	*available = cap - owned;
	return 1;
}

/* ------------------------------------------------------------------------- */
/* Overridden entry points                                                   */
/* ------------------------------------------------------------------------- */

NVMLSHIM_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
NVMLSHIM_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory);

/*
 * v1's used is documented as reserved plus allocated, whereas the capped number
 * is the sum over this container's processes and so excludes the driver's own
 * reserve. The difference is the couple of hundred MiB the driver sets aside
 * for bookkeeping, and attributing it to one tenant of a shared GPU would be
 * the less accurate choice.
 */
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
	nvmlReturn_t rc;

	if (real_memory_info == NULL)
		return NVML_ERROR_FUNCTION_NOT_FOUND;

	rc = real_memory_info(device, memory);
	if (rc != NVML_SUCCESS || memory == NULL)
		return rc;

	(void)apply_cap(device, &memory->total, &memory->used, &memory->free);
	return rc;
}

/*
 * reserved is passed through untouched: it describes the driver's own claim on
 * the physical device, which the container's limit does not change.
 */
nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory)
{
	nvmlReturn_t rc;

	if (real_memory_info_v2 == NULL)
		return NVML_ERROR_FUNCTION_NOT_FOUND;

	/* Called first so the driver can validate the caller's version tag; on
	 * a mismatch it fails and nothing of the caller's struct is touched. */
	rc = real_memory_info_v2(device, memory);
	if (rc != NVML_SUCCESS || memory == NULL)
		return rc;

	(void)apply_cap(device, &memory->total, &memory->used, &memory->free);
	return rc;
}

/* ------------------------------------------------------------------------- */
/* Startup                                                                   */
/* ------------------------------------------------------------------------- */

/* Appends one version of the process query, if this driver exports it. */
static void register_proc_query(const char *name, size_t stride)
{
	proc_query_fn fn;

	RESOLVE_FN(fn, real_handle, name);
	if (fn == NULL || proc_query_count >= ARRAY_LEN(proc_queries))
		return;

	proc_queries[proc_query_count].query = fn;
	proc_queries[proc_query_count].stride = stride;
	proc_queries[proc_query_count].name = name;
	proc_query_count++;
}

static void resolve_own_calls(void)
{
	RESOLVE_FN(real_memory_info, real_handle, "nvmlDeviceGetMemoryInfo");
	RESOLVE_FN(real_memory_info_v2, real_handle, "nvmlDeviceGetMemoryInfo_v2");
	RESOLVE_FN(real_device_get_index, real_handle, "nvmlDeviceGetIndex");

	/* Newest first. _v2 and _v3 share the nvmlProcessInfo_t layout; only v1
	 * is narrower, and all three put usedGpuMemory at the same offset, which
	 * is what lets one reader walk any of them given the stride. */
	register_proc_query("nvmlDeviceGetComputeRunningProcesses_v3", sizeof(nvmlProcessInfo_v2_t));
	register_proc_query("nvmlDeviceGetComputeRunningProcesses_v2", sizeof(nvmlProcessInfo_v2_t));
	register_proc_query("nvmlDeviceGetComputeRunningProcesses", sizeof(nvmlProcessInfo_v1_t));
}

static void fill_trampoline_slots(void)
{
	const char *name = nvmlshim_symbol_names;
	unsigned long missing = 0;
	unsigned long i;

	for (i = 0; i < nvmlshim_symbol_count; i++) {
		nvmlshim_slots[i] = dlsym(real_handle, name);
		if (nvmlshim_slots[i] == NULL) {
			missing++;
			shim_log("driver does not export %s; that trampoline will return "
				 "NVML_ERROR_FUNCTION_NOT_FOUND", name);
		}
		name += strlen(name) + 1;
	}

	shim_log("forwarding %lu symbols (%lu unresolved)", nvmlshim_symbol_count, missing);
}

static void configure_caps(void)
{
	const char *spec;

	if (env_equals(ENV_ENABLE, "0")) {
		shim_log("%s=0: forwarding only, nothing will be capped", ENV_ENABLE);
		return;
	}

	spec = getenv(ENV_MEM_LIMIT);
	if (spec == NULL || *spec == '\0') {
		shim_log("%s is not set; reporting the real device", ENV_MEM_LIMIT);
		return;
	}

	if (!parse_cap_spec(spec)) {
		device_cap_count = 0;
		shim_log("could not parse %s=\"%.200s\"; reporting the real device",
			 ENV_MEM_LIMIT, spec);
		return;
	}

	caps_active = 1;
	shim_log("%u device limit(s) parsed from %s", device_cap_count, ENV_MEM_LIMIT);
}

/*
 * Runs while the caller's dlopen of libnvidia-ml.so.1 is still in progress, so
 * everything below has to be finished before that dlopen returns -- the
 * trampolines have no initialisation hook of their own by design, since adding
 * one would mean touching registers they are supposed to leave alone.
 */
static void nvmlshim_init(void)
{
	debug_enabled = env_equals(ENV_DEBUG, "1");
	locate_self();

	real_handle = open_real_nvml();
	if (real_handle == NULL) {
		/* Nothing to fall back to: without the driver there is no
		 * behaviour to be transparent about. Every slot stays NULL and
		 * every call reports the absence rather than crashing. */
		shim_log("no real NVML available; all calls will return NVML_ERROR_FUNCTION_NOT_FOUND");
		return;
	}

	fill_trampoline_slots();
	resolve_own_calls();
	configure_caps();
}
