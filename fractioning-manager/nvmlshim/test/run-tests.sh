#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Exercises the shadow library against the real driver on a GPU host.
#
# Every fail-open assertion is paired with a check that the shim was actually
# loaded. Without that pairing those cases are untrustworthy: "reports the true
# total" is equally the answer when the shim degrades correctly and when it was
# never on the search path at all, so a completely broken shadow mechanism would
# show up as extra passes rather than as failures.

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
build_dir="${BUILD_DIR:-$root/build}"
shim="$build_dir/libnvidia-ml.so.1"
memquery="$build_dir/memquery"

passes=0
failures=0

pass() { printf 'PASS  %s\n' "$*"; passes=$((passes + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }
info() { printf '      %s\n' "$*"; }

check_eq() {
	local what="$1" expected="$2" actual="$3"
	if [[ "$expected" == "$actual" ]]; then
		pass "$what (= $actual)"
	else
		fail "$what: expected '$expected', got '$actual'"
	fi
}

# with_env <shim|bare> [VAR=VALUE ...] -- <command...>
#
# Builds the environment explicitly rather than inheriting it: LD_LIBRARY_PATH
# and every variable this component reads are cleared first, so a value left in
# the caller's shell cannot decide a result.
with_env() {
	local mode="$1"; shift
	local -a env_args=(env
		-u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT
		-u GPU_FRACTIONING_NVML_SHIM
		-u GPU_FRACTIONING_NVML_SHIM_DEBUG
		-u GPU_FRACTIONING_REAL_NVML)

	if [[ "$mode" == shim ]]; then
		env_args+=("LD_LIBRARY_PATH=$build_dir")
	else
		env_args+=(-u LD_LIBRARY_PATH)
	fi

	while [[ $# -gt 0 && "$1" != "--" ]]; do
		env_args+=("$1")
		shift
	done
	shift

	"${env_args[@]}" "$@"
}

# query <shim|bare> <field> [VAR=VALUE ...]
query() {
	local mode="$1" field="$2"; shift 2
	with_env "$mode" "$@" -- nvidia-smi \
		--query-gpu="$field" --format=csv,noheader,nounits -i 0 2>/dev/null
}

# loaded_shim reports whether the shadow library initialised, read from its own
# debug output rather than inferred from the numbers it produced.
#
# The output is captured and matched rather than piped into grep -q: grep -q
# exits on the first match, the SIGPIPE that lands on nvidia-smi becomes the
# left-hand status of the pipeline, and under `set -o pipefail` a successful
# match would then read as a failed check.
loaded_shim() {
	local diagnostics
	diagnostics="$(with_env shim GPU_FRACTIONING_NVML_SHIM_DEBUG=1 "$@" -- nvidia-smi \
		--query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>&1 >/dev/null)"
	[[ "$diagnostics" == *"nvml-shim["* ]]
}

echo "=== nvmlshim self-test ==="
echo

if [[ ! -f "$shim" ]]; then
	echo "FAIL  $shim does not exist; run make first" >&2
	exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
	echo "FAIL  nvidia-smi not found; these tests need a GPU host" >&2
	exit 1
fi
if [[ ! -x "$memquery" ]]; then
	echo "FAIL  $memquery does not exist; run make test" >&2
	exit 1
fi

# --------------------------------------------------------------------------
echo "--- 0. structure ---"

soname="$(readelf -d "$shim" | sed -n 's/.*SONAME).*\[\(.*\)\].*/\1/p')"
check_eq "SONAME" "libnvidia-ml.so.1" "$soname"

# A DT_NEEDED on the driver would resolve straight back to this library: both
# carry the same SONAME and this one is first on the search path.
if readelf -d "$shim" | grep -q 'NEEDED.*libnvidia-ml'; then
	fail "shim declares a DT_NEEDED on libnvidia-ml, which would resolve to itself"
else
	pass "shim has no DT_NEEDED on libnvidia-ml"
fi

expected_exports="$(grep -c '^nvml' "$root/symbols.txt")"
actual_exports="$(nm -D --defined-only "$shim" | awk '$2 == "T" && $3 ~ /^nvml/' | wc -l)"
check_eq "exported entry-point count" "$expected_exports" "$actual_exports"

missing="$(comm -23 \
	<(grep '^nvml' "$root/symbols.txt" | LC_ALL=C sort) \
	<(nm -D --defined-only "$shim" | awk '$2 == "T" && $3 ~ /^nvml/ { print $3 }' | LC_ALL=C sort))"
if [[ -z "$missing" ]]; then
	pass "no entry point from symbols.txt is missing from the shim"
else
	fail "entry points missing from the shim: $(tr '\n' ' ' <<<"$missing")"
fi

echo

# --------------------------------------------------------------------------
echo "--- 1. baseline, without the shim ---"

baseline_total="$(query bare memory.total)"
if [[ "$baseline_total" =~ ^[0-9]+$ ]] && (( baseline_total > 1024 )); then
	pass "nvidia-smi reports memory.total = $baseline_total MiB"
else
	fail "nvidia-smi without the shim reported '$baseline_total'"
fi

echo

# --------------------------------------------------------------------------
echo "--- 2. capped total ---"

if loaded_shim CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=4096M; then
	pass "shadow library loaded (its constructor ran)"
else
	fail "shadow library did NOT load; everything below is meaningless"
fi

check_eq "nvidia-smi memory.total under 0=4096M" "4096" \
	"$(query shim memory.total CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=4096M)"

# Both entry points, in bytes, because nvidia-smi only calls _v2 on this driver.
three_gib=$((3 * 1024 * 1024 * 1024))
if mem_out="$(with_env shim CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=3G -- "$memquery" 2>&1)"; then
	info "$(tr '\n' '|' <<<"$mem_out")"
	check_eq "nvmlDeviceGetMemoryInfo (v1) total under 0=3G" "$three_gib" \
		"$(sed -n 's/^v1 total=\([0-9]*\).*/\1/p' <<<"$mem_out")"
	check_eq "nvmlDeviceGetMemoryInfo_v2 total under 0=3G" "$three_gib" \
		"$(sed -n 's/^v2 total=\([0-9]*\).*/\1/p' <<<"$mem_out")"

	# reserved describes the driver's claim on the physical device and is
	# specified to pass through untouched.
	shim_reserved="$(sed -n 's/.*reserved=\([0-9]*\).*/\1/p' <<<"$mem_out")"
	bare_reserved="$(with_env bare -- "$memquery" 2>/dev/null |
		sed -n 's/.*reserved=\([0-9]*\).*/\1/p')"
	check_eq "v2 reserved passes through unchanged" "$bare_reserved" "$shim_reserved"
else
	fail "memquery failed through the shim: $mem_out"
fi

echo

# --------------------------------------------------------------------------
echo "--- 3. consistency ---"

read -r c_total c_used c_free <<<"$(query shim memory.total,memory.used,memory.free \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=4096M | tr -d ',')"
info "nvidia-smi: total=$c_total used=$c_used free=$c_free (MiB)"

if [[ "$c_total" =~ ^[0-9]+$ && "$c_used" =~ ^[0-9]+$ && "$c_free" =~ ^[0-9]+$ ]]; then
	if (( c_used + c_free == c_total )); then
		pass "memory.free + memory.used == memory.total"
	else
		fail "memory.free + memory.used = $((c_used + c_free)), expected $c_total"
	fi
	if (( c_free <= c_total )); then
		pass "memory.free <= memory.total"
	else
		fail "memory.free ($c_free) exceeds memory.total ($c_total)"
	fi
else
	fail "could not read the capped memory triple"
fi

# The same invariant in bytes, where nvidia-smi's MiB rounding cannot hide a
# one-unit discrepancy.
if mem_out="$(with_env shim CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=3G -- "$memquery" 2>&1)"; then
	for version in v1 v2; do
		line="$(grep "^$version " <<<"$mem_out")"
		b_total="$(sed -n 's/.*total=\([0-9]*\).*/\1/p' <<<"$line")"
		b_free="$(sed -n 's/.*free=\([0-9]*\).*/\1/p' <<<"$line")"
		b_used="$(sed -n 's/.*used=\([0-9]*\).*/\1/p' <<<"$line")"
		if (( b_free + b_used == b_total )); then
			pass "$version free + used == total exactly, in bytes"
		else
			fail "$version free($b_free) + used($b_used) != total($b_total)"
		fi
	done
fi

echo

# --------------------------------------------------------------------------
echo "--- 4. fail open: no limit configured ---"

if loaded_shim; then
	pass "shadow library loaded with no limit set"
else
	fail "shadow library did not load"
fi
check_eq "memory.total with the shim but no limit" "$baseline_total" "$(query shim memory.total)"

echo

# --------------------------------------------------------------------------
echo "--- 5. fail open: malformed limit ---"

for bad in "garbage" "0=" "=4096M" "0=0M" "0=4096Q" "0=1G,junk" "0=99999999999999999999G" "   "; do
	if loaded_shim "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=$bad"; then
		check_eq "limit '$bad' falls back to the real total" "$baseline_total" \
			"$(query shim memory.total "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=$bad")"
	else
		fail "shadow library did not load for limit '$bad'"
	fi
done

echo

# --------------------------------------------------------------------------
echo "--- 6. opt-out ---"

if loaded_shim GPU_FRACTIONING_NVML_SHIM=0 CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=4096M; then
	pass "shadow library loaded with the opt-out set"
else
	fail "shadow library did not load with the opt-out set"
fi
check_eq "GPU_FRACTIONING_NVML_SHIM=0 reports the real total" "$baseline_total" \
	"$(query shim memory.total GPU_FRACTIONING_NVML_SHIM=0 CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=4096M)"

echo

# --------------------------------------------------------------------------
echo "--- 7. forwarding intact ---"

identity_args=(--query-gpu=name,driver_version,uuid --format=csv)
check_eq "name, driver_version and uuid are unchanged through the shim" \
	"$(with_env bare -- nvidia-smi "${identity_args[@]}" 2>/dev/null)" \
	"$(with_env shim CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=4096M -- \
		nvidia-smi "${identity_args[@]}" 2>/dev/null)"

# The full table and the exhaustive dump are the real regression checks: they
# walk a wide swath of the forwarded entry points, any one of which would fault
# or error if its trampoline were wrong.
declare -a forwarding_cases=(
	"the default table|"
	"the exhaustive dump|-q"
	"the compute-process query|--query-compute-apps=pid,used_memory --format=csv,noheader"
	"topology|topo -m"
	"per-process accounting|--query-accounted-apps=pid --format=csv,noheader"
)
for case_spec in "${forwarding_cases[@]}"; do
	label="${case_spec%%|*}"
	args="${case_spec#*|}"
	# shellcheck disable=SC2086
	if with_env shim CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=4096M -- \
		nvidia-smi $args >/dev/null 2>&1; then
		pass "'nvidia-smi $args' exits 0 through the shim ($label)"
	else
		status=$?
		# Compare against the driver: a subcommand this GPU does not
		# support fails identically with and without the shim, and that
		# is not a forwarding regression.
		# shellcheck disable=SC2086
		if with_env bare -- nvidia-smi $args >/dev/null 2>&1; then
			fail "'nvidia-smi $args' failed only through the shim (exit $status)"
		else
			pass "'nvidia-smi $args' fails identically with and without the shim ($label)"
		fi
	fi
done

echo

# --------------------------------------------------------------------------
echo "--- 8. degraded paths, against the synthetic driver ---"

# A healthy GPU host cannot be made to produce a driver with missing entry
# points, a failing process query, or a container over its own limit, so these
# run against test/fakenvml.c instead. It also pins the arithmetic to exact
# figures rather than to whatever happens to be resident on this machine.
gib=$((1024 * 1024 * 1024))
fake_lib="$build_dir/libfakenvml.so"
fake_total=$((100 * gib))
fake_reserved=$((1 * gib))

# fake_check <label> <expected total> <expected used> <expected free> [VAR=VALUE ...]
fake_check() {
	local label="$1" want_total="$2" want_used="$3" want_free="$4"; shift 4
	local out status

	out="$(with_env shim "GPU_FRACTIONING_REAL_NVML=$fake_lib" "$@" -- "$build_dir/shimcheck" 2>&1)"
	status=$?
	if (( status != 0 )); then
		# 139 is SIGSEGV, which is what jumping through an unresolved
		# slot would produce.
		fail "$label: shimcheck exited $status ${out:+($out)}"
		return
	fi

	local version line got_total got_used got_free
	for version in v1 v2; do
		line="$(grep "^$version " <<<"$out")"
		got_total="$(sed -n 's/.*total=\([0-9]*\).*/\1/p' <<<"$line")"
		got_used="$(sed -n 's/.*used=\([0-9]*\).*/\1/p' <<<"$line")"
		got_free="$(sed -n 's/.* free=\([0-9]*\).*/\1/p' <<<"$line")"
		if [[ "$got_total" == "$want_total" && "$got_used" == "$want_used" &&
		      "$got_free" == "$want_free" ]]; then
			pass "$label ($version total=$got_total used=$got_used free=$got_free)"
		else
			fail "$label ($version): expected total=$want_total used=$want_used free=$want_free, got total=$got_total used=$got_used free=$got_free"
		fi
	done
}

# An unresolved forwarding slot must return NVML_ERROR_FUNCTION_NOT_FOUND. The
# synthetic driver exports only what the shim itself calls, so every other
# trampoline has a NULL slot; reaching this assertion at all proves none of
# them jumped through it.
unresolved_out="$(with_env shim "GPU_FRACTIONING_REAL_NVML=$fake_lib" \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G -- "$build_dir/shimcheck" 2>&1)"
unresolved_status=$?
if (( unresolved_status != 0 )); then
	fail "shimcheck crashed against a driver missing most entry points (exit $unresolved_status)"
else
	pass "a driver missing 427 of 429 forwarded entry points does not crash the caller"
	for symbol in nvmlDeviceGetName nvmlSystemGetDriverVersion; do
		check_eq "unresolved $symbol returns NVML_ERROR_FUNCTION_NOT_FOUND" "13" \
			"$(sed -n "s/^unresolved $symbol rc=\([0-9-]*\).*/\1/p" <<<"$unresolved_out")"
	done
fi

fake_check "one process of 1 GiB under an 8G limit" \
	$((8 * gib)) $((1 * gib)) $((7 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G

fake_check "three processes of 1 GiB are summed" \
	$((8 * gib)) $((3 * gib)) $((5 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G FAKE_NVML_PROC_COUNT=3

fake_check "no running processes reports zero used" \
	$((8 * gib)) 0 $((8 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G FAKE_NVML_PROC_COUNT=0

fake_check "an NVML_VALUE_NOT_AVAILABLE entry is skipped, not summed" \
	$((8 * gib)) $((1 * gib)) $((7 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G FAKE_NVML_PROC_COUNT=2 FAKE_NVML_PROC_NOT_AVAILABLE=1

# The container cannot really exceed its own MPS limit, so a sum above the cap
# means the list was not container-scoped. Clamping keeps free from underflowing.
fake_check "usage above the limit is clamped, leaving free at zero" \
	$((8 * gib)) $((8 * gib)) 0 \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G FAKE_NVML_PROC_BYTES=$((20 * gib))

# Process query broken: report the driver's own used, still under the cap,
# rather than inventing a number.
fake_check "a failing process query falls back to the driver's used value" \
	$((8 * gib)) $((8 * gib)) 0 \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G FAKE_NVML_PROC_FAIL=1

fake_check "the limit for device 0 is chosen out of a multi-device list" \
	$((8 * gib)) $((1 * gib)) $((7 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G,1=4G

# Without a resolvable index a single limit is unambiguous and is applied.
fake_check "one limit still applies when the device index cannot be resolved" \
	$((8 * gib)) $((1 * gib)) $((7 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G FAKE_NVML_NO_INDEX=1

# With several, there is nothing to match against, so nothing is capped.
fake_check "several limits and no resolvable index reports the real device" \
	"$fake_total" $((80 * gib)) $((20 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G,1=4G FAKE_NVML_NO_INDEX=1

# A limit naming only devices this container does not have is not this
# container's limit.
fake_check "a limit naming only other devices reports the real device" \
	"$fake_total" $((80 * gib)) $((20 * gib)) \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=3=8G,4=4G

reserved_out="$(with_env shim "GPU_FRACTIONING_REAL_NVML=$fake_lib" \
	CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G -- "$build_dir/shimcheck" 2>/dev/null)"
check_eq "v2 reserved is passed through even when capping" "$fake_reserved" \
	"$(sed -n 's/.*reserved=\([0-9]*\).*/\1/p' <<<"$reserved_out")"

echo
echo "=== $passes passed, $failures failed ==="
(( failures == 0 ))
