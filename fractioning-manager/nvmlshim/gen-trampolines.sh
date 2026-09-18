#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Emits the x86-64 forwarding trampolines for every NVML entry point the shim
# does not implement itself, and verifies the committed symbol list still
# matches the driver when one is installed.
#
# There are 431 nvml* entry points and we have signatures for none of them, so
# the forwarders cannot be C functions: a C wrapper has to name its arguments
# and its return type, and getting either wrong corrupts the call. The
# trampolines instead tail-jump through a pointer slot without touching the
# argument registers or the stack, which is correct for every signature,
# including the struct-return and varargs ones, precisely because it never
# claims to know what they are.
#
#   ./gen-trampolines.sh                     # emit from the committed list
#   ./gen-trampolines.sh --update            # re-capture the list from the driver
#   ./gen-trampolines.sh --nvml /path/to.so  # verify against a specific driver

set -euo pipefail

# Deterministic output regardless of the builder's locale: sort and comm must
# agree on collation or comm rejects its own inputs as unsorted.
export LC_ALL=C

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
symbols_file="$script_dir/symbols.txt"
overrides_file="$script_dir/overrides.txt"
out_file="$script_dir/generated/trampolines.S"
real_nvml=""
update=0
verify=1

# Where a driver may be installed. Only used to verify or refresh the committed
# list; the emitted code never depends on a driver being present here.
NVML_SEARCH_GLOBS=(
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*[0-9]
  /usr/lib64/libnvidia-ml.so.*[0-9]
  /usr/lib/libnvidia-ml.so.*[0-9]
  /usr/local/nvidia/lib64/libnvidia-ml.so.*[0-9]
  /usr/local/nvidia/lib/libnvidia-ml.so.*[0-9]
)

usage() {
  sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --update)     update=1; shift ;;
    --no-verify)  verify=0; shift ;;
    --nvml)       real_nvml="${2:?--nvml needs a path}"; shift 2 ;;
    --out)        out_file="${2:?--out needs a path}"; shift 2 ;;
    -h|--help)    usage 0 ;;
    *)            echo "gen-trampolines.sh: unknown argument '$1'" >&2; usage 1 ;;
  esac
done

# strip_comments reads one of the committed .txt lists and yields bare symbols.
strip_comments() {
  grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$1" | sort -u
}

find_real_nvml() {
  local candidate
  for candidate in "${NVML_SEARCH_GLOBS[@]}"; do
    # An unmatched glob stays literal, so test for a real file rather than
    # trusting expansion.
    [[ -f "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; }
  done
  return 1
}

# read_driver_symbols lists the exported nvml* text symbols of a driver build.
read_driver_symbols() {
  nm -D --defined-only "$1" | awk '$2 == "T" && $3 ~ /^nvml/ { print $3 }' | sort -u
}

[[ -r "$symbols_file" ]]   || { echo "gen-trampolines.sh: missing $symbols_file" >&2; exit 1; }
[[ -r "$overrides_file" ]] || { echo "gen-trampolines.sh: missing $overrides_file" >&2; exit 1; }

if [[ -z "$real_nvml" && ( $update -eq 1 || $verify -eq 1 ) ]]; then
  real_nvml="$(find_real_nvml || true)"
fi

if [[ -n "$real_nvml" ]] && ! command -v nm >/dev/null 2>&1; then
  echo "gen-trampolines.sh: nm not available, skipping driver verification" >&2
  real_nvml=""
fi

if [[ -n "$real_nvml" ]]; then
  driver_symbols="$(read_driver_symbols "$real_nvml")"
  committed_symbols="$(strip_comments "$symbols_file")"

  if [[ "$driver_symbols" != "$committed_symbols" ]]; then
    if [[ $update -eq 1 ]]; then
      # Keep the header, replace the body, so provenance survives the refresh.
      { grep -e '^[[:space:]]*#' "$symbols_file"; printf '%s\n' "$driver_symbols"; } \
        > "$symbols_file.tmp"
      mv "$symbols_file.tmp" "$symbols_file"
      echo "gen-trampolines.sh: updated symbols.txt from $real_nvml ($(printf '%s\n' "$driver_symbols" | wc -l) symbols)"
    else
      echo "gen-trampolines.sh: committed symbols.txt does not match $real_nvml" >&2
      echo "  '<' present only in symbols.txt, '>' present only in the driver:" >&2
      diff <(printf '%s\n' "$committed_symbols") <(printf '%s\n' "$driver_symbols") \
        | sed 's/^/  /' >&2 || true
      echo "  Re-run with --update once you have confirmed the driver change is intended." >&2
      exit 1
    fi
  elif [[ $update -eq 1 ]]; then
    echo "gen-trampolines.sh: symbols.txt already matches $real_nvml"
  fi
fi

symbols="$(strip_comments "$symbols_file")"
overrides="$(strip_comments "$overrides_file")"

# An override that the driver does not export means nvmlshim.c is defining an
# entry point nothing will ever call, which is a sign the list has rotted.
missing_overrides="$(comm -13 <(printf '%s\n' "$symbols") <(printf '%s\n' "$overrides"))"
if [[ -n "$missing_overrides" ]]; then
  echo "gen-trampolines.sh: overrides.txt names symbols absent from symbols.txt:" >&2
  printf '  %s\n' $missing_overrides >&2
  exit 1
fi

forwarded="$(comm -23 <(printf '%s\n' "$symbols") <(printf '%s\n' "$overrides"))"
forwarded_count="$(printf '%s\n' "$forwarded" | wc -l)"

mkdir -p "$(dirname "$out_file")"

{
  cat <<'HEADER'
/* Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * GENERATED by gen-trampolines.sh from symbols.txt -- do not edit.
 *
 * One tail-jumping forwarder per NVML entry point the shim does not implement
 * itself. Each loads its slot and jumps, leaving every argument register, the
 * stack and the return value exactly as the caller left them, which is why
 * this works without knowing any of the 429 signatures.
 *
 * %r11 is the scratch register used to hold the target. It is the only
 * register that is both caller-saved and unused by the SysV AMD64 calling
 * convention for argument passing, the static chain (%r10) and the varargs
 * vector-register count (%al), so clobbering it cannot disturb a call we are
 * forwarding blind.
 *
 * The endbr64 is present because a caller reaches these through a dlsym'd
 * function pointer, which is an indirect branch. We deliberately do NOT emit
 * a .note.gnu.property advertising IBT support: the real driver library has no
 * such note, so IBT is already disabled process-wide for NVML consumers, and
 * claiming it here while tail-jumping into un-annotated driver code would turn
 * a working process into a faulting one.
 */

	.text
	.p2align 4
/*
 * Reached when the driver in this container does not export the symbol the
 * committed list expected. Returning NVML_ERROR_FUNCTION_NOT_FOUND is what
 * NVML itself returns for an entry point a given driver version lacks, so
 * callers already have a path for it. Jumping through the NULL slot instead
 * would kill a GPU workload over a reporting shim.
 */
.Lnvmlshim_unresolved:
	movl	$13, %eax
	ret

HEADER

  printf '%s\n' "$forwarded" | awk '
    {
      printf "\t.p2align 4\n"
      printf "\t.globl\t%s\n", $0
      printf "\t.type\t%s, @function\n", $0
      printf "%s:\n", $0
      printf "\tendbr64\n"
      printf "\tmovq\tnvmlshim_slots+%d(%%rip), %%r11\n", NR * 8 - 8
      printf "\ttestq\t%%r11, %%r11\n"
      printf "\tje\t.Lnvmlshim_unresolved\n"
      printf "\tjmp\t*%%r11\n"
      printf "\t.size\t%s, .-%s\n\n", $0, $0
    }'

  cat <<HEADER
/*
 * One pointer per forwarded symbol, in the same order as the name blob below,
 * filled once by the constructor. Hidden so the trampolines can reach it with
 * a plain RIP-relative load: a default-visibility data symbol would be
 * interposable, and the load above would then disagree with what the C code
 * sees through the GOT.
 */
	.section .bss
	.p2align 4
	.globl	nvmlshim_slots
	.hidden	nvmlshim_slots
	.type	nvmlshim_slots, @object
	.size	nvmlshim_slots, $((forwarded_count * 8))
nvmlshim_slots:
	.zero	$((forwarded_count * 8))

/*
 * The names, packed as consecutive NUL-terminated strings. A blob rather than
 * an array of pointers so the table needs no relocations at all, which keeps
 * it in .rodata instead of dragging in a relocated .data.rel.ro section for
 * what is only ever walked once.
 */
	.section .rodata
	.p2align 4
	.globl	nvmlshim_symbol_count
	.hidden	nvmlshim_symbol_count
	.type	nvmlshim_symbol_count, @object
	.size	nvmlshim_symbol_count, 8
nvmlshim_symbol_count:
	.quad	$forwarded_count

	.globl	nvmlshim_symbol_names
	.hidden	nvmlshim_symbol_names
	.type	nvmlshim_symbol_names, @object
nvmlshim_symbol_names:
HEADER

  printf '%s\n' "$forwarded" | awk '{ printf "\t.asciz\t\"%s\"\n", $0 }'

  cat <<'FOOTER'

/* Mark the stack non-executable; a hand-written .S gets no such note for free. */
	.section .note.GNU-stack,"",@progbits
FOOTER
} > "$out_file"

echo "gen-trampolines.sh: wrote $out_file ($forwarded_count forwarded, $(printf '%s\n' "$overrides" | wc -l) overridden)"
