#!/usr/bin/env python3
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate THIRD-PARTY.txt from the modules actually linked into the shipped binaries.

Scope is deliberately narrower than `go list -m all`: that reports the whole
module graph, including test-only and tooling dependencies that are never
distributed. Attribution is owed for what we ship, so this walks `go list -deps`
from each binary's main package instead.

License texts are emitted once per distinct text and cross-referenced, rather
than repeated per module — reproducing the same Apache-2.0 text 55 times would
add ~600 KB of identical bytes without adding information.

Usage:
    python3 hack/gen-third-party.py            # write THIRD-PARTY.txt
    python3 hack/gen-third-party.py --check    # fail if THIRD-PARTY.txt is stale
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIRST_PARTY = "github.com/kai-scheduler/gpu-fractioning"
OUTPUT = REPO / "THIRD-PARTY.txt"

# (binary name, module directory, main package) — the binaries shipped in the
# images we publish.
BINARIES = [
    ("operator", "operator", "./cmd/..."),
    ("fractiond", ".", "./fractioning-manager/fractiond/cmd"),
    ("mpsd", ".", "./fractioning-manager/mpsd/cmd"),
    ("metricsd", "fractioning-manager/metricsd", "./cmd"),
]

# The platforms the release images are built for. Dependency sets are
# GOOS/GOARCH-dependent — prometheus/client_golang, for instance, only pulls in
# prometheus/procfs on Linux — so resolving against the host would make the
# output depend on whoever ran the generator. Resolve against what we ship and
# take the union.
TARGET_PLATFORMS = [("linux", "amd64"), ("linux", "arm64")]

LICENSE_FILE_RE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE)([._-].*)?$", re.IGNORECASE)


def go(args: list[str], cwd: Path, goos: str | None = None, goarch: str | None = None) -> str:
    env = dict(os.environ, GOWORK="off", CGO_ENABLED="1")
    if goos:
        env["GOOS"] = goos
    if goarch:
        env["GOARCH"] = goarch
    return subprocess.run(
        ["go", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout


def module_cache() -> Path:
    return Path(go(["env", "GOMODCACHE"], REPO).strip())


def escape(path: str) -> str:
    """Apply the Go module cache's uppercase escaping (Foo -> !foo)."""
    return re.sub(r"[A-Z]", lambda m: "!" + m.group(0).lower(), path)


def linked_modules() -> dict[str, set[str]]:
    """Map "module@version" -> set of binaries that link it."""
    result: dict[str, set[str]] = {}
    for name, moddir, pkg in BINARIES:
        for goos, goarch in TARGET_PLATFORMS:
            out = go(
                ["list", "-deps", "-f",
                 "{{if .Module}}{{.Module.Path}}@{{.Module.Version}}{{end}}", pkg],
                REPO / moddir,
                goos=goos,
                goarch=goarch,
            )
            for line in out.splitlines():
                line = line.strip()
                if not line or line.startswith(FIRST_PARTY):
                    continue
                result.setdefault(line, set()).add(name)
    return result


def license_files(cache: Path, module: str, version: str) -> list[Path]:
    root = cache / (escape(module) + "@" + escape(version))
    if not root.is_dir():
        raise SystemExit(
            f"module not in cache: {module}@{version}\nRun `go mod download` first."
        )
    found = [
        Path(p)
        for p in glob.glob(str(root / "*"))
        if os.path.isfile(p) and LICENSE_FILE_RE.match(os.path.basename(p))
    ]
    if not found:
        raise SystemExit(f"no license file found for {module}@{version} in {root}")
    return sorted(found)


def source_url(module: str) -> str:
    special = {
        "cel.dev/expr": "https://github.com/google/cel-spec",
        "gomodules.xyz/jsonpatch/v2": "https://github.com/gomodules/jsonpatch",
    }
    if module in special:
        return special[module]
    prefixes = [
        ("golang.org/x/", "https://cs.opensource.google/go/x/"),
        ("k8s.io/", "https://github.com/kubernetes/"),
        ("sigs.k8s.io/", "https://github.com/kubernetes-sigs/"),
    ]
    for prefix, base in prefixes:
        if module.startswith(prefix):
            return base + module[len(prefix):].split("/")[0]
    if module.startswith("go.opentelemetry.io/"):
        return "https://github.com/open-telemetry/opentelemetry-go"
    if module.startswith("go.uber.org/"):
        return "https://github.com/uber-go/" + module.split("/")[-1]
    if module.startswith("go.yaml.in/"):
        return "https://github.com/yaml/go-yaml"
    if module.startswith("google.golang.org/grpc"):
        return "https://github.com/grpc/grpc-go"
    if module.startswith("google.golang.org/protobuf"):
        return "https://github.com/protocolbuffers/protobuf-go"
    if module.startswith("google.golang.org/genproto"):
        return "https://github.com/googleapis/go-genproto"
    if module.startswith("gopkg.in/evanphx/"):
        return "https://github.com/evanphx/json-patch"
    if module.startswith("gopkg.in/yaml"):
        return "https://github.com/go-yaml/yaml"
    if module.startswith("gopkg.in/inf"):
        return "https://github.com/go-inf/inf"
    return "https://" + module


def render() -> str:
    cache = module_cache()
    mods = linked_modules()

    # text hash -> (text, [module@version that ship it])
    texts: dict[str, tuple[str, list[str]]] = {}
    entries = []
    for key in sorted(mods):
        module, _, version = key.rpartition("@")
        blocks = []
        for path in license_files(cache, module, version):
            # Explicit UTF-8: several license files carry non-ASCII copyright
            # holders, and a locale-dependent default would decode them
            # differently on a CI runner than on a developer machine, making
            # the generated file non-reproducible.
            text = path.read_text(encoding="utf-8", errors="replace")
            digest = hashlib.sha256(text.encode()).hexdigest()[:12]
            texts.setdefault(digest, (text, []))[1].append(f"{key} ({path.name})")
            blocks.append((path.name, digest))
        entries.append((key, module, version, sorted(mods[key]), blocks))

    out = []
    out.append("THIRD-PARTY SOFTWARE NOTICES AND INFORMATION")
    out.append("=" * 78)
    out.append("")
    out.append("gpu-fractioning")
    out.append("Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.")
    out.append("SPDX-License-Identifier: Apache-2.0")
    out.append("")
    out.append(
        "This project incorporates the third-party open source components listed"
    )
    out.append(
        "below. They are statically linked into the binaries shipped in the"
    )
    out.append(
        "gpu-fractioning container images. Each component remains subject to its"
    )
    out.append("own license, reproduced in the LICENSE TEXTS section.")
    out.append("")
    out.append("This file is generated. Regenerate it with `make third-party`.")
    out.append("")
    out.append("Components not listed here, and why:")
    out.append("")
    out.append(
        "  * NVML (libnvidia-ml.so.1) is dynamically loaded by mpsd"
    )
    out.append(
        "    and metricsd at runtime."
    )
    out.append(
        "    It is part of the NVIDIA driver, is supplied by the NVIDIA container"
    )
    out.append("    runtime, and is not redistributed by this project.")
    out.append(
        "  * nvidia-cuda-mps-control is executed as a separate process by mpsd. It"
    )
    out.append("    comes from the CUDA base image and is not added by this project.")
    out.append(
        "  * Contents of the container base images are covered by those images'"
    )
    out.append("    own notices; this project installs no additional OS packages.")
    out.append(
        "  * Build, test and development tooling is not distributed in any artifact."
    )
    out.append("")
    out.append("=" * 78)
    out.append("COMPONENTS")
    out.append("=" * 78)
    out.append("")
    for key, module, version, bins, blocks in entries:
        out.append(module)
        out.append(f"    Version:      {version}")
        out.append(f"    Source:       {source_url(module)}")
        out.append(f"    Linked into:  {', '.join(bins)}")
        out.append(
            "    License:      "
            + ", ".join(f"{name} [text {digest}]" for name, digest in blocks)
        )
        out.append("")

    out.append("=" * 78)
    out.append("LICENSE TEXTS")
    out.append("=" * 78)
    out.append("")
    out.append(
        "Each distinct license text is reproduced once and referenced above by its"
    )
    out.append("[text <id>] marker.")
    out.append("")
    for digest in sorted(texts, key=lambda d: (-len(texts[d][1]), d)):
        text, users = texts[digest]
        out.append("-" * 78)
        out.append(f"text {digest} — applies to:")
        for u in sorted(users):
            out.append(f"    {u}")
        out.append("-" * 78)
        out.append("")
        out.append(text.rstrip("\n"))
        out.append("")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the file is stale")
    args = parser.parse_args()

    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            print(
                f"{OUTPUT.name} is out of date. Run `make third-party` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.name} is up to date.")
        return 0
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
