#!/usr/bin/env python3
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail if the chart's CRD and the generated CRD disagree on their CEL rules.

There are two CRDs in this repository. controller-gen writes
operator/config/crd/bases/ from the kubebuilder markers on the Go types, and
that is the one developers read. The chart ships its own hand-maintained copy in
operator/charts/templates/crd.yaml, deliberately trimmed (complex sub-objects
become x-kubernetes-preserve-unknown-fields rather than full schemas), and that
is the one actually installed on a cluster.

The trimming is fine. Losing a validation rule is not: add a
+kubebuilder:validation:XValidation marker, regenerate, forget the chart, and the
rule is enforced in every developer's head and on no real cluster. Nothing else
in the build notices, because both files are individually valid.

So this compares only the x-kubernetes-validations rules, keyed by the schema
path they sit on, in both directions.
"""

import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

REPO = Path(__file__).resolve().parent.parent
CHART = REPO / "operator" / "charts"
GENERATED_DIR = REPO / "operator" / "config" / "crd" / "bases"


def rendered_chart_crds():
    """Render the chart and return its CRDs, so this checks what installs."""
    try:
        out = subprocess.run(
            ["helm", "template", "crd-check", str(CHART)],
            check=True, capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        sys.exit("helm is required to render the chart CRD")
    except subprocess.CalledProcessError as err:
        sys.exit(f"helm template failed:\n{err.stderr}")

    return {
        doc["metadata"]["name"]: doc
        for doc in yaml.safe_load_all(out)
        if doc and doc.get("kind") == "CustomResourceDefinition"
    }


def generated_crds():
    crds = {}
    for path in sorted(GENERATED_DIR.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if doc and doc.get("kind") == "CustomResourceDefinition":
                crds[doc["metadata"]["name"]] = doc
    return crds


def rules(node, path="", found=None):
    """Collect (schema path, CEL rule) pairs from an OpenAPI schema."""
    if found is None:
        found = set()
    if not isinstance(node, dict):
        return found

    for validation in node.get("x-kubernetes-validations", []):
        found.add((path or ".", validation.get("rule", "")))

    for name, prop in (node.get("properties") or {}).items():
        rules(prop, f"{path}.{name}", found)
    if isinstance(node.get("items"), dict):
        rules(node["items"], f"{path}[]", found)
    if isinstance(node.get("additionalProperties"), dict):
        rules(node["additionalProperties"], f"{path}{{}}", found)

    return found


def crd_rules(crd):
    found = set()
    for version in crd.get("spec", {}).get("versions", []):
        schema = version.get("schema", {}).get("openAPIV3Schema", {})
        rules(schema, f"{version['name']}", found)
    return found


def main():
    chart, generated = rendered_chart_crds(), generated_crds()

    problems = []
    for name in sorted(set(chart) | set(generated)):
        if name not in chart:
            problems.append(f"{name}: present in the generated CRD, missing from the chart")
            continue
        if name not in generated:
            problems.append(f"{name}: present in the chart, missing from the generated CRD")
            continue

        chart_rules, generated_rules = crd_rules(chart[name]), crd_rules(generated[name])
        for path, rule in sorted(generated_rules - chart_rules):
            problems.append(
                f"{name}: rule on {path} is generated but not in the chart, "
                f"so it is not enforced on a real install: {rule}"
            )
        for path, rule in sorted(chart_rules - generated_rules):
            problems.append(
                f"{name}: rule on {path} is in the chart but not generated, "
                f"so regenerating will silently drop it: {rule}"
            )

    if problems:
        print("CRD validation rules disagree between the chart and controller-gen:\n")
        for problem in problems:
            print(f"  - {problem}")
        print("\nUpdate operator/charts/templates/crd.yaml to match, then re-run.")
        return 1

    print("CRD validation rules agree between the chart and controller-gen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
