#!/usr/bin/env python3
"""Build a per-run manifest with the planner and selected edge agent services."""

from __future__ import annotations

import os
from pathlib import Path
import random
import re
import sys

import yaml


PLANNER = "agent-network-planner"
ZONE_LABEL = "agent-network.io/zone"
TASK_FILES = {
    "agent_faults_MDOC": ("locustfile.py",),
    "agent_faults_MAR": ("locustfile_MARBLEBench.py",),
}
TOPOLOGY_MODES = {"dynamic", "edge-preference", "cloud-preference"}


def with_zone_affinity(raw: str, item: dict, mode: str) -> str:
    """Keep the selected Pod's routing metadata consistent with the run mode."""
    if item["kind"] != "Deployment":
        return raw
    env = item["spec"]["template"]["spec"]["containers"][0].get("env", [])
    zone_values = [entry.get("value") for entry in env if entry["name"] == "NACOS_META_ZONE"]
    affinity = [entry.get("value") for entry in env
                if entry["name"] == "NACOS_META_ZONE_AFFINITY"]
    if len(zone_values) != 1 or zone_values[0] not in {"cloud", "edge"} or len(affinity) > 1:
        raise ValueError(f"invalid zone metadata in {item['metadata']['name']}")
    zone = zone_values[0]
    enabled = mode == f"{zone}-preference"
    if not enabled:
        if affinity and str(affinity[0]).lower() in {"true", "1", "yes"}:
            raise ValueError(f"{item['metadata']['name']} has zone affinity incompatible with {mode}")
        return raw
    if affinity:
        if str(affinity[0]).lower() not in {"true", "1", "yes"}:
            raise ValueError(f"{item['metadata']['name']} must enable zone affinity for {mode}")
        return raw
    pattern = (r"(?m)^(?P<indent>\s*)- name: NACOS_META_ZONE\n"
               r"(?P=indent)  value: " + zone + r"\n")
    raw, replacements = re.subn(
        pattern,
        lambda match: match.group(0) + match.group("indent")
        + '- name: NACOS_META_ZONE_AFFINITY\n'
        + match.group("indent") + '  value: "true"\n',
        raw,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"cannot inject zone affinity into {item['metadata']['name']}")
    return raw


def edge_services_from_documents(documents: list[dict]) -> set[str]:
    edge_services = set()
    for item in documents:
        if item["kind"] != "Deployment":
            continue
        name = item["metadata"]["name"]
        expected_zone = "edge" if name.endswith("-edge") else "cloud"
        selector = item["spec"]["template"]["spec"].get("nodeSelector", {})
        if selector.get(ZONE_LABEL) != expected_zone or "kubernetes.io/hostname" in selector:
            raise ValueError(f"{name} must select {ZONE_LABEL}={expected_zone} without a hostname")
        if expected_zone == "edge":
            edge_services.add(name.removesuffix("-edge"))
    if PLANNER not in edge_services:
        raise ValueError(f"manifest must contain {PLANNER}-edge")
    return edge_services


def choose_services(folder: str, available: set[str], target: str, count: int = 3,
                    rng: random.Random | None = None) -> list[str]:
    if folder not in TASK_FILES:
        raise ValueError(f"unexpected manifest directory: {folder}")
    if count < 1 or count > len(available):
        raise ValueError(f"EDGE_SERVICE_SAMPLE_COUNT must be within 1..{len(available)} edge services")
    base = target.removesuffix("-edge")
    if base not in {PLANNER, "common"} and base not in available:
        raise ValueError(f"target has no edge Deployment: {target}")
    override = os.environ.get("EDGE_SERVICES", "").strip()
    if override:
        chosen = [name.removesuffix("-edge") for name in re.split(r"[\s,]+", override)]
        if len(chosen) != count or len(set(chosen)) != count:
            raise ValueError(f"EDGE_SERVICES must list exactly {count} distinct non-planner services")
        if set(chosen) - available:
            raise ValueError(f"EDGE_SERVICES contains unknown/non-edge services: {sorted(set(chosen) - available)}")
        if base in available and base not in chosen:
            raise ValueError(f"EDGE_SERVICES must include fault target {base}")
        return chosen

    chosen = [base] if base in available else []
    others = sorted(available - set(chosen))
    chosen.extend((rng or random.SystemRandom()).sample(others, count - len(chosen)))
    return chosen


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: select_edge_deployments.py SOURCE TARGET BENCHMARK SELECTED", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = sys.argv[2]
    folder = sys.argv[3]
    selected_path = Path(sys.argv[4])
    if folder not in TASK_FILES:
        raise ValueError(f"unexpected manifest directory: {folder}")
    mode = os.environ.get("TOPOLOGY_MODE", "dynamic")
    if mode not in TOPOLOGY_MODES:
        raise ValueError(f"unknown TOPOLOGY_MODE: {mode}")
    raw_documents = [part.strip("\n") + "\n" for part in
                     re.split(r"(?m)^---\s*$", source.read_text(encoding="utf-8"))
                     if part.strip()]
    documents = [(yaml.safe_load(raw), raw) for raw in raw_documents]
    if any("summarizer" in item["metadata"]["name"] for item, _ in documents):
        raise ValueError(f"{source} still contains the removed summarizer service")
    available = edge_services_from_documents([item for item, _ in documents]) - {PLANNER}
    count = int(os.environ.get("EDGE_SERVICE_SAMPLE_COUNT", "3"))
    chosen = choose_services(folder, available, target, count)
    selected = []
    for item, raw in documents:
        name = item["metadata"]["name"]
        if name == f"{PLANNER}-edge-nodeport":
            selected.append(raw)
        elif name.endswith("-edge"):
            base = name.removesuffix("-edge")
            if base == PLANNER or base in chosen:
                selected.append(with_zone_affinity(raw, item, mode))
        else:
            selected.append(with_zone_affinity(raw, item, mode))
    selected_names = {yaml.safe_load(raw)["metadata"]["name"] for raw in selected}
    for base in [PLANNER, *chosen]:
        if f"{base}-edge" not in selected_names:
            raise ValueError(f"{source} is missing {base}-edge")
    if f"{PLANNER}-edge-nodeport" not in selected_names:
        raise ValueError(f"{source} is missing the edge planner NodePort")
    selected_path.write_text("---\n".join(selected), encoding="utf-8")
    print(" ".join(chosen))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"edge deployment selection failed: {error}", file=sys.stderr)
        raise SystemExit(2)
