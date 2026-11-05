#!/usr/bin/env python3
"""Read MDOC and MAR task definitions without starting Locust or gevent."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import random
import re


HERE = Path(__file__).resolve().parent


def literal_assignment(filename: str, name: str):
    path = HERE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path}: missing literal assignment {name}")


def marble_original_definitions() -> list[dict]:
    filename = "locustfile_MARBLEBench_orig.py"
    scenarios = literal_assignment(filename, "SCENARIO_SPECS")
    fallbacks = literal_assignment(filename, "FALLBACK_TASKS")
    bench_dir = Path(os.getenv("MARBLE_BENCH_DIR", str((HERE / "../../../MARBLE/multiagentbench").resolve())))
    max_tasks = int(os.getenv("MARBLE_MAX_TASKS_PER_SCENARIO", "20"))
    seed = os.getenv("MARBLE_SAMPLE_SEED", "").strip()
    sampler = random.Random(int(seed)) if seed else random
    definitions = []
    for scenario, (service, relative_path, task_type) in scenarios.items():
        prompts = []
        try:
            with (bench_dir / relative_path).open(encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    content = (record.get("task") or {}).get("content")
                    if content and content.strip():
                        prompts.append(content.strip())
        except (OSError, json.JSONDecodeError):
            prompts = []
        if max_tasks and len(prompts) > max_tasks:
            prompts = sampler.sample(prompts, max_tasks)
        if not prompts:
            fallback = fallbacks.get(scenario)
            prompts = [fallback] if fallback else []
        for prompt in prompts:
            definitions.append({"name": service, "scenario": scenario,
                                "type": task_type, "services": [service], "prompt": prompt})
    return definitions


def dataset_definitions(dataset: str) -> tuple[list[dict], dict]:
    if dataset == "MARBLEBench":
        definitions = literal_assignment("locustfile_MARBLEBench.py", "TASK_DEFINITIONS")
        definitions.extend(marble_original_definitions())
        resources = {}
    elif dataset == "MDOC":
        definitions = literal_assignment("locustfile.py", "TASK_DEFINITIONS")
        resources = literal_assignment("locustfile.py", "FILE_RESOURCES")
    else:
        raise ValueError(f"unknown AGENT_FAULT_DATASET: {dataset}")
    if not definitions:
        raise RuntimeError(f"no workload tasks for {dataset}")
    return definitions, resources


def definitions_for_service(definitions: list[dict], service: str) -> list[dict]:
    service = service.removesuffix("-edge")
    if service not in {"agent-network-planner", "common"}:
        matching = [item for item in definitions
                    if service in item.get("services", [item["name"]])]
        if matching:
            return matching
    return definitions


def parse_task_indices(raw: str, pool_size: int) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("AGENT_FAULT_TASK_INDEX must be an integer list") from error
        if not isinstance(values, list):
            raise ValueError("AGENT_FAULT_TASK_INDEX must be an integer list")
    else:
        tokens = [part for part in re.split(r"[\s,]+", raw) if part]
        try:
            values = [int(part) for part in tokens]
        except ValueError as error:
            raise ValueError("AGENT_FAULT_TASK_INDEX must contain integers") from error
    if not values or any(type(value) is not int for value in values):
        raise ValueError("AGENT_FAULT_TASK_INDEX must contain integers")
    if len(values) != len(set(values)):
        raise ValueError("AGENT_FAULT_TASK_INDEX contains duplicate indices")
    if any(value < 0 or value >= pool_size for value in values):
        raise ValueError(f"AGENT_FAULT_TASK_INDEX must be within 0..{pool_size - 1}")
    return values
