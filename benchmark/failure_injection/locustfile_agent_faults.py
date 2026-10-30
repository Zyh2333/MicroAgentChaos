#!/usr/bin/env python3
"""Agent Service workload through the real planner's /service/unify endpoint.

The request contract follows agent-network/examples/distributed_real/send_task.py:
    POST /service/unify {"flowId": "AgentNetworkPlannerGroup/AgentNetworkPlanner", "task": ...}
Task text comes from the existing MAR and MDOC workload definitions.
"""
from __future__ import annotations

import importlib.util
import os
import random
import uuid
from functools import lru_cache
from pathlib import Path

from locust import HttpUser, between, task

PLANNER = "AgentNetworkPlannerGroup/AgentNetworkPlanner"
HERE = Path(__file__).resolve().parent
DATASET = os.getenv("AGENT_FAULT_DATASET", "MARBLEBench")
FAULT_SERVICE = os.getenv("FAULT_SERVICE", "agent-network-marble-research")


def _load_legacy(filename: str):
    """Reuse task definitions without exposing legacy HttpUser classes to Locust."""
    spec = importlib.util.spec_from_file_location("_agent_fault_tasks_" + filename.replace(".", "_"), HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def task_pool():
    if DATASET == "MARBLEBench":
        intent = _load_legacy("locustfile_MARBLEBench.py")
        original = _load_legacy("locustfile_MARBLEBench_orig.py")
        definitions = list(intent.MARBLEBenchUser.TASK_DEFINITIONS)
        definitions += list(original.MARBLEBenchOrigUser.TASK_DEFINITIONS)
        resources = {}
    elif DATASET == "MDOC":
        mdoc = _load_legacy("locustfile.py")
        definitions = list(mdoc.MDOCUser.TASK_DEFINITIONS)
        resources = mdoc.MDOCUser.FILE_RESOURCES
    else:
        raise ValueError(f"unknown AGENT_FAULT_DATASET: {DATASET}")
    if not definitions:
        raise RuntimeError(f"no workload tasks for {DATASET}")
    # The planner is invoked on every request; the summarizer normally follows
    # completion. Other faults should favor tasks declared for the target service.
    if FAULT_SERVICE not in {"agent-network-planner", "agent-network-summarizer"}:
        matching = [item for item in definitions
                    if FAULT_SERVICE in item.get("services", [item["name"]])]
        if matching:
            definitions = matching
    return definitions, resources


def resolve_prompt(item: dict, resources: dict) -> str:
    prompt = item["prompt"]
    file_type = item.get("requires_file")
    if file_type:
        urls = resources.get(file_type, [])
        if not urls:
            raise RuntimeError(f"no file resources for {file_type}")
        prompt = prompt.format(url=random.choice(urls))
    return prompt


class AgentFaultUser(HttpUser):
    wait_time = between(5, 10)

    @task
    def send_planner_task(self):
        definitions, resources = task_pool()
        item = random.choice(definitions)
        prompt = resolve_prompt(item, resources)
        request_id = uuid.uuid4()
        task_name = item["name"]
        task_type = item.get("type", "TASK")
        preview = " ".join(prompt.split())[:120]
        print(f"id:{request_id} {DATASET}/{task_type}/{task_name} | {preview}")
        with self.client.post(
            "/service/unify",
            json={"flowId": PLANNER, "task": prompt},
            name=f"{DATASET}/{task_type}/{task_name}",
            timeout=600,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code} for {task_name}")
