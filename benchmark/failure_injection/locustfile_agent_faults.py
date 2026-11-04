#!/usr/bin/env python3
"""Agent Service workload through the real planner's /service/unify endpoint.

The request contract follows agent-network/examples/distributed_real/send_task.py:
    POST /service/unify {"flowId": "AgentNetworkPlannerGroup/AgentNetworkPlanner", "task": ...}
Task text comes from the existing MAR and MDOC workload definitions.
"""
from __future__ import annotations

import os
import random
import uuid
from functools import lru_cache
from urllib.parse import urlsplit

from locust import HttpUser, between, task
from agent_fault_task_catalog import (
    dataset_definitions as load_dataset_definitions,
    definitions_for_service as filter_definitions_for_service,
    parse_task_indices,
)

PLANNER = "AgentNetworkPlannerGroup/AgentNetworkPlanner"
DATASET = os.getenv("AGENT_FAULT_DATASET", "MARBLEBench")
FAULT_SERVICE = os.getenv("FAULT_SERVICE", "agent-network-marble-research")
TASK_SERVICE = FAULT_SERVICE.removesuffix("-edge")
TASK_INDEX = os.getenv("AGENT_FAULT_TASK_INDEX", "").strip()
PLANNER_HOSTS_RAW = os.getenv("AGENT_FAULT_PLANNER_HOSTS", "").strip()


def planner_hosts() -> tuple[str, ...]:
    if not PLANNER_HOSTS_RAW:
        return ()
    hosts = tuple(value.strip().rstrip("/") for value in PLANNER_HOSTS_RAW.split(","))
    if len(hosts) != 2 or len(set(hosts)) != 2 or any(
        urlsplit(host).scheme not in {"http", "https"} or not urlsplit(host).netloc
        for host in hosts
    ):
        raise ValueError("AGENT_FAULT_PLANNER_HOSTS must contain two distinct HTTP URLs")
    return hosts


PLANNER_HOSTS = planner_hosts()


@lru_cache(maxsize=1)
def dataset_definitions():
    return load_dataset_definitions(DATASET)


def definitions_for_service(service: str):
    definitions, resources = dataset_definitions()
    return filter_definitions_for_service(definitions, service), resources


@lru_cache(maxsize=1)
def task_pool():
    definitions, resources = definitions_for_service(TASK_SERVICE)
    if TASK_INDEX:
        indices = parse_task_indices(TASK_INDEX, len(definitions))
        definitions = [definitions[index] for index in indices]
        print(f"selected task indices for {TASK_SERVICE}: {indices}", flush=True)
    return definitions, resources


def resolve_prompt(item: dict, resources: dict) -> str:
    prompt = item["prompt"]
    file_type = item.get("requires_file")
    if file_type:
        urls = resources.get(file_type, [])
        if not urls:
            raise RuntimeError(f"no file resources for {file_type}")
        prompt = prompt.format(url=urls[0] if TASK_INDEX else random.choice(urls))
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
        planner_host = random.choice(PLANNER_HOSTS) if PLANNER_HOSTS else None
        request_url = f"{planner_host}/service/unify" if planner_host else "/service/unify"
        print(f"id:{request_id} {DATASET}/{task_type}/{task_name} "
              f"planner_entry:{planner_host or 'configured-host'} | {preview}")
        with self.client.post(
            request_url,
            json={"flowId": PLANNER, "task": prompt},
            name=f"{DATASET}/{task_type}/{task_name}",
            timeout=600,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code} for {task_name}")
