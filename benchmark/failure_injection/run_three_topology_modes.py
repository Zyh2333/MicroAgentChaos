#!/usr/bin/env python3
"""Run the same sampled tasks through edge, cloud, and dynamic topologies."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import yaml

from agent_fault_task_catalog import (
    dataset_definitions as load_dataset_definitions,
    definitions_for_service as filter_definitions_for_service,
    parse_task_indices,
)
from select_edge_deployments import PLANNER, choose_services, edge_services_from_documents


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_ROOT = REPO_ROOT / "benchmark" / "data"
FOLDERS = {"agent_faults_MDOC": "MDOC", "agent_faults_MAR": "MARBLEBench"}
MODES = ("edge-preference", "cloud-preference", "dynamic")
RUNTIME_KEYS = (
    "FAULT_TYPES", "FAULT_MODE", "INJECT_FAULT", "SAMPLES_PER_FAULT",
    "LOCUST_USERS", "LOCUST_SPAWN_RATE", "WARMUP_SECONDS", "BASELINE_SECONDS",
    "PRE_FAULT_SECONDS", "FAULT_SECONDS", "RECOVERY_SECONDS",
    "LOCUST_STOP_TIMEOUT", "MANAGE_DEPLOYMENTS", "DEPLOYMENTS_FILE",
    "HOST_IP", "LOCUST_HOST", "PLANNER_NODEPORT", "MARBLE_BENCH_DIR",
    "MARBLE_MAX_TASKS_PER_SCENARIO", "LOCUSTFILE", "PYTHON",
    "AGENT_FAULT_PLANNER_HOSTS", "EDGE_SERVICE_SAMPLE_COUNT",
)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def validate_planner_hosts(raw: str) -> None:
    if not raw:
        return
    hosts = [value.strip().rstrip("/") for value in raw.split(",")]
    if len(hosts) != 2 or len(set(hosts)) != 2 or any(
        urlsplit(host).scheme not in {"http", "https"} or not urlsplit(host).netloc
        for host in hosts
    ):
        raise ValueError("AGENT_FAULT_PLANNER_HOSTS must contain two distinct HTTP URLs")


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def log(run_dir: Path, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def services_for_run(folder: str, requested: str, inject_fault: bool,
                     edge_count: int) -> list[str]:
    directory = HERE / folder
    available = {path.name.removeprefix("sum_chaos_cloud_").removesuffix(".sh")
                 for path in directory.glob("sum_chaos_cloud_*.sh")}
    available.discard("common")
    if requested != "all":
        service = requested.removesuffix("-edge")
        if service not in available and not (service == "common" and inject_fault):
            raise ValueError(f"unknown service: {requested}")
        return [service]
    override = os.environ.get("EDGE_SERVICES", "").strip()
    if override:
        import re
        group = [name.removesuffix("-edge") for name in re.split(r"[\s,]+", override)]
        if len(group) != edge_count or len(set(group)) != edge_count or any(name not in available or name == PLANNER for name in group):
            raise ValueError(f"EDGE_SERVICES must list {edge_count} distinct non-planner services")
        services = [PLANNER, *group]
    else:
        services = [PLANNER, *sorted(available - {PLANNER})]
    if inject_fault:
        services.append("common")
    return services


def edge_deployments(folder: str) -> set[str]:
    source = HERE / folder / "deployments.yaml"
    documents = [item for item in yaml.safe_load_all(source.read_text(encoding="utf-8")) if item]
    return edge_services_from_documents(documents)


def task_fingerprint(item: dict, resources: dict) -> str:
    file_type = item.get("requires_file")
    file_url = (resources.get(file_type) or [None])[0] if file_type else None
    payload = json.dumps({"definition": item, "file_url": file_url},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_workload(dataset: str, marble_seed: str, runtime_env: dict | None = None):
    if runtime_env is not None:
        for name in ("MARBLE_BENCH_DIR", "MARBLE_MAX_TASKS_PER_SCENARIO"):
            if name in runtime_env:
                os.environ[name] = runtime_env[name]
            else:
                os.environ.pop(name, None)
    os.environ["AGENT_FAULT_DATASET"] = dataset
    os.environ["MARBLE_SAMPLE_SEED"] = marble_seed
    definitions, resources = load_dataset_definitions(dataset)
    return lambda service: (filter_definitions_for_service(definitions, service), resources)


def create_selection(folder: str, requested: str, task_count: int,
                     run_dir: Path) -> dict:
    dataset = FOLDERS[folder]
    if task_count < 1:
        raise ValueError("TASK_SAMPLE_COUNT must be a positive integer")
    edge_count = int(os.environ.get("EDGE_SERVICE_SAMPLE_COUNT", "3"))
    if edge_count < 1:
        raise ValueError("EDGE_SERVICE_SAMPLE_COUNT must be a positive integer")
    inject_fault = os.environ.get("INJECT_FAULT", "true").lower() not in {"false", "0", "no", "off"}
    services = services_for_run(folder, requested, inject_fault, edge_count)
    seed_text = os.environ.get("TASK_SELECTION_SEED", "").strip()
    seed = int(seed_text) if seed_text else secrets.randbits(63)
    marble_seed = os.environ.get("MARBLE_SAMPLE_SEED", "").strip() or str(seed % (2 ** 32))
    dynamic_zone = os.environ.get("DYNAMIC_FAULT_ZONE", "").strip()
    if not dynamic_zone:
        dynamic_zone = random.Random(seed).choice(("cloud", "edge"))
    if dynamic_zone not in {"cloud", "edge"}:
        raise ValueError("DYNAMIC_FAULT_ZONE must be cloud or edge")
    validate_planner_hosts(os.environ.get("AGENT_FAULT_PLANNER_HOSTS", ""))
    definitions_for_service = load_workload(dataset, marble_seed)
    available = edge_deployments(folder) - {PLANNER}
    explicit = os.environ.get("AGENT_FAULT_TASK_INDEX", "").strip()
    selected = {}
    for service in services:
        definitions, resources = definitions_for_service(service)
        if explicit:
            indices = parse_task_indices(explicit, len(definitions))
            if os.environ.get("TASK_SAMPLE_COUNT") and len(indices) != task_count:
                raise ValueError(f"{service}: TASK_SAMPLE_COUNT disagrees with AGENT_FAULT_TASK_INDEX")
        else:
            if task_count > len(definitions):
                raise ValueError(f"{service} has {len(definitions)} tasks; requested {task_count}")
            service_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{service}".encode()).digest()[:8], "big")
            indices = sorted(random.Random(service_seed).sample(range(len(definitions)), task_count))
        edge_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:edge-services:{service}".encode()).digest()[:8], "big")
        edge_services = choose_services(folder, available, service, edge_count,
                                        random.Random(edge_seed))
        selected[service] = {
            "pool_size": len(definitions),
            "indices": indices,
            "edge_services": edge_services,
            "tasks": [
                {"index": index, "name": definitions[index]["name"],
                 "sha256": task_fingerprint(definitions[index], resources)}
                for index in indices
            ],
        }
    run_id = run_dir.name
    return {
        "schema_version": 1,
        "created_at": now(),
        "folder": folder,
        "dataset": dataset,
        "requested_service": requested,
        "services": services,
        "modes": list(MODES),
        "task_sample_count": len(next(iter(selected.values()))["indices"]),
        "edge_service_sample_count": edge_count,
        "selection_seed": seed,
        "marble_sample_seed": marble_seed,
        "dynamic_fault_zone": dynamic_zone,
        "selection": selected,
        "runtime_env": {key: os.environ[key] for key in RUNTIME_KEYS if key in os.environ},
        "data_prefix_base": f"{dataset}/agent_faults/topology_runs/{run_id}",
    }


def verify_selection(plan: dict) -> None:
    definitions_for_service = load_workload(plan["dataset"], plan["marble_sample_seed"],
                                            plan["runtime_env"])
    available = edge_deployments(plan["folder"])
    for service in plan["services"]:
        saved = plan["selection"][service]
        definitions, resources = definitions_for_service(service)
        if len(definitions) != saved["pool_size"]:
            raise ValueError(f"{service}: task pool size changed since selection")
        for task in saved["tasks"]:
            index = task["index"]
            if task_fingerprint(definitions[index], resources) != task["sha256"]:
                raise ValueError(f"{service}: task at index {index} changed since selection")
        if any(name not in available for name in [PLANNER, *saved["edge_services"]]):
            raise ValueError(f"{service}: edge deployment choices changed since selection")


def run(plan: dict, run_dir: Path) -> int:
    progress_file = run_dir / "progress.json"
    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    completed = {tuple(pair) for pair in progress["completed"]}
    folder = plan["folder"]
    for mode in plan["modes"]:
        for service in plan["services"]:
            key = (mode, service)
            if key in completed:
                log(run_dir, f"SKIP completed {mode} {service}")
                continue
            entry = plan["selection"][service]
            indices = entry["indices"]
            env = dict(os.environ)
            for name in RUNTIME_KEYS:
                env.pop(name, None)
            env.update(plan["runtime_env"])
            for name in ("HOST_IP", "LOCUST_HOST", "PLANNER_NODEPORT",
                         "AGENT_FAULT_PLANNER_HOSTS"):
                if name in os.environ:
                    env[name] = os.environ[name]
            if mode != "dynamic":
                env.pop("AGENT_FAULT_PLANNER_HOSTS", None)
            env.update({
                "TOPOLOGY_MODE": mode,
                "FAULT_ZONE": "edge" if mode == "edge-preference" else
                              "cloud" if mode == "cloud-preference" else plan["dynamic_fault_zone"],
                "EDGE_SERVICES": " ".join(entry["edge_services"]),
                "EDGE_SERVICE_SAMPLE_COUNT": str(len(entry["edge_services"])),
                "AGENT_FAULT_TASK_INDEX": ",".join(map(str, indices)),
                "MARBLE_SAMPLE_SEED": plan["marble_sample_seed"],
                "DATA_PREFIX": f"{plan['data_prefix_base']}/{mode}/load-{env.get('LOCUST_USERS', '64')}",
            })
            wrapper = HERE / folder / f"sum_chaos_cloud_{service}.sh"
            fault_target = (env["FAULT_ZONE"] if env.get("INJECT_FAULT", "true").lower()
                            not in {"false", "0", "no", "off"} else "none")
            log(run_dir, f"START {mode} {service} indices={indices} fault_target_zone={fault_target} edge_services={entry['edge_services']}")
            with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
                result = subprocess.run(["bash", str(wrapper)], env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                log(run_dir, f"FAILED {mode} {service} exit={result.returncode}; resume with --resume {run_dir}")
                return result.returncode
            completed.add(key)
            progress["completed"] = [list(pair) for pair in sorted(completed)]
            save_json(progress_file, progress)
            log(run_dir, f"DONE {mode} {service}")
    log(run_dir, "ALL THREE TOPOLOGY MODES COMPLETE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", choices=FOLDERS)
    parser.add_argument("service", help="base agent-network service name, common, or all")
    parser.add_argument("--run-dir", type=Path, help="directory for the new selection and run log")
    parser.add_argument("--resume", type=Path, help="reuse a prior selection and skip completed pairs")
    parser.add_argument("--plan-only", action="store_true", help="save choices without starting workloads")
    args = parser.parse_args()
    if args.run_dir and args.resume:
        parser.error("--run-dir and --resume cannot be combined")
    if not args.resume and not (os.environ.get("HOST_IP") or os.environ.get("LOCUST_HOST") or args.plan_only):
        parser.error("set HOST_IP or LOCUST_HOST before starting workloads")
    if not args.resume and not args.plan_only and not (
        os.environ.get("HOST_IP") or os.environ.get("AGENT_FAULT_PLANNER_HOSTS")
    ):
        parser.error("dynamic mode requires HOST_IP or AGENT_FAULT_PLANNER_HOSTS")
    if args.resume:
        run_dir = args.resume.resolve()
        plan = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
        if plan["folder"] != args.folder or plan["requested_service"] != args.service:
            raise ValueError("resume folder/service differs from saved selection")
        if not args.plan_only and not any(
            os.environ.get(name) or plan["runtime_env"].get(name)
            for name in ("HOST_IP", "LOCUST_HOST")
        ):
            parser.error("set HOST_IP or LOCUST_HOST before starting workloads")
        if not args.plan_only and not any(
            os.environ.get(name) or plan["runtime_env"].get(name)
            for name in ("HOST_IP", "AGENT_FAULT_PLANNER_HOSTS")
        ):
            parser.error("dynamic mode requires HOST_IP or AGENT_FAULT_PLANNER_HOSTS")
        validate_planner_hosts(os.environ.get("AGENT_FAULT_PLANNER_HOSTS") or
                               plan["runtime_env"].get("AGENT_FAULT_PLANNER_HOSTS", ""))
        verify_selection(plan)
        log(run_dir, f"RESUME using saved task indices from {run_dir / 'selection.json'}")
    else:
        run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        run_id += f"-{secrets.token_hex(4)}"
        run_dir = (args.run_dir or DATA_ROOT / FOLDERS[args.folder] /
                   "agent_faults" / "topology_runs" / run_id).resolve()
        task_count = int(os.environ.get("TASK_SAMPLE_COUNT", "1"))
        plan = create_selection(args.folder, args.service, task_count, run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "selection.json", plan)
        save_json(run_dir / "progress.json", {"completed": []})
        log(run_dir, f"SAVED selection before first run: {run_dir / 'selection.json'}")
        if plan["runtime_env"].get("INJECT_FAULT", "true").lower() in {"false", "0", "no", "off"}:
            log(run_dir, "DYNAMIC planner requests use both zones; no fault target for normal data")
        else:
            log(run_dir, f"DYNAMIC fault target zone: {plan['dynamic_fault_zone']} (planner requests use both zones)")
        for service in plan["services"]:
            entry = plan["selection"][service]
            log(run_dir, f"SELECTED {service} indices={entry['indices']} edge_services={entry['edge_services']}")
    if args.plan_only:
        log(run_dir, "PLAN ONLY; no workloads started")
        return 0
    return run(plan, run_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"topology run failed: {error}", file=sys.stderr)
        raise SystemExit(2)
