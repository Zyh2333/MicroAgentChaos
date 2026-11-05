#!/usr/bin/env python3
"""Run and incrementally resume sampled faults across cloud/edge topologies."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
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
FAULT_MODES = ("single", "persistent", "intermittent", "burst")
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


def topology_modes(raw: str | None) -> list[str]:
    if raw is None:
        return list(MODES)
    modes = [item for item in re.split(r"[\s,]+", raw.strip()) if item]
    if not modes or any(item not in MODES for item in modes):
        raise ValueError(f"TOPOLOGY_MODES must list modes from {', '.join(MODES)}")
    return list(dict.fromkeys(modes))


def fault_modes(runtime: dict) -> list[str]:
    value = runtime.get("FAULT_MODE", "")
    if value and value not in FAULT_MODES:
        raise ValueError(f"FAULT_MODE must be one of {', '.join(FAULT_MODES)}")
    return [value] if value else list(FAULT_MODES)


def faults_for_service(folder: str, service: str, runtime: dict) -> list[str]:
    if runtime.get("INJECT_FAULT", "true").lower() in {"false", "0", "no", "off"}:
        return ["normal"]
    raw = runtime.get("FAULT_TYPES")
    if raw is None:
        wrapper = HERE / folder / f"sum_chaos_cloud_{service}.sh"
        match = re.search(r'^\s*read -r -a faults <<< "\$\{FAULT_TYPES:-([^}]*)\}"',
                          wrapper.read_text(encoding="utf-8"), re.M)
        if not match:
            raise ValueError(f"cannot read default fault types from {wrapper}")
        faults = match.group(1).split()
    else:
        faults = [item for item in re.split(r"[\s,]+", raw.strip()) if item]
    if not faults or len(faults) != len(set(faults)) or any(
        not re.fullmatch(r"f[1-4]_[a-z0-9_]+", item) for item in faults
    ):
        raise ValueError("FAULT_TYPES must contain distinct valid fault names")
    return faults


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


def select_service(folder: str, service: str, task_count: int, edge_count: int,
                   seed: int, definitions_for_service, explicit: str = "") -> dict:
    available = edge_deployments(folder) - {PLANNER}
    definitions, resources = definitions_for_service(service)
    if explicit:
        indices = parse_task_indices(explicit, len(definitions))
        if len(indices) != task_count:
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
    return {
        "pool_size": len(definitions),
        "indices": indices,
        "edge_services": edge_services,
        "tasks": [
            {"index": index, "name": definitions[index]["name"],
             "sha256": task_fingerprint(definitions[index], resources)}
            for index in indices
        ],
    }


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
    explicit = os.environ.get("AGENT_FAULT_TASK_INDEX", "").strip()
    if explicit and not os.environ.get("TASK_SAMPLE_COUNT"):
        first_definitions, _ = definitions_for_service(services[0])
        task_count = len(parse_task_indices(explicit, len(first_definitions)))
    selected = {}
    for service in services:
        selected[service] = select_service(folder, service, task_count, edge_count,
                                           seed, definitions_for_service, explicit)
    run_id = run_dir.name
    return {
        "schema_version": 1,
        "created_at": now(),
        "folder": folder,
        "dataset": dataset,
        "requested_service": requested,
        "services": services,
        "modes": topology_modes(os.environ.get("TOPOLOGY_MODES")),
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


def sample_service(plan: dict, mode: str, service: str) -> str:
    zone = ("edge" if mode == "edge-preference" else "cloud" if mode == "cloud-preference"
            else plan["dynamic_fault_zone"])
    return f"{service}-edge" if zone == "edge" else service


def reconcile_progress(plan: dict, run_dir: Path, progress: dict) -> dict:
    """Import finished samples from old progress and recover a post-collection crash."""
    if isinstance(progress.get("completed"), list):
        progress = {"schema_version": 2, "completed": {}}
    completed = progress["completed"]
    load_users = plan.get("runtime_env", {}).get("LOCUST_USERS", "64")
    for mode in MODES:
        for service in plan["services"]:
            target = sample_service(plan, mode, service)
            base = run_dir / mode / f"load-{load_users}"
            if service == "common":
                base /= target
            if not base.is_dir():
                continue
            for persistence in (*FAULT_MODES, "normal"):
                parent = base if persistence == "normal" else base / persistence
                name = service if persistence == "normal" and mode == "dynamic" else target
                label = parent / f"{name}_label.txt"
                if not label.is_file():
                    continue
                labeled = set(re.findall(r"^=== (.+?) ===$", label.read_text(encoding="utf-8"), re.M))
                counts: dict[str, int] = {}
                for sample in parent.iterdir():
                    if not sample.is_dir() or sample.name not in labeled:
                        continue
                    match = re.fullmatch(rf"{re.escape(name)}_(f[1-4]_[a-z0-9_]+|normal)_(\d+)", sample.name)
                    if not match or not (sample / "task.log").is_file() or not (
                        sample / "agent-network/graph/call_chains.json").is_file():
                        continue
                    fault = match.group(1)
                    if persistence != "normal" and not (sample / "proxy_stats.json").is_file():
                        continue
                    counts[fault] = counts.get(fault, 0) + 1
                for fault, count in counts.items():
                    leaf = completed.setdefault(mode, {}).setdefault(service, {}).setdefault(fault, {})
                    leaf[persistence] = max(leaf.get(persistence, 0), count)
    return progress


def runtime_for_resume(plan: dict) -> dict:
    runtime = dict(plan["runtime_env"])
    for name in RUNTIME_KEYS:
        if name in os.environ and name not in {"MARBLE_BENCH_DIR", "MARBLE_MAX_TASKS_PER_SCENARIO"}:
            runtime[name] = os.environ[name]
    if runtime.get("LOCUST_USERS", "64") != plan["runtime_env"].get("LOCUST_USERS", "64"):
        raise ValueError("LOCUST_USERS cannot change within one run directory")
    return runtime


def run(plan: dict, run_dir: Path, selected_modes: list[str], selected_services: list[str],
        runtime: dict) -> int:
    progress_file = run_dir / "progress.json"
    progress = reconcile_progress(plan, run_dir, json.loads(progress_file.read_text(encoding="utf-8")))
    save_json(progress_file, progress)
    completed = progress["completed"]
    folder = plan["folder"]
    target_count = int(runtime.get("SAMPLES_PER_FAULT", "1"))
    if target_count < 1:
        raise ValueError("SAMPLES_PER_FAULT must be a positive integer")
    injecting = runtime.get("INJECT_FAULT", "true").lower() not in {"false", "0", "no", "off"}
    for mode in selected_modes:
        for service in selected_services:
            entry = plan["selection"][service]
            indices = entry["indices"]
            wrapper = HERE / folder / f"sum_chaos_cloud_{service}.sh"
            zone = "edge" if mode == "edge-preference" else "cloud" if mode == "cloud-preference" else plan["dynamic_fault_zone"]
            for fault in faults_for_service(folder, service, runtime):
                for persistence in fault_modes(runtime) if injecting else ["normal"]:
                    previous = completed.get(mode, {}).get(service, {}).get(fault, {}).get(persistence, 0)
                    remaining = target_count - previous
                    if remaining <= 0:
                        log(run_dir, f"SKIP completed {mode} {service} {fault} {persistence} samples={previous}")
                        continue
                    if injecting:
                        manifest = HERE / folder / sample_service(plan, mode, service) / persistence / f"{fault}.yaml"
                        if not manifest.is_file():
                            raise ValueError(f"missing fault manifest: {manifest}")
                    env = dict(os.environ)
                    for name in RUNTIME_KEYS:
                        env.pop(name, None)
                    env.update(runtime)
                    if mode != "dynamic":
                        env.pop("AGENT_FAULT_PLANNER_HOSTS", None)
                    env.update({
                        "TOPOLOGY_MODE": mode, "FAULT_ZONE": zone,
                        "EDGE_SERVICES": " ".join(entry["edge_services"]),
                        "EDGE_SERVICE_SAMPLE_COUNT": str(len(entry["edge_services"])),
                        "AGENT_FAULT_TASK_INDEX": ",".join(map(str, indices)),
                        "MARBLE_SAMPLE_SEED": plan["marble_sample_seed"],
                        "DATA_PREFIX": f"{plan['data_prefix_base']}/{mode}/load-{env.get('LOCUST_USERS', '64')}",
                        "FAULT_TYPES": fault, "SAMPLES_PER_FAULT": str(remaining),
                    })
                    if injecting:
                        env["FAULT_MODE"] = persistence
                    log(run_dir, f"START {mode} {service} {fault} {persistence} samples={previous}+{remaining} indices={indices} fault_target_zone={zone if injecting else 'none'} edge_services={entry['edge_services']}")
                    with (run_dir / "run.log").open("a", encoding="utf-8") as stream:
                        result = subprocess.run(["bash", str(wrapper)], env=env, stdout=stream,
                                                stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        progress = reconcile_progress(plan, run_dir, progress)
                        save_json(progress_file, progress)
                        log(run_dir, f"FAILED {mode} {service} {fault} {persistence} exit={result.returncode}; resume with --resume {run_dir}")
                        return result.returncode
                    leaf = completed.setdefault(mode, {}).setdefault(service, {}).setdefault(fault, {})
                    leaf[persistence] = previous + remaining
                    save_json(progress_file, progress)
                    log(run_dir, f"DONE {mode} {service} {fault} {persistence} samples={leaf[persistence]}")
    log(run_dir, "SELECTED TOPOLOGY/SERVICE/FAULT COMBINATIONS COMPLETE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", choices=FOLDERS)
    parser.add_argument("service", help="base agent-network service name, common, or all")
    parser.add_argument("--run-dir", type=Path, help="directory for the new selection and run log")
    parser.add_argument("--resume", type=Path, help="reuse saved task selection and skip completed fault samples")
    parser.add_argument("--plan-only", action="store_true", help="save choices without starting workloads")
    args = parser.parse_args()
    if args.run_dir and args.resume:
        parser.error("--run-dir and --resume cannot be combined")
    selected_modes = topology_modes(os.environ.get("TOPOLOGY_MODES"))
    if not args.resume and not args.plan_only and any(mode != "dynamic" for mode in selected_modes) and not (
        os.environ.get("HOST_IP") or os.environ.get("LOCUST_HOST")
    ):
        parser.error("set HOST_IP or LOCUST_HOST for preference topologies")
    if not args.resume and not args.plan_only and "dynamic" in selected_modes and not (
        os.environ.get("HOST_IP") or os.environ.get("AGENT_FAULT_PLANNER_HOSTS")
    ):
        parser.error("dynamic mode requires HOST_IP or AGENT_FAULT_PLANNER_HOSTS")
    if args.resume:
        run_dir = args.resume.resolve()
        plan = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
        if plan["folder"] != args.folder:
            raise ValueError("resume folder differs from saved selection")
        runtime = runtime_for_resume(plan)
        inject_fault = runtime.get("INJECT_FAULT", "true").lower() not in {"false", "0", "no", "off"}
        selected_services = services_for_run(args.folder, args.service, inject_fault,
                                             plan["edge_service_sample_count"])
        if not args.plan_only and any(mode != "dynamic" for mode in selected_modes) and not any(
            runtime.get(name)
            for name in ("HOST_IP", "LOCUST_HOST")
        ):
            parser.error("set HOST_IP or LOCUST_HOST for preference topologies")
        if not args.plan_only and "dynamic" in selected_modes and not any(
            runtime.get(name)
            for name in ("HOST_IP", "AGENT_FAULT_PLANNER_HOSTS")
        ):
            parser.error("dynamic mode requires HOST_IP or AGENT_FAULT_PLANNER_HOSTS")
        validate_planner_hosts(runtime.get("AGENT_FAULT_PLANNER_HOSTS", ""))
        verify_selection(plan)
        missing_services = [name for name in selected_services if name not in plan["selection"]]
        if missing_services:
            definitions_for_service = load_workload(plan["dataset"], plan["marble_sample_seed"],
                                                    plan["runtime_env"])
            for name in missing_services:
                plan["selection"][name] = select_service(
                    args.folder, name, plan["task_sample_count"],
                    plan["edge_service_sample_count"], plan["selection_seed"],
                    definitions_for_service)
                plan["services"].append(name)
            verify_selection(plan)
            save_json(run_dir / "selection.json", plan)
            log(run_dir, f"EXTENDED selection with services={missing_services}")
        plan["modes"] = list(dict.fromkeys([*plan["modes"], *selected_modes]))
        save_json(run_dir / "selection.json", plan)
        progress_file = run_dir / "progress.json"
        save_json(progress_file, reconcile_progress(
            plan, run_dir, json.loads(progress_file.read_text(encoding="utf-8"))))
        log(run_dir, f"RESUME using saved task indices from {run_dir / 'selection.json'}")
    else:
        run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        run_id += f"-{secrets.token_hex(4)}"
        run_dir = (args.run_dir or DATA_ROOT / FOLDERS[args.folder] /
                   "agent_faults" / "topology_runs" / run_id).resolve()
        task_count = int(os.environ.get("TASK_SAMPLE_COUNT", "1"))
        plan = create_selection(args.folder, args.service, task_count, run_dir)
        selected_services = plan["services"]
        runtime = plan["runtime_env"]
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "selection.json", plan)
        save_json(run_dir / "progress.json", {"schema_version": 2, "completed": {}})
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
    return run(plan, run_dir, selected_modes, selected_services, runtime)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"topology run failed: {error}", file=sys.stderr)
        raise SystemExit(2)
