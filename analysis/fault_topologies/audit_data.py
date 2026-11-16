#!/usr/bin/env python3
"""Read-only source audit for the motivation-1 design; not the final analysis.

Run from any directory: python3 analysis/fault_topologies/audit_data.py
Only writes audit CSVs into this directory's tables/ folder.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "benchmark/data/MDOC/agent_faults"
MODES = ("edge-preference", "cloud-preference", "dynamic")
# Documented node labels in benchmark/failure_injection/CLOUD_EDGE_TOPOLOGY.md.
NODE_ZONE = {**dict.fromkeys(("node-15", "node-171", "node-227"), "cloud"),
             **dict.fromkeys(("node-219", "node-221"), "edge")}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def labels(base):
    """Resolve repeated aggregate/per-persistence blocks to real sample paths."""
    result = {}
    for path in sorted(base.rglob("*label.txt")):
        current = None
        for line in path.read_text().splitlines() + ["=== END ==="]:
            match = re.fullmatch(r"=== (\S+) ===", line)
            if match:
                if current:
                    name, meta = current
                    persistence = meta.get("persistence", "")
                    parent = path.parent
                    if parent.name != persistence and persistence:
                        parent /= persistence
                    sample = parent / name
                    if sample in result and result[sample] != meta:
                        raise ValueError(f"Conflicting duplicate labels: {sample}")
                    result[sample] = meta
                current = (match[1], {}) if match[1] != "END" else None
            elif current and ":" in line:
                key, value = line.split(":", 1)
                current[1][key.strip()] = value.strip()
    return result


def write_csv(name, rows):
    path = HERE / "tables" / name
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_audit(path, meta, cohort, run_id, selection):
    graph = path / "agent-network/graph/call_chains.json"
    chains = json.loads(graph.read_text()) if graph.exists() else []
    task_ids = {c.get("task_id") for c in chains} - {None, ""}
    fires = []
    for source in sorted(path.glob("chaos/log/*_fires.jsonl")):
        for line in source.read_text().splitlines():
            if line.strip():
                fires.append(json.loads(line))
    true_fires = [f for f in fires if f.get("effect_applied") is True]
    mapped = {f.get("taskId") for f in true_fires if f.get("taskId")} & task_ids
    spans = [s for c in chains for level in c.get("levels", [])
             for s in (level.get("spans") or {}).values()]
    targets = Counter(f.get("upstream", "unknown") for f in true_fires)
    target_zones = Counter((f.get("routing_headers") or {}).get(
        "X-Agent-Target-Zone", "unknown") for f in true_fires)
    placements = set()
    placement_source = path / "agent-network/metrics/graph.csv"
    if placement_source.exists():
        with placement_source.open() as source:
            for row in csv.DictReader(source):
                if row["destination"].startswith("node-"):
                    # Retain deployment and edge suffix, strip ReplicaSet hash/Pod id.
                    deployment = re.sub(r"-[a-z0-9]+-[a-z0-9]+$", "", row["source"])
                    placements.add((deployment, row["destination"]))
    service = meta.get("service", path.parent.parent.name).removesuffix("-edge")
    selected = selection.get("selection", {}).get(service, {})
    return {
        "cohort": cohort, "run_id": run_id,
        "topology_mode": meta.get("topology_mode", "unrecorded"),
        "sample_path": str(path.relative_to(ROOT)), "directory_exists": path.is_dir(),
        "label_present": bool(meta), "graph_present": graph.exists(),
        "service": service, "fault_type": meta.get("fault_type", "unrecorded"),
        "persistence": meta.get("persistence", "none" if meta.get("fault_type") == "normal" else path.parent.name),
        "sample": meta.get("sample", "unrecorded"),
        "fault_injection": meta.get("fault_injection", "unrecorded"),
        "fault_zone": meta.get("fault_zone", "unrecorded"),
        "load_users": meta.get("load_users", "unrecorded"),
        "task_set_hash": digest(sorted(t["sha256"] for t in selected.get("tasks", []))) if selected else "",
        "edge_set_hash": digest(sorted(selected.get("edge_services", []))) if selected else "",
        "label_edge_services": meta.get("edge_services", ""),
        "task_indices": meta.get("task_indices", ""),
        "placement_hash": digest(sorted(placements)) if placements else "",
        "deployment_zone_hash": digest(sorted({(d, NODE_ZONE.get(n, "unknown")) for d, n in placements})) if placements else "",
        "unmapped_placement_nodes": ";".join(sorted({n for _, n in placements if n not in NODE_ZONE})),
        "deployment_node_pairs": len(placements),
        "n_graph_tasks": len(chains),
        "task_status_counts": json.dumps(dict(sorted(Counter(str(c.get("task_status")) for c in chains).items()))),
        "n_graph_spans": len(spans),
        "n_failed_spans_proxy": sum(s.get("status") == 3 or bool(s.get("error")) for s in spans),
        "n_caller_spans": sum(len(c.get("caller_spans", [])) for c in chains),
        "n_fire_rows": len(fires), "n_effect_applied_true": len(true_fires),
        "n_effect_applied_false": sum(f.get("effect_applied") is False for f in fires),
        "n_effect_applied_unspecified": sum("effect_applied" not in f for f in fires),
        "n_true_fire_rows_with_task_id": sum(bool(f.get("taskId")) for f in true_fires),
        "n_true_fire_rows_with_target_zone": sum(bool((f.get("routing_headers") or {}).get("X-Agent-Target-Zone")) for f in true_fires),
        "n_true_fire_graph_hit_tasks": len(mapped),
        "true_fire_targets": json.dumps(dict(sorted(targets.items()))),
        "true_fire_target_zones": json.dumps(dict(sorted(target_zones.items()))),
    }


def main():
    samples, runs = [], []
    for run in sorted((DATA / "topology_runs").iterdir()):
        if not run.is_dir():
            continue
        selection = json.loads((run / "selection.json").read_text())
        metadata = labels(run)
        candidates = set(metadata)
        candidates.update(p.parents[2] for p in run.rglob("call_chains.json"))
        candidates.update(p.parent for p in run.rglob("task.log"))
        batch = [sample_audit(p, metadata.get(p, {}), "topology_runs", run.name, selection)
                 for p in sorted(candidates)]
        samples.extend(batch)
        row = {"run_id": run.name, "selected_modes": ",".join(selection["modes"]),
               "selected_task_count": selection["task_sample_count"],
               "edge_services": ";".join(next(iter(selection["selection"].values()))["edge_services"]),
               "dynamic_fault_zone": selection["dynamic_fault_zone"]}
        for mode in MODES:
            present = [s for s in batch if s["topology_mode"] == mode]
            row[mode + "_graphs"] = sum(s["graph_present"] for s in present)
            row[mode + "_faults"] = ";".join(sorted({s["fault_type"] for s in present if s["graph_present"]}))
        runs.append(row)
    common = DATA / "abnormal/load-64/common"
    metadata = labels(common)
    candidates = set(metadata) | {p.parents[2] for p in common.rglob("call_chains.json")}
    candidates.update(p.parent for p in common.rglob("task.log"))
    samples.extend(sample_audit(p, metadata.get(p, {}), "legacy_common", "legacy_common", {})
                   for p in sorted(candidates))
    write_csv("audit_samples.csv", samples)
    write_csv("audit_runs.csv", runs)
    groups = defaultdict(dict)
    for s in samples:
        if s["cohort"] == "topology_runs" and s["graph_present"]:
            key = (s["run_id"], s["service"], s["fault_type"], s["persistence"], s["sample"], s["load_users"])
            if s["topology_mode"] in groups[key]:
                raise ValueError(f"Duplicate mode in pairing group: {key}")
            groups[key][s["topology_mode"]] = s
    pairs = []
    for key, modes in sorted(groups.items()):
        values = list(modes.values())
        pairs.append(dict(zip(("run_id", "service", "fault_type", "persistence", "sample", "load_users"), key)) | {
            "available_modes": ";".join(m for m in MODES if m in modes),
            "complete_triplet": len(modes) == 3,
            "same_task_set": len({s["task_set_hash"] for s in values}) == 1 and all(s["task_set_hash"] for s in values),
            "same_edge_set": len({s["edge_set_hash"] for s in values}) == 1 and all(s["edge_set_hash"] for s in values),
            "same_observed_deployment_node_pairs": len({s["placement_hash"] for s in values}) == 1 and all(s["placement_hash"] for s in values),
            "same_observed_deployment_zone_pairs": len({s["deployment_zone_hash"] for s in values}) == 1 and all(s["deployment_zone_hash"] for s in values),
            "fault_zones": ";".join(m + ":" + modes[m]["fault_zone"] for m in MODES if m in modes),
        })
    write_csv("audit_pairs.csv", pairs)
    outcomes = []
    for pair in pairs:
        if not pair["complete_triplet"]:
            continue
        for mode in MODES:
            sample = groups[tuple(pair[k] for k in (
                "run_id", "service", "fault_type", "persistence", "sample", "load_users"))][mode]
            counts = json.loads(sample["task_status_counts"])
            terminal = counts.get("0", 0) + counts.get("7", 0)
            outcomes.append({k: sample[k] for k in (
                "run_id", "topology_mode", "fault_type", "persistence", "fault_zone", "load_users",
                "n_graph_tasks", "n_graph_spans", "n_failed_spans_proxy", "n_effect_applied_true",
                "n_true_fire_graph_hit_tasks")} | {
                    "window_basis": "entire_collected_graph_unadjusted",
                    "n_status_0": counts.get("0", 0), "n_status_7": counts.get("7", 0),
                    "task_fail_rate_terminal": counts.get("0", 0) / terminal if terminal else "",
                })
    write_csv("audit_triplet_outcomes.csv", outcomes)
    print(json.dumps({"run_count": len(runs), "sample_count": len(samples),
                      "topology_graph_samples": sum(s["cohort"] == "topology_runs" and s["graph_present"] for s in samples),
                      "legacy_graph_samples": sum(s["cohort"] == "legacy_common" and s["graph_present"] for s in samples),
                      "complete_triplets": sum(p["complete_triplet"] for p in pairs),
                      "complete_fault_triplets": sum(p["complete_triplet"] and p["fault_type"] != "normal" for p in pairs)}, indent=2))


if __name__ == "__main__":
    main()
