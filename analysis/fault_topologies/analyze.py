#!/usr/bin/env python3
"""Real-data motivation-1 analysis: distribution and breadth figures, plus a table.

Run: MPLCONFIGDIR=/private/tmp/fault-topologies-matplotlib .venv/bin/python
     analysis/fault_topologies/analyze.py
"""
import bisect
import csv
import datetime as dt
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

from audit_data import DATA, NODE_ZONE, ROOT, labels

HERE = Path(__file__).resolve().parent
MODES = ("cloud-preference", "edge-preference", "dynamic")
MODE_LABEL = dict(zip(MODES, ("Cloud-first", "Edge-first", "Dynamic")))
CN_MODE = dict(zip(MODES, ("云优先", "边优先", "不区分")))
ZONE_CODE = {"cloud": "C", "edge": "E", "external": "External"}
SEGMENTS = ("C→C", "C→E", "E→C", "E→E", "C→External", "E→External")
TRACE_FILES = ("normal.pkl", "abnormal.pkl", "inbound.pkl", "outbound.pkl",
               "inbound_half.pkl", "outbound_half.pkl", "abnormal_half.pkl")


def service(raw):
    if not raw:
        return None
    if raw.startswith("AgentNetworkPlannerGroup/"):
        return "agent-network-planner"
    head = raw.split("/")[0].removesuffix("-edge")
    return head if head.startswith("agent-network-") else None


def failed(span):
    return span.get("status") == 3 or bool(span.get("error"))


def is_model_call(fire):
    """Identify the API, not a provider alias (qwen/deepseek can change)."""
    return (fire.get("path", "").rstrip("/").endswith("/chat/completions")
            and service(fire.get("upstream")) is None)


def source_zone(raw):
    if not service(raw):
        return "unknown"
    return "edge" if raw.split("/")[0].endswith("-edge") else "cloud"


def epoch(value):
    return int(re.search(r"\((\d{10})\)", value)[1])


def write_csv(relative, rows, fields=None):
    path = HERE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover():
    candidates = defaultdict(dict)
    for run in sorted((DATA / "topology_runs").iterdir()):
        if not run.is_dir():
            continue
        selection = json.loads((run / "selection.json").read_text())
        for path, meta in labels(run).items():
            if meta.get("fault_injection") != "true" or meta.get("load_users") != "64":
                continue
            if not (path / "agent-network/graph/call_chains.json").is_file():
                continue
            base = meta["service"].removesuffix("-edge")
            chosen = selection["selection"][base]
            assert sorted(map(int, meta["task_indices"].split(","))) == sorted(chosen["indices"])
            assert sorted(meta["edge_services"].split()) == sorted(chosen["edge_services"])
            group = (run.name, base, meta["fault_type"], meta["persistence"],
                     meta["sample"], meta["load_users"])
            mode = meta["topology_mode"]
            assert mode not in candidates[group], (group, mode)
            candidates[group][mode] = (path, meta)
    return {"|".join(key): value for key, value in sorted(candidates.items())
            if set(value) == set(MODES)}


def endpoint_maps(path):
    """Trace endpoints provide IP→Pod identities, not fire→source identities."""
    ip_pods = defaultdict(set)
    for name in TRACE_FILES:
        source = path / "agent-network/trace" / name
        if not source.is_file():
            continue
        # Collector uses append mode: consume all pickle objects, not only the first.
        with source.open("rb") as inp:
            while True:
                try:
                    batch = pickle.load(inp)
                except EOFError:
                    break
                for trace in batch:
                    for pair in trace.get("call_instance", []):
                        for endpoint in pair:
                            match = re.fullmatch(r"([^\.]+)\..*-(\d+\.\d+\.\d+\.\d+)", endpoint)
                            if match:
                                ip_pods[match[2]].add(match[1])
    pod_nodes = defaultdict(dict)
    source = path / "agent-network/metrics/graph.csv"
    with source.open() as inp:
        for row in csv.DictReader(inp):
            if not row["destination"].startswith("node-"):
                continue
            time = dt.datetime.fromisoformat(row["timestamp"]).replace(tzinfo=dt.timezone.utc).timestamp()
            observations = pod_nodes[row["source"]]
            assert time not in observations or observations[time] == row["destination"]
            observations[time] = row["destination"]
    return ip_pods, {pod: sorted(values.items()) for pod, values in pod_nodes.items()}


def resolve_target(ip, time, ip_pods, pod_nodes):
    candidates = []
    for pod in sorted(ip_pods.get(ip, [])):
        observations = pod_nodes.get(pod, [])
        if not observations:
            continue
        index = bisect.bisect_right([t for t, _ in observations], time) - 1
        if index >= 0 and 0 <= time - observations[index][0] <= 10:
            candidates.append((pod, observations[index][1]))
    if len(candidates) == 1:
        pod, node = candidates[0]
        return pod, node, "ip_trace_pod_deployment_node"
    return "", "", "ambiguous" if len(candidates) > 1 else "not_resolved"


def load_hits(group, mode, path, meta):
    ip_pods, pod_nodes = endpoint_maps(path)
    start, stop = epoch(meta["fault_start"]), epoch(meta["fault_end"])
    rows, seen = [], set()
    qc = Counter()
    for source in sorted(path.glob("chaos/log/*_fires.jsonl")):
        for line_no, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fire = json.loads(line)
            qc["fire_rows"] += 1
            if fire.get("effect_applied") is not True:
                qc["not_effect_applied_true"] += 1
                continue
            if not start <= fire["t"] < stop:
                qc["outside_fault_window"] += 1
                continue
            # A duplicate requires identical complete event content, not just task ID.
            fingerprint = hashlib.sha256(json.dumps(fire, sort_keys=True).encode()).hexdigest()
            if fingerprint in seen:
                qc["exact_duplicate_events"] += 1
                continue
            seen.add(fingerprint)
            headers = fire.get("routing_headers") or {}
            src_raw, upstream = fire.get("source_service", ""), fire.get("upstream", "")
            src, target = service(src_raw), service(upstream)
            src_zone = source_zone(src_raw)
            url = headers.get("X-Upstream-Url", "")
            match = re.search(r"(?:^|//)(\d+\.\d+\.\d+\.\d+)(?::|/|$)", url)
            ip = match[1] if match else ""
            pod, node, basis = resolve_target(ip, fire["t"], ip_pods, pod_nodes)
            tag_zone = headers.get("X-Agent-Target-Zone", "unknown")
            mapped_zone = NODE_ZONE.get(node, "unknown")
            if tag_zone in ("cloud", "edge") and mapped_zone in ("cloud", "edge"):
                assert tag_zone == mapped_zone, (source, line_no, tag_zone, mapped_zone)
            target_zone = mapped_zone if mapped_zone != "unknown" else tag_zone
            model_call = is_model_call(fire)
            if model_call:
                target_zone = "external"
            elif target is None:
                target_zone = "unknown"
            segment = (ZONE_CODE[src_zone] + "→" + ZONE_CODE[target_zone]
                       if src_zone in ZONE_CODE and target_zone in ZONE_CODE else "unknown")
            rows.append({
                "group_id": group, "mode": mode,
                "fire_id": str(source.relative_to(ROOT)) + ":" + str(line_no),
                "time": fire["t"], "fault_type": meta["fault_type"],
                "task_id": fire.get("taskId") or "",
                "source_service": src_raw, "source_zone": src_zone,
                "target_service": upstream, "target_ip": ip,
                "target_pod": pod, "target_node": node, "target_zone": target_zone,
                "perturbed_service": (src if model_call else target) or "unknown",
                "call_kind": "model" if model_call else "agent" if target else "unknown",
                "segment": segment, "target_mapping_basis": basis,
                "source_mapping_basis": "deployment_service_zone_only",
                "target_zone_basis": "external_dependency" if model_call else
                    "target_node" if mapped_zone != "unknown" else "routing_header",
                "fault_zone": meta["fault_zone"],
                "source_path": str(source.relative_to(ROOT)), "source_line": line_no,
            })
    qc["kept_hits"] = len(rows)
    qc["resolved_target_pod"] = sum(bool(row["target_pod"]) for row in rows)
    qc["agent_call_hits"] = sum(bool(service(row["target_service"])) for row in rows)
    qc["unknown_segment"] = sum(row["segment"] == "unknown" for row in rows)
    qc["model_upstreams_json"] = json.dumps(sorted({r["target_service"] for r in rows
                                                   if r["call_kind"] == "model"}))
    return rows, qc


def task_data(group, mode, path, hits, include_unhit=False):
    roots_by_task = defaultdict(set)
    for hit in hits:
        if hit["task_id"] and service(hit["target_service"]):
            roots_by_task[hit["task_id"]].add(service(hit["target_service"]))
    source = path / "agent-network/graph/call_chains.json"
    tasks = json.loads(source.read_text())
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    service_rows, edge_rows, task_rows, path_rows = [], [], [], []
    matched_ids = set()
    for task in tasks:
        tid = task["task_id"]
        if tid not in roots_by_task and not include_unhit:
            continue
        if tid in roots_by_task:
            matched_ids.add(tid)
        roots = roots_by_task.get(tid, set())
        info = defaultdict(lambda: {"final_failed": False, "any_attempt_failed": False,
                                    "evidence_vertices": set()})
        vertices = set()
        for level in task.get("levels", []):
            for raw_vertex in level.get("vertexes", []):
                if service(raw_vertex):
                    vertices.add(service(raw_vertex))
            for vertex, span in (level.get("spans") or {}).items():
                planned = service(vertex)
                if not planned:
                    continue
                vertices.add(planned)
                # Executor can differ from the planned service after functional fallback.
                # Attribute the actual attempt's error to executor, never to its alias.
                final_service = service(span.get("executor")) or planned
                info[planned]["evidence_vertices"].add(vertex)
                info[final_service]["evidence_vertices"].add(vertex)
                if failed(span):
                    info[final_service]["final_failed"] = True
                    info[final_service]["any_attempt_failed"] = True
                for attempt in span.get("attempts", []):
                    actual = service(attempt.get("executor")) or planned
                    info[actual]["evidence_vertices"].add(vertex)
                    if failed(attempt):
                        info[actual]["any_attempt_failed"] = True
        edges = set()
        for index, edge in enumerate(task.get("edges", [])):
            u, v = service(edge.get("source")), service(edge.get("destination"))
            if u:
                vertices.add(u)
            if v:
                vertices.add(v)
            if u and v and u != v:
                edges.add((u, v))
        common = {"group_id": group, "mode": mode, "task_id": tid}
        for s, values in sorted(info.items()):
            service_rows.append(common | {"service": s, "executed": True,
                "is_root": s in roots, "final_failed": values["final_failed"],
                "any_attempt_failed": values["any_attempt_failed"],
                "evidence_vertices_json": json.dumps(sorted(values["evidence_vertices"])),
                "graph_path": str(source.relative_to(ROOT))})
        for u, v in sorted(edges):
            edge_rows.append(common | {"source_service": u, "target_service": v,
                "graph_path": str(source.relative_to(ROOT)),
                "edge_records_json": json.dumps([i for i, e in enumerate(task.get("edges", []))
                    if (service(e.get("source")), service(e.get("destination"))) == (u, v)])})
        bad = {s for s, values in info.items() if values["any_attempt_failed"]}
        adjacency = defaultdict(set)
        for u, v in edges:
            adjacency[u].add(v)
        # Depth is undefined if any directly hit root is absent from the explicit graph.
        depth = 0 if roots and roots <= vertices else ""
        patterns = set()
        if depth != "":
            for root in sorted(roots):
                queue, seen = deque([(root, (root,))]), {root}
                while queue:
                    current, sequence = queue.popleft()
                    # Another independently hit root is not propagation beyond roots.
                    if current in bad - roots and len(sequence) > 1:
                        depth = max(depth, len(sequence) - 1)
                        patterns.add(sequence)
                    for nxt in sorted(adjacency[current]):
                        if nxt not in seen:
                            seen.add(nxt)
                            queue.append((nxt, sequence + (nxt,)))
        for sequence in sorted(patterns):
            path_rows.append(common | {"path_pattern": "→".join(sequence),
                "length_hops": len(sequence) - 1,
                "evidence": "explicit_dependency_candidate",
                "graph_path": str(source.relative_to(ROOT))})
        task_rows.append(common | {"B": len(bad), "B_out": len(bad - roots) if roots else "",
            "has_linked_fire": bool(roots),
            "R": len(bad) / len(info) if info else "", "D": depth,
            "n_services": len(info), "root_services_json": json.dumps(sorted(roots)),
            "failed_services_json": json.dumps(sorted(bad)),
            "roots_complete": bool(roots) and roots <= vertices, "n_path_patterns": len(patterns),
            "task_status": task.get("task_status"),
            "graph_path": str(source.relative_to(ROOT))})
    return service_rows, edge_rows, task_rows, path_rows, {
        "hit_task_ids": len(roots_by_task), "matched_hit_tasks": len(matched_ids),
        "n_graph_tasks": len(tasks),
        "unmatched_hit_tasks": len(set(roots_by_task) - matched_ids),
        "depth_valid_tasks": sum(row["D"] != "" for row in task_rows)}


def distributions(hits, groups):
    rows, matrices = [], {}
    for kind, key in [("service", "perturbed_service"), ("segment", "segment")]:
        categories = sorted({h[key] for h in hits if h[key] != "unknown"})
        if kind == "segment":
            categories = [s for s in SEGMENTS if s in categories]
        counters = {(g, m): Counter(h[key] for h in hits if h["group_id"] == g and h["mode"] == m)
                    for g in groups for m in MODES}
        valid_groups = [g for g in groups if all(sum(v for k, v in counters[g, m].items()
                         if k != "unknown") > 0 for m in MODES)]
        matrix = np.zeros((len(MODES), len(categories)))
        for mi, mode in enumerate(MODES):
            for ci, cat in enumerate(categories):
                values = []
                for group in valid_groups:
                    counts = counters[group, mode]
                    denominator = sum(v for k, v in counts.items() if k != "unknown")
                    numerator = counts[cat]
                    values.append(numerator / denominator)
                    rows.append({"group_id": group, "mode": mode, "kind": kind,
                        "category": cat, "numerator": numerator, "denominator": denominator,
                        "share": numerator / denominator, "unknown_count": counts["unknown"],
                        "aggregation": "within_group", "n_groups": 1, "groups_json": ""})
                value = float(np.mean(values))
                matrix[mi, ci] = value
                rows.append({"group_id": "", "mode": mode, "kind": kind, "category": cat,
                    "numerator": sum(values), "denominator": len(values), "share": value,
                    "unknown_count": sum(counters[g, mode]["unknown"] for g in valid_groups),
                    "aggregation": "equal_group", "n_groups": len(valid_groups),
                    "groups_json": json.dumps(valid_groups)})
        assert np.allclose(matrix.sum(axis=1), 1)
        matrices[kind] = (categories, matrix)
    return rows, matrices


def propagation_summary(tasks, paths, groups):
    from auxiliary import summarize
    return summarize(tasks, paths, groups)


def plot_heatmaps(matrices):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    services, a = matrices["service"]
    segments, b = matrices["segment"]
    short = {"agent-network-" + key: label for key, label in [
        ("planner", "Planner"), ("csv-gen", "CSV Gen"), ("direction", "Direction"),
        ("excel-gen", "Excel Gen"), ("excel-parsing", "Excel Parse"), ("image", "Image"),
        ("image-gen", "Image Gen"), ("ocr", "OCR"), ("pdf-gen", "PDF Gen"),
        ("pdf-parsing", "PDF Parse"), ("word-gen", "Word Gen"), ("word-parsing", "Word Parse")]}
    # Same color scale across both distributions; no per-panel rescaling.
    vmax = np.ceil(max(a.max(), b.max()) * 10) / 10
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("Blues")
    fig = plt.figure(figsize=(11.9, 3.6), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[len(services), len(segments), .28],
                           left=.115, right=.93, bottom=.37, top=.93, wspace=.12)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    for index, (ax, matrix, columns, caption) in enumerate(zip(axes, [a, b],
        [[short.get(s, s.removeprefix("agent-network-")) for s in services], segments],
        ["(a) Fault-service distribution", "(b) Network-segment distribution"])):
        # Explicit vector cells avoid the PDF backend's multi-image rendering issue.
        # Both exports use exactly these same cell colors and normalization.
        for (r, c), value in np.ndenumerate(matrix):
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1,
                                  facecolor=cmap(norm(value)), edgecolor="none"))
        ax.set_xlim(-.5, len(columns) - .5)
        ax.set_ylim(2.5, -.5)
        ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        ax.set_yticks(range(3), [MODE_LABEL[m] for m in MODES] if index == 0 else [""] * 3)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.set_xlabel(caption, labelpad=12)
        ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.45, alpha=.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for (r, c), value in np.ndenumerate(matrix):
            rgb = cmap(norm(value))[:3]
            light = np.dot(rgb, [.2126, .7152, .0722])
            ax.text(c, r, f"{value:.1%}", ha="center", va="center", fontsize=8.5,
                    color="black" if light > .48 else "white")
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
    axes[0].set_ylabel("Cooperation mode", labelpad=8)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(grid[2]))
    from matplotlib.ticker import PercentFormatter
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    cbar.set_label("Share of fault hits", labelpad=8)
    cbar.outline.set_linewidth(.7)
    for suffix in ("png", "pdf"):
        output = HERE / "figures" / ("fig1_fault_distribution." + suffix)
        output.parent.mkdir(exist_ok=True)
        fig.savefig(output, dpi=350, facecolor="white")
    plt.close(fig)


def write_report(summary, coverage, hits, groups, eligible):
    from auxiliary import write_tables
    table_lines = write_tables(summary)
    total_hits = len(hits)
    resolved = sum(int(c["resolved_target_pod"]) for c in coverage)
    agent_hits = sum(int(c["agent_call_hits"]) for c in coverage)
    with (HERE / "tables/distributions.csv").open() as inp:
        distribution = list(csv.DictReader(inp))
    n_distribution = {kind: next(r["n_groups"] for r in distribution
                        if r["kind"] == kind and r["aggregation"] == "equal_group")
                      for kind in ("service", "segment")}
    report = ["# 动机实验 1：真实数据结果", "",
        f"当前数据有 {len(groups)} 组完整三模式对照（{len(groups)*3} 个样本），"
        f"保留故障窗内实际生效命中 {total_hits:,} 次。图 1 服务／网络段分别采用 "
        f"{n_distribution['service']}／{n_distribution['segment']} 组，要求组内三模式均有有效命中分母；"
        "零分母组整体退出比例图，仍保留在箱线图和辅助表。先组内归一化，再对有效组等权平均；不是原始次数混池。",
        "", "![故障分布](figures/fig1_fault_distribution.png)", "",
        "图注：横轴为受扰服务或有向区域段，纵轴为协同模式，颜色／格内数值为命中份额，颜色越深占比越高。"
        "服务区域配置相同，具体节点并非严格固定；故障注入区域随优先模式改变。"
        "External 为模型 API 依赖（按 /chat/completions 识别，包括 qwen/deepseek）。"
        "部分新增 dynamic 样本的模型上游标识不同，底层模型一致性未经确认，不能视为固定模型条件下的纯策略因果对照。"
        "这里只比较故障命中分布，不解释为故障发生率或策略优劣。", "",
        f"target 映射：{agent_hits:,} 次 Agent 调用命中中 {resolved:,} 次恢复到目标 Pod 与节点；"
        "网络段仍可使用目标区域头。source 仅使用服务名标识区域，没有推断逐请求源 Pod。", "",
        "## 每次故障运行的影响广度", "",
        "![故障运行影响广度](figures/fig2_fault_breadth.png)", "",
        f"三项使用全部 {len(groups)} 组对照，每种模式各 {len(groups)} 次故障运行。"
        "服务数为该样本完整采集执行图中出现过程错误的服务并集，任务数为过程有错误或终态失败的任务数，"
        "网络段数为故障窗内真命中的已知区域段去重数。"
        "这是运行级异常观测与故障暴露统计，不要求模型 fire 中有 taskId，也不声称所有异常都由本次故障因果引起。"
        "慢故障的纯延迟和截断结果的语义损害不能被错误状态完全识别。每个散点为一次运行，箱体 Q1–Q3、横线中位数、须为 1.5 IQR 规则。"
        "网络段计数一致时箱体会退化为线。原始计数见 `tables/fault_run_breadth.csv`，分位数见 `tables/fault_run_breadth_summary.csv`。", "",
        "## 全部对照组的辅助指标表", "",
        f"表格纳入全部 {len(eligible)} 组；B 按完整采集图的全部任务计算。B_out/D 保留任务级故障根定义，"
        "模型故障等无法关联根的记录为 NA；每项显示有效组数，不把未知当成 0。", ""]
    report += table_lines
    report += ["", "B=每条任务中出现过程失败的逻辑服务数；B_out=其中直接命中根之外的服务数；"
        "D=根到下游非根异常服务的最短见证距离的最大值。B_out/D 的分母仅包含具备根证据的任务。",
        "", "各组均值和有效数见 tables/propagation_summary.csv；全部任务记录见 intermediate/run_task_metrics.csv。",
        "", "重跑：`MPLCONFIGDIR=/private/tmp/fault-topologies-matplotlib .venv/bin/python analysis/fault_topologies/analyze.py`。"]
    (HERE / "results.md").write_text("\n".join(report) + "\n")


def main():
    groups = discover()
    if not groups:
        raise ValueError("No complete three-mode fault groups")
    all_hits, coverage, services, edges, tasks, paths, run_tasks = [], [], [], [], [], [], []
    for group, modes in groups.items():
        for mode in MODES:
            path, meta = modes[mode]
            hits, qc = load_hits(group, mode, path, meta)
            sr, er, tr, pr, tq = task_data(group, mode, path, hits, include_unhit=True)
            hit_ids = {t["task_id"] for t in tr if t["has_linked_fire"]}
            all_hits.extend(hits)
            run_tasks.extend(tr)
            services.extend(r for r in sr if r["task_id"] in hit_ids)
            edges.extend(r for r in er if r["task_id"] in hit_ids)
            tasks.extend(t for t in tr if t["has_linked_fire"])
            paths.extend(pr)
            coverage.append({"group_id": group, "mode": mode,
                "sample_path": str(path.relative_to(ROOT)),
                "fault_type": meta["fault_type"], "persistence": meta["persistence"],
                "fault_start": epoch(meta["fault_start"]), "fault_end": epoch(meta["fault_end"]),
                "window_start": epoch(meta["window_start"]), "window_end": epoch(meta["window_end"]),
                **dict(qc), **tq})
            print(group.split("|")[0], mode, len(hits), "hits", tq["matched_hit_tasks"],
                  "matched tasks", len(tr), "graph tasks", flush=True)
    distribution_rows, matrices = distributions(all_hits, groups)
    summary, propagation_rows, eligible = propagation_summary(run_tasks, paths, groups)
    for name, rows in [("intermediate/hits.csv", all_hits),
        ("intermediate/task_services.csv", services), ("intermediate/dependency_edges.csv", edges),
        ("intermediate/task_metrics.csv", tasks), ("intermediate/propagation_paths.csv", paths),
        ("intermediate/run_task_metrics.csv", run_tasks),
        ("tables/coverage.csv", coverage), ("tables/distributions.csv", distribution_rows),
        ("tables/propagation_summary.csv", summary), ("tables/propagation.csv", propagation_rows)]:
        write_csv(name, rows)
    plot_heatmaps(matrices)
    from breadth import write_outputs
    write_outputs(coverage, all_hits, run_tasks, groups)
    write_report(summary, coverage, all_hits, groups, eligible)
    from build_readme import write_readme
    write_readme()
    print(json.dumps({"complete_groups": len(groups), "in_window_hits": len(all_hits),
        "propagation_groups": len(eligible),
        "means": [s for s in summary if s["aggregation"] == "equal_group"]}, indent=2))


if __name__ == "__main__":
    main()
