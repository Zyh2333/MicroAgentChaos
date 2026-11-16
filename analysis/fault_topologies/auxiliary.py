"""All-group auxiliary table with explicit per-metric coverage and missing roots."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODES = ("cloud-preference", "edge-preference", "dynamic")
CN = ("云优先", "边优先", "不区分")
EN = ("Cloud-first", "Edge-first", "Dynamic")
METRICS = ("B", "B_out", "D")


def number(value):
    return None if value is None or value == "" else float(value)


def summarize(tasks, paths, groups):
    group_ids = list(groups)
    by_sample = defaultdict(list)
    for task in tasks:
        by_sample[task["group_id"], task["mode"]].append(task)
    summary, details = [], []
    for group in group_ids:
        for mode in MODES:
            selected = by_sample[group, mode]
            values = {metric: [number(t.get(metric)) for t in selected
                               if number(t.get(metric)) is not None] for metric in METRICS}
            stats = {"group_id": group, "mode": mode, "aggregation": "within_group",
                "n_groups": 1, "n_tasks": len(selected),
                "n_hit_tasks": sum(t.get("has_linked_fire") in (True, "True") for t in selected),
                "n_B_valid": len(values["B"]), "n_B_out_valid": len(values["B_out"]),
                "n_depth_valid": len(values["D"]),
                "n_tasks_with_downstream_path": sum(int(t.get("n_path_patterns", 0)) > 0 for t in selected),
                "groups_json": json.dumps([group])}
            for metric in METRICS:
                v = values[metric]
                stats.update({"sum_" + metric: sum(v),
                    "mean_" + metric: float(np.mean(v)) if v else "",
                    "max_" + metric: max(v) if v else "",
                    "n_groups_" + metric: int(bool(v)),
                    "groups_" + metric + "_json": json.dumps([group] if v else [])})
            summary.append(stats)
            for task in selected:
                details.append({"group_id": group, "mode": mode, "kind": "task_metrics",
                    "task_id": task["task_id"], "B": task.get("B", ""),
                    "B_out": task.get("B_out", ""), "D": task.get("D", ""),
                    "has_linked_fire": task.get("has_linked_fire", False),
                    "aggregation": "within_group"})
    for mode in MODES:
        selected = [s for s in summary if s["mode"] == mode]
        row = {"group_id": "", "mode": mode, "aggregation": "equal_group",
            "n_groups": len(group_ids), "n_tasks": sum(s["n_tasks"] for s in selected),
            "n_hit_tasks": sum(s["n_hit_tasks"] for s in selected),
            "n_B_valid": sum(s["n_B_valid"] for s in selected),
            "n_B_out_valid": sum(s["n_B_out_valid"] for s in selected),
            "n_depth_valid": sum(s["n_depth_valid"] for s in selected),
            "n_tasks_with_downstream_path": sum(s["n_tasks_with_downstream_path"] for s in selected),
            "groups_json": json.dumps(group_ids)}
        for metric in METRICS:
            valid = [s for s in selected if s["mean_" + metric] != ""]
            row.update({"sum_" + metric: sum(s["sum_" + metric] for s in selected),
                "mean_" + metric: float(np.mean([s["mean_" + metric] for s in valid])) if valid else "",
                "max_" + metric: max(s["max_" + metric] for s in valid) if valid else "",
                "n_groups_" + metric: len(valid),
                "groups_" + metric + "_json": json.dumps([s["group_id"] for s in valid])})
        summary.append(row)
    for path in paths:
        if path["group_id"] in group_ids:
            details.append(path | {"kind": "path_pattern", "aggregation": "within_group"})
    return summary, details, group_ids


def display(row, metric):
    value = row["mean_" + metric]
    return "NA" if value == "" else f"{float(value):.3f}"


def write_tables(summary):
    aggregate = [next(s for s in summary if s["mode"] == mode and s["aggregation"] == "equal_group")
                 for mode in MODES]
    lines = ["| 协同模式 | 总组数 | 采集任务数 | 平均 B（有效组） | 平均 B_out（有效组） | 平均 D（有效组） |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in zip(CN, aggregate):
        cells = [f"{display(row, metric)}（{row['n_groups_'+metric]}/{row['n_groups']}）"
                 for metric in METRICS]
        lines.append(f"| {name} | {row['n_groups']} | {row['n_tasks']} | " + " | ".join(cells) + " |")
    lines += ["", "表覆盖全部对照，先组内平均，再对指标可计算的组等权。B 使用完整采集图的全部任务；"
              "B_out 和 D 仅使用有任务级根证据的任务，缺失值保持 NA，不补 0。括号为有效组数／总组数。",
              "", "| 协同模式 | B 有效任务 | B_out 有效任务 | D 有效任务 |",
              "| --- | ---: | ---: | ---: |"]
    for name, row in zip(CN, aggregate):
        lines.append(f"| {name} | {row['n_B_valid']} | {row['n_B_out_valid']} | {row['n_depth_valid']} |")
    (HERE / "tables/propagation_summary.md").write_text("\n".join(lines) + "\n")
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Mode & Groups & Tasks & $\overline{B}$ (groups) & $\overline{B}_{\mathrm{out}}$ (groups) & $\overline{D}$ (groups) \\",
           r"\midrule"]
    for name, row in zip(EN, aggregate):
        cells = [f"{display(row, metric)} ({row['n_groups_'+metric]}/{row['n_groups']})" for metric in METRICS]
        tex.append(f"{name} & {row['n_groups']} & {row['n_tasks']} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}",
            "% B: all collected tasks. B_out and D: only tasks with root evidence.",
            "% Group means are averaged equally over available groups; missing is not zero."]
    (HERE / "tables/propagation_summary.tex").write_text("\n".join(tex) + "\n")
    return lines
