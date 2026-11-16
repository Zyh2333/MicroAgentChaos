"""All-run observed symptoms and fault exposure, without per-hit causal claims.

Every boxplot observation is one sample, not one task or one dataset directory.
Service/task errors use the whole collected graph; segment counts use true fires.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
MODES = ("cloud-preference", "edge-preference", "dynamic")
LABELS = ("Cloud-first", "Edge-first", "Dynamic")
COLORS = ("#BADCCB", "#66B5A4", "#287F9D")
METRICS = ("n_affected_services", "n_affected_tasks", "n_affected_segments")


def compute_breadth(coverage, hits, task_metrics, eligible_groups):
    """Use every provided graph task, even if fires have no taskId (model faults)."""
    eligible = set(eligible_groups)
    by_task = defaultdict(dict)
    by_hit = defaultdict(list)
    for task in task_metrics:
        key = (task["group_id"], task["mode"])
        if task["task_id"] in by_task[key]:
            raise ValueError(f"Duplicate task observation: {key}, {task['task_id']}")
        by_task[key][task["task_id"]] = task
    for hit in hits:
        by_hit[hit["group_id"], hit["mode"]].append(hit)
    rows = []
    for sample in coverage:
        if sample["group_id"] not in eligible:
            continue
        key = (sample["group_id"], sample["mode"])
        sample_hits = by_hit[key]
        hit_ids = {h["task_id"] for h in sample_hits if h["task_id"]}
        run_tasks = by_task[key]
        missing_tasks = hit_ids - set(run_tasks)
        missing_task_ids = sum(not h["task_id"] for h in sample_hits)
        errors = {tid: t for tid, t in by_task[key].items()
                  if int(t["B"]) > 0 or str(t.get("task_status")) == "0"}
        bad_services = {s for t in errors.values() for s in json.loads(t["failed_services_json"])}
        unknown_segments = sum(h["segment"] == "unknown" for h in sample_hits)
        segments = {h["segment"] for h in sample_hits if h["segment"] != "unknown"}
        expected_tasks = sample.get("n_graph_tasks")
        graph_complete = expected_tasks is not None and len(run_tasks) == int(expected_tasks)
        network_complete = not unknown_segments
        run_id, target, fault, persistence, index, load = sample["group_id"].split("|")
        rows.append({"group_id": sample["group_id"], "mode": sample["mode"],
            "run_id": run_id, "fault_type": fault, "persistence": persistence,
            "sample_index": index, "load_users": load, "sample_path": sample["sample_path"],
            "n_hit_tasks": len(hit_ids), "n_matched_hit_tasks": len(hit_ids - missing_tasks),
            "n_tasks_in_collected_graph": len(run_tasks),
            "n_affected_services": len(bad_services) if graph_complete else "",
            "n_affected_tasks": len(errors) if graph_complete else "",
            "n_affected_segments": len(segments) if network_complete else "",
            "affected_services_json": json.dumps(sorted(bad_services)),
            "affected_task_ids_json": json.dumps(sorted(errors)),
            "affected_segments_json": json.dumps(sorted(segments)),
            "n_unmatched_hit_tasks": len(missing_tasks),
            "n_hits_without_task_id": missing_task_ids,
            "n_hits_with_unknown_segment": unknown_segments,
            "breadth_complete": graph_complete and network_complete,
            "task_basis": "all_collected_graph_tasks_process_error_or_terminal_failure",
            "network_basis": "all_true_hit_segments_independent_of_task_linkage"})
    # Use the same complete triplets and observations for all three metrics.
    modes_by_group = defaultdict(set)
    for row in rows:
        if row["breadth_complete"]:
            modes_by_group[row["group_id"]].add(row["mode"])
    complete_groups = {g for g, modes in modes_by_group.items() if modes == set(MODES)}
    for row in rows:
        row["included_in_boxplot"] = row["group_id"] in complete_groups
    return rows


def summarize_breadth(rows):
    summary = []
    for metric in METRICS:
        for mode in MODES:
            values = [float(r[metric]) for r in rows
                      if r["mode"] == mode and r["included_in_boxplot"]]
            if not values:
                raise ValueError(f"No complete per-run observations: {metric}, {mode}")
            stats = cbook.boxplot_stats(values, whis=1.5)[0]
            summary.append({"metric": metric, "mode": mode, "n_fault_runs": len(values),
                "min": min(values), "q1": stats["q1"], "median": stats["med"],
                "q3": stats["q3"], "max": max(values), "mean": float(np.mean(values)),
                "whisker_low": stats["whislo"], "whisker_high": stats["whishi"],
                "n_outliers": len(stats["fliers"]),
                "outlier_values_json": json.dumps(stats["fliers"].tolist())})
    return summary


def plot_breadth(rows):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45))
    fig.subplots_adjust(left=.065, right=.985, top=.91, bottom=.25, wspace=.30)
    titles = ("Observed abnormal services", "Observed abnormal tasks", "Fault-hit segments")
    rng = np.random.default_rng(20261109)
    for index, (ax, metric, title) in enumerate(zip(axes, METRICS, titles)):
        values = [np.array([r[metric] for r in rows
                  if r["mode"] == mode and r["included_in_boxplot"]], dtype=float)
                  for mode in MODES]
        ax.set_facecolor("#F6F7F8")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="white", linewidth=1.2)
        boxes = ax.boxplot(values, positions=(1, 2, 3), widths=.52, whis=1.5,
            patch_artist=True, showfliers=False,
            medianprops={"color": "#27323B", "linewidth": 1.25},
            whiskerprops={"color": "#5C646C", "linewidth": .9},
            capprops={"color": "#5C646C", "linewidth": .9},
            boxprops={"edgecolor": "#48535E", "linewidth": .9})
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
        for position, data in enumerate(values, 1):
            # Jitter changes only x-position; y is the exact integer count.
            ax.scatter(position + rng.uniform(-.10, .10, len(data)), data,
                       s=13, color="#26343F", alpha=.48, linewidths=0, zorder=3)
        ax.set_xticks((1, 2, 3), LABELS, rotation=20, ha="right")
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", direction="out", width=.7, length=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        maximum = max(max(v) for v in values)
        ax.set_ylim(0, max(1, maximum) * 1.16)
        ax.set_ylabel("Number of services" if index == 0 else
                      "Number of tasks" if index == 1 else "Number of region segments")
        ax.set_title(f"({chr(97+index)}) {title}", fontsize=12, pad=9)
        ax.text(.97, .97, f"n = {len(values[0])} per mode", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#59636D")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8C9298")
            ax.spines[side].set_linewidth(.7)
    fig.text(.525, .035, "Each point denotes one fault run; boxes show the median and IQR; whiskers: 1.5 IQR.",
             ha="center", fontsize=9, color="#555E66")
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig2_fault_breadth.{suffix}", dpi=350, facecolor="white")
    plt.close(fig)


def write_outputs(coverage, hits, task_metrics, eligible_groups):
    rows = compute_breadth(coverage, hits, task_metrics, eligible_groups)
    summary = summarize_breadth(rows)
    for name, data in [("fault_run_breadth.csv", rows), ("fault_run_breadth_summary.csv", summary)]:
        path = HERE / "tables" / name
        with path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    plot_breadth(rows)
    return rows, summary
