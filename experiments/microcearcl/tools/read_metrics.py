"""CSV metrics inventory and summaries; -1/NaN are missing, not healthy."""
import csv
import math
from statistics import median

NAME = "read_metrics"
PARAMETERS = {"service": {"type": "string"}}
DESCRIPTION = "汇总原生 CSV 指标，保留单位未知及缺测状态，不将绝对耗时视为注入延迟。"


def run(sample):
    if NAME in sample.cache:
        return sample.cache[NAME]
    files = {}
    for file in sorted(sample.ns("metrics").glob("*.csv")):
        if file.name == "graph.csv":
            continue
        values, missing = {}, 0
        with file.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                for key, raw in row.items():
                    if key is None or key == "timestamp":
                        continue
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value) or value < 0:
                        missing += 1
                    else:
                        values.setdefault(key, []).append(value)
        files[str(file.relative_to(sample.path))] = {
            "missing_cells": missing,
            "columns": {key: {"n": len(v), "median": median(v), "min": min(v), "max": max(v)}
                        for key, v in values.items()}}
    result = {"files": files, "diagnostic_role": "context_only",
              "note": "CSV 单位未自动推断；同窗口/负载/任务对照需调用方验证。"}
    if sample.mask_csv_gen_auth:
        from ..utils.background_noise import csv_service
        for data in files.values():
            for column, summary in data['columns'].items():
                if csv_service(column.split('&')[0]):
                    summary['background_contaminated'] = True
                    summary['note'] = '已知 csv-gen 数据污染；保留原值，不能单独用于归因。'
    # The three primary KPI frames are loaded by the shared MicroARCL reader.
    if all(sample.ns("metrics").joinpath(n).is_file() for n in
           ("success_rate.csv", "latency.csv", "svc_qps.csv")):
        import agent_rca as R
        frames = sample.cache.get("kpi")
        if frames is None:
            frames = R.load_kpi(str(sample.ns("metrics")))
            sample.cache["kpi"] = frames
        result["detected_changes"] = []
        for name, frame in zip(("success_rate", "latency", "qps"), frames):
            for col in frame.columns:
                if col == "timestamp":
                    continue
                mask = R._series_spike(frame[col] if name != "success_rate" else 1 - frame[col].where(frame[col] >= 0))
                if mask.any():
                    result["detected_changes"].append({
                        "service": col.split("&")[0], "metric": name, "count": int(mask.sum()),
                        "first": str(frame.loc[mask, "timestamp"].iloc[0]),
                        "last": str(frame.loc[mask, "timestamp"].iloc[-1]),
                        "note": "单序列稳健偏移检测，只支持异常观测，不确定层级"})
    sample.cache[NAME] = result
    return result
