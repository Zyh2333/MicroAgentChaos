"""Two distinct target dimensions; all ranking formulas reuse MicroARCL helpers."""
import statistics

from eval_utils import _topk_metrics

from .rules import LAYERS


def ranking_metrics(rows, dimension):
    eligible = [r for r in rows if r.get("true_" + dimension)]
    key = "layer_rank" if dimension == "layers" else "fault_rank"
    acc, avg = _topk_metrics(eligible, key)
    return {"n": len(eligible), "ACC@K": acc, "AVG@N": avg,
            "MRR": sum(1 / r[key] for r in eligible if r.get(key, 0) > 0) / len(eligible) if eligible else None,
            "top1_accuracy": acc[1] if eligible else None}


def token_metrics(rows):
    """Aggregate per-sample ``token_usage``; missing usage is ignored."""
    usage = [r.get("token_usage") or {} for r in rows]

    def values(key):
        return [u[key] for u in usage if isinstance(u.get(key), (int, float))]

    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "cached_tokens", "uncached_prompt_tokens", "reasoning_tokens",
                "max_prompt_tokens", "rounds", "usage_rounds", "tool_calls"):
        found = values(key)
        result[key] = ({"sum": sum(found), "mean": statistics.mean(found),
                        "median": statistics.median(found)} if found else None)
    totals = values("total_tokens")
    usage_rounds = values("usage_rounds")
    calls = values("tool_calls")
    result["tokens_per_round"] = (round(sum(totals) / sum(usage_rounds))
                                  if totals and sum(usage_rounds) else None)
    result["tokens_per_tool_call"] = (round(sum(totals) / sum(calls))
                                      if totals and sum(calls) else None)
    result["samples"] = len(totals)
    return result


def evaluate(rows, breakdowns=True):
    labeled = [r for r in rows if r.get("true_layers")]
    total_true = sum(len(r["true_layers"]) for r in labeled)
    retained = sum(len(set(r["true_layers"]) & set(r["candidate_layers"])) for r in labeled)
    candidates = sum(len(r["candidate_layers"]) for r in labeled)
    excluded = sum(len(set(r["true_layers"]) & set(r["excluded_layers"])) for r in labeled)
    supported = sum(len(set(r["true_layers"]) & set(r.get("supported_layers", []))) for r in labeled)
    vetoed_supported = sum(len(set(r["true_layers"]) & set(r.get("vetoed_supported_layers", [])))
                           for r in labeled)
    typed = [r for r in rows if r.get("true_types")]
    result = {"samples": len(rows), "completed": sum(r["status"] == "success" for r in rows),
            "layer_ranking": ranking_metrics(rows, "layers"),
            "fault_type_ranking": ranking_metrics(rows, "types"),
            "layer_recall": retained / total_true if total_true else None,
            "layer_precision": retained / candidates if candidates else None,
            "false_elimination_rate": excluded / total_true if total_true else None,
            "support_recall": supported / total_true if total_true else None,
            "support_failure_rate": 1 - supported / total_true if total_true else None,
            "cf_false_elimination_rate": vetoed_supported / total_true if total_true else None,
            "candidate_reduction_rate": sum(1 - len(r["candidate_layers"]) / len(LAYERS) for r in labeled) / len(labeled) if labeled else None,
            "layer_exact_candidate_set_accuracy": sum(set(r["candidate_layers"]) == set(r["true_layers"]) for r in labeled) / len(labeled) if labeled else None,
            "fault_type_exact_set_accuracy": sum(set(r["fault_order"]) == set(r["true_types"]) for r in typed) / len(typed) if typed else None,
            "fault_type_recall": sum(len(set(r["fault_order"]) & set(r["true_types"])) for r in typed) / sum(len(r["true_types"]) for r in typed) if typed else None,
            "unexplained": sum(r.get("diagnostic_status") == "Unexplained" for r in rows),
            "out_of_taxonomy": sum(r.get("unexplained_reason") == "OutOfTaxonomy" for r in rows),
            "insufficient_support": sum(r.get("unexplained_reason") == "InsufficientSupport" for r in rows),
            "candidates_unranked": sum(r.get("unexplained_reason") == "CandidatesUnranked" for r in rows),
            "salvaged": sum(bool(r.get("salvaged")) for r in rows),
            "token_usage": token_metrics(rows),
            "unlabeled_layers": len(rows) - len(labeled), "unlabeled_types": len(rows) - len(typed)}
    if breakdowns:
        result["breakdowns"] = {field: {value: evaluate([r for r in rows if r.get(field, "") == value], False)
                                      for value in sorted({r.get(field, "") for r in rows})}
                                for field in ("topology_mode", "fault_mode", "load", "category", "fault")}
        from .io import TOPOLOGY_MODES, FAULT_MODES
        result["conditions"] = {topology: {
            "metrics": evaluate([r for r in rows if r.get("topology_mode") == topology], False),
            "fault_modes": {mode: evaluate([r for r in rows if r.get("topology_mode") == topology
                                           and r.get("fault_mode") == mode], False) for mode in FAULT_MODES}}
            for topology in TOPOLOGY_MODES}
    return result


def projection(rows, dimension):
    """Only label-bearing rows enter shared summaries, avoiding fake ground truth."""
    return [{**r, "gt": ",".join(r["true_" + dimension]),
             "rank": r["layer_rank" if dimension == "layers" else "fault_rank"],
             "top1": next(iter(r["layer_order" if dimension == "layers" else "fault_order"]), None),
             "root_cause": ""} for r in rows if r.get("true_" + dimension)]
