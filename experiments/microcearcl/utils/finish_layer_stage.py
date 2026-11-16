"""Internal workflow utility: finish_layer_stage.

Stage 1 closes when every blocking work item is decided.  Optional R
strengthening and C controls never block.  ``complete`` lets the LLM end the
stage with layers left unsubmitted (single-candidate early exit); those layers
are frozen as ``skipped`` and cannot inflate the candidate set.
"""
import json

from ..rules import TYPES
from . import worklist

NAME = "finish_layer_stage"


def run(agent, args):
    complete = bool(args.get("complete"))
    if not worklist.readers_done(agent):
        raise ValueError("Agent 必须先调用指标、日志、执行图观测工具（缺失也应记录）")
    remaining = worklist.required(agent.assessments, skip_unsubmitted=complete)
    if remaining:
        raise ValueError("仍有必需项未完成：" + json.dumps(remaining, ensure_ascii=False))
    early_exit = worklist.early_exit(agent, complete)
    skipped = worklist.freeze_skipped(agent) if complete else []
    agent.stage = 2
    candidates = worklist.candidates(agent.assessments)
    support = {layer: [{'rule_id': row['rule_id'], 'evidence_id': row['evidence_id']}
                       for row in agent.assessments[layer].get('support', [])][:6]
               for layer in candidates}
    return {"stage": 2, "candidates": candidates, "strict_candidates": candidates,
            "fault_types": {k: TYPES[k] for k in candidates},
            "controls": {k: agent.assessments[k].get('controls', []) for k in candidates},
            "cf_states": {k: agent.assessments[k].get('cf_states', {}) for k in candidates},
            "support_evidence": support,
            "optional_work": worklist.optional(agent.assessments),
            "excluded_layers": [k for k, v in agent.assessments.items()
                                if v.get("state") == "excluded"],
            "unsupported_layers": [k for k, v in agent.assessments.items()
                                   if v.get("state") == "unsupported"],
            "parked_layers": worklist.parked(agent.assessments),
            "skipped_layers": skipped,
            "early_exit": early_exit,
            "next": "只在 candidate_layers 内排序和细分类；支持证据仅用于排序，"
                    "细分类必须引用第二阶段新读的记录"}
