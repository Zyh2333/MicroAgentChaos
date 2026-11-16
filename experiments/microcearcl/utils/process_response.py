"""Apply incremental layer judgments through internal validators.

Each submission is a delta: only the layers/rules the LLM advances are merged.
Validation is all-or-nothing; provenance is still checked per citation, and
tool-level ``observation-N`` ids expand to reviewed record ids first.
"""
from copy import deepcopy

from . import verify_counterfactual, assess_layer, finish_layer_stage, finish_diagnosis, worklist
from .evidence_ids import expand, reviewed_for
from ..rules import LAYERS, predictions

ENTRY_FIELDS = (("support", "support"), ("contradictions", "contradiction"), ("controls", "control"))


def _error(code, message, **details):
    return {'code': code, 'message': message, **details}


def _normalize(agent, assessment):
    """Expand observation ids and inherit expectation/explanation on a delta."""
    layer = assessment.get('layer')
    prior = agent.assessments.get(layer) or {}
    assessment = deepcopy(assessment)
    assessment['expectation'] = (assessment.get('expectation') or prior.get('expectation')
                                 or '按当前层规则复核')
    assessment['explanation'] = (assessment.get('explanation') or prior.get('explanation')
                                 or '增量提交；依据见所引用记录')
    reviews, reviewed, _ = reviewed_for(agent, layer) if layer in LAYERS else (None, set(), set())
    if layer in LAYERS and not reviews:
        raise ValueError('先调用 submit_layer 提交该层（内部会完成来源核验）')
    expanded = {}
    for field in ('support', 'contradictions', 'controls'):
        entries = assessment.get(field)
        if not isinstance(entries, list):
            continue
        result = []
        for entry in entries:
            if not isinstance(entry, dict):
                result.append(entry)
                continue
            ids, mapping = expand([entry.get('evidence_id')], reviewed, f'{field}.evidence_id')
            expanded.update(mapping)
            for eid in ids:
                result.append({**entry, 'evidence_id': eid})
        assessment[field] = result
    cf = assessment.get('cf')
    if isinstance(cf, list):
        for entry in cf:
            if not isinstance(entry, dict):
                continue
            for field in ('evidence_ids', 'control_ids', 'coverage_evidence_ids'):
                ids, mapping = expand(entry.get(field), reviewed, f'cf.{field}')
                expanded.update(mapping)
                entry[field] = ids
    if 'support_coverage_ids' in assessment:
        ids, mapping = expand(assessment.get('support_coverage_ids') or [], reviewed,
                              'support_coverage_ids')
        expanded.update(mapping)
        assessment['support_coverage_ids'] = ids
    assessment['_auto_expanded'] = expanded
    return assessment


def _verify_entries(agent, assessment, field, role):
    entries = assessment.get(field, [])
    if not isinstance(entries, list):
        raise ValueError(f"{field} 必须为数组")
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"{field}[{index}] 须为 {{evidence_id,rule_id,note}} 对象")
        result = verify_counterfactual.run(agent, {
            "layer": assessment.get("layer"), "role": role,
            "evidence_id": entry.get("evidence_id"), "rule_id": entry.get("rule_id"),
            "expectation": assessment.get("expectation"), "explanation": assessment.get("explanation")})
        if not result["verified"]:
            raise ValueError(f"{field}[{index}] 引用无效：{entry.get('evidence_id')} / {entry.get('rule_id')}")


def _verify_cf(agent, assessment):
    entries = assessment.get("cf", [])
    if not isinstance(entries, list):
        raise ValueError("cf 必须为数组")
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"cf[{index}] 须为对象")
        for field in ("evidence_ids", "control_ids", "coverage_evidence_ids"):
            ids = entry.get(field, [])
            if not isinstance(ids, list) or any(not isinstance(eid, str) for eid in ids):
                raise ValueError(f"cf[{index}].{field} 必须为记录 ID 数组")
            for eid in ids:
                result = verify_counterfactual.run(agent, {
                    "layer": assessment.get("layer"), "role": "counterfactual",
                    "evidence_id": eid, "rule_id": entry.get("rule_id"),
                    "expectation": assessment.get("expectation"),
                    "explanation": entry.get("explanation") or assessment.get("explanation")})
                if not result["verified"]:
                    raise ValueError(f"cf[{index}] 引用无效：{eid} / {entry.get('rule_id')}")


def _verify_support_coverage(agent, assessment):
    layer = assessment.get("layer")
    rule = predictions(layer)[0] if layer in LAYERS else None
    for eid in assessment.get("support_coverage_ids") or []:
        result = verify_counterfactual.run(agent, {
            "layer": layer, "role": "coverage", "evidence_id": eid, "rule_id": rule,
            "expectation": assessment.get("expectation"),
            "explanation": assessment.get("explanation")})
        if not result["verified"]:
            raise ValueError(f"support_coverage_ids 引用无效：{eid}")


def _compact(update):
    keys = ('layer', 'state', 'support_state', 'support_confirmed', 'parked',
            'cached_rules', 'optional_rules', 'cf_states', 'veto_by', 'gaps', 'ignored')
    return {key: update[key] for key in keys if key in update}


def _ledger(agent):
    """Five-line state summary so the model never needs to re-submit a layer."""
    return {layer: {'state': record.get('state'),
                    'support': record.get('support_state'),
                    'cf': record.get('cf_states', {}),
                    'cached': record.get('supported_rules', [])}
            for layer, record in agent.assessments.items()}


def _cite_index(agent, observations=4, records=3):
    """Recent Stage-1 record ids so citations do not need another lookup round."""
    groups = {}
    for eid, fact in agent.registry.evidence.items():
        if fact.get('kind') == 'raw_observation' and fact.get('stage') == 1:
            groups.setdefault(eid.split(':record:')[0], []).append(eid)
    order = sorted(groups, key=lambda obs: int(obs.split('-')[1]))[-observations:]
    return {obs: groups[obs][:records] for obs in order}


def run(agent, payload):
    if not isinstance(payload, dict):
        raise ValueError("判断必须为 JSON object")
    if agent.stage == 2:
        return finish_diagnosis.run(agent, payload)
    assessments = payload.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        raise ValueError("第一阶段需返回非空 assessments 数组")
    if "stage1_complete" in payload and not isinstance(payload["stage1_complete"], bool):
        raise ValueError("stage1_complete 必须为布尔值")
    complete = payload.get("stage1_complete") is True
    snapshot = deepcopy(agent.assessments), set(agent.verified), agent.stage
    before = deepcopy(agent.assessments)
    updates, errors, seen = [], [], set()
    try:
        for index, assessment in enumerate(assessments):
            if not isinstance(assessment, dict):
                errors.append(_error('assessment_type', '每层判断必须为对象', index=index))
                continue
            layer = assessment.get("layer")
            if layer in seen:
                errors.append(_error('duplicate_layer', '同一轮不能重复提交同一层', layer=layer))
                continue
            seen.add(layer)
            prior = agent.assessments.get(layer) or {}
            if prior.get("state") in ("excluded", "skipped"):
                # A pruned layer is monotone: ignore its resubmission without
                # spending a round on citations for a decision already made.
                updates.append(assess_layer.run(agent, {"layer": layer}))
                continue
            try:
                assessment = _normalize(agent, assessment)
                for field, role in ENTRY_FIELDS:
                    _verify_entries(agent, assessment, field, role)
                _verify_cf(agent, assessment)
                _verify_support_coverage(agent, assessment)
                updates.append(assess_layer.run(agent, assessment))
            except (ValueError, TypeError, KeyError) as exc:
                code = 'layer_delta_invalid'
                if '未在该层 check 中复核' in str(exc):
                    code = 'rule_not_reviewed'
                elif ('只能引用已复核' in str(exc) or '引用无效' in str(exc)
                      or '无法解析' in str(exc)):
                    code = 'evidence_invalid'
                errors.append(_error(code, str(exc), layer=layer))
        if errors:
            raise finish_diagnosis.ValidationErrors(errors)
        result = {"assessments": [_compact(update) for update in updates],
                  "stage": agent.stage, "candidates": agent._candidates(),
                  "strict_candidates": worklist.candidates(agent.assessments),
                  "worklist": worklist.summary(agent),
                  "ledger": _ledger(agent),
                  "cite_index": _cite_index(agent),
                  "pruning": worklist.effects(before, agent.assessments)}
        if worklist.ready(agent, complete=complete):
            result["transition"] = finish_layer_stage.run(agent, {"complete": complete})
            result["stage"] = agent.stage
            result["candidates"] = agent._candidates()
            result["worklist"] = worklist.summary(agent)
            result["next"] = "已进入 Stage 2：只在 strict_candidates 内排序和细分类"
        else:
            remaining = result["worklist"]["required"]
            if not remaining and not result["worklist"]["readers_done"]:
                result["next"] = ("必需项已清空；还需调用 read_metrics、read_logs、"
                                  "read_execution_graph 才能进入 Stage 2")
            else:
                result["next"] = (f"仍有 {len(remaining)} 个必需项；只提交本轮推进的层，"
                                  "已关闭项不需要重复提交")
        return result
    except (ValueError, TypeError, KeyError):
        agent.assessments, agent.verified, agent.stage = snapshot
        raise
