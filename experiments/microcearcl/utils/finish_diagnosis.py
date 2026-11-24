"""Internal workflow utility: finish_diagnosis."""
import json

from ..rules import BOUNDARY_TYPES, LAYERS, TYPES
from . import worklist
from .observation_evidence import raw_fact

NAME = "finish_diagnosis"


class ValidationErrors(ValueError):
    """All failures from one submission, so the model can correct them at once."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__(json.dumps(errors, ensure_ascii=False))


def _error(code, message, **details):
    return {'code': code, 'message': message, **details}


def _evidence_stage(registry, eid):
    return registry.evidence.get(eid, {}).get('stage')


def _layer_sets(agent):
    records = {layer: (agent.assessments.get(layer) or {}) for layer in LAYERS}
    states = {layer: records[layer].get('state', 'unassessed') for layer in LAYERS}
    candidates = worklist.candidates(agent.assessments)
    excluded = [layer for layer in LAYERS if states[layer] == 'excluded']
    unsupported = [layer for layer in LAYERS if states[layer] == 'unsupported']
    parked = [layer for layer in LAYERS if records[layer].get('parked')]
    skipped = [layer for layer in LAYERS if states[layer] == 'skipped']
    return candidates, excluded, unsupported, states, parked, skipped


def _unexplained_reason(states):
    if not states:
        return 'Unassessed'
    values = ['unsupported' if state in (None, 'unassessed', 'skipped') else state
              for state in states.values()]
    if all(state == 'excluded' for state in values):
        return 'OutOfTaxonomy'
    if all(state in ('unsupported', 'unresolved') for state in values):
        return 'InsufficientSupport'
    return 'Mixed'


def _ranking_basis(agent, args, candidates):
    basis = args.get('ranking_basis') or {}
    gaps, control_ids = [], []
    if not isinstance(basis, dict):
        gaps.append('ranking_basis 必须为对象')
        basis = {}
    control_ids = basis.get('control_evidence_ids', [])
    if not isinstance(control_ids, list) or any(not isinstance(eid, str) for eid in control_ids):
        gaps.append('ranking_basis.control_evidence_ids 必须为记录 ID 数组')
        control_ids = []
    unknown = [eid for eid in control_ids if not raw_fact(agent.registry, eid)]
    if unknown:
        gaps.append('ranking_basis 引用了未读取的对照记录')
    if len(candidates) > 1 and not control_ids:
        gaps.append('多个候选但未引用 C 对照；排序可信度未由对照加强')
    return {'control_evidence_ids': [eid for eid in control_ids if eid not in unknown],
            'explanation': basis.get('explanation') if isinstance(basis.get('explanation'), str) else '',
            'gaps': gaps}


def run(agent, args):
    errors, validated = [], []
    candidates, excluded, unsupported, states, parked, skipped = _layer_sets(agent)
    order = args.get('layer_order')
    if not isinstance(order, list) or any(not isinstance(x, str) for x in order):
        errors.append(_error('layer_order_type', 'layer_order 必须为层名数组',
                             field='layer_order', candidates=candidates))
        order = []
    elif len(set(order)) != len(order):
        errors.append(_error('layer_order_duplicate', 'layer_order 不能包含重复层',
                             field='layer_order', layer_order=order))
    not_candidate = [x for x in order if x not in candidates]
    if not_candidate:
        errors.append(_error(
            'layer_order_not_candidate', 'layer_order 只能包含有支持的 candidate 层',
            field='layer_order', layers=not_candidate, candidates=candidates,
            excluded_layers=excluded, unsupported_layers=unsupported, parked_layers=parked))

    diagnoses = args.get('diagnoses')
    if not isinstance(diagnoses, list):
        errors.append(_error('diagnoses_type', 'diagnoses 必须为数组', field='diagnoses'))
        diagnoses = []
    seen = set()
    for index, diagnosis in enumerate(diagnoses):
        prefix = {'diagnosis_index': index}
        if not isinstance(diagnosis, dict):
            errors.append(_error('diagnosis_type', '每条 diagnosis 必须为对象', **prefix))
            continue
        layer, kind = diagnosis.get('layer'), diagnosis.get('fault_type')
        key = f'{layer}/{kind}'
        prefix.update({'diagnosis': key, 'layer': layer, 'fault_type': kind})
        if not isinstance(layer, str) or layer not in candidates:
            errors.append(_error('diagnosis_layer', '诊断层必须是 candidate 且出现在 layer_order 中',
                                 candidates=candidates, layer_order=order, **prefix))
        elif layer not in order:
            errors.append(_error('diagnosis_layer_order', '诊断层必须出现在 layer_order 中',
                                 candidates=candidates, layer_order=order, **prefix))
        if not isinstance(kind, str) or kind not in TYPES.get(layer, ()):
            errors.append(_error('diagnosis_type_unknown', '类型必须来自该候选层的已知类型',
                                 allowed_types=list(TYPES.get(layer, ())), **prefix))
        if key in seen:
            errors.append(_error('diagnosis_duplicate', '同一 layer/fault_type 不能重复', **prefix))
        seen.add(key)
        ids = diagnosis.get('evidence_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(eid, str) for eid in ids):
            errors.append(_error('evidence_ids', '诊断需引用非空 evidence_ids 数组',
                                 actual_stage=agent.stage, **prefix))
            ids = []
        absent = [eid for eid in ids
                  if eid not in agent.registry.evidence and eid not in agent.registry.mechanisms]
        if absent:
            errors.append(_error('evidence_unknown', '诊断引用了不存在的证据 ID',
                                 evidence_ids=absent, actual_stage=agent.stage, **prefix))
        raw_ids = [eid for eid in ids if raw_fact(agent.registry, eid, stage=2)]
        staged_out = [eid for eid in ids if eid not in absent and not raw_fact(agent.registry, eid, stage=2)]
        if not raw_ids:
            errors.append(_error(
                'evidence_stage', '细分类须至少引用一条第二阶段新记录（类型判别签名）',
                evidence_ids=ids, actual_stage=agent.stage,
                evidence_stages={eid: _evidence_stage(agent.registry, eid) for eid in ids}, **prefix))
        if staged_out:
            errors.append(_error(
                'evidence_stage_mixed',
                '以下引用不是第二阶段原始记录；Stage 1 记录请移到 corroboration_evidence_ids',
                evidence_ids=staged_out,
                evidence_stages={eid: _evidence_stage(agent.registry, eid) for eid in staged_out}, **prefix))
        corroboration = diagnosis.get('corroboration_evidence_ids', [])
        if not isinstance(corroboration, list) or any(not isinstance(eid, str) for eid in corroboration):
            errors.append(_error('corroboration_ids',
                                 'corroboration_evidence_ids 必须为记录 ID 数组',
                                 field='corroboration_evidence_ids', **prefix))
            corroboration = []
        corr_absent = [eid for eid in corroboration
                       if eid not in agent.registry.evidence and eid not in agent.registry.mechanisms]
        if corr_absent:
            errors.append(_error('corroboration_unknown',
                                 'corroboration 引用了不存在的证据 ID',
                                 evidence_ids=corr_absent, **prefix))
        corr_staged = [eid for eid in corroboration if eid not in corr_absent
                       and not raw_fact(agent.registry, eid, stage=1)]
        if corr_staged:
            errors.append(_error('corroboration_stage',
                                 'corroboration_evidence_ids 只接受 Stage 1 原始记录',
                                 evidence_ids=corr_staged, actual_stage=agent.stage,
                                 evidence_stages={eid: _evidence_stage(agent.registry, eid)
                                                  for eid in corr_staged}, **prefix))
        if raw_ids and (not isinstance(diagnosis.get('explanation'), str)
                        or not diagnosis['explanation'].strip()):
            errors.append(_error('explanation', '须提供基于所引用观测的简短解释',
                                 evidence_ids=raw_ids, **prefix))
        if kind in BOUNDARY_TYPES:
            boundary = diagnosis.get('boundary_evidence_ids')
            if not isinstance(boundary, list) or not boundary:
                errors.append(_error(
                    'boundary_missing', '该类型区分请求丢弃与响应丢弃，须引用 compare_presence 的跨端对照',
                    field='boundary_evidence_ids', actual_stage=agent.stage, **prefix))
                boundary = []
            for eid in boundary:
                fact = raw_fact(agent.registry, eid, stage=2)
                tool = agent.registry.evidence.get(eid, {}).get('tool')
                if not fact or tool != 'compare_presence':
                    errors.append(_error(
                        'boundary_evidence', 'boundary_evidence_ids 须为第二阶段的 compare_presence 记录',
                        evidence_id=eid, tool=tool, actual_stage=agent.stage, **prefix))
        if all(e.get('diagnosis') != key for e in errors):
            validated.append({**diagnosis, 'corroboration_evidence_ids': corroboration,
                              'basis': 'llm_inference', 'validation': 'provenance_only'})

    if errors:
        raise ValidationErrors(errors)
    basis = _ranking_basis(agent, args, candidates)
    if diagnoses:
        status = 'Explained'
    elif order:
        status = 'Partial'
    else:
        status = 'Unexplained'
    reason = None
    if not candidates:
        reason = _unexplained_reason(states)
    elif not order:
        reason = 'CandidatesUnranked'
    agent.final = {"layer_order": order, "diagnoses": validated,
                   "fault_order": [f"{d['layer'].lower()}_{d['fault_type']}" for d in diagnoses],
                   "ranking_basis": basis,
                   "explanation": args.get("explanation"),
                   "diagnostic_status": status, "unexplained_reason": reason,
                   "candidate_layers": candidates, "excluded_layers": excluded,
                   "unsupported_layers": unsupported,
                   "parked_layers": parked, "skipped_layers": skipped,
                   "ranking_gaps": basis['gaps']}
    return agent.final


def salvage(agent, payload):
    """Keep the last submitted ordering when the round budget runs out.

    Only structurally valid pieces are kept: the last known-layer ordering and
    diagnoses that cite Stage-2 records and have an explanation.  The termination
    stays round_budget_exhausted; `salvaged` marks the relaxed validation.
    """
    order = []
    for layer in payload.get('layer_order') or []:
        if isinstance(layer, str) and layer in LAYERS and layer not in order:
            order.append(layer)
    validated, seen = [], set()
    for diagnosis in payload.get('diagnoses') or []:
        if not isinstance(diagnosis, dict):
            continue
        layer, kind = diagnosis.get('layer'), diagnosis.get('fault_type')
        key = f'{layer}/{kind}'
        if layer not in order or kind not in TYPES.get(layer, ()) or key in seen:
            continue
        ids = diagnosis.get('evidence_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(eid, str) for eid in ids):
            continue
        if any(eid not in agent.registry.evidence and eid not in agent.registry.mechanisms for eid in ids):
            continue
        if not any(raw_fact(agent.registry, eid, stage=2) for eid in ids):
            continue
        if not isinstance(diagnosis.get('explanation'), str) or not diagnosis['explanation'].strip():
            continue
        seen.add(key)
        validated.append({**diagnosis, 'basis': 'llm_inference', 'validation': 'provenance_only'})
    return {"layer_order": order, "diagnoses": validated,
            "fault_order": [f"{d['layer'].lower()}_{d['fault_type']}" for d in validated],
            "explanation": payload.get('explanation'),
            "diagnostic_status": "Partial" if order or validated else "Unexplained",
            "salvaged": True, "salvage_reason": "round_budget_exhausted"}
