"""Check evidence availability for a layer hypothesis without classifying log text."""
from ..rules import LAYERS, counterfactuals, predictions, rule_lookup
from .evidence_ids import expand
from .observation_evidence import raw_fact

PARAMETERS = {
    'hypothesis': {'type': 'string'}, 'expectation': {'type': 'string'},
    'rule_ids': {'type': 'array', 'items': {'type': 'string'},
                 'description': '本次复核的该层 R1–R4 支持规则 ID'},
    'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
    'control_ids': {'type': 'array', 'items': {'type': 'string'}},
    'cf_rule_ids': {'type': 'array', 'items': {'type': 'string'},
                    'description': '本次复核的该层 CF 必要性签名规则 ID'},
    'cf_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
    'cf_control_ids': {'type': 'array', 'items': {'type': 'string'}},
    'cf_coverage_ids': {'type': 'array', 'items': {'type': 'string'}},
    'support_coverage_ids': {'type': 'array', 'items': {'type': 'string'},
                             'description': '声明该层 support absent 时的覆盖范围记录'},
}


def _recent_records(registry, stage, limit=8):
    """Recently read record ids, so a bad citation gets an actionable hint."""
    ids = [eid for eid, fact in registry.evidence.items()
           if fact.get('kind') == 'raw_observation' and fact.get('stage') == stage]
    return '、'.join(ids[-limit:]) or '（当前阶段没有可引用记录）'


def _rows(registry, ids, stage, field='evidence_ids'):
    if not isinstance(ids, list) or any(not isinstance(eid, str) for eid in ids):
        raise ValueError('证据字段必须为记录 ID 数组')
    ids, _ = expand(ids, set(), field, registry=registry, stage=stage)
    rows = []
    for eid in ids:
        fact = raw_fact(registry, eid, stage=stage)
        if not fact:
            # Name the offending id: a bare "not readable" message left the
            # model guessing which citation to replace.
            raise ValueError(f'check 无法引用 {eid}：它不是当前阶段已读取的记录；'
                             f'可引用示例：{_recent_records(registry, stage)}')
        rows.append({'evidence_id': eid, 'text': fact['text'], 'references': fact['references']})
    return rows


def check(registry, layer, args, stage):
    if layer not in LAYERS:
        raise ValueError('未知故障域')
    rules = args.get('rule_ids', [])
    cf_rules = args.get('cf_rule_ids', [])
    for name, values, allowed in (('rule_ids', rules, predictions(layer)),
                                  ('cf_rule_ids', cf_rules, counterfactuals(layer))):
        if not isinstance(values, list) or any(not isinstance(rid, str) for rid in values):
            raise ValueError(f'{name} 必须为规则 ID 数组')
        for rid in values:
            if rid not in allowed:
                raise ValueError(f'{rid} 不属于 {layer} 的 {name} 范围')
    rows = _rows(registry, args.get('evidence_ids', []), stage)
    comparison = _rows(registry, args.get('control_ids', []), stage)
    cf_evidence = _rows(registry, args.get('cf_evidence_ids', []), stage)
    cf_controls = _rows(registry, args.get('cf_control_ids', []), stage)
    cf_coverage = _rows(registry, args.get('cf_coverage_ids', []), stage)
    support_coverage = _rows(registry, args.get('support_coverage_ids', []), stage,
                             'support_coverage_ids')
    gaps = []
    for field in ('hypothesis', 'expectation'):
        if not args.get(field, '').strip():
            gaps.append('缺少 ' + field)
    if not rules:
        gaps.append('未标注 R 支持规则（rule_ids）')
    if not cf_rules:
        gaps.append('未复核 CF 必要性签名（cf_rule_ids）')
    if not rows:
        gaps.append('缺少实际观测引用')
    if not comparison:
        gaps.append('未提供对照引用；是否足以支持假设需 LLM 说明')
    if cf_evidence and not cf_controls:
        gaps.append('CF 证据缺少匹配对照')
    if cf_evidence and not cf_coverage:
        gaps.append('CF 证据缺少覆盖引用；absent 只能记为 unknown')
    lookup = rule_lookup()
    result = {'layer': layer, 'stage': stage, 'hypothesis': args.get('hypothesis'),
        'expectation': args.get('expectation'),
        'rules': [{'id': rid, 'description': lookup[rid]['description']} for rid in rules],
        'cf_rules': [{'id': rid, 'description': lookup[rid]['description']} for rid in cf_rules],
        'evidence': rows, 'controls': comparison,
        'cf_evidence': cf_evidence, 'cf_controls': cf_controls, 'cf_coverage': cf_coverage,
        'support_coverage': support_coverage,
        'gaps': gaps, 'validation': 'provenance_and_completeness',
        'semantic_verdict': '由 LLM 根据原文与对照判断，不由日志模板判定'}
    registry.verifications.append(result)
    return result
