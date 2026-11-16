"""Merge LLM layer judgments incrementally; the engine caches committed rules.

A submission is a delta.  Omitted fields keep their previous value, so a
CF-only round never wipes support and a support-only round never resurrects a
vetoed layer.  Support is trivalent, mirroring CF: ``present`` (R evidence),
``absent`` (confirmed no support, coverage + control required) and ``unknown``
(probed but unconfirmed).  Confirmed absence and a CF veto close the layer;
``unknown`` keeps it visible as parked without satisfying Gate 1.
"""
from copy import deepcopy

from ..rules import LAYERS, controls, counterfactuals, predictions
from ..registry import LAYER_TO_TOOL
from .evidence_ids import reviewed_for
from .observation_evidence import raw_fact

NAME = 'assess_layer'
CF_STATES = ('present', 'absent', 'unknown')
SUPPORT_STATES = ('present', 'absent', 'unknown')
LAYER_STATES = ('unsubmitted', 'candidate', 'unsupported', 'excluded', 'skipped')


def blank(layer, state='unsubmitted'):
    return {'layer': layer, 'state': state, 'support': [], 'contradictions': [],
            'controls': [], 'cf': [], 'cf_states': {}, 'veto_by': [],
            'support_rules': {}, 'supported_rules': [], 'cached_rules': [],
            'optional_rules': [], 'support_state': 'unsubmitted',
            'support_confirmed': False, 'parked': False, 'support_coverage_ids': [],
            'cf_gaps': [], 'gaps': [], 'auto_expanded': {},
            'expectation': '', 'explanation': '',
            'assessment_basis': 'llm_inference', 'validation': 'provenance_only'}


def _entries(value, field):
    if not isinstance(value, list):
        raise ValueError(f'{field} 必须为数组')
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f'{field} 的每项须为 {{"evidence_id","rule_id","note"}} 对象')
        evidence_id, rule = item.get('evidence_id'), item.get('rule_id')
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError(f'{field} 缺少 evidence_id')
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError(f'{field} 的 {evidence_id} 缺少 rule_id；必须标注所引用的规则')
        result.append({'evidence_id': evidence_id, 'rule_id': rule,
                       'note': item.get('note') if isinstance(item.get('note'), str) else ''})
    return result


def _ids(value, field):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f'{field} 必须为记录 ID 数组')
    return value


def _unique(entries, *keys):
    out, seen = [], set()
    for entry in entries:
        key = tuple(entry.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _cf_entries(agent, layer, value, reviewed, verified):
    if not isinstance(value, list):
        raise ValueError('cf 必须为数组')
    allowed = set(counterfactuals(layer))
    result, seen = [], set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError('cf 的每项须为对象')
        rule, state = item.get('rule_id'), item.get('state')
        if rule not in allowed:
            raise ValueError(f'{rule} 不属于 {layer} 的 CF 规则')
        if rule in seen:
            raise ValueError(f'CF 规则 {rule} 不能重复')
        seen.add(rule)
        if state not in CF_STATES:
            raise ValueError('cf.state 必须是 present、absent 或 unknown')
        evidence = _ids(item.get('evidence_ids'), 'cf.evidence_ids')
        controls_ = _ids(item.get('control_ids'), 'cf.control_ids')
        coverage = _ids(item.get('coverage_evidence_ids'), 'cf.coverage_evidence_ids')
        gaps = []
        if state in ('present', 'absent') and not evidence:
            gaps.append('缺少签名观测引用')
        if state == 'absent' and not controls_:
            gaps.append('缺少匹配对照')
        if state == 'absent' and not coverage:
            gaps.append('缺少覆盖引用')
        for eid in evidence + controls_ + coverage:
            if eid not in reviewed or not raw_fact(agent.registry, eid, stage=1):
                valid = '、'.join(sorted(reviewed)[:8]) or '（无）'
                raise ValueError(f'cf 只能引用已复核的 Stage 1 记录：{eid}；可引用示例：{valid}')
            if (eid, 'counterfactual') not in verified:
                raise ValueError(f'{eid} 未完成 CF 来源核验')
        if state == 'absent' and gaps:
            # Missing coverage/control cannot veto: downgrade to unknown.
            state = 'unknown'
        if state == 'absent' and not (isinstance(item.get('explanation'), str)
                                      and item['explanation'].strip()):
            raise ValueError('cf absent 必须说明签名为何没有出现以及覆盖范围')
        result.append({'rule_id': rule, 'state': state, 'evidence_ids': evidence,
                       'control_ids': controls_, 'coverage_evidence_ids': coverage,
                       'gaps': gaps,
                       'explanation': item.get('explanation') if isinstance(item.get('explanation'), str) else ''})
    return result


def _merge_cf(prior, incoming):
    merged = {entry['rule_id']: deepcopy(entry) for entry in prior}
    ignored = []
    for entry in incoming:
        rule = entry['rule_id']
        old = merged.get(rule)
        if old is None:
            merged[rule] = entry
        elif old['state'] == 'present':
            if entry['state'] == 'present':
                old['evidence_ids'] = sorted(set(old['evidence_ids']) | set(entry['evidence_ids']))
                old['control_ids'] = sorted(set(old['control_ids']) | set(entry['control_ids']))
                old['coverage_evidence_ids'] = sorted(
                    set(old['coverage_evidence_ids']) | set(entry['coverage_evidence_ids']))
            else:
                ignored.append(rule)
        elif old['state'] == 'absent' and entry['state'] != 'absent':
            ignored.append(rule)  # a veto is monotone inside Stage 1
        else:
            merged[rule] = entry
    return [merged[rule] for rule in sorted(merged)], ignored


def _merge_support(prior_state, entries, explicit, prior_entries, provided):
    """Return (state, merged_entries, ignored) under the monotone lattice."""
    merged = _unique(list(prior_entries) + list(entries), 'evidence_id', 'rule_id')
    if entries:
        target = 'present'
    elif explicit is not None:
        if explicit not in SUPPORT_STATES:
            raise ValueError('support_state 必须是 present、absent 或 unknown')
        target = explicit
    elif provided:
        target = 'unknown'
    else:
        target = prior_state
    ignored = False
    if prior_state == 'present' and target != 'present':
        target, ignored = 'present', True
    elif prior_state == 'absent' and target == 'unknown':
        target, ignored = 'absent', True
    if target == 'present' and not merged:
        raise ValueError('support_state=present 必须附至少一条 R 支持证据')
    if target == 'absent' and entries:
        raise ValueError('support 非空时不能声明 absent')
    return target, merged, ignored


def run(agent, args):
    layer = args['layer']
    if layer not in LAYERS:
        raise ValueError('未知故障域')
    prior = deepcopy(agent.assessments.get(layer) or blank(layer))
    if prior.get('state') in ('excluded', 'skipped'):
        return {'layer': layer, **prior, 'ignored': 'layer_closed',
                'candidates': agent._candidates()}
    reviews, reviewed, reviewed_rules = reviewed_for(agent, layer)
    if not reviews:
        raise ValueError('先调用 submit_layer 提交该层（或先用 ' + LAYER_TO_TOOL[layer] + ' 复核）')

    provided = 'support' in args
    support = _entries(args.get('support', []), 'support') if provided else []
    contradictions = _entries(args.get('contradictions', []), 'contradictions')
    used_controls = _entries(args.get('controls', []), 'controls')
    for entries, value, allowed in ((support, 1, set(predictions(layer))),
                                     (contradictions, 0, set(predictions(layer))),
                                     (used_controls, None, set(controls(layer)))):
        for entry in entries:
            eid, rule = entry['evidence_id'], entry['rule_id']
            if rule not in allowed:
                raise ValueError(f'{rule} 不属于 {layer} 的 R/C 规则范围')
            if value is not None and rule not in reviewed_rules:
                raise ValueError(f'{rule} 未在该层 check 中复核；先复核对应规则')
            if eid not in reviewed or not raw_fact(agent.registry, eid, stage=1):
                valid = '、'.join(sorted(reviewed)[:8]) or '（无）'
                raise ValueError(f'只能引用已复核的 Stage 1 记录：{eid}；可引用示例：{valid}')
            if value is not None and (eid, value) not in agent.verified:
                raise ValueError(f'{eid} 未按所引规则完成来源核验')

    support_state, merged_support, ignored = _merge_support(
        prior.get('support_state', 'unsubmitted'), support,
        args.get('support_state'), prior.get('support'), provided)
    coverage = _ids(args.get('support_coverage_ids', prior.get('support_coverage_ids', [])),
                    'support_coverage_ids')
    for eid in coverage:
        if eid not in reviewed or not raw_fact(agent.registry, eid, stage=1):
            raise ValueError(f'support_coverage_ids 只能引用已复核的 Stage 1 记录：{eid}')
        if (eid, 'coverage') not in agent.verified:
            raise ValueError(f'{eid} 未完成覆盖来源核验')

    merged_controls = _unique(prior.get('controls', []) + used_controls, 'evidence_id', 'rule_id')
    merged_contradictions = _unique(prior.get('contradictions', []) + contradictions,
                                    'evidence_id', 'rule_id')
    gaps = list(prior.get('gaps', []))
    if support_state == 'absent' and (not coverage or not merged_controls):
        support_state = 'unknown'
        gaps.append('support absent 缺少覆盖引用或匹配对照；降级为 unknown（暂时没找到）')

    cf, ignored_cf = _merge_cf(prior.get('cf', []),
                               _cf_entries(agent, layer, args.get('cf', []),
                                           reviewed, agent.verified))
    cf_states = {entry['rule_id']: entry['state'] for entry in cf}
    veto_by = sorted(rule for rule, state in cf_states.items() if state == 'absent')
    supported_rules = sorted({entry['rule_id'] for entry in merged_support})
    has_support = support_state == 'present'

    if veto_by or prior.get('state') == 'excluded':
        state = 'excluded'
    elif support_state == 'unsubmitted':
        state = 'unsubmitted'
    elif has_support:
        state = 'candidate'
    else:
        state = 'unsupported'

    optional_rules = ([rule for rule in predictions(layer) if rule not in supported_rules]
                      if state in ('candidate',) or support_state == 'unknown' else [])
    record = {
        'layer': layer, 'state': state, 'support': merged_support,
        'contradictions': merged_contradictions, 'controls': merged_controls,
        'cf': cf, 'cf_states': cf_states, 'veto_by': veto_by,
        'support_rules': {rule: {'state': 'present',
                                 'evidence_ids': sorted({e['evidence_id'] for e in merged_support
                                                         if e['rule_id'] == rule})}
                          for rule in supported_rules},
        'supported_rules': supported_rules, 'cached_rules': supported_rules,
        'optional_rules': optional_rules, 'support_state': support_state,
        'support_confirmed': support_state in ('present', 'absent'),
        'parked': support_state == 'unknown', 'support_coverage_ids': coverage,
        'weak_conflict': has_support and bool(merged_contradictions),
        'support_at_veto': has_support and state == 'excluded',
        'cf_gaps': [gap for entry in cf for gap in entry['gaps']],
        'gaps': list(dict.fromkeys(
            gaps + ([f'已忽略已闭合规则的重提：{", ".join(ignored_cf)}'] if ignored_cf else [])
            + (['已忽略对 present support 的撤销'] if ignored else []))),
        'auto_expanded': {**prior.get('auto_expanded', {}), **(args.get('_auto_expanded') or {})},
        'expectation': args.get('expectation') or prior.get('expectation', ''),
        'explanation': args.get('explanation') or prior.get('explanation', ''),
        'assessment_basis': 'llm_inference', 'validation': 'provenance_only'}
    if prior.get('state') == 'excluded':
        record['state'] = 'excluded'
        record['veto_by'] = prior.get('veto_by', veto_by)
    agent.assessments[layer] = record
    return {'layer': layer, **record, 'candidates': agent._candidates()}
