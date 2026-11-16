"""Submit one layer delta: provenance review and ledger commit in one call.

This is the LLM-facing Stage-1 entry point.  ``check_layer`` only records a
review; this tool performs the same completeness review internally and then
merges the delta through ``process_response``.  A tool-call round can therefore
advance the ledger directly, without spending a separate tool-free turn on a
JSON submission.
"""
from ..rules import LAYERS
from ..utils import worklist
from ..utils.check_observations import check
from ..utils.finish_diagnosis import ValidationErrors
from ..utils.process_response import run as process_response

NAME = 'submit_layer'
DESCRIPTION = ('提交一个故障层的增量判断：内部完成来源核验、规则复核并合并到 ledger。'
               '每次调用只提交一个层；同一轮可多次调用以推进多个层。')

ENTRY = {'type': 'object',
         'properties': {
             'evidence_id': {'type': 'string'},
             'rule_id': {'type': 'string'},
             'note': {'type': 'string'}},
         'required': ['evidence_id', 'rule_id'],
         'additionalProperties': False}
CF_ENTRY = {'type': 'object',
            'properties': {
                'rule_id': {'type': 'string'},
                'state': {'type': 'string', 'enum': ['present', 'absent', 'unknown']},
                'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                'control_ids': {'type': 'array', 'items': {'type': 'string'}},
                'coverage_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                'explanation': {'type': 'string'}},
            'required': ['rule_id', 'state'],
            'additionalProperties': False}
PARAMETERS = {
    'layer': {'type': 'string'},
    'hypothesis': {'type': 'string'},
    'expectation': {'type': 'string'},
    'support': {'type': 'array', 'items': ENTRY},
    'contradictions': {'type': 'array', 'items': ENTRY},
    'controls': {'type': 'array', 'items': ENTRY},
    'cf': {'type': 'array', 'items': CF_ENTRY},
    'support_state': {'type': 'string', 'enum': ['present', 'absent', 'unknown']},
    'support_coverage_ids': {'type': 'array', 'items': {'type': 'string'}},
    'explanation': {'type': 'string'},
    'stage1_complete': {'type': 'boolean'},
}
DELTA_FIELDS = ('layer', 'support', 'contradictions', 'controls', 'cf', 'support_state',
                'support_coverage_ids', 'expectation', 'explanation')


def _objects(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _field(entries, name):
    return [entry[name] for entry in entries
            if isinstance(entry.get(name), str) and entry[name].strip()]


def _id_list(entries, name):
    return [eid for entry in entries for eid in (entry.get(name) or [])
            if isinstance(eid, str) and eid.strip()]


def _unique(values):
    return list(dict.fromkeys(values))


def _review_arguments(arguments):
    """Build the internal check payload from one submit_layer delta."""
    support = _objects(arguments.get('support'))
    contradictions = _objects(arguments.get('contradictions'))
    controls = _objects(arguments.get('controls'))
    cf = _objects(arguments.get('cf'))
    evidence = support + contradictions + controls
    return {
        'layer': arguments.get('layer'),
        'hypothesis': arguments.get('hypothesis', ''),
        'expectation': arguments.get('expectation', ''),
        'rule_ids': _unique(_field(support + contradictions, 'rule_id')),
        'evidence_ids': _unique(_field(evidence, 'evidence_id')),
        'control_ids': _unique(_field(controls, 'evidence_id')),
        'cf_rule_ids': _unique(_field(cf, 'rule_id')),
        'cf_evidence_ids': _unique(_id_list(cf, 'evidence_ids')),
        'cf_control_ids': _unique(_id_list(cf, 'control_ids')),
        'cf_coverage_ids': _unique(_id_list(cf, 'coverage_evidence_ids')),
        'support_coverage_ids': [eid for eid in (arguments.get('support_coverage_ids') or [])
                                 if isinstance(eid, str)],
    }


def _delta(arguments):
    """Copy the delta and give optional CF arrays their documented empty default."""
    delta = {key: arguments[key] for key in DELTA_FIELDS if key in arguments}
    entries = delta.get('cf')
    if isinstance(entries, list):
        delta['cf'] = [
            {**entry, **{field: entry.get(field) or []
                         for field in ('evidence_ids', 'control_ids', 'coverage_evidence_ids')}}
            if isinstance(entry, dict) else entry
            for entry in entries]
    return delta


def _failure(agent, layer, message, code='layer_delta_invalid', review=None):
    return {'committed': False, 'layer': layer,
            'errors': [{'code': code, 'message': message}],
            'review_gaps': (review or {}).get('gaps', []),
            'worklist': worklist.summary(agent),
            'rounds_remaining': agent.max_rounds - len(agent.history)}


def run(agent, arguments, stage):
    layer = arguments.get('layer')
    if stage != 1:
        return _failure(agent, layer,
                        'submit_layer 只在 Stage 1 可用；Stage 2 请使用 submit_diagnosis')
    if layer not in LAYERS:
        return _failure(agent, layer, 'layer 必须来自规则注册表')
    if not any(key in arguments for key in
               ('support', 'support_state', 'contradictions', 'controls', 'cf',
                'support_coverage_ids', 'stage1_complete')):
        return _failure(agent, layer,
                        '提交至少需要 support/support_state/cf 等字段之一；'
                        '未评估层写 support: []')
    try:
        review = check(agent.registry, layer, _review_arguments(arguments), stage)
    except (ValueError, TypeError, KeyError) as exc:
        return _failure(agent, layer, str(exc), code='review_invalid')
    payload = {'assessments': [_delta(arguments)]}
    if 'stage1_complete' in arguments:
        payload['stage1_complete'] = arguments['stage1_complete']
    try:
        result = process_response(agent, payload)
    except ValidationErrors as exc:
        failure = _failure(agent, layer, exc.errors[0].get('message', '提交校验失败'),
                           review=review)
        failure['errors'] = exc.errors
        return failure
    except (ValueError, TypeError, KeyError) as exc:
        return _failure(agent, layer, str(exc), review=review)
    round_no = len(agent.history)
    update = next((item for item in result.get('assessments', [])
                   if item.get('layer') == layer), {})
    if result.get('pruning'):
        event = {'round': round_no, **result['pruning']}
        agent.pruning_log.append(event)
        agent._emit('prune', event)
    if result.get('transition'):
        agent.stage_two_round = round_no
        agent._emit('stage_transition', {'round': round_no, **result['transition']})
    ignored = update.get('ignored')
    if not ignored:
        agent.committed = True
        agent.stage_one_commits += 1
        agent.last_commit_round = round_no
    return {
        'committed': not ignored, 'layer': layer,
        'ignored': ignored,
        'state': update.get('state'),
        'support_state': update.get('support_state'),
        'support_confirmed': update.get('support_confirmed'),
        'parked': update.get('parked'),
        'cached_rules': update.get('cached_rules', []),
        'optional_rules': update.get('optional_rules', []),
        'cf_states': update.get('cf_states', {}),
        'veto_by': update.get('veto_by', []),
        'gaps': update.get('gaps', []),
        'review_gaps': review.get('gaps', []),
        'pruning': result.get('pruning'),
        'worklist': result.get('worklist'),
        'ledger': result.get('ledger'),
        'candidates': result.get('candidates'),
        'transition': result.get('transition'),
        'next': result.get('next'),
        'rounds_remaining': agent.max_rounds - round_no,
    }
