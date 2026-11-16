"""Submit the Stage-2 diagnosis through a tool call, at any round.

Stage 1 commits go through ``submit_layer``; this is its Stage-2 counterpart.
It can be called as soon as the model has enough evidence, so a validation
failure still leaves rounds to repair instead of consuming the final round.
"""
from copy import deepcopy

from ..utils.finish_diagnosis import ValidationErrors
from ..utils.process_response import run as process_response

NAME = 'submit_diagnosis'
DESCRIPTION = ('提交 Stage 2 最终诊断：layer_order、diagnoses、ranking_basis。'
               '准备充分即可调用；校验失败会返回 errors，可继续检索后再次提交。')

DIAGNOSIS = {'type': 'object',
             'properties': {
                 'layer': {'type': 'string'},
                 'fault_type': {'type': 'string'},
                 'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                 'corroboration_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                 'boundary_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                 'explanation': {'type': 'string'}},
             'required': ['layer', 'fault_type', 'evidence_ids', 'explanation']}
PARAMETERS = {
    'layer_order': {'type': 'array', 'items': {'type': 'string'}},
    'diagnoses': {'type': 'array', 'items': DIAGNOSIS},
    'ranking_basis': {'type': 'object'},
    'explanation': {'type': 'string'},
}


def run(agent, arguments, stage):
    if stage != 2:
        return {'committed': False,
                'errors': [{'code': 'stage', 'message': 'submit_diagnosis 只在 Stage 2 可用'}]}
    # Keep the last Stage-2 payload even if validation fails, so budget
    # exhaustion can salvage the ordering and any valid diagnoses.
    if isinstance(arguments.get('layer_order'), list):
        agent.last_stage_two = deepcopy(arguments)
    agent.context_stats['diagnosis_submissions'] += 1
    try:
        result = process_response(agent, arguments)
    except ValidationErrors as exc:
        return {'committed': False, 'errors': exc.errors,
                'rounds_remaining': agent.max_rounds - len(agent.history)}
    except (ValueError, TypeError, KeyError) as exc:
        return {'committed': False,
                'errors': [{'code': 'diagnosis_invalid', 'message': str(exc)}],
                'rounds_remaining': agent.max_rounds - len(agent.history)}
    return {'committed': True, 'diagnosis': result,
            'diagnostic_status': result.get('diagnostic_status'),
            'layer_order': result.get('layer_order'),
            'fault_order': result.get('fault_order'),
            'rounds_remaining': agent.max_rounds - len(agent.history)}
