"""Verify cited data provenance and rule scope; hypothesis semantics belong to the LLM."""
from ..rules import LAYERS, controls, counterfactuals, predictions
from .observation_evidence import raw_fact

NAME = 'verify_counterfactual'
ROLES = {'support': 1, 'contradiction': 0, 'control': 'control',
         'counterfactual': 'counterfactual', 'coverage': 'coverage'}


def run(agent, args):
    if any(not isinstance(args.get(k), str) or not args[k].strip() for k in ('expectation', 'explanation')):
        raise ValueError('需说明预期、实际观测与推断理由')
    layer = args['layer']
    role = args.get('role')
    evidence_id, rule = args.get('evidence_id'), args.get('rule_id')
    if role not in ROLES:
        raise ValueError('role 必须是 support、contradiction、control、counterfactual 或 coverage')
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError('需引用记录级 evidence_id；observation-N 会由引擎展开')
    allowed = {'control': controls(layer), 'counterfactual': counterfactuals(layer),
               'support': predictions(layer), 'contradiction': predictions(layer),
               'coverage': tuple(predictions(layer)) + tuple(counterfactuals(layer))
               + tuple(controls(layer))}[role]
    fact = raw_fact(agent.registry, evidence_id, stage=1)
    valid = (layer in LAYERS and fact is not None and isinstance(rule, str)
             and rule in allowed and agent.registry.evidence.get(evidence_id, {}).get('tool') is not None)
    if valid:
        agent.verified.add((evidence_id, ROLES[role]))
    return {'verified': valid, 'evidence_id': evidence_id, 'rule_id': rule, 'role': role,
            'scope': '仅核验记录来源与规则归属；推断语义未由程序判定'}
