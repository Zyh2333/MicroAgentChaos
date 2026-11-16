"""Normalize cited ids: tool-level observations expand to their record ids.

The LLM frequently cites ``observation-15`` when it means
``observation-15:record:3``.  Expansion is scope-limited: assessment citations
only expand to records that were already reviewed through ``check_layer``;
``check_layer`` inputs expand to the records read in the current stage.  No
expansion crosses a stage or an observation.
"""
import re

OBSERVATION = re.compile(r'^observation-\d+$')
RECORD = re.compile(r'^(observation-\d+):record:(\d+)$')


def _order(evidence_id):
    match = RECORD.match(evidence_id)
    if match:
        return int(match.group(1).split('-')[1]), int(match.group(2))
    if OBSERVATION.match(evidence_id):
        return int(evidence_id.split('-')[1]), -1
    return 0, 0


def records_for(registry, observation, stage=None):
    """Record ids belonging to one observation, restricted to a stage."""
    prefix = observation + ':record:'
    return sorted((eid for eid, fact in registry.evidence.items()
                   if eid.startswith(prefix) and fact.get('kind') == 'raw_observation'
                   and (stage is None or fact.get('stage') == stage)), key=_order)


def observation_ids(registry, stage=None):
    return sorted({eid.split(':record:')[0] for eid, fact in registry.evidence.items()
                   if fact.get('kind') == 'raw_observation'
                   and (stage is None or fact.get('stage') == stage)},
                  key=lambda eid: int(eid.split('-')[1]))


def _sample(ids, limit=12):
    ordered = sorted(ids, key=_order)
    return ordered[:limit]


def _fail(field, evidence_id, valid):
    hint = '、'.join(_sample(valid)) or '（当前没有可引用记录）'
    raise ValueError(f'{field} 的 {evidence_id} 无法解析为该范围内的记录 ID；'
                     f'可引用示例：{hint}')


def expand(ids, reviewed, field, registry=None, stage=None):
    """Expand tool-level ids; return (ids, {original: expanded}).

    ``reviewed`` is the allowed record set for assessment citations.  When
    ``registry``/``stage`` are given (check_layer inputs), the allowed set is
    the records read in that stage instead.
    """
    if not isinstance(ids, list) or any(not isinstance(eid, str) for eid in ids):
        raise ValueError(f'{field} 必须为记录 ID 数组')
    out, expanded = [], {}
    for evidence_id in ids:
        if not OBSERVATION.match(evidence_id):
            out.append(evidence_id)
            continue
        if registry is None:
            matches = [eid for eid in reviewed if eid.startswith(evidence_id + ':record:')]
            valid = reviewed
        else:
            matches = records_for(registry, evidence_id, stage)
            valid = _stage_records(registry, stage)
        if not matches:
            _fail(field, evidence_id, valid)
        matches = sorted(set(matches), key=_order)
        out.extend(matches)
        expanded[evidence_id] = matches
    return list(dict.fromkeys(out)), expanded


def _stage_records(registry, stage):
    return [eid for eid, fact in registry.evidence.items()
            if fact.get('kind') == 'raw_observation' and fact.get('stage') == stage]


def reviewed_for(agent, layer, stage=1):
    """Records and rule ids the LLM has reviewed for one layer."""
    reviews = [r for r in agent.registry.verifications
               if r.get('layer') == layer and r.get('stage') == stage]
    reviewed = {row['evidence_id'] for review in reviews
                for key in ('evidence', 'controls', 'cf_evidence', 'cf_controls',
                            'cf_coverage', 'support_coverage')
                for row in review.get(key, [])}
    reviewed_rules = {rule['id'] for review in reviews
                      for key in ('rules', 'cf_rules') for rule in review.get(key, [])}
    return reviews, reviewed, reviewed_rules
