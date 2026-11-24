"""Required/optional work derived from the incremental layer ledger.

Stage 1 only blocks on what can still change the candidate set: an
unsubmitted layer's support verdict, and an open CF verdict for a supported
layer.  Satisfied R rules are cached (``supported_rules``), the rest of that
layer's R rules become optional strengthening, and C controls never block.
"""
from ..rules import LAYERS, controls, counterfactuals, predictions

REQUIRED_READERS = ('read_metrics', 'read_logs', 'read_execution_graph')
# Unresolved layers remain optional, like previously parked unsupported layers.
NONBLOCKING_STATES = ('excluded', 'unsupported', 'skipped', 'unresolved')


def record(assessments, layer):
    return assessments.get(layer) or {}


def candidates(assessments):
    return [layer for layer in LAYERS
            if record(assessments, layer).get('state') == 'candidate']


def visible_candidates(assessments):
    """Unassessed layers stay visible so a backend failure does not hide them."""
    return [layer for layer in LAYERS
            if record(assessments, layer).get('state') in (None, 'unsubmitted', 'candidate')]


def parked(assessments):
    return [layer for layer in LAYERS if record(assessments, layer).get('parked')]


def skipped(assessments):
    return [layer for layer in LAYERS
            if record(assessments, layer).get('state') == 'skipped']


def readers_done(agent):
    return set(REQUIRED_READERS) <= set(agent.registry.calls)


def required(assessments, skip_unsubmitted=False):
    """Work items that must be decided before Stage 2."""
    items = []
    for layer in LAYERS:
        current = record(assessments, layer)
        state = current.get('state')
        if state in NONBLOCKING_STATES:
            continue
        if state in (None, 'unsubmitted'):
            if not skip_unsubmitted:
                items.append({'layer': layer, 'kind': 'support',
                              'rules': list(predictions(layer))})
            continue
        open_cf = [rule for rule in counterfactuals(layer)
                   if rule not in current.get('cf_states', {})]
        if open_cf:
            items.append({'layer': layer, 'kind': 'cf', 'rules': open_cf})
    return items


def optional(assessments):
    """Non-blocking strengthening: remaining R rules and C controls."""
    items = []
    for layer in LAYERS:
        current = record(assessments, layer)
        state = current.get('state')
        cached = set(current.get('supported_rules', []))
        if state == 'candidate':
            rest = [rule for rule in predictions(layer) if rule not in cached]
            if rest:
                items.append({'layer': layer, 'kind': 'support_optional', 'rules': rest,
                              'status': 'candidate'})
            if controls(layer):
                items.append({'layer': layer, 'kind': 'control', 'rules': list(controls(layer)),
                              'status': 'candidate'})
        elif current.get('parked') and state not in ('excluded', 'skipped'):
            # An excluded layer is monotone: its rules were closed by the veto
            # and must not reappear as optional strengthening.
            rest = [rule for rule in predictions(layer) if rule not in cached]
            if rest:
                items.append({'layer': layer, 'kind': 'support_optional', 'rules': rest,
                              'status': 'parked'})
    return items


def ready(agent, complete=False):
    """Readers done and no blocking work; ``complete`` skips unsubmitted layers."""
    return readers_done(agent) and not required(agent.assessments, skip_unsubmitted=complete)


def early_exit(agent, complete=False):
    """True when ``complete`` closes the stage with layers left unsubmitted."""
    return bool(complete) and ready(agent, complete=True) and any(
        record(agent.assessments, layer).get('state') in (None, 'unsubmitted')
        for layer in LAYERS)


def skipped_record(layer):
    return {'layer': layer, 'state': 'skipped', 'support': [], 'contradictions': [],
            'controls': [], 'cf': [], 'cf_states': {}, 'veto_by': [],
            'support_state': 'unsubmitted', 'support_confirmed': False, 'parked': False,
            'supported_rules': [], 'cached_rules': [], 'optional_rules': [],
            'support_coverage_ids': [], 'cf_gaps': [], 'gaps': [],
            'explanation': '未在 Stage 1 内评估；由单候选早停跳过',
            'assessment_basis': 'engine_prune', 'validation': 'skipped'}


def freeze_skipped(agent):
    """Materialize unsubmitted layers as skipped so they never inflate candidates."""
    frozen = []
    for layer in LAYERS:
        current = agent.assessments.get(layer)
        if current is None or current.get('state') in (None, 'unsubmitted'):
            agent.assessments[layer] = skipped_record(layer)
            frozen.append(layer)
    return frozen


def effects(before, after):
    """Pruning effects of one submission, for the compact feedback payload."""
    cached, pruned, closed, reasons = [], [], [], {}
    for layer in LAYERS:
        old = before.get(layer) or {}
        new = after.get(layer) or {}
        state = new.get('state')
        confirmed_absent = state == 'unsupported' and not new.get('parked')
        if (state == 'excluded' or confirmed_absent) and old.get('state') != state:
            pruned.append(layer)
            reasons[layer] = 'CF veto' if state == 'excluded' else 'confirmed absent'
            closed.extend(predictions(layer) + counterfactuals(layer) + controls(layer))
        if new.get('state') == 'candidate' and old.get('state') != 'candidate':
            cached.extend(new.get('supported_rules', []))
        for rule in new.get('cf_states', {}):
            if rule not in old.get('cf_states', {}):
                closed.append(rule)
    return {'cached_rules': sorted(set(cached)), 'pruned_layers': pruned,
            'closed_rules': sorted(set(closed)), 'reasons': reasons}


def summary(agent):
    return {'required': required(agent.assessments),
            'optional': optional(agent.assessments),
            'candidates': candidates(agent.assessments),
            'parked': parked(agent.assessments),
            'skipped': skipped(agent.assessments),
            'readers_done': readers_done(agent)}
