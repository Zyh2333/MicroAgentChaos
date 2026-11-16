"""Validate provenance of LLM-cited observations without interpreting their text."""


def raw_fact(registry, evidence_id, stage=None):
    fact = registry.evidence.get(evidence_id, {})
    if (fact.get('kind') != 'raw_observation' or not fact.get('text', '').strip()
            or not fact.get('references') or stage is not None and fact.get('stage') != stage):
        return None
    return fact
