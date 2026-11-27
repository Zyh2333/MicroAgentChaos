"""Mechanism propositions, evidence bundles and graph-driven query needs.

A proposition is a rule instance from ``rules.yaml`` (``F1-R1``) plus the evidence
the agent cites for it.  Support strength is *not* the number of citations:

* duplicate citations of the same record collapse to one observation,
* records linked by SAME or DEPENDENT collapse to one evidence bundle,
* derived records (aggregates/comparisons) stay in the bundle of their sources.

So ``strength = graph.bundle_count(evidence_ids)``.  A proposition with five
citations and one bundle is exactly the "false support" RQ2 measures.
"""
from collections import Counter
from dataclasses import dataclass, field

from .schema import PropositionState

SUPPORT = "support"
CONTRADICTION = "contradiction"
CONTROL = "control"
CF = "cf"
DIAGNOSIS = "diagnosis"
CORROBORATION = "corroboration"
BOUNDARY = "boundary"
KINDS = (SUPPORT, CONTRADICTION, CONTROL, CF, DIAGNOSIS, CORROBORATION, BOUNDARY)


@dataclass
class Claim:
    rule_id: str
    kind: str = SUPPORT
    evidence_ids: list = field(default_factory=list)
    control_ids: list = field(default_factory=list)
    state: str | None = None
    note: str = ""

    @property
    def layer(self):
        for separator in ("-", ":"):
            if separator in self.rule_id:
                return self.rule_id.split(separator)[0]
        return self.rule_id


@dataclass
class Proposition:
    id: str
    rule: str
    layer: str
    state: str
    claims: list
    evidence_ids: list
    distinct_observations: int = 0
    bundles: int = 0
    duplicate_citations: int = 0
    dependent_corroboration: int = 0
    subjects: tuple = ()
    gaps: list = field(default_factory=list)
    controls_independent: bool | None = None
    # Layer-level statuses, repeated on every rule of the layer so the view can
    # decide whether a supported rule is actually closed.
    cf: dict = field(default_factory=dict)          # CF rule -> present|absent|unknown
    controls: dict = field(default_factory=dict)    # C rule -> present
    cf_refuted: tuple = ()                          # CF rules that validly veto the layer

    @property
    def supported(self):
        return self.state == PropositionState.SUPPORTED.value

    @property
    def single_source(self):
        return self.supported and self.bundles == 1 and self.distinct_observations > 1

    @property
    def single_bundle(self):
        """Supported but resting on one evidence bundle, whatever the citation count."""
        return self.supported and self.bundles <= 1

    def to_dict(self):
        return {"id": self.id, "rule": self.rule, "layer": self.layer, "state": self.state,
                "claims": len(self.claims), "evidence_ids": len(self.evidence_ids),
                "distinct_observations": self.distinct_observations, "bundles": self.bundles,
                "duplicate_citations": self.duplicate_citations,
                "dependent_corroboration": self.dependent_corroboration,
                "single_source": self.single_source, "single_bundle": self.single_bundle,
                "subjects": list(self.subjects),
                "gaps": self.gaps, "controls_independent": self.controls_independent,
                "cf": dict(self.cf), "controls": dict(self.controls),
                "cf_refuted": list(self.cf_refuted)}


def _unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def assess(graph, claims, indep=None):
    """Turn flat claims into propositions with graph-derived strength.

    ``indep`` optionally supplies a memoized ``graph.independent`` lookup; the
    removal-combination search evaluates the same evidence pairs repeatedly.
    """
    lookup = indep or graph.independent
    grouped = {}
    order = []
    for claim in claims:
        key = claim.rule_id
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(claim)

    # CF/C statuses are layer-level: a support rule is closed only when the
    # layer's necessity signature has been decided, not merely because it has
    # one supporting bundle.
    cf_by_layer = {}
    controls_by_layer = {}
    for claim in claims:
        layer = claim.layer
        if claim.kind == CF:
            cf_by_layer.setdefault(layer, {})[claim.rule_id] = claim.state or "unknown"
        elif claim.kind == CONTROL:
            controls_by_layer.setdefault(layer, {})[claim.rule_id] = "present"

    propositions = []
    all_ids = set(graph.observations)
    for rule_id in order:
        group = grouped[rule_id]
        evidence_ids = _unique([i for claim in group if claim.kind in (SUPPORT, DIAGNOSIS, CF)
                                for i in claim.evidence_ids])
        known = [i for i in evidence_ids if i in all_ids]
        unknown = [i for i in evidence_ids if i not in all_ids]
        distinct = len({graph.class_of(i) for i in known})
        bundles = graph.bundle_count(known)
        counts = Counter()
        for i in known:
            for entity in graph.observations[i].features.entities:
                if entity.count(".") == 3 or entity.startswith("node-"):
                    continue
                counts[entity] += 1
        subjects = tuple(name for name, _ in counts.most_common(3))

        controls = _unique([i for claim in group if claim.kind in (CF, CONTROL)
                            for i in claim.control_ids + claim.evidence_ids])
        controls_known = [i for i in controls if i in all_ids]
        independent_control = None
        if known and controls_known:
            independent_control = any(lookup(s, c) for s in known for c in controls_known)

        refutations = [claim for claim in group
                       if claim.kind == CF and (claim.state == "absent")]
        gaps = []
        if not evidence_ids:
            state = PropositionState.UNRESOLVED.value
            gaps.append("no_evidence")
        elif refutations:
            if independent_control:
                state = PropositionState.REFUTED.value
            else:
                state = PropositionState.UNRESOLVED.value
                gaps.append("refutation_depends_on_support")
        else:
            state = PropositionState.SUPPORTED.value
            if bundles == 1 and distinct > 1:
                gaps.append("single_bundle")
        if unknown:
            gaps.append("unresolved_evidence_ids")

        propositions.append(Proposition(
            id=f"{rule_id}::{subjects[0]}" if subjects else rule_id, rule=rule_id,
            layer=group[0].layer, state=state, claims=group, evidence_ids=evidence_ids,
            distinct_observations=distinct, bundles=bundles,
            duplicate_citations=len(evidence_ids) - distinct,
            dependent_corroboration=max(0, distinct - bundles),
            subjects=tuple(subjects), gaps=gaps, controls_independent=independent_control,
            cf=dict(cf_by_layer.get(group[0].layer, {})),
            controls=dict(controls_by_layer.get(group[0].layer, {}))))

    # An absent CF is a veto only when its proposition is ``refuted`` (absent
    # with an independent control).  Otherwise the necessity signature is still
    # open: report it as unknown, not as a decided absent.
    invalid_absent = {}
    for proposition in propositions:
        if (proposition.state != PropositionState.REFUTED.value
                and any(claim.kind == CF for claim in proposition.claims)
                and proposition.cf.get(proposition.rule) == "absent"):
            invalid_absent.setdefault(proposition.layer, set()).add(proposition.rule)
    for proposition in propositions:
        for rule in invalid_absent.get(proposition.layer, ()):
            if proposition.cf.get(rule) == "absent":
                proposition.cf[rule] = "unknown"

    # A CF veto is valid only when its absent signature had an independent
    # control; that is exactly the CF proposition state ``refuted``.
    refuted_by_layer = {}
    for proposition in propositions:
        if (proposition.state == PropositionState.REFUTED.value
                and any(claim.kind == CF for claim in proposition.claims)):
            refuted_by_layer.setdefault(proposition.layer, set()).add(proposition.rule)
    for proposition in propositions:
        proposition.cf_refuted = tuple(sorted(refuted_by_layer.get(proposition.layer, ())))
    return propositions


def query_needs(propositions):
    """Graph-driven next-step needs: what would separate the remaining candidates.

    The mapping is by *gap type* and tool capability, never by fault type, so it
    transfers to unseen observations and unseen fault families.
    """
    by_layer = {}
    for proposition in propositions:
        if proposition.state == PropositionState.UNRESOLVED.value:
            by_layer.setdefault(proposition.layer, []).append(proposition)

    needs = []
    for proposition in propositions:
        if proposition.state != PropositionState.UNRESOLVED.value:
            continue
        if "no_evidence" in proposition.gaps:
            reason = "coverage"
        elif "refutation_depends_on_support" in proposition.gaps:
            reason = "independence"
        elif proposition.bundles <= 1:
            reason = "independence"
        else:
            reason = "contrast"
        rivals = [p for p in by_layer.get(proposition.layer, ()) if p.rule != proposition.rule]
        needs.append({"proposition": proposition.id, "layer": proposition.layer,
                      "reason": reason, "subjects": list(proposition.subjects),
                      "expected_discrimination": 1 + len(rivals),
                      "rivals": [p.id for p in rivals]})
    needs.sort(key=lambda need: (-need["expected_discrimination"], need["proposition"]))
    return needs


def summary(propositions):
    states = Counter(p.state for p in propositions)
    return {"propositions": len(propositions), "states": dict(states),
            "supported": states.get(PropositionState.SUPPORTED.value, 0),
            "refuted": states.get(PropositionState.REFUTED.value, 0),
            "unresolved": states.get(PropositionState.UNRESOLVED.value, 0),
            "single_source_support": sum(1 for p in propositions if p.single_source),
            "single_bundle_support": sum(1 for p in propositions if p.single_bundle),
            "citations": sum(len(p.evidence_ids) for p in propositions),
            "distinct_observations": sum(p.distinct_observations for p in propositions),
            "bundles": sum(p.bundles for p in propositions),
            "duplicate_citations": sum(p.duplicate_citations for p in propositions),
            "dependent_corroboration": sum(p.dependent_corroboration for p in propositions)}
