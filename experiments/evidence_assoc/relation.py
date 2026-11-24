"""Pairwise relation inference with explicit abstention.

Four relations only: SAME, DEPENDENT, INDEPENDENT, UNKNOWN.

Two rules keep this from degrading into similarity clustering:

* UNKNOWN is the default.  A pair is promoted only when the promoted relation is
  backed by the dimensions that are actually available.
* INDEPENDENT requires *positive* separation: a different producer (service/pod),
  no shared reference or identifier, and a gap beyond the observed execution.

v0.2 keeps the decision ladder but drops weighted scores from the decision:
identity-based branches use a fixed confidence, similarity-based branches use the
unweighted mean of the available dimensions.
"""
from dataclasses import dataclass

from .compat import combine, compatibility
from .extract import (SCOPE_KEYS, producer_separated, scope_conflict, shared_identifiers,
                      shared_refs)
from .schema import Relation


@dataclass(frozen=True)
class Policy:
    """One irreducible constant.

    Text-only dependence needs a floor for "similar enough"; everything else is
    derived: the time window is the observed execution duration of this corpus,
    structural comparisons use ordinal levels, and independence is defined as
    *below* the linking floor rather than with its own constant.
    """
    link_semantic: float = 0.62


@dataclass
class Verdict:
    a: str
    b: str
    relation: Relation
    confidence: float
    basis: tuple
    reason: str
    order: int = 0                     # -1: a before b, 1: b before a, 0: unknown
    detail: dict = None

    @property
    def pair(self):
        return tuple(sorted((self.a, self.b)))

    def to_dict(self):
        return {"a": self.a, "b": self.b, "relation": self.relation.value,
                "confidence": round(self.confidence, 4), "basis": list(self.basis),
                "reason": self.reason, "order": self.order, "detail": self.detail or {}}


def _order(a, b):
    span_a, span_b = a.features.span, b.features.span
    if span_a is None or span_b is None:
        return 0
    if span_a[1] <= span_b[0]:
        return -1
    if span_b[1] <= span_a[0]:
        return 1
    return 0


def classify(a, b, corpus, topology=None, policy=None):
    policy = policy or Policy()
    if a.id == b.id:
        return Verdict(a.id, b.id, Relation.SAME, 1.0, ("same_observation",), "identity")

    # ---- source identity and lineage -------------------------------------
    shared = shared_refs(a, b)
    if shared:
        if a.anchor is not None and a.anchor == b.anchor:
            if a.text.strip() == b.text.strip() or a.content_key == b.content_key:
                return Verdict(a.id, b.id, Relation.SAME, 1.0, ("same_record",), "same_record",
                               detail={"refs": sorted(shared)[:4]})
            # Aggregate rows share a coarse location while each row is a different
            # comparison.  Same source, different fact.
            return Verdict(a.id, b.id, Relation.DEPENDENT, 0.85, ("same_source_anchor",),
                           "shared_source_anchor", detail={"refs": sorted(shared)[:4]})
        lineage = bool(set(a.derived_from) & set(b.refs) or set(b.derived_from) & set(a.refs))
        if lineage or a.is_aggregate or b.is_aggregate:
            return Verdict(a.id, b.id, Relation.DEPENDENT, 0.95, ("derived_from_source",),
                           "lineage", detail={"refs": sorted(shared)[:4]})
        return Verdict(a.id, b.id, Relation.DEPENDENT, 0.80, ("shared_source",), "shared_source",
                       detail={"refs": sorted(shared)[:4]})

    ids = shared_identifiers(a, b)
    dims = compatibility(a, b, corpus, topology)
    semantic, structural, contextual = dims
    score, coverage = combine(dims)
    order = _order(a, b)
    detail = {"dimensions": {d.name: d.to_dict() for d in dims},
              "combined": score, "coverage": coverage}

    # ---- identity anchors ------------------------------------------------
    scope_ids = {item for item in ids if item[0] in SCOPE_KEYS}
    if ids:
        # A shared identifier is an identity fact, not a text similarity: a
        # request line and its result line share almost no text.
        reason = "shared_scope" if scope_ids else "shared_identifier"
        basis = ("shared_scope_identifier" if scope_ids else "shared_identifier",) + semantic.basis
        return Verdict(a.id, b.id, Relation.DEPENDENT, 0.90 if scope_ids else 0.70,
                       basis, reason, order,
                       {**detail, "identifiers": sorted((scope_ids or ids))[:6]})

    # ---- repeated content ------------------------------------------------
    if a.content_key == b.content_key:
        if scope_conflict(a, b):
            return Verdict(a.id, b.id, Relation.UNKNOWN, 0.0,
                           ("identical_content", "scope_conflict"),
                           "repeated_content_distinct_scope", 0, detail)
        # Identical text at different source locations is not one fact: the same
        # line re-emitted, a retry, or two services sharing a logger look alike.
        return Verdict(a.id, b.id, Relation.UNKNOWN, 0.0,
                       ("identical_content", "distinct_source"), "repeated_content", 0, detail)

    # ---- text-only dependence -------------------------------------------
    # The time window is the corpus's longest observed execution: two records
    # can be one execution only if their gap fits inside an execution that was
    # actually observed here.
    span = corpus.execution_span if corpus.execution_span else corpus.time_scale
    gap = (contextual.detail or {}).get("gap_seconds")
    if (gap is not None and gap <= span
            and semantic.value is not None and semantic.value >= policy.link_semantic
            and coverage >= 2):
        return Verdict(a.id, b.id, Relation.DEPENDENT, score,
                       ("semantic", "within_execution_span"), "paraphrase_temporal",
                       order, {**detail, "execution_span": span})

    # ---- positive independence ------------------------------------------
    if (gap is not None and gap > span
            and producer_separated(a, b)
            and semantic.value is not None and semantic.value < policy.link_semantic
            and not ids and not shared):
        return Verdict(a.id, b.id, Relation.INDEPENDENT, round(1.0 - score, 4),
                       ("producer_separation", "beyond_execution_span", "below_link_floor"),
                       "resolved_separation", 0, {**detail, "execution_span": span})

    if coverage == 0:
        reason = "no_basis"
    elif coverage < len(dims):
        reason = "insufficient_coverage"
    else:
        reason = "weak_evidence"
    return Verdict(a.id, b.id, Relation.UNKNOWN, 0.0, (), reason, order, detail)
