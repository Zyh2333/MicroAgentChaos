"""Graph predicates kept by the state-driven agent.

The semantic state is pushed every round and the model chooses any layer, tool
or submission.  This module keeps the predicates those decisions are checked
against:

* ``layer_support``  - descriptive support facts for one layer (bundles,
  sources, corroboration) using the unified independence predicate;
* ``QueryLedger``    - exact and window-contained repeats are redundant;
* ``claim_guard``    - an ``absent`` counterfactual needs an independent control.

Every check is a predicate over the graph (bundle equality, source identity,
independence), never a configured number.
"""
import json

WINDOW_KEYS = ("start", "end", "start_line", "offset", "limit")


# ------------------------------------------------------------------ evidence
def cited(assessments, layer):
    record = assessments.get(layer) or {}
    return [entry["evidence_id"] for entry in record.get("support") or ()
            if isinstance(entry, dict) and isinstance(entry.get("evidence_id"), str)]


def source_of(graph, identifier):
    observation = graph.observations.get(identifier)
    anchor = getattr(observation, "anchor", None)
    return getattr(anchor, "file", None)


def known(observations, identifiers):
    return [i for i in identifiers if i in observations]


def corroborated(graph, identifiers):
    """Two cited observations from one corpus, independent by the unified predicate.

    ``graph.independent`` is the same predicate the INDEPENDENT edge and the
    counterfactual control guard use: different bundle, no shared reference or
    identifier, a different producer, and a gap beyond the observed execution.
    """
    items = known(graph.observations, identifiers)
    for i, left in enumerate(items):
        for right in items[i + 1:]:
            if graph.independent(left, right):
                return True
    return False


def layer_support(graph, assessments, layer):
    """Descriptive support facts; decisions use the predicates above."""
    identifiers = cited(assessments, layer)
    known_ids = known(graph.observations, identifiers)
    sources = {source_of(graph, i) for i in known_ids}
    sources.discard(None)
    bundles = graph.bundle_count(known_ids)
    # The ledger is authoritative once a claim exists, so a retracted proposition
    # reads 0 instead of falling back to its (still present) citations.
    in_degree = getattr(graph, "support_in_degree", lambda *_: bundles)(layer, known_ids)
    diversity = getattr(graph, "support_diversity",
                        lambda *_: len(sources))(layer, known_ids)
    return {"citations": len(identifiers),
            "distinct": len({graph.class_of(i) for i in known_ids}),
            "bundles": bundles, "in_degree": in_degree, "diversity": diversity,
            "sources": len(sources),
            "corroborated": corroborated(graph, identifiers)}


# -------------------------------------------------------------------- ledger
class QueryLedger:
    """Coverage ledger: exact repeats and window-contained repeats are redundant."""

    def __init__(self):
        self.exact = {}
        self.entries = []

    @staticmethod
    def _canonical(arguments):
        return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)

    def redundant(self, tool, arguments):
        key = (tool, self._canonical(arguments))
        if key in self.exact:
            return self.exact[key]
        for prior_tool, prior_args, prior_key in self.entries:
            if prior_tool != tool:
                continue
            if self._contains(prior_args, arguments or {}):
                return prior_key
        return None

    @staticmethod
    def _contains(prior, current):
        prior_core = {k: v for k, v in prior.items() if k not in WINDOW_KEYS}
        current_core = {k: v for k, v in current.items() if k not in WINDOW_KEYS}
        if prior_core != current_core:
            return False
        if "start" not in prior or "end" not in prior:
            return False
        if not isinstance(current.get("start"), (int, float)) or not isinstance(
                current.get("end"), (int, float)):
            return False
        return prior["start"] <= current["start"] and prior["end"] >= current["end"]

    def add(self, tool, arguments, evidence_id=None):
        key = (tool, self._canonical(arguments))
        self.exact.setdefault(key, evidence_id or key)
        self.entries.append((tool, dict(arguments or {}), evidence_id or key))


def claim_guard(graph, payload, resolve):
    """Reject a circular refutation before it reaches the ledger."""
    errors = []
    support = []
    for entry in payload.get("support") or ():
        if isinstance(entry, dict):
            items, _ = resolve([entry.get("evidence_id")])
            support.extend(items)
    for index, entry in enumerate(payload.get("cf") or (), 1):
        if not isinstance(entry, dict) or entry.get("state") != "absent":
            continue
        covered, unknown = resolve(entry.get("evidence_ids") or [])
        controls, missing_controls = resolve(
            (entry.get("control_ids") or []) + (entry.get("coverage_evidence_ids") or []))
        if unknown or missing_controls:
            errors.append({"code": "evidence_invalid", "index": index,
                           "message": "反事实引用无法解析为已读记录，不能判 absent"})
            continue
        if not controls:
            errors.append({"code": "control_missing", "index": index,
                           "message": "判 absent 需要匹配对照；请先读取对照组"})
            continue
        if support and not any(graph.independent(s, c)
                               for s in support for c in controls):
            errors.append({"code": "refutation_depends_on_support", "index": index,
                           "message": "对照与支持证据属于同一证据包，否决无效；"
                                      "请提供独立来源的对照（不同执行域或不同服务）"})
        if not covered:
            errors.append({"code": "coverage_missing", "index": index,
                           "message": "判 absent 需要覆盖记录，说明你检查了哪些范围"})
    return errors
