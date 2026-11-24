"""Offline audit of baseline artifacts: RQ1 relation quality and RQ2 support strength.

RQ1 needs gold relations that are independent of the association algorithm.  Two
structural facts qualify:

* two evidence ids with the same file+location are the same record  -> gold SAME
* two records sharing a trace/task/request identifier belong to one execution
  -> gold DEPENDENT

Everything else stays unlabeled.  Unlabeled pairs are never counted as negatives,
and UNKNOWN predictions are counted as abstentions rather than errors, so the
report can state precision at a stated coverage instead of hiding abstention.
"""
from collections import Counter, defaultdict
from pathlib import Path

from .adapters import (ServiceGraph, TopologyIndex, claims_from_artifact,
                       observations_from_artifact, service_graph_from_sample)
from .associate import Associator
from .propositions import (DIAGNOSIS, SUPPORT, CORROBORATION, assess, query_needs,
                           summary as propositions_summary)
from .relation import Policy
from .schema import Relation

SCOPE_KEYS = ("trace_id", "traceid", "task_id", "taskid", "flowid", "request_id", "requestid")


def gold_relation(a, b):
    if a.anchor is not None and a.anchor == b.anchor:
        if a.text.strip() == b.text.strip():
            return Relation.SAME
        # Same coarse source location, different content: a shared source, not one fact.
        return Relation.DEPENDENT
    left, right = a.features.identifier_map, b.features.identifier_map
    for key in SCOPE_KEYS:
        if left.get(key) and left.get(key) == right.get(key):
            return Relation.DEPENDENT
    return None


def _gold_pairs(graph):
    """Indexed enumeration of structural gold pairs - no O(n^2) scan."""
    buckets = defaultdict(list)
    for observation in graph.observations.values():
        if observation.anchor is not None:
            buckets[("ref", observation.anchor.key)].append(observation.id)
        identifiers = observation.features.identifier_map
        for key in SCOPE_KEYS:
            if identifiers.get(key):
                buckets[("scope", key, identifiers[key])].append(observation.id)
    pairs = set()
    for members in buckets.values():
        ordered = sorted(set(members))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1:]:
                pairs.add((left, right))
    return pairs


def _decided(graph):
    """Relation asserted by the graph, read from class/bundle membership.

    Reading the edge table directly is unreliable: a pair can gain a DEPENDENT
    edge before a later SAME merge puts both endpoints in one class.  Class and
    bundle membership are the authoritative statements.
    """
    roots = {observation_id: graph.bundle_of(observation_id) for observation_id in graph.observations}
    independent = {tuple(sorted((edge.a, edge.b))) for edge in graph.edges.values()
                   if edge.relation == Relation.INDEPENDENT}
    return {"roots": roots, "independent": independent}


def _score(gold, predicted):
    if predicted is None:
        return "abstain"
    if gold == Relation.SAME:
        return "correct" if predicted == Relation.SAME else (
            "partial" if predicted == Relation.DEPENDENT else "error")
    if gold == Relation.DEPENDENT:
        return "correct" if predicted == Relation.DEPENDENT else (
            "partial" if predicted == Relation.SAME else "error")
    return "abstain"


def rq1(graph):
    decided = _decided(graph)
    roots, independent = decided["roots"], decided["independent"]
    reverse = {tuple(sorted((edge.a, edge.b))): edge for edge in graph.edges.values()}
    counts = Counter()
    by_gold = defaultdict(Counter)
    contradictions = []
    for pair in _gold_pairs(graph):
        if pair[0] not in graph.observations or pair[1] not in graph.observations:
            continue
        gold = gold_relation(graph.observations[pair[0]], graph.observations[pair[1]])
        if gold is None:
            continue
        if graph.class_of(pair[0]) == graph.class_of(pair[1]):
            predicted = Relation.SAME
        elif roots[pair[0]] == roots[pair[1]]:
            predicted = Relation.DEPENDENT
        elif pair in independent:
            predicted = Relation.INDEPENDENT
        else:
            predicted = None
        outcome = _score(gold, predicted)
        counts[outcome] += 1
        by_gold[gold.value][outcome] += 1
        if outcome == "error" and len(contradictions) < 20:
            edge = reverse.get(pair)
            contradictions.append({"a": pair[0], "b": pair[1], "gold": gold.value,
                                   "predicted": predicted.value if predicted else None,
                                   "basis": list(edge.basis) if edge else [],
                                   "confidence": edge.confidence if edge else None})
    decided = counts["correct"] + counts["partial"] + counts["error"]
    labelled = decided + counts["abstain"]
    return {
        "gold_pairs": labelled,
        "decided": decided,
        "correct": counts["correct"],
        "partial": counts["partial"],
        "errors": counts["error"],
        "abstained": counts["abstain"],
        "coverage": round(decided / labelled, 4) if labelled else 0.0,
        "precision": round((counts["correct"] + 0.5 * counts["partial"]) / decided, 4) if decided else 0.0,
        "strict_precision": round(counts["correct"] / decided, 4) if decided else 0.0,
        "by_gold": {k: dict(v) for k, v in by_gold.items()},
        "contradictions": contradictions,
    }


def _claim_table(graph, claims, kind):
    """Aggregate claims by rule: citations vs distinct observations vs bundles."""
    groups = defaultdict(list)
    for claim in claims:
        if claim.kind == kind and claim.evidence_ids:
            groups[claim.rule_id].extend(claim.evidence_ids)
    rows = []
    for rule, ids in sorted(groups.items()):
        known = [i for i in ids if i in graph.observations]
        distinct = len({graph.class_of(i) for i in known})
        rows.append({"rule": rule,
                     "citations": len(ids),
                     "distinct_observations": distinct,
                     "bundles": graph.bundle_count(known),
                     "co_sourced": len(known) - distinct})
    return rows


def rq2(graph, claims, propositions):
    layer_rows = _claim_table(graph, claims, SUPPORT)
    diagnosis_rows = _claim_table(graph, claims, DIAGNOSIS)
    corroboration = _claim_table(graph, claims, CORROBORATION)
    # Corroboration only adds information if it is outside every diagnosis bundle.
    diagnoses, extras = defaultdict(list), defaultdict(list)
    for claim in claims:
        if claim.kind == DIAGNOSIS:
            diagnoses[claim.rule_id].extend(claim.evidence_ids)
        elif claim.kind == CORROBORATION:
            extras[claim.rule_id].extend(claim.evidence_ids)
    dependency = 0
    for rule_id, ids in extras.items():
        support_bundles = set(graph.bundles_of(diagnoses.get(rule_id, [])))
        if ids and any(graph.bundle_of(i) in support_bundles for i in ids
                       if i in graph.observations):
            dependency += 1
    return {
        "propositions": propositions_summary(propositions),
        "layer_support": layer_rows,
        "layer_support_single_source": sum(1 for row in layer_rows if row["bundles"] == 1),
        "diagnosis": diagnosis_rows,
        "diagnosis_single_source": sum(1 for row in diagnosis_rows if row["bundles"] == 1),
        "corroboration": corroboration,
        "corroboration_inside_diagnosis_bundle": dependency,
        "top_single_source": [p.to_dict() for p in propositions if p.single_source][:10],
    }


def build(artifact, sample_path=None, policy=None):
    observations, vocabulary = observations_from_artifact(artifact)
    topology = None
    path = sample_path or (artifact.get("sample_metadata") or {}).get("sample_path_absolute") \
        or (artifact.get("sample_metadata") or {}).get("sample_path")
    if path and Path(path).is_dir():
        services = service_graph_from_sample(path, vocabulary)
        if services.size:
            topology = TopologyIndex(services)
    associator = Associator(policy or Policy(), topology=topology)
    associator.ingest(observations)
    associator.resolve()
    graph = associator.graph
    claims, unknown = claims_from_artifact(artifact, graph)
    propositions = assess(graph, claims)
    return {"associator": associator, "graph": graph, "claims": claims,
            "unknown_citations": unknown, "propositions": propositions,
            "vocabulary": vocabulary}


def audit_artifact(path, policy=None):
    import json
    path = Path(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    built = build(artifact, policy=policy)
    graph = built["graph"]
    evaluation = artifact.get("evaluation") or {}
    report = {
        "artifact": str(path),
        "sample": path.stem,
        "case": evaluation.get("case"),
        "truth": {"layers": evaluation.get("true_layers"), "types": evaluation.get("true_types")},
        "diagnosis": {"layer_order": (artifact.get("diagnosis") or {}).get("layer_order"),
                      "fault_order": (artifact.get("diagnosis") or {}).get("fault_order")},
        "observations": len(graph.observations),
        "graph": graph.summary(),
        "association": dict(built["associator"].stats),
        "unknown_citations": len(built["unknown_citations"]),
        "rq1": rq1(graph),
        "rq2": rq2(graph, built["claims"], built["propositions"]),
        "needs": query_needs(built["propositions"])[:10],
    }
    return report


def aggregate(reports):
    totals = Counter()
    gold = Counter()
    for report in reports:
        totals["artifacts"] += 1
        totals["observations"] += report["observations"]
        totals["merged_duplicates"] += report["graph"]["merged_duplicates"]
        totals["dependent_edges"] += report["graph"]["edges"]["DEPENDENT"]
        totals["unknown_pairs"] += report["graph"]["unknown_total"]
        totals["independent_pairs"] += report["graph"]["edges"]["INDEPENDENT"]
        totals["citations"] += report["rq2"]["propositions"]["citations"]
        totals["distinct_observations"] += report["rq2"]["propositions"]["distinct_observations"]
        totals["bundles"] += report["rq2"]["propositions"]["bundles"]
        totals["single_source_support"] += report["rq2"]["propositions"]["single_source_support"]
        for key in ("gold_pairs", "decided", "correct", "partial", "errors", "abstained"):
            gold[key] += report["rq1"][key]
    decided = gold["decided"]
    return {
        "artifacts": totals["artifacts"],
        "observations": totals["observations"],
        "merged_duplicates": totals["merged_duplicates"],
        "duplicate_rate": round(totals["merged_duplicates"] / totals["observations"], 4)
        if totals["observations"] else 0.0,
        "dependent_edges": totals["dependent_edges"],
        "independent_pairs": totals["independent_pairs"],
        "unknown_pairs": totals["unknown_pairs"],
        "unknown_share": round(totals["unknown_pairs"] / max(1, totals["unknown_pairs"] +
                                                            totals["dependent_edges"] +
                                                            totals["independent_pairs"]), 4),
        "rq1": {**{k: gold[k] for k in ("gold_pairs", "decided", "correct", "partial",
                                        "errors", "abstained")},
                "precision": round((gold["correct"] + 0.5 * gold["partial"]) / decided, 4)
                if decided else 0.0,
                "coverage": round(decided / gold["gold_pairs"], 4) if gold["gold_pairs"] else 0.0},
        "rq2": {"citations": totals["citations"],
                "distinct_observations": totals["distinct_observations"],
                "bundles": totals["bundles"],
                "single_source_support": totals["single_source_support"],
                "citation_to_bundle_ratio": round(totals["citations"] / totals["bundles"], 3)
                if totals["bundles"] else 0.0},
    }
