"""Semantic evidence view pushed to the model every round.

The graph already knows what is duplicated, what shares an execution chain and
what is independent.  Counts of those relations are not actionable for a
language model; records, provenance and relations are.

Everything here is deterministic:

* rule text comes from the caller's registry, never from this package;
* a record summary is an excerpt of the observation text (fallback: the masked
  template, then the origin id), plus status/exception when present;
* ``same`` / ``dependent`` / ``independent`` labels come from the existing
  union-find queries and the stored INDEPENDENT edges;
* the two sentences ``graph_reading`` and ``open`` are generated from those
  facts, never by an LLM call.

The view reports what is missing; the model chooses the tool from its
description.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .extract import producers
from .schema import Relation

DEFAULT_MAX_PROPOSITIONS = 12
DEFAULT_MAX_RECORDS = 6
DEFAULT_MAX_INDEPENDENT = 2
DEFAULT_SUMMARY_CHARS = 1000
DEFAULT_MAX_BUNDLES = 10
DEFAULT_MAX_SIGNALS = 5
DEFAULT_MAX_PRODUCERS = 5
EVIDENCE_ROLES = ("support", "related", "independent")

# ------------------------------------------------------------------ summaries
def _first_line(text):
    for line in (text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line
    return ""


def _excerpt(observation, limit):
    """One deterministic line: text excerpt, else masked template, else origin."""
    text = _first_line(observation.text)
    if not text:
        text = " ".join((observation.features.template or "").split())
    if not text:
        text = observation.origin or observation.id
    features = observation.features
    extras = []
    if features.status is not None:
        extras.append(f"status={features.status}")
    if features.exception:
        extras.append(features.exception)
    missing = [extra for extra in extras if extra not in text]
    if missing:
        text = text + " [" + ", ".join(missing) + "]"
    if len(text) > limit:
        text = text[:limit - 1].rstrip() + "…"
    return text


def summary(observation, limit=DEFAULT_SUMMARY_CHARS):
    """Public one-line summary used by the transcript projection."""
    return _excerpt(observation, limit)


def _stamp_label(value):
    try:
        return datetime.fromtimestamp(value, timezone.utc).strftime("%m-%d %H:%M:%S")
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _time_label(observation):
    span = observation.features.span
    return _stamp_label(span[0]) if span else None


def _time_span_label(observations):
    stamps = [row.features.span for row in observations if row.features.span]
    if not stamps:
        return None
    start = _stamp_label(min(span[0] for span in stamps))
    end = _stamp_label(max(span[1] for span in stamps))
    if not start:
        return None
    return start if end in (None, start) else f"{start}–{end}"


def producer_label(observation):
    """Who emitted the record, or ``None`` when provenance cannot say.

    Only a ``<pod>_<container>.log`` reference or a provenance-derived entity
    names a producer.  Data files such as ``call_chains.json`` or
    ``svc_metric.csv`` are sources, not producers.
    """
    names = sorted(producers(observation))
    if names:
        return names[0]
    anchor = observation.anchor
    if anchor is not None and anchor.file.endswith(".log"):
        return anchor.file.rsplit("/", 1)[-1]
    return None


def _record(observation, relation, bundle, summary_chars, copies=1):
    row = {"id": observation.id, "rel": relation, "bundle": bundle,
           "time": _time_label(observation),
           "summary": _excerpt(observation, summary_chars)}
    producer = producer_label(observation)
    if producer:
        row["producer"] = producer
    if copies > 1:
        row["copies"] = copies
    return row


# -------------------------------------------------------------------- support
def support_representatives(proposition, graph):
    """One observation per SAME class, in citation order.

    Returns ``(items, duplicates)`` where each item is
    ``(observation, class_root, bundle_root)``.
    """
    seen, items, duplicates = set(), [], 0
    for evidence_id in proposition.evidence_ids:
        observation = graph.observations.get(evidence_id)
        if observation is None:
            continue
        class_root = graph.class_of(evidence_id)
        if class_root in seen:
            duplicates += 1
            continue
        seen.add(class_root)
        items.append((observation, class_root, graph.bundle_of(evidence_id)))
    return items, duplicates


def relation_label(class_root, bundle_root, class_sizes, bundle_sizes):
    if class_sizes.get(class_root, 1) > 1:
        return "same"
    if bundle_sizes.get(bundle_root, 1) > 1:
        return "dependent"
    return "single"


def _support_records(proposition, graph, class_sizes, bundle_sizes, bundle_labels,
                     max_records, summary_chars):
    """Cited records, one per SAME class, ordered along the chain."""
    items, duplicates = support_representatives(proposition, graph)
    rows = []
    for observation, class_root, bundle_root in items:
        start = observation.features.span[0] if observation.features.span else float("inf")
        rows.append((start, observation, class_root, bundle_root))
    rows.sort(key=lambda item: (item[0], item[1].id))
    shown = rows[:max_records]
    records = [_record(observation,
                       relation_label(class_root, bundle_root, class_sizes, bundle_sizes),
                       bundle_labels.get(bundle_root), summary_chars,
                       class_sizes.get(class_root, 1))
               for _, observation, class_root, bundle_root in shown]
    return records, duplicates, max(0, len(rows) - len(shown))


def independent_adjacency(graph):
    """INDEPENDENT edges as an adjacency map, built once per view."""
    adjacency = defaultdict(list)
    for edge in graph.edges.values():
        if edge.relation != Relation.INDEPENDENT:
            continue
        adjacency[edge.a].append((edge.b, edge.confidence))
        adjacency[edge.b].append((edge.a, edge.confidence))
    return adjacency


def _text_hints(observation):
    """Semantic anchors for relevance: text entities plus the masked template."""
    hints = {entity for entity, source in observation.features.entity_sources
             if source == "text"}
    template = observation.features.template
    if template:
        hints.add("template:" + template)
    return hints


def independent_candidates(proposition, graph, adjacency):
    """INDEPENDENT neighbours that share a semantic anchor with the support.

    An INDEPENDENT edge means "a different producer, not the same execution";
    it does not mean "about the same thing".  Without a shared text entity or
    template the neighbour would be an unrelated record, so it is omitted.
    Returns ``(confidence, observation)`` ordered by confidence then id.
    """
    support_ids = set(proposition.evidence_ids)
    support_hints = set()
    for evidence_id in support_ids:
        observation = graph.observations.get(evidence_id)
        if observation is not None:
            support_hints |= _text_hints(observation)
    if not support_hints:
        return []
    found = {}
    for evidence_id in support_ids:
        for other, confidence in adjacency.get(evidence_id, ()):
            if other in support_ids or other in found:
                continue
            observation = graph.observations.get(other)
            if observation is None or not (_text_hints(observation) & support_hints):
                continue
            found[other] = (confidence, observation)
    return sorted(found.values(), key=lambda item: (-item[0], item[1].id))


def _independent_records(proposition, graph, adjacency, bundle_labels,
                         max_independent, summary_chars):
    order = independent_candidates(proposition, graph, adjacency)
    return [_record(observation, "independent",
                    bundle_labels.get(graph.bundle_of(observation.id)), summary_chars)
            for _, observation in order[:max_independent]]


# ------------------------------------------------------------------- overview
def _overview(graph, bundle_sizes, bundle_labels, max_bundles, max_signals,
              max_producers, summary_chars, include_ids=True, inline_samples=True):
    """Summarize the graph, with inline samples or evidence table references."""
    top_roots = sorted(bundle_sizes,
                       key=lambda root: (-bundle_sizes[root], root))[:max_bundles]
    top_set = set(top_roots)
    members = defaultdict(list)
    errors = {}
    producer_counts = Counter()
    for observation in graph.observations.values():
        root = graph.bundle_of(observation.id)
        if root in top_set:
            members[root].append(observation)
        producer = producer_label(observation)
        if producer:
            producer_counts[producer] += 1
        status = observation.features.status
        exception = observation.features.exception
        if (status is not None and status >= 400) or exception:
            source = producer or observation.tool or "unknown"
            key = (source, status, exception)
            entry = errors.get(key)
            if entry is None:
                entry = {"source": source, "status": status, "exception": exception,
                         "records": 0,
                         "time": _time_label(observation)}
                if inline_samples:
                    entry["sample"] = _excerpt(observation, summary_chars)
                if include_ids:
                    entry["id"] = observation.id
                errors[key] = entry
            entry["records"] += 1

    bundles = []
    for root in top_roots:
        rows = members.get(root) or []
        if not rows:
            continue
        rows.sort(key=lambda row: (row.features.span[0] if row.features.span else float("inf"),
                                   row.id))
        # Missing provenance is valid (e.g. a graph record joined to a log).
        # Keep its evidence in the bundle, but only sort known producer names.
        producer_names = {producer_label(row) for row in rows}
        producer_names.discard(None)
        bundle_row = {"bundle": bundle_labels.get(root),
                      "size": bundle_sizes[root],
                      "producers": sorted(producer_names)[:4],
                      "time": _time_span_label(rows)}
        if inline_samples:
            bundle_row["sample"] = _excerpt(rows[0], summary_chars)
        if include_ids:
            bundle_row["id"] = rows[0].id
        bundles.append(bundle_row)
    signals = sorted(errors.values(),
                     key=lambda row: (-row["records"], row["source"]))[:max_signals]
    producers_rows = [{"producer": name, "records": count}
                      for name, count in producer_counts.most_common(max_producers)]
    return {"largest_bundles": bundles, "error_signals": signals,
            "top_producers": producers_rows}


# ------------------------------------------------------------------- sentences
def _reading(proposition, support, duplicates, independent):
    if not support:
        return "该规则尚无已引用记录。"
    kinds = Counter(row["rel"] for row in support)
    parts = [f"共 {len(proposition.evidence_ids)} 条引用，折叠为 {proposition.bundles} 个证据包"]
    if duplicates:
        parts.append(f"{duplicates} 条是重复引用")
    if kinds.get("dependent"):
        parts.append(f"{kinds['dependent']} 条是同链上的不同记录")
    if kinds.get("single"):
        parts.append(f"{kinds['single']} 条是单独记录")
    if independent:
        parts.append(f"另有 {len(independent)} 条独立来源观测")
    else:
        parts.append("没有独立来源观测")
    return "；".join(parts) + "。"


def _open(proposition):
    gaps = set(proposition.gaps)
    if proposition.state == "unresolved":
        if "no_evidence" in gaps:
            return "尚无支持证据：需要能支持或反驳该规则的观测。"
        if "refutation_depends_on_support" in gaps:
            return "反事实否决的对照与支持证据同包，否决无效；需要独立来源的对照。"
        if "single_bundle" in gaps or proposition.bundles <= 1:
            return "支持只来自一个证据包或一个 producer；需要另一个来源的观测。"
        return "证据仍不能区分竞争规则；需要能区分它们的对照。"
    if proposition.state == "supported":
        # One bundle is enough for support.  Whether the layer is closed is
        # decided by its CF status in ``_action_open``, not by this rule alone.
        return "已支持（一个证据包即可成立）；无需继续为该规则取证，转向 action=investigate 的规则。"
    if proposition.state == "refuted":
        return "已被反事实否决；无需继续取证。"
    return ""


def _cf_rule(rule):
    return "-CF" in rule


def _control_rule(rule):
    return "-C" in rule and "-CF" not in rule


def _action_open(proposition, cf_status, control_status, cf_missing, vetoed,
                 is_cf, is_control, competing, rule_text):
    """One layer-level decision applied to every rule of that layer.

    Support is necessary but not sufficient: a supported rule stays open while
    its necessity signature (CF) is unknown.  A valid CF veto closes the layer.
    C controls only matter when several supported layers compete.
    """
    if vetoed:
        text = "；".join(_claim_text(rule, rule_text) for rule in vetoed)
        return "closed", f"反事实否决：{text}"
    if cf_missing:
        text = "；".join(_claim_text(rule, rule_text) for rule in cf_missing)
        return "investigate", f"必要性签名未判定：{text}"
    if is_cf:
        if proposition.state == "supported":
            return "closed", "必要性签名已出现；不否决该层。"
        return "investigate", _open(proposition)
    if is_control:
        missing = [rule for rule, status in control_status.items() if status == "missing"]
        if competing and missing:
            text = "；".join(_claim_text(rule, rule_text) for rule in missing)
            return "investigate", f"多个层有支持且未被否决；{text} 用于区分排序。"
        if competing:
            return "closed", "已有对照；可用于多个候选的排序。"
        return "closed", "只有一个支持层；对照仅用于排序，无需补证。"
    if proposition.state == "unresolved":
        return "investigate", _open(proposition)
    return "closed", _open(proposition)


def _claim_text(rule, rule_text):
    """Registry values may be plain strings or ``{description: ...}`` dicts."""
    value = rule_text.get(rule)
    if isinstance(value, dict):
        return value.get("description") or rule
    if value:
        return value
    return rule.replace(":", " ")


def evaluate_propositions(propositions, rule_text=None, cf_rules=None, control_rules=None):
    """Rank propositions by action, CF gap, unresolved state and bundle count."""
    rule_text = rule_text or {}
    cf_rules = cf_rules or {}
    control_rules = control_rules or {}

    support_layers = set()
    for proposition in propositions:
        if (proposition.state == "supported" and not proposition.cf_refuted
                and not _cf_rule(proposition.rule)
                and not _control_rule(proposition.rule)):
            support_layers.add(proposition.layer)
    competing = len(support_layers) >= 2

    evaluated = []
    for proposition in propositions:
        layer = proposition.layer
        expected_cf = cf_rules.get(layer) or tuple(proposition.cf)
        cf_status = {rule: proposition.cf.get(rule, "missing") for rule in expected_cf}
        expected_controls = control_rules.get(layer) or tuple(proposition.controls)
        control_status = {rule: ("present" if rule in proposition.controls else "missing")
                          for rule in expected_controls}
        cf_missing = [rule for rule, status in cf_status.items()
                      if status in ("unknown", "missing")]
        vetoed = list(proposition.cf_refuted)
        action, open_text = _action_open(
            proposition, cf_status, control_status, cf_missing, vetoed,
            _cf_rule(proposition.rule), _control_rule(proposition.rule),
            competing, rule_text)
        # CF gaps outrank generic "no evidence" rules: support is already there,
        # only the necessity signature is missing.
        rank = (0 if action == "investigate" else 1,
                0 if cf_missing else 1,
                0 if proposition.state == "unresolved" else 1,
                -proposition.bundles, -len(proposition.evidence_ids), proposition.rule)
        evaluated.append((rank, proposition, cf_status, control_status,
                          action, open_text))
    evaluated.sort(key=lambda item: item[0])
    return evaluated


def graph_state(propositions, graph, assessments=None, rule_text=None,
                layers=None, round_no=None,
                cf_rules=None, control_rules=None,
                max_propositions=DEFAULT_MAX_PROPOSITIONS,
                max_records=DEFAULT_MAX_RECORDS,
                max_independent=DEFAULT_MAX_INDEPENDENT,
                max_bundles=DEFAULT_MAX_BUNDLES,
                max_signals=DEFAULT_MAX_SIGNALS,
                max_producers=DEFAULT_MAX_PRODUCERS,
                summary_chars=DEFAULT_SUMMARY_CHARS,
                selection=None, include_ids=True):
    """Render open propositions as a bounded semantic digest.

    ``cf_rules``/``control_rules`` map a layer to its expected rule ids.  A
    supported rule stays ``investigate`` while the layer's CF is unknown; a
    valid CF veto closes it; C controls only matter when several layers compete.
    ``assessments``/``layers`` only report layers that have not been assessed.

    With ``selection``, proposition roles and overview samples refer to
    ``state["evidence"]`` by id. Selected evidence is listed in ranked order.
    ``None`` renders inline records with the fixed per-rule caps.
    """
    class_sizes = Counter()
    bundle_sizes = Counter()
    for observation_id in graph.observations:
        class_sizes[graph.class_of(observation_id)] += 1
        bundle_sizes[graph.bundle_of(observation_id)] += 1
    ordered_roots = sorted(bundle_sizes, key=lambda root: (-bundle_sizes[root], root))
    bundle_labels = {root: f"B{index}" for index, root in enumerate(ordered_roots, 1)}
    adjacency = independent_adjacency(graph) if selection is None else None
    rule_text = rule_text or {}

    evidence = {}

    def add_evidence(evidence_id):
        if evidence_id not in evidence:
            observation = graph.observations[evidence_id]
            class_root = graph.class_of(evidence_id)
            bundle_root = graph.bundle_of(evidence_id)
            record = _record(observation,
                             relation_label(class_root, bundle_root, class_sizes, bundle_sizes),
                             bundle_labels[bundle_root], summary_chars,
                             class_sizes[class_root])
            del record["id"]
            evidence[evidence_id] = record

    evaluated = evaluate_propositions(propositions, rule_text, cf_rules, control_rules)
    rows = []
    for _, proposition, cf_status, control_status, action, open_text in evaluated[:max_propositions]:
        if selection is None:
            support, duplicates, omitted = _support_records(
                proposition, graph, class_sizes, bundle_sizes, bundle_labels,
                max_records, summary_chars)
            independent = _independent_records(
                proposition, graph, adjacency, bundle_labels, max_independent, summary_chars)
            references = {"support": support, "independent": independent}
            reading = _reading(proposition, support, duplicates, independent)
        else:
            chosen = selection.by_rule.get(proposition.rule) or {}
            references = {
                role: [evidence_id for evidence_id in chosen.get(role, ())
                       if evidence_id in graph.observations]
                for role in EVIDENCE_ROLES
            }
            for ids in references.values():
                for evidence_id in ids:
                    add_evidence(evidence_id)
            items, duplicates = support_representatives(proposition, graph)
            visible = set(references["support"])
            omitted = sum(1 for observation, _, _ in items
                          if observation.id not in visible)
            support = [evidence[evidence_id] for evidence_id in references["support"]]
            reading = _reading(proposition, support, duplicates, references["independent"])
            if omitted and not support:
                reading = f"本轮未展示已引用记录（{omitted} 条去重引用）。"
        row = {"rule": proposition.rule,
               "claim": _claim_text(proposition.rule, rule_text),
               "layer": proposition.layer,
               "state": proposition.state,
               "action": action,
               "cf": cf_status,
               "controls": control_status,
               "bundles": proposition.bundles,
               **references,
               "graph_reading": reading,
               "open": open_text}
        if omitted:
            row["omitted"] = omitted
        rows.append(row)

    state = {"records": len(graph.observations),
             "evidence_bundles": len(bundle_sizes),
             "propositions": rows}
    # Before any proposition carries records, the model still needs the graph's
    # concrete content: largest bundles, error signals and main producers.
    if not any(row["support"] or row.get("related") for row in rows) and graph.observations:
        overview = _overview(graph, bundle_sizes, bundle_labels, max_bundles,
                             max_signals, max_producers, summary_chars,
                             include_ids=include_ids or selection is not None,
                             inline_samples=selection is None)
        if overview["largest_bundles"] or overview["error_signals"] or overview["top_producers"]:
            state["overview"] = overview
            if selection is not None:
                for group in ("largest_bundles", "error_signals"):
                    for sample in overview[group]:
                        add_evidence(sample["id"])
    if selection is not None:
        evidence_order = dict.fromkeys([*selection.active, *evidence])
        state["evidence"] = {evidence_id: evidence[evidence_id]
                             for evidence_id in evidence_order if evidence_id in evidence}
    if round_no is not None:
        state = {"round": round_no, **state}
    if assessments is not None and layers:
        unassessed = [layer for layer in layers
                      if (assessments.get(layer) or {}).get("state") in (None, "unsubmitted")]
        if unassessed:
            state["unassessed_layers"] = unassessed
    return state
