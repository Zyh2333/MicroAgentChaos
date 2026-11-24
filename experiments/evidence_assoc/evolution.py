"""Activity ledger for the combinatorial counterfactual context.

M2 scores joint evidence removals against the current open set
(``GER_RCA_full.md`` §3.1) and fills the context in lexicographic order
within a character budget (``§3.3``, greedy v1).  The ledger derives four
activity states from the rendered state:

* ``ACTIVE``   - selected into this round's state;
* ``DORMANT``  - eligible (open closure) but outside the budget;
* ``ARCHIVED`` - outside the open closure; the record stays in the graph;
* ``REACTIVATED`` - an archived record that re-enters the eligible set.

Selection is reproducible, including capped sampling, and uses no model calls.
"""
from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import combinations
from math import comb
from random import Random

from .propositions import assess
from .schema import Relation
from .view import (DEFAULT_MAX_PROPOSITIONS, EVIDENCE_ROLES, evaluate_propositions,
                   independent_adjacency, independent_candidates, producer_label,
                   support_representatives)

ACTIVE = "ACTIVE"
DORMANT = "DORMANT"
ARCHIVED = "ARCHIVED"
DEFAULT_COMBINATION_SIZE = 2
DEFAULT_MAX_COMBINATIONS = 20000


@dataclass
class Selection:
    """Contribution-ranked evidence chosen for one round."""
    order: list = field(default_factory=list)          # every eligible id, ranked
    active: list = field(default_factory=list)         # ids inside the budget
    by_rule: dict = field(default_factory=dict)        # rule -> evidence ids by role
    contributions: dict = field(default_factory=dict)  # id -> (dec, cov, str, dis)
    costs: dict = field(default_factory=dict)
    omitted: dict = field(default_factory=dict)        # rule -> omitted count by role
    spent: int = 0
    budget: int = 0
    counterfactual: dict = field(default_factory=dict)

    def summary(self):
        return {"selected": len(self.active), "eligible": len(self.order),
                "spent_chars": self.spent, "budget_chars": self.budget,
                "omitted": self.omitted, "counterfactual": self.counterfactual}


def _state_ids(state, graph):
    """Evidence ids rendered by one ``graph_state`` call."""
    found = set(state.get("evidence") or ())
    if "evidence" not in state:
        for row in state.get("propositions") or ():
            for role in EVIDENCE_ROLES:
                for record in row.get(role) or ():
                    found.add(record.get("id"))
    overview = state.get("overview") or {}
    for row in overview.get("largest_bundles") or ():
        found.add(row.get("id"))
    for row in overview.get("error_signals") or ():
        found.add(row.get("id"))
    found.discard(None)
    return {item for item in found if item in graph.observations}


def _decision_signature(propositions):
    states = {p.rule: p.state for p in propositions}
    vetoes = {rule for p in propositions for rule in p.cf_refuted}
    return states, vetoes


def _decision_test(propositions, graph):
    """Compare only affected layers and cache repeated removals of their citations."""
    by_layer = defaultdict(list)
    for p in propositions:
        by_layer[p.layer].extend(p.claims)
    baseline = {layer: _decision_signature(assess(graph, claims))
                for layer, claims in by_layer.items()}
    references = {layer: {i for claim in claims for i in claim.evidence_ids + claim.control_ids}
                  for layer, claims in by_layer.items()}

    @lru_cache(maxsize=4096)
    def layer_changed(layer, removed):
        filtered = [replace(claim, evidence_ids=[i for i in claim.evidence_ids if i not in removed],
                            control_ids=[i for i in claim.control_ids if i not in removed])
                    for claim in by_layer[layer]]
        states, vetoes = _decision_signature(assess(graph, filtered))
        old_states, old_vetoes = baseline[layer]
        return states != old_states or bool(old_vetoes - vetoes)

    def changed(removed):
        for layer, ids in references.items():
            affected = removed & ids
            if affected and layer_changed(layer, affected):
                return True
        return False

    return changed


def _coverage_test(propositions, graph, candidates, size):
    """Coverage falls when an entire cited bundle or producer bucket is removed."""
    lost_by = set()
    for p in propositions:
        groups = list(graph.bundles_of(p.evidence_ids).values())
        producers = defaultdict(list)
        for i in p.evidence_ids:
            observation = graph.observations.get(i)
            producer = producer_label(observation) if observation else None
            if producer:
                producers[producer].append(i)
        groups.extend(producers.values())
        for group in groups:
            ids = frozenset(group)
            if len(ids) <= size and ids <= candidates:
                lost_by.add(ids)

    return lambda removed: any(group <= removed for group in lost_by)


def _structure_test(graph, candidates):
    """Detect splits of surviving nodes anywhere in the SAME-contracted graph."""
    adjacency = defaultdict(set)
    for edge in graph.edges.values():
        if edge.relation != Relation.DEPENDENT:
            continue
        left, right = graph.class_of(edge.a), graph.class_of(edge.b)
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    class_bundles = {i: (graph.class_of(i), graph.bundle_of(i)) for i in candidates
                     if graph.class_of(i) in adjacency}

    @lru_cache(maxsize=4096)
    def bundle_split(removed_classes):
        # Every surviving piece of the original bundle touches a removed node.
        boundary = set().union(*(adjacency[node] for node in removed_classes))
        boundary.difference_update(removed_classes)
        if len(boundary) < 2:
            return False

        first = boundary.pop()
        seen = {first}
        stack = [first]
        while stack and boundary:
            for node in adjacency[stack.pop()]:
                if node not in removed_classes and node not in seen:
                    seen.add(node)
                    boundary.discard(node)
                    stack.append(node)
        return bool(boundary)

    def changed(removed):
        by_bundle = defaultdict(set)
        for i in removed:
            if i in class_bundles:
                node, bundle = class_bundles[i]
                by_bundle[bundle].add(node)
        return any(bundle_split(frozenset(nodes)) for nodes in by_bundle.values())

    return changed


def _removal_combinations(candidates, size, total, limit):
    if total <= limit:
        yield from combinations(candidates, size)
        return
    # Sample distinct groups without materializing the combinatorial search space.
    rng = Random(0)
    sampled = set()
    while len(sampled) < limit:
        group = tuple(sorted(rng.sample(candidates, size)))
        if group not in sampled:
            sampled.add(group)
            yield group


def _combination_scores(propositions, graph, candidates, size, limit):
    """Apply each rejection bit to every member, then leave ranking to the caller."""
    if size < 1 or limit < 1:
        raise ValueError("M2 combination size and limit must be positive")
    ids = sorted(candidates)
    size = min(size, len(ids))
    total = comb(len(ids), size) if size else 0
    scores = {i: [0, 0, 0] for i in ids}
    tested = 0
    rejected = dict.fromkeys(("dec", "cov", "str"), 0)
    if total:
        tests = {
            "dec": _decision_test(propositions, graph),
            "cov": _coverage_test(propositions, graph, candidates, size),
            "str": _structure_test(graph, candidates),
        }
        for group in _removal_combinations(ids, size, total, limit):
            removed = frozenset(group)
            tested += 1
            for index, (name, changed) in enumerate(tests.items()):
                if changed(removed):
                    rejected[name] += 1
                    for i in group:
                        scores[i][index] += 1
    stats = {"size": size, "total": total, "tested": tested, "rejected": rejected}
    return scores, stats


class ActivityLedger:
    def __init__(self):
        self.states = {}
        self.activity_log = []
        self.reactivation_log = []
        self.last_selection = None
        self._previous = {}
        self._fresh = set()
        self._eligible = []

    # ------------------------------------------------------------- ingest hook
    def on_new_observation(self, ids):
        """Record ids ingested by the latest tool call (reactivation candidates)."""
        self._fresh.update(item for item in (ids or ()) if isinstance(item, str))
        return []

    # ------------------------------------------------------------ contribution
    @staticmethod
    def _cost(observation):
        text = " ".join((observation.text or "").split())
        if not text:
            text = observation.features.template or observation.origin or observation.id
        return min(len(text), 1000) + 96

    @staticmethod
    def _delta_dis(v, hypotheses):
        for hypothesis in hypotheses or ():
            nodes = (hypothesis.get("involved_nodes") if isinstance(hypothesis, dict)
                     else getattr(hypothesis, "involved_nodes", None)) or ()
            if v in nodes:
                return 1
        return 0

    def _owners(self, propositions, graph):
        """Map each evidence id to its role for each candidate proposition."""
        adjacency = independent_adjacency(graph)
        owners = {}
        components = {component["root"]: component["members"]
                      for component in graph.components()}

        def claim_owner(observation_id, kind, index, rule):
            owners.setdefault(observation_id, {}).setdefault((index, rule), kind)

        for index, proposition in enumerate(propositions):
            items, _ = support_representatives(proposition, graph)
            support_classes = {class_root for _, class_root, _ in items}
            for observation, _, _ in items:
                claim_owner(observation.id, "support", index, proposition.rule)
            for bundle_root in dict.fromkeys(root for _, _, root in items):
                for member in components.get(bundle_root, ()):
                    if graph.class_of(member) not in support_classes:
                        claim_owner(member, "related", index, proposition.rule)
            for _, observation in independent_candidates(proposition, graph, adjacency):
                if graph.class_of(observation.id) not in support_classes:
                    claim_owner(observation.id, "independent", index, proposition.rule)
        return owners

    def eligible(self, propositions, graph):
        return list(self._owners(propositions, graph))

    # ---------------------------------------------------------------- ranking
    def select(self, propositions, graph, budget=12000, hypotheses=(), *,
               combination_size=DEFAULT_COMBINATION_SIZE,
               max_combinations=DEFAULT_MAX_COMBINATIONS):
        """Score removal combinations and fill the budget in lexicographic order."""
        budget = max(0, int(budget))
        owners = self._owners(propositions, graph)
        costs = {i: self._cost(graph.observations[i]) for i in owners}
        scores, stats = _combination_scores(propositions, graph, costs.keys(),
                                              combination_size, max_combinations)
        contributions = {i: (*score, self._delta_dis(i, hypotheses)) for i, score in scores.items()}

        def key(v):
            observation = graph.observations[v]
            start = observation.features.span[0] if observation.features.span else float("inf")
            index = min(i for i, _ in owners[v])
            return (-contributions[v][0], -contributions[v][1], -contributions[v][2],
                    index, start, v)

        order = sorted(contributions, key=key)
        by_rule = {p.rule: {role: [] for role in EVIDENCE_ROLES} for p in propositions}
        omitted = {p.rule: dict.fromkeys(EVIDENCE_ROLES, 0) for p in propositions}
        spent, active = 0, []
        for v in order:
            if spent + costs[v] > budget:
                for (_, rule), role in owners[v].items():
                    omitted[rule][role] += 1
                continue
            spent += costs[v]
            active.append(v)
            for (_, rule), role in owners[v].items():
                by_rule[rule][role].append(v)
        selection = Selection(order=order, active=active, by_rule=by_rule,
                              contributions=contributions, costs=costs,
                              omitted=omitted, spent=spent, budget=budget, counterfactual=stats)
        self.last_selection = selection
        self._eligible = list(order)
        return selection

    # ---------------------------------------------------------------- activity
    def refresh(self, state, graph, propositions=None, selection=None, round_no=None):
        """Derive the four states from the rendered state and log transitions."""
        active = _state_ids(state, graph)
        if selection is not None:
            eligible = list(selection.order)
        elif propositions is not None:
            eligible = self.eligible(propositions, graph)
        else:
            eligible = list(self._eligible)
        eligible = set(eligible) | active
        cited = {i for p in (propositions or ()) for i in p.evidence_ids}
        fresh_bundles = {graph.bundle_of(i) for i in self._fresh
                         if i in graph.observations}
        states, events = {}, []
        for observation_id in graph.observations:
            if observation_id in active:
                states[observation_id] = ACTIVE
            elif observation_id in eligible:
                states[observation_id] = DORMANT
            else:
                states[observation_id] = ARCHIVED
        for observation_id, current in states.items():
            if (self._previous.get(observation_id) == ARCHIVED
                    and current in (ACTIVE, DORMANT)):
                bundle = graph.bundle_of(observation_id)
                if fresh_bundles and bundle in fresh_bundles:
                    cause = "linked"
                elif observation_id in cited:
                    cause = "cited"
                else:
                    cause = "selected"
                events.append({"round": round_no, "id": observation_id,
                               "cause": cause, "to": current, "bundle": bundle})
        self._previous = states
        self.states = states
        self._fresh.clear()
        protected = (sum(1 for delta in selection.contributions.values() if delta[3])
                     if selection is not None else 0)
        summary = {
            "active": sum(1 for state in states.values() if state == ACTIVE),
            "dormant": sum(1 for state in states.values() if state == DORMANT),
            "archived": sum(1 for state in states.values() if state == ARCHIVED),
            "selected": len(active),
            "protected": protected,
            "budget_chars": getattr(selection, "budget", None),
            "spent_chars": getattr(selection, "spent", None),
            "reactivated": events[:16],
            "note": "本轮上下文只含选中记录；可用 read_evidence(id) 回读"}
        entry = {
            "round": round_no,
            "mode": "contribution" if selection is not None else "semantic",
            "active": summary["active"], "dormant": summary["dormant"],
            "archived": summary["archived"], "selected": summary["selected"],
            "protected": protected,
            "budget": summary["budget_chars"], "spent": summary["spent_chars"],
            "order": list(selection.order) if selection is not None else list(eligible),
            "contributions": ({v: list(delta) for v, delta in selection.contributions.items()}
                              if selection is not None else {}),
            "omitted": dict(selection.omitted) if selection is not None else {}}
        if selection is not None:
            entry["counterfactual"] = selection.counterfactual
        self.activity_log.append(entry)
        self.reactivation_log.extend(events)
        return summary

    def ranked_propositions(self, propositions, rule_text=None, cf_rules=None,
                            control_rules=None, max_propositions=DEFAULT_MAX_PROPOSITIONS):
        """The same display order ``view.graph_state`` uses, capped."""
        return [item[1] for item in evaluate_propositions(
            propositions, rule_text, cf_rules, control_rules)][:max_propositions]
