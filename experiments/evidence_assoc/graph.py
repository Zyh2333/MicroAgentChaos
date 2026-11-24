"""Dual-relation evidence graph with global consistency constraints.

Two relation families share one vertex set:

* ``E_T`` topology constraints  - deployment/zone/call structure, used by the
  structural dimension.  A topology edge is *potential* influence, not causality.
* ``E_E`` evidence relations    - SAME / DEPENDENT inferred between observations.

Local pairwise decisions are cheap and therefore wrong in predictable ways.  The
graph enforces global structure before an edge becomes usable:

1. SAME is a union-find equivalence; a merge is refused when the two observations
   carry provably different execution scopes (the "repeated identical line" trap).
2. DEPENDENT must stay acyclic - a backwards or cyclic claim is demoted.
3. Each vertex has a bounded number of DEPENDENT edges; excess edges are demoted
   to UNKNOWN.  Prefer a missing edge over a confident wrong one.
4. Independence is a query over the resulting structure, not a stored similarity:
   two observations are independent only when they are in different evidence
   bundles, share neither a source reference nor an identifier, were emitted by
   different producers (service/pod), and lie beyond the longest observed
   execution.
"""
from collections import Counter, defaultdict

from .extract import (SCOPE_KEYS, producer_separated, shared_identifiers, shared_refs,
                      scope_conflict)
from .relation import Policy
from .schema import Edge, Relation


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        self.parent.setdefault(item, item)
        self.rank.setdefault(item, 0)

    def find(self, item):
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class EvidenceGraph:
    def __init__(self, policy=None, degree_budget=6, unknown_sample=200, observer=None):
        self.policy = policy or Policy()
        self.degree_budget = degree_budget
        self.unknown_sample_cap = unknown_sample
        # Optional advisory sink for a differential log.  It observes decisions
        # but can never change them; ``None`` keeps the graph pure.
        self.observer = observer
        self.observations = {}
        self.same = UnionFind()          # SAME classes (identity / duplicate dedup)
        self.chain = UnionFind()         # SAME + DEPENDENT (evidence bundles)
        self.edges = {}
        self.dependent = defaultdict(set)  # keyed by current bundle root
        self.unknown_reasons = Counter()
        self.unknown_samples = []
        self.demotions = Counter()
        self.same_basis = Counter()
        # G2: written edges are retractable.  ``support`` is the E_S ledger
        # (proposition -> bundle root -> cited observations); ``retracted`` keeps
        # the audit trail of withdrawn evidence edges.
        self.support = {}
        self.retracted = []
        self.retracted_support = []
        # Observed execution duration for this corpus; set by the associator.
        # Independence requires a gap beyond it.
        self.corpus_span = None
        self.corpus = None
        self.evaluated_pairs = 0
        self.no_anchor_pairs = 0

    # ---------------------------------------------------------------- events
    def observe(self, callback):
        """Attach a differential event sink for graph evolution logging."""
        self.observer = callback

    def _observe(self, event):
        if self.observer is not None:
            self.observer(event)

    def _edge_event(self, op, relation, a, b, verdict, reason):
        return {"type": "edge", "op": op, "relation": relation, "a": a, "b": b,
                "confidence": round(verdict.confidence, 4),
                "basis": list(verdict.basis), "reason": reason}

    # ---------------------------------------------------------------- ingest
    def add_observation(self, observation):
        self.observations[observation.id] = observation
        self.same.add(observation.id)
        self.chain.add(observation.id)

    def _class_union(self, a, b):
        """Merge two SAME classes; rekey the dependent adjacency onto the new root."""
        ra, rb = self.same.find(a), self.same.find(b)
        if ra == rb:
            self.chain.union(a, b)
            return ra
        root = self.same.union(a, b)
        absorbed = rb if root == ra else ra
        moved = self.dependent.pop(absorbed, None)
        if moved:
            self.dependent[root].update(moved)
        for targets in self.dependent.values():
            if absorbed in targets:
                targets.discard(absorbed)
                targets.add(root)
        self.chain.union(a, b)
        return root

    def _degree(self, class_root):
        return len(self.dependent.get(class_root, ()))

    def _reaches(self, source, target):
        """Reachability over DEPENDENT edges between SAME classes."""
        start, goal = self.same.find(source), self.same.find(target)
        if start == goal:
            return True
        seen, stack = {start}, [start]
        while stack:
            node = stack.pop()
            for neighbor in self.dependent.get(node, ()):
                if neighbor == goal:
                    return True
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return False

    def add(self, verdict):
        a, b, relation = verdict.a, verdict.b, verdict.relation
        if relation == Relation.SAME:
            if self.same.find(a) == self.same.find(b):
                return "duplicate"
            if scope_conflict(self.observations[a], self.observations[b]):
                self.demotions["same_scope_conflict"] += 1
                return self._unknown(verdict, "same_scope_conflict")
            self._class_union(a, b)
            self.same_basis[verdict.reason] += 1
            self.edges[("same",) + tuple(sorted((a, b)))] = Edge(
                a, b, relation, verdict.confidence, verdict.basis, verdict.detail or {})
            self._observe(self._edge_event("add", "SAME", a, b, verdict, verdict.reason))
            return "merged"

        if relation == Relation.DEPENDENT:
            source, target = (a, b) if verdict.order <= 0 else (b, a)
            ra, rb = self.same.find(source), self.same.find(target)
            if ra == rb:
                self.demotions["self_dependence"] += 1
                self._observe(self._edge_event("demote", "DEPENDENT", source, target,
                                               verdict, "self_dependence"))
                return "demoted"
            if self._reaches(rb, ra):
                self.demotions["cycle"] += 1
                return self._unknown(verdict, "temporal_cycle")
            if self._degree(ra) >= self.degree_budget or self._degree(rb) >= self.degree_budget:
                self.demotions["degree_budget"] += 1
                return self._unknown(verdict, "degree_budget")
            self.edges[("dependent", source, target)] = Edge(
                source, target, relation, verdict.confidence, verdict.basis, verdict.detail or {})
            self.dependent[ra].add(rb)
            self.chain.union(source, target)
            self._observe(self._edge_event("add", "DEPENDENT", source, target,
                                           verdict, verdict.reason))
            return "added"

        if relation == Relation.INDEPENDENT:
            if self.chain.find(a) == self.chain.find(b):
                self.demotions["independent_conflict"] += 1
                self._observe(self._edge_event("demote", "INDEPENDENT", a, b,
                                               verdict, "independent_conflict"))
                return "demoted"
            self.edges[("independent",) + tuple(sorted((a, b)))] = Edge(
                a, b, relation, verdict.confidence, verdict.basis, verdict.detail or {})
            self._observe(self._edge_event("add", "INDEPENDENT", a, b, verdict, verdict.reason))
            return "added"

        return self._unknown(verdict, verdict.reason)

    def _unknown(self, verdict, reason):
        self.unknown_reasons[reason] += 1
        if len(self.unknown_samples) < self.unknown_sample_cap:
            self.unknown_samples.append({**verdict.to_dict(), "reason": reason})
        self._observe({"type": "unknown", "relation": verdict.relation.value,
                       "a": verdict.a, "b": verdict.b,
                       "confidence": round(verdict.confidence, 4), "reason": reason})
        return "unknown"

    # --------------------------------------------------------------- revision
    @staticmethod
    def edge_key(relation, a, b):
        if relation == Relation.DEPENDENT:
            return ("dependent", a, b)
        return (relation.value.lower(),) + tuple(sorted((a, b)))

    def retract_edge(self, key, reason):
        """Withdraw one written evidence edge and rebuild connectivity."""
        edge = self.edges.pop(key, None)
        if edge is None:
            return False
        self.retracted.append({**edge.to_dict(), "retracted_by": reason})
        self._observe({"type": "edge", "op": "retract", "relation": edge.relation.value,
                       "a": edge.a, "b": edge.b, "reason": reason})
        self.rebuild()
        return True

    def rebuild(self):
        """Recompute connectivity from the surviving edges.

        Only connectivity is rebuilt: the surviving set was admitted under the
        cycle and degree constraints, and removing edges can only relax them.
        """
        self.same, self.chain = UnionFind(), UnionFind()
        self.dependent = defaultdict(set)
        for observation_id in self.observations:
            self.same.add(observation_id)
            self.chain.add(observation_id)
        same_edges = sorted((e for e in self.edges.values() if e.relation == Relation.SAME),
                            key=lambda edge: -edge.confidence)
        for edge in same_edges:
            self.same.union(edge.a, edge.b)
            self.chain.union(edge.a, edge.b)
        dependent = sorted((e for e in self.edges.values()
                            if e.relation == Relation.DEPENDENT),
                           key=lambda edge: -edge.confidence)
        for edge in dependent:
            ra, rb = self.same.find(edge.a), self.same.find(edge.b)
            if ra == rb:
                continue
            self.dependent[ra].add(rb)
            self.chain.union(edge.a, edge.b)

    def reconcile(self, involved=None):
        """Withdraw SAME edges that merged observations with conflicting scopes.

        Called after new merges: a later record can prove that an earlier merge
        joined two different execution scopes.  The weakest edge on the class is
        withdrawn and the component rebuilt, so classes can split.
        """
        involved = set(involved or ())
        withdrawn = 0
        while True:
            target = None
            for members in self.classes().values():
                if len(members) < 2 or (involved and not involved.intersection(members)):
                    continue
                observations = [self.observations[m] for m in members]
                for i in range(len(observations)):
                    for j in range(i + 1, len(observations)):
                        if scope_conflict(observations[i], observations[j]):
                            target = members
                            break
                    if target:
                        break
                if target:
                    break
            if target is None:
                break
            member_set = set(target)
            candidates = [key for key, edge in self.edges.items()
                          if edge.relation == Relation.SAME
                          and edge.a in member_set and edge.b in member_set]
            if not candidates:
                break
            weakest = min(candidates, key=lambda key: self.edges[key].confidence)
            self.retract_edge(weakest, "scope_conflict")
            withdrawn += 1
            if withdrawn >= 100:
                break
        return withdrawn

    # ---------------------------------------------------------------- support
    def claim(self, proposition, evidence_ids):
        """Write or revise the E_S edges of one proposition; return the diff."""
        new = {}
        for evidence_id in evidence_ids or ():
            if evidence_id not in self.observations:
                continue
            new.setdefault(self.bundle_of(evidence_id), []).append(evidence_id)
        previous = self.support.get(proposition, {})
        removed = [root for root in previous if root not in new]
        added = [root for root in new if root not in previous]
        self.support[proposition] = new
        if removed:
            self.retracted_support.append({"proposition": proposition, "bundles": removed,
                                           "reason": "support_revised"})
        self._observe({"type": "claim", "proposition": proposition,
                       "added": len(added), "removed": len(removed), "in_degree": len(new),
                       "bundles_total": len(new), "added_bundles": sorted(added)[:16],
                       "removed_bundles": sorted(removed)[:16]})
        return {"added": len(added), "removed": len(removed), "in_degree": len(new)}

    def retract_claim(self, proposition, reason="proposition_refuted"):
        """Remove every E_S edge of a proposition (refutation / exclusion)."""
        previous = self.support.get(proposition)
        if not previous:
            return False
        # Keep an empty entry so in-degree reads 0 instead of falling back to the
        # citation count, which would count withdrawn support again.
        self.support[proposition] = {}
        self.retracted_support.append({"proposition": proposition,
                                       "bundles": list(previous), "reason": reason})
        self._observe({"type": "claim_retracted", "proposition": proposition,
                       "reason": reason, "bundles_total": len(previous)})
        return True

    def execution_span(self):
        """Longest observed duration of one execution, or ``None``.

        Groups observations by shared execution identifiers and takes the widest
        observed span.  This is the corpus-defined answer to "how long can one
        execution last", used instead of a fixed time window.
        """
        groups = defaultdict(list)
        for observation in self.observations.values():
            identifiers = observation.features.identifier_map
            for key in SCOPE_KEYS:
                value = identifiers.get(key)
                if value:
                    groups[(key, value)].append(observation)
        spans = []
        for members in groups.values():
            stamps = [o.features.span for o in members if o.features.span]
            if len(stamps) >= 2:
                spans.append(max(s[1] for s in stamps) - min(s[0] for s in stamps))
        return max(spans) if spans else None

    def support_in_degree(self, proposition, fallback_ids=()):
        if proposition in self.support:
            return len(self.support[proposition])
        return self.bundle_count(fallback_ids)

    def support_diversity(self, proposition, fallback_ids=()):
        """Distinct source signatures among supporting bundles.

        Five bundles that all cite one log file are one perspective, not five;
        this is the quantity D(p) uses to separate "many in-edges" from
        "discriminating in-edges".
        """
        groups = self.support.get(proposition)
        if not groups:
            groups = self.bundles_of(fallback_ids)
        signatures = set()
        for members in groups.values():
            files = frozenset(self.observations[m].anchor.file for m in members
                              if m in self.observations and self.observations[m].anchor)
            signatures.add(files)
        signatures.discard(frozenset())
        return len(signatures)

    # ------------------------------------------------------------------ view
    def class_of(self, observation_id):
        return self.same.find(observation_id)

    def class_degree(self, observation_id):
        """Number of outgoing DEPENDENT neighbors of the observation's SAME class."""
        if observation_id not in self.observations:
            return 0
        return len(self.dependent.get(self.same.find(observation_id), ()))

    def bundle_of(self, observation_id):
        return self.chain.find(observation_id)

    def independent(self, a, b):
        """The single independence predicate.

        Used by the INDEPENDENT edge, the counterfactual control guard and the
        planner's corroboration check.  A pair is independent when it is in
        different evidence bundles, shares no reference or identifier, was
        emitted by a different producer (service/pod), and lies beyond the
        longest execution observed in this corpus.
        """
        if a not in self.observations or b not in self.observations:
            return None
        if self.bundle_of(a) == self.bundle_of(b):
            return False
        left, right = self.observations[a], self.observations[b]
        if shared_refs(left, right) or shared_identifiers(left, right):
            return False
        if not producer_separated(left, right):
            return False
        if self.corpus is not None:
            from .compat import semantic as semantic_score
            score = semantic_score(left, right, self.corpus)
            if score.value is not None and score.value >= self.policy.link_semantic:
                return False
        span = self.corpus_span
        if not span:
            return False
        left_span, right_span = left.features.span, right.features.span
        if left_span is None or right_span is None:
            return False
        gap = max(0.0, max(left_span[0], right_span[0]) - min(left_span[1], right_span[1]))
        return gap > span

    def bundle_count(self, evidence_ids):
        return len({self.bundle_of(i) for i in evidence_ids if i in self.observations})

    def bundles_of(self, evidence_ids):
        groups = defaultdict(list)
        for evidence_id in evidence_ids:
            if evidence_id in self.observations:
                groups[self.bundle_of(evidence_id)].append(evidence_id)
        return dict(groups)

    def neighbors(self, observation_id):
        rows = []
        for edge in self.edges.values():
            if observation_id in (edge.a, edge.b):
                rows.append(edge)
        return sorted(rows, key=lambda e: -e.confidence)

    def classes(self):
        groups = defaultdict(list)
        for observation_id in self.observations:
            groups[self.class_of(observation_id)].append(observation_id)
        return {root: sorted(members) for root, members in groups.items()}

    def components(self):
        groups = defaultdict(list)
        for observation_id in self.observations:
            groups[self.bundle_of(observation_id)].append(observation_id)
        ordered = sorted(groups.values(), key=lambda m: (-len(m), sorted(m)[0]))
        return [{"root": self.bundle_of(members[0]), "size": len(members),
                 "members": sorted(members)} for members in ordered]

    def observation_stats(self):
        """Entity/provenance coverage, for auditing and ablation."""
        stats = Counter()
        for observation in self.observations.values():
            features = observation.features
            if features.entities:
                stats["with_entities"] += 1
            sources = dict(features.entity_sources)
            if "text" in sources.values():
                stats["text_entities"] += 1
            if "ref" in sources.values():
                stats["ref_entities"] += 1
            if not features.entities and not features.identifiers:
                stats["without_structure"] += 1
            if features.has_time:
                stats["with_time"] += 1
        return {"observations": len(self.observations), **stats}

    def same_class_stats(self):
        """Sanity signals for over-merging: cross-file classes and wide time spans."""
        cross_file = wide = 0
        sizes, spans = [], []
        for members in self.classes().values():
            files = {self.observations[m].anchor.file for m in members
                     if self.observations[m].anchor is not None}
            if len(files) > 1:
                cross_file += 1
            stamps = [self.observations[m].features.span for m in members
                      if self.observations[m].features.span]
            if len(stamps) > 1:
                span = max(s[1] for s in stamps) - min(s[0] for s in stamps)
                spans.append(span)
                if span > 60:
                    wide += 1
            sizes.append(len(members))
        return {"cross_file_classes": cross_file, "wide_classes_over_60s": wide,
                "max_class_size": max(sizes) if sizes else 0,
                "mean_class_size": round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
                "median_class_time_span": round(sorted(spans)[len(spans) // 2], 3) if spans else None,
                "by_basis": dict(self.same_basis.most_common())}

    def summary(self):
        classes = self.classes()
        components = self.components()
        return {
            "observations": len(self.observations),
            "same_classes": len(classes),
            "merged_duplicates": len(self.observations) - len(classes),
            "same_class_quality": self.same_class_stats(),
            "observation_stats": self.observation_stats(),
            "support": {"propositions": len(self.support),
                        "support_edges": sum(len(bundles) for bundles in self.support.values()),
                        "retracted_edges": len(self.retracted),
                        "retracted_support": len(self.retracted_support)},
            "edges": {relation.value: sum(1 for edge in self.edges.values()
                                          if edge.relation == relation)
                      for relation in (Relation.SAME, Relation.DEPENDENT, Relation.INDEPENDENT)},
            "unknown": dict(self.unknown_reasons.most_common()),
            "unknown_total": sum(self.unknown_reasons.values()),
            "demotions": dict(self.demotions.most_common()),
            "evaluated_pairs": self.evaluated_pairs,
            "no_anchor_pairs": self.no_anchor_pairs,
            "evidence_bundles": len(components),
            "largest_bundle": components[0]["size"] if components else 0,
            "mean_bundle_size": round(sum(c["size"] for c in components) / len(components), 3)
            if components else 0.0,
            "degree_budget": self.degree_budget,
        }

    def to_dict(self, limit=200):
        return {"summary": self.summary(), "classes": list(self.classes().values())[:limit],
                "components": self.components()[:limit],
                "edges": [edge.to_dict() for edge in list(self.edges.values())[:limit]],
                "unknown_samples": self.unknown_samples[:50],
                "observations": [o.to_dict() for o in list(self.observations.values())[:limit]]}
