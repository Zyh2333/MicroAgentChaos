"""Read-only bridge between the running agent and the evidence association layer.

The agent keeps its R/CF/C protocol untouched.  This bridge only *observes*:
after every data tool call it ingests the newly registered evidence records,
recomputes relations, and exposes a compact view that the LLM may consult.

Nothing here can change a diagnosis; the graph is advisory, and every claim it
makes is traceable to evidence ids the agent already has.

An optional ``event_sink`` receives structured differential events (edge add /
demote / retract, claims, per-sync counts) for offline evolution analysis.  The
sink is observational: it cannot alter a decision.
"""
from collections import Counter

from experiments.evidence_assoc.adapters import (TopologyIndex, claims_from_diagnosis,
                                                 claims_from_layers, observations_from_evidence,
                                                 service_graph_from_sample)
from experiments.evidence_assoc.associate import Associator
from experiments.evidence_assoc.hypothesis import proposition_claims
from experiments.evidence_assoc.propositions import assess, query_needs
from experiments.evidence_assoc.relation import Policy

from ..utils.native_data import services


# Refusals that would have become an edge are logged individually; "not enough
# evidence yet" verdicts are aggregated per sync (they are both numerous and
# re-derived on every sync, so individual lines would dominate the log).
LOGGED_UNKNOWN = ("degree_budget", "temporal_cycle", "same_scope_conflict")


class EvidenceGraphBridge:
    def __init__(self, sample, policy=None, degree_budget=6, event_sink=None, context=None):
        self.sample = sample
        self.event_sink = event_sink
        self.context = context
        self.sync_index = 0
        self.tool = None
        self._pending_unknown = Counter()
        self.vocabulary = tuple(services(sample))
        self.associator = Associator(policy or Policy(), degree_budget=degree_budget,
                                     topology=self._topology())
        self.ingested = set()
        self.last_sync = 0
        self.last_fresh_ids = []
        if event_sink is not None:
            self.associator.graph.observe(self._graph_event)

    def _topology(self):
        try:
            graph = service_graph_from_sample(self.sample.path, self.vocabulary)
        except Exception:
            return None
        return TopologyIndex(graph) if graph.size else None

    def _stamp(self, event):
        """Attach the agent's current round/stage and the ingest context."""
        round_no, stage = self.context() if self.context is not None else (None, None)
        return {**event, "round": round_no, "stage": stage,
                "sync_index": self.sync_index, "tool": self.tool}

    def _graph_event(self, event):
        if self.event_sink is None:
            return
        if event.get("type") == "unknown" and event.get("reason") not in LOGGED_UNKNOWN:
            self._pending_unknown[event.get("reason")] += 1
            return
        self.event_sink(self._stamp(event))

    def begin_tool(self, name):
        """Attribute subsequent agent-side events to this tool call."""
        self.tool = name

    def record(self, event):
        """Emit one agent-side event (claim outcomes, layer support)."""
        if self.event_sink is not None:
            self.event_sink(self._stamp(dict(event)))

    def sync(self, evidence, tool=None):
        """Ingest evidence records registered since the last sync."""
        self.last_fresh_ids = []
        fresh = {}
        for evidence_id, fact in (evidence or {}).items():
            if evidence_id in self.ingested or not isinstance(fact, dict):
                continue
            if fact.get("kind") != "raw_observation" or not (fact.get("text") or "").strip():
                continue
            self.ingested.add(evidence_id)
            fresh[evidence_id] = fact
        if not fresh:
            return 0
        self.tool = tool
        self.sync_index += 1
        rows = observations_from_evidence(fresh, self.vocabulary)
        self.associator.ingest(rows)
        self.associator.resolve()
        self.last_sync = len(rows)
        self.last_fresh_ids = sorted(fresh)
        self._emit_sync(len(rows), fresh)
        return len(rows)

    def _emit_sync(self, added, fresh):
        """One summary line per ingest batch: the state the agent now sees."""
        if self.event_sink is None:
            return
        graph = self.associator.graph
        corpus = self.associator.corpus
        counts = Counter(edge.relation.value for edge in graph.edges.values())
        self.event_sink(self._stamp({
            "type": "sync", "added_observations": added,
            "evidence_ids": sorted(fresh)[:16],
            "observations": len(graph.observations),
            "edges": {relation: counts.get(relation, 0)
                      for relation in ("SAME", "DEPENDENT", "INDEPENDENT")},
            "bundles": len(graph.components()),
            "unknown_delta": dict(self._pending_unknown),
            "unknown_total": sum(graph.unknown_reasons.values()),
            "time_scale_seconds": round(corpus.time_scale, 3),
            "execution_span_seconds": (round(corpus.execution_span, 3)
                                       if corpus.execution_span else None)}))
        self._pending_unknown.clear()

    # --------------------------------------------------------------- views
    def propositions(self, layers=None, diagnosis=None):
        graph = self.associator.graph
        claims, _ = claims_from_layers(layers or {}, graph)
        extra, _ = claims_from_diagnosis(diagnosis, graph)
        if graph.hypothesis_memory is not None:
            extra.extend(proposition_claims(graph.hypothesis_memory.proposition_evidence))
        return assess(graph, claims + extra)

    def summary(self):
        return {"graph": self.associator.graph.summary(),
                "association": dict(self.associator.stats),
                "last_sync": self.last_sync}

    def components(self, limit=10):
        return self.associator.graph.components()[:limit]

    def edges_of(self, evidence_id):
        graph = self.associator.graph
        if evidence_id not in graph.observations:
            return {"error": "未知 evidence_id", "known_observations": len(graph.observations)}
        return {"evidence_id": evidence_id,
                "same_class": graph.class_of(evidence_id),
                "bundle": graph.bundle_of(evidence_id),
                "neighbors": [{"other": edge.b if edge.a == evidence_id else edge.a,
                               "relation": edge.relation.value,
                               "confidence": round(edge.confidence, 3),
                               "basis": list(edge.basis)} for edge in graph.neighbors(evidence_id)]}

    def needs(self, layers=None, diagnosis=None, limit=8):
        return query_needs(self.propositions(layers, diagnosis))[:limit]

    def support(self, layers=None, diagnosis=None, limit=20):
        rows = [p.to_dict() for p in self.propositions(layers, diagnosis)
                if p.state != "unresolved"]
        rows.sort(key=lambda row: (-row["bundles"], row["id"]))
        return rows[:limit]

    def graph_summary(self):
        """Compact summary safe to store in every artifact checkpoint."""
        summary = self.summary()
        memory = self.associator.graph.hypothesis_memory
        if memory is not None:
            summary.update(memory.snapshot())
        summary["largest_components"] = [
            {"size": component["size"], "members": component["members"][:8]}
            for component in self.components(5)]
        return summary
