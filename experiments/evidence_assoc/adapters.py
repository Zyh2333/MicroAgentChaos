"""Adapters: baseline registry evidence -> observations; artifacts -> claims.

The baseline already funnels every record through ``Registry.evidence`` with a
stable ``observation-N:record:i`` handle, a tool name, a stage, the record text
and file/location references.  That is enough for the observation contract; no
change to the running agent is required to analyse past runs.
"""
import json
import re
import statistics
from collections import defaultdict, deque
from pathlib import Path

from .extract import observation, vocabulary_from_messages
from .hypothesis import proposition_claims
from .propositions import (BOUNDARY, CF, CONTRADICTION, CONTROL, CORROBORATION,
                           DIAGNOSIS, SUPPORT, Claim)
from .schema import Observation


def _query_of(evidence_id):
    return evidence_id.split(":record:")[0] if ":record:" in evidence_id else evidence_id


def observations_from_evidence(evidence, vocabulary=()):
    """Skip tool-level handles; keep record-level observations with non-empty text."""
    rows = []
    for evidence_id, fact in (evidence or {}).items():
        if not isinstance(fact, dict) or fact.get("kind") != "raw_observation":
            continue
        if not (fact.get("text") or "").strip():
            continue
        rows.append(observation(evidence_id, fact, vocabulary, query=_query_of(evidence_id)))
    return rows


def observations_from_artifact(artifact, vocabulary=None):
    vocabulary = tuple(vocabulary) if vocabulary else vocabulary_from_messages(artifact.get("messages"))
    return observations_from_evidence(artifact.get("rules") or {}, vocabulary), vocabulary


def expand_ids(graph, evidence_ids):
    """Expand tool-level handles to the record ids the registry actually stored."""
    resolved, unknown = [], []
    for evidence_id in evidence_ids or ():
        if not isinstance(evidence_id, str):
            continue
        if evidence_id in graph.observations:
            resolved.append(evidence_id)
            continue
        children = sorted(o for o in graph.observations if o.startswith(evidence_id + ":record:"))
        if children:
            resolved.extend(children)
        else:
            unknown.append(evidence_id)
    return resolved, unknown


def _ids(value):
    if isinstance(value, str):
        return [value]
    return [item for item in value or () if isinstance(item, str)]


def claims_from_layers(layers, graph):
    claims, unknown = [], []
    for layer, record in (layers or {}).items():
        if not isinstance(record, dict):
            continue
        for entry in record.get("support") or ():
            if not isinstance(entry, dict):
                continue
            resolved, missing = expand_ids(graph, _ids(entry.get("evidence_id")))
            unknown.extend(missing)
            claims.append(Claim(rule_id=entry.get("rule_id") or layer, kind=SUPPORT,
                                evidence_ids=resolved, note=str(entry.get("note") or "")))
        for entry in record.get("contradictions") or ():
            if not isinstance(entry, dict):
                continue
            resolved, missing = expand_ids(graph, _ids(entry.get("evidence_id")))
            unknown.extend(missing)
            claims.append(Claim(rule_id=entry.get("rule_id") or layer, kind=CONTRADICTION,
                                evidence_ids=resolved))
        for entry in record.get("controls") or ():
            if not isinstance(entry, dict):
                continue
            resolved, missing = expand_ids(graph, _ids(entry.get("evidence_id")))
            unknown.extend(missing)
            claims.append(Claim(rule_id=entry.get("rule_id") or layer, kind=CONTROL,
                                evidence_ids=resolved))
        for entry in record.get("cf") or ():
            if not isinstance(entry, dict):
                continue
            rule_id = entry.get("rule_id") or f"{layer}-CF"
            resolved, missing = expand_ids(graph, _ids(entry.get("evidence_ids")))
            controls, missing_controls = expand_ids(
                graph, _ids(entry.get("control_ids")) + _ids(entry.get("coverage_evidence_ids")))
            unknown.extend(missing)
            unknown.extend(missing_controls)
            if not resolved and not controls:
                # An empty CF statement contains no refutable content.
                continue
            claims.append(Claim(rule_id=rule_id, kind=CF, evidence_ids=resolved,
                                control_ids=controls, state=entry.get("state"),
                                note=str(entry.get("explanation") or "")))
    return claims, unknown


def claims_from_diagnosis(diagnosis, graph):
    claims, unknown = [], []
    if not isinstance(diagnosis, dict):
        return claims, unknown
    for entry in diagnosis.get("diagnoses") or ():
        if not isinstance(entry, dict):
            continue
        rule_id = f"{entry.get('layer', '')}:{entry.get('fault_type', '')}".strip(":")
        resolved, missing = expand_ids(graph, _ids(entry.get("evidence_ids")))
        unknown.extend(missing)
        claims.append(Claim(rule_id=rule_id, kind=DIAGNOSIS, evidence_ids=resolved,
                            note=str(entry.get("explanation") or "")))
        for kind, key in ((CORROBORATION, "corroboration_evidence_ids"),
                          (BOUNDARY, "boundary_evidence_ids")):
            resolved, missing = expand_ids(graph, _ids(entry.get(key)))
            unknown.extend(missing)
            if resolved:
                claims.append(Claim(rule_id=rule_id, kind=kind, evidence_ids=resolved))
    return claims, unknown


def claims_from_artifact(artifact, graph):
    """Project the baseline's own assertions onto the graph's observation ids."""
    layer_claims, unknown = claims_from_layers(artifact.get("layers") or {}, graph)
    diagnosis_claims, missing = claims_from_diagnosis(artifact.get("diagnosis"), graph)
    promoted = proposition_claims((artifact.get("evidence_graph") or {}).get("proposition_evidence", {}))
    return layer_claims + diagnosis_claims + promoted, unknown + missing


# ------------------------------------------------------------- topology index
def _normalise(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


class ServiceGraph:
    """Undirected service-level call graph recovered from execution traces."""

    def __init__(self, edges=(), vocabulary=()):
        self.adjacency = defaultdict(set)
        self.vocabulary = sorted({v for v in vocabulary if v}, key=len, reverse=True)
        self.normalised = {_normalise(v): v for v in self.vocabulary}
        # Observed end-to-end execution durations in this run (seconds).  The
        # median is the typical execution and calibrates the time window, so no
        # fixed window is configured anywhere.
        self.execution_durations = []
        for left, right in edges:
            self._link(left, right)

    def resolve(self, vertex):
        """Map a span vertex to a known service, else its group prefix."""
        key = _normalise(str(vertex))
        for name, value in self.normalised.items():
            if name and key.startswith(name):
                return value
        return str(vertex).split("/")[0]

    def _link(self, left, right):
        if not left or not right or left == right:
            return
        self.adjacency[left].add(right)
        self.adjacency[right].add(left)

    def distance(self, targets, sources, limit=4):
        """Hop distance between two entity sets; None when unreachable in ``limit``."""
        targets = [t for t in targets if t in self.adjacency]
        sources = {s for s in sources if s in self.adjacency}
        if not targets or not sources:
            return None
        if set(targets) & sources:
            return 0
        frontier, seen = deque((t, 0) for t in targets), set(targets)
        while frontier:
            node, depth = frontier.popleft()
            if depth >= limit:
                continue
            for neighbor in self.adjacency.get(node, ()):
                if neighbor in sources:
                    return depth + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append((neighbor, depth + 1))
        return None

    @property
    def execution_span(self):
        """Median observed execution duration; ``None`` when unobserved."""
        durations = [value for value in self.execution_durations if value > 0]
        return statistics.median(durations) if durations else None

    @property
    def size(self):
        return len(self.adjacency)


class TopologyIndex:
    """Structural distance between two observations, masked when undecidable."""

    def __init__(self, services: ServiceGraph, zones=None):
        self.services = services
        self.zones = zones or {}

    @property
    def execution_span(self):
        return self.services.execution_span

    def _entities(self, observation):
        return {e for e in observation.features.entities if e in self.services.adjacency}

    def distance(self, a, b):
        left, right = self._entities(a), self._entities(b)
        if not left or not right:
            return None
        return self.services.distance(left, right)


def service_graph_from_sample(sample_path, vocabulary=()):
    """Read service-level call edges from ``agent-network/graph/call_chains.json``."""
    sample_path = Path(sample_path)
    path = sample_path / "agent-network/graph/call_chains.json"
    if not path.is_file():
        return ServiceGraph(vocabulary=vocabulary)
    try:
        chains = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ServiceGraph(vocabulary=vocabulary)
    graph = ServiceGraph(vocabulary=vocabulary)
    if not isinstance(chains, list):
        return graph
    for chain in chains:
        total = 0.0
        for level in chain.get("levels") or ():
            for edge in level.get("edges") or ():
                graph._link(graph.resolve(edge.get("source", "")), graph.resolve(edge.get("destination", "")))
            durations = [span.get("time") for span in (level.get("spans") or {}).values()
                         if isinstance(span, dict) and isinstance(span.get("time"), (int, float))]
            if durations:
                total += max(durations)
        if total > 0:
            graph.execution_durations.append(total)
    return graph


def observations_for_registry(registry, vocabulary=()):
    """Live bridge entry point: convert the registry's current evidence dict."""
    return observations_from_evidence(registry.evidence, vocabulary)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
