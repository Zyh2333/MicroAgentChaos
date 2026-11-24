"""Incremental association: candidate generation, pair inference, graph commit.

Candidate generation is a *blocking* step, not a scoring shortcut: a pair that
shares no informative token, entity, identifier or distinct template cannot reach
DEPENDENT under the rules in :mod:`relation` anyway, so it is recorded as
``no_anchor`` instead of being silently dropped.  Counts are reported so coverage
stays auditable.

Candidates are generated at ``resolve`` time, after the corpus is complete, so
document frequency reflects the whole observation set rather than the first few
records.
"""
from collections import Counter

from .compat import Corpus
from .graph import EvidenceGraph
from .relation import Policy, classify
from .schema import Relation


def _content_entities(features):
    """Only text-derived entities block candidate generation.

    Provenance-derived entities (service/pod from the file path) are used by the
    structural dimension for scoring, but they must not turn every same-service
    record into a candidate pair - that is O(n^2) work whose ceiling is a 0.25
    structural level that cannot admit a dependency on its own.
    """
    sources = dict(features.entity_sources)
    if not sources:
        return features.entities
    return frozenset(entity for entity, source in sources.items() if source == "text")


class BlockIndex:
    """Inverted index over token/entity/identifier/template/time-bucket keys."""

    def __init__(self):
        self.postings = {}

    @staticmethod
    def _push(table, key, value):
        table.setdefault(key, []).append(value)

    def add(self, observation, bucket_seconds=300):
        features = observation.features
        for token in features.tokens:
            self._push(self.postings, ("token", token), observation.id)
        for entity in _content_entities(features):
            self._push(self.postings, ("entity", entity), observation.id)
        for item in features.identifiers:
            self._push(self.postings, ("identifier", item), observation.id)
        if features.template:
            self._push(self.postings, ("template", features.template), observation.id)
        if features.span:
            self._push(self.postings, ("bucket", int(features.span[0] // bucket_seconds)),
                       observation.id)

    def get(self, kind, key):
        return self.postings.get((kind, key), ())

    def candidates(self, observation, total, cap=250, max_df_ratio=0.10, template_df_ratio=0.05):
        """Score shared anchors; accept two weak anchors or one strong anchor."""
        score = Counter()
        features = observation.features
        max_df = max(20, int(total * max_df_ratio))
        for token in features.tokens:
            postings = self.get("token", token)
            if len(postings) > max_df:      # ubiquitous token: no discriminative value
                continue
            for other in postings:
                if other != observation.id:
                    score[other] += 1
        for entity in _content_entities(features):
            for other in self.get("entity", entity):
                if other != observation.id:
                    score[other] += 3
        for item in features.identifiers:
            for other in self.get("identifier", item):
                if other != observation.id:
                    score[other] += 5
        if features.template and len(self.get("template", features.template)) <= max(
                20, int(total * template_df_ratio)):
            for other in self.get("template", features.template):
                if other != observation.id:
                    score[other] += 2
        if features.span:
            for other in self.get("bucket", int(features.span[0] // 300)):
                if other != observation.id:
                    score[other] += 1
        accepted = [(value, key) for key, value in score.items() if value >= 2]
        accepted.sort(key=lambda pair: (-pair[0], pair[1]))
        capped = len(accepted) > cap
        return [key for _, key in accepted[:cap]], capped, len(accepted)


class Associator:
    def __init__(self, policy=None, degree_budget=6, topology=None, candidate_cap=250):
        self.policy = policy or Policy()
        self.graph = EvidenceGraph(self.policy, degree_budget=degree_budget)
        self.corpus = Corpus()
        self.index = BlockIndex()
        self.topology = topology
        self.candidate_cap = candidate_cap
        self.unprocessed = []
        self.evaluated = set()
        self.stats = Counter()

    def add_observation(self, observation):
        if observation.id in self.graph.observations:
            return False
        self.graph.add_observation(observation)
        self.corpus.add(observation)
        self.index.add(observation)
        self.unprocessed.append(observation.id)
        self.stats["observations"] += 1
        return True

    def ingest(self, observations):
        return [o for o in observations if self.add_observation(o)]

    def _pairs(self):
        total = len(self.graph.observations)
        pairs = []
        for observation_id in self.unprocessed:
            observation = self.graph.observations.get(observation_id)
            if observation is None:
                continue
            candidates, capped, accepted = self.index.candidates(observation, total, self.candidate_cap)
            if not candidates:
                self.stats["no_anchor_observations"] += 1
            if capped:
                self.stats["candidate_cap_hits"] += 1
                self.stats["candidate_pairs_dropped"] += accepted - self.candidate_cap
            for other in candidates:
                pair = tuple(sorted((other, observation_id)))
                if pair not in self.evaluated:
                    self.evaluated.add(pair)
                    pairs.append(pair)
        self.unprocessed = []
        return pairs

    def resolve(self):
        """Classify pending pairs, then commit in confidence order.

        Highest-confidence relations land first; the graph then refuses merges or
        edges that would break equivalence, acyclicity or the degree budget.  This
        is a greedy approximation of the maximum-confidence consistent subgraph.
        """
        observations = self.graph.observations
        # The time window is the longest execution step observed in this run
        # (execution graph), else the widest observed chain, else the sampling
        # scale.  No fixed window is configured anywhere.
        span = getattr(self.topology, "execution_span", None)
        self.corpus.execution_span = (span or self.graph.execution_span()
                                      or self.corpus.time_scale)
        # The graph answers independence with the same corpus calibration.
        self.graph.corpus = self.corpus
        self.graph.corpus_span = self.corpus.execution_span
        verdicts = [classify(observations[a], observations[b], self.corpus,
                             topology=self.topology, policy=self.policy)
                    for a, b in self._pairs() if a in observations and b in observations]
        self.graph.evaluated_pairs += len(verdicts)
        rank = {Relation.SAME: 0, Relation.DEPENDENT: 1, Relation.INDEPENDENT: 2, Relation.UNKNOWN: 3}
        verdicts.sort(key=lambda v: (rank[v.relation], -v.confidence, v.a, v.b))
        outcomes = Counter()
        involved = set()
        for verdict in verdicts:
            outcome = self.graph.add(verdict)
            outcomes[outcome] += 1
            if outcome == "merged":
                involved.update((verdict.a, verdict.b))
        # A later record can prove an earlier merge joined two execution scopes.
        if involved:
            withdrawn = self.graph.reconcile(involved)
            if withdrawn:
                self.stats["reconciled_splits"] += withdrawn
        self.stats.update({f"commit_{k}": v for k, v in outcomes.items()})
        return outcomes

    def summary(self):
        return {"graph": self.graph.summary(), "association": dict(self.stats),
                "time_scale_seconds": round(self.corpus.time_scale, 3),
                "execution_span_seconds": (round(self.corpus.execution_span, 3)
                                           if self.corpus.execution_span else None)}
