"""Three compatibility dimensions, each independently maskable.

v0.2 removes all heuristic weights from the score:

* between dimensions: **unweighted mean** over the available dimensions;
* ``semantic``  : ``max(text overlap, template identity)`` - no weights;
* ``structural``: an *ordinal* evidence level (same execution > same scope anchor >
  same replica > same deployment unit > none);
* ``contextual``: ``exp(-gap / scale)`` - no weights.

A masked dimension (``value is None``) is never treated as a negative observation.
Text overlap uses inverse document frequency over the observations seen so far, so
a log template emitted by every service in every run cannot manufacture a high
semantic score.
"""
import math
from collections import Counter

from .extract import entity_kind, shared_identifiers, shared_refs
from .schema import Dimension

# Ordinal structural levels.  The numbers are the four admissible levels, not
# fitted weights: a higher level subsumes the evidence of a lower one.
LEVEL_EXECUTION = 1.0      # same trace / span: one execution
LEVEL_SCOPE = 0.75         # same task / endpoint / identifier: one execution scope
LEVEL_REPLICA = 0.5        # same pod
LEVEL_DEPLOYMENT = 0.25    # same node / service
LEVEL_NONE = 0.0


class Corpus:
    """Corpus statistics over the observations currently in the graph."""

    def __init__(self, observations=()):
        self.token_df = Counter()
        self.template_df = Counter()
        self.entity_df = Counter()
        self.identifier_df = Counter()
        self.n = 0
        self.time_scale = 60.0
        # Longest observed execution in this corpus; set by the associator from
        # the graph.  Falls back to the inter-observation scale before any
        # execution has been observed.
        self.execution_span = None
        for observation in observations:
            self.add(observation)

    def add(self, observation):
        features = observation.features
        self.token_df.update(set(features.tokens))
        if features.template:
            self.template_df[features.template] += 1
        self.entity_df.update(features.entities)
        self.identifier_df.update(set(features.identifiers))
        self.n += 1

    def idf(self, term):
        df = self.token_df.get(term, 0)
        return math.log((self.n + 1.0) / (df + 0.5)) + 0.05

    def weight(self, terms):
        return sum(self.idf(t) for t in terms)

    def template_shared_by_majority(self, value):
        """True when the template is emitted by at least half the corpus."""
        return bool(self.n) and self.template_df.get(value, 0) * 2 >= self.n


def build_corpus(observations):
    corpus = Corpus()
    stamps = []
    for observation in observations:
        corpus.add(observation)
        if observation.features.has_time:
            stamps.append(observation.features.span[0])
    stamps.sort()
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b > a)
    if gaps:
        median = gaps[len(gaps) // 2]
        corpus.time_scale = min(600.0, max(1.0, median * 10.0))
    return corpus


def semantic(a, b, corpus):
    """Semantic compatibility = max(text overlap, template identity).

    Both components are in [0, 1]; taking the maximum means "the strongest piece
    of textual evidence decides", with no weights to tune.
    """
    fa, fb = a.features, b.features
    if not fa.tokens and not fb.tokens:
        return Dimension("semantic", None, ("empty_text",))
    shared = fa.tokens & fb.tokens
    left, right = corpus.weight(fa.tokens), corpus.weight(fb.tokens)
    union = corpus.weight(fa.tokens | fb.tokens)
    weighted = corpus.weight(shared)
    containment = weighted / min(left, right) if min(left, right) > 0 else 0.0
    jaccard = weighted / union if union > 0 else 0.0
    content = min(1.0, max(containment, jaccard))

    identical_template = bool(fa.template) and fa.template == fb.template
    # A template shared by the majority is a background pattern, not evidence.
    template = 1.0 if identical_template and not corpus.template_shared_by_majority(
        fa.template) else 0.0

    value = max(content, template)
    basis = []
    if content >= 0.5:
        basis.append("token_overlap")
    if template:
        basis.append("distinctive_template")
    elif identical_template:
        basis.append("common_template")
    return Dimension("semantic", round(value, 4), tuple(basis),
                     {"token_content": round(content, 4), "template": template})


def _structural_signals(a, b):
    fa, fb = a.features, b.features
    signals = {}
    ids_a, ids_b = fa.identifier_map, fb.identifier_map
    for key in ("trace_id", "traceid"):
        if ids_a.get(key) and ids_a.get(key) == ids_b.get(key):
            signals["shared_trace"] = LEVEL_EXECUTION
    if ids_a.get("span_id") and ids_a.get("span_id") == ids_b.get("span_id"):
        signals["shared_span"] = LEVEL_EXECUTION
    for key in ("task_id", "taskid", "subtaskid", "flowid", "request_id", "requestid"):
        if ids_a.get(key) and ids_a.get(key) == ids_b.get(key):
            signals["shared_task"] = LEVEL_SCOPE
    shared_ids = set(fa.identifiers) & set(fb.identifiers)
    if shared_ids:
        # Any shared identity anchor; a stronger scope signal, if present,
        # already occupies the same or a higher level.
        signals.setdefault("shared_identifier", LEVEL_SCOPE)
    if ids_a.get("attempt") and ids_a.get("attempt") == ids_b.get("attempt"):
        signals["shared_attempt"] = LEVEL_DEPLOYMENT

    typed = {"ip": set(), "node": set(), "pod": set(), "service": set()}
    for entity in fa.entities:
        typed[entity_kind(entity)].add(entity)
    other = {"ip": set(), "node": set(), "pod": set(), "service": set()}
    for entity in fb.entities:
        other[entity_kind(entity)].add(entity)
    if typed["ip"] & other["ip"]:
        signals["shared_endpoint"] = LEVEL_SCOPE
    if typed["pod"] & other["pod"]:
        signals["shared_pod"] = LEVEL_REPLICA
    if typed["node"] & other["node"]:
        signals["shared_node"] = LEVEL_DEPLOYMENT
    if typed["service"] & other["service"]:
        # A service name is the weakest structural fact: a busy service emits
        # thousands of unrelated lines.
        signals["shared_service"] = LEVEL_DEPLOYMENT
    return signals


def structural(a, b, topology=None):
    """Structural compatibility as an ordinal level in {0, 0.25, 0.5, 0.75, 1}."""
    signals = _structural_signals(a, b)
    fa, fb = a.features, b.features
    if topology is not None:
        distance = topology.distance(a, b)
        if distance is not None:
            signals["topology_distance"] = {0: LEVEL_DEPLOYMENT, 1: LEVEL_SCOPE,
                                            2: LEVEL_REPLICA}.get(distance, LEVEL_DEPLOYMENT)
    if not signals:
        if not (fa.entities or fb.entities or fa.identifiers or fb.identifiers):
            return Dimension("structural", None, ("no_structure",))
        return Dimension("structural", LEVEL_NONE, ("no_shared_structure",))
    level = max(signals.values())
    return Dimension("structural", level,
                     tuple(sorted(signals, key=lambda k: -signals[k]))[:4],
                     {"signals": {k: v for k, v in signals.items()}})


def contextual(a, b, corpus):
    """Contextual compatibility = exp(-gap / scale); masked when a stamp is missing."""
    fa, fb = a.features, b.features
    if fa.span is None or fb.span is None:
        return Dimension("contextual", None, ("no_time",))
    (a_start, a_end), (b_start, b_end) = fa.span, fb.span
    gap = max(0.0, max(a_start, b_start) - min(a_end, b_end))
    scale = corpus.time_scale
    value = math.exp(-gap / scale) if gap > 0 else 1.0
    basis = ("time_overlap",) if gap == 0 else (
        "time_near",) if gap <= scale else ("time_distant",)
    return Dimension("contextual", round(value, 4), basis,
                     {"gap_seconds": round(gap, 3), "scale_seconds": round(scale, 3)})


def compatibility(a, b, corpus, topology=None):
    return [semantic(a, b, corpus), structural(a, b, topology), contextual(a, b, corpus)]


def combine(dims):
    """Unweighted mean over the *available* dimensions, plus their count."""
    values = [dimension.value for dimension in dims if dimension.available]
    return (round(sum(values) / len(values), 4) if values else 0.0), len(values)
