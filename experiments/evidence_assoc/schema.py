"""Open-world observation and relation schema.

An observation is O = (X, S, C):

* ``X`` content  - the record text and its extracted features.
* ``S`` source   - file/location references, the tool that produced it, the query
  that returned it, and the records it was derived from.
* ``C`` context  - time span, entities, identifiers and status/exception values.

Every field is optional.  Nothing here requires ``taskId``, ``serviceName`` or
``statusCode`` to exist; extractors fill what they can and mark what they cannot.
"""
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1


class Relation(str, Enum):
    SAME = "SAME"
    DEPENDENT = "DEPENDENT"
    INDEPENDENT = "INDEPENDENT"
    UNKNOWN = "UNKNOWN"


class PropositionState(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


# Derivation kinds.  ``direct`` is a record observed as-is; the others are
# produced by aggregating, comparing or summarising other records and therefore
# inherit dependence on their inputs.
DIRECT, AGGREGATE, COMPARISON, METADATA = "direct", "aggregate", "comparison", "metadata"


@dataclass(frozen=True)
class Ref:
    file: str
    location: str

    @classmethod
    def of(cls, value):
        if not isinstance(value, dict):
            return None
        file = value.get("file")
        if not file:
            return None
        return cls(str(file), str(value.get("location", "")))

    @property
    def key(self):
        return f"{self.file}#{self.location}"


def refs_of(values):
    out = []
    for value in values or ():
        ref = Ref.of(value)
        if ref is not None and ref not in out:
            out.append(ref)
    return tuple(out)


@dataclass(frozen=True)
class Features:
    """Extracted, all-optional features.  Absent means masked, never negative."""
    tokens: frozenset = frozenset()
    template: str = ""
    identifiers: tuple = ()          # sorted ((key, value), ...)
    entities: frozenset = frozenset()
    status: int | None = None
    start: float | None = None
    end: float | None = None
    exception: str | None = None
    truncated: bool = False
    # Which provenance produced each entity: ("text" | "ref").  Kept separate so
    # path-derived structure can be audited and ablated.
    entity_sources: tuple = ()

    @property
    def identifier_map(self):
        return {k: v for k, v in self.identifiers}

    @property
    def has_time(self):
        return self.start is not None or self.end is not None

    @property
    def span(self):
        start = self.start if self.start is not None else self.end
        end = self.end if self.end is not None else self.start
        return (start, end) if start is not None else None


@dataclass
class Observation:
    """A single addressable observation instance."""
    id: str
    origin: str                                   # evidence_id in the baseline registry
    kind: str = "text"
    tool: str | None = None
    stage: int | None = None
    query: str | None = None
    text: str = ""
    refs: tuple = ()
    features: Features = field(default_factory=Features)
    derivation: str = DIRECT
    derived_from: tuple = ()

    @property
    def anchor(self):
        return self.refs[0] if self.refs else None

    @property
    def identity_key(self):
        """Stable source identity: the primary reference, else the origin id."""
        return self.anchor.key if self.anchor else f"origin:{self.origin}"

    @property
    def content_key(self):
        f = self.features
        payload = "|".join([f.template, repr(f.identifiers), str(f.status)])
        return sha1(payload.encode()).hexdigest()[:16]

    @property
    def is_aggregate(self):
        return self.derivation in (AGGREGATE, COMPARISON)

    def to_dict(self):
        return {"id": self.id, "origin": self.origin, "kind": self.kind, "tool": self.tool,
                "stage": self.stage, "query": self.query, "derivation": self.derivation,
                "identity_key": self.identity_key, "content_key": self.content_key,
                "refs": [r.key for r in self.refs],
                "derived_from": [r.key for r in self.derived_from],
                "features": {"template": self.features.template,
                             "identifier_keys": sorted(self.features.identifier_map),
                             "entities": sorted(self.features.entities),
                             "entity_sources": dict(self.features.entity_sources),
                             "status": self.features.status,
                             "time": self.features.span,
                             "tokens": len(self.features.tokens),
                             "truncated": self.features.truncated},
                "text": self.text[:400]}


@dataclass
class Edge:
    a: str
    b: str
    relation: Relation
    confidence: float
    basis: tuple = ()
    detail: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.a, self.b) if self.relation != Relation.SAME else tuple(sorted((self.a, self.b)))

    def to_dict(self):
        return {"a": self.a, "b": self.b, "relation": self.relation.value,
                "confidence": round(self.confidence, 4), "basis": list(self.basis),
                "detail": self.detail}


@dataclass
class Dimension:
    """One compatibility dimension.  ``value is None`` means masked, not negative."""
    name: str
    value: float | None
    basis: tuple = ()
    detail: dict = field(default_factory=dict)

    @property
    def available(self):
        return self.value is not None

    def to_dict(self):
        return {"value": None if self.value is None else round(self.value, 4),
                "basis": list(self.basis), **self.detail}
