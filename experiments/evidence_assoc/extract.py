"""Schema-free feature extraction from arbitrary observation text.

No extractor is mandatory: each one returns ``None``/empty when the corresponding
pattern is absent, and callers must treat that as masked.  Patterns are generic
syntax (timestamps, ``key=value``, IPs, HTTP status) rather than dataset fields.
"""
import re
from dataclasses import replace
from datetime import datetime, timezone

from .schema import (AGGREGATE, COMPARISON, DIRECT, METADATA, Features, Observation,
                     Ref, refs_of)

TRUNCATION_MARKER = re.compile(r"…\[tool display truncated (\d+) chars\]$")
STAMP = re.compile(r"\b(\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:?\d\d)?)\b")
EPOCH = re.compile(r"\b(1[6-9]\d{8}(?:\.\d+)?)\b")
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
POD = re.compile(r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+)*-[a-z0-9]{6,10}-[a-z0-9]{5})\b")
NODE = re.compile(r"\bnode-\d+\b")
UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
# Hex token that must contain both a digit and a letter, and must not be part of a
# decimal/scientific literal (``0.0994415283203125`` is a float, not an id).
LONG_ID = re.compile(r"(?<![\w.\-])(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)"
                     r"[0-9a-fA-F]{12,64}(?![\w.\-])")
# ``key=value`` in logs and ``"key": "value"`` in JSON traces; the optional quote
# around the key is what lets trace_id/task_id be recovered from structured text.
PAIR = re.compile(r"\b([A-Za-z][A-Za-z0-9_\-]{1,30})\"?\s*[=:]\s*\"?([A-Za-z0-9_\-\.]{6,128})\"?")
HTTP_STATUS = (
    re.compile(r"\b(?:http_status|status[ _]?code|status)\s*[=:]\s*\"?([1-5]\d\d)\b", re.I),
    re.compile(r"\bHTTP/\d(?:\.\d)?\s+([1-5]\d\d)\b"),
    re.compile(r"\"status\"\s*:\s*([1-5]\d\d)\b"),
)
EXCEPTION = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Exception|Error))\b")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{1,}|[一-鿿]{2,}")
DIGITS = re.compile(r"\d+")
# Provenance shapes.  A log file is ``<pod>_<container>.log``; a metric column is
# ``column/<service>&<stat>``.  These are syntactic conventions, not dataset
# field names, and only shape-valid candidates become entities.
LOG_STEM = re.compile(r"^(?P<pod>[^/]+)_(?P<container>[A-Za-z0-9][A-Za-z0-9_.\-]*)\.log$")
COLUMN_LOCATION = re.compile(r"^column/(?P<service>[A-Za-z0-9_.\-]+)&")
SERVICE_NAME = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

STOPWORDS = frozenset("""
the a an and or of to in on at for with from by is are was were be been it this that
if then else not no yes null none true false none ok error info debug warn warning
""".split())
# Keys whose value identifies an execution/session scope.  Used only for conflict
# detection, never required to exist.
SCOPE_KEYS = frozenset({"task_id", "taskid", "subtaskid", "subtask_id", "trace_id", "traceid",
                        "request_id", "requestid", "flowid", "flow_id", "span_id", "attempt",
                        "session_id", "run_id"})
IGNORED_KEYS = frozenset({"timestamp", "time", "level", "severity", "line", "line_no",
                          "location", "ref", "n", "count", "bytes", "offset", "limit",
                          "cost", "token", "tokens", "latency", "duration",
                          # log levels and logger/module names are not identities
                          "warning", "warn", "error", "info", "debug", "notice", "critical",
                          "fatal", "logger", "module", "name", "msg", "message", "file",
                          "path", "bucket_name", "thread", "process"})
# A key or value only counts as an identity anchor when it looks like one.
ID_SUFFIXES = ("id", "uuid", "key", "hash", "token", "trace", "span", "flow",
               "identifier", "request", "session", "run")
ID_SHAPE = re.compile(r"[A-Za-z0-9_\-]{6,128}")

DERIVATION_BY_TOOL = {
    "log_census": AGGREGATE,
    "read_metrics": AGGREGATE,
    "analyze_latency": AGGREGATE,
    "compare_presence": COMPARISON,
    "read_topology": METADATA,
    "read_runtime_events": DIRECT,
}


def epoch(value):
    """Parse ISO/epoch timestamps; nanoseconds are truncated to microseconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(" ", "T", 1) if " " in value and "T" not in value else value.strip()
    text = re.sub(r"(\.\d{6})\d+", r"\1", text).replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()


def times(text):
    """Return (start, end) from all timestamps found, or (None, None)."""
    found = []
    for match in STAMP.finditer(text):
        value = epoch(match.group(1))
        if value is not None:
            found.append(value)
    for match in EPOCH.finditer(text):
        found.append(float(match.group(1)))
    if not found:
        return None, None
    return min(found), max(found)


def status(text):
    for pattern in HTTP_STATUS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def exception(text):
    match = EXCEPTION.search(text)
    return match.group(1) if match else None


def _identifier_pair(key, value):
    """Accept a ``key=value`` pair only when it can plausibly anchor an identity.

    ``WARNING:urllib3.connectionpool`` and ``ip=10.244.1.103`` both match a generic
    ``key:value`` pattern but neither is an execution identifier; treating them as
    anchors chained unrelated log lines into one bundle during calibration.
    """
    if key in IGNORED_KEYS or value.lower() in STOPWORDS:
        return False
    if "." in value or not ID_SHAPE.fullmatch(value):
        return False
    if not any(ch.isdigit() for ch in value):
        return False
    return key.endswith(ID_SUFFIXES) or len(value) >= 12


def identifiers(text):
    pairs, seen, values = [], set(), set()
    for key, value in PAIR.findall(text):
        key = key.lower()
        if not _identifier_pair(key, value):
            continue
        item = (key, value)
        if item not in seen:
            seen.add(item)
            values.add(value)
            pairs.append(item)
    # A long hex token already captured under a named key keeps that key; the
    # generic "id" alias is only added for tokens no key claimed.
    for pattern in (UUID, LONG_ID):
        for match in pattern.finditer(text):
            if match.group(0) in values:
                continue
            item = ("id", match.group(0))
            if item not in seen:
                seen.add(item)
                values.add(item[1])
                pairs.append(item)
    return tuple(sorted(pairs))


def entities(text, vocabulary=()):
    found = set(match.group(0) for match in IPV4.finditer(text))
    found.update(match.group(1) for match in POD.finditer(text))
    found.update(match.group(0) for match in NODE.finditer(text))
    lowered = text.lower()
    for name in vocabulary or ():
        if name and name.lower() in lowered:
            found.add(name)
    return frozenset(found)


def entity_kind(value):
    """Classify an entity string so structural weights do not need name heuristics."""
    if value.count(".") == 3 and all(part.isdigit() for part in value.split(".")):
        return "ip"
    if value.startswith("node-") and value[5:].isdigit():
        return "node"
    if POD.fullmatch(value):
        # A Kubernetes pod ends with <replicaset-hash 6-10>-<pod-hash 5>; the
        # replicaset hash carries a digit.  The pod hash itself may be all
        # letters, and a service prefix like ``agent-network-image`` also matches
        # the shape, so the digit check is what separates them.
        _service, replicaset, _suffix = value.rsplit("-", 2)
        if any(ch.isdigit() for ch in replicaset):
            return "pod"
    return "service"


def template(text):
    """Digit/identifier-masked skeleton.  Shared templates are weak evidence only."""
    value = STAMP.sub("<ts>", text)
    value = IPV4.sub("<ip>", value)
    value = UUID.sub("<id>", value)
    value = LONG_ID.sub("<id>", value)
    value = DIGITS.sub("#", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value[:400]


def tokens(text):
    out = set()
    for token in WORD.findall(text.lower()):
        if len(token) < 2 or token in STOPWORDS or token.isdigit():
            continue
        out.add(token)
    return frozenset(out)


def strip_truncation(text):
    match = TRUNCATION_MARKER.search(text)
    if not match:
        return text, False
    return text[:match.start()].rstrip(), True


def producers(observation):
    """Services/pods that *emitted* the record, from provenance only.

    Independence is about the observer, not about the entities the record
    mentions: two callers observing the same target share its IP but are
    independent perspectives.
    """
    return frozenset(entity for entity, source in observation.features.entity_sources
                     if source == "ref")


def producer_separated(left, right):
    """Whether the two records were emitted by different producers."""
    left_producers, right_producers = producers(left), producers(right)
    if left_producers and right_producers:
        return not (left_producers & right_producers)
    left_file = left.anchor.file if left.anchor else None
    right_file = right.anchor.file if right.anchor else None
    return bool(left_file and right_file and left_file != right_file)


def entities_from_refs(refs, text_entities=(), vocabulary=()):
    """Entities recoverable from provenance: service/pod names live in the path.

    Only shape-valid candidates are admitted, and shared directories that do not
    encode a pod or a service column yield nothing.  The result maps entity to
    the provenance that produced it, so text- and path-derived structure stay
    auditable and can be ablated.
    """
    found = {}
    lowered = {name.lower() for name in vocabulary or ()}
    for ref in refs or ():
        stem = ref.file.rsplit("/", 1)[-1]
        match = LOG_STEM.match(stem)
        if match:
            pod = match.group("pod")
            if entity_kind(pod) == "pod":
                found.setdefault(pod, "ref")
            container = match.group("container")
            if SERVICE_NAME.fullmatch(container) and (
                    container in lowered or container not in text_entities):
                found.setdefault(container, "ref")
        match = COLUMN_LOCATION.match(ref.location or "")
        if match:
            service = match.group("service")
            if SERVICE_NAME.fullmatch(service):
                found.setdefault(service, "ref")
    return {entity: source for entity, source in found.items()
            if entity not in text_entities}


def features(text, vocabulary=()):
    body, truncated = strip_truncation(text)
    start, end = times(body)
    found = entities(body, vocabulary)
    return Features(tokens=tokens(body), template=template(body), identifiers=identifiers(body),
                    entities=found, status=status(body), start=start, end=end,
                    exception=exception(body), truncated=truncated,
                    entity_sources=tuple(sorted((entity, "text") for entity in found)))


def derivation_of(tool):
    return DERIVATION_BY_TOOL.get(tool, DIRECT)


def observation(evidence_id, fact, vocabulary=(), query=None, text=None):
    """Build an Observation from a baseline registry evidence record.

    Entities come from the text *and* from the reference path.  The second source
    is why a log line whose body never names its service still has structure.
    """
    raw = fact.get("text") if text is None else text
    if raw is None:
        raw = ""
    refs = refs_of(fact.get("references"))
    extracted = features(raw, vocabulary)
    from_refs = entities_from_refs(refs, set(extracted.entities), vocabulary)
    if from_refs:
        sources = dict(extracted.entity_sources)
        sources.update(from_refs)
        extracted = replace(extracted, entities=frozenset(extracted.entities) | set(from_refs),
                            entity_sources=tuple(sorted(sources.items())))
    derivation = derivation_of(fact.get("tool"))
    derived_from = refs if derivation in (AGGREGATE, COMPARISON) else ()
    kind = "log" if fact.get("tool") in ("read_logs", "search_logs", "read_exceptions") else (
        fact.get("tool") or "text")
    return Observation(id=evidence_id, origin=evidence_id, kind=kind, tool=fact.get("tool"),
                       stage=fact.get("stage"), query=query, text=raw, refs=refs,
                       features=extracted, derivation=derivation,
                       derived_from=derived_from)


def vocabulary_from_messages(messages):
    """Read the declared service list from the baseline user message, if present."""
    import json
    for message in messages or ():
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (ValueError, TypeError):
            continue
        names = payload.get("services")
        if isinstance(names, list):
            return tuple(str(n) for n in names if isinstance(n, str))
    return ()


def shared_refs(a, b):
    return {ref.key for ref in a.refs} & {ref.key for ref in b.refs}


def shared_identifiers(a, b):
    left, right = a.features.identifier_map, b.features.identifier_map
    shared = set()
    for key in set(left) & set(right):
        if left[key] == right[key]:
            shared.add((key, left[key]))
    return shared


def scope_conflict(a, b):
    """Same content but provably different execution scope."""
    left, right = a.features.identifier_map, b.features.identifier_map
    for key in set(left) & set(right):
        if key in SCOPE_KEYS and left[key] != right[key]:
            return True
    return False
