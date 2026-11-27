"""Pack the graph object sent in a round under one serialized-character limit."""
from collections import defaultdict
from itertools import zip_longest

from experiments.evidence_assoc.view import EVIDENCE_ROLES
from .transcript_budget import size


POLICY = "shared_graph_json_v1"


def _admit(state, edits, limit):
    """Keep a whole display unit, including any records it references, or roll it back."""
    applied = []
    for container, key, value in edits:
        if isinstance(container, list):
            container.append(value)
        elif key not in container:
            container[key] = value
        else:
            continue
        applied.append((container, key))
    if size(state) <= limit:
        return True
    for container, key in reversed(applied):
        if isinstance(container, list):
            container.pop()
        else:
            del container[key]
    return False


def _evidence_units(source, state, rows):
    table = state.get("evidence")
    references = defaultdict(list)
    for original, row in rows:
        for role in EVIDENCE_ROLES:
            for record in original.get(role, ()):
                edit = (row[role], None, record)
                if table is None:
                    yield [edit]
                else:
                    references[record].append(edit)

    groups = source.get("overview", {})
    overview = {group: [] for group in groups}
    for group, records in groups.items():
        for record in records:
            edits = [(state, "overview", overview), (overview[group], None, record)]
            if table is not None and "id" in record:
                references[record["id"]].extend(edits)
            else:
                yield edits

    # The contribution view's evidence table already has the M2 ranking.
    for eid, record in source.get("evidence", {}).items():
        yield [(table, eid, record), *references[eid]]


def _memory_units(memory, state):
    if not memory:
        return
    nodes, pairs = {}, []
    view = {**memory, "nodes": nodes, "pairs": pairs}
    attach = (state, "hypotheses", view)
    for pair in memory["pairs"]:
        yield [attach, *((nodes, key, memory["nodes"][key]) for key in pair["ids"]),
               (pairs, None, pair)]
    for key, node in memory["nodes"].items():
        if key not in nodes:
            yield [attach, (nodes, key, node)]


def pack_state(source, memory, limit):
    """Alternate ranked evidence and hypothesis units. Either may use all remaining space."""
    if limit < 2:
        raise ValueError("--state-budget 至少为 2 个字符，以容纳空 JSON 对象")
    state = {}
    for key in ("records", "round", "state_view", "unassessed_layers", "evidence_bundles"):
        if key in source:
            _admit(state, [(state, key, source[key])], limit)

    # Reserve count widths only. The ledger is updated once, from the final visible IDs.
    counts = dict.fromkeys(("active", "dormant", "archived", "protected", "selected"), source["records"])
    _admit(state, [(state, "activity", counts)], limit)
    _admit(state, [(state, "propositions", [])], limit)
    if "evidence" in source and not _admit(state, [(state, "evidence", {})], limit):
        return state

    rows = []
    for original in source["propositions"]:
        row = {k: ([] if k in EVIDENCE_ROLES else v) for k, v in original.items()
               if k not in ("graph_reading", "omitted")}
        if "propositions" in state and _admit(state, [(state["propositions"], None, row)], limit):
            rows.append((original, row))

    evidence = _evidence_units(source, state, rows)
    hypotheses = _memory_units(memory, state)
    for units in zip_longest(evidence, hypotheses):
        for edits in units:
            if edits:
                _admit(state, edits, limit)

    for original, row in rows:
        omitted = original.get("omitted", 0) + len(original["support"]) - len(row["support"])
        if omitted:
            _admit(state, [(row, "omitted", omitted)], limit)
        if all(original.get(k) == row.get(k) for k in EVIDENCE_ROLES):
            _admit(state, [(row, "graph_reading", original["graph_reading"])], limit)
    return state


def finish_activity(state, activity, limit):
    """Replace the reserved counters and include as many reactivation events as fit."""
    if "activity" in state:
        counts = state["activity"]
        for key in counts:
            counts[key] = activity[key]
        if _admit(state, [(counts, "reactivated", [])], limit):
            for event in activity["reactivated"]:
                if not _admit(state, [(counts["reactivated"], None, event)], limit):
                    break
        _admit(state, [(counts, "note", activity["note"])], limit)
        if _admit(state, [(counts, "budget_chars", limit), (counts, "spent_chars", limit)], limit):
            # The reported size includes the decimal digits of the field itself.
            while counts["spent_chars"] != size(state):
                counts["spent_chars"] = size(state)
