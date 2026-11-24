"""Append-only differential log of evidence-graph changes.

One JSON object per line, written the moment a decision happens, so the graph
evolution survives a crashed or budget-exhausted run.  Unlike the artifact (which
is rewritten every round and only keeps the terminal graph), this file is never
truncated and records *why* each edge was admitted, refused or withdrawn.

Event types:

* ``edge``           - ``op`` is ``add`` / ``demote`` / ``retract``; carries the
  relation, endpoints, confidence, basis and reason.
* ``unknown``        - an individually logged refusal (degree budget, cycle,
  scope conflict); high-volume "not enough evidence" verdicts are aggregated
  into the next ``sync`` event instead.
* ``sync``           - one tool call's ingest batch: graph counts after the
  resolve, the corpus calibration, and the per-reason unknown delta.
* ``claim`` / ``claim_retracted`` - E_S support ledger revisions.
* ``layer_support``  - the support facts the agent's feedback uses.
"""
import json
from pathlib import Path

from .conversation_log import timestamp


class GraphEventLog:
    def __init__(self, artifact):
        artifact = Path(artifact)
        self.path = artifact.with_suffix(".graph_events.jsonl")
        self.sequence = 0

    def write(self, event):
        if not isinstance(event, dict) or not event:
            return
        self.sequence += 1
        record = {"sequence": self.sequence, "timestamp": timestamp(), **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
