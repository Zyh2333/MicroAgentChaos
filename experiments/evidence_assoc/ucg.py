"""Unexplained Case Graph (UCG) snapshots.

This module implements **only** the first stage of §7 in ``METHOD.md``: freezing
the localization graph of one unexplained sample into a self-contained, label-free
snapshot.  Cross-sample mining, candidate-rule generation and the validation
protocol are intentionally not implemented here.

Snapshots are written one file per sample and never merged.  Similar log lines or
coarse source anchors (e.g. ``compare_presence`` rows with ``location="file"``)
would otherwise chain two unrelated failure paths together across samples.

Label policy: a snapshot contains the system's own inference output (terminal
status, propositions, edges) and the observations it was built from.  It never
contains the ground-truth label, the fault type, the failure mode, the case name,
or anything under ``chaos/``.
"""
import hashlib
import json
from pathlib import Path

from .audit import build

SCHEMA_VERSION = 1
TERMINAL_REASONS = ("OutOfTaxonomy", "InsufficientSupport", "Mixed")
EDGE_CAP = 5000
MEMBER_CAP = 50


def _safe_metadata(artifact, source):
    """Whitelist: identity and deployment condition only, never the answer."""
    metadata = artifact.get("sample_metadata") or {}
    return {
        "sample_id": metadata.get("sample_id") or Path(source).stem,
        "topology_mode": metadata.get("topology_mode"),
        "load": metadata.get("load"),
        "source_artifact": Path(source).name if source else None,
        "label_policy": "no ground-truth label, fault type, failure mode, case name, or chaos data",
    }


def _terminal(artifact):
    diagnosis = artifact.get("diagnosis") if isinstance(artifact.get("diagnosis"), dict) else {}
    return {
        "termination": artifact.get("termination"),
        "stage": artifact.get("stage"),
        "diagnostic_status": diagnosis.get("diagnostic_status"),
        "unexplained_reason": diagnosis.get("unexplained_reason"),
        "candidate_layers": artifact.get("candidate_layers") or [],
        "parked_layers": artifact.get("parked_layers") or [],
        "excluded_layers": artifact.get("excluded_layers") or [],
        "supported_layers": [layer for layer, record in (artifact.get("layers") or {}).items()
                             if isinstance(record, dict) and record.get("support_state") == "present"],
        "layer_order": diagnosis.get("layer_order") or [],
    }


def is_unexplained(artifact):
    """True when the run ends without a diagnosis, or explicitly unexplained."""
    terminal = _terminal(artifact)
    return bool(terminal["unexplained_reason"]) or terminal["diagnostic_status"] == "Unexplained"


def _observation_row(observation):
    features = observation.features
    return {
        "id": observation.id,
        "origin": observation.origin,
        "tool": observation.tool,
        "stage": observation.stage,
        "query": observation.query,
        "derivation": observation.derivation,
        "identity_key": observation.identity_key,
        "content_key": observation.content_key,
        "refs": [ref.key for ref in observation.refs],
        "derived_from": [ref.key for ref in observation.derived_from],
        "features": {
            "template": features.template[:240],
            "identifiers": [list(item) for item in features.identifiers],
            "entities": sorted(features.entities),
            "status": features.status,
            "span": features.span,
            "truncated": features.truncated,
        },
        "text_preview": observation.text[:240],
        "text_sha1": hashlib.sha1(observation.text.encode("utf-8", "replace")).hexdigest(),
    }


def _bundle_row(graph, members):
    internal = [edge for edge in graph.edges.values()
                if edge.a in members and edge.b in members]
    stamps = [graph.observations[m].features.span for m in members
              if graph.observations[m].features.span]
    tools = {}
    for member in members:
        tool = graph.observations[member].tool or "unknown"
        tools[tool] = tools.get(tool, 0) + 1
    return {
        "bundle_id": graph.bundle_of(members[0]),
        "size": len(members),
        "members": sorted(members)[:MEMBER_CAP],
        "members_omitted": max(0, len(members) - MEMBER_CAP),
        "tools": dict(sorted(tools.items())),
        "relations": {relation: sum(1 for edge in internal if edge.relation.value == relation)
                      for relation in ("SAME", "DEPENDENT")},
        "time_span": [min(s[0] for s in stamps), max(s[1] for s in stamps)] if stamps else None,
    }


def snapshot(graph, propositions, source=None, artifact=None, built=None):
    """Build the label-free snapshot for one sample."""
    artifact = artifact or {}
    observations = list(graph.observations.values())
    stamps = [o.features.span for o in observations if o.features.span is not None]
    components = graph.components()
    edges = list(graph.edges.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "unexplained_case_graph",
        "sample": _safe_metadata(artifact, source),
        "window": {"start": min(s[0] for s in stamps) if stamps else None,
                   "end": max(s[1] for s in stamps) if stamps else None},
        "services": list((built or {}).get("vocabulary") or ()),
        "terminal": _terminal(artifact),
        "observations": [_observation_row(o) for o in observations],
        "bundles": [_bundle_row(graph, component["members"]) for component in components],
        "edges": [edge.to_dict() for edge in edges[:EDGE_CAP]],
        "edges_omitted": max(0, len(edges) - EDGE_CAP),
        "propositions": [proposition.to_dict() for proposition in propositions],
        "coverage": {
            "observations": len(observations),
            "bundles": len(components),
            "merged_duplicates": graph.summary()["merged_duplicates"],
            "edges": graph.summary()["edges"],
            "unknown_reasons": graph.summary()["unknown"],
            "demotions": graph.summary()["demotions"],
        },
        "provenance": {
            "method": "evidence_assoc/ucg",
            "degree_budget": graph.degree_budget,
            "policy": dict(getattr(graph.policy, "__dict__", {})),
        },
    }


def from_artifact(path):
    """Build a snapshot from a saved MicroCEARCL artifact (no labels included)."""
    path = Path(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    built = build(artifact)
    return snapshot(built["graph"], built["propositions"], source=path, artifact=artifact,
                    built=built)


def write_pool(paths, out_dir, responses=None, include_all=False):
    """Write one UCG file per sample plus a manifest; never merge samples."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    responses = set(responses or TERMINAL_REASONS)
    items, skipped = [], []
    for path in paths:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if not include_all:
            if not is_unexplained(artifact):
                skipped.append({"artifact": Path(path).name, "why": "explained"})
                continue
            reason = _terminal(artifact)["unexplained_reason"]
            if reason and reason not in responses:
                skipped.append({"artifact": Path(path).name, "why": reason})
                continue
        snapshot_doc = from_artifact(path)
        if not snapshot_doc["observations"]:
            # Older artifacts registered only tool-level handles, so no graph can
            # be rebuilt.  An empty snapshot would be misleading.
            skipped.append({"artifact": Path(path).name, "why": "no_record_level_evidence"})
            continue
        sample_id = snapshot_doc["sample"]["sample_id"]
        target = out / f"{sample_id}.ucg.json"
        target.write_text(json.dumps(snapshot_doc, ensure_ascii=False, indent=1), encoding="utf-8")
        items.append({"sample_id": sample_id, "file": target.name,
                      "reason": snapshot_doc["terminal"]["unexplained_reason"],
                      "observations": len(snapshot_doc["observations"]),
                      "bundles": len(snapshot_doc["bundles"])})
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "ucg_pool",
                "count": len(items), "items": items, "skipped": skipped,
                "policy": "one file per sample; snapshots are not merged"}
    (out / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    return manifest
