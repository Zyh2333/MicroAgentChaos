"""Project the append-only transcript into a bounded request context.

Full messages stay in the artifact for audit.  The request view keeps the most
recent round verbatim and folds older tool results into evidence references:

* tool results that touch ACTIVE/DORMANT evidence keep ``id + state + summary``;
* tool results whose evidence is entirely ARCHIVED keep the ids and a note;
* older state notices collapse into a one-line marker.

Messages are replaced, never removed, so assistant ``tool_calls`` and their
``tool`` replies stay paired.  The projection is deterministic for identical
inputs.
"""
import json


def size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _truncate(text, limit):
    if not text or len(text) <= limit:
        return text
    return text[:max(0, limit - 1)].rstrip() + "…"


def project_messages(messages, states, call_evidence, summarize=None, budget=24000,
                     keep_from=0, max_evidence=6, summary_chars=120):
    """Return ``(projected_messages, stats)``.

    ``keep_from`` is the first message index kept verbatim (the latest round).
    ``states`` maps evidence id to ACTIVE/DORMANT/ARCHIVED; ``call_evidence``
    maps a tool_call id to the evidence ids that call registered.
    """
    projected = []
    stats = {"tool_folded": 0, "notices_folded": 0,
             "before_chars": sum(size(message) for message in messages)}
    for index, message in enumerate(messages):
        role = message.get("role")
        if index >= keep_from or role == "system":
            projected.append(message)
            continue
        if role == "tool":
            ids = [item for item in (call_evidence.get(message.get("tool_call_id")) or ())
                   if states.get(item)]
            if ids:
                live = [item for item in ids if states[item] in ("ACTIVE", "DORMANT")]
                if live:
                    evidence = []
                    for item in live[:max_evidence]:
                        row = {"id": item, "state": states[item]}
                        if summarize is not None:
                            summary = summarize(item)
                            if summary:
                                row["summary"] = _truncate(summary, summary_chars)
                        evidence.append(row)
                    content = {"projected": True, "evidence": evidence,
                               "omitted": max(0, len(live) - max_evidence),
                               "note": "本轮上下文按贡献选择；其余记录可用 read_evidence(id) 回读"}
                else:
                    content = {"projected": True, "archived": len(ids),
                               "evidence_ids": ids[:8],
                               "note": "该结果已归档；可用 read_evidence(id) 回读"}
                projected.append({**message,
                                  "content": json.dumps(content, ensure_ascii=False)})
                stats["tool_folded"] += 1
                continue
        if role == "user":
            try:
                payload = json.loads(message.get("content") or "{}")
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict) and "graph" in payload:
                content = {"projected": True, "round": payload.get("round"),
                           "note": "历史轮次状态已折叠；当前状态见本轮 graph"}
                projected.append({**message,
                                  "content": json.dumps(content, ensure_ascii=False)})
                stats["notices_folded"] += 1
                continue
        projected.append(message)
    stats["after_chars"] = sum(size(message) for message in projected)
    stats["budget"] = budget
    stats["keep_from"] = keep_from
    return projected, stats
