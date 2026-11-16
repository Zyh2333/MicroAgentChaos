"""Suppress known auth evidence while retaining topology and independent faults."""
from .background_noise import policy, csv_service


def filter_text(sample, value, service, ref):
    """Apply the same provenance mask recursively to detailed graph queries."""
    if isinstance(value, str):
        return policy(sample).text(value, service=service, channel="graph_evidence", ref=ref)
    if isinstance(value, list):
        return [filter_text(sample, v, service, {**ref, "item": i}) for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: filter_text(sample, v, k if csv_service(k) else service, {**ref, "key": k})
                for k, v in value.items()}
    return value


def span_failed(sample, span, service, ref):
    mask = policy(sample)
    error = span.get("error")
    filtered = mask.text(error, service=service, channel="graph_spans", ref=ref) if isinstance(error, str) else error
    own = bool(filtered) or (span.get("status") == 3 and not (error and not filtered))
    return own or any(span_failed(sample, attempt, attempt.get("executor") or service,
                                 {**ref, "attempt": i}) for i, attempt in enumerate(span.get("attempts") or []))


def explicit_error(sample, span, service, ref):
    error = span.get("error")
    remaining = policy(sample).text(error, service=service, channel="graph_spans", ref=ref) if isinstance(error, str) else error
    return bool(remaining) or any(explicit_error(sample, a, a.get("executor") or service,
                                               {**ref, "attempt": i}) for i, a in enumerate(span.get("attempts") or []))


def filter_graph(sample, graph, independent_failure=False):
    mask = policy(sample)
    if not mask.enabled:
        return False
    for vertex in graph.vertexes.values():
        if csv_service(vertex.key):
            for text in vertex.error_texts:
                mask.learn(text, vertex.key)
    csv_auth = any(csv_service(v.key) and any(mask.related(t, v.key) for t in v.error_texts)
                   for v in graph.vertexes.values())
    changed = False
    from execution_graph import _match_signatures
    for vertex in graph.vertexes.values():
        texts = [mask.text(text, service=vertex.key, graph_csv_auth=csv_auth, channel="graph_evidence",
                          ref={"trace_id": graph.trace_id, "vertex": vertex.key, "text": i})
                 for i, text in enumerate(vertex.error_texts)]
        if texts == vertex.error_texts:
            continue
        changed = True
        vertex.error_texts = [text for text in texts if text]
        vertex.signatures = sorted({s for text in vertex.error_texts for s in _match_signatures(text)})
        vertex.failed = bool(vertex.signatures) or vertex.status_failed
        if not vertex.signatures and vertex.status == 3:
            vertex.status = None
    if changed and not independent_failure and not any(v.signatures for v in graph.vertexes.values()):
        # Terminal failure can be a consequence of the masked background. Mark
        # it unknown rather than converting it into successful execution.
        if graph.task_status in (0, 3, 8):
            graph.task_status = None
        for vertex in graph.vertexes.values():
            if vertex.status_failed:
                vertex.status_failed = False
                vertex.failed = False
                vertex.status = None
    return changed
