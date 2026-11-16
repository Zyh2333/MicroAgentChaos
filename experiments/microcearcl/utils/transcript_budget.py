"""Bound a tool result before it enters the append-only transcript.

Full records already live in ``registry.evidence``; the transcript only needs
enough to reason and to cite.  Previewed records keep their ``evidence_id`` so
``read_evidence`` can expand any of them later.
"""
import json

DEFAULT_BUDGET = 12_000
DEFAULT_RECORDS = 24
DEFAULT_TEXT = 240
MIN_TEXT = 80


def size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _shrink(value, records, text):
    if isinstance(value, str):
        if len(value) <= text:
            return value
        return value[:text] + f'…[+{len(value) - text} chars; read_evidence 回读全文]'
    if isinstance(value, list):
        kept = [_shrink(item, records, text) for item in value[:records]]
        if len(value) > records:
            kept.append({'omitted_records': len(value) - records,
                         'returned_records': records,
                         'note': '其余记录用 read_evidence(该结果的 evidence_id) 回读'})
        return kept
    if isinstance(value, dict):
        return {key: _shrink(item, records, text) for key, item in value.items()}
    return value


def _stub(output, original):
    data = output.get('data') if isinstance(output, dict) else None
    keys = sorted(data) if isinstance(data, dict) else []
    return {'evidence_id': output.get('evidence_id') if isinstance(output, dict) else None,
            'data': {'omitted': True, 'original_chars': original, 'top_level_keys': keys,
                     'note': '结果超出转录预算；用 read_evidence(<evidence_id>) 按记录回读'},
            'input_filter': output.get('input_filter') if isinstance(output, dict) else None}


def bounded(output, budget=DEFAULT_BUDGET, records=DEFAULT_RECORDS, text=DEFAULT_TEXT):
    """Return a preview of ``output`` whose JSON form stays within ``budget``."""
    original = size(output)
    while True:
        view = _shrink(output, max(1, records), max(MIN_TEXT, text))
        if size(view) <= budget:
            return view
        if records <= 1 and text <= MIN_TEXT:
            return _stub(output, original)
        records = max(1, records // 2)
        text = max(MIN_TEXT, text // 2)
