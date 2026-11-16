"""Generic display bounds without dataset-specific text rewriting."""

MAX_TEXT = 1200


def clean(value, limit=12):
    if isinstance(value, str):
        if len(value) <= MAX_TEXT:
            return value
        return value[:MAX_TEXT] + f'…[tool display truncated {len(value) - MAX_TEXT} chars]'
    if isinstance(value, list):
        result = [clean(x, limit) for x in value[:limit]]
        if len(value) > limit:
            result.append({'omitted_records': len(value) - limit, 'returned_records': limit})
        return result
    if isinstance(value, dict):
        return {k: clean(v, limit) for k, v in value.items()}
    return value
