"""Read, write, and remove fields using simple JSON paths."""
import re
from typing import Any, List

_INDEX_RE = re.compile(r"^(\w+)\[(\d+)\]$")


def parse_tokens(path: str) -> List:
    """Parse a path into keys and indices, such as a.b[0]."""
    if path in ("$", "", None):
        return []
    p = path.lstrip("$").lstrip(".")
    tokens: List = []
    for part in p.split("."):
        if not part:
            continue
        m = _INDEX_RE.match(part)
        if m:
            tokens.append(m.group(1))
            tokens.append(int(m.group(2)))
        elif part.isdigit():
            tokens.append(int(part))
        else:
            tokens.append(part)
    return tokens


def jp_get(data: Any, path: str, default: Any = None) -> Any:
    try:
        cur = data
        for tok in parse_tokens(path):
            cur = cur[tok]
        return cur
    except (KeyError, IndexError, TypeError):
        return default


def jp_set(data: Any, path: str, value: Any) -> Any:
    """Set an existing path; ignore missing keys or indices."""
    tokens = parse_tokens(path)
    if not tokens:
        return value
    try:
        cur = data
        for tok in tokens[:-1]:
            cur = cur[tok]
        cur[tokens[-1]] = value
    except (KeyError, IndexError, TypeError):
        pass
    return data


def jp_remove(data: Any, path: str) -> Any:
    """Remove a path if it exists."""
    tokens = parse_tokens(path)
    if not tokens:
        return data
    try:
        cur = data
        for tok in tokens[:-1]:
            cur = cur[tok]
        last = tokens[-1]
        if isinstance(cur, dict):
            cur.pop(last, None)
        elif isinstance(cur, list) and isinstance(last, int) and 0 <= last < len(cur):
            cur.pop(last)
    except (KeyError, IndexError, TypeError):
        pass
    return data
