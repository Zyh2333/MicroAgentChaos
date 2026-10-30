"""Match and apply HTTP fault rules, then log each injection."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

try:
    from .jsonpath import jp_get, jp_set, jp_remove
    from .taxonomy import FaultRule
except ImportError:
    from jsonpath import jp_get, jp_set, jp_remove  # type: ignore
    from taxonomy import FaultRule  # type: ignore


@dataclass
class RequestCtx:
    method: str
    path: str
    headers: dict
    upstream: Optional[str] = None
    taskId: Optional[str] = None

    def header(self, name: str) -> Optional[str]:
        # Header names are case insensitive.
        low = name.lower()
        for k, v in self.headers.items():
            if k.lower() == low:
                return v
        return None


@dataclass
class Outcome:
    """Result of applying rules at one request or response stage."""
    short_circuit: Optional[str] = None   # None | "abort" | "drop"
    status: int = 500
    body: Any = None
    delay_ms: float = 0.0
    body_mutated: bool = False
    body_value: Any = None
    fired: List[dict] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.short_circuit is not None


class ProxyFaultEngine:

    def __init__(self, rules: Optional[List[FaultRule]] = None,
                 seed: int = 42, log_path: Optional[str] = None,
                 max_fires: int = 10000):
        self._rules: List[FaultRule] = list(rules or [])
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._log_path = log_path
        self._log_fp = open(log_path, "a", encoding="utf-8") if log_path else None
        self.fire_log: List[dict] = []
        self._total_fires = 0
        self._max_fires = max_fires  # Limit total injected faults.

    def set_rules(self, rules: List[FaultRule]):
        with self._lock:
            self._rules = list(rules)

    def clear_rules(self):
        """Clear rules and counters; keep the audit log."""
        with self._lock:
            self._rules = []
            self._total_fires = 0

    def rules(self) -> List[FaultRule]:
        return list(self._rules)

    def reset_counts(self):
        with self._lock:
            for r in self._rules:
                r._count = 0
            self._total_fires = 0

    def stats(self) -> dict:
        return {
            "total_fires": self._total_fires,
            "rules": [{**r.labels(), "count": r._count,
                       "enabled": r.enabled, "exhausted": r.exhausted()}
                      for r in self._rules],
        }

    def _matches(self, rule: FaultRule, ctx: RequestCtx, body: Any) -> bool:
        m = rule.match
        if m.upstream and m.upstream != ctx.upstream:
            return False
        if m.methods and ctx.method.upper() not in m.methods:
            return False
        if m.path_prefix and not ctx.path.startswith(m.path_prefix):
            return False
        if m.path_regex and not re.search(m.path_regex, ctx.path):
            return False
        for hk, hv in m.header_equals.items():
            if ctx.header(hk) != hv:
                return False
        for hk, hv in m.header_regex.items():
            cur = ctx.header(hk)
            if cur is None or not re.search(hv, cur):
                return False
        for path, expect in m.body_equals.items():
            if jp_get(body, path) != expect:
                return False
        return True

    def _try_fire(self, rule: FaultRule) -> bool:
        """Atomically decide whether this rule fires and update its count."""
        with self._lock:
            if not rule.enabled:
                return False
            if rule.exhausted():
                return False
            if rule.eff_probability < 1.0 and self._rng.random() > rule.eff_probability:
                return False
            rule._count += 1
            # Count but skip the first min_count matches.
            if rule.min_count > 0 and rule._count <= rule.min_count:
                return False
            if self._total_fires >= self._max_fires:
                return False
            self._total_fires += 1
            return True

    def _corrupt_unicode(self, text: str) -> str:
        chars = list(text)
        with self._lock:
            for _ in range(max(1, len(chars) // 5)):
                i = self._rng.randint(0, len(chars) - 1)
                chars[i] = chr(self._rng.randint(0x2600, 0x26FF))
        return "".join(chars)

    def _corrupt(self, text: str, mode: str) -> str:
        if mode == "mojibake":
            try:
                return text.encode("utf-8").decode("latin-1")
            except (UnicodeDecodeError, UnicodeEncodeError):
                return self._corrupt_unicode(text)
        if mode == "broken_json":
            return text[: max(1, len(text) // 2)] + "\x00\x00"
        return self._corrupt_unicode(text)

    def apply(self, stage: str, ctx: RequestCtx, body: Any) -> Outcome:
        out = Outcome(body_value=body)
        for rule in self._rules:
            if rule.stage != stage:
                continue
            if not self._matches(rule, ctx, body):
                continue
            if not self._try_fire(rule):
                continue

            self._record(rule, ctx)
            out.fired.append({**rule.labels(), "count": rule._count,
                              "description": rule.description})

            if rule.action == "abort":
                out.short_circuit = "abort"
                out.status = rule.status
                out.body = rule.body if rule.body is not None else {
                    "error": {"type": "fault_injected", "rule": rule.id,
                              "category": rule.category.value,
                              "message": rule.description or "fault injected by proxy pod"}}
                return out
            if rule.action == "drop":
                out.short_circuit = "drop"
                return out

            if rule.action == "delay":
                out.delay_ms += float(rule.value if isinstance(rule.value, (int, float)) else 2000)
                continue

            if out.body_value is None:
                # Structured edits require a JSON body.
                continue

            if rule.action == "set_json":
                out.body_value = jp_set(_deep(out, body), rule.target_path, rule.value)
                out.body_mutated = True
            elif rule.action == "remove_json":
                out.body_value = jp_remove(_deep(out, body), rule.target_path)
                out.body_mutated = True
            elif rule.action == "corrupt":
                orig = jp_get(out.body_value, rule.target_path)
                if isinstance(orig, str) and orig:
                    mode = rule.value if isinstance(rule.value, str) else "unicode"
                    jp_set(_deep(out, body), rule.target_path, self._corrupt(orig, mode))
                    out.body_mutated = True
            elif rule.action == "truncate":
                orig = jp_get(out.body_value, rule.target_path)
                ratio = rule.value if isinstance(rule.value, (int, float)) and 0 < rule.value < 1 else 0.5
                if isinstance(orig, (str, list)) and orig:
                    jp_set(_deep(out, body), rule.target_path, orig[: max(1, int(len(orig) * ratio))])
                    out.body_mutated = True
            elif rule.action == "duplicate":
                # The server owns the cross-request response cache.
                # Record the duplicate action for the server.
                pass
        return out

    def _record(self, rule: FaultRule, ctx: RequestCtx):
        entry = {
            "t": round(time.time(), 3),
            **rule.labels(),
            "count": rule._count,
            "method": ctx.method,
            "path": ctx.path,
            "upstream": ctx.upstream,
            "description": rule.description,
            "taskId": ctx.taskId
        }
        self.fire_log.append(entry)
        if self._log_fp:
            self._log_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._log_fp.flush()

    def close(self):
        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None


def _deep(out: Outcome, original: Any) -> Any:
    """Copy the body before its first mutation."""
    import copy
    if not out.body_mutated:
        out.body_value = copy.deepcopy(original)
    return out.body_value
