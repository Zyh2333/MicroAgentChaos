"""Match and apply HTTP fault rules, then log each injection."""
from __future__ import annotations

import copy
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


_MISSING = object()


@dataclass
class RequestCtx:
    method: str
    path: str
    headers: dict
    upstream: Optional[str] = None
    taskId: Optional[str] = None
    query: list[tuple[str, str]] = field(default_factory=list)
    request_body_bytes: Optional[int] = None
    response_status: Optional[int] = None
    response_content_type: Optional[str] = None
    response_body_bytes: Optional[int] = None

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
    query_value: list[tuple[str, str]] = field(default_factory=list)
    query_mutated: bool = False
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
        self._resp_cache: dict[tuple[Optional[str], str], Any] = {}

    def set_rules(self, rules: List[FaultRule]):
        with self._lock:
            self._rules = list(rules)
            self._resp_cache.clear()

    def clear_rules(self):
        """Clear rules and counters; keep the audit log."""
        with self._lock:
            self._rules = []
            self._total_fires = 0
            self._resp_cache.clear()

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
        out = Outcome(body_value=body, query_value=list(ctx.query))
        cache_key = (ctx.upstream, ctx.path)
        previous_response = None
        if stage == "response":
            with self._lock:
                if cache_key in self._resp_cache:
                    previous_response = copy.deepcopy(self._resp_cache[cache_key])
        replayed = False
        for rule in self._rules:
            if rule.stage != stage:
                continue
            if not self._matches(rule, ctx, body):
                continue
            if not self._try_fire(rule):
                continue

            out.fired.append({**rule.labels(), "count": rule._count,
                              "description": rule.description})
            details = self._apply_action(rule, stage, ctx, body, out, previous_response)
            self._record(rule, ctx, details=details)
            if rule.action == "duplicate" and details["effect_applied"]:
                replayed = True
            if out.is_terminal:
                return out
        if stage == "response" and body is not None and not replayed:
            with self._lock:
                self._resp_cache[cache_key] = copy.deepcopy(body)
        return out

    def _apply_action(self, rule: FaultRule, stage: str, ctx: RequestCtx, original: Any,
                      out: Outcome, previous_response: Any) -> dict:
        action = rule.action
        details = {"effect_applied": False}

        if action in {"abort", "drop"}:
            before = {"body": copy.deepcopy(out.body_value)}
            if stage == "request":
                before["query"] = list(out.query_value)
            else:
                before["status"] = ctx.response_status
            out.short_circuit = action
            if action == "abort":
                out.status = rule.status
                out.body = rule.body if rule.body is not None else {
                    "error": {"type": "fault_injected", "rule": rule.id,
                              "category": rule.category.value,
                              "message": rule.description or "fault injected by proxy pod"}}
            else:
                out.status = 504
                message = "connection dropped (timeout)" if stage == "request" else "response dropped (timeout)"
                out.body = {"error": {"type": "fault_injected", "message": message},
                            "fired": copy.deepcopy(out.fired)}
            response_body = (copy.deepcopy(out.body) if isinstance(out.body, (dict, list))
                             else str(out.body) if out.body is not None else "")
            details.update(effect_applied=True, before=before,
                           after={"status": out.status, "body": response_body},
                           upstream_called=stage == "response")
            return details

        if action == "delay":
            delay = float(rule.value if isinstance(rule.value, (int, float)) else 2000)
            before = out.delay_ms
            out.delay_ms += delay
            details.update(effect_applied=delay > 0, before={"delay_ms": before},
                           after={"delay_ms": out.delay_ms}, requested_delay_ms=delay)
            if delay <= 0:
                details["skip_reason"] = "non_positive_delay"
            return details

        if action in {"set_query", "remove_query"}:
            before = list(out.query_value)
            key = rule.target_path[2:]
            if action == "remove_query":
                out.query_value = [(k, v) for k, v in before if k != key]
            else:
                value = str(rule.value)
                new_query = []
                replaced = False
                for k, v in before:
                    if k == key:
                        if not replaced:
                            new_query.append((key, value))
                            replaced = True
                    else:
                        new_query.append((k, v))
                if not replaced:
                    new_query.append((key, value))
                out.query_value = new_query
                details["configured_value"] = value
            out.query_mutated = True
            details.update(effect_applied=before != out.query_value,
                           target_path=rule.target_path, before=before,
                           after=list(out.query_value))
            if before == out.query_value:
                details["skip_reason"] = "target_missing" if action == "remove_query" else "value_unchanged"
            return details

        if action not in {"set_json", "remove_json", "corrupt", "truncate", "duplicate"}:
            return details

        if action == "duplicate":
            details.update(before=copy.deepcopy(out.body_value),
                           after=copy.deepcopy(out.body_value),
                           cache_hit=previous_response is not None)
            if out.body_value is None:
                details["skip_reason"] = "json_body_unavailable"
            elif stage != "response" or previous_response is None:
                details["skip_reason"] = "cache_miss" if stage == "response" else "request_stage"
            else:
                out.body_value = copy.deepcopy(previous_response)
                out.body_mutated = True
                details.update(effect_applied=True, after=copy.deepcopy(out.body_value),
                               content_changed=details["before"] != details["after"])
            return details

        details["target_path"] = rule.target_path
        before = _target_state(out.body_value, rule.target_path)
        details["before"] = before
        if out.body_value is None:
            details.update(after=before, skip_reason="json_body_unavailable")
            if action == "truncate":
                details["truncation_applied"] = False
            return details

        if action == "set_json":
            out.body_value = jp_set(_deep(out, original), rule.target_path, rule.value)
            out.body_mutated = True
            details["configured_value"] = copy.deepcopy(rule.value)
        elif action == "remove_json":
            out.body_value = jp_remove(_deep(out, original), rule.target_path)
            out.body_mutated = True
        elif action == "corrupt":
            value = before["value"]
            mode = rule.value if isinstance(rule.value, str) else "unicode"
            details["mode"] = mode
            if isinstance(value, str) and value:
                jp_set(_deep(out, original), rule.target_path, self._corrupt(value, mode))
                out.body_mutated = True
            else:
                details["skip_reason"] = "target_not_nonempty_string"
        elif action == "truncate":
            value = before["value"]
            ratio = rule.value if isinstance(rule.value, (int, float)) and 0 < rule.value < 1 else 0.5
            details["ratio"] = ratio
            if isinstance(value, (str, list)) and value:
                shortened = value[: max(1, int(len(value) * ratio))]
                if shortened != value:
                    jp_set(_deep(out, original), rule.target_path, shortened)
                    out.body_mutated = True
            else:
                details["skip_reason"] = "target_not_nonempty_string_or_list"

        after = _target_state(out.body_value, rule.target_path)
        changed = before != after
        details.update(after=after, effect_applied=changed, content_changed=changed)
        if action == "truncate":
            details.update(truncation_applied=changed)
            if isinstance(before["value"], (str, list)):
                details.update(content_before=before["value"], content_after=after["value"],
                               length_before=len(before["value"]), length_after=len(after["value"]))
        if not changed and "skip_reason" not in details:
            details["skip_reason"] = "value_unchanged" if after["exists"] else "target_missing"
        return details

    def _record(self, rule: FaultRule, ctx: RequestCtx, details: Optional[dict] = None):
        entry = {
            "t": round(time.time(), 3),
            **rule.labels(),
            "count": rule._count,
            "method": ctx.method,
            "path": ctx.path,
            "upstream": ctx.upstream,
            "description": rule.description,
            "taskId": ctx.taskId,
            "request_query": list(ctx.query),
            "request_content_type": ctx.header("Content-Type"),
            "request_body_bytes": ctx.request_body_bytes,
            "response_status": ctx.response_status if rule.stage == "response" else None,
            "response_content_type": ctx.response_content_type if rule.stage == "response" else None,
            "response_body_bytes": ctx.response_body_bytes if rule.stage == "response" else None,
            "routing_headers": {name: value for name in (
                "X-Upstream", "X-Upstream-Url", "X-Agent-Target-Zone",
                "X-Agent-Degrade-Stage", "X-Forwarded-For")
                if (value := ctx.header(name)) is not None},
            "effective_probability": rule.eff_probability,
            "effective_max_count": rule.eff_max_count,
        }
        if details:
            entry.update(details)
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
    if not out.body_mutated:
        out.body_value = copy.deepcopy(original)
    return out.body_value


def _target_state(body: Any, path: str) -> dict:
    if body is None:
        return {"exists": False, "value": None}
    value = jp_get(body, path, _MISSING)
    return {"exists": value is not _MISSING,
            "value": copy.deepcopy(value) if value is not _MISSING else None}
