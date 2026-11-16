"""Shared observation contract. Never discover labels or injection configuration."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import math


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reference(file, location, **details):
    return {"file": str(file), "location": str(location), **details}


def evidence(refs=(), reason="证据不足", value=None, **details):
    return {"value": value, "reason": reason, "count": len(refs),
            "references": list(refs)[:8], "details": details}


def observed(refs, reason, **details):
    return evidence(refs, reason, 1 if refs else None, **details)


def successful(event):
    return number(event.get("status")) and 200 <= event["status"] < 300


def failed(event):
    return number(event.get("status")) and event["status"] >= 400


def same_zone(event):
    a, b = event.get("source_zone"), event.get("target_zone")
    return a == b if a and b else None


def content(event, key):
    value = event.get(key)
    if isinstance(value, str):
        return value
    return None


def query_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {x[0]: x[1] for x in value if isinstance(x, (list, tuple)) and len(x) == 2}
    return None


@dataclass
class Sample:
    path: Path
    baseline: Path | None = None
    mode: str = "runtime"
    start: float | None = None
    end: float | None = None
    cache: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    namespace: str = "agent-network"
    mask_csv_gen_auth: bool = True
    topology_file: Path | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        self.baseline = Path(self.baseline) if self.baseline else None
        if not self.path.is_dir():
            raise ValueError(f"sample 目录不存在: {self.path}")
        if self.baseline and not self.baseline.is_dir():
            raise ValueError(f"baseline 目录不存在: {self.baseline}")
        if self.mode not in ("runtime", "proxy-audit"):
            raise ValueError("mode must be runtime or proxy-audit")
        if any(v is not None and not number(v) for v in (self.start, self.end)):
            raise ValueError("时间窗口必须使用有限 epoch 秒")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("start 必须小于 end")

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    def json(self, relative, default=None):
        file = self.path / relative
        if not file.is_file():
            return default
        try:
            return json.loads(file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            self.warn(f"{relative}: {type(error).__name__}")
            return default

    def in_window(self, timestamp):
        if self.start is None and self.end is None:
            return True
        if not number(timestamp):
            self.warn("存在无绝对时间的记录；指定窗口时不将其作为窗内证据")
            return False
        return ((self.start is None or timestamp >= self.start)
                and (self.end is None or timestamp < self.end))

    def complete(self, kind, events):
        """Coverage is an explicit collector contract, never inferred from file presence.

        It applies only to full-run canonical telemetry, never fault-hit logs.
        Counts and scope must match; malformed input invalidates this contract.
        """
        if self.mode != "runtime" or self.start is not None or self.end is not None:
            return False
        mask = self.cache.get("csv_auth_mask")
        if mask and mask.enabled and (mask.known_background_present or any(mask.records.values())):
            return False  # Suppressed errors cannot become evidence that a boundary is healthy.
        coverage = self.json("observations/coverage.json", {})
        entry = coverage.get(kind, {}) if isinstance(coverage, dict) else {}
        native = [e for e in events if e.get("origin") == "telemetry" and e.get("kind") == kind]
        return (not self.cache.get("telemetry_invalid", False)
                and entry.get("complete") is True and entry.get("scope") == "whole_sample"
                and type(entry.get("expected_records")) is int
                and entry["expected_records"] == len(native) and len(native) > 0)

    def normal(self):
        if "baseline" not in self.cache:
            self.cache["baseline"] = Sample(self.baseline, namespace=self.namespace,
                                             mask_csv_gen_auth=self.mask_csv_gen_auth,
                                             topology_file=self.topology_file) if self.baseline else None
        return self.cache["baseline"]

    def ns(self, suffix):
        return self.path / self.namespace / suffix


def layer_result(facts, mechanisms=None, healthy=None):
    """CF tests the no-fault hypothesis; its violation supports the faulty layer.

    A failed support predicate is unknown, not contradictory. Only verified,
    complete boundary observations can supply a contradiction (value=0).
    """
    positives = [v for k, v in facts.items() if k.startswith("r") and v["value"] == 1]
    if healthy is not None:
        facts["cf1"] = healthy
    elif positives:
        facts["cf1"] = evidence([r for v in positives for r in v["references"]],
                                "观测违反该层无故障假设；支持故障层，不是排除故障层", 1)
    else:
        facts["cf1"] = evidence(reason="无故障假设尚未被证伪；不等于该层已排除")
    return {"facts": facts, "mechanisms": mechanisms or {}}
