"""Fault rule models for matching, actions, and execution limits."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Category(str, Enum):
    F1_RUNTIME = "F1_runtime_env"
    F2_SHARED_DEP = "F2_shared_dep_config"
    F3_INTERFACE = "F3_service_interface"
    F4_EXEC_CTRL = "F4_agent_exec_control"


class Scope(str, Enum):
    CALL = "call"
    TASK = "task"
    SERVICE = "service"
    CROSS_SERVICE = "cross_service"


class Persistence(str, Enum):
    SINGLE = "single"
    PERSISTENT = "persistent"
    INTERMITTENT = "intermittent"
    BURST = "burst"


class Propagation(str, Enum):
    LOCAL = "local"
    DOWNSTREAM = "downstream"
    CROSS_TASK = "cross_task"
    CROSS_SERVICE = "cross_service"


class Recovery(str, Enum):
    AUTO = "auto"
    RETRY = "retry"
    CONFIG = "config"
    RESTART = "restart"


_PERSISTENCE_GATE: Dict[Persistence, Dict[str, Any]] = {
    Persistence.SINGLE: {"max_count": 1, "probability": 1.0},
    Persistence.PERSISTENT: {"max_count": sys.maxsize, "probability": 1.0},
    Persistence.INTERMITTENT: {"max_count": sys.maxsize, "probability": 0.3},
    Persistence.BURST: {"max_count": 10, "probability": 1.0},
}


@dataclass
class Match:
    """Match requests by all specified conditions."""
    upstream: Optional[str] = None
    methods: List[str] = field(default_factory=list)
    path_prefix: Optional[str] = None
    path_regex: Optional[str] = None
    header_equals: Dict[str, str] = field(default_factory=dict)
    header_regex: Dict[str, str] = field(default_factory=dict)
    body_equals: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Match":
        d = d or {}
        return Match(
            upstream=d.get("upstream"),
            methods=[m.upper() for m in d.get("methods", [])],
            path_prefix=d.get("path_prefix"),
            path_regex=d.get("path_regex"),
            header_equals={k: str(v) for k, v in (d.get("header_equals") or {}).items()},
            header_regex={k: str(v) for k, v in (d.get("header_regex") or {}).items()},
            body_equals=dict(d.get("body_equals") or {}),
        )


ACTIONS = {
    "abort", "delay", "drop", "set_json", "remove_json",
    "set_query", "remove_query",
    "corrupt", "truncate", "duplicate",
}

STAGES = {"request", "response"}


@dataclass
class FaultRule:
    id: str
    category: Category
    description: str = ""

    scope: Scope = Scope.CALL
    persistence: Persistence = Persistence.SINGLE
    propagation: Propagation = Propagation.LOCAL
    recovery: Recovery = Recovery.AUTO

    match: Match = field(default_factory=Match)

    stage: str = "response"       # request | response
    action: str = "abort"
    target_path: str = "$"
    value: Any = None
    status: int = 500
    body: Optional[Any] = None

    probability: Optional[float] = None
    max_count: Optional[int] = None
    min_count: int = 0
    enabled: bool = True

    _count: int = field(default=0, repr=False)

    @property
    def eff_probability(self) -> float:
        if self.probability is not None:
            return self.probability
        return _PERSISTENCE_GATE[self.persistence]["probability"]

    @property
    def eff_max_count(self) -> int:
        if self.max_count is not None:
            return self.max_count
        return _PERSISTENCE_GATE[self.persistence]["max_count"]

    def exhausted(self) -> bool:
        mc = self.eff_max_count
        return mc > 0 and self._count >= mc

    def labels(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "scope": self.scope.value,
            "persistence": self.persistence.value,
            "propagation": self.propagation.value,
            "recovery": self.recovery.value,
            "stage": self.stage,
            "action": self.action,
        }

    @staticmethod
    def from_dict(d: dict) -> "FaultRule":
        def _enum(enum_cls, val, default):
            if val is None:
                return default
            # Accept short and full category names.
            for member in enum_cls:
                if val == member.value or member.value.startswith(str(val) + "_") \
                        or member.name == val or member.name.split("_")[0] == val:
                    return member
            raise ValueError(f"无法识别 {enum_cls.__name__} 取值: {val!r}")

        stage = d.get("stage", "response")
        if stage not in STAGES:
            raise ValueError(f"stage 必须是 {STAGES}, 收到 {stage!r}")
        action = d.get("action", "abort")
        if action not in ACTIONS:
            raise ValueError(f"action 必须是 {ACTIONS}, 收到 {action!r}")
        if action in {"set_query", "remove_query"}:
            if stage != "request":
                raise ValueError(f"{action} 只能用于 request 阶段")
            target_path = d.get("target_path", "$")
            if not isinstance(target_path, str) or not target_path.startswith("$.") \
                    or not target_path[2:] or any(c in target_path[2:] for c in ".[]"):
                raise ValueError(f"{action} 的 target_path 必须是单个查询参数，如 $.taskId")
            if action == "set_query" and d.get("value") is None:
                raise ValueError("set_query 需要非 null 的 value")

        return FaultRule(
            id=d["id"],
            category=_enum(Category, d.get("category"), Category.F1_RUNTIME),
            description=d.get("description", ""),
            scope=_enum(Scope, d.get("scope"), Scope.CALL),
            persistence=_enum(Persistence, d.get("persistence"), Persistence.SINGLE),
            propagation=_enum(Propagation, d.get("propagation"), Propagation.LOCAL),
            recovery=_enum(Recovery, d.get("recovery"), Recovery.AUTO),
            match=Match.from_dict(d.get("match")),
            stage=stage,
            action=action,
            target_path=d.get("target_path", "$"),
            value=d.get("value"),
            status=int(d.get("status", 500)),
            body=d.get("body"),
            probability=d.get("probability"),
            max_count=d.get("max_count"),
            min_count=int(d.get("min_count", 0)),
            enabled=bool(d.get("enabled", True)),
        )


def load_rules(raw: dict) -> List[FaultRule]:
    """Load rules from a parsed faults list."""
    faults = raw.get("faults") if isinstance(raw, dict) else None
    if not isinstance(faults, list):
        raise ValueError("故障清单需包含顶层 'faults' 列表")
    rules = [FaultRule.from_dict(item) for item in faults]
    ids = [r.id for r in rules]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"故障 id 重复: {sorted(dup)}")
    return rules
