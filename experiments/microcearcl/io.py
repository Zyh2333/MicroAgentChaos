"""MicroARCL-compatible discovery, KPI input, label isolation and output paths."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

from eval_utils import parse_labels
from .core import Sample

ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_MODES = ("cloud-preference", "edge-preference", "dynamic")
FAULT_MODES = ("persistent", "intermittent", "burst")


@dataclass
class Case:
    name: str
    path: Path
    topology_mode: str
    fault_mode: str
    load: str
    label: dict  # evaluation-only: never passed to Sample or the LLM


def discover(root, topology_modes=None, fault_modes=None, limit=None):
    """Same *_label.txt block parser and sorted sibling-sample discovery.

    Recursive discovery additionally supports current topology_runs/load/group/
    persistence layouts. Aggregate labels are deduplicated by the real sample.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"数据根目录不存在: {root}")
    cases = {}
    topology_modes = set(TOPOLOGY_MODES if topology_modes is None else topology_modes)
    fault_modes = set(FAULT_MODES if fault_modes is None else fault_modes)
    if topology_modes - set(TOPOLOGY_MODES) or fault_modes - set(FAULT_MODES):
        raise ValueError("协同模式或故障模式不受支持")
    parents = sorted({p.parent for p in root.rglob("*_label.txt")})
    for parent in parents:
        for name, label in sorted(parse_labels(str(parent)).items()):
            if Path(name).name != name or name in (".", ".."):
                continue
            candidates = [parent / name]
            persistence = label.get("persistence", "")
            if persistence and Path(persistence).name == persistence:
                candidates.append(parent / persistence / name)
            for path in candidates:
                if not path.is_dir():
                    continue
                load = next((p.name for p in path.parents if p.name.startswith("load-")), "unrecorded")
                def mode(field, choices):
                    parts = (root.name, *path.relative_to(root).parts[:-1])
                    inferred = next((part for part in reversed(parts) if part in choices), None)
                    explicit = label.get(field)
                    if explicit in choices and inferred and explicit != inferred:
                        raise ValueError(f"{field} 标签与目录不一致")
                    return explicit if explicit and explicit != "unrecorded" else inferred
                topology = mode("topology_mode", TOPOLOGY_MODES)
                fault_mode = mode("persistence", FAULT_MODES)
                key = path.resolve()
                if key in cases:
                    old = cases[key].label
                    for field in ("fault_type", "service", "fault_layer", "topology_mode", "persistence"):
                        if old.get(field) and label.get(field) and old[field] != label[field]:
                            raise ValueError("重复标签冲突，不能静默选择真实答案")
                    old.update(label)
                    cases[key].topology_mode = topology or cases[key].topology_mode
                    cases[key].fault_mode = fault_mode or cases[key].fault_mode
                else:
                    cases[key] = Case(name, path, topology, fault_mode, load, dict(label))
    result = sorted((c for c in cases.values() if c.topology_mode in topology_modes and c.fault_mode in fault_modes),
                    key=lambda c: (c.topology_mode, c.fault_mode, c.load, str(c.path)))
    return result[:limit] if limit is not None else result


def load_sample(path, baseline=None, mode="runtime", namespace="agent-network", mask_csv_gen_auth=True,
                topology_file=None):
    """Reuse MicroARCL's exact primary KPI loader; other modalities stay lazy."""
    sample = Sample(Path(path), baseline=baseline, mode=mode, namespace=namespace,
                    mask_csv_gen_auth=mask_csv_gen_auth, topology_file=topology_file)
    metrics = sample.ns("metrics")
    files = ("success_rate.csv", "latency.csv", "svc_qps.csv")
    if all((metrics / f).is_file() for f in files):
        from agent_rca import load_kpi
        try:
            frames = load_kpi(str(metrics))
            sample.cache["kpi"] = frames
            sample.cache["services"] = [c for c in frames[0].columns if c != "timestamp"]
        except Exception as exc:
            sample.cache["services"] = []
            sample.warn(f"KPI 读取失败：{type(exc).__name__}；继续使用其他模态，不推断指标正常")
    else:
        sample.cache["services"] = []
        sample.warn("三份主要 KPI CSV 不齐；保留未知，不伪造正常指标")
    return sample


def sample_key(case):
    return hashlib.sha256(str(case.path).encode()).hexdigest()[:16]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Stream the indented encoder into the file. json.dumps(indent=...) is the pure-Python
    # encoder and materializes one chunk list for the whole document; on the repeated,
    # growing checkpoints of a long agent run that fragments memory into hundreds of MB.
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _values(raw):
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, TypeError):
        pass
    return [v for v in re.split(r"[,;\s]+", str(raw)) if v]


def truth(label):
    """Only explicit layer/type labels; legacy cpu/mem do not acquire invented types."""
    from .rules import TYPES, LAYERS
    kinds = _values(label.get("fault_types") or label.get("fault_type"))
    known = {f"{layer.lower()}_{kind}" for layer, values in TYPES.items() for kind in values}
    types = sorted({k.lower() for k in kinds if isinstance(k, str) and k.lower() in known})
    layers = {k[:2].upper() for k in types}
    for value in _values(label.get("fault_layers") or label.get("fault_layer") or label.get("layer")):
        if isinstance(value, str) and value.upper() in LAYERS:
            layers.add(value.upper())
    return sorted(layers), types
