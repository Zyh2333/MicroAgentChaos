"""Load the declarative fault-domain registry. The engine contains no layer text.

Rules live in rules.yaml (or MICROCEARCL_RULES).  R (r1-r4) are support rules,
CF (cf1) is a necessary-signature rule and C (c1-c2) are controls.  Adding a
layer or type is a registry edit; no code change is required.
"""
import os
from pathlib import Path

import yaml

KEYS = ("r1", "r2", "r3", "r4", "cf1", "c1", "c2")
PREDICTION_KEYS = ("r1", "r2", "r3", "r4")
COUNTERFACTUAL_KEYS = ("cf1",)
CONTROL_KEYS = ("c1", "c2")
DEFAULT_REGISTRY = Path(__file__).with_name("rules.yaml")


def _require(condition, path, message):
    if not condition:
        raise ValueError(f"规则注册表 {path}: {message}")


def load_registry(path=None):
    """Load and validate a registry; return layer/type/rule mappings."""
    file = Path(path or os.environ.get("MICROCEARCL_RULES") or DEFAULT_REGISTRY)
    _require(file.is_file(), file, "文件不存在")
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"规则注册表 {file}: YAML 解析失败: {error}") from error
    _require(isinstance(data, dict), file, "顶层必须是对象")
    entries = data.get("layers")
    _require(isinstance(entries, list) and entries, file, "layers 必须为非空数组")
    layers, names, types, rules, signatures, seen_types = [], {}, {}, {}, {}, set()
    for index, entry in enumerate(entries):
        where = f"layers[{index}]"
        _require(isinstance(entry, dict), file, f"{where} 必须是对象")
        layer = entry.get("id")
        _require(isinstance(layer, str) and layer.strip(), file, f"{where}.id 必须为非空字符串")
        _require(layer not in names, file, f"重复层级 {layer}")
        name = entry.get("name")
        _require(isinstance(name, str) and name.strip(), file, f"{layer}.name 必须为非空字符串")
        layer_types = entry.get("types")
        _require(isinstance(layer_types, list) and layer_types
                 and all(isinstance(x, str) and x.strip() for x in layer_types),
                 file, f"{layer}.types 必须为非空字符串数组")
        for kind in layer_types:
            _require(kind not in seen_types, file, f"重复类型 {kind}")
            seen_types.add(kind)
        rule_text = entry.get("rules")
        _require(isinstance(rule_text, dict) and set(rule_text) == set(KEYS),
                 file, f"{layer}.rules 必须恰好包含 {', '.join(KEYS)}")
        for key in KEYS:
            _require(isinstance(rule_text[key], str) and rule_text[key].strip(),
                     file, f"{layer}.{key} 必须为非空字符串")
        # Type signatures are optional for custom registries, but when present
        # they must cover exactly the declared types and keep the same shape.
        signature_map = entry.get("type_signatures")
        if signature_map is not None:
            _require(isinstance(signature_map, dict), file,
                     f"{where}.type_signatures 必须是对象")
            _require(set(signature_map) == set(layer_types), file,
                     f"{where}.type_signatures 必须覆盖且仅覆盖 types: "
                     f"{', '.join(layer_types)}")
            cleaned = {}
            for kind, signature in signature_map.items():
                _require(isinstance(signature, dict), file,
                         f"{where}.{kind} 签名必须是对象")
                description = signature.get("description")
                _require(isinstance(description, str) and description.strip(), file,
                         f"{where}.{kind}.description 必须为非空字符串")
                phenomena = signature.get("phenomena")
                _require(isinstance(phenomena, list) and phenomena
                         and all(isinstance(v, str) and v.strip() for v in phenomena),
                         file, f"{where}.{kind}.phenomena 必须为非空字符串数组")
                cleaned[kind] = dict(signature)
            signatures[layer] = cleaned
        else:
            signatures[layer] = {}
        layers.append(layer)
        names[layer] = name
        types[layer] = tuple(layer_types)
        rules[layer] = [rule_text[key] for key in KEYS]
    boundary = data.get("boundary_types", [])
    _require(isinstance(boundary, list)
             and all(isinstance(x, str) and x.strip() for x in boundary),
             file, "boundary_types 必须为字符串数组")
    _require(set(boundary) <= seen_types, file, "boundary_types 必须来自已声明类型")
    return {"path": str(file), "layers": tuple(layers), "names": names,
            "types": types, "rules": rules, "type_signatures": signatures,
            "boundary_types": tuple(boundary)}


_REGISTRY = load_registry()
LAYERS = _REGISTRY["layers"]
LAYER_NAMES = _REGISTRY["names"]
TYPES = _REGISTRY["types"]
RULES = _REGISTRY["rules"]
TYPE_SIGNATURES = _REGISTRY["type_signatures"]
BOUNDARY_TYPES = set(_REGISTRY["boundary_types"])


def registry():
    return _REGISTRY


def type_signature(layer, kind):
    """Signature for one layer/type, or ``None`` when none is configured."""
    return TYPE_SIGNATURES.get(layer, {}).get(kind)


def rule_id(layer, key):
    return f"{layer}-{key.upper()}"


def predictions(layer):
    return tuple(rule_id(layer, key) for key in PREDICTION_KEYS)


def counterfactuals(layer):
    return tuple(rule_id(layer, key) for key in COUNTERFACTUAL_KEYS)


def controls(layer):
    return tuple(rule_id(layer, key) for key in CONTROL_KEYS)


def rule_lookup():
    return {rule_id(layer, key): {"layer": layer, "key": key, "description": text}
            for layer in LAYERS for key, text in zip(KEYS, RULES[layer])}


def matrix():
    return [{"id": rule_id(layer, key), "layer": layer, "description": text}
            for layer in LAYERS for key, text in zip(KEYS, RULES[layer])]


def contract():
    """Human-readable layer contract for the exploration prompt.

    Only mechanism definitions from this registry are exposed.  No sample paths,
    directory names, injection labels or dataset-specific log strings are read.
    """
    lines = ["故障域定义（R 为支持规则，CF 为必要性签名，C 为对照）："]
    for layer in LAYERS:
        lines.append(f"{layer} {LAYER_NAMES[layer]}（可诊断类型：{', '.join(TYPES[layer])}）")
        for key, text in zip(KEYS, RULES[layer]):
            lines.append(f"  {rule_id(layer, key)}: {text}")
    return "\n".join(lines)


def type_contract():
    """Mechanism plus typical phenomena for every declared type.

    The rest (counter-signatures, boundaries, distinctions) is left to the
    agent; the registry only says what the fault *is* and what it usually
    looks like.
    """
    lines = ["故障类型说明（description 为机制，phenomena 为典型现象）："]
    for layer in LAYERS:
        for kind in TYPES[layer]:
            signature = type_signature(layer, kind)
            if not signature:
                continue
            lines.append(f"{layer.lower()}_{kind}：{signature['description']}")
            lines.append("  典型现象: " + "；".join(signature.get("phenomena") or ()))
    return "\n".join(lines)
