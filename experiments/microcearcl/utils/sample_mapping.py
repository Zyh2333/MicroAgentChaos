"""Human-readable sample identities for hashed artifacts; never sent to the LLM."""
import json
from pathlib import Path
from ..io import sample_key, write_json


def sample_metadata(case):
    key = sample_key(case)
    return {"sample_id": key, "sample_name": case.name, "sample_path": str(case.path),
            "sample_path_absolute": str(case.path.resolve()),
            "topology_mode": case.topology_mode, "fault_mode": case.fault_mode, "load": case.load,
            "fault_type": case.label.get("fault_type"),
            "artifact": f"{key}.json", "conversation_jsonl": f"{key}.conversation.jsonl",
            "conversation_log": f"{key}.conversation.log"}


def write_mapping(folder, cases, *, dataset, tag, data_root, identities=None, variant=None):
    """Write the full planned sample list before diagnosing the first sample."""
    path = Path(folder) / "mapping.json"
    payload = (json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": 1, "hash_scheme": "sha256(str(case.path).encode('utf-8'))[:16]",
        "artifact_paths_relative_to": "mapping.json parent directory", "batches": [], "samples": {}})
    if variant is not None:
        payload["agent_variant"] = variant
    batch = {"dataset": dataset, "tag": tag, "batch_id": path.parent.name,
             "data_root": str(data_root), "data_root_absolute": str(Path(data_root).resolve())}
    if batch not in payload["batches"]:
        payload["batches"].append(batch)
    for entry in identities if identities is not None else map(sample_metadata, cases):
        key = entry["sample_id"]
        existing = payload["samples"].get(key)
        if existing and existing["sample_path"] != entry["sample_path"] and identities is None:
            raise ValueError("样本短哈希冲突，不能覆盖已有 mapping")
        payload["samples"][key] = entry
    write_json(path, payload)
    return path
