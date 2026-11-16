"""Resume a batch by skipping samples with a terminal result in earlier runs.

Sample identity is semantic (dataset, topology, fault mode, load, sample name)
plus the sample's path relative to its data root.  The semantic key survives a
different ``--root`` or different topology/fault filters; the relative path
disambiguates same-named samples inside one root.
"""
import json
from pathlib import Path

NON_TERMINAL = {None, "running", "pending"}
TERMINAL = {"completed", "round_budget_exhausted", "tool_budget_exhausted",
            "backend_error", "execution_error"}


class ResumeIndex:
    def __init__(self):
        self.executed = set()       # semantic keys
        self.executed_rel = set()   # relpath keys
        self._rel_by_sem = {}       # semantic key -> relpath keys
        self.sources = {}           # semantic key -> mapping path
        self.mappings = []

    def add(self, semantic, relative, source):
        self.executed.add(semantic)
        rels = self._rel_by_sem.setdefault(semantic, set())
        if relative is not None:
            self.executed_rel.add(relative)
            rels.add(relative)
        self.sources.setdefault(semantic, str(source))

    def skip(self, semantic, relative):
        """Return the source path (or True), else None.

        A relative-path match is authoritative.  The semantic key is only a
        fallback when the earlier mapping has no relative path: same-named
        samples in different subdirectories must not be conflated.
        """
        if relative is not None:
            if relative in self.executed_rel:
                return True
            if self._rel_by_sem.get(semantic):
                return None
        if semantic in self.executed:
            return self.sources.get(semantic, True)
        return None


def _terminal_row(row, only_success=False):
    if not isinstance(row, dict):
        return False
    if row.get("status", "failed") not in ("success", "failed"):
        return False
    if only_success and row.get("status") != "success":
        return False
    return row.get("termination") not in NON_TERMINAL


def _terminal_artifact(path, only_success=False):
    """Fallback for batches whose results file is missing."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    termination = data.get("termination")
    if termination in NON_TERMINAL:
        return False
    if only_success:
        return (data.get("evaluation") or {}).get("status") == "success"
    return "evaluation" in data or termination in TERMINAL


def _semantic(entry, dataset):
    name = entry.get("sample_name") or Path(str(entry.get("sample_path", ""))).name
    return (dataset, entry.get("topology_mode"), entry.get("fault_mode"),
            entry.get("load"), name)


def _relative(entry, batch):
    absolute, root = entry.get("sample_path_absolute"), batch.get("data_root_absolute")
    if absolute and root:
        try:
            return Path(absolute).resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return None


def _batch_for(entry, batches):
    absolute = entry.get("sample_path_absolute")
    if absolute:
        for batch in batches:
            root = batch.get("data_root_absolute")
            if not root:
                continue
            try:
                Path(absolute).resolve().relative_to(Path(root).resolve())
                return batch
            except ValueError:
                continue
    return batches[0] if batches else {}


def _result_rows(path):
    """Rows from results-<tag>_<ts>.json next to this batch's samples dir."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    batches = payload.get("batches") or [{}]
    ts = path.parent.name
    base = path.parents[2] if len(path.parents) >= 3 else path.parent
    rows, seen = [], set()
    for batch in batches:
        patterns = [f"results-*_{ts}.json"]
        if batch.get("tag"):
            patterns.insert(0, f"results-{batch['tag']}_{ts}.json")
        for pattern in patterns:
            for results_path in sorted(base.glob(pattern)):
                if results_path in seen:
                    continue
                seen.add(results_path)
                try:
                    data = json.loads(results_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                rows.extend(data.get("rows") or [])
    return rows


def _mapping_paths(output_root=None, sources=None):
    paths = []
    if output_root:
        paths.extend(sorted(Path(output_root).glob("samples/*/mapping.json")))
    for source in sources or []:
        path = Path(source)
        if not path.exists():
            raise ValueError(f"resume source 不存在: {source}")
        if path.is_file() and path.name == "mapping.json":
            paths.append(path)
        elif path.is_dir():
            direct = path / "mapping.json"
            if direct.is_file():
                paths.append(direct)
            else:
                paths.extend(sorted(path.rglob("mapping.json")))
        else:
            raise ValueError(f"resume source 不是 mapping.json 或目录: {source}")
    return sorted({p.resolve() for p in paths})


def load_index(dataset, output_root=None, sources=None, only_success=False):
    """Build the executed-sample index from mapping.json + results/artifacts."""
    index = ResumeIndex()
    paths = _mapping_paths(output_root=output_root, sources=sources)
    index.mappings = [str(path) for path in paths]
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取 resume mapping: {path}: {exc}") from exc
        batches = payload.get("batches") or [{}]
        entries = payload.get("samples") or {}
        rows = _result_rows(path)
        executed_ids = {row.get("sample_id") for row in rows
                        if _terminal_row(row, only_success=only_success)}
        for sample_id, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            batch = _batch_for(entry, batches)
            semantic = _semantic(entry, dataset)
            relative = _relative(entry, batch)
            if sample_id in executed_ids:
                index.add(semantic, (semantic[0], *semantic[1:4], relative)
                          if relative else None, path)
                continue
            artifact = path.parent / str(entry.get("artifact") or "")
            if (entry.get("artifact") and artifact.is_file()
                    and _terminal_artifact(artifact, only_success=only_success)):
                index.add(semantic, (semantic[0], *semantic[1:4], relative)
                          if relative else None, path)
        # Results can outlive a mapping edit; keep terminal rows that no longer
        # have a mapping entry.
        for row in rows:
            if not _terminal_row(row, only_success=only_success) or row.get("sample_id") in entries:
                continue
            semantic = (dataset, row.get("topology_mode"), row.get("fault_mode"),
                        row.get("load"), row.get("case"))
            index.add(semantic, None, path)
    return index


def case_keys(case, root, dataset):
    semantic = (dataset, case.topology_mode, case.fault_mode, case.load, case.name)
    try:
        relative = Path(case.path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        relative = None
    key = (semantic[0], semantic[1], semantic[2], semantic[3], relative) if relative else None
    return semantic, key


def filter_cases(cases, root, dataset, index):
    keep, skipped = [], []
    for case in cases:
        semantic, relative = case_keys(case, root, dataset)
        source = index.skip(semantic, relative)
        if source is None:
            keep.append(case)
            continue
        skipped.append({"case": case.name, "sample_path": str(case.path),
                        "topology_mode": case.topology_mode, "fault_mode": case.fault_mode,
                        "load": case.load, "fault_type": case.label.get("fault_type"),
                        "reason": "already_executed",
                        "source": source if isinstance(source, str) else None})
    return keep, skipped


def summary(index):
    return {"mappings": list(index.mappings),
            "executed_samples": len(index.executed)}
