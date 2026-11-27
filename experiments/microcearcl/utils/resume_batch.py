"""Update an explicitly selected batch, keeping one current result per sample."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import json
import sys
import time

import results_log
from . import resume
from .conversation_log import timestamp
from .sample_mapping import sample_metadata
from ..io import write_json


class ResumeBatch:
    def __init__(self, sources, dataset):
        # The first source is the write destination; further sources only inform
        # skip selection. A broad directory must identify one destination.
        paths = resume._mapping_paths(sources=sources[:1])
        if len(paths) != 1:
            raise ValueError("原地续跑的第一个 --resume-from 必须指向一个 mapping.json 或单个批次目录")
        self.mapping_path = paths[0]
        self.folder = self.mapping_path.parent
        if self.folder.parent.name != "samples":
            raise ValueError("原地续跑需要 samples/<批次>/mapping.json 布局")
        self.out = self.folder.parents[1]
        self.ts = self.folder.name
        self.mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        batches = self.mapping.get("batches") or []
        if not batches or any(batch.get("dataset") != dataset for batch in batches):
            raise ValueError("--resume-from 的数据集与 --dataset 不一致")
        tags = {batch["tag"] for batch in batches if batch.get("tag")}
        if len(tags) != 1:
            raise ValueError("原地续跑需要批次具有唯一 tag")
        self.tag = tags.pop()
        self.results_path = self.out / f"results-{self.tag}_{self.ts}.json"
        self.payload = (json.loads(self.results_path.read_text(encoding="utf-8"))
                        if self.results_path.is_file() else {})
        by_id = {row["sample_id"]: row for row in self.payload.get("rows", [])}
        # Recover evaluated artifacts when an interruption occurred between the
        # atomic artifact write and the aggregate write.
        for sid, entry in self.mapping.get("samples", {}).items():
            artifact = self.folder / (entry.get("artifact") or f"{sid}.json")
            if artifact.is_file():
                value = json.loads(artifact.read_text(encoding="utf-8"))
                if value.get("evaluation"):
                    by_id[sid] = value["evaluation"]
                elif value.get("termination") in resume.NON_TERMINAL:
                    by_id.pop(sid, None)
        self.rows = list(by_id.values())

    def identity(self, case, root, dataset):
        """Keep the old artifact id even when the input root has moved."""
        identity = sample_metadata(case)
        semantic, relative = resume.case_keys(case, root, dataset)
        batches = self.mapping["batches"]
        for sid, entry in self.mapping.get("samples", {}).items():
            if resume._semantic(entry, dataset) != semantic:
                continue
            old_relative = resume._relative(entry, resume._batch_for(entry, batches))
            if relative is not None and old_relative is not None and relative[-1] != old_relative:
                continue
            identity.update(sample_id=sid, artifact=entry.get("artifact") or f"{sid}.json",
                            conversation_jsonl=entry.get("conversation_jsonl") or f"{sid}.conversation.jsonl",
                            conversation_log=entry.get("conversation_log") or f"{sid}.conversation.log")
            return identity
        if identity["sample_id"] in self.mapping.get("samples", {}):
            raise ValueError("样本短哈希冲突，不能覆盖已有 mapping")
        return identity

    def selection(self, current, skipped):
        """Keep batch-wide coverage and expose this invocation separately."""
        previous = self.payload.get("selection") or {}
        entries = json.loads(self.mapping_path.read_text(encoding="utf-8"))["samples"]
        included_paths = {entry["sample_path"] for entry in entries.values()}
        combined = {item["sample_path"]: item
                    for item in self.payload.get("skipped_samples", []) + skipped
                    if item["sample_path"] not in included_paths}
        selected = {**previous, **current,
                    "discovered": max(previous.get("discovered", 0), current["discovered"]),
                    "selected": len(entries),
                    "skipped": len(combined), "current_run": deepcopy(current)}
        return selected, list(combined.values())

    def reset_sample(self, identity):
        """Clear a superseded attempt before tools start, including stale errors."""
        artifact = self.folder / identity["artifact"]
        # An atomic pending marker makes the artifact authoritative even if the
        # process stops before the old aggregate row has been removed.
        write_json(artifact, {"termination": "pending", "sample_metadata": identity,
                              "sample_mapping": str(self.mapping_path.relative_to(self.out))})
        for path in (self.folder / identity["conversation_jsonl"],
                     self.folder / identity["conversation_log"],
                     artifact.with_suffix(".graph_events.jsonl")):
            path.unlink(missing_ok=True)

    @contextmanager
    def open(self, headers):
        log_path = self.out / f"batch-{self.tag}_{self.ts}.log"
        started = time.perf_counter()
        with log_path.open("a", encoding="utf-8") as output, \
                redirect_stdout(results_log._Tee(sys.stdout, output)), \
                redirect_stderr(results_log._Tee(sys.stderr, output)):
            print(f"\nResume start: {timestamp()} (in place)")
            print(f"Overall log: {log_path}")
            for line in headers:
                print(line)
            try:
                yield self.out, self.ts
            finally:
                print(f"\nResume wall-clock: {time.perf_counter() - started:.1f}s")
