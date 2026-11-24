"""Batch front/back match MicroARCL; the middle is an LLM tool-calling loop."""
import argparse
import contextlib
import json
import time
from pathlib import Path

import results_log
from agent_rca import _service_rank
from .agent import CounterfactualAgent
from .graph_agent import CounterfactualGraphAgent
from .evaluation import evaluate, projection
from .io import ROOT, TOPOLOGY_MODES, FAULT_MODES, discover, load_sample, truth, write_json
from .reporting import summarize, write_sublogs
from .llm import ChatBackend, ReplayBackend
from .rules import LAYERS
from .utils.conversation_log import ConversationLog, error_info
from .utils.graph_events import GraphEventLog
from .utils.sample_mapping import sample_metadata, write_mapping
from .utils.injection_hits import select as select_hit_samples
from .utils import resume as resume_utils


def best_rank(order, targets):
    ranks = [_service_rank(order, target) for target in targets]
    return min((r for r in ranks if r), default=0)


def run_batch(args, backend_factory=None):
    dataset = "MARBLEBench" if args.dataset == "MAR" else args.dataset
    root = Path(args.root) if args.root else ROOT / "benchmark/data" / dataset / "agent_faults/topology_runs"
    discovered = discover(root, args.topology_modes, args.fault_modes)
    if not discovered:
        raise ValueError("没有可用的带标签 sample；检查 --root、--topology-modes 和 --fault-modes")
    cases, skipped = select_hit_samples(discovered, getattr(args, 'skip_unhit_samples', True))
    output_dataset = "MAR" if dataset == "MARBLEBench" else dataset
    resume_info, resume_skipped = None, []
    if getattr(args, 'resume', False) or getattr(args, 'resume_from', None):
        resume_root = results_log.out_root(output_dataset, "MicroCEARCL")
        index = resume_utils.load_index(
            output_dataset,
            output_root=resume_root if getattr(args, 'resume', False) else None,
            sources=getattr(args, 'resume_from', None),
            only_success=getattr(args, 'resume_retry_failed', False))
        cases, resume_skipped = resume_utils.filter_cases(cases, root, output_dataset, index)
        resume_info = resume_utils.summary(index)
        print(f"Resume: {resume_info['executed_samples']} executed sample keys from "
              f"{len(resume_info['mappings'])} mapping(s); skipped {len(resume_skipped)}.",
              flush=True)
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        print("No samples remain after selection/resume; writing an empty batch record.",
              flush=True)
    def factory():
        if backend_factory:
            return backend_factory()
        return ReplayBackend(args.replay) if args.backend == "replay" else ChatBackend(
            args.model, args.base_url, args.timeout, args.max_tokens)
    # Fail fast on missing credentials before creating success-looking result logs.
    # One backend (and HTTP connection pool) serves the whole batch: a fresh client
    # per sample leaked sockets and memory across long runs.
    backend = factory() if cases else None
    rows = []
    filter_tag = "csv_auth_masked" if args.mask_csv_gen_auth else "unmasked"
    tag = args.tag or (f"microcearcl_{args.backend}_{args.evidence_mode}_{filter_tag}_"
                       f"{output_dataset}" + ("_graph" if getattr(args, "agent", "cot") == "graph"
                                              else ""))
    headers = [f"MicroCloudEdgeAgentRootCauseLocalization root={root}",
               f"backend={args.backend}; mode={args.evidence_mode}; "
               f"agent={getattr(args, 'agent', 'cot')}; targets=layer,fault_type",
               f"mask_csv_gen_auth={args.mask_csv_gen_auth}",
               f"evidence_graph={getattr(args, 'evidence_graph', False) or getattr(args, 'agent', 'cot') == 'graph'}",
               "Ground truth is used only after diagnosis; no service-rank metric."]
    if resume_info is not None:
        headers.append(f"resume={len(resume_info['mappings'])} mapping(s); "
                       f"skipped_executed={len(resume_skipped)}")
    with contextlib.ExitStack() as cleanup, \
            results_log.open_batch(output_dataset, "MicroCEARCL", tag, headers) as (out_root, ts):
        if backend is not None and hasattr(backend, "close"):
            cleanup.callback(backend.close)
        out = Path(out_root)
        mapping_path = write_mapping(out / "samples" / ts, cases, dataset=output_dataset, tag=tag, data_root=root)
        mapping_relative = str(mapping_path.relative_to(out))
        print(f"Sample mapping: {mapping_path}")
        selection = {'discovered': len(discovered), 'selected': len(cases), 'skipped': len(skipped),
                     'limit': args.limit, 'skip_unhit_samples': getattr(args, 'skip_unhit_samples', True),
                     'resume': resume_info, 'resume_skipped': len(resume_skipped)}
        write_json(out / f'selection-{tag}_{ts}.json',
                   {'selection': selection, 'skipped_samples': skipped,
                    'resume_skipped': resume_skipped})
        def save_results():
            write_json(out / f"results-{tag}_{ts}.json", {'sample_mapping': mapping_relative,
                'selection': selection, 'skipped_samples': skipped, 'resume_skipped': resume_skipped,
                'rows': rows, 'metrics': evaluate(rows)})
        save_results()
        print(f"Found {len(discovered)} labeled samples; selected={len(cases)}, "
              f"skipped={len(skipped)}, resume_skipped={len(resume_skipped)}.")
        for item in skipped:
            print(f"[skip] {item['case']}: {item['reason']}")
        for item in resume_skipped:
            print(f"[resume-skip] {item['case']}: {item['reason']}")
        for case in cases:
            gt_layers, gt_types = truth(case.label)
            results_log.sample_banner(f"{case.topology_mode}/{case.fault_mode}", case.name,
                                      ",".join(gt_layers), case.label.get("fault_type", ""))
            started = time.perf_counter()
            identity = sample_metadata(case)
            artifact = mapping_path.parent / identity["artifact"]
            conversation = ConversationLog(artifact)
            graph_events = GraphEventLog(artifact)
            result, error = {}, None
            try:
                sample = load_sample(case.path, args.baseline, args.evidence_mode,
                    mask_csv_gen_auth=args.mask_csv_gen_auth, topology_file=getattr(args, 'topology_file', None))
                read_seconds = time.perf_counter() - started
                if getattr(args, "agent", "cot") == "graph":
                    agent = CounterfactualGraphAgent(
                        backend, args.max_rounds, args.max_tool_calls,
                        degree_budget=getattr(args, "evidence_degree_budget", 6),
                        commit_every=args.commit_every,
                        graph_event_sink=graph_events.write)
                else:
                    agent = CounterfactualAgent(
                        backend, args.max_rounds, args.max_tool_calls,
                        commit_every=args.commit_every,
                        evidence_graph=getattr(args, "evidence_graph", False),
                        evidence_degree_budget=getattr(args, "evidence_degree_budget", 6),
                        graph_event_sink=graph_events.write)
                def checkpoint(value):
                    write_json(artifact, {**value, "sample_metadata": identity, "sample_mapping": mapping_relative})
                    last = next(iter(value.get("trajectory", [])[-1:]), {})
                    progress = (last.get('round', 0), value['stage'], value['termination'])
                    if progress != getattr(checkpoint, "last_progress", None):
                        memory = value.get("timing") or {}
                        print(f"[agent] round={progress[0]} stage={value['stage']} "
                              f"candidates={value['candidate_layers']} state={value['termination']} "
                              f"rss={memory.get('rss_mb')}MB peak={memory.get('peak_rss_mb')}MB", flush=True)
                        checkpoint.last_progress = progress
                result = agent.diagnose(sample, checkpoint=checkpoint, event_sink=conversation.emit)
            except Exception as exc:
                error = type(exc).__name__
                conversation.emit("execution_error", {"error": error_info(exc)})
                if artifact.is_file():
                    result = json.loads(artifact.read_text(encoding="utf-8"))
                result["termination"] = "execution_error"
                read_seconds = time.perf_counter() - started
            # Evaluation metadata is attached only after the agent returns.
            # A budget-exhausted run may still carry a salvaged last ordering.
            diagnosis = result.get("diagnosis") or {}
            completed = result.get("termination") == "completed"
            layer_order = diagnosis.get("layer_order", []) if isinstance(diagnosis, dict) else []
            fault_order = diagnosis.get("fault_order", []) if isinstance(diagnosis, dict) else []
            row = {"case": case.name, "load": case.load,
                   "sample_id": identity["sample_id"], "sample_path": identity["sample_path"],
                   "sample_path_absolute": identity["sample_path_absolute"], "sample_mapping": mapping_relative,
                   "fault": case.label.get("fault_type", ""), "category": "+".join(gt_layers) or "unlabeled",
                   "true_layers": gt_layers, "true_types": gt_types,
                   "status": "success" if completed else "failed", "error": error,
                   "termination": result.get("termination", "input_error"),
                   "backend": args.backend, "evidence_mode": args.evidence_mode,
                   "mask_csv_gen_auth": args.mask_csv_gen_auth,
                   "topology_mode": case.topology_mode, "fault_mode": case.fault_mode,
                   "persistence": case.fault_mode,
                   "candidate_layers": result.get("candidate_layers", []),
                   "excluded_layers": result.get("excluded_layers", []),
                   "unsupported_layers": result.get("unsupported_layers", []),
                   "supported_layers": [k for k, v in result.get("layers", {}).items()
                                        if v.get("support_state") == "present"],
                   "vetoed_supported_layers": [k for k, v in result.get("layers", {}).items()
                                               if v.get("state") == "excluded"
                                               and v.get("support_state") == "present"],
                   "parked_layers": result.get("parked_layers", []),
                   "skipped_layers": result.get("skipped_layers", []),
                   "rules_cached": sum(len(v.get("supported_rules", []))
                                       for v in result.get("layers", {}).values()),
                   "rules_pruned": sum(len(entry.get("closed_rules", []))
                                       for entry in result.get("pruning_log", [])),
                   "rounds_to_stage2": (result.get("context") or {}).get("stage_two_round"),
                   "commit_every": args.commit_every,
                   "submit_checkpoints": (result.get("context") or {}).get("submit_checkpoints", 0),
                   "context_max_prompt_tokens": (result.get("context") or {}).get("max_prompt_tokens"),
                   "token_usage": result.get("token_usage") or {},
                   "truncations": (result.get("context") or {}).get("truncations", 0),
                   "layer_order": layer_order, "fault_order": fault_order,
                   "layer_rank": best_rank(layer_order, gt_layers), "fault_rank": best_rank(fault_order, gt_types),
                   "diagnostic_status": diagnosis.get("diagnostic_status"),
                   "unexplained_reason": diagnosis.get("unexplained_reason") if isinstance(diagnosis, dict) else None,
                   "salvaged": bool(diagnosis.get("salvaged")) if isinstance(diagnosis, dict) else False,
                   "timing": {"read": read_seconds, **result.get("timing", {}), "total": time.perf_counter() - started},
                   "artifact": str(artifact.relative_to(out)),
                   "conversation_jsonl": str(conversation.jsonl_path.relative_to(out)),
                   "conversation_log": str(conversation.text_path.relative_to(out)),
                   "graph_events_log": (str(graph_events.path.relative_to(out))
                                        if graph_events.path.is_file() else None)}
            write_json(artifact, {**result, "sample_metadata": identity, "sample_mapping": mapping_relative, "evaluation": row})
            rows.append(row)
            save_results()
            tokens = (row.get("token_usage") or {}).get("total_tokens")
            print(f"[eval] layer rank={row['layer_rank']} fault-type rank={row['fault_rank']} "
                  f"retained={row['candidate_layers']} tokens={tokens} "
                  f"termination={row['termination']}")
        categories = sorted({r["category"] for r in rows})
        for dimension in ("layers", "types"):
            summarize(tag + "_" + dimension, projection(rows, dimension), categories=categories)
        print("\nTwo-dimensional metrics:")
        print(json.dumps(evaluate(rows, False), ensure_ascii=False, indent=2))
    for dimension in ("layers", "types"):
        write_sublogs(out_root, ts, tag + "_" + dimension, projection(rows, dimension), categories=categories)
    write_json(Path(out_root) / f"metrics-{tag}_{ts}.json", evaluate(rows))
    return rows


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("MDOC", "MAR", "MARBLEBench"), default="MDOC")
    ap.add_argument("--root", help="默认 benchmark/data/<dataset>/agent_faults/topology_runs")
    ap.add_argument("--topology-modes", nargs="+", choices=TOPOLOGY_MODES, default=list(TOPOLOGY_MODES), help="第一层：协同模式")
    ap.add_argument("--fault-modes", nargs="+", choices=FAULT_MODES, default=list(FAULT_MODES), help="第二层：故障模式")
    ap.add_argument("--limit", type=int,
                    help="在跳过已执行样本之后，本次最多执行的新样本数")
    ap.add_argument("--resume", action="store_true",
                    help="扫描当前 dataset 的历史 mapping.json，跳过已有终态结果的 sample")
    ap.add_argument("--resume-from", nargs="+",
                    help="指定历史 mapping.json 或 samples/<ts> 目录（可多个）；--root/topology/fault 可不同")
    ap.add_argument("--resume-retry-failed", action="store_true",
                    help="resume 时只跳过成功样本，重跑失败/预算耗尽样本")
    ap.add_argument("--baseline", help="可选，同条件正常 sample，用于请求处理延迟对照")
    ap.add_argument('--topology-file', help='可选，节点到区域/CIDR 的观测元数据 JSON；不写入提示词')
    ap.add_argument('--skip-unhit-samples', action=argparse.BooleanOptionalAction, default=True,
                    help='默认跳过注入命中为 0 或缺少命中计数的 sample；单独记录且不计入准确率')
    ap.add_argument("--backend", choices=("llm", "replay"), default="llm")
    ap.add_argument("--agent", choices=("cot", "graph"), default="cot",
                    help="cot=原 CoT baseline（默认）；graph=图结构引导的新主循环")
    ap.add_argument("--replay", help="离线回放 JSON；仅验证管线，不报告为真实 LLM 结果")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-rounds", type=int, default=32)
    ap.add_argument("--max-tool-calls", type=int, default=100)
    ap.add_argument("--commit-every", type=int, default=5,
                    help="连续 N 轮无 Stage-1 提交时强制一次 submit 检查点；0 关闭周期检查点")
    ap.add_argument("--evidence-mode", choices=("runtime",), default="runtime")
    ap.add_argument("--evidence-graph", action=argparse.BooleanOptionalAction, default=False,
                    help="只读证据图：额外暴露 read_evidence_graph 工具与对应提示词；不改变诊断协议")
    ap.add_argument("--evidence-degree-budget", type=int, default=6,
                    help="证据图中每个观测允许的最大 DEPENDENT 度数；超出降级为 UNKNOWN")
    ap.add_argument("--mask-csv-gen-auth", action=argparse.BooleanOptionalAction, default=True,
                    help="保留已知 csv-gen 数据污染屏蔽；不作为通用诊断分类规则")
    ap.add_argument("--tag")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if args.backend == "replay" and not args.replay:
        ap.error("--backend replay 需要 --replay")
    if any(v <= 0 for v in (args.max_rounds, args.max_tool_calls, args.timeout, args.max_tokens)):
        ap.error("预算和 timeout 必须大于零")
    if args.commit_every < 0:
        ap.error("--commit-every 不能为负数（0 表示关闭周期检查点）")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit 必须大于零")
    if args.tag and (Path(args.tag).name != args.tag or args.tag in (".", "..")):
        ap.error("--tag 必须为单个文件名")
    run_batch(args)


if __name__ == "__main__":
    main()
