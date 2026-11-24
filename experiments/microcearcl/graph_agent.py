"""State-driven graph agent.

The CoT baseline ``CounterfactualAgent`` decides what to read from the transcript.
This agent pushes the evidence graph's semantic state into every round:

* each open proposition carries the records it cites -- content, producer, time
  and relation (same / dependent / single) -- plus independent neighbours;
* the model sees all open propositions at once, advances any layer, and calls
  any available tool;
* constraints live on the result side: bundle-counted support, independent
  controls for refutations, and the query ledger for redundancy;
* Stage-1 submission uses the baseline's checkpoint unchanged: deadline or
  periodic after ``commit_every`` rounds without a commit, regardless of graph
  activity.

The model keeps the part it is uniquely good at: deciding what a record means and
which rule it supports.  Stage 2 keeps the baseline protocol untouched.  The pull
tool ``read_evidence_graph`` belongs to the CoT ablation.
"""
import hashlib
import json
import time
from copy import deepcopy

from experiments.evidence_assoc import planner as graph_planner
from experiments.evidence_assoc.adapters import claims_from_layers, expand_ids
from experiments.evidence_assoc.evolution import (DEFAULT_COMBINATION_SIZE,
                                                   DEFAULT_MAX_COMBINATIONS, ActivityLedger)
from experiments.evidence_assoc.propositions import assess
from experiments.evidence_assoc.view import graph_state, summary

from .agent import CounterfactualAgent, _frame
from .prompts import GRAPH_CONTRIBUTION_SYSTEM, GRAPH_SYSTEM
from .registry import DIAGNOSES, READERS, SUBMITS, Registry
from .rules import LAYERS, counterfactuals, controls as layer_controls, rule_lookup
from .utils import worklist
from .utils.background_noise import POLICY, policy as background_policy
from .utils.conversation_log import error_info, timestamp
from .utils.evidence_graph import EvidenceGraphBridge
from .utils import finish_layer_stage
from .utils.finish_diagnosis import ValidationErrors, salvage
from .utils.process_response import run as process_response
from .utils.context_projection import project_messages
from .utils.transcript_budget import bounded, size as json_size


class CounterfactualGraphAgent(CounterfactualAgent):
    def __init__(self, backend, max_rounds=32, max_tool_calls=100, degree_budget=6,
                 commit_every=5, graph_event_sink=None, state_view="semantic",
                 state_budget=12000, project_transcript=False, context_budget=24000,
                 m2_combination_size=DEFAULT_COMBINATION_SIZE,
                 m2_max_combinations=DEFAULT_MAX_COMBINATIONS):
        super().__init__(backend, max_rounds=max_rounds, max_tool_calls=max_tool_calls,
                         commit_every=commit_every, evidence_graph=True,
                         evidence_degree_budget=degree_budget,
                         graph_event_sink=graph_event_sink)
        self.prompt = GRAPH_CONTRIBUTION_SYSTEM if state_view == "contribution" else GRAPH_SYSTEM
        self.state_view = state_view
        self.state_budget = state_budget
        self.m2_combination_size = m2_combination_size
        self.m2_max_combinations = m2_max_combinations
        self.project_transcript = project_transcript
        self.context_budget = context_budget

    # ------------------------------------------------------------------ graph
    def _graph(self):
        return self.evidence_graph.associator.graph

    def _resolve(self, identifiers):
        return expand_ids(self._graph(), identifiers or [])

    def _schemas(self, keep=None, stage=None):
        """B2 pushes the semantic state; the pull tool belongs to the B1 ablation."""
        schemas = super()._schemas(keep=keep, stage=stage)
        return [schema for schema in schemas
                if schema["function"]["name"] != "read_evidence_graph"]

    def _state(self, round_no):
        """Semantic digest of the open propositions, pushed every round."""
        graph = self._graph()
        # Parked, excluded and unsupported layers keep their citations in the ledger.
        open_layers = {layer: record for layer, record in self.assessments.items()
                       if (record or {}).get("state") not in
                       worklist.NONBLOCKING_STATES}
        claims, _ = claims_from_layers(open_layers, graph)
        propositions = assess(graph, claims)
        cf_rules = {layer: list(counterfactuals(layer)) for layer in LAYERS}
        control_rules = {layer: list(layer_controls(layer)) for layer in LAYERS}
        ranked = self.activity.ranked_propositions(
            propositions, rule_text=rule_lookup(), cf_rules=cf_rules,
            control_rules=control_rules)
        selection = None
        if self.state_view == "contribution":
            selection = self.activity.select(
                ranked, graph, budget=self.state_budget,
                combination_size=self.m2_combination_size,
                max_combinations=self.m2_max_combinations)
        state = graph_state(
            propositions, graph, self.assessments,
            rule_text=rule_lookup(), layers=LAYERS, round_no=round_no,
            cf_rules=cf_rules, control_rules=control_rules,
            selection=selection, include_ids=True)
        state["state_view"] = self.state_view
        state["activity"] = self.activity.refresh(
            state, graph, propositions=ranked, selection=selection, round_no=round_no)
        return state

    def _graph_fingerprint(self):
        graph = self._graph()
        return (len(graph.observations), len(graph.edges),
                sum(len(bundles) for bundles in graph.support.values()),
                len(graph.retracted), len(graph.retracted_support))

    def _state_signature(self):
        signature = {}
        for layer, record in self.assessments.items():
            if not isinstance(record, dict):
                continue
            signature[layer] = (record.get("state"), record.get("support_state"),
                                tuple(sorted((record.get("cf_states") or {}).items())))
        return signature

    def _delta(self):
        current = self._state_signature()
        changed = {layer: {"state": value[0], "support": value[1], "cf": dict(value[2])}
                   for layer, value in current.items()
                   if self._last_state.get(layer) != value}
        self._last_state = current
        return changed

    # --------------------------------------------------------------- dispatch
    def _record_claim(self, layer):
        """Write or withdraw the layer's E_S edges after a submission."""
        graph = self._graph()
        record = self.assessments.get(layer) or {}
        cited = [entry.get("evidence_id") for entry in record.get("support") or ()
                 if isinstance(entry, dict) and entry.get("evidence_id")]
        if record.get("state") == "excluded":
            # A vetoed layer keeps no E_S edges: its support is withdrawn.
            graph.retract_claim(layer, "layer_excluded")
        elif cited:
            graph.claim(layer, cited)
        support = graph_planner.layer_support(graph, self.assessments, layer)
        support["sufficient"] = support["corroborated"]
        if self.evidence_graph is not None:
            self.evidence_graph.record({"type": "layer_support", "layer": layer,
                                        "state": record.get("state"), **support})
        return support

    def _dispatch(self, name, args, allowed=None):
        if self.evidence_graph is not None:
            self.evidence_graph.begin_tool(name)
        if self.stage == 1 and name in READERS:
            prior = self.ledger.redundant(name, args)
            if prior:
                self.context_stats["redundant_queries"] += 1
                return {"redundant": True, "already_covered_by": prior,
                        "note": "该查询已被历史查询覆盖；请引用已有 evidence_id，或缩小范围后重试"}
        if name == "submit_layer":
            errors = graph_planner.claim_guard(self._graph(), args, self._resolve)
            if errors:
                if self.evidence_graph is not None:
                    self.evidence_graph.record({"type": "claim_rejected",
                                                "layer": args.get("layer"),
                                                "errors": errors})
                return {"committed": False, "errors": errors,
                        "next": "按 errors 修复后重新提交；不要重复提交同一批引用"}
        result = super()._dispatch(name, args, allowed)
        if name in READERS:
            evidence_id = result.get("evidence_id") if isinstance(result, dict) else None
            self.ledger.add(name, args, evidence_id)
        if name == "submit_layer" and isinstance(result, dict):
            support = self._record_claim(args.get("layer"))
            result["graph_support"] = support
            if not support["sufficient"]:
                result["graph_next"] = ("该层的支持只来自一个来源或一条链；"
                                        "需要不同服务/不同执行域的独立记录")
        return result

    def _guard_payload(self, payload):
        if self.stage != 1 or not isinstance(payload, dict):
            return []
        errors = []
        for assessment in payload.get("assessments") or ():
            if not isinstance(assessment, dict):
                continue
            for error in graph_planner.claim_guard(self._graph(), assessment, self._resolve):
                errors.append({**error, "layer": assessment.get("layer")})
        return errors

    def _sync_evidence_graph(self, name):
        """Ingest the call's records and feed fresh ids to the activity ledger."""
        super()._sync_evidence_graph(name)
        activity = getattr(self, "activity", None)
        if self.evidence_graph is None or activity is None:
            return
        fresh = getattr(self.evidence_graph, "last_fresh_ids", None)
        if fresh:
            activity.on_new_observation(fresh)

    def _evidence_summary(self, evidence_id):
        observation = self._graph().observations.get(evidence_id)
        return summary(observation, 160) if observation is not None else None

    @staticmethod
    def _collect_evidence(value):
        found = []

        def walk(item):
            if isinstance(item, dict):
                token = item.get("evidence_id")
                if isinstance(token, str):
                    found.append(token)
                for child in item.values():
                    walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)

        walk(value)
        return found

    def _evidence_ids_of(self, value):
        graph = self._graph()
        ids = []
        for token in self._collect_evidence(value):
            if token in graph.observations and token not in ids:
                ids.append(token)
        return ids

    def _request_messages(self):
        """Full transcript or its bounded projection, plus projection stats."""
        if not self.project_transcript:
            return self.messages, {}
        keep_from = self._round_starts[-2] if len(self._round_starts) >= 2 else 0
        projected, stats = project_messages(
            self.messages, self.activity.states, self._call_evidence,
            summarize=self._evidence_summary, budget=self.context_budget,
            keep_from=keep_from)
        if stats["after_chars"] > self.context_budget:
            current = self._round_starts[-1] if self._round_starts else len(self.messages)
            projected, stats = project_messages(
                self.messages, self.activity.states, self._call_evidence,
                summarize=self._evidence_summary, budget=self.context_budget,
                keep_from=current, max_evidence=2, summary_chars=60)
        return projected, stats

    def _result(self, termination, started):
        value = super()._result(termination, started)
        value["state_log"] = deepcopy(self.state_log)
        value["state_view"] = self.state_view
        value["stalled_rounds"] = self.stalled_rounds
        value["current_state"] = self.last_state
        value["activity_log"] = deepcopy(self.activity.activity_log)
        value["reactivation_log"] = deepcopy(self.activity.reactivation_log)
        value["selection_stats"] = (self.activity.last_selection.summary()
                                    if self.activity.last_selection is not None else None)
        value["projection_stats"] = deepcopy(self.projection_log)
        return value

    # ------------------------------------------------------------------- loop
    def diagnose(self, sample, checkpoint=None, event_sink=None):
        self.registry = Registry(sample)
        self.registry.agent = self
        self.assessments, self.verified = {}, set()
        self.stage, self.final = 1, None
        self.history, self.exchanges, self.drafts = [], [], []
        self.last_stage_two = None
        self.observations, self.read_seconds = {}, 0.0
        self.peak_rss_mb = 0.0
        self.pruning_log, self.max_prompt_tokens, self.stage_two_round = [], 0, None
        self._reset_token_stats()
        self.committed, self.stage_one_commits, self.last_commit_round = False, 0, 0
        self.context_stats = {"tool_result_chars": 0, "sent_tool_chars": 0, "truncations": 0,
                              "duplicate_queries": 0, "compacted_results": 0,
                              "submit_checkpoints": 0, "diagnosis_submissions": 0,
                              "evidence_graph_syncs": 0, "evidence_graph_observations": 0,
                              "redundant_queries": 0, "stalled_rounds": 0,
                              "state_changed_rounds": 0,
                              "evidence_edges_retracted": 0,
                              "support_edges_retracted": 0,
                              "active_evidence": 0, "dormant_evidence": 0,
                              "archived_evidence": 0, "reactivated": 0,
                              "projected_messages": 0, "projected_chars": 0}
        self.evidence_graph = EvidenceGraphBridge(
            sample, degree_budget=self.evidence_degree_budget,
            event_sink=self.graph_event_sink,
            context=lambda: (self.round_no, self.stage))
        self.ledger = graph_planner.QueryLedger()
        self.activity = ActivityLedger()
        self.projection_log = []
        self._call_evidence = {}
        self._round_starts = []
        self.state_log, self.last_state = [], None
        self.stalled_rounds = 0
        self._last_state = {}
        self._checkpoint_callback, self._event_sink = checkpoint, event_sink
        self.started = time.perf_counter()

        backend_config = (self.backend.configuration() if hasattr(self.backend, "configuration")
                          else {"backend": self.backend.name, "model": self.backend.model})
        allowed_keys = ("backend", "model", "base_url", "timeout_seconds", "max_retries",
                        "request_parameters", "replay_file")
        self.configuration = {**{k: backend_config[k] for k in allowed_keys if k in backend_config},
                              "agent": "graph",
                              "input_filters": {"mask_csv_gen_auth": sample.mask_csv_gen_auth,
                                                "policy": POLICY},
                              "max_rounds": self.max_rounds, "max_tool_calls": self.max_tool_calls,
                              "commit_every": self.commit_every,
                              "state_view": self.state_view,
                              "state_budget": self.state_budget,
                              "project_transcript": self.project_transcript,
                              "context_budget": self.context_budget,
                              "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest()}
        if self.state_view == "contribution":
            self.configuration.update(m2_combination_size=self.m2_combination_size,
                                      m2_max_combinations=self.m2_max_combinations)
        self.messages = [{"role": "system", "content": self.prompt},
                         {"role": "user", "content": json.dumps({
                             "request": "由证据图引导，逐步补足证据并提交两阶段诊断。",
                             "services": sample.cache.get("services", []),
                             "modalities": {"metrics": sample.ns("metrics").is_dir(),
                                            "logs": sample.ns("log").is_dir(),
                                            "execution_graph": sample.ns("graph").is_dir()},
                             "input_filter": background_policy(sample).report()},
                             ensure_ascii=False)}]
        self._message_hasher = hashlib.sha256()
        for message in self.messages:
            self._message_hasher.update(_frame(message))
        self._emit("session_start", {"configuration": self.configuration,
                                     "messages": self.messages})
        self._save("running")

        calls = 0
        for round_no in range(1, self.max_rounds + 1):
            self.round_no = round_no
            if self.stage == 1 and worklist.ready(self):
                transition = finish_layer_stage.run(self, {})
                self.stage_two_round = round_no
                self._emit("stage_transition", {"round": round_no, **transition})
            before_fingerprint = self._graph_fingerprint()
            before_signature = self._state_signature()

            self._round_starts.append(len(self.messages))
            state = self._state(round_no)
            self.last_state = state
            self.state_log.append({"round": round_no, "stage": self.stage, "state": state})
            activity = state.get("activity") or {}
            self.context_stats["active_evidence"] = activity.get("active", 0)
            self.context_stats["dormant_evidence"] = activity.get("dormant", 0)
            self.context_stats["archived_evidence"] = activity.get("archived", 0)
            self.context_stats["reactivated"] += len(activity.get("reactivated") or ())
            if self.stage == 1:
                checkpoint_reason = self._stage1_checkpoint(round_no)
                if checkpoint_reason:
                    tools = self._schemas(keep=SUBMITS)
                    allowed = SUBMITS
                    self.context_stats["submit_checkpoints"] += 1
                    notice = {"round": round_no, "stage": 1,
                              "submit_checkpoint": checkpoint_reason,
                              "rounds_remaining": self.max_rounds - round_no,
                              "state": self._stage_state(),
                              "graph": state,
                              "next": "本轮只提供 submit_layer：至少提交一个层（support_state=unknown 也可）。"
                                      "Stage 1 关闭后，Stage 2 还需要读取新证据和最终提交的轮次。"}
                    self._emit("submit_checkpoint", {"round": round_no, "notice": notice})
                else:
                    tools = self._schemas()
                    allowed = None
                    if not state.get("records"):
                        next_text = "先读取日志、指标和执行图；之后用 submit_layer 提交已复核的层。"
                    elif state.get("overview"):
                        next_text = ("尚无层级判断；根据 overview 的已读证据，"
                                     "用 submit_layer 提交已复核的层。")
                    else:
                        next_text = "按 graph 取证；需要时用 submit_layer 提交已复核的层。"
                    notice = {"round": round_no, "stage": 1, "graph": state,
                              "delta": self._delta(), "next": next_text}
            else:
                final_submit = round_no == self.max_rounds
                if final_submit:
                    tools = self._schemas(keep=DIAGNOSES)
                    allowed = DIAGNOSES
                    notice = {"round": round_no, "stage": 2, "final": True,
                              "graph": state,
                              "next": "本轮只提供 submit_diagnosis；也可直接返回 JSON"}
                else:
                    tools = self._schemas()
                    allowed = None
                    notice = {"round": round_no, "stage": 2,
                              "candidates": self._strict_candidates(),
                              "graph": state,
                              "next": "只在 candidate_layers 内排序与细分类"}
            self._append_message({"role": "user",
                                  "content": json.dumps(notice, ensure_ascii=False)})
            self._emit("state", {"round": round_no, "notice": notice})

            parameters = (self.backend.request_parameters()
                          if hasattr(self.backend, "request_parameters")
                          else {"model": self.backend.model})
            request_messages, projection = self._request_messages()
            request = {**parameters, "message_count": len(request_messages),
                       "messages_sha256": hashlib.sha256(
                           json.dumps(request_messages, ensure_ascii=False).encode()).hexdigest(),
                       "tool_count": len(tools or []),
                       "tools_sha256": hashlib.sha256(
                           json.dumps(tools or [], ensure_ascii=False,
                                      sort_keys=True).encode()).hexdigest()}
            if projection:
                request["projection"] = {key: projection[key] for key in
                                         ("tool_folded", "notices_folded",
                                          "before_chars", "after_chars", "budget")}
                self.projection_log.append({"round": round_no, **projection})
                self.context_stats["projected_messages"] += (
                    projection["tool_folded"] + projection["notices_folded"])
                self.context_stats["projected_chars"] = projection["after_chars"]
            exchange = {"round": round_no, "stage": self.stage, "started_at": timestamp(),
                        "configuration": deepcopy(self.configuration), "request": request,
                        "status": "pending"}
            turn = {"round": round_no, "stage": self.stage, "summary": "", "tool_calls": []}
            self.exchanges.append(exchange)
            self.history.append(turn)
            self._emit("llm_request", exchange)
            self._save("running")
            try:
                message, metadata = self.backend.complete(request_messages, tools)
            except Exception as exc:
                exchange.update(status="backend_error", error=error_info(exc),
                                finished_at=timestamp())
                turn["error"] = type(exc).__name__
                self._emit("llm_error", {"round": round_no, "error": exchange["error"]})
                return self._finish("backend_error")
            exchange.update(status="response_received",
                            response={"message": deepcopy(message),
                                      "metadata": deepcopy(metadata)},
                            finished_at=timestamp())
            turn.update(metadata)
            turn["assistant_message"] = deepcopy(message)
            usage = metadata.get("usage") or {}
            self._record_usage(usage)
            self._emit("llm_response", {"round": round_no, **exchange["response"]})
            if metadata.get("finish_reason") == "length" and not (message.get("tool_calls") or []):
                self.context_stats["truncations"] += 1
                exchange["status"] = "response_truncated"
                self._append_message(deepcopy(message))
                turn["summary"] = message.get("content") or ""
                feedback = {"role": "user", "content": json.dumps({
                    "error": "输出被 max_completion_tokens 截断，未解析任何判断。",
                    "next": "本轮只提交一个层的增量 delta，或先调用一个工具。",
                    "rounds_remaining": self.max_rounds - round_no}, ensure_ascii=False)}
                self._append_message(feedback)
                self._emit("response_truncated", {"round": round_no, "usage": usage,
                                                  "message": feedback})
                self._save("running")
                continue
            if not self._valid_message(message):
                exchange["status"] = "invalid_assistant_message"
                turn["error"] = "invalid_assistant_message"
                feedback = {"role": "user",
                            "content": "返回格式无效，请使用规范的 function tool_calls 或阶段 JSON。"}
                self._append_message(feedback)
                self._emit("feedback", {"round": round_no, "message": feedback})
                self._save("running")
                continue
            self._append_message(deepcopy(message))
            turn["summary"] = message.get("content") or ""
            self._save("running")
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                parsed = None
                try:
                    content = (message.get("content") or "").strip()
                    if content.startswith("```json") and content.endswith("```"):
                        content = content[7:-3].strip()
                    parsed = json.loads(content)
                    guard = self._guard_payload(parsed)
                    if guard:
                        update = {"committed": False, "errors": guard,
                                  "next": "按 errors 修复后重新提交；不要重复提交同一批引用"}
                    else:
                        if self.stage == 2 and isinstance(parsed, dict) and isinstance(
                                parsed.get("layer_order"), list):
                            self.last_stage_two = deepcopy(parsed)
                        submitted_stage = self.stage
                        update = process_response(self, parsed)
                        self.committed = True
                        if submitted_stage == 1:
                            self.stage_one_commits += 1
                            self.last_commit_round = round_no
                except ValidationErrors as exc:
                    update = {"errors": exc.errors,
                              "next": "按 error.code 逐项修复后重新提交；不要只改格式。"}
                    if self.stage == 2:
                        self.drafts.append({"round": round_no, "stage": self.stage,
                                            "payload": parsed, "errors": exc.errors})
                except (ValueError, TypeError, KeyError) as exc:
                    update = {"error": str(exc),
                              "next": "继续查询观测，或按当前阶段的 JSON 格式提交判断。"}
                    if self.stage == 2 and isinstance(parsed, dict):
                        self.drafts.append({"round": round_no, "stage": self.stage,
                                            "payload": parsed,
                                            "errors": [{"code": type(exc).__name__,
                                                        "message": str(exc)}]})
                if isinstance(update, dict):
                    update.setdefault("rounds_remaining", self.max_rounds - round_no)
                    update.setdefault("stage", self.stage)
                    pruning = update.get("pruning")
                    if pruning:
                        event = {"round": round_no, **pruning}
                        self.pruning_log.append(event)
                        self._emit("prune", event)
                    if update.get("worklist"):
                        self._emit("worklist_update", {"round": round_no, **update["worklist"]})
                    transition = update.get("transition")
                    if transition:
                        self.stage_two_round = round_no
                        self._emit("stage_transition", {"round": round_no, **transition})
                turn["workflow"] = update
                feedback = {"role": "user", "content": json.dumps(update, ensure_ascii=False)}
                self._append_message(feedback)
                self._emit("workflow", {"round": round_no, "update": update,
                                        "message": feedback})
                self._save("running")
            for call in tool_calls:
                if calls >= self.max_tool_calls:
                    return self._finish("tool_budget_exhausted")
                calls += 1
                function = call.get("function") or {}
                name = function.get("name", "")
                entry = {"id": call["id"], "name": name,
                         "arguments_raw": function.get("arguments"), "arguments": None,
                         "status": "running"}
                turn["tool_calls"].append(entry)
                self._emit("tool_call", {"round": round_no, "call": call})
                self._save("running")
                try:
                    args = json.loads(function.get("arguments", "{}"))
                    entry["arguments"] = args
                    output = self._dispatch(name, args, allowed=allowed)
                    entry["status"] = "completed"
                except (ValueError, TypeError, KeyError) as exc:
                    output = {"error": str(exc)}
                    entry.update(status="error", error_type=type(exc).__name__)
                except Exception as exc:
                    output = {"error": type(exc).__name__,
                              "note": "数据工具失败；此结果不支持排除"}
                    entry.update(status="error", error_type=type(exc).__name__)
                entry["result"] = output
                if name in READERS:
                    self._call_evidence[call["id"]] = self._evidence_ids_of(output)
                original = json_size(output)
                if isinstance(output, dict) and output.get("duplicate_of"):
                    self.context_stats["duplicate_queries"] += 1
                    view = output
                else:
                    view = bounded(output)
                    sent = json_size(view)
                    self.context_stats["tool_result_chars"] += original
                    self.context_stats["sent_tool_chars"] += sent
                    if sent < original:
                        self.context_stats["compacted_results"] += 1
                        self._emit("transcript_compacted", {
                            "round": round_no, "call_id": call["id"], "name": name,
                            "original_chars": original, "sent_chars": sent})
                tool_message = {"role": "tool", "tool_call_id": call["id"],
                                "content": json.dumps(view, ensure_ascii=False,
                                                      allow_nan=False)}
                self._append_message(tool_message)
                self._emit("tool_result", {"round": round_no, "call_id": call["id"],
                                           "result": output, "message": tool_message,
                                           "status": entry["status"]})
            exchange["status"] = "processed"
            moved = (self._graph_fingerprint() != before_fingerprint
                     or self._state_signature() != before_signature)
            turn["state_changed"] = moved
            self.context_stats["state_changed_rounds"] += int(moved)
            if moved:
                self.stalled_rounds = 0
            else:
                self.stalled_rounds += 1
                self.context_stats["stalled_rounds"] += 1
            self.context_stats["evidence_edges_retracted"] = len(self._graph().retracted)
            self.context_stats["support_edges_retracted"] = len(self._graph().retracted_support)
            self._emit("round_end", {"round": round_no, "stage": self.stage,
                                     "candidates": self._candidates()})
            self._save("completed" if self.final else "running")
            if self.final is not None:
                return self._finish("completed")
        return self._finish("round_budget_exhausted")
