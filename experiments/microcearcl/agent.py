"""LLM-led detection → counterfactual verification → narrowing → classification."""
import json
import hashlib
import time
from copy import deepcopy
from .registry import Registry, CHECKS, DIAGNOSES, SUBMITS
from .prompts import SYSTEM
from .rules import LAYERS, controls, counterfactuals, predictions
from .utils import worklist
from .utils.observation_summary import READERS
from .utils.process_response import run as process_response
from .utils.finish_diagnosis import ValidationErrors, salvage
from .utils.conversation_log import error_info, timestamp
from .utils.background_noise import policy as background_policy, POLICY
from .utils.transcript_budget import bounded, size as json_size


def digest(value):
    """Stable content identity for a JSON-compatible request snapshot."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _frame(message):
    return json.dumps(message, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode() + b"\n"


def messages_digest(messages):
    """Digest of an append-only message list; equal to the streaming digest below."""
    hasher = hashlib.sha256()
    for message in messages:
        hasher.update(_frame(message))
    return hasher.hexdigest()


def resident_mb():
    """Process RSS, or None when psutil is unavailable. Diagnostic only."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1048576, 1)
    except Exception:
        return None


class CounterfactualAgent:
    def __init__(self, backend, max_rounds=32, max_tool_calls=100, commit_every=5):
        self.backend = backend
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        # A submit checkpoint is forced once this many rounds pass without a
        # Stage-1 commit.  0 disables periodic checkpoints (the end-game force
        # still applies).
        self.commit_every = commit_every

    def diagnose(self, sample, checkpoint=None, event_sink=None):
        self.registry = Registry(sample)
        self.registry.agent = self
        self.assessments, self.verified = {}, set()
        self.stage, self.final = 1, None
        self.history, self.exchanges = [], []
        self.drafts = []
        self.last_stage_two = None
        self.observations, self.read_seconds = {}, 0.0
        self.peak_rss_mb = 0.0
        self.pruning_log, self.max_prompt_tokens, self.stage_two_round = [], 0, None
        self.committed, self.stage_one_commits, self.last_commit_round = False, 0, 0
        self.context_stats = {'tool_result_chars': 0, 'sent_tool_chars': 0,
                              'truncations': 0, 'duplicate_queries': 0,
                              'compacted_results': 0, 'submit_checkpoints': 0,
                              'diagnosis_submissions': 0}
        self._checkpoint_callback, self._event_sink = checkpoint, event_sink
        self.started = time.perf_counter()
        backend_config = (self.backend.configuration() if hasattr(self.backend, "configuration")
                          else {"backend": self.backend.name, "model": self.backend.model})
        allowed = ("backend", "model", "base_url", "timeout_seconds", "max_retries", "request_parameters", "replay_file")
        self.configuration = {**{k: backend_config[k] for k in allowed if k in backend_config},
                              "input_filters": {"mask_csv_gen_auth": sample.mask_csv_gen_auth, "policy": POLICY},
                              "max_rounds": self.max_rounds, "max_tool_calls": self.max_tool_calls,
                              "commit_every": self.commit_every,
                              "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()}
        self.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps({
            "request": "检测异常，进行反事实验证与两阶段诊断。", "services": sample.cache.get("services", []),
            "modalities": {"metrics": sample.ns("metrics").is_dir(), "logs": sample.ns("log").is_dir(),
                           "execution_graph": sample.ns("graph").is_dir()},
            "evidence_mode": sample.mode, "input_filter": background_policy(sample).report()}, ensure_ascii=False)}]
        # Streaming digest: O(new messages) per round instead of re-encoding the
        # whole growing history, which churned hundreds of MB per long run.
        self._message_hasher = hashlib.sha256()
        for message in self.messages:
            self._message_hasher.update(_frame(message))
        self._emit("session_start", {"configuration": self.configuration, "messages": self.messages})
        self._save("running")
        calls = 0
        for round_no in range(1, self.max_rounds + 1):
            # Reserve the last round for a submission: stage 2 gets no tools and
            # one explicit chance to return the final JSON.
            final_submit = self.stage == 2 and round_no == self.max_rounds
            # An exploration-only run never reaches Stage 2.  Near the budget,
            # revoke data tools when the previous round did not commit a delta.
            stage1_force = (self.stage == 1 and self.max_rounds >= 8
                            and round_no >= self.max_rounds - 4
                            and self.last_commit_round < round_no - 1)
            # Periodic checkpoint: after N rounds without a Stage-1 commit,
            # expose only submit_layer until one delta lands.  This replaces the
            # old per-round "please commit next round" nudge.
            needs_commit = bool(worklist.required(self.assessments))
            periodic_checkpoint = (self.stage == 1 and self.commit_every > 0
                                   and needs_commit
                                   and round_no - self.last_commit_round >= self.commit_every)
            checkpoint = (self.stage == 1 and needs_commit
                          and (stage1_force or periodic_checkpoint))
            if final_submit:
                notice = {"role": "user", "content": (
                    "程序进入最终提交轮：本轮只提供 submit_diagnosis；也可以直接返回 JSON。"
                    "请依据已读证据提交最可靠的诊断；未通过校验的草稿会被保留。")}
                self._append_message(notice)
                self._emit("final_submission_round", {"round": round_no, "message": notice})
            elif checkpoint:
                notice = {"role": "user", "content": json.dumps({
                    "submit_checkpoint": "deadline" if stage1_force else "periodic",
                    "round": round_no,
                    "rounds_remaining": self.max_rounds - round_no,
                    "state": self._stage_state(),
                    "next": "本轮只提供 submit_layer：至少提交一个层（support_state=unknown 也可）。"
                            "Stage 1 关闭后，Stage 2 还需要读取新证据和最终提交的轮次。"},
                    ensure_ascii=False)}
                self._append_message(notice)
                self.context_stats["submit_checkpoints"] += 1
                self._emit("submit_checkpoint", {"round": round_no, "message": notice})
            if final_submit:
                tools = self._schemas(keep=DIAGNOSES)
                allowed = DIAGNOSES
            elif checkpoint:
                # Keep the submission tool available: a tool-free round cannot
                # close a layer, and data tools would let the checkpoint be skipped.
                tools = self._schemas(keep=SUBMITS)
                allowed = SUBMITS
            else:
                tools = self._schemas()
                allowed = None
            parameters = (self.backend.request_parameters() if hasattr(self.backend, "request_parameters")
                          else {"model": self.backend.model})
            # Snapshot identity, not a copy of the growing history: messages are append-only,
            # so request N is exactly messages[:message_count] and its digest proves it.
            # Storing the full history per exchange turned memory and checkpoints quadratic.
            request = {**parameters, "message_count": len(self.messages),
                       "messages_sha256": self._message_hasher.copy().hexdigest(),
                       "tool_count": len(tools or []), "tools_sha256": digest(tools or [])}
            exchange = {"round": round_no, "stage": self.stage, "started_at": timestamp(),
                        "configuration": deepcopy(self.configuration), "request": request, "status": "pending"}
            turn = {"round": round_no, "stage": self.stage, "summary": "", "tool_calls": []}
            self.exchanges.append(exchange)
            self.history.append(turn)
            self._emit("llm_request", exchange)
            self._save("running")
            try:
                message, metadata = self.backend.complete(self.messages, tools)
            except Exception as exc:
                exchange.update(status="backend_error", error=error_info(exc), finished_at=timestamp())
                turn["error"] = type(exc).__name__
                self._emit("llm_error", {"round": round_no, "error": exchange["error"]})
                return self._finish("backend_error")
            exchange.update(status="response_received", response={"message": deepcopy(message),
                            "metadata": deepcopy(metadata)}, finished_at=timestamp())
            turn.update(metadata)
            turn["assistant_message"] = deepcopy(message)
            usage = metadata.get("usage") or {}
            if isinstance(usage.get("prompt_tokens"), int):
                self.max_prompt_tokens = max(self.max_prompt_tokens, usage["prompt_tokens"])
            self._emit("llm_response", {"round": round_no, **exchange["response"]})
            if metadata.get("finish_reason") == "length" and not (message.get("tool_calls") or []):
                # A truncated JSON submission is unparseable; ask for a one-layer delta
                # instead of spending another round on the same oversized payload.
                self.context_stats["truncations"] += 1
                exchange["status"] = "response_truncated"
                self._append_message(deepcopy(message))
                turn["summary"] = message.get("content") or ""
                feedback = {"role": "user", "content": json.dumps({
                    "error": "本轮输出达到 max_completion_tokens 被截断，未解析任何判断。",
                    "next": "本轮只提交一个层的增量 delta（或先调用一个工具）；"
                            "不要把多层放进同一次 JSON。",
                    "rounds_remaining": self.max_rounds - round_no}, ensure_ascii=False)}
                self._append_message(feedback)
                self._emit("response_truncated", {"round": round_no,
                                                  "usage": usage, "message": feedback})
                self._save("running")
                continue
            if not self._valid_message(message):
                exchange["status"] = "invalid_assistant_message"
                turn["error"] = "invalid_assistant_message"
                feedback = {"role": "user", "content": "返回格式无效，请使用规范的 function tool_calls 或阶段 JSON。"}
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
                    # Remember the last ordering a Stage-2 submission actually contained,
                    # even if the submission is rejected, so budget exhaustion can keep it.
                    if self.stage == 2 and isinstance(parsed, dict) and isinstance(parsed.get('layer_order'), list):
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
                    update = {"error": str(exc), "next": "继续查询观测，或按当前阶段的 JSON 格式提交判断。"}
                    if self.stage == 2 and isinstance(parsed, dict):
                        self.drafts.append({"round": round_no, "stage": self.stage, "payload": parsed,
                                            "errors": [{"code": type(exc).__name__, "message": str(exc)}]})
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
                self._emit("workflow", {"round": round_no, "update": update, "message": feedback})
                self._save("running")
            for call in tool_calls:
                if calls >= self.max_tool_calls:
                    return self._finish("tool_budget_exhausted")
                calls += 1
                function = call.get("function") or {}
                name = function.get("name", "")
                entry = {"id": call["id"], "name": name, "arguments_raw": function.get("arguments"),
                         "arguments": None, "status": "running"}
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
                    output = {"error": type(exc).__name__, "note": "数据工具失败；此结果不支持排除"}
                    entry.update(status="error", error_type=type(exc).__name__)
                entry["result"] = output
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
                                "content": json.dumps(view, ensure_ascii=False, allow_nan=False)}
                self._append_message(tool_message)
                self._emit("tool_result", {"round": round_no, "call_id": call["id"], "result": output,
                                           "message": tool_message, "status": entry["status"]})
                # No artifact checkpoint here: the journal already fsynced the result,
                # and the next pre-dispatch/round-end save persists it. Rewriting the
                # whole artifact after every call multiplied file-watcher churn (open
                # editors reload the multi-MB JSON each time) and I/O cost.
            exchange["status"] = "processed"
            self._emit("round_end", {"round": round_no, "stage": self.stage, "candidates": self._candidates()})
            self._save("completed" if self.final else "running")
            if self.final is not None:
                return self._finish("completed")
        return self._finish("round_budget_exhausted")

    def _append_message(self, message):
        self.messages.append(message)
        self._message_hasher.update(_frame(message))

    def _emit(self, event, payload):
        if self._event_sink:
            self._event_sink(event, deepcopy(payload))

    def _save(self, termination):
        result = self._result(termination, self.started)
        if self._checkpoint_callback:
            self._checkpoint_callback(result)
        return result

    def _finish(self, termination):
        # Budget exhaustion still submits the last Stage-2 ordering.  The run is
        # not marked completed, but its ranking and any valid diagnoses are kept.
        if termination == "round_budget_exhausted" and self.final is None and self.last_stage_two is not None:
            self.final = salvage(self, self.last_stage_two)
            self._emit("salvaged_diagnosis", {"round": len(self.history), "diagnosis": self.final})
        self._emit("session_end", {"termination": termination, "stage": self.stage,
                                   "candidates": self._candidates(), "diagnosis": self.final})
        return self._save(termination)

    def _schemas(self, keep=None, stage=None):
        return self.registry.schemas(keep, self.stage if stage is None else stage)

    @staticmethod
    def _valid_message(message):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return False
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            return False
        ids = set()
        for call in calls:
            if (not isinstance(call, dict) or not isinstance(call.get("id"), str)
                    or call["id"] in ids or not isinstance(call.get("function"), dict)
                    or not isinstance(call["function"].get("name"), str)
                    or not isinstance(call["function"].get("arguments"), str)):
                return False
            ids.add(call["id"])
        return True

    def _dispatch(self, name, args, allowed=None):
        if not isinstance(args, dict):
            raise ValueError("工具参数必须为 JSON object")
        if self.final is not None:
            raise ValueError("诊断已完成")
        if allowed is not None and name not in allowed:
            raise ValueError(f"本轮只允许 {', '.join(allowed)}")
        if name not in self.registry.modules:
            raise ValueError("未知观测或诊断工具；流程控制函数不对 LLM 暴露")
        if name in SUBMITS and self.stage == 2:
            raise ValueError("第二阶段请使用 submit_diagnosis")
        if name in DIAGNOSES and self.stage != 2:
            raise ValueError("submit_diagnosis 只在 Stage 2 可用")
        layer = args.get("layer") if name in CHECKS else None
        if self.stage == 2 and layer and layer not in self._strict_candidates():
            raise ValueError("第二阶段只能检查保留的候选层级")
        started = time.perf_counter()
        try:
            result = self.registry.invoke(name, args, self.stage)
        except Exception as exc:
            if name in READERS and name in self.registry.calls:
                self.observations[name] = {"status": "error", "error": type(exc).__name__}
            raise
        else:
            if (name in READERS and isinstance(result, dict)
                    and not result.get("duplicate_of")):
                self.observations[name] = {"status": "loaded", **result}
            return result
        finally:
            if name in READERS:
                self.read_seconds += time.perf_counter() - started

    @property
    def exposed_mechanisms(self):
        return self.registry.exposed_mechanisms

    def _candidates(self):
        # Gate 1: support is required. Unassessed layers stay visible so backend
        # failures do not silently drop the whole search space.
        return worklist.visible_candidates(self.assessments)

    def _stage_state(self):
        """Compact checkpoint state: required work, cached/pruned rules, revisits."""
        ledger, pruned = {}, []
        for layer in LAYERS:
            record = self.assessments.get(layer) or {}
            state = record.get('state', 'unsubmitted')
            ledger[layer] = {'state': state,
                             'support': record.get('support_state', 'unsubmitted'),
                             'cf': record.get('cf_states', {}),
                             'cached': record.get('supported_rules', [])}
            # ``pruned`` is monotone: only vetoed/skipped layers belong here.
            # Parked (unknown) layers stay revisitable and appear in optional.
            if state in ('excluded', 'skipped'):
                pruned.append({'layer': layer, 'state': state,
                               'reason': 'CF veto' if state == 'excluded' else 'skipped',
                               'closed_rules': list(predictions(layer))
                                               + list(counterfactuals(layer))
                                               + list(controls(layer))})
        return {'required': worklist.required(self.assessments),
                'optional': worklist.optional(self.assessments),
                'candidates': worklist.candidates(self.assessments),
                'visible': self._candidates(),
                'parked': worklist.parked(self.assessments),
                'ledger': ledger, 'pruned': pruned,
                'last_commit_round': self.last_commit_round}

    def _strict_candidates(self):
        return worklist.candidates(self.assessments)

    def _result(self, termination, started):
        # Checkpoints are serialized synchronously by the caller; copying the whole
        # (growing) transcript on every save only doubled peak memory.
        rss_mb = resident_mb()
        if rss_mb is not None:
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        ledger = {layer: {key: value.get(key) for key in
                          ("state", "support_state", "support_confirmed", "parked",
                           "supported_rules", "optional_rules", "cf_states", "veto_by", "gaps")}
                  for layer, value in self.assessments.items()}
        return {"method": "MicroCEARCL", "backend": self.backend.name, "model": self.backend.model,
                "termination": termination, "evidence_mode": self.registry.sample.mode,
                "configuration": deepcopy(self.configuration),
                "messages": self.messages, "exchanges": self.exchanges,
                "stage": self.stage, "candidate_layers": self._candidates(),
                "strict_candidate_layers": self._strict_candidates(),
                "excluded_layers": [k for k, v in self.assessments.items() if v.get("state") == "excluded"],
                "unsupported_layers": [k for k, v in self.assessments.items() if v.get("state") == "unsupported"],
                "parked_layers": [k for k, v in self.assessments.items() if v.get("parked")],
                "skipped_layers": [k for k, v in self.assessments.items() if v.get("state") == "skipped"],
                "layers": self.assessments, "rule_ledger": ledger,
                "layer_verifications": deepcopy(self.registry.verifications), "rules": self.registry.evidence, "observations": self.observations,
                "mechanisms": self.registry.mechanisms if self.stage == 2 else {},
                "diagnosis": self.final, "diagnosis_drafts": self.drafts,
                "pruning_log": self.pruning_log, "worklist": worklist.summary(self),
                "context": {**self.context_stats, "max_prompt_tokens": self.max_prompt_tokens,
                            "commit_every": self.commit_every,
                            "stage_two_round": self.stage_two_round,
                            "stage_one_commits": self.stage_one_commits,
                            "stage_one_committed": self.committed},
                "trajectory": self.history,
                "warnings": self.registry.sample.warnings,
                "input_filter": deepcopy(background_policy(self.registry.sample).report()),
                "timing": {"read_observations": self.read_seconds, "localize": time.perf_counter() - started,
                           "rss_mb": rss_mb, "peak_rss_mb": round(self.peak_rss_mb, 1)}}
