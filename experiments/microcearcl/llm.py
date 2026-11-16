"""OpenAI-compatible tool calling, plus clearly labeled offline replay."""
import json
import os
from pathlib import Path
from .utils.conversation_log import safe_endpoint


class ChatBackend:
    name = "llm"

    def __init__(self, model=None, base_url=None, timeout=90, max_tokens=8192):
        from openai import OpenAI
        api_key = os.environ.get("MICROCEARCL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("MICROCEARCL_MODEL")
        if not api_key or not self.model:
            raise ValueError("配置 MICROCEARCL_MODEL 与 MICROCEARCL_API_KEY（或 OPENAI_API_KEY）后运行 LLM")
        self.client = OpenAI(api_key=api_key,
                             base_url=base_url or os.environ.get("MICROCEARCL_BASE_URL") or None,
                             timeout=timeout, max_retries=2)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = 2
        self.endpoint = base_url or os.environ.get("MICROCEARCL_BASE_URL") or str(self.client.base_url)

    def request_parameters(self):
        return {"model": self.model, "tool_choice": "auto", "max_completion_tokens": self.max_tokens}

    def configuration(self):
        return {"backend": self.name, "model": self.model, "base_url": safe_endpoint(self.endpoint),
                "timeout_seconds": self.timeout, "max_retries": self.max_retries,
                "request_parameters": self.request_parameters()}

    def close(self):
        """Release the HTTP connection pool when a batch ends."""
        self.client.close()

    def complete(self, messages, tools):
        parameters = self.request_parameters()
        if tools is not None:
            parameters["tools"] = tools
        else:
            # Final submission rounds pass None: omit tools and tool_choice.
            parameters.pop("tool_choice", None)
        response = self.client.chat.completions.create(messages=messages, **parameters)
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        # Keep only documented assistant message fields, never internal reasoning.
        message = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "refusal")}
        response_info = {key: getattr(response, key, None) for key in
                         ("id", "model", "created", "system_fingerprint", "service_tier", "_request_id")}
        return message, {"usage": response.usage.model_dump() if response.usage else {},
                         "finish_reason": choice.finish_reason,
                         "response_info": {k: v for k, v in response_info.items() if v is not None}}


class ReplayBackend:
    """Recorded assistant messages. This verifies plumbing, not LLM accuracy."""
    name = "replay"
    model = "offline-replay"

    def __init__(self, path):
        self.path = str(path)
        self.turns = iter(json.loads(Path(path).read_text(encoding="utf-8"))["turns"])

    def request_parameters(self):
        return {"model": self.model}

    def configuration(self):
        return {"backend": self.name, "model": self.model, "replay_file": self.path,
                "request_parameters": self.request_parameters()}

    def close(self):
        pass

    def complete(self, messages, tools):
        try:
            message = next(self.turns)
        except StopIteration as exc:
            raise RuntimeError("回放耗尽，尚未完成诊断") from exc
        return message, {"usage": {}, "finish_reason": "replay"}
