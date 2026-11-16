"""Temporary, provenance-based masking of the known CSV generator key error."""
import json
import re
from collections import defaultdict

MARKER = "[CSV_AUTH_BACKGROUND_MASKED]"
POLICY = "csv-gen-invalid-api-key-v1"
BAD_KEY = re.compile(r"invalid_api_key|(?:invalid|incorrect)[\s-]+api[\s-]*key", re.I)
IDS = re.compile(r'''["'](?:id|request_id|requestId)["']\s*:\s*["']([\w-]{8,})["']''')
AUTH_PREFIX = re.compile(r'''(?:\[Error\]\s*)?(?:Failed to chat with LLM:\s*)?(?:[\w.]*AuthenticationError:\s*)?Error code:\s*401\s*-\s*''', re.I)
OTHER_FAILURE = re.compile(
    r"(?:status\s*[=:]\s*|Error code:\s*|HTTP/\d(?:\.\d)?\s+)(?:400|403|404|408|429|5\d\d)\b"
    r"|\b(?:timeout|timed out|connection refused|quota|rate.?limit|truncated|duplicate response)\b"
    r"|\b(?!AuthenticationError\b|Error\b)[A-Z]\w*(?:Error|Exception)\b", re.I)
OTHER_HTTP = re.compile(r"(?:status\s*[=:]\s*|Error code:\s*|HTTP/\d(?:\.\d)?\s+)(?:400|403|404|408|429|5\d\d)\b", re.I)


def csv_service(value):
    name = str(value or "").split("/", 1)[0]
    return (name in ("CSVGeneratorAgentGroup", "csv-gen")
            or re.fullmatch(r"agent-network-csv-gen(?:-edge)?(?:-[a-z0-9]+-[a-z0-9]+)?", name) is not None)


def request_ids(text):
    return set(IDS.findall(text.replace('\\"', '"').replace("\\'", "'")))


class CsvAuthMask:
    def __init__(self, sample):
        self.sample = sample
        self.enabled = sample.mask_csv_gen_auth
        self.seeded = False
        self.ids = set()
        self.known_background_present = False
        self.records = defaultdict(set)
        self.references = []

    def seed(self):
        if self.seeded or not self.enabled:
            return
        self.seeded = True
        for file in sorted(self.sample.ns("log").glob("*.log")):
            deployment = file.name.split("_", 1)[0]
            if not csv_service(deployment):
                continue
            with file.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if BAD_KEY.search(line):
                        self.learn(line, deployment)

    def learn(self, text, service):
        if self.enabled and csv_service(service) and BAD_KEY.search(text):
            self.known_background_present = True
            self.ids.update(request_ids(text))

    def record(self, channel, ref):
        key = json.dumps(ref, sort_keys=True, ensure_ascii=False)
        if key not in self.records[channel]:
            self.records[channel].add(key)
            if len(self.references) < 16:
                self.references.append({"channel": channel, **ref})

    def related(self, text, service=None, graph_csv_auth=False):
        if not self.enabled or not BAD_KEY.search(text):
            return False
        self.seed()
        self.learn(text, service)
        ids = request_ids(text)
        csv_summary = (not ids and self.known_background_present and re.search(r"\bcsv\b", text, re.I)
                       and any(key in text for key in ("agent_network_summarize_reasoning", "agent_network_final_result")))
        return (csv_service(service) or bool(ids & self.ids)
                or (not ids and (graph_csv_auth or "agent-network-csv-gen" in text
                                  or "CSVGeneratorAgentGroup" in text)) or bool(csv_summary))

    @staticmethod
    def payload_end(text, start):
        """Find the end of the small provider error object, preserving adjacent faults."""
        if start >= len(text) or text[start] != "{":
            return None
        depth, quote, escaped = 0, None, False
        for i in range(start, len(text)):
            char = text[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif quote:
                if char == quote:
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        return None

    def text(self, text, *, service=None, graph_csv_auth=False, channel, ref):
        if not isinstance(text, str) or not self.related(text, service, graph_csv_auth):
            return text
        spans = []
        for match in AUTH_PREFIX.finditer(text):
            end = self.payload_end(text, match.end())
            if end and self.related(text[match.start():end], service, graph_csv_auth):
                spans.append((match.start(), end))
        remaining = text
        for start, end in reversed(spans):
            remaining = remaining[:start] + MARKER + remaining[end:]
        # A second auth failure with another request ID remains independent.
        foreign_ids = request_ids(remaining) - self.ids
        if not OTHER_FAILURE.search(remaining) and not foreign_ids:
            self.record(channel, ref)
            return ""
        if remaining != text:
            self.record(channel, ref)
            if OTHER_HTTP.search(remaining):
                remaining += " [observed HTTP failure]"
        return remaining

    def event(self, event, raw, *, channel, ref):
        if not self.enabled or event.get("kind") != "model_call" or event.get("status") != 401:
            return False
        text = json.dumps(raw, ensure_ascii=False)
        if csv_service(event.get("source")) or self.related(text, event.get("source")):
            self.known_background_present = True
            self.record(channel, ref)
            return True
        return False

    def report(self):
        return {"enabled": self.enabled, "policy": POLICY,
                "known_background_present": self.known_background_present,
                "seed_request_ids": len(self.ids),
                "masked_records": {k: len(v) for k, v in self.records.items()},
                "references": list(self.references),
                "note": "仅屏蔽 csv-gen 已知 key 错误及可关联副本；被屏蔽不等于正常。"
                        "csv-gen 指标可能仍受污染，不能单独用于归因。" if self.enabled else "未启用临时屏蔽。"}


def policy(sample):
    if "csv_auth_mask" not in sample.cache:
        sample.cache["csv_auth_mask"] = CsvAuthMask(sample)
    return sample.cache["csv_auth_mask"]
