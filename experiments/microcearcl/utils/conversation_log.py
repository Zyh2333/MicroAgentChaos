"""Durable transcripts of visible messages, requests and tool execution."""
from datetime import datetime
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo
from uuid import uuid4


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="milliseconds")


def safe_endpoint(value):
    """Record endpoint identity without URL credentials or query tokens."""
    parsed = urlsplit(str(value))
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def error_info(error):
    # SDK error strings can contain headers/payloads; retain structured identity.
    result = {"type": type(error).__name__}
    for name in ("status_code", "request_id"):
        value = getattr(error, name, None)
        if isinstance(value, (str, int)):
            result[name] = value
    return result


class ConversationLog:
    def __init__(self, artifact):
        artifact = Path(artifact)
        self.jsonl_path = artifact.with_suffix(".conversation.jsonl")
        self.text_path = artifact.with_suffix(".conversation.log")
        self.sequence = 0
        self.session_id = uuid4().hex

    def emit(self, event, payload):
        """Write each event before proceeding, including before external calls."""
        self.sequence += 1
        record = {"session_id": self.session_id, "sequence": self.sequence,
                  "timestamp": timestamp(), "event": event, **payload}
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{record['timestamp']}] #{self.sequence} {event}\n")
            # Streamed: the indented pure-Python encoder otherwise builds the whole
            # payload chunk list in memory, which compounds across long transcripts.
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
