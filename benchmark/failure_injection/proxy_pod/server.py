"""HTTP reverse proxy with fault injection and runtime rule controls."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aiohttp import web, ClientSession, ClientTimeout

try:
    from .taxonomy import load_rules, FaultRule
    from .engine import ProxyFaultEngine, RequestCtx, Outcome
except ImportError:
    from taxonomy import load_rules, FaultRule  # type: ignore
    from engine import ProxyFaultEngine, RequestCtx, Outcome  # type: ignore


# Exclude hop-by-hop headers when forwarding.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


@dataclass
class ProxyConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8900
    upstreams: Dict[str, str] = field(default_factory=dict)
    default_upstream: Optional[str] = None
    routes: List[Tuple[str, str]] = field(default_factory=list)  # (path_prefix, upstream_name)
    upstream_header: str = "X-Upstream"
    upstream_url_header: str = "X-Upstream-Url"
    allow_header_upstream: bool = True
    taskId_header: str = "X-TaskId"
    control_prefix: str = "/__fault__"
    force_non_stream: bool = False
    timeout_s: float = 300.0

    @staticmethod
    def from_yaml(d: Optional[dict]) -> "ProxyConfig":
        d = (d or {}).get("proxy", {}) if d else {}
        listen = d.get("listen", "0.0.0.0:8900")
        host, _, port = listen.partition(":")
        cfg = ProxyConfig(
            listen_host=host or "0.0.0.0",
            listen_port=int(port or 8900),
            upstreams=dict(d.get("upstreams") or {}),
            default_upstream=d.get("default_upstream"),
            routes=[(r["path_prefix"], r["upstream"]) for r in (d.get("routes") or [])],
            upstream_header=d.get("upstream_header", "X-Upstream"),
            upstream_url_header=d.get("upstream_url_header", "X-Upstream-Url"),
            allow_header_upstream=bool(d.get("allow_header_upstream", True)),
            taskId_header=d.get("taskId_header", "X-TaskId"),
            control_prefix=d.get("control_prefix", "/__fault__"),
            force_non_stream=bool(d.get("force_non_stream", False)),
            timeout_s=float(d.get("timeout_s", 300.0)),
        )
        return cfg


def _normalize_base(base: Optional[str]) -> Optional[str]:
    """Normalize an upstream address, adding http:// when needed."""
    if not base:
        return base
    base = base.strip()
    if "://" not in base:
        base = "http://" + base
    return base


def _parse_yaml_or_json(text: str) -> dict:
    """Parse YAML or JSON configuration."""
    text = text.strip()
    if not text:
        return {}
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


class ProxyApp:

    def __init__(self, cfg: ProxyConfig, engine: ProxyFaultEngine):
        self.cfg = cfg
        self.engine = engine
        self._session: Optional[ClientSession] = None
        self._resp_cache: Dict[str, dict] = {}

    async def start(self):
        self._session = ClientSession(timeout=ClientTimeout(total=self.cfg.timeout_s))

    async def stop(self):
        if self._session:
            await self._session.close()
        self.engine.close()

    def resolve_upstream(self, request: web.Request) -> Tuple[Optional[str], Optional[str]]:
        name = request.headers.get(self.cfg.upstream_header)
        if not name:
            for prefix, up in self.cfg.routes:
                if request.path.startswith(prefix):
                    name = up
                    break
        if not name:
            name = self.cfg.default_upstream

        base = None
        if self.cfg.allow_header_upstream:
            base = request.headers.get(self.cfg.upstream_url_header)
        if not base and name:
            base = self.cfg.upstreams.get(name)

        # Use the URL as the name when no logical name is set.
        if not name and base:
            name = base
        return name, _normalize_base(base)

    async def ctl_health(self, request: web.Request):
        return web.json_response({"status": "ok", "rules": len(self.engine.rules())})

    async def ctl_stats(self, request: web.Request):
        return web.json_response(self.engine.stats())

    async def ctl_get_rules(self, request: web.Request):
        return web.json_response({"rules": [r.labels() for r in self.engine.rules()]})

    async def ctl_set_rules(self, request: web.Request):
        text = await request.text()
        try:
            rules = load_rules(_parse_yaml_or_json(text))
        except Exception as e:
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=400)
        self.engine.set_rules(rules)
        return web.json_response({"loaded": len(rules),
                                  "rules": [r.labels() for r in rules]})

    async def ctl_reset(self, request: web.Request):
        self.engine.reset_counts()
        return web.json_response({"status": "counts reset"})

    async def ctl_clear(self, request: web.Request):
        self.engine.clear_rules()
        self._resp_cache.clear()
        return web.json_response({"status": "cleared", "rules": 0})

    async def handle(self, request: web.Request) -> web.StreamResponse:
        up_name, up_base = self.resolve_upstream(request)
        if up_base is None:
            return web.json_response(
                {"error": "no upstream resolved",
                 "hint": f"设置 default_upstream / routes / 请求头 {self.cfg.upstream_header} "
                         f"(逻辑名)或 {self.cfg.upstream_url_header} (物理地址)"}, status=502)

        raw = await request.read()
        req_json = self._try_json(request.headers.get("Content-Type", ""), raw)
        ctx = RequestCtx(method=request.method, path=request.path,
                         headers=dict(request.headers), upstream=up_name, taskId=request.headers.get(self.cfg.taskId_header))

        out = self.engine.apply("request", ctx, req_json)
        if out.delay_ms > 0:
            await asyncio.sleep(out.delay_ms / 1000.0)
        if out.short_circuit == "abort":
            return self._abort_response(out)
        if out.short_circuit == "drop":
            # Return 504 for a dropped connection without hanging the caller.
            return web.json_response(
                {"error": {"type": "fault_injected", "message": "connection dropped (timeout)"},
                 "fired": out.fired}, status=504)

        send_body = raw
        if req_json is not None:
            body_obj = out.body_value if out.body_mutated else req_json
            if self.cfg.force_non_stream and isinstance(body_obj, dict) and body_obj.get("stream"):
                body_obj = copy.deepcopy(body_obj) if not out.body_mutated else body_obj
                body_obj["stream"] = False
                out.body_mutated = True
            if out.body_mutated:
                send_body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")

        # Avoid duplicating /v1 in the upstream path.
        path_qs = request.path_qs
        if up_base.rstrip("/").endswith("/v1") and request.path.startswith("/v1/"):
            path_qs = path_qs[3:]
        url = up_base.rstrip("/") + path_qs
        fwd_headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in _HOP_BY_HOP}
        try:
            async with self._session.request(
                request.method, url, data=send_body or None, headers=fwd_headers,
            ) as upstream_resp:
                resp_bytes = await upstream_resp.read()
                resp_status = upstream_resp.status
                resp_ctype = upstream_resp.headers.get("Content-Type", "")
                resp_headers = {k: v for k, v in upstream_resp.headers.items()
                                if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"}
        except asyncio.TimeoutError:
            return web.json_response({"error": "upstream timeout"}, status=504)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"upstream error: {type(e).__name__}: {e}"}, status=502)

        # Pass SSE responses through without body edits.
        if "text/event-stream" in resp_ctype:
            return web.Response(body=resp_bytes, status=resp_status, headers=resp_headers)

        resp_json = self._try_json(resp_ctype, resp_bytes)

        rout = self.engine.apply("response", ctx, resp_json)
        if rout.delay_ms > 0:
            await asyncio.sleep(rout.delay_ms / 1000.0)
        if rout.short_circuit == "abort":
            return self._abort_response(rout)
        if rout.short_circuit == "drop":
            return web.json_response(
                {"error": {"type": "fault_injected", "message": "response dropped (timeout)"},
                 "fired": rout.fired}, status=504)

        # Replay the previous response for this upstream and path.
        cache_key = f"{up_name}:{request.path}"
        dup = any(f.get("action") == "duplicate" for f in rout.fired)
        if resp_json is not None and isinstance(resp_json, dict):
            if dup and cache_key in self._resp_cache:
                body_obj = self._resp_cache[cache_key]
                resp_bytes = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
            elif rout.body_mutated:
                resp_bytes = json.dumps(rout.body_value, ensure_ascii=False).encode("utf-8")
            # Cache this response for a later duplicate action.
            if not dup:
                self._resp_cache[cache_key] = resp_json
            resp_headers.pop("Content-Length", None)

        return web.Response(body=resp_bytes, status=resp_status, headers=resp_headers)

    @staticmethod
    def _try_json(content_type: str, raw: bytes):
        if not raw:
            return None
        if "json" not in content_type.lower():
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    @staticmethod
    def _abort_response(out: Outcome) -> web.Response:
        body = out.body
        if isinstance(body, (dict, list)):
            resp = web.json_response(body, status=out.status)
        else:
            resp = web.Response(text=str(body) if body is not None else "", status=out.status)
        return resp


def build_app(cfg: ProxyConfig, engine: ProxyFaultEngine) -> web.Application:
    proxy = ProxyApp(cfg, engine)
    app = web.Application()
    app["proxy"] = proxy

    cp = cfg.control_prefix.rstrip("/")
    app.router.add_get(f"{cp}/health", proxy.ctl_health)
    app.router.add_get(f"{cp}/stats", proxy.ctl_stats)
    app.router.add_get(f"{cp}/rules", proxy.ctl_get_rules)
    app.router.add_post(f"{cp}/rules", proxy.ctl_set_rules)
    app.router.add_post(f"{cp}/reset", proxy.ctl_reset)
    app.router.add_post(f"{cp}/clear", proxy.ctl_clear)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)

    async def _on_start(app_):
        await app_["proxy"].start()

    async def _on_cleanup(app_):
        await app_["proxy"].stop()

    app.on_startup.append(_on_start)
    app.on_cleanup.append(_on_cleanup)
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="Agent Service 故障注入代理 Pod")
    p.add_argument("--config", default="config/faults.example.yaml", help="YAML 故障清单(含 proxy 段与 faults 列表)")
    p.add_argument("--start-without-faults", action="store_true",
                   help="启动时仅加载代理配置，故障规则由控制接口稍后注入")
    p.add_argument("--listen", help="监听地址, 如 0.0.0.0:8900")
    p.add_argument("--upstream", action="append", default=[],
                   help="上游定义 name=base_url, 可重复")
    p.add_argument("--default-upstream", help="默认上游名")
    p.add_argument("--log", help="命中日志(JSONL)输出路径")
    p.add_argument("--force-non-stream", action="store_true",
                   help="把 LLM 请求的 stream 置 false, 以便对响应体注入故障")
    args = p.parse_args(argv)

    raw = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = _parse_yaml_or_json(f.read()) or {}

    cfg = ProxyConfig.from_yaml(raw)
    if args.listen:
        host, _, port = args.listen.partition(":")
        cfg.listen_host, cfg.listen_port = host or cfg.listen_host, int(port or cfg.listen_port)
    for item in args.upstream:
        name, _, url = item.partition("=")
        cfg.upstreams[name] = url
    if args.default_upstream:
        cfg.default_upstream = args.default_upstream
    if args.force_non_stream:
        cfg.force_non_stream = True

    rules = load_rules(raw) if raw.get("faults") and not args.start_without_faults else []
    log_path = args.log or os.getenv("FAULT_LOG")
    engine = ProxyFaultEngine(rules=rules, log_path=log_path)

    app = build_app(cfg, engine)
    print(f"[proxy-pod] listening on {cfg.listen_host}:{cfg.listen_port} | "
          f"upstreams={cfg.upstreams} default={cfg.default_upstream} | rules={len(rules)}")
    web.run_app(app, host=cfg.listen_host, port=cfg.listen_port, print=None)


if __name__ == "__main__":
    main()
