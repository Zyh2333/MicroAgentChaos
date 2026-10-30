#!/usr/bin/env python3
"""Manage in-memory rules of the existing chaos/fault-proxy Deployment.

Read the ConfigMap embedded in a full per-fault Kubernetes manifest. The
Kubernetes objects are not applied: kubectl exec calls the control API inside
the proxy container without restarting the proxy or deleting its Service.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def extract_rules(path: Path) -> bytes:
    with path.open(encoding="utf-8") as source:
        documents = [item for item in yaml.safe_load_all(source) if item]
    maps = [item for item in documents if item.get("kind") == "ConfigMap"
            and item.get("metadata", {}).get("name") == "fault-proxy-config"
            and item.get("metadata", {}).get("namespace") == "chaos"]
    if len(maps) != 1:
        raise ValueError("清单中需要恰好一个 chaos/fault-proxy-config ConfigMap")
    config = yaml.safe_load(maps[0]["data"]["faults.yaml"])
    faults = config.get("faults")
    if not isinstance(faults, list) or len(faults) != 1:
        raise ValueError("每个故障清单必须恰好包含一条 faults 规则")
    return yaml.safe_dump({"faults": faults}, allow_unicode=True, sort_keys=False).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="热加载或清空现有 fault-proxy 的规则")
    parser.add_argument("action", choices=("load", "clear", "reset", "stats"))
    parser.add_argument("manifest", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "load":
        if args.manifest is None:
            parser.error("load 需要单故障 K8s YAML 路径")
        try:
            payload = extract_rules(args.manifest)
        except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    else:
        if args.manifest is not None:
            parser.error(f"{args.action} 不接受 YAML 路径")
        payload = b""

    endpoint = {"load": "rules", "clear": "clear", "reset": "reset", "stats": "stats"}[args.action]
    method = "GET" if args.action == "stats" else "POST"
    remote = (
        "import sys, urllib.request\n"
        "data = sys.stdin.buffer.read()\n"
        f"url = 'http://127.0.0.1:18800/__fault__/{endpoint}'\n"
        f"req = urllib.request.Request(url, data=data if '{method}' == 'POST' else None, "
        f"method='{method}', headers={{'Content-Type': 'application/yaml'}})\n"
        "with urllib.request.urlopen(req, timeout=15) as response:\n"
        "    print(response.read().decode('utf-8'))\n"
    )
    cmd = ["kubectl", "--request-timeout=30s", "-n", "chaos", "exec", "-i", "deployment/fault-proxy", "--",
           "python", "-c", remote]
    try:
        result = subprocess.run(cmd, input=payload, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False, timeout=45)
    except subprocess.TimeoutExpired:
        print("fault-proxy 控制命令超过 45 秒，已中止", file=sys.stderr)
        return 124
    if result.returncode:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        return result.returncode
    output = result.stdout.decode("utf-8", errors="replace").strip()
    try:
        obj = json.loads(output)
    except json.JSONDecodeError:
        print(output)
        return 1
    if args.action == "load" and obj.get("loaded") != 1:
        print(output, file=sys.stderr)
        return 1
    if args.action == "clear" and (obj.get("status") != "cleared" or obj.get("rules") != 0):
        print(output, file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
