#!/usr/bin/env python3

import argparse
import heapq
import json
import math
import os
import re
import shutil
import signal
import tempfile
import time

from kubernetes import client, config
from kubernetes.stream import stream

from datetime import datetime, timezone


def timestamp_to_rfc3339(timestamp):
    """
    Convert a Unix timestamp to Kubernetes RFC3339 UTC time.
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_logs(
        namespace,
        start_timestamp,
        end_timestamp,
        output_dir,
):
    config.load_kube_config()

    v1 = client.CoreV1Api()

    pods = v1.list_namespaced_pod(
        namespace=namespace
    ).items

    print(f"Namespace: {namespace}")
    print(f"Start: {start_timestamp}")
    print(f"End:   {end_timestamp}")
    print(f"Pods:  {len(pods)}")

    os.makedirs(output_dir, exist_ok=True)

    for pod in pods:

        pod_name = pod.metadata.name
        statuses = (getattr(pod.status, "container_statuses", None) or []) if pod.status else []
        restart_counts = {status.name: status.restart_count for status in statuses}

        containers = [
            c.name
            for c in pod.spec.containers
        ]

        for container_name in containers:

            if container_name == "istio-proxy":
                continue

            # Kubernetes returns only the current container by default. A restart
            # hides its predecessor unless previous=True is collected separately.
            for previous in (False, True):
                if previous and restart_counts.get(container_name, 0) < 1:
                    continue
                suffix = "_previous" if previous else ""
                output_file = os.path.join(
                    output_dir, f"{pod_name}_{container_name}{suffix}.log"
                )
                print(f"Collecting: {pod_name}/{container_name}{suffix}")

                try:
                    since_seconds = max(1, math.ceil(time.time() - start_timestamp))
                    logs = v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=namespace,
                        container=container_name,
                        timestamps=True,
                        since_seconds=since_seconds,
                        previous=previous,
                        _preload_content=False,
                    )

                    with open(output_file, "w", encoding="utf-8") as f:
                        for line in logs:
                            if isinstance(line, bytes):
                                line = line.decode("utf-8", errors="replace")
                            line = line.rstrip("\n")
                            if not line:
                                continue
                            log_timestamp = _kube_log_time(line)
                            if log_timestamp is None:
                                continue
                            if log_timestamp > end_timestamp:
                                break
                            if log_timestamp >= start_timestamp:
                                f.write(line + "\n")

                    # A restarted container may have no logs in this window.
                    # Do not leave an empty previous-instance placeholder.
                    if previous and os.path.getsize(output_file) == 0:
                        os.unlink(output_file)

                except client.exceptions.ApiException as e:
                    print(f"Failed: {pod_name}/{container_name}{suffix}: {e}")


def safe_filename(name):
    """
    Replace characters unsafe for Pod or container filenames.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _kube_log_time(line):
    """Parse the timestamp prefix added by Kubernetes (up to nanoseconds)."""
    token = line.partition(" ")[0]
    token = re.sub(r"(\.\d{6})\d+(?=Z$|[+-]\d\d:\d\d$)", r"\1", token)
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def watch_logs(namespace, start_timestamp, output_dir, interval=5):
    """Persist service logs during a run, including containers that restart.

    Polling is used because a Kubernetes preStop hook does not run for crashes
    or OOM kills. The final collector still fetches the last previous instance.
    """
    config.load_kube_config()
    v1 = client.CoreV1Api()
    os.makedirs(output_dir, exist_ok=True)
    state = {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        try:
            pods = v1.list_namespaced_pod(namespace=namespace).items
            for pod in pods:
                statuses = (getattr(pod.status, "container_statuses", None) or []) if pod.status else []
                restart_counts = {s.name: s.restart_count for s in statuses}
                for container in pod.spec.containers:
                    name = container.name
                    if name == "istio-proxy":
                        continue
                    key = (pod.metadata.name, name)
                    count = restart_counts.get(name, 0)
                    # Start from the observed count: restarts before this watch
                    # began are not restarts between its polls.
                    item = state.setdefault(key, {"second": int(start_timestamp) - 1,
                                                  "seen": set(), "restarts": count})
                    if count > item["restarts"] + 1:
                        print(f"WARNING: {key} restarted {count - item['restarts']} times between polls; "
                              "older container logs may be unavailable", flush=True)
                    # Read the previous instance first, before advancing the
                    # watermark with lines from the new current instance.
                    for previous in (True, False):
                        if previous and count < 1:
                            continue
                        since = max(int(start_timestamp), item["second"] - 1)
                        try:
                            response = v1.read_namespaced_pod_log(
                                name=key[0], namespace=namespace, container=name,
                                timestamps=True,
                                since_seconds=max(1, math.ceil(time.time() - since)),
                                previous=previous, _preload_content=False,
                            )
                            try:
                                path = os.path.join(output_dir, f"{key[0]}_{name}.log")
                                with open(path, "a", encoding="utf-8") as out:
                                    for raw in response:
                                        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                                        line = line.rstrip("\n")
                                        ts = _kube_log_time(line)
                                        if ts is None or ts < start_timestamp:
                                            continue
                                        second = int(ts)
                                        if second < item["second"]:
                                            continue
                                        if second > item["second"]:
                                            item["second"] = second
                                            item["seen"].clear()
                                        if line not in item["seen"]:
                                            out.write(line + "\n")
                                            item["seen"].add(line)
                            finally:
                                response.close()
                        except client.exceptions.ApiException as exc:
                            print(f"WARNING: log poll failed for {key} previous={previous}: {exc}", flush=True)
                    item["restarts"] = count
        except Exception as exc:
            print(f"WARNING: log watch scan failed: {exc}", flush=True)
        if not stopping:
            time.sleep(interval)


def merge_live_logs(final_dir, live_dir, start_timestamp=None, end_timestamp=None):
    """Merge the run-time snapshot and final current/previous logs by timestamp."""
    has_live_dir = os.path.isdir(live_dir)
    if not has_live_dir:
        print(f"WARNING: live log directory is missing: {live_dir}")
    os.makedirs(final_dir, exist_ok=True)
    names = {name for name in os.listdir(live_dir) if name.endswith(".log")} if has_live_dir else set()
    names.update(f"{name[:-13]}.log" for name in os.listdir(final_dir)
                 if name.endswith("_previous.log"))
    for name in sorted(names):
        canonical = os.path.join(final_dir, name)
        previous = os.path.join(final_dir, f"{name[:-4]}_previous.log")
        sources = [path for path in (os.path.join(live_dir, name), canonical, previous)
                   if os.path.isfile(path)]
        if not sources:
            continue
        fd, tmp = tempfile.mkstemp(prefix=".log-merge-", dir=final_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                handles = [open(path, encoding="utf-8") for path in sources]
                try:
                    lines = heapq.merge(*handles, key=lambda line: _kube_log_time(line) or 0)
                    last_second = None
                    seen = set()
                    for line in lines:
                        ts = _kube_log_time(line)
                        if ts is None:
                            continue
                        if start_timestamp is not None and ts < start_timestamp:
                            continue
                        if end_timestamp is not None and ts > end_timestamp:
                            continue
                        second = int(ts)
                        if second != last_second:
                            last_second = second
                            seen.clear()
                        if line not in seen:
                            out.write(line)
                            seen.add(line)
                finally:
                    for handle in handles:
                        handle.close()
            os.replace(tmp, canonical)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    if has_live_dir:
        shutil.rmtree(live_dir)


def collect_proxy_logs(start_timestamp, end_timestamp, output_dir):
    """Collect fault-proxy stdout and its structured fire log for one window.

    The proxy writes fault hits to an emptyDir file, not stdout.  Fetch that
    file while the Pod still exists; Kubernetes container logs alone omit it.
    """
    config.load_kube_config()
    v1 = client.CoreV1Api()
    os.makedirs(output_dir, exist_ok=True)
    pods = v1.list_namespaced_pod(namespace="chaos", label_selector="app=fault-proxy").items
    if not pods:
        print("[proxy logs] no app=fault-proxy Pod found in namespace chaos")
        return

    since_seconds = max(1, math.ceil(time.time() - start_timestamp))
    for pod in pods:
        pod_name = pod.metadata.name
        container_names = {c.name for c in pod.spec.containers}
        statuses = (getattr(pod.status, "container_statuses", None) or []) if pod.status else []
        restart_counts = {s.name: s.restart_count for s in statuses}
        for container_name in sorted(container_names):
            for previous in (False, True):
                if previous and restart_counts.get(container_name, 0) < 1:
                    continue
                suffix = "_previous" if previous else ""
                filename = f"{safe_filename(pod_name)}_{safe_filename(container_name)}{suffix}.log"
                try:
                    raw = v1.read_namespaced_pod_log(
                        name=pod_name, namespace="chaos", container=container_name,
                        timestamps=True, since_seconds=since_seconds, previous=previous,
                    )
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as out:
                        for line in raw.splitlines():
                            ts = _kube_log_time(line)
                            if ts is not None and start_timestamp <= ts <= end_timestamp:
                                out.write(line + "\n")
                    print(f"[proxy logs] collected chaos/{pod_name}/{container_name}{suffix}")
                except Exception as exc:  # Keep collecting other proxy log sources.
                    print(f"[proxy logs] failed chaos/{pod_name}/{container_name}{suffix}: {exc}")

        # Fault hits are written by engine.py to the Deployment's --log path.
        # Use Pod exec before the emptyDir is removed; filter by the JSONL `t` field.
        try:
            raw = stream(
                v1.connect_get_namespaced_pod_exec, pod_name, "chaos",
                container="fault-proxy",
                command=["cat", "/var/log/fault-proxy/fires.jsonl"],
                stderr=False, stdin=False, stdout=True, tty=False,
            )
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            filename = f"{safe_filename(pod_name)}_fires.jsonl"
            hits = 0
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as out:
                for line in raw.splitlines():
                    try:
                        entry = json.loads(line)
                        ts = float(entry["t"])
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                        continue
                    if start_timestamp <= ts <= end_timestamp:
                        out.write(line + "\n")
                        hits += 1
            print(f"[proxy logs] collected {hits} fault hits from chaos/{pod_name}")
        except Exception as exc:  # Exec transports may raise websocket errors.
            print(f"[proxy logs] failed to read fires.jsonl from chaos/{pod_name}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect logs from all pods in a Kubernetes namespace for a time range"
    )

    parser.add_argument(
        "--namespace",
        "-n",
        required=True,
        help="Kubernetes namespace"
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start time, for example: 2026-09-04 10:00:00"
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End time, for example: 2026-09-04 11:00:00"
    )

    parser.add_argument(
        "--output",
        "-o",
        default="./k8s_logs",
        help="Log output directory (default: ./k8s_logs)"
    )

    parser.add_argument(
        "--previous",
        action="store_true",
        help="Also collect logs from previous containers"
    )

    args = parser.parse_args()

    start_time = parse_time(args.start)
    end_time = parse_time(args.end)

    if end_time <= start_time:
        raise ValueError(
            "End time must be later than start time"
        )

    collect_logs(
        namespace=args.namespace,
        start_time=start_time,
        end_time=end_time,
        output_dir=args.output,
        include_previous=args.previous,
    )


def collect_logs_(start_time, end_time, namespace, output_dir):
    collect_logs(
        namespace=namespace,
        start_timestamp=start_time,
        end_timestamp=end_time,
        output_dir=output_dir,
        # include_previous=args.previous,
    )


if __name__ == "__main__":
    main()
