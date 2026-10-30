#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
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

    start_rfc3339 = timestamp_to_rfc3339(start_timestamp)

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

        containers = [
            c.name
            for c in pod.spec.containers
        ]

        for container_name in containers:

            if container_name == "istio-proxy":
                continue

            output_file = os.path.join(
                output_dir,
                f"{pod_name}_{container_name}.log"
            )

            print(
                f"Collecting: "
                f"{pod_name}/{container_name}"
            )

            try:
                since_seconds = max(
                    1,
                    int(time.time() - start_timestamp)
                )

                logs = v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container_name,
                    timestamps=True,
                    since_seconds=since_seconds,
                    _preload_content=False,
                )

                with open(output_file, "w", encoding="utf-8") as f:

                    for line in logs:
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")

                        line = line.rstrip("\n")

                        if not line:
                            continue

                        try:
                            # Kubernetes timestamps=True prefixes each log line with a timestamp:
                            #
                            # 2026-09-04T06:20:31.123456789Z log content
                            #
                            timestamp_str = line.split(" ", 1)[0]

                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1] + "+00:00"

                            # datetime.fromisoformat accepts at most microsecond precision.
                            if "." in timestamp_str:
                                main_part, fraction = timestamp_str.split(".", 1)

                                # The fractional part may include a timezone offset:
                                # 012475323+00:00
                                fraction = fraction[:6]

                                timestamp_str = f"{main_part}.{fraction}+00:00"

                            log_dt = datetime.fromisoformat(timestamp_str)
                            log_timestamp = log_dt.timestamp()

                            # Stop once the requested window ends.
                            if log_timestamp > end_timestamp:
                                break

                            # since_seconds is approximate, so enforce the start bound here too.
                            if log_timestamp < start_timestamp:
                                continue

                        except (ValueError, IndexError):
                            # Skip lines without a parseable timestamp.
                            continue

                        f.write(line + "\n")

            except client.exceptions.ApiException as e:
                print(
                    f"Failed: "
                    f"{pod_name}/{container_name}: "
                    f"{e}"
                )


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
