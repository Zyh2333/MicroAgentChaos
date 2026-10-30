#!/usr/bin/env bash
# Run the nine proxy fault scenarios for agent-network-pdf-gen sequentially.
set -euo pipefail
service="agent-network-pdf-gen"
total_count="${SAMPLES_PER_FAULT:-1}"
# read -r -a faults <<< "${FAULT_TYPES:-f1_unavailable f1_slow f1_network_drop f2_auth f2_quota f3_missing_task_id f3_response_drop f4_no_tool_calls f4_truncated_result}"
read -r -a faults <<< "${FAULT_TYPES:-f1_unavailable}"
export LOCUST_USERS="${LOCUST_USERS:-3}"
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
export INJECT_FAULT="${INJECT_FAULT:-true}"
if [[ -z "${LOCUST_HOST:-}" ]]; then
  : "${HOST_IP:?Set HOST_IP to a Kubernetes node address, or set LOCUST_HOST}"
  export LOCUST_HOST="http://${HOST_IP}:${PLANNER_NODEPORT:-31882}"
fi
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
