#!/usr/bin/env bash
# Run all agent_faults_MDOC services sequentially, one proxy fault at a time.
set -euo pipefail
total_count="${SAMPLES_PER_FAULT:-1}"
read -r -a faults <<< "${FAULT_TYPES:-f1_unavailable f1_slow f1_network_drop f2_auth f2_quota f3_missing_task_id f3_response_drop f4_no_tool_calls f4_truncated_result}"
services=(
    agent-network-planner
    agent-network-csv-gen
    agent-network-direction
    agent-network-excel-gen
    agent-network-excel-parsing
    agent-network-image
    agent-network-image-gen
    agent-network-ocr
    agent-network-pdf-gen
    agent-network-pdf-parsing
    agent-network-summarizer
    agent-network-word-gen
    agent-network-word-parsing
)
export LOCUST_USERS="${LOCUST_USERS:-5}"
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
export INJECT_FAULT="${INJECT_FAULT:-true}"
if [[ -z "${LOCUST_HOST:-}" ]]; then
  : "${HOST_IP:?Set HOST_IP to a Kubernetes node address, or set LOCUST_HOST}"
  export LOCUST_HOST="http://${HOST_IP}:${PLANNER_NODEPORT:-31882}"
fi
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for service in "${services[@]}"; do
  echo "########## $service ##########"
  bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
done
