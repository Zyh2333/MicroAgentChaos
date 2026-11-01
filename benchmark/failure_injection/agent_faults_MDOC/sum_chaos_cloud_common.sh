#!/usr/bin/env bash
# Run global proxy fault scenarios across all agent services.
set -euo pipefail
service="common"
total_count="${SAMPLES_PER_FAULT:-1}"
# read -r -a faults <<< "${FAULT_TYPES:-f1_unavailable f1_slow f1_network_drop f2_auth f2_quota f3_missing_task_id f3_response_drop f4_repeated_execution f4_truncated_result}"
read -r -a faults <<< "${FAULT_TYPES:-f1_unavailable f1_slow f1_network_drop f3_missing_task_id f3_response_drop}"
export LOCUST_USERS="${LOCUST_USERS:-64}"
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
export INJECT_FAULT="${INJECT_FAULT:-true}"
if [[ -z "${LOCUST_HOST:-}" ]]; then
  : "${HOST_IP:?Set HOST_IP to a Kubernetes node address, or set LOCUST_HOST}"
  export LOCUST_HOST="http://${HOST_IP}:${PLANNER_NODEPORT:-31882}"
fi
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${INJECT_FAULT:-true}" =~ ^(false|FALSE|0|no|NO|off|OFF)$ ]]; then
  exec bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
fi
case "${FAULT_MODE:-}" in
  "") modes=(single persistent intermittent burst) ;;
  single|persistent|intermittent|burst) modes=("$FAULT_MODE") ;;
  *) echo "FAULT_MODE must be single, persistent, intermittent, or burst" >&2; exit 2 ;;
esac
for mode in "${modes[@]}"; do
  echo "########## FAULT_MODE=$mode ##########"
  FAULT_MODE="$mode" bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
done
