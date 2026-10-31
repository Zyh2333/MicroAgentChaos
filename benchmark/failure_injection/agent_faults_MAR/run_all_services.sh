#!/usr/bin/env bash
# Run all agent_faults_MAR services sequentially, one proxy fault at a time.
set -euo pipefail
total_count="${SAMPLES_PER_FAULT:-1}"
read -r -a faults <<< "${FAULT_TYPES:-f1_unavailable f1_slow f1_network_drop f2_auth f2_quota f3_missing_task_id f3_response_drop f4_repeated_execution f4_truncated_result}"
services=(
    agent-network-marble-research
    agent-network-marble-web
    agent-network-marble-coding
    agent-network-marble-database
    agent-network-marble-world
    agent-network-marble-werewolf
    agent-network-marble-minecraft
    agent-network-planner
    agent-network-summarizer
)
export LOCUST_USERS="${LOCUST_USERS:-64}"
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
export INJECT_FAULT="${INJECT_FAULT:-true}"
if [[ -z "${LOCUST_HOST:-}" ]]; then
  : "${HOST_IP:?Set HOST_IP to a Kubernetes node address, or set LOCUST_HOST}"
  export LOCUST_HOST="http://${HOST_IP}:${PLANNER_NODEPORT:-31882}"
fi
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${INJECT_FAULT:-true}" =~ ^(false|FALSE|0|no|NO|off|OFF)$ ]]; then
  modes=("")
else
  case "${FAULT_MODE:-}" in
    "") modes=(single persistent intermittent burst) ;;
    single|persistent|intermittent|burst) modes=("$FAULT_MODE") ;;
    *) echo "FAULT_MODE must be single, persistent, intermittent, or burst" >&2; exit 2 ;;
  esac
fi
for service in "${services[@]}"; do
  echo "########## $service ##########"
  for mode in "${modes[@]}"; do
    if [[ -n "$mode" ]]; then
      echo "########## FAULT_MODE=$mode ##########"
      FAULT_MODE="$mode" bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
    else
      bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
    fi
  done
done
