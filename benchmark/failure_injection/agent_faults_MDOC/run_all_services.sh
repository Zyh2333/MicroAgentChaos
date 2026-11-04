#!/usr/bin/env bash
# Run all agent_faults_MDOC services sequentially, one proxy fault at a time.
set -euo pipefail
total_count="${SAMPLES_PER_FAULT:-1}"
read -r -a faults <<< "${FAULT_TYPES:-f1_unavailable f1_slow f1_network_drop f2_auth f2_quota f3_missing_task_id f3_response_drop f4_repeated_execution f4_truncated_result}"
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
    agent-network-word-gen
    agent-network-word-parsing
)
export TOPOLOGY_MODE="${TOPOLOGY_MODE:-dynamic}"
case "$TOPOLOGY_MODE" in
  edge-preference)
    [[ -z "${FAULT_ZONE:-}" || "$FAULT_ZONE" == edge ]] || { echo "edge-preference requires FAULT_ZONE=edge" >&2; exit 2; }
    export FAULT_ZONE=edge
    ;;
  cloud-preference)
    [[ -z "${FAULT_ZONE:-}" || "$FAULT_ZONE" == cloud ]] || { echo "cloud-preference requires FAULT_ZONE=cloud" >&2; exit 2; }
    export FAULT_ZONE=cloud
    ;;
  dynamic)
    if [[ -z "${FAULT_ZONE:-}" ]]; then
      if (( RANDOM % 2 )); then FAULT_ZONE=edge; else FAULT_ZONE=cloud; fi
    fi
    export FAULT_ZONE
    ;;
  *) echo "TOPOLOGY_MODE must be edge-preference, cloud-preference, or dynamic" >&2; exit 2 ;;
esac
case "$FAULT_ZONE" in
  edge)
    for i in "${!services[@]}"; do
      services[$i]="${services[$i]}-edge"
    done
    ;;
  cloud) ;;
  *) echo "FAULT_ZONE must be cloud or edge" >&2; exit 2 ;;
esac
export LOCUST_USERS="${LOCUST_USERS:-64}"
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
export INJECT_FAULT="${INJECT_FAULT:-true}"
if [[ "$INJECT_FAULT" =~ ^(true|TRUE|1|yes|YES|on|ON)$ ]]; then
  if [[ "$FAULT_ZONE" == edge ]]; then services+=(common-edge); else services+=(common); fi
fi
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${EDGE_SERVICES:-}" ]]; then
  read -r -a requested_services <<< "${EDGE_SERVICES//,/ }"
  edge_count="${EDGE_SERVICE_SAMPLE_COUNT:-3}"
  [[ "$edge_count" =~ ^[1-9][0-9]*$ ]] || { echo "EDGE_SERVICE_SAMPLE_COUNT must be a positive integer" >&2; exit 2; }
  [[ ${#requested_services[@]} -eq $edge_count ]] || { echo "EDGE_SERVICES must list exactly $edge_count services" >&2; exit 2; }
  selected_services=(agent-network-planner)
  if [[ "$FAULT_ZONE" == edge ]]; then selected_services=(agent-network-planner-edge); fi
  for requested in "${requested_services[@]}"; do
    base="${requested%-edge}"
    [[ "$base" != agent-network-planner && "$base" != common && -d "$dir/$base" ]] || { echo "invalid EDGE_SERVICES entry: $requested" >&2; exit 2; }
    target="$base"
    if [[ "$FAULT_ZONE" == edge ]]; then target="${base}-edge"; fi
    for existing in "${selected_services[@]}"; do
      [[ "$existing" != "$target" ]] || { echo "duplicate EDGE_SERVICES entry: $requested" >&2; exit 2; }
    done
    selected_services+=("$target")
  done
  if [[ "$INJECT_FAULT" =~ ^(true|TRUE|1|yes|YES|on|ON)$ ]]; then
    if [[ "$FAULT_ZONE" == edge ]]; then selected_services+=(common-edge); else selected_services+=(common); fi
  fi
  services=("${selected_services[@]}")
fi
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
