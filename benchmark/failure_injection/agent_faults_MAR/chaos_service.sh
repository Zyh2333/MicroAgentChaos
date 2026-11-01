#!/usr/bin/env bash
# Collect baseline -> load -> one proxy fault -> recovery for one Agent Service.
set -euo pipefail

usage() {
  echo "usage: $0 <service> [samples_per_fault] [f1_unavailable ... f4_truncated_result]" >&2
  exit 2
}
[[ $# -ge 1 ]] || usage
service="$1"
total_count="${2:-1}"
[[ "$total_count" =~ ^[1-9][0-9]*$ ]] || usage
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
fault_types=("$@")
if [[ ${#fault_types[@]} -eq 0 ]]; then
  fault_types=(f1_unavailable f1_slow f1_network_drop f2_auth f2_quota \
               f3_missing_task_id f3_response_drop f4_repeated_execution f4_truncated_result)
fi

case "${INJECT_FAULT:-true}" in
  true|TRUE|1|yes|YES|on|ON) inject_fault=true ;;
  false|FALSE|0|no|NO|off|OFF) inject_fault=false; fault_types=(normal) ;;
  *) echo "INJECT_FAULT must be true or false" >&2; exit 2 ;;
esac

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deployments_file="${DEPLOYMENTS_FILE:-$dir/deployments.yaml}"
case "$(basename "$dir")" in
  agent_faults_MAR) dataset=MARBLEBench; default_locustfile="$dir/../locustfile_agent_faults.py" ;;
  agent_faults_MDOC) dataset=MDOC; default_locustfile="$dir/../locustfile_agent_faults.py" ;;
  *) echo "unexpected script directory: $dir" >&2; exit 2 ;;
esac
[[ -d "$dir/$service" ]] || { echo "unknown service: $service" >&2; exit 2; }
mode_for_fault() {
  if [[ -n "${FAULT_MODE:-}" ]]; then
    case "$FAULT_MODE" in
      single|persistent|intermittent|burst) printf '%s\n' "$FAULT_MODE" ;;
      *) echo "FAULT_MODE must be single, persistent, intermittent, or burst" >&2; return 2 ;;
    esac
    return
  fi
  case "$1" in
    f1_network_drop|f3_response_drop) echo intermittent ;;
    f4_no_tool_calls|f4_truncated_result) echo single ;;
    *) echo persistent ;;
  esac
}
if [[ "$inject_fault" == true ]]; then
  for fault in "${fault_types[@]}"; do
    mode="$(mode_for_fault "$fault")"
    [[ "$fault" =~ ^f[1-4]_[a-z0-9_]+$ && -f "$dir/$service/$mode/$fault.yaml" ]] || {
      echo "missing/invalid fault YAML: $dir/$service/$mode/$fault.yaml" >&2; exit 2;
    }
  done
fi

repo_root="$(cd "$dir/../../.." && pwd)"
benchmark_root="$repo_root/benchmark"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
collector="$benchmark_root/data-collector/Main.py"
proxy_rules="$dir/../proxy_rules.py"
locustfile="${LOCUSTFILE:-$default_locustfile}"
locust_users="${LOCUST_USERS:-64}"
locust_spawn_rate="${LOCUST_SPAWN_RATE:-10}"
locust_stop_timeout="${LOCUST_STOP_TIMEOUT:-600}"
[[ "$locust_stop_timeout" =~ ^[1-9][0-9]*$ ]] || { echo "LOCUST_STOP_TIMEOUT must be a positive integer" >&2; exit 2; }
if [[ -n "${LOCUST_HOST:-}" ]]; then
  locust_host="$LOCUST_HOST"
else
  [[ -n "${HOST_IP:-}" ]] || { echo "set LOCUST_HOST or HOST_IP" >&2; exit 2; }
  locust_host="http://${HOST_IP}:${PLANNER_NODEPORT:-31882}"
fi
if [[ "$inject_fault" == true ]]; then
  default_prefix="$dataset/agent_faults/abnormal/load-${locust_users}"
else
  default_prefix="$dataset/agent_faults/normal/load-${locust_users}"
fi
data_base_prefix="${DATA_PREFIX:-$default_prefix}"
if [[ "$service" == common ]]; then
  [[ "$data_base_prefix" == */common ]] || data_base_prefix="$data_base_prefix/common"
fi

# Match the old MAR 20-minute window by default; unlike the old MDOC script,
# FAULT_SECONDS is nonzero so a fault can actually be observed.
warmup_seconds="${WARMUP_SECONDS:-60}"
baseline_seconds="${BASELINE_SECONDS:-300}"
pre_fault_seconds="${PRE_FAULT_SECONDS:-120}"
fault_seconds="${FAULT_SECONDS:-180}"
recovery_seconds="${RECOVERY_SECONDS:-600}"
for seconds in "$warmup_seconds" "$baseline_seconds" "$pre_fault_seconds" "$fault_seconds" "$recovery_seconds"; do
  [[ "$seconds" =~ ^[0-9]+$ ]] || { echo "window durations must be nonnegative integers" >&2; exit 2; }
done
(( fault_seconds > 0 )) || { echo "FAULT_SECONDS must be positive" >&2; exit 2; }
locust_run_seconds=$((pre_fault_seconds + fault_seconds + locust_stop_timeout + 120))
manage_deployments="${MANAGE_DEPLOYMENTS:-true}"

locust_pid=""
fault_active=false
deployed=false
proxy_deployed=false
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ "$fault_active" == true ]]; then
    "$python_bin" "$proxy_rules" clear >/dev/null 2>&1 || true
  fi
  if [[ -n "$locust_pid" ]]; then
    kill -TERM "$locust_pid" 2>/dev/null || true
    wait "$locust_pid" 2>/dev/null || true
  fi
  if [[ "$proxy_deployed" == true ]]; then
    kubectl -n chaos delete deployment/fault-proxy service/fault-proxy configmap/fault-proxy-config \
      --ignore-not-found --wait=false >/dev/null 2>&1 || true
  fi
  if [[ "$deployed" == true && "$manage_deployments" == true ]]; then
    kubectl delete -f "$deployments_file" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

phase() {
  printf '[%s] %s: %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$1" "$2"
}

wait_for_agent_deployments() {
  local run_id="$1" deployments deployment deadline remaining
  deployments="$(kubectl -n agent-network get deployments -o name)"
  [[ -n "$deployments" ]] || { echo "no Deployments found in agent-network" >&2; return 1; }
  deadline=$(( $(date +%s) + 600 ))
  phase "$run_id" "waiting for all agent-network Deployments to roll out"
  while IFS= read -r deployment; do
    [[ -n "$deployment" ]] || continue
    remaining=$((deadline - $(date +%s)))
    if (( remaining <= 0 )); then
      echo "timed out waiting for all agent-network Deployments" >&2
      return 1
    fi
    kubectl -n agent-network rollout status "$deployment" --timeout="${remaining}s"
  done <<< "$deployments"
}

run_window() {
  local fault="$1" idx yaml="" mode="" proxy_manifest run_id user_dir data_dir task_log label_file aggregate_label_file label_entry data_prefix sample_parent sample_dir suffix latest_idx=0
  data_prefix="$data_base_prefix"
  if [[ "$inject_fault" == true ]]; then
    mode="$(mode_for_fault "$fault")"
    yaml="$dir/$service/$mode/$fault.yaml"
    data_prefix="$data_base_prefix/$mode"
  fi
  proxy_manifest="${yaml:-$dir/../proxy_pod/deploy/k8s.yaml}"
  sample_parent="$benchmark_root/data/$data_prefix"
  mkdir -p "$sample_parent"
  for sample_dir in "$sample_parent/${service}_${fault}_"*; do
    [[ -d "$sample_dir" ]] || continue
    suffix="${sample_dir#"$sample_parent/${service}_${fault}_"}"
    [[ "$suffix" =~ ^[0-9]+$ ]] || continue
    if (( 10#$suffix > latest_idx )); then latest_idx=$((10#$suffix)); fi
  done
  idx=$((latest_idx + 1))
  while :; do
    run_id="${service}_${fault}_${idx}"
    user_dir="$data_prefix/$run_id"
    data_dir="$benchmark_root/data/$user_dir"
    if mkdir "$data_dir" 2>/dev/null; then break; fi
    if [[ -e "$data_dir" ]]; then
      idx=$((idx + 1))
      continue
    fi
    echo "cannot create sample directory: $data_dir" >&2
    return 1
  done
  task_log="$data_dir/task.log"
  label_file="$benchmark_root/data/$data_prefix/${service}_label.txt"
  aggregate_label_file="$benchmark_root/data/$data_base_prefix/${service}_label.txt"

  phase "$run_id" "applying fault-proxy manifest"
  proxy_deployed=true
  # Remove a leftover proxy from an interrupted or manually started run so its
  # emptyDir logs cannot leak into this sample.
  if kubectl -n chaos get deployment/fault-proxy >/dev/null 2>&1; then
    kubectl -n chaos delete deployment/fault-proxy service/fault-proxy configmap/fault-proxy-config \
      --ignore-not-found --wait=true
  fi
  kubectl apply -f "$proxy_manifest"
  kubectl -n chaos rollout status deployment/fault-proxy --timeout=600s
  if [[ "$manage_deployments" == true ]]; then
    kubectl apply -f "$deployments_file"
    deployed=true
  fi
  wait_for_agent_deployments "$run_id"
  # A clean baseline is required even when the previous run was interrupted.
  phase "$run_id" "clearing proxy rules"
  "$python_bin" "$proxy_rules" clear >/dev/null
  phase "$run_id" "warmup ${warmup_seconds}s"
  sleep "$warmup_seconds"

  local ws_e ws_h ls_e ls_h fs_e fs_h fe_e fe_h lstop_e lstop_h we_e we_h
  ws_e="$(date +%s)"; ws_h="$(date +'%Y-%m-%d %H:%M:%S')"
  phase "$run_id" "baseline ${baseline_seconds}s"
  sleep "$baseline_seconds"

  ls_e="$(date +%s)"; ls_h="$(date +'%Y-%m-%d %H:%M:%S')"
  phase "$run_id" "starting Locust; task log: $task_log"
  AGENT_FAULT_DATASET="$dataset" FAULT_SERVICE="$service" locust -f "$locustfile" --host="$locust_host" --headless \
    --only-summary -u "$locust_users" -r "$locust_spawn_rate" -t "${locust_run_seconds}s" \
    --stop-timeout "$locust_stop_timeout" \
    >> "$task_log" 2>&1 &
  locust_pid=$!
  phase "$run_id" "load without fault ${pre_fault_seconds}s"
  sleep "$pre_fault_seconds"

  if [[ "$inject_fault" == true ]]; then
    phase "$run_id" "injecting $fault"
    "$python_bin" "$proxy_rules" load "$yaml" > "$data_dir/proxy_load.json"
    fault_active=true
    "$python_bin" "$proxy_rules" reset >/dev/null
  fi
  fs_e="$(date +%s)"; fs_h="$(date +'%Y-%m-%d %H:%M:%S')"
  phase "$run_id" "fault window ${fault_seconds}s"
  sleep "$fault_seconds"

  if [[ "$inject_fault" == true ]]; then
    "$python_bin" "$proxy_rules" stats > "$data_dir/proxy_stats.json" || \
      echo "warning: proxy stats unavailable for $run_id" >&2
  fi
  "$python_bin" "$proxy_rules" clear >/dev/null
  fault_active=false
  fe_e="$(date +%s)"; fe_h="$(date +'%Y-%m-%d %H:%M:%S')"
  phase "$run_id" "fault cleared; stopping Locust and waiting up to ${locust_stop_timeout}s for active requests"

  kill -TERM "$locust_pid" 2>/dev/null || true
  wait "$locust_pid" 2>/dev/null || true
  locust_pid=""
  lstop_e="$(date +%s)"; lstop_h="$(date +'%Y-%m-%d %H:%M:%S')"
  phase "$run_id" "recovery ${recovery_seconds}s"
  sleep "$recovery_seconds"
  we_e="$(date +%s)"; we_h="$(date +'%Y-%m-%d %H:%M:%S')"

  label_entry="$({
    echo "=== $run_id ==="
    echo "service:      $service"
    echo "fault_type:   $fault"
    if [[ "$inject_fault" == true ]]; then echo "persistence:  $mode"; fi
    echo "fault_injection: $inject_fault"
    if [[ "$inject_fault" == true ]]; then
      echo "root_cause:   $service/$fault"
    else
      echo "root_cause:   none"
    fi
    echo "sample:       $idx"
    echo "load_users:   $locust_users"
    echo "locust_stop_timeout: ${locust_stop_timeout}s"
    echo "window_start: $ws_h  ($ws_e)"
    echo "load_start:   $ls_h  ($ls_e)"
    echo "fault_start:  $fs_h  ($fs_e)"
    echo "fault_end:    $fe_h  ($fe_e)"
    echo "load_stop:    $lstop_h  ($lstop_e)"
    echo "window_end:   $we_h  ($we_e)"
    echo "window_range: $ws_e $we_e"
    echo "data_dir:     data/${user_dir}"
    echo "task_log:     data/${user_dir}/task.log"
    echo "service_logs: data/${user_dir}/agent-network/log"
    echo "proxy_logs:   data/${user_dir}/chaos/log"
    if [[ "$inject_fault" == true ]]; then
      echo "proxy_stats:   data/${user_dir}/proxy_stats.json"
    fi
    echo
  })"
  if [[ "$label_file" == "$aggregate_label_file" ]]; then
    printf '%s\n\n' "$label_entry" >> "$label_file"
  else
    printf '%s\n\n' "$label_entry" | tee -a "$label_file" "$aggregate_label_file" >/dev/null
  fi

  phase "$run_id" "collecting logs before proxy deletion"
  ( cd "$benchmark_root" && "$python_bin" "$collector" \
      --user "$user_dir" --start "$ws_e" --end "$we_e" ) || \
      echo "warning: data collection failed for $run_id" >&2

  kubectl -n chaos delete deployment/fault-proxy service/fault-proxy configmap/fault-proxy-config \
    --ignore-not-found --wait=true
  proxy_deployed=false
  if [[ "$manage_deployments" == true ]]; then
    kubectl delete -f "$deployments_file" --ignore-not-found --wait=true
    deployed=false
  fi
  echo "completed $run_id -> data/$user_dir"
}

for fault in "${fault_types[@]}"; do
  for ((sample=1; sample<=total_count; sample++)); do
    run_window "$fault"
  done
done
echo "all windows complete for $service -> data/$data_base_prefix"
