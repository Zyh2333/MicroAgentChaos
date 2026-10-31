#!/usr/bin/env bash
#
# Generic chaos data-collection orchestrator.
#
# For a given service it runs N samples of every requested fault type. Each
# sample is one complete 20-minute data-collection window with this timeline:
#
#   [ 0m ,  5m)  baseline        : no load, no fault
#   [ 5m , 10m)  load injection  : locust headless, 5 min
#   [ 7m , 10m)  fault injection : chaos-mesh, 3 min  (starts 2 min after load)
#   [10m , 20m)  recovery        : load stopped, fault removed, keep observing
#
# For every sample a machine-parseable label block is appended to
#   <dir>/<service>_label.txt
# containing the root cause and every phase boundary (human time + epoch),
# so downstream tooling only has to parse this file to know, for each window,
# the root-cause service/fault and the exact time ranges to pull metrics for.
#
# Usage:
#   bash chaos_service.sh <service> [sample_count] [fault_type ...]
#   INJECT_FAULT=false bash chaos_service.sh <service> [sample_count]
#
#   sample_count  defaults to 3
#   fault_type    defaults to: cpu_load mem_load net_latency pod_failure pod_kill
#                 (each must have a matching <service>/<fault_type>.yaml)
#   INJECT_FAULT defaults to true. When false, no Chaos Mesh YAML is applied
#                and N healthy/normal windows are collected instead.

service="$1"
total_count="${2:-3}"
if [ "$#" -ge 2 ]; then shift 2; else shift 1; fi
fault_types=("$@")
if [ "${#fault_types[@]}" -eq 0 ]; then
    fault_types=(cpu_load mem_load net_latency pod_failure pod_kill)
fi

# ---- fault injection switch ------------------------------------------------
# Accepted true values: true/1/yes/on; false values: false/0/no/off.
case "${INJECT_FAULT:-true}" in
    true|TRUE|1|yes|YES|on|ON) INJECT_FAULT=true ;;
    false|FALSE|0|no|NO|off|OFF) INJECT_FAULT=false ;;
    *)
        echo "INJECT_FAULT must be true or false (also accepts 1/0, yes/no, on/off)." >&2
        exit 2
        ;;
esac

# Normal-data mode performs one healthy window per sample instead of repeating
# the same healthy window once for every configured fault type.
if [ "$INJECT_FAULT" = "false" ]; then
    fault_types=(normal)
fi

dir="$(cd "$(dirname "$0")" && pwd)"

# Project root holds .venv; benchmark root holds data-collector/ and data/.
REPO_ROOT="$(cd "$dir/../../.." && pwd)"
BENCHMARK_ROOT="$REPO_ROOT/benchmark"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
COLLECTOR="$BENCHMARK_ROOT/data-collector/Main.py"

# ---- load (locust) configuration -------------------------------------------
# All overridable via environment variables, e.g. LOCUST_USERS=10 bash ...
LOCUST_HOST="${LOCUST_HOST:-http://${HOST_IP:?Set HOST_IP or LOCUST_HOST}:35696}"
LOCUST_USERS="${LOCUST_USERS:-3}"
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
# Use the mix workload to run intent and original-bench users at their configured weights.
LOCUSTFILE="${LOCUSTFILE:-$dir/../locustfile_MARBLEBench_mix.py}"
# Pass the intent:original workload ratio to the two Locust User classes.
# Export these values for Locust; enough users are needed to reflect the ratio.
export MARBLE_WEIGHT_INTENT="${MARBLE_WEIGHT_INTENT:-1}"
export MARBLE_WEIGHT_ORIG="${MARBLE_WEIGHT_ORIG:-1}"
# Prefix for each window output directory (including task.log), relative to benchmark/data/.
if [ "$INJECT_FAULT" = "true" ]; then
    default_data_prefix="MARBLEBench/abnormal/load-${LOCUST_USERS}"
else
    default_data_prefix="MARBLEBench/normal/load-${LOCUST_USERS}"
fi
DATA_PREFIX="${DATA_PREFIX:-$default_data_prefix}"

# Store labels beside the data at data/<DATA_PREFIX>/<service>_label.txt.
label_file="$BENCHMARK_ROOT/data/${DATA_PREFIX}/${service}_label.txt"
mkdir -p "$BENCHMARK_ROOT/data/${DATA_PREFIX}"

# ---- window timing (seconds) -----------------------------------------------
BASELINE=$((5 * 60))    # no-load baseline before load starts
PRE_FAULT=$((2 * 60))   # load-only, before fault is injected
FAULT=$((3 * 60))       # fault duration
RECOVERY=$((10 * 60))   # post-fault observation (no load, no fault)
LOAD_DURATION=$((PRE_FAULT + FAULT))  # load runs from 5m mark until fault ends

# Best-effort cleanup so an interrupt never leaves load or chaos running.
locust_pid=""
current_yaml=""
cleanup() {
    # Clear the trap before cleanup: network latency can stall the Chaos Mesh finalizer.
    # A second Ctrl+C can then stop the script instead of reentering cleanup.
    trap - INT TERM
    [ -n "$locust_pid" ] && kill "$locust_pid" 2>/dev/null
    # --wait=false avoids blocking on the Chaos Mesh finalizer.
    [ -n "$current_yaml" ] && kubectl delete -f "$current_yaml" -n chaos-mesh --ignore-not-found --wait=false 2>/dev/null
    kubectl delete -f "$dir/deployments.yaml" --ignore-not-found --wait=false 2>/dev/null
}
trap 'echo "interrupted, cleaning up..."; cleanup; exit 130' INT TERM

run_window() {
    fault="$1"
    idx="$2"
    run_id="${service}_${fault}_${idx}"
    yaml=""
    if [ "$INJECT_FAULT" = "true" ]; then
        yaml="$dir/$service/${fault}.yaml"
        if [ ! -f "$yaml" ]; then
            echo "!!! [$run_id] missing yaml: $yaml -- skipped"
            return
        fi
    fi
    current_yaml="$yaml"

    # per-window data output dir (relative to benchmark/data/) + task log path
    user_dir="${DATA_PREFIX}/${run_id}"
    data_dir="$BENCHMARK_ROOT/data/${user_dir}"
    task_log="$data_dir/task.log"
    mkdir -p "$data_dir"

    # (re)deploy the target services and wait until they are ready, so the
    # baseline window measures a healthy service rather than pod cold-start.
    kubectl apply -f "$dir/deployments.yaml" 2>/dev/null
    echo "    [$run_id] waiting for deployments to be ready..."
    kubectl rollout status -f "$dir/deployments.yaml" --timeout=600s 2>/dev/null

    sleep 60

    echo ">>> [$run_id] window start $(date +'%Y-%m-%d %H:%M:%S')"

    # window_start
    ws_e=$(date +%s); ws_h="$(date +'%Y-%m-%d %H:%M:%S')"

    # phase 0: baseline (no load, no fault)
    sleep "$BASELINE"

    # phase 1: start load (headless, self-terminates after LOAD_DURATION)
    # FAULT_SERVICE biases Locust traffic toward tasks using the affected service.
    ls_e=$(date +%s); ls_h="$(date +'%Y-%m-%d %H:%M:%S')"
    FAULT_SERVICE="$service" locust -f "$LOCUSTFILE" --host="$LOCUST_HOST" --headless \
        --only-summary \
        -u "$LOCUST_USERS" -r "$LOCUST_SPAWN_RATE" -t "${LOAD_DURATION}s" \
        >> "$task_log" 2>&1 &
    locust_pid=$!
    echo "    [$run_id] load started (pid $locust_pid) at $ls_h -> $task_log"
    sleep "$PRE_FAULT"

    # pod_failure and pod_kill lose the old container logs on restart.
    # Snapshot logs before injection; Main.py --merge-prefault restores logs
    # from pods that disappeared by the final collection.
    prefault_tmp=""
    # case "$fault" in
    #     pod_failure|pod_kill)
    #         prefault_tmp=".log_prefault_tmp"
    #         echo "    [$run_id] pre-fault log snapshot (pod will restart; capture logs first)"
    #         ( cd "$BENCHMARK_ROOT" && "$PYTHON" "$COLLECTOR" --user "$user_dir" \
    #             --start "$ws_e" --end "$(date +%s)" \
    #             --logs-only --log-subdir "$prefault_tmp" ) \
    #             || echo "!!! [$run_id] pre-fault log snapshot failed"
    #         ;;
    # esac

    # phase 2: inject fault, or keep the equivalent interval healthy when the
    # switch is off so normal and abnormal windows retain the same timeline.
    fs_e=$(date +%s); fs_h="$(date +'%Y-%m-%d %H:%M:%S')"
    if [ "$INJECT_FAULT" = "true" ]; then
        kubectl apply -f "$yaml" -n chaos-mesh
        echo "    [$run_id] fault injected at $fs_h"
    else
        echo "    [$run_id] normal observation interval started at $fs_h (no fault YAML)"
    fi
    sleep "$FAULT"

    # fault/normal observation interval end
    if [ "$INJECT_FAULT" = "true" ]; then
        # Network latency can delay Chaos Mesh finalization; delete asynchronously
        # so recovery can finish in the background.
        kubectl delete -f "$yaml" -n chaos-mesh --ignore-not-found --wait=false
    fi
    fe_e=$(date +%s); fe_h="$(date +'%Y-%m-%d %H:%M:%S')"
    if [ "$INJECT_FAULT" = "true" ]; then
        echo "    [$run_id] fault removed at $fe_h"
    else
        echo "    [$run_id] normal observation interval ended at $fe_h"
    fi

    # load stops at the same moment (locust -t elapsed); make sure it is gone
    wait "$locust_pid" 2>/dev/null
    locust_pid=""
    lstop_e=$(date +%s); lstop_h="$(date +'%Y-%m-%d %H:%M:%S')"

    # phase 3: recovery observation
    sleep "$RECOVERY"
    we_e=$(date +%s); we_h="$(date +'%Y-%m-%d %H:%M:%S')"
    current_yaml=""

    # append the label block
    {
        echo "=== $run_id ==="
        echo "service:      $service"
        echo "fault_type:   $fault"
        echo "fault_injection: $INJECT_FAULT"
        if [ "$INJECT_FAULT" = "true" ]; then
            echo "root_cause:   $service/$fault"
        else
            echo "root_cause:   none"
        fi
        echo "sample:       $idx"
        echo "load_users:   $LOCUST_USERS"
        echo "window_start: $ws_h  ($ws_e)"
        echo "load_start:   $ls_h  ($ls_e)"
        echo "fault_start:  $fs_h  ($fs_e)"
        echo "fault_end:    $fe_h  ($fe_e)"
        echo "load_stop:    $lstop_h  ($lstop_e)"
        echo "window_end:   $we_h  ($we_e)"
        echo "window_range: $ws_e $we_e"
        echo "data_dir:     data/${user_dir}"
        echo "task_log:     data/${user_dir}/task.log"
        case "$fault" in
            pod_failure|pod_kill)
                echo "log_note:     Pod restarted; pre-fault logs of disappeared pods are in log/*_log_prefault.log" ;;
        esac
        echo ""
    } >> "$label_file"

    # Collect this window before deleting the deployment and its pods.
    # Restore pre-fault logs for pods removed by pod_failure or pod_kill.
    echo "    [$run_id] collecting window data before teardown -> data/${user_dir}"
    merge_arg=()
    [ -n "$prefault_tmp" ] && merge_arg=(--merge-prefault "$prefault_tmp")
    ( cd "$BENCHMARK_ROOT" && "$PYTHON" "$COLLECTOR" --user "$user_dir" --start "$ws_e" --end "$we_e" \
        "${merge_arg[@]}" ) \
        || echo "!!! [$run_id] data collection failed (see output above)"

    kubectl delete -f "$dir/deployments.yaml" 2>/dev/null
    echo "<<< [$run_id] window done $we_h"
}

echo "### service=$service  samples=$total_count  inject_fault=$INJECT_FAULT  windows=[${fault_types[*]}]"
for fault in "${fault_types[@]}"; do
    for ((idx = 1; idx <= total_count; idx++)); do
        run_window "$fault" "$idx"
    done
done
echo "### all windows finished for $service -> $label_file"
