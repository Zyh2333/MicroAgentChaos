
import json
import time
import os
import pymysql
import requests
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from collections import Counter

TASK_STATUS_MAP = {
    0: "New",
    1: "Running",
    2: "Succeeded",
    3: "Failed",
    4: "Canceled",
    5: "Paused",
    6: "Manual assistance",
    7: "Completed",
    8: "Completion failed"
}


DB_CONFIG = {
    'host': '192.168.31.15',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'taskScheduling',
    'charset': 'utf8mb4'
}
FILE_SERVER_BASE_URL = "http://192.168.31.15:12104/api/sys/storage/file/download"  # Replace with the actual file server address.
# SAVE_DIR = "./task_files_20260524_prod"  # Root directory for downloaded files.
# CHECKPOINT_FILE = os.path.join(SAVE_DIR, "checkpoint.json")
REPORT_INTERVAL = 1000

DOWNLOAD_COUNT_MAP = {}


task_total_time = [0]
task_total_token = [0]
agent_subs_nums = [0]
tbot_subs_nums = [0]
agent_subs_map = Counter()
tbot_subs_map = Counter()
task_trace_map = Counter()
task_time_map = Counter()
task_token_map = Counter()
agent_vertex_time = [0]
rpa_vertex_time = [0]
agent_vertex_token = [0]
rpa_vertex_token = [0]
agent_vertex_nums = [0]
rpa_vertex_nums = [0]
success_agent_vertex_map = Counter()
fail_agent_vertex_map = Counter()
success_rpa_vertex_map = Counter()
fail_rpa_vertex_map = Counter()
contribution_agent_counter = Counter()
contribution_rpa_counter = Counter()
contribution_agent_map = Counter()
contribution_rpa_map = Counter()
success_task_map = [2, 7]  # Status codes treated as successful.
result_location = ['description', 'title']  # Fields used to locate a task result.

COUNTER_STATE_NAMES = [
    "agent_subs_map",
    "tbot_subs_map",
    "task_trace_map",
    "task_time_map",
    "task_token_map",
    "success_agent_vertex_map",
    "fail_agent_vertex_map",
    "success_rpa_vertex_map",
    "fail_rpa_vertex_map",
    "contribution_agent_counter",
    "contribution_rpa_counter",
    "contribution_agent_map",
    "contribution_rpa_map",
]

LIST_STATE_NAMES = [
    "task_total_time",
    "task_total_token",
    "agent_subs_nums",
    "tbot_subs_nums",
    "agent_vertex_time",
    "rpa_vertex_time",
    "agent_vertex_token",
    "rpa_vertex_token",
    "agent_vertex_nums",
    "rpa_vertex_nums",
]


def record_vertex(vertex, success):

    if success:
        if 'Group' in vertex:
            success_agent_vertex_map[vertex] += 1
        else:
            success_rpa_vertex_map[vertex] += 1
    else:
        if 'Group' in vertex:
            fail_agent_vertex_map[vertex] += 1
        else:
            fail_rpa_vertex_map[vertex] += 1


def add_vertex(vertex, cost_time, token):
    if cost_time > 10000:
        cost_time /= 1000

    if 'Group' in vertex:
        agent_vertex_nums[0] += 1
        agent_vertex_time[0] += min(3600, cost_time)
        agent_vertex_token[0] += token
    else:
        rpa_vertex_nums[0] += 1
        rpa_vertex_time[0] += min(3600, cost_time)
        rpa_vertex_token[0] += token


def similarity_with_length(current, target):
    if current is None or target is None:
        return 0.0
    current = str(current).replace(" ", "")
    target = str(target).replace(" ", "")
    cnt_current = Counter(current)
    cnt_target = Counter(target)
    intersection = cnt_current & cnt_target
    union = cnt_current | cnt_target
    if not union:
        return 0.0
    return sum(intersection.values()) / sum(union.values())


def similarity_ignore_length(current, target):
    if current is None or target is None:
        return 0.0
    current = str(current).replace(" ", "")
    target = str(target).replace(" ", "")
    cnt_target = Counter(target)
    all_count = sum(cnt_target.values())
    if all_count == 0:
        return 0.0
    for c in current:
        if c in cnt_target:
            cnt_target[c] -= 1
            if cnt_target[c] == 0:
                del cnt_target[c]
    return 1 - sum(cnt_target.values()) / all_count


def add_contribution(front_contribution, final_output, group, span):
    cur_value = get_value(span)
    cur_contribution = 0
    cur_contribution += similarity_with_length(cur_value, final_output) * 0.8
    cur_contribution += similarity_ignore_length(cur_value, final_output) * 0.2
    if 'Group' in group:
        contribution_agent_counter[group] += 1
        contribution_agent_map[group] += max(cur_contribution - front_contribution, 0)
    else:
        contribution_rpa_counter[group] += 1
        contribution_rpa_map[group] += max(cur_contribution - front_contribution, 0)
    return max(front_contribution, cur_contribution)


def get_value(span):
    res = None
    if 'results' in span:
        for result in span['results']:
            # if 'description' in result and 'result' in result['description']:
            #     if 'value' in result:
            #         return result['value']
            # if 'name' in result and ('result' in result['name'] or 'flag' in result['name']):
            #     if 'value' in result:
            #         return result['value']
            # if 'title' in result and ('result' in result['title'] or 'flag' in result['title']):
            #     if 'value' in result:
            #         return result['value']
            if 'value' in result:
                res = result['value']
    return res


def my_merge(my_map):
    if 7 in my_map:
        my_map[2] += my_map[7]
        my_map.pop(7)
    if 8 in my_map:
        my_map[3] += my_map[8]
        my_map.pop(8)
    return my_map


def overview(tasks):
    status_counter = Counter(task['status'] for task in tasks)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    overview_path = os.path.join(SAVE_DIR, "overview.txt")

    with open(overview_path, "w", encoding="utf-8") as f:
        f.write(f"Total tasks: {len(tasks)}\n\n")
        f.write("Tasks by status:\n")
        for status_code in sorted(TASK_STATUS_MAP):
            count = status_counter.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count}\n")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()

    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        checkpoint = json.load(f)

    for name in LIST_STATE_NAMES:
        if name in checkpoint:
            globals()[name][0] = checkpoint[name]

    int_key_counters = {
        "agent_subs_map",
        "tbot_subs_map",
        "task_trace_map",
        "task_time_map",
        "task_token_map",
    }
    for name in COUNTER_STATE_NAMES:
        if name not in checkpoint:
            continue
        key_parser = int if name in int_key_counters else str
        globals()[name].clear()
        globals()[name].update({key_parser(k): v for k, v in checkpoint[name].items()})

    processed_task_ids = set(str(task_id) for task_id in checkpoint.get("processed_task_ids", []))
    print(f"Checkpoint found; completed tasks: {len(processed_task_ids)}")
    return processed_task_ids


def save_checkpoint(processed_task_ids):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    checkpoint = {
        "processed_task_ids": sorted(processed_task_ids),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name in LIST_STATE_NAMES:
        checkpoint[name] = globals()[name][0]
    for name in COUNTER_STATE_NAMES:
        checkpoint[name] = dict(globals()[name])

    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CHECKPOINT_FILE)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator


def write_analyze_reports(tasks_num, task_map, processed_count):
    analyze_dir = os.path.join(SAVE_DIR, "analyze")
    os.makedirs(analyze_dir, exist_ok=True)

    with open(os.path.join(analyze_dir, "task_cost_overview.txt"), "w", encoding="utf-8") as f:
        f.write(f"Processed tasks: {processed_count} / {tasks_num}\n")
        f.write(f"Average task duration: {task_total_time[0]} / {tasks_num} = {safe_ratio(task_total_time[0], tasks_num)}\n")
        f.write(f"Average tokens per task: {task_total_token[0]} / {tasks_num} = {safe_ratio(task_total_token[0], tasks_num)}\n")
    with open(os.path.join(analyze_dir, "subtask_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total agent subtasks: {agent_subs_nums[0]}\n")
        f.write("Agent subtasks by status:\n")
        for status_code in sorted(agent_subs_map):
            count = agent_subs_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {agent_subs_nums[0]}"
                    f" = {safe_ratio(count, agent_subs_nums[0])}\n")
        f.write(f"\nTotal tbot subtasks: {tbot_subs_nums[0]}\n")
        f.write("Tbot subtasks by status:\n")
        for status_code in sorted(tbot_subs_map):
            count = tbot_subs_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {tbot_subs_nums[0]}"
                    f" = {safe_ratio(count, tbot_subs_nums[0])}\n")
    with open(os.path.join(analyze_dir, "task_detail_analysis.txt"), "w", encoding="utf-8") as f:
        f.write("Average call-chain length by task status:\n")
        for status_code in sorted(task_trace_map):
            count = task_trace_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {len(task_map[status_code])}"
                    f" = {safe_ratio(count, len(task_map[status_code]))}\n")
        f.write("Average duration by task status:\n")
        for status_code in sorted(task_time_map):
            count = task_time_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {len(task_map[status_code])}"
                    f" = {safe_ratio(count, len(task_map[status_code]))}\n")
        f.write("Average token use by task status:\n")
        for status_code in sorted(task_token_map):
            count = task_token_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {len(task_map[status_code])} ="
                    f" {safe_ratio(count, len(task_map[status_code]))}\n")
    with open(os.path.join(analyze_dir, "vertex_overview.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total vertices: {agent_vertex_nums[0] + rpa_vertex_nums[0]}\n")
        f.write(f"Agent vertices: {agent_vertex_nums[0]}\n")
        f.write(f"Agent vertex duration: {agent_vertex_time[0]}\n")
        f.write(f"Agent vertex tokens: {agent_vertex_token[0]}\n")
        f.write(f"Tbot vertices: {rpa_vertex_nums[0]}\n")
        f.write(f"Tbot vertex duration: {rpa_vertex_time[0]}\n")
        f.write(f"Tbot vertex tokens: {rpa_vertex_token[0]}\n")
    with open(os.path.join(analyze_dir, "vertex_counts_and_success_rates.txt"), "w", encoding="utf-8") as f:
        f.write("agentVertexes: \n")
        for vertex in success_agent_vertex_map:
            count = success_agent_vertex_map.get(vertex, 0)
            total = count + fail_agent_vertex_map.get(vertex, 0)
            f.write(f"  [{vertex}]: {total}\n")
            f.write(f"  Success rate: {safe_ratio(count, total)}\n")
        f.write(f"tbotVertexes: \n")
        for vertex in success_rpa_vertex_map:
            count = success_rpa_vertex_map.get(vertex, 0)
            total = count + fail_rpa_vertex_map.get(vertex, 0)
            f.write(f"  [{vertex}]: {total}\n")
            f.write(f"  Success rate: {safe_ratio(count, total)}\n")
        f.write("\nAgent vertex success rate: ")
        f.write(f"{safe_ratio(sum(success_agent_vertex_map.values()), sum(success_agent_vertex_map.values()) + sum(fail_agent_vertex_map.values()))}\n")
        f.write("Tbot vertex success rate: ")
        f.write(f"{safe_ratio(sum(success_rpa_vertex_map.values()), sum(success_rpa_vertex_map.values()) + sum(fail_rpa_vertex_map.values()))}\n")
        f.write("Overall vertex success rate: ")
        success_vertex_count = sum(success_agent_vertex_map.values()) + sum(success_rpa_vertex_map.values())
        vertex_count = success_vertex_count + sum(fail_agent_vertex_map.values()) + sum(fail_rpa_vertex_map.values())
        f.write(f"{safe_ratio(success_vertex_count, vertex_count)}\n")
    with open(os.path.join(analyze_dir, "vertex_contributions.txt"), "w", encoding="utf-8") as f:
        f.write("agent vertex: \n")
        for vertex in contribution_agent_counter:
            count = contribution_agent_counter.get(vertex, 0)
            f.write(f"  [{vertex}]: {count}\n")
            f.write(f"  Average contribution: {safe_ratio(contribution_agent_map.get(vertex, 0), count)}\n")
        f.write(f"\ntbot vertex: \n")
        for vertex in contribution_rpa_counter:
            count = contribution_rpa_counter.get(vertex, 0)
            f.write(f"  [{vertex}]: {count}\n")
            f.write(f"  Average contribution: {safe_ratio(contribution_rpa_map.get(vertex, 0), count)}\n")


def print_progress(processed_count, tasks_num):
    print(
        f"Processed {processed_count} / {tasks_num} tasks, "
        f"agent subtasks: {agent_subs_nums[0]}, tbot subtasks: {tbot_subs_nums[0]}, "
        f"total vertices: {agent_vertex_nums[0] + rpa_vertex_nums[0]}, "
        f"total task duration: {task_total_time[0]}, total task tokens: {task_total_token[0]}"
    )



def fetch_data():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)


    cursor.execute("SELECT * FROM sys_task")
    tasks = cursor.fetchall()

    task_map = defaultdict(list)
    graph_ids = set()

    for task in tasks:
        status = task['status']
        task_id = task['id']
        task_map[status].append(task)
        graph_ids.add(task['graph_id'])


    cursor.execute("SELECT * FROM sys_agent_subtask")
    agent_subtasks = cursor.fetchall()

    cursor.execute("SELECT * FROM sys_tbot_subtask")
    tbot_subtasks = cursor.fetchall()


    cursor.execute(f"SELECT * FROM sys_graph WHERE id IN ({','.join(['%s'] * len(graph_ids))})", list(graph_ids))
    graph_rows = cursor.fetchall()
    graph_path_map = {g['id']: g['file_path'] for g in graph_rows}

    cursor.close()
    conn.close()

    return tasks, task_map, agent_subtasks, tbot_subtasks, graph_path_map



def download_file(file_path, save_path, status):
    download_count = DOWNLOAD_COUNT_MAP.get(file_path, 0)
    if download_count >= 3:
        return
    DOWNLOAD_COUNT_MAP[file_path] = download_count + 1
    url = f"{FILE_SERVER_BASE_URL}?user=agent&fileName={file_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        if not os.path.exists(save_path + ".json"):
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path + '.json', 'wb') as f:
                    f.write(response.content)
                print(f"Download succeeded: {save_path}")
            else:
                print(f"Download failed: {url}, HTTP status: {response.status_code}")
        with open(save_path + '.json', encoding='utf-8') as load_f:
            load_dict = json.load(load_f)
            if 'time' in load_dict:
                task_total_time[0] += load_dict['time']
                task_time_map[status] += load_dict['time']
            if 'token' in load_dict:
                task_total_token[0] += int(load_dict['token'])
                task_token_map[status] += int(load_dict['token'])
            if 'total_level' in load_dict:
                task_trace_map[status] += load_dict['total_level']
            if 'level_details' in load_dict:
                final_output = None
                front_contribution = 0
                if status in task_trace_map:
                    final_output = " "
                    for level in load_dict['level_details']:
                        if level['level'] >= len(load_dict['level_details']) and 'level_spans' in level:
                            for span in level['level_spans']:
                                tmp_value = get_value(level['level_spans'][span])
                                if tmp_value is not None:
                                    final_output = tmp_value
                                    break
                for level in load_dict['level_details']:
                    route_num = 0
                    if 'level_routes' in level:
                        for route in level['level_routes']:
                            route_num += 1
                            if route_num >= 2:
                                if 'Group' in route and '/' not in route and level['level'] >= len(load_dict['level_details']) and level['level_routes'][route]:
                                    for next_group in level['level_routes'][route]:
                                        record_vertex(next_group.split('/')[0], False)
                                        add_vertex(next_group.split('/')[0], 0, 0)
                                continue
                            group = route.split('/')[0]
                            if final_output is not None and 'level_spans' in level:
                                for span in level['level_spans']:
                                    front_contribution = add_contribution(front_contribution, final_output,
                                                                          group, level['level_spans'][span])
                                    break
                            # A group without a scheduled child fails only at the final level.
                            if 'Group' not in route or '/' in route:
                                # The planner group follows the same final-level rule.
                                if group == 'AgentNetworkPlannerGroup':
                                    if level['level'] >= len(load_dict['level_details']):
                                        record_vertex(group, False)
                                    else:
                                        record_vertex(group, True)
                                    if 'level_spans' in level:
                                        for span in level['level_spans']:
                                            cost_time = level['level_spans'][span].get('time', 0)
                                            token = level['level_spans'][span].get('token', 0)
                                            add_vertex(group, cost_time, token)
                                    else:
                                        add_vertex(group, 0, 0)
                                elif 'level_spans' in level:
                                    for span in level['level_spans']:
                                        cost_time = level['level_spans'][span].get('time', 0)
                                        token = level['level_spans'][span].get('token', 0)
                                        add_vertex(group, cost_time, token)
                                        if 'status' in level['level_spans'][span]:
                                            record_vertex(group, level['level_spans'][span]['status'] in success_task_map)
                                        elif 'results' in level['level_spans'][span]:
                                            success = False
                                            has_success_message = False
                                            for result in level['level_spans'][span]['results']:
                                                location = result_location[0]
                                                if 'description' in result and result['description'] == '':
                                                    location = result_location[1]
                                                # TODO: Verify this result classification against the upstream implementation.
                                                if location in result and '成功' in result[location] and '信息' not in result[location]:
                                                    has_success_message = True
                                                    special = False  # Some services use different success criteria.
                                                    if "".join(filter(str.isdigit, result[location])) != '':
                                                        special = True
                                                    if 'value' in result:
                                                        if special:
                                                            if "".join(filter(str.isdigit, result[location]))[0] in str(result['value']).lower():
                                                                success = True
                                                        elif 'true' in str(result['value']).lower() or '1' in str(result['value']) or '成功' in str(result['value']):
                                                            success = True
                                                    break
                                            if has_success_message:
                                                record_vertex(group, success)
                                            else:
                                                record_vertex(group, False)
                                        else:
                                            # Older tasks treat the presence of level_spans as success.
                                            record_vertex(group, True)
                                else:
                                    record_vertex(group, False)
                                    add_vertex(group, 0, 0)
                            else:
                                if 'level_routes' in level:
                                    cost_time = 0
                                    token = 0
                                    if 'level_spans' in level:
                                        for span in level['level_spans']:
                                            cost_time = level['level_spans'][span].get('time', 0)
                                            token = level['level_spans'][span].get('token', 0)
                                    record_vertex(group, True)
                                    add_vertex(group, cost_time, token)
                                    if level['level'] >= len(load_dict['level_details']) and level['level_routes'][route]:
                                        # Missing next-level services indicate failure of the scheduled service.
                                        for next_group in level['level_routes'][route]:
                                            record_vertex(next_group.split('/')[0], False)
                                            add_vertex(next_group.split('/')[0], 0, 0)
                                else:
                                    record_vertex(group, False)
                                    add_vertex(group, 0, 0)
            else:
                print(load_dict)
    except Exception as e:
        print(f"Download or read error: {url}, error: {e}; retrying...")
        time.sleep(0.5)
        download_file(file_path, save_path, status)


# Candidate task time columns vary across schema versions.
TASK_TIME_KEYS = ['create_time', 'start_time', 'gmt_create', 'begin_time', 'update_time', 'end_time']


def _db_timezone():
    """Timezone used by MySQL DATETIME fields, which carry no timezone info.

    distributed_real writes gmt_create from containers running in UTC. Override
    this when collecting data from a deployment that writes local wall time.
    """
    return ZoneInfo(os.getenv('GRAPH_TASK_TIMEZONE', 'UTC'))


def _task_epoch(task):
    """Parse an epoch-second timestamp from a task row, or return None."""
    for k in TASK_TIME_KEYS:
        v = task.get(k)
        if v is None:
            continue
        if isinstance(v, datetime):
            dt = v
            return dt.replace(tzinfo=_db_timezone()).timestamp() if dt.tzinfo is None else dt.timestamp()
        if isinstance(v, (int, float)):
            # Normalize millisecond timestamps to seconds.
            return float(v) / (1000.0 if v > 1e12 else 1.0)
        try:
            dt = datetime.fromisoformat(str(v).strip().replace('Z', '+00:00'))
            return dt.replace(tzinfo=_db_timezone()).timestamp() if dt.tzinfo is None else dt.timestamp()
        except Exception:
            continue
    return None


def fetch_tasks_in_window(config):
    """Fetch tasks and graph paths from MySQL for the configured time window.

    Return (task_row, graph_path) pairs; include all rows if no time column is known.
    """
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT * FROM sys_task")
    tasks = cursor.fetchall()

    start = getattr(config, 'start', None)
    end = getattr(config, 'end', None)
    task_epochs = [(task, _task_epoch(task)) for task in tasks]
    has_time = any(epoch is not None for _, epoch in task_epochs)

    if has_time and start is not None and end is not None:
        selected = [task for task, epoch in task_epochs
                    if epoch is not None and start <= epoch <= end]
        print(f"Selected {len(selected)} / {len(tasks)} tasks in window [{start}, {end}]")
        if not selected:
            recent = max((epoch for _, epoch in task_epochs if epoch is not None), default=None)
            if recent is not None:
                print(f"Latest task epoch: {recent:.0f}; DB DATETIME timezone: "
                      f"{os.getenv('GRAPH_TASK_TIMEZONE', 'UTC')}")
    else:
        selected = tasks
        print(f"No task time column or window found; processing all {len(selected)} tasks")

    graph_ids = {t['graph_id'] for t in selected if t.get('graph_id') is not None}
    graph_path_map = {}
    if graph_ids:
        cursor.execute(
            f"SELECT * FROM sys_graph WHERE id IN ({','.join(['%s'] * len(graph_ids))})",
            list(graph_ids),
        )
        graph_path_map = {g['id']: g['file_path'] for g in cursor.fetchall()}

    cursor.close()
    conn.close()

    return [(t, graph_path_map.get(t.get('graph_id'))) for t in selected]


def fetch_graph_json(file_path, save_path, retries=3):
    """Download and parse graph.json, reusing a local copy when available."""
    json_path = save_path + '.json'
    url = f"{FILE_SERVER_BASE_URL}?user=agent&fileName={file_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for attempt in range(retries):
        try:
            if not os.path.exists(json_path):
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    print(f"Download failed: {url}, HTTP status: {resp.status_code}")
                    return None
                with open(json_path, 'wb') as f:
                    f.write(resp.content)
            with open(json_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Download or read error: {url}, error: {e} (attempt {attempt + 1}/{retries})")
            time.sleep(0.5)
    return None


SPAN_FIELDS = ('status', 'type', 'role', 'executor', 'error', 'degrade_stage',
               'zone_pref', 'token', 'cost', 'time')
ATTEMPT_FIELDS = ('status', 'executor', 'error', 'degrade_stage', 'zone_pref',
                  'token', 'cost', 'time')


def _span_summary(span):
    """Keep execution metadata and ordered retries without large message bodies."""
    summary = {key: span[key] for key in SPAN_FIELDS if key in span}
    attempts = span.get('attempts') or []
    summary['attempt_count'] = len(attempts)
    summary['attempts'] = [
        {'attempt': index, **{key: item[key] for key in ATTEMPT_FIELDS if key in item}}
        for index, item in enumerate(attempts, start=1)
    ]
    return summary


def parse_call_chain(graph):
    """Build an ordered call chain with caller spans and retry history.

    level_routes maps each caller to its callees at each level. level_spans
    records the logical vertex; a caller span also identifies the real executor
    used by a fallback. Increasing level order yields the execution sequence.
    """
    details = sorted(graph.get('level_details', []) or [], key=lambda d: d.get('level', 0))

    levels = []
    edges = []
    caller_spans = []
    for lvl in details:
        level_no = lvl.get('level')
        routes = lvl.get('level_routes', {}) or {}
        source_spans = lvl.get('level_spans', {}) or {}
        vertexes = list(dict.fromkeys((lvl.get('level_vertexes') or []) +
                                     list(routes) + list(source_spans)))
        spans = {vertex: _span_summary(span) for vertex, span in source_spans.items()}
        for vertex, span in spans.items():
            if span.get('role') == 'caller':
                caller_spans.append({'level': level_no, 'vertex': vertex, **span})
        level_edges = []
        for caller, callees in routes.items():
            if not callees:
                continue
            for callee in callees.keys():
                level_edges.append({'source': caller, 'destination': callee})
                edges.append({'level': level_no, 'source': caller, 'destination': callee})
        levels.append({'level': level_no, 'vertexes': vertexes,
                       'edges': level_edges, 'spans': spans})

    # Record each vertex at its first level to form the ordered path.
    seen = set()
    path = []
    for lvl in levels:
        for v in lvl['vertexes']:
            if v not in seen:
                seen.add(v)
                path.append(v)
    # Append callees missing from level_vertexes to the path tail.
    for e in edges:
        if e['destination'] not in seen:
            seen.add(e['destination'])
            path.append(e['destination'])

    return {
        'trace_id': graph.get('trace_id'),
        'total_level': graph.get('total_level'),
        'vertexes_count': graph.get('vertexes_count'),
        'participated_vertexes': graph.get('participated_vertexes'),
        'path': path,              # Ordered vertex path.
        'chain_length': len(path),
        'levels': levels,          # Vertices and edges by level.
        'edges': edges,            # Flattened edges with level.
        'caller_spans': caller_spans,
    }


def collect_graph(config, out_dir):
    """Collect ordered graph.json call chains for tasks in the time window.

    Write one ordered chain per trace to <out_dir>/call_chains.json.
    Cache source files in <out_dir>/graph_json/ for reuse and inspection.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, 'graph_json')
    os.makedirs(raw_dir, exist_ok=True)

    tasks = fetch_tasks_in_window(config)
    chains = []
    for task, graph_path in tasks:
        if not graph_path:
            continue
        save_path = os.path.join(raw_dir, str(task.get('id')))
        graph = fetch_graph_json(graph_path, save_path)
        if not graph or 'level_details' not in graph:
            continue
        chain = parse_call_chain(graph)
        chain['task_id'] = task.get('id')
        chain['task_status'] = task.get('status')
        chains.append(chain)

    out_file = os.path.join(out_dir, 'call_chains.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(chains, f, ensure_ascii=False, indent=2)
    print(f"Collected {len(chains)} call chains -> {out_file}")
    return chains



def main():
    tasks, task_map, agent_subtasks, tbot_subtasks, graph_path_map = fetch_data()

    overview(tasks)
    tasks_num = len(tasks)
    processed_task_ids = load_checkpoint()
    processed_count = len(processed_task_ids)
    print_progress(processed_count, tasks_num)
    over = False

    for status, tasks in task_map.items():
        if over:
            break
        status_dir = os.path.join(SAVE_DIR, f"status_{status}")
        for task in tasks:
            if over:
                break
            task_id = task['id']
            task_id_key = str(task_id)
            if task_id_key in processed_task_ids:
                continue

            task_dir = os.path.join(status_dir, task_id_key)
            os.makedirs(task_dir, exist_ok=True)


            graph_id = task['graph_id']
            if graph_id in graph_path_map:
                # file_path = graph_path_map[graph_id]
                file_path = task_id_key
                local_path = os.path.join(task_dir, os.path.basename(file_path))
                download_file(graph_path_map[graph_id], local_path, status)



            agent_subs_sorted = sorted(
                filter(lambda s: s['task_id'] == task_id, agent_subtasks),
                key=lambda x: x['start_time']
            )
            tbot_subs_sorted = sorted(
                filter(lambda s: s['task_id'] == task_id, tbot_subtasks),
                key=lambda x: x['start_time']
            )

            subtask_file = os.path.join(task_dir, "subtasks.txt")
            with open(subtask_file, "w", encoding="utf-8") as f:
                f.write("Agent Subtasks:\n")
                for sub in agent_subs_sorted:
                    f.write(f"  [{sub['status']}] {sub['start_time']} {sub['name']} ({sub['type']})\n")
                    agent_subs_nums[0] += 1
                    agent_subs_map[sub['status']] += 1

                f.write("\nTBot Subtasks:\n")
                for sub in tbot_subs_sorted:
                    f.write(f"  [{sub['status']}] {sub['start_time']} {sub['name']} - {sub['flow_id']}\n")
                    tbot_subs_nums[0] += 1
                    tbot_subs_map[sub['status']] += 1

            processed_task_ids.add(task_id_key)
            processed_count += 1
            save_checkpoint(processed_task_ids)
            if processed_count % REPORT_INTERVAL == 0 or processed_count == tasks_num:
                write_analyze_reports(tasks_num, task_map, processed_count)
                print_progress(processed_count, tasks_num)


    write_analyze_reports(tasks_num, task_map, processed_count)
    save_checkpoint(processed_task_ids)

    print("Data processing complete.")


if __name__ == "__main__":
    main()
