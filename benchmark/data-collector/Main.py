import sys
import re
import argparse
import shutil

# import pkl_concat
from Config import Config
import MetricCollector
import LogCollector
from handler import Trace, Log
import time
from util.utils import *
import os

if __name__ == "__main__":
    # namespaces = ['bookinfo', 'hipster', 'hipster2', 'cloud-sock-shop', 'horsecoder-test', 'openclaw']
    namespaces = ['agent-network']
    config = Config()


    class Simple:
        def __init__(self, label, begin, end):
            self.label = label
            self.begin = begin
            self.end = end


    def read_label_logs(label_file, simple_list: [Simple]):
        if simple_list is None:
            simple_list = []
        file_path = label_file
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return simple_list
                for line in lines:
                    if 'cpu_' in line or 'mem_' in line or 'net_' in line:
                        root_cause = line.strip()
                        simple = Simple(root_cause, None, None)
                    elif 'start create' in line:
                        begin = line[:18]
                    elif 'finish delete' in line:
                        end = line[:18]
                        simple.begin = timestamp_2_time_string(time_string_2_timestamp(begin) - 360)
                        simple.end = timestamp_2_time_string(time_string_2_timestamp(end) + 360)
                        simple_list.append(simple)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    def parse_label_file(label_file):
        """Parse a fault label file into a list of fault groups.

        Each `=== xxx ===` block describes one service fault; a file can contain several.
        Each group contains window_start, window_end, root_cause, and other fields.
        A parenthesized epoch value is also stored under `<field>_ts`.
        """
        groups = []
        current = None
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Failed to read label file {label_file}: {e}")
            return groups
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('===') and line.endswith('==='):
                if current is not None:
                    groups.append(current)
                current = {'name': line.strip('= ').strip()}
            elif current is not None and ':' in line:
                key, _, val = line.partition(':')
                key, val = key.strip(), val.strip()
                current[key] = val
                m = re.search(r'\((\d+)\)', val)
                if m:
                    current[key + '_ts'] = int(m.group(1))
        if current is not None:
            groups.append(current)
        # Prefer window_range for the full collection window.
        for g in groups:
            parts = g.get('window_range', '').split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                g['window_start_ts'] = int(parts[0])
                g['window_end_ts'] = int(parts[1])
        return groups

    def user_dir_for(label_file, group_name):
        """Build the output directory from the label directory and fault group name."""
        data_root = os.path.abspath('./data')
        label_dir = os.path.dirname(os.path.abspath(label_file))
        rel = os.path.relpath(label_dir, data_root)
        if rel == '.':
            return group_name
        return os.path.join(rel, group_name)

    def collect_for_config(config, namespaces):
        """Collect metrics for the complete window in config."""
        # Proxy fault records live in chaos/emptyDir, outside agent-network stdout.
        try:
            LogCollector.collect_proxy_logs(
                config.start, config.end,
                os.path.join('./data', str(config.user), 'chaos', 'log'))
        except Exception as e:
            print(f"Failed to collect fault-proxy logs: {e}")
        for n in namespaces:
            config.namespace = n
            config.svcs.clear()
            config.pods.clear()
            data_folder = './data/' + str(config.user) + '/' + config.namespace
            print('Collecting data for [' + config.namespace + ']')
            MetricCollector.collect(config, os.path.join(data_folder, 'metrics'), True)
            Trace.collect(config, os.path.join(data_folder, 'trace'))
            LogCollector.collect_logs_(config.start, config.end, n, os.path.join(data_folder, 'log'))
            # Log.collect(config, os.path.join(data_folder, 'log'))
            config.pods.clear()
            # Merge trace data into the original file when enabled.
            # pkl_concat.data_concat(data_folder, data_folder)
            # Store ordered graph.json call chains alongside metrics, traces, and logs.
            try:
                import GraphCollector
                GraphCollector.collect_graph(config, os.path.join(data_folder, 'graph'))
            except Exception as e:
                print(f"Failed to collect call chains: {e}")
        # Collect node metrics.
        node_folder = './data/' + str(config.user) + '/node'
        print('Collecting node data')
        MetricCollector.collect_node(config, node_folder, True)

    def merge_prefault_logs(config, namespaces, tmp_subdir):
        """Merge pre-fault logs from pods that disappeared before final collection.

        Compare snapshots with final logs by <pod>_<container>.log filename.
        Keep only missing pods under the _log_prefault suffix, then remove the snapshot directory.
        """
        for n in namespaces:
            data_folder = './data/' + str(config.user) + '/' + n
            log_dir = os.path.join(data_folder, 'log')
            tmp_dir = os.path.join(data_folder, tmp_subdir)
            if not os.path.isdir(tmp_dir):
                continue
            os.makedirs(log_dir, exist_ok=True)
            existing = set(os.listdir(log_dir))
            kept = 0
            for fn in os.listdir(tmp_dir):
                if not fn.endswith('.log'):
                    continue
                if fn in existing:
                    continue  # Final collection already has this pod log.
                dst = os.path.join(log_dir, f"{fn[:-4]}_log_prefault.log")
                shutil.move(os.path.join(tmp_dir, fn), dst)
                kept += 1
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if kept:
                print(f"[{n}] {kept} pre-fault logs from disappeared pods merged into log/ (suffix: _log_prefault)")

    # chaos_service.sh uses this mode before deleting the deployment.
    # --user, --start, and --end identify the output directory and full window.
    #   python Main.py --user <dir> --start <epoch> --end <epoch>
    # Without these options, process the label_files list below.
    parser = argparse.ArgumentParser(description='Collect all metrics in the specified time window')
    parser.add_argument('--user', help='Output directory (config.user), relative to ./data')
    parser.add_argument('--start', type=int, help='Window start time (epoch seconds)')
    parser.add_argument('--end', type=int, help='Window end time (epoch seconds)')
    parser.add_argument('--logs-only', action='store_true',
                        help='Collect logs only (skip metrics, traces, graphs, and nodes); '
                             'save logs before a pod_failure or pod_kill restart')
    parser.add_argument('--log-subdir', default='log',
                        help='Log subdirectory (default: log); use a temporary name for pre-fault snapshots')
    parser.add_argument('--merge-prefault', metavar='TMP_SUBDIR',
                        help='After final collection, merge logs of disappeared pods from the temporary snapshot '
                             'into log/ with the _log_prefault suffix, then remove the temporary directory')
    cli_args, _ = parser.parse_known_args()

    if cli_args.user and cli_args.start and cli_args.end:
        config.user = cli_args.user
        config.start = cli_args.start
        config.end = cli_args.end
        config.duration = config.end - config.start

        if cli_args.logs_only:
            print(f"==> Collecting logs for one window [{config.start} ~ {config.end}] "
                  f"-> data/{config.user}/<ns>/{cli_args.log_subdir}")
            try:
                LogCollector.collect_proxy_logs(
                    config.start, config.end,
                    os.path.join('./data', str(config.user), 'chaos', cli_args.log_subdir))
            except Exception as e:
                print(f"Failed to collect fault-proxy logs: {e}")
            for n in namespaces:
                config.namespace = n
                data_folder = './data/' + str(config.user) + '/' + n
                LogCollector.collect_logs_(
                    config.start, config.end, n,
                    os.path.join(data_folder, cli_args.log_subdir))
            sys.exit(0)

        print(f"==> Collecting one window [{config.start} ~ {config.end}] "
              f"({config.duration}s) -> data/{config.user}")
        collect_for_config(config, namespaces)
        if cli_args.merge_prefault:
            merge_prefault_logs(config, namespaces, cli_args.merge_prefault)
        sys.exit(0)

    # Each label file can contain several fault groups and collection windows.
    label_files = [
        'data/MDOC/abnormal/load-5/agent-network-pdf-parsing_label.txt',
    ]

    for label_file in label_files:
        groups = parse_label_file(label_file)
        print(f'{label_file} contains {len(groups)} fault groups')
        for g in groups:
            if 'window_start_ts' not in g or 'window_end_ts' not in g:
                print(f"Skipping {g.get('name')}: missing full window (window_start/window_end)")
                continue
            config.user = user_dir_for(label_file, g['name'])
            config.start = g['window_start_ts']
            config.end = g['window_end_ts']
            config.duration = config.end - config.start
            print(f"==> Collecting {g['name']}  root_cause={g.get('root_cause')}  "
                  f"[{config.start} ~ {config.end}] ({config.duration}s) -> data/{config.user}")
            collect_for_config(config, namespaces)
