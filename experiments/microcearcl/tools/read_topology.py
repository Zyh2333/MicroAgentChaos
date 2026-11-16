"""Join trace endpoint → Pod → time-matched deployment node → configured zone/CIDR."""
from bisect import bisect_right
from collections import defaultdict
import csv
import ipaddress
import json
from pathlib import Path
import pickle
import re

from ..core import reference
from ..utils.native_data import epoch

NAME = 'read_topology'
DESCRIPTION = '查询实例、节点、端点 IP 与已记录区域/网段，返回关联来源和歧义；不会从服务名猜实际落点。'
PARAMETERS = {'service': {'type': 'string'}, 'target_ip': {'type': 'string'},
              'timestamp': {'type': 'number'}, 'offset': {'type': 'integer', 'minimum': 0}}
TRACE_FILES = ('normal.pkl', 'abnormal.pkl', 'inbound.pkl', 'outbound.pkl',
               'inbound_half.pkl', 'outbound_half.pkl')
POD = re.compile(r'(.+)-[a-z0-9]{6,10}-[a-z0-9]{5}$')


class TelemetryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('telemetry must contain only primitive containers')


def load(sample):
    if NAME in sample.cache:
        return sample.cache[NAME]
    config_path = (Path(sample.topology_file) if sample.topology_file else
                   sample.path / 'observations/topology.json')
    if not config_path.is_file() and not sample.topology_file:
        config_path = Path(__file__).resolve().parents[1] / 'topology.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    nodes = config.get('nodes', {})
    placements, ip_pods = defaultdict(list), defaultdict(dict)
    path = sample.ns('metrics/graph.csv')
    if path.is_file():
        with path.open() as handle:
            for i, row in enumerate(csv.DictReader(handle), 2):
                node, stamp = row.get('destination', ''), epoch(row.get('timestamp'))
                if not node or stamp is None or node not in nodes and not node.startswith('node-'):
                    continue
                placements[row['source']].append({'timestamp': stamp, 'node': node,
                    'ref': reference(str(path.relative_to(sample.path)), i)})
    for entries in placements.values():
        entries.sort(key=lambda r: r['timestamp'])
    for name in TRACE_FILES:
        path = sample.ns('trace') / name
        if not path.is_file():
            continue
        try:
            with path.open('rb') as handle:
                batch_no = 0
                while True:
                    try:
                        batch = TelemetryUnpickler(handle).load()
                    except EOFError:
                        break
                    batch_no += 1
                    for i, trace in enumerate(batch):
                        if not isinstance(trace, dict):
                            continue
                        for pair in trace.get('call_instance', []):
                            for endpoint in pair:
                                if not isinstance(endpoint, str):
                                    continue
                                match = re.fullmatch(r'([^\.]+)\..*-(\d+\.\d+\.\d+\.\d+)', endpoint)
                                if match:
                                    ip_pods[match[2]][match[1]] = reference(
                                        str(path.relative_to(sample.path)), f'batch/{batch_no}/trace/{i}/call_instance')
        except (OSError, ValueError, TypeError, pickle.UnpicklingError) as error:
            sample.warn(f'trace/{name}: {type(error).__name__}; endpoint mapping unavailable')
    result = {'placements': dict(placements), 'ip_pods': dict(ip_pods), 'nodes': nodes,
              'config_ref': {'file': str(config_path), 'location': '/nodes',
                             'provenance': config.get('provenance')}}
    sample.cache[NAME] = result
    return result


def resolve(sample, *, pod=None, ip=None, timestamp=None):
    data = load(sample)
    candidates = [pod] if pod else list(data['ip_pods'].get(ip, {}))
    resolved = []
    for candidate in candidates:
        entries = data['placements'].get(candidate, [])
        stamp = epoch(timestamp)
        if not entries or stamp is None:
            continue
        i = bisect_right([r['timestamp'] for r in entries], stamp) - 1
        if i < 0 or stamp - entries[i]['timestamp'] > 30:
            continue
        # Simultaneous conflicting placement observations stay ambiguous.
        chosen_time = entries[i]['timestamp']
        matches = [r for r in entries if r['timestamp'] == chosen_time]
        if len({r['node'] for r in matches}) != 1:
            continue
        entry = entries[i]
        node_config = data['nodes'].get(entry['node'], {})
        if isinstance(node_config, str):
            node_config = {'zone': node_config}
        refs = [entry['ref'], data['config_ref']]
        if ip:
            refs.insert(0, data['ip_pods'][ip][candidate])
        resolved.append({'pod': candidate, 'node': entry['node'], 'zone': node_config.get('zone'),
                         'subnet': node_config.get('subnet') or node_config.get('cidr'),
                         'references': refs})
    if len(resolved) != 1:
        return {'resolved': False, 'reason': 'ambiguous' if resolved or len(candidates) > 1 else 'not_observed'}
    # Reused IPs cannot establish an actual endpoint even when only one Pod has recent placement.
    if ip and len(candidates) != 1:
        return {'resolved': False, 'reason': 'reused_ip'}
    return {'resolved': True, **resolved[0]}


def query(sample, service=None, target_ip=None, timestamp=None, offset=0, limit=12):
    if type(offset) is not int or offset < 0:
        raise ValueError('offset 必须非负')
    if target_ip:
        ipaddress.ip_address(target_ip)
        if timestamp is None:
            data = load(sample)
            return {'resolved': False, 'reason': 'timestamp_required',
                    'candidates': [{'pod': pod, 'reference': ref}
                                   for pod, ref in data['ip_pods'].get(target_ip, {}).items()]}
        return resolve(sample, ip=target_ip, timestamp=timestamp)
    data = load(sample)
    rows = []
    for pod, entries in sorted(data['placements'].items()):
        match = POD.fullmatch(pod)
        name = match[1] if match else pod
        if service and name != service:
            continue
        if timestamp is not None:
            rows.append({'service': name, **resolve(sample, pod=pod, timestamp=timestamp)})
            continue
        for node in sorted({r['node'] for r in entries}):
            subset = [r for r in entries if r['node'] == node]
            config = data['nodes'].get(node, {})
            rows.append({'service': name, 'pod': pod, 'node': node,
                         'node_metadata': config, 'first': subset[0]['timestamp'], 'last': subset[-1]['timestamp'],
                         'ips': [ip for ip, pods in data['ip_pods'].items() if pod in pods],
                         'references': [subset[0]['ref'], subset[-1]['ref'], data['config_ref']]})
    return {'placements': rows[offset:offset + limit], 'total': len(rows),
            'next_offset': offset + limit if offset + limit < len(rows) else None,
            'note': '区域来自节点元数据；物理 CIDR 仅在配置明确记录时返回，不从节点编号或 IP 前缀猜测。'}
