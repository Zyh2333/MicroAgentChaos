#!/usr/bin/env python3
"""Auditable run-level data for motivation 2. No injected metadata in features."""
from collections import defaultdict, deque
from pathlib import Path
import json
import hashlib
import re

import numpy as np
import pandas as pd
from observations import completion_times,early_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'benchmark/data/MDOC/agent_faults'
COMMON = DATA / 'abnormal/load-64/common'
LAYERS = ['F0', 'F1', 'F2', 'F3', 'F4']
SCOPE = ['affected_services', 'affected_tasks', 'affected_task_ratio', 'propagation_breadth']
LOCAL = ['node_error_rate', 'link_error_rate', 'mean_span_seconds',
         'p95_span_seconds', 'retry_rate']
TOPOLOGIES = ['cloud-preference', 'edge-preference', 'dynamic', 'unrecorded']
PERSISTENCES = ['burst', 'intermittent', 'persistent', 'single']
EXCLUDED_RUN_IDS = {'20261104-012022-3391d406', '20261104-034933-2183a1ad'}


def proxy_stats_info(path):
    """A missing counter is unknown, never an observed zero."""
    source = path/'proxy_stats.json'
    if not source.is_file():
        return {'proxy_stats_file':str(source),'proxy_stats_status':'missing_file','proxy_total_fires':None}
    payload = json.loads(source.read_text())
    if 'total_fires' not in payload:
        return {'proxy_stats_file':str(source),'proxy_stats_status':'missing_total_fires','proxy_total_fires':None}
    total = payload['total_fires']
    if isinstance(total,bool) or not isinstance(total,(int,float)) or not np.isfinite(total) or total<0:
        raise ValueError(f'Invalid total_fires in {source}: {total!r}')
    return {'proxy_stats_file':str(source),'proxy_stats_status':'recorded','proxy_total_fires':total}


def no_fault_reasons(path, stats):
    reasons=[]
    if any(path.resolve().is_relative_to((DATA/'topology_runs'/run).resolve()) for run in EXCLUDED_RUN_IDS):
        reasons.append('explicit_no_fault_run')
    if stats['proxy_stats_status']=='recorded' and stats['proxy_total_fires']==0:
        reasons.append('proxy_total_fires_zero')
    return reasons


def service(name):
    if not name:
        return None
    if name.startswith('AgentNetworkPlannerGroup/'):
        return 'agent-network-planner'
    head = name.split('/')[0].removesuffix('-edge')
    return head if head.startswith('agent-network-') else None


def failed(span):
    return span.get('status') == 3 or bool(span.get('error'))


def labels(root):
    """Resolve only real siblings; aggregate labels do not create extra runs."""
    found = {}
    for path in sorted(root.rglob('*label.txt')):
        name, meta = None, {}
        for line in path.read_text().splitlines() + ['=== END ===']:
            match = re.fullmatch(r'=== (\S+) ===', line.strip())
            if match:
                if name:
                    parents = [path.parent]
                    if meta.get('persistence'):
                        parents.append(path.parent / meta['persistence'])
                    candidates = [p / name for p in parents if (p / name).is_dir()]
                    for sample in candidates:
                        if sample in found:
                            # Aggregate and persistence labels may omit optional keys.
                            for key in ('fault_type', 'fault_injection', 'fault_start', 'fault_end'):
                                if key in meta and key in found[sample]:
                                    assert meta[key] == found[sample][key], (sample, key)
                        found.setdefault(sample, {}).update(meta)
                name, meta = (match[1], {}) if match[1] != 'END' else (None, {})
            elif name and ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
    return found


def epoch(value):
    return int(re.search(r'\((\d{10})\)', value)[1])


def load_fires(path, meta):
    raw, kept, seen = 0, [], set()
    qc = {'n_effect_unspecified': 0, 'n_effect_false': 0, 'n_duplicates': 0, 'n_outside_window': 0}
    start, end = epoch(meta['fault_start']), epoch(meta['fault_end'])
    for source in sorted(path.glob('chaos/log/*_fires.jsonl')):
        for line_no, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            f = json.loads(line)
            raw += 1
            key = json.dumps(f, sort_keys=True)
            if key in seen:
                qc['n_duplicates'] += 1
                continue
            seen.add(key)
            qc['n_effect_unspecified'] += int('effect_applied' not in f)
            qc['n_effect_false'] += int(f.get('effect_applied') is False)
            qc['n_outside_window'] += int(not start <= f['t'] < end)
            if f.get('effect_applied') is True and start <= f['t'] < end:
                kept.append({**f, '_file': str(source), '_line': line_no})
    return {'n_raw_fires': raw, **qc}, kept


def reachable(start, adjacency):
    visited, queue = set(), deque(adjacency.get(start, ()))
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        queue.extend(adjacency.get(v, ()))
    return visited


def task_signals(trace):
    """Process errors only: never use terminal task_status in feature extraction.

    Propagation proxy uses earliest error vertices in the recorded layered DAG.
    Actual fallback services count as affected, without inventing their edges.
    """
    vertices, abnormal, adjacency = set(), set(), defaultdict(set)
    node_n = node_bad = retries = 0
    seconds, span_errors = [], {}
    for level in trace.get('levels', []):
        level_no = level['level']
        for key, span in (level.get('spans') or {}).items():
            planned = service(key)
            if planned is None:
                continue
            vertices.add(planned)
            attempts = span.get('attempts') or []
            process_bad = failed(span) or any(failed(a) for a in attempts)
            span_errors[(level_no, key)] = process_bad
            node_n += 1
            node_bad += int(process_bad)
            retries += int(len(attempts) > 1 or span.get('attempt_count', 1) > 1)
            duration = span.get('time')
            if isinstance(duration, (int, float)) and duration >= 0:
                seconds.append(float(duration))
            if failed(span):
                abnormal.add(service(span.get('executor')) or planned)
            for attempt in attempts:
                if failed(attempt):
                    abnormal.add(service(attempt.get('executor')) or planned)
    link_n = link_bad = unmatched_edges = 0
    for edge in trace.get('edges', []):
        src, dst = service(edge.get('source')), service(edge.get('destination'))
        origin = (edge.get('level'), edge.get('source'))
        target = (edge.get('level', -2) + 1, edge.get('destination'))
        if src and dst and origin in span_errors and target in span_errors:
            adjacency[origin].add(target)
            link_n += 1
            link_bad += int(span_errors[target])
        elif src and dst:
            unmatched_edges += 1
    # Propagation entity = (task_id, level, recorded execution vertex). Actual fallback
    # services still contribute to affected_services, without inventing edges.
    observed_errors = {k for k, bad in span_errors.items() if bad}
    root_set, propagated = set(), set()
    cross_service_paths = []
    breadth = 0.0
    if observed_errors:
        descendants = {v: reachable(v, adjacency) for v in observed_errors}
        # Keep recorded cross-service anomaly paths as audit evidence. Several
        # disconnected service errors alone do not establish propagation.
        for origin in sorted(observed_errors):
            queue = deque([(origin, [origin])]);seen = {origin}
            while queue:
                current, path = queue.popleft()
                if (current != origin and current in observed_errors and
                        service(current[1]) != service(origin[1])):
                    cross_service_paths.append(path)
                    break
                for target in sorted(adjacency.get(current, ())):
                    if target not in seen:
                        seen.add(target);queue.append((target, path + [target]))
        root_set = {v for v in observed_errors
                    if not any(v in descendants[u] for u in observed_errors - {v})}
        if not root_set:
            breadth = np.nan
        else:
            propagated = (set().union(*(descendants[v] for v in root_set)) & observed_errors) - root_set
            # Disconnected error cycles must not silently disappear.
            if root_set | propagated != observed_errors:
                breadth = np.nan
            else:
                breadth = float(len(propagated))
    return {'task_id': trace.get('task_id'), 'failed_services': sorted(abnormal),
            'observed_error_vertices': sorted(observed_errors),
            'cross_service_error_paths': cross_service_paths,
            'observed_roots': sorted(root_set), 'propagated_services': sorted(propagated),
            'propagation_breadth': breadth, 'node_n': node_n, 'node_bad': node_bad,
            'link_n': link_n, 'link_bad': link_bad, 'retry_spans': retries,
            'span_seconds': seconds, 'task_status': trace.get('task_status'),
            'n_unmatched_edges': unmatched_edges}


def summarize_tasks(traces, n_submitted):
    ts = [task_signals(t) for t in traces]
    ids = [t['task_id'] for t in ts]
    assert len(ids) == len(set(ids)) and None not in ids, 'Missing/duplicate task_id'
    affected = set().union(*(set(t['failed_services']) for t in ts)) if ts else set()
    n_bad_tasks = sum(bool(t['failed_services']) for t in ts)
    n_nodes = sum(t['node_n'] for t in ts)
    n_links = sum(t['link_n'] for t in ts)
    durations = [v for t in ts for v in t['span_seconds']]
    completed = [t for t in ts if t['task_status'] in (0, 7)]
    n_failed = sum(t['task_status'] == 0 for t in completed)
    breadth = sum(t['propagation_breadth'] for t in ts) if ts else np.nan
    return {'n_submitted': n_submitted, 'n_graph_tasks': len(ts),
            'n_terminal_tasks': len(completed), 'n_failed_tasks': n_failed,
            'n_unobserved_tasks': max(0, n_submitted - len(ts)),
            'graph_coverage': len(ts) / n_submitted if n_submitted else np.nan,
            'affected_services': len(affected) if ts else np.nan,
            'affected_tasks': n_bad_tasks if ts else np.nan,
            'affected_task_ratio': n_bad_tasks / n_submitted if n_submitted else np.nan,
            'propagation_breadth': breadth,
            'node_error_rate': sum(t['node_bad'] for t in ts) / n_nodes if n_nodes else np.nan,
            'link_error_rate': sum(t['link_bad'] for t in ts) / n_links if n_links else np.nan,
            'mean_span_seconds': np.mean(durations) if durations else np.nan,
            'p95_span_seconds': np.percentile(durations, 95) if durations else np.nan,
            'retry_rate': sum(t['retry_spans'] for t in ts) / n_nodes if n_nodes else np.nan,
            'task_failure_rate': n_failed / len(completed) if completed else np.nan,
            'n_nodes': n_nodes, 'n_links': n_links,
            'n_unmatched_edges': sum(t['n_unmatched_edges'] for t in ts),
            'affected_services_json': json.dumps(sorted(affected)),
            'n_unresolved_propagation_tasks': sum(not np.isfinite(t['propagation_breadth']) for t in ts),
            'outcomes_complete': bool(n_submitted and len(completed) == len(ts) == n_submitted)}, ts


def discover(extra_roots=()):
    selected, audit = {}, []
    sources = [(COMMON, 'common'), (DATA / 'topology_runs', 'topology_runs')]
    sources.extend((Path(p), 'extra') for p in extra_roots)
    for root, cohort in sources:
        for path, meta in labels(root).items():
            injection = meta.get('fault_injection')
            graph = path / 'agent-network/graph/call_chains.json'
            stats = proxy_stats_info(path)
            reasons = no_fault_reasons(path,stats)
            if injection not in ('true', 'false'):
                reasons.append('unverified_injection_label')
            if not graph.is_file():
                reasons.append('missing_graph')
            if not (path / 'task.log').is_file():
                reasons.append('missing_submission_log')
            reason = ';'.join(reasons)
            sample_id = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            audit.append({'sample_id': sample_id, 'included': not reason,
                          'reason': reason, 'cohort': cohort, **stats, **{k: meta.get(k, '') for k in
                          ['fault_type', 'fault_injection', 'load_users', 'topology_mode', 'persistence']}})
            if not reason:
                selected[path] = (meta, cohort, sample_id)
    # Audit pending F0 directories even when no label block has yet been written.
    for path in (DATA / 'topology_runs').rglob('*_f0_*'):
        if path.is_dir() and not (path / 'agent-network/graph/call_chains.json').is_file():
            audit.append({'sample_id': str(path.relative_to(ROOT)), 'included': False,
                          'reason': 'F0_pending_no_graph_or_label', 'cohort': 'pending',
                          'fault_type': 'f0', 'fault_injection': '', 'load_users': '64'})
    labelled_paths={p for p,(_,cohort,_) in selected.items() if cohort=='topology_runs'}
    audited_paths={a['sample_id'] for a in audit}
    for graph in (DATA/'topology_runs').rglob('call_chains.json'):
        path=graph.parents[2];sample_id=str(path.relative_to(ROOT))
        if path not in labelled_paths and sample_id not in audited_paths:
            audit.append({'sample_id':sample_id,'included':False,'reason':'graph_without_resolvable_label',
                          'cohort':'topology_runs','fault_type':'unknown','fault_injection':'unknown','load_users':''})
    for path, meta in labels(DATA / 'normal').items():
        audit.append({'sample_id': str(path.relative_to(ROOT)), 'included': False,
                      'reason': 'normal_directory_has_fault_injection_true' if meta.get('fault_injection') == 'true' else 'outside_selected_control_cohort',
                      'cohort': 'legacy_normal_directory', **{k: meta.get(k, '') for k in
                      ['fault_type', 'fault_injection', 'load_users']}})
    return selected, audit


def build_dataframe(extra_roots=()):
    selected, audit = discover(extra_roots)
    rows, tasks, events = [], [], []
    for path, (meta, cohort, sample_id) in sorted(selected.items()):
        fire_qc, fires = load_fires(path, meta)
        injection = meta['fault_injection'] == 'true'
        if not injection and fires:
            raise ValueError(f'Normal-labelled run has applied fires: {path}')
        n_submit = len(re.findall(r'^id:[0-9a-fA-F-]+\s', (path / 'task.log').read_text(), re.M))
        signals, ts = summarize_tasks(json.loads((path / 'agent-network/graph/call_chains.json').read_text()), n_submit)
        timing, completion = completion_times(path, ts, epoch(meta['fault_start']), epoch(meta['fault_end']))
        early = early_features(ts, completion, epoch(meta['fault_start']), epoch(meta['fault_end']),
                               timing['completion_times_complete'])
        fault_type = meta['fault_type']
        layer = fault_type[:2].upper() if injection else 'normal'
        # Same normal collection/configuration block is kept in one fold.
        run_parts = [p for p in path.parts if re.fullmatch(r'\d{8}-\d{6}-[0-9a-f]+', p)]
        normal_block = meta.get('collection_group') or (run_parts[-1] if run_parts else str(path.parent))
        topology = meta.get('topology_mode', 'unrecorded')
        persistence = (meta.get('persistence') or path.parent.name) if injection else 'none'
        assert topology in TOPOLOGIES, (path, topology)
        assert not injection or persistence in PERSISTENCES, (path, persistence)
        selection = {'task_indices':sorted(map(int,meta.get('task_indices','').split(','))) if meta.get('task_indices') else [],
                     'edge_services':sorted(meta.get('edge_services','').split()),
                     'load_users':int(meta['load_users']), 'target_service':meta['service'].removesuffix('-edge')}
        config_id = hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()[:16] if selection['task_indices'] else 'unrecorded'
        workload_id=hashlib.sha256(json.dumps(selection['task_indices']).encode()).hexdigest()[:12] if selection['task_indices'] else 'unrecorded'
        deployment_id=hashlib.sha256(json.dumps(selection['edge_services']).encode()).hexdigest()[:12] if meta.get('edge_services') else 'unrecorded'
        task_classes=re.findall(r'^id:[0-9a-fA-F-]+\s+MDOC/([^/\s]+)/', (path/'task.log').read_text(), re.M)
        task_class_counts={kind:task_classes.count(kind) for kind in sorted(set(task_classes))}
        providers = sorted({f.get('upstream','') for f in fires
                            if f.get('path','').rstrip('/').endswith('/chat/completions') and service(f.get('upstream')) is None})
        rows.append({'sample_id': sample_id, 'cohort': cohort, 'layer': layer,
                     'fault_type': fault_type, 'persistence': persistence,
                     'topology_mode':topology, 'load_users':int(meta['load_users']),
                     'run_id':run_parts[-1] if run_parts else 'legacy_common',
                     'configuration_id':config_id, 'fault_zone':meta.get('fault_zone','unrecorded'),
                     'workload_id':workload_id,'edge_deployment_id':deployment_id,
                     'task_class_profile':'+'.join(task_class_counts) if task_class_counts else 'unrecorded',
                     'task_class_counts_json':json.dumps(task_class_counts,sort_keys=True),
                     'injected_target':meta.get('service','unrecorded'),
                     'task_indices':meta.get('task_indices',''), 'edge_services':meta.get('edge_services',''),
                     'model_upstreams_json':json.dumps(providers),
                     'detection_label': int(injection),
                     'cv_group': fault_type if injection else 'normal:' + normal_block,
                     **proxy_stats_info(path),
                     **fire_qc, 'n_true_window_hits': len(fires),
                     'hit_verification': 'verified_applied_window_events' if fires else 'label_only_no_verified_window_effect',
                     'window_start': epoch(meta['fault_start']), 'window_end': epoch(meta['fault_end']),
                     'propagation_basis': 'first_observed_error_vertex_descendants_proxy', **signals,
                     **timing, **early})
        for t in ts:
            tasks.append({'sample_id': sample_id, **completion[t['task_id']], **{k: (json.dumps(v) if isinstance(v, list) else v)
                                                    for k, v in t.items()}})
        for f in fires:
            events.append({'sample_id': sample_id, 'file': f['_file'], 'line': f['_line'],
                           **{k: f.get(k, '') for k in ['t', 'category', 'taskId', 'upstream', 'source_service']}})
        print(f'{layer}: {path.name}: submitted={n_submit}, graph={len(ts)}, true_hits={len(fires)}')
    return pd.DataFrame(rows), pd.DataFrame(tasks), pd.DataFrame(events), pd.DataFrame(audit)
