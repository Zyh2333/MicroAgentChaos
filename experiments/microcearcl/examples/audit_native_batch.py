"""Offline data-tool smoke audit of a saved mapping; no LLM or fault inference."""
import argparse
import json
from pathlib import Path
from ..core import Sample
from ..io import write_json
from ..registry import Registry
from ..utils.injection_hits import inspect


def audit(mapping_file, output):
    mapping = json.loads(Path(mapping_file).read_text())
    rows = []
    for meta in mapping['samples'].values():
        path = Path(meta['sample_path_absolute'])
        eligibility = inspect(path)
        if not eligibility['eligible']:
            row = {'sample_id': meta['sample_id'], 'name': meta['sample_name'], 'skipped': eligibility}
        else:
            registry = Registry(Sample(path))
            logs = registry.invoke('read_logs', {'limit': 3})['data']
            events = registry.invoke('read_runtime_events', {'limit': 3})['data']
            graph = registry.invoke('read_execution_graph', {'limit': 1})['data']
            row = {'sample_id': meta['sample_id'], 'name': meta['sample_name'],
                'log_files_visible': len([f for f in logs['files'] if 'file' in f]),
                'log_records_returned': len(logs['records']), 'next_log_cursor': logs['next_cursor'],
                'collector_events_available': events['available'], 'graph_tasks': graph['task_count'],
                'warnings': registry.sample.warnings}
        rows.append(row)
        print(meta['sample_name'], 'skip' if 'skipped' in row else 'data_read', flush=True)
        write_json(output, {'kind': 'offline data access smoke test, not LLM accuracy', 'rows': rows})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mapping')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    audit(args.mapping, args.out)
