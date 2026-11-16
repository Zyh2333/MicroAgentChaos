"""Evaluation-only eligibility; never passed to Sample, tools, or model context."""
import json
from pathlib import Path


def inspect(path):
    path = Path(path)
    stats_path = path / 'proxy_stats.json'
    raw_hits, effective_hits, valid_records, invalid = None, None, 0, 0
    if stats_path.is_file():
        try:
            data = json.loads(stats_path.read_text())
            count = data.get('total_fires')
            if type(count) is int and count >= 0:
                raw_hits = count
        except (ValueError, OSError, AttributeError):
            invalid += 1
    explicit, applied = 0, 0
    for file in sorted((path / 'chaos/log').glob('*_fires.jsonl')):
        with file.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError('not an object')
                except ValueError:
                    invalid += 1
                    continue
                valid_records += 1
                if type(record.get('effect_applied')) is bool:
                    explicit += 1
                    applied += record['effect_applied']
    if raw_hits is None and valid_records:
        raw_hits = valid_records
    # Complete explicit effect records take precedence over match counts/cache misses.
    if explicit and explicit == valid_records and not invalid:
        effective_hits = applied
    count = effective_hits if effective_hits is not None else raw_hits
    eligible = count is not None and count > 0
    return {'eligible': eligible, 'raw_hits': raw_hits, 'effective_hits': effective_hits,
            'audit_records': valid_records, 'invalid_records': invalid,
            'reason': None if eligible else 'zero_injection_hits' if count == 0 else 'missing_injection_hit_count'}


def select(cases, enabled=True):
    selected, skipped = [], []
    for case in cases:
        observation = inspect(case.path) if enabled else {'eligible': True, 'filter_disabled': True}
        if observation['eligible']:
            selected.append(case)
        else:
            skipped.append({'case': case.name, 'sample_path': str(case.path),
                            'topology_mode': case.topology_mode, 'fault_mode': case.fault_mode, **observation})
    return selected, skipped
