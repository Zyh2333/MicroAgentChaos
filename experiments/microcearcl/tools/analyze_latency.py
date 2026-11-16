"""Within-series latency baselines with missingness and temporal provenance."""
import csv
import math
from statistics import median

from ..core import reference
from ..utils.background_noise import csv_service
from ..utils.native_data import epoch

NAME = 'analyze_latency'
DESCRIPTION = '分析 latency.csv 的时序波动、历史窗口基线和持续抬升，返回时间、幅度、样本数与原始行号。'
PARAMETERS = {'service': {'type': 'string'},
              'quantile': {'type': 'string', 'enum': ['p50', 'p90', 'p99']},
              'start': {'type': 'number'}, 'end': {'type': 'number'},
              'offset': {'type': 'integer', 'minimum': 0}}


def run(sample):
    if NAME in sample.cache:
        return sample.cache[NAME]
    file = sample.ns('metrics/latency.csv')
    if not file.is_file():
        return {'series': [], 'note': 'latency.csv 不存在'}
    columns, missing = {}, {}
    with file.open() as handle:
        for line_no, row in enumerate(csv.DictReader(handle), 2):
            stamp = epoch(row.get('timestamp'))
            if stamp is None or not sample.in_window(stamp):
                continue
            for name, raw in row.items():
                if name == 'timestamp' or not name:
                    continue
                try:
                    value = float(raw)
                except (ValueError, TypeError):
                    continue
                if value < 0 or not math.isfinite(value):
                    missing[name] = missing.get(name, 0) + 1
                    continue
                columns.setdefault(name, []).append({'timestamp': stamp, 'value': value, 'line': line_no})
    summaries = []
    for name, points in columns.items():
        points.sort(key=lambda p: p['timestamp'])
        gaps = [b['timestamp'] - a['timestamp'] for a, b in zip(points, points[1:])
                if b['timestamp'] > a['timestamp']]
        cadence = median(gaps) if gaps else None
        changes = []
        # A baseline uses only preceding points. Require three consecutive elevated observations.
        for i in range(4, len(points) - 2):
            past, current = points[max(0, i - 12):i], points[i:i + 3]
            window = past + current
            if cadence is None or any(b['line'] != a['line'] + 1 or
                                      b['timestamp'] - a['timestamp'] > cadence * 3
                                      for a, b in zip(window, window[1:])):
                continue
            values = [p['value'] for p in past]
            baseline = median(values)
            mad = median(abs(x - baseline) for x in values)
            threshold = max(3 * 1.4826 * mad, abs(baseline) * .25, 1e-9)
            delta = median(p['value'] for p in current) - baseline
            if all(p['value'] > baseline + threshold for p in current):
                if changes and current[0]['timestamp'] <= changes[-1]['end']:
                    continue
                changes.append({'start': current[0]['timestamp'], 'end': current[-1]['timestamp'],
                    'baseline': baseline, 'current': median(p['value'] for p in current), 'delta': delta,
                    'ratio': (baseline + delta) / baseline if baseline > 0 else None,
                    'mad': mad, 'threshold': threshold, 'baseline_n': len(past), 'elevated_n': 3,
                    'references': [reference(str(file.relative_to(sample.path)), p['line'], column=name)
                                   for p in (past[0], past[-1], current[0], current[-1])]})
        service, _, quantile = name.partition('&')
        summaries.append({'column': name, 'service': service, 'quantile': quantile,
            'n': len(points), 'missing': missing.get(name, 0), 'cadence_seconds': cadence,
            'min': min(p['value'] for p in points), 'median': median(p['value'] for p in points),
            'max': max(p['value'] for p in points), 'first': points[0]['timestamp'], 'last': points[-1]['timestamp'],
            'background_contaminated': sample.mask_csv_gen_auth and csv_service(service),
            'changes': changes})
    result = {'series': summaries, 'unit': 'native CSV units (not assumed to be milliseconds)',
              'method': 'preceding 4–12 points, median/MAD; 3 consecutive points exceed max(3 robust sigma, 25% baseline)',
              'note': '时序抬升支持延迟异常，不能单独区分模型、网络和服务处理延迟；无稳定对照不补造基线。'}
    sample.cache[NAME] = result
    return result


def query(sample, service=None, quantile=None, start=None, end=None, offset=0, limit=12):
    if type(offset) is not int or offset < 0:
        raise ValueError('offset 必须非负')
    if quantile is not None and quantile not in ('p50', 'p90', 'p99'):
        raise ValueError('quantile 必须是 p50/p90/p99')
    if start is not None and end is not None and start >= end:
        raise ValueError('start 必须小于 end')
    result = run(sample)
    rows = []
    for item in result['series']:
        if service and item['service'] != service or quantile and item['quantile'] != quantile:
            continue
        changes = [c for c in item['changes'] if (start is None or c['end'] >= start)
                   and (end is None or c['start'] < end)]
        rows.append({**item, 'changes': changes})
    rows.sort(key=lambda r: (not bool(r['changes']), r['column']))
    return {**result, 'series': rows[offset:offset + limit], 'total_series': len(rows),
            'next_offset': offset + limit if offset + limit < len(rows) else None}
