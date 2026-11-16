"""Read collector JSONL as data; never derive events from log wording or fault audits."""
import json
from ..core import reference
from ..utils.native_data import epoch
from ..utils.background_noise import policy

NAME = 'read_runtime_events'
DESCRIPTION = '按行读取采集器 observations/events.jsonl 原始 JSON 记录，保留字段与来源；不从日志推断事件或分类。'
PARAMETERS = {'start_line': {'type': 'integer', 'minimum': 1}}


def run(sample):
    if NAME in sample.cache:
        return sample.cache[NAME]
    path = sample.path / 'observations/events.jsonl'
    rows = []
    if path.is_file():
        with path.open(encoding='utf-8', errors='replace') as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    sample.warn(f'observations/events.jsonl:{line_no}: invalid JSON')
                    continue
                if not sample.in_window(epoch(data.get('timestamp')) if isinstance(data, dict) else None):
                    continue
                if isinstance(data, dict) and policy(sample).event(data, data, channel='telemetry_events',
                        ref=reference('observations/events.jsonl', line_no)):
                    continue
                rows.append({'data': data, 'ref': reference('observations/events.jsonl', line_no)})
    sample.cache[NAME] = rows
    return rows


def query(sample, start_line=1, limit=12):
    if type(start_line) is not int or start_line < 1:
        raise ValueError('start_line 必须为正整数')
    rows = [r for r in run(sample) if int(r['ref']['location']) >= start_line]
    return {'records': rows[:limit], 'next_line': int(rows[limit]['ref']['location']) if len(rows) > limit else None,
            'available': (sample.path / 'observations/events.jsonl').is_file(),
            'note': '仅读取采集器原始记录；chaos 注入命中日志不作为诊断输入。'}


def report(sample):
    return query(sample)
