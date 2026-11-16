"""Generic signature census over original logs; no fault classification."""
import re
from collections import Counter

from ..utils.native_data import logs

NAME = 'log_census'
DESCRIPTION = ('按服务和通用签名统计原始日志：HTTP 状态码与异常类名；返回次数、首末时间、'
               '时间桶和原文引用。只做计数，不判断故障域或类型。')
PARAMETERS = {
    'service': {'type': 'string'},
    'start': {'type': 'number', 'description': 'UTC epoch 秒，包含'},
    'end': {'type': 'number', 'description': 'UTC epoch 秒，不包含'},
    'bucket_seconds': {'type': 'integer', 'minimum': 10, 'maximum': 3600},
}
STATUS = re.compile(r'(?i)(?:\bstatus(?:_code)?\s*[=:]\s*|\bhttp(?:/\d(?:\.\d)?)?\s+|\bcode\s*[:=]\s*)(\d{3})\b')
EXCEPTION = re.compile(r'\b([A-Z][A-Za-z0-9_.]*(?:Error|Exception|Warning))\b')


def _bucket(timestamp, width):
    start = int(timestamp // width * width)
    return start, start + width


def query(sample, service=None, start=None, end=None, bucket_seconds=60, limit=12):
    if type(bucket_seconds) is not int or not 10 <= bucket_seconds <= 3600:
        raise ValueError('bucket_seconds 必须为 10–3600 的整数')
    key = ('log_census', service, start, end, bucket_seconds)
    if key in sample.cache:
        return sample.cache[key]
    signatures, scanned = {}, 0
    for record in logs(sample, service, start, end):
        scanned += 1
        text, stamp = record['text'], record['timestamp']
        for kind, pattern in (('http_status', STATUS), ('exception', EXCEPTION)):
            for match in pattern.finditer(text):
                value = match.group(1)
                item = signatures.setdefault((record['service'], kind, value), {
                    'service': record['service'], 'kind': kind, 'value': value,
                    'count': 0, 'first': None, 'last': None,
                    'buckets': Counter(), 'references': [], 'samples': []})
                item['count'] += 1
                if stamp is not None:
                    item['first'] = stamp if item['first'] is None else min(item['first'], stamp)
                    item['last'] = stamp if item['last'] is None else max(item['last'], stamp)
                    item['buckets'][_bucket(stamp, bucket_seconds)] += 1
                if len(item['references']) < 3 and record['ref'] not in item['references']:
                    item['references'].append(record['ref'])
                if len(item['samples']) < 3:
                    item['samples'].append({'ref': record['ref'], 'text': text[:600]})
    ordered = sorted(signatures.values(), key=lambda x: (-x['count'], x['service'], x['kind'], x['value']))
    rows = []
    for item in ordered[:limit]:
        buckets = [{'start': a, 'end': b, 'count': count}
                   for (a, b), count in sorted(item['buckets'].items())]
        rows.append({
            'service': item['service'], 'kind': item['kind'], 'value': item['value'],
            'count': item['count'], 'first': item['first'], 'last': item['last'],
            'buckets': buckets, 'references': item['references'], 'samples': item['samples'],
            'text': (f"{item['service']} {item['kind']}={item['value']} count={item['count']} "
                     f"first={item['first']} last={item['last']}")})
    result = {
        'window': {'start': start, 'end': end, 'bucket_seconds': bucket_seconds},
        'scanned_records': scanned, 'services': sorted({row['service'] for row in rows}),
        'signatures': rows, 'omitted_signatures': max(0, len(ordered) - limit),
        'note': ('通用 HTTP/异常签名计数；签名出现不证明故障层，未出现也不能单独证明健康。'
                 '时间桶只表示观测分布，不表示故障窗。')}
    sample.cache[key] = result
    return result


def run(sample, service=None):
    return query(sample, service=service)
