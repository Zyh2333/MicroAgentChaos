"""Cross-service presence check for one identifier; no fault classification."""
from ..core import reference
from ..utils.native_data import log_identity, logs, services as known_services

NAME = 'compare_presence'
DESCRIPTION = ('在同一时间窗内按服务统计给定任务/请求标识符的出现次数与首末时间，'
               '用于检查一次调用是否到达目标端；返回扫描文件和原文引用，不判断故障类型。')
PARAMETERS = {
    'identifier': {'type': 'string', 'description': '来自已读日志的任务、请求或追踪标识符'},
    'services': {'type': 'array', 'items': {'type': 'string'},
                 'description': '要对照的服务；省略时扫描全部已发现服务'},
    'start': {'type': 'number', 'description': 'UTC epoch 秒，包含'},
    'end': {'type': 'number', 'description': 'UTC epoch 秒，不包含'},
}


def query(sample, identifier, services=None, start=None, end=None, limit=12):
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError('identifier 必须为非空字符串')
    if services is not None and (not isinstance(services, list)
                                 or any(not isinstance(x, str) for x in services)):
        raise ValueError('services 必须为服务名数组')
    selected = list(services) if services else known_services(sample)
    unknown = [x for x in selected if x not in known_services(sample)]
    if unknown:
        raise ValueError('services 必须来自已读取的数据清单: ' + ', '.join(unknown))
    key = ('compare_presence', identifier, tuple(selected), start, end)
    if key in sample.cache:
        return sample.cache[key]
    files_by_service = {}
    for path in sample.ns('log').glob('*.log'):
        files_by_service.setdefault(log_identity(path)[0], []).append(str(path.relative_to(sample.path)))
    needle, rows = identifier.casefold(), []
    for service in selected:
        files = sorted(files_by_service.get(service, []))
        count, first, last, samples = 0, None, None, []
        for record in logs(sample, service, start, end):
            if needle not in record['text'].casefold():
                continue
            count += 1
            stamp = record['timestamp']
            if stamp is not None:
                first = stamp if first is None else min(first, stamp)
                last = stamp if last is None else max(last, stamp)
            if len(samples) < 3:
                samples.append({'ref': record['ref'], 'text': record['text'][:600]})
        refs = [sample_ref['ref'] for sample_ref in samples]
        if not refs:
            refs = [reference(name, 'file', scanned=True) for name in files]
        rows.append({
            'service': service, 'count': count, 'first': first, 'last': last,
            'scanned_files': files, 'references': refs, 'samples': samples,
            'text': (f"{service} identifier={identifier} count={count} "
                     f"first={first} last={last} scanned_files={len(files)}")})
    result = {
        'identifier': identifier, 'window': {'start': start, 'end': end},
        # Map keyed by service: every requested service is visible, including count=0.
        'services': {row['service']: row for row in rows},
        'service_count': len(rows),
        'scanned_files': sum(len(row['scanned_files']) for row in rows),
        'note': ('只统计标识符在同一时间窗内的出现情况；目标端 0 次表示在已扫描文件中未出现，'
                 '需同时说明文件、时间窗和标识符是否可用于关联。出现次数不等于执行成功。')}
    sample.cache[key] = result
    return result


def run(sample, identifier, services=None):
    return query(sample, identifier, services=services)
