"""Literal log search with pagination, context, and stable source locations."""
from collections import deque
from ..utils.native_data import logs

NAME = 'search_logs'
DESCRIPTION = '按关键词、服务、task ID、时间检索原生日志，返回文件行号、上下文和下一页。'
PARAMETERS = {
    'query': {'type': 'string', 'description': '大小写不敏感的字面关键词；空串浏览日志'},
    'service': {'type': 'string'}, 'task_id': {'type': 'string'},
    'start': {'type': 'number', 'description': 'UTC epoch 秒，包含'},
    'end': {'type': 'number', 'description': 'UTC epoch 秒，不包含'},
    'offset': {'type': 'integer', 'minimum': 0},
    'context': {'type': 'integer', 'minimum': 0, 'maximum': 3},
}


def query(sample, query='', service=None, task_id=None, start=None, end=None,
          offset=0, context=1, limit=12):
    if not isinstance(query, str) or task_id is not None and not isinstance(task_id, str):
        raise ValueError('query/task_id 必须为字符串')
    def matches(record):
        text = record['text'].casefold()
        return query.casefold() in text and (task_id is None or task_id.casefold() in text)
    return collect(sample, matches, service, start, end, offset, context, limit)


def collect(sample, predicate, service=None, start=None, end=None, offset=0, context=1, limit=12):
    """Read-only pagination shared by literal search and generic exception browsing."""
    if type(offset) is not int or offset < 0 or type(context) is not int or not 0 <= context <= 3:
        raise ValueError('offset 非负，context 为 0–3')
    found, count, before, pending = [], 0, deque(maxlen=context), []
    previous_file = None
    for record in logs(sample, service, start, end):
        file = record['ref']['file']
        if file != previous_file:
            before.clear()
            pending = []
            previous_file = file
        for item, remaining in list(pending):
            item['after'].append(record)
        pending = [(item, remaining - 1) for item, remaining in pending if remaining > 1]
        if predicate(record):
            if count >= offset and len(found) < limit:
                item = {**record, 'before': list(before), 'after': []}
                found.append(item)
                if context:
                    pending.append((item, context))
            count += 1
        before.append(record)
    return {'matches': found, 'total_matches': count, 'offset': offset,
            'next_offset': offset + len(found) if count > offset + len(found) else None}
