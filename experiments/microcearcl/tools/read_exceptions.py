"""Read exception records verbatim; never interpret their business meaning."""
from .search_logs import collect

NAME = 'read_exceptions'
DESCRIPTION = '读取通用异常记录、Traceback 和 ERROR 日志的原文、上下文、文件行号；不分类异常含义。'
PARAMETERS = {'service': {'type': 'string'}, 'start': {'type': 'number'}, 'end': {'type': 'number'},
              'offset': {'type': 'integer', 'minimum': 0},
              'context': {'type': 'integer', 'minimum': 0, 'maximum': 3}}


def is_exception(record):
    """Recognize logging/Python record structure, not fault-specific words or fields."""
    text = record['text']
    if 'Traceback (most recent call last):' in text:
        return True
    for word in text.split()[:12]:
        if word.strip('[]:').upper() == 'ERROR':
            return True
        if word.endswith(':'):
            name = word[:-1].rsplit('.', 1)[-1]
            if name.isidentifier() and name.endswith(('Error', 'Exception')):
                return True
    return False


def query(sample, service=None, start=None, end=None, offset=0, context=2, limit=12):
    page = collect(sample, is_exception, service, start, end, offset, context, limit)
    return {'exceptions': page['matches'], 'total_exceptions': page['total_matches'],
            'offset': offset, 'next_offset': page['next_offset'],
            'interpretation': '原始异常观测；其原因、故障域和类型由 LLM 结合其他文件推理。'}
