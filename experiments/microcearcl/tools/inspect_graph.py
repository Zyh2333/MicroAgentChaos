"""Query native detailed graphs without selecting traces by injection labels."""

from ..utils.background_graph import filter_text

NAME = 'inspect_graph'
DESCRIPTION = '按 task/trace ID 或执行服务查询详细执行图的路径、参数、结果、异常和实际 executor。'
PARAMETERS = {'task_id': {'type': 'string'}, 'service': {'type': 'string'},
              'offset': {'type': 'integer', 'minimum': 0}}


def query(sample, task_id=None, service=None, offset=0, limit=12):
    if type(offset) is not int or offset < 0:
        raise ValueError('offset 必须非负')
    matches, count = [], 0
    for file in sorted(sample.ns('graph/graph_json').glob('*.json')):
        if task_id is not None and file.stem != task_id:
            continue
        relative = str(file.relative_to(sample.path))
        data = sample.json(relative, {})
        identity = data.get('task_id') or data.get('trace_id') or file.stem
        for i, level in enumerate(data.get('level_details', [])):
            for vertex, span in (level.get('level_spans') or {}).items():
                executor = span.get('executor') or vertex
                executors = sorted({e for e in [span.get('executor')] +
                                    [a.get('executor') for a in span.get('attempts', [])] if e})
                if service and service != executor and service not in executors and service not in vertex.split('/'):
                    continue
                if count >= offset and len(matches) < limit:
                    safe = {}
                    for key in ('error', 'params', 'results', 'messages'):
                        safe[key] = filter_text(sample, span.get(key), executor, {'file': relative})
                    matches.append({'task_id': identity, 'level': level.get('level'), 'vertex': vertex,
                        'executor': executor, 'observed_executors': executors,
                        'status': span.get('status'), 'time_seconds': span.get('time'),
                        'attempts': filter_text(sample, span.get('attempts', []), executor, {'file': relative}),
                        'routes': filter_text(sample, (level.get('level_routes') or {}).get(vertex, {}), executor, {'file': relative}),
                        **safe, 'ref': {'file': relative, 'location': f'/level_details/{i}/level_spans/{vertex}'}})
                count += 1
    return {'spans': matches, 'total_spans': count, 'offset': offset,
            'next_offset': offset + len(matches) if count > offset + len(matches) else None,
            'note': '图路径表示执行关系；不把重试或重复文本视为重复执行。'}
