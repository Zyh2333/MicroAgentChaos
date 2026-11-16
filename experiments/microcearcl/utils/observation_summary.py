"""Format data queries without semantic log selection or fault labels."""
import importlib
from ..core import reference
from .summarize import clean
from .native_data import services

READERS = ('read_metrics', 'read_logs', 'read_execution_graph', 'read_runtime_events',
           'search_logs', 'log_census', 'compare_presence', 'inspect_graph', 'read_topology',
           'analyze_latency', 'read_exceptions')
EXPLORERS = tuple(name for name in READERS if name != 'read_metrics')


def read(sample, name, service=None, limit=12, **arguments):
    if name not in READERS:
        raise ValueError('未知数据读取函数')
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('limit 必须是 1–100 整数')
    if service is not None and service not in services(sample):
        raise ValueError('service 必须来自已读取的数据清单')
    module = importlib.import_module(f'..tools.{name}', __package__)
    if name != 'read_metrics':
        result = module.query(sample, limit=limit, **({'service': service} if service is not None else {}), **arguments)
    else:
        if arguments:
            raise ValueError('不支持的读取参数')
        result = module.run(sample)
        files = {}
        for file, data in result['files'].items():
            columns = [{'column': k, **v, 'ref': reference(file, 'column/' + k)}
                       for k, v in data['columns'].items() if service is None or k.split('&')[0] == service]
            files[file] = {'missing_cells': data['missing_cells'], 'column_summaries': columns}
        result = {**result, 'files': files, 'detected_changes': [v for v in result.get('detected_changes', [])
                  if service is None or v['service'] == service]}
    return clean(result, limit)
