"""Browse log files in physical line order; no semantic selection or ranking."""
from ..core import reference
from ..utils.background_noise import policy
from ..utils.native_data import epoch, log_identity

NAME = 'read_logs'
DESCRIPTION = ('按文件和服务泛读原始日志；未指定文件时跨服务分层取样，返回行号、游标和原文，不预筛异常。')
PARAMETERS = {'service': {'type': 'string'}, 'file': {'type': 'string', 'description': '返回清单中的文件名或相对路径'},
              'start_line': {'type': 'integer', 'minimum': 1}}


def _slice(sample, path, start_line, quota):
    """Read up to quota physical lines; physical order and provenance are preserved."""
    name, pod = log_identity(path)
    relative = str(path.relative_to(sample.path))
    rows, cursor = [], None
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line_no, line in enumerate(handle, 1):
            if line_no < start_line:
                continue
            text = policy(sample).text(line, service=name, channel='logs',
                ref={'file': relative, 'line': line_no})
            if line.strip() and not text.strip():
                continue
            if len(rows) >= quota:
                cursor = {'file': relative, 'start_line': line_no}
                break
            text = text.rstrip('\r\n')
            stamp = epoch(text.split(maxsplit=1)[0]) if text.strip() else None
            rows.append({'service': name, 'pod': pod, 'timestamp': stamp,
                         'text': text, 'ref': reference(relative, line_no)})
    return rows, cursor


def query(sample, service=None, file=None, start_line=1, limit=12):
    if type(start_line) is not int or start_line < 1:
        raise ValueError('start_line 必须为正整数')
    available = sorted(sample.ns('log').glob('*.log'))
    # Inventory is a map, so generic list display bounds never hide available files.
    inventory = {str(p.relative_to(sample.path)):
                 {'service': log_identity(p)[0], 'bytes': p.stat().st_size} for p in available}
    selected = [p for p in available if service is None or log_identity(p)[0] == service]
    if file is not None:
        matching = [p for p in selected if file in (p.name, str(p.relative_to(sample.path)))]
        if not matching:
            raise ValueError('file 必须来自当前 sample 的日志清单')
        selected = matching
    # No explicit target: sample across services instead of exhausting the
    # alphabetically first file.  This is coverage, not semantic ranking.
    if file is None and service is None and len(selected) > 1:
        quota = max(1, limit // len(selected))
        rows, cursors = [], []
        for path in selected:
            if len(rows) >= limit:
                break
            part, cursor = _slice(sample, path, start_line, min(quota, limit - len(rows)))
            rows.extend(part)
            if cursor:
                cursors.append(cursor)
        return {'files': inventory, 'records': rows, 'next_cursor': cursors[0] if cursors else None,
                'cursors': cursors, 'selection': 'cross_service_spread',
                'sampled_files': min(len(selected), limit), 'total_files': len(selected),
                'note': '跨服务分层取样；按 cursors 继续读取，日志语义由 LLM 判断。'}
    rows, cursor = [], None
    for index, path in enumerate(selected):
        lower = start_line if index == 0 else 1
        part, cursor = _slice(sample, path, lower, limit - len(rows))
        rows.extend(part)
        if cursor or len(rows) >= limit:
            break
    return {'files': inventory, 'records': rows, 'next_cursor': cursor,
            'selection': 'physical_line_order', 'note': '包含普通、空白和异常行；日志语义由 LLM 判断。'}


def run(sample, service=None):
    return query(sample, service=service)
