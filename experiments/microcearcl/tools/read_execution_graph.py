"""Paginated native graphs; preserve observations without text-based failure selection."""
from ..core import reference
from ..utils.background_graph import filter_text

NAME = 'read_execution_graph'
DESCRIPTION = '按原始顺序读取执行图、路径与 span；保留成功和失败记录，不匹配日志关键词或预判根因。'
PARAMETERS = {'offset': {'type': 'integer', 'minimum': 0}}


def query(sample, offset=0, limit=12):
    if type(offset) is not int or offset < 0:
        raise ValueError('offset 必须非负')
    relative = f'{sample.namespace}/graph/call_chains.json'
    chains = sample.json(relative, [])
    if not isinstance(chains, list):
        sample.warn('call_chains.json 必须是数组')
        chains = []
    rows = [{'data': filter_text(sample, chain, None, reference(relative, f'/{i}')),
             'ref': reference(relative, f'/{i}')}
            for i, chain in enumerate(chains[offset:offset + limit], offset)]
    files = [str(p.relative_to(sample.path)) for p in sorted(sample.ns('graph/graph_json').glob('*.json'))]
    return {'task_count': len(chains), 'records': rows, 'detail_files': files,
            'next_offset': offset + len(rows) if offset + len(rows) < len(chains) else None}


def run(sample):
    return query(sample)
