"""Read full record text for an evidence_id already returned by a tool.

Transcripts only carry a bounded preview; this tool expands a record (or all
records under an observation) so citations stay precise without re-reading the
source files.
"""
from ..utils.evidence_ids import records_for

NAME = 'read_evidence'
DESCRIPTION = '按 evidence_id 回读已读取记录的完整文本；支持 observation-N 与 observation-N:record:i。'
PARAMETERS = {'evidence_id': {'type': 'string'},
              'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50}}


def run(registry, arguments):
    evidence_id = arguments.get('evidence_id')
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError('需要 evidence_id（observation-N 或 observation-N:record:i）')
    limit = arguments.get('limit', 12)
    fact = registry.evidence.get(evidence_id)
    if fact and fact.get('kind') == 'raw_observation':
        return {'evidence_id': evidence_id, 'text': fact.get('text', ''),
                'references': fact.get('references', []), 'tool': fact.get('tool'),
                'stage': fact.get('stage')}
    if evidence_id.startswith('observation-') and ':record:' not in evidence_id:
        ids = records_for(registry, evidence_id)
        if not ids:
            raise ValueError(f'{evidence_id} 没有已读取记录；先查询数据，或检查 ID 粒度')
        rows = [{'evidence_id': rid, 'text': registry.evidence[rid].get('text', ''),
                 'references': registry.evidence[rid].get('references', [])}
                for rid in ids[:limit]]
        return {'evidence_id': evidence_id, 'count': len(ids), 'returned_records': len(rows),
                'records': rows}
    raise ValueError(f'未知 evidence_id：{evidence_id}；使用 observation-N 或 observation-N:record:i')
