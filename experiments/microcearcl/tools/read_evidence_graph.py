"""Inspect the evidence graph built from records already read in this run.

The graph is advisory and independently computed from the tool outputs the agent
has already seen.  It answers three questions the transcript makes hard:

* which citations are the *same* record (and therefore not extra corroboration),
* which citations belong to one execution chain rather than independent sources,
* which propositions are still unresolved and what kind of evidence would separate
  them.

It never classifies a fault and never reads a label.
"""
NAME = 'read_evidence_graph'
DESCRIPTION = ('查询由已读记录构建的证据图：证据包、重复引用、依赖/独立关系、'
               '命题的独立支持数与未解决命题的下一步查询建议。不判定故障类型。')
PARAMETERS = {
    'view': {'type': 'string', 'enum': ['summary', 'bundles', 'edges', 'support', 'needs']},
    'evidence_id': {'type': 'string', 'description': '查看单条观测的关系（配合 view=edges）'},
    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
}


def run(registry, arguments):
    bridge = getattr(registry.agent, 'evidence_graph', None)
    if bridge is None:
        raise ValueError('证据图未启用；本轮请使用原始观测工具')
    view = arguments.get('view', 'summary')
    limit = arguments.get('limit', 12)
    if view == 'edges':
        evidence_id = arguments.get('evidence_id')
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError('view=edges 需要 evidence_id')
        result = bridge.edges_of(evidence_id)
        result['neighbors'] = result.get('neighbors', [])[:limit]
        return result
    if view == 'bundles':
        return {'components': bridge.components(limit),
                'note': '同一组件内的记录不构成相互独立的支持'}
    if view == 'support':
        layers = getattr(registry.agent, 'assessments', {})
        diagnosis = getattr(registry.agent, 'final', None) or getattr(
            registry.agent, 'last_stage_two', None)
        return {'support': bridge.support(layers, diagnosis, limit),
                'note': 'bundles 是该命题的独立证据包数；citations 是引用条数'}
    if view == 'needs':
        layers = getattr(registry.agent, 'assessments', {})
        diagnosis = getattr(registry.agent, 'final', None) or getattr(
            registry.agent, 'last_stage_two', None)
        return {'needs': bridge.needs(layers, diagnosis, limit),
                'note': '按可区分度排序；工具建议依据缺口类型，不依据故障类型'}
    summary = bridge.graph_summary()
    summary['observations'] = len(registry.evidence)
    return summary
