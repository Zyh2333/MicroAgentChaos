"""Data queries and layer hypothesis reviews, without semantic log templates."""
import importlib
import json
from .core import number
from .rules import LAYERS
from .utils.observation_summary import READERS, read
from .utils.native_data import services

CHECKS = ('check_layer',)
SUBMITS = ('submit_layer',)
DIAGNOSES = ('submit_diagnosis',)
SUBMISSION_TOOLS = SUBMITS + DIAGNOSES
EXTRA = ('read_evidence',)
LAYER_TO_TOOL = {layer: 'check_layer' for layer in LAYERS}
MODULES = READERS + CHECKS + EXTRA + SUBMITS + DIAGNOSES
# check_layer remains available internally (tests, dry runs) but the LLM-facing
# entry points are submit_layer (Stage 1) and submit_diagnosis (Stage 2).
PUBLIC = tuple(name for name in MODULES if name not in CHECKS)
STAGE_TOOLS = {1: SUBMITS, 2: DIAGNOSES}


def referenced_records(value):
    if isinstance(value, dict):
        if value.get('ref') or value.get('references'):
            yield value
        else:
            for child in value.values():
                yield from referenced_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from referenced_records(child)


class Registry:
    def __init__(self, sample):
        self.sample = sample
        self.modules = {name: importlib.import_module(f'{__package__}.tools.{name}') for name in MODULES}
        self.evidence, self.mechanisms, self.calls = {}, {}, []
        self.exposed_mechanisms = set()
        self.verifications = []
        self.dedupe = {}
        self.agent = None  # set by CounterfactualAgent; submit_layer needs the ledger

    def schemas(self, keep=None, stage=1):
        allowed = STAGE_TOOLS.get(stage, ())
        names = [name for name in PUBLIC
                 if name not in SUBMISSION_TOOLS or name in allowed]
        if keep is not None:
            names = [name for name in names if name in keep]
        return [{'type': 'function', 'function': {'name': name, 'description': module.DESCRIPTION,
            'parameters': {'type': 'object', 'properties': {**({'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100}} if name in READERS else {}),
                **getattr(module, 'PARAMETERS', {})}, 'additionalProperties': False}}}
                for name, module in ((name, self.modules[name]) for name in names)]

    def invoke(self, name, arguments=None, stage=1):
        if name not in self.modules:
            raise ValueError('未知数据工具；不提供日志规则分类工具')
        arguments = {} if arguments is None else arguments
        properties = {**({'limit': {'type': 'integer', 'minimum': 1, 'maximum': 100}} if name in READERS else {}),
                      **getattr(self.modules[name], 'PARAMETERS', {})}
        if not isinstance(arguments, dict) or set(arguments) - set(properties):
            raise ValueError('不支持的工具参数')
        for key, value in arguments.items():
            schema = properties[key]
            kind = schema['type']
            if (kind == 'string' and not isinstance(value, str) or kind == 'integer' and type(value) is not int
                    or kind == 'number' and not number(value) or kind == 'array' and not isinstance(value, list)):
                raise ValueError(f'{key} 参数类型错误')
            if 'minimum' in schema and value < schema['minimum'] or 'maximum' in schema and value > schema['maximum']:
                raise ValueError(f'{key} 超出允许范围')
            if 'enum' in schema and value not in schema['enum']:
                raise ValueError(f'{key} 不在允许选项中')
        if name in SUBMISSION_TOOLS:
            if self.agent is None:
                raise ValueError(f'{name} 需要 agent 上下文')
            return self.modules[name].run(self.agent, arguments, stage)
        if name in CHECKS:
            self.calls.append(name)
            from .utils.summarize import clean
            return clean(self.modules[name].review(self, arguments, stage))
        if name in EXTRA:
            self.calls.append(name)
            return self.modules[name].run(self, arguments)
        service = arguments.get('service')
        if service is not None and service not in services(self.sample):
            raise ValueError('service 必须来自已读取的数据清单')
        limit = arguments.get('limit', 12)
        key = (stage, name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
        if key in self.dedupe:
            prior = self.dedupe[key]
            return {'evidence_id': prior, 'duplicate_of': prior,
                    'note': '相同查询本阶段已执行；直接引用该 evidence_id 下的记录，'
                            '或用 read_evidence 展开'}
        self.calls.append(name)
        result = read(self.sample, name, service, limit,
                      **{k: v for k, v in arguments.items() if k not in ('service', 'limit')})
        eid = f'observation-{len(self.calls)}'
        self.evidence[eid] = {'value': None, 'tool': name, 'reason': '数据查询；语义由 LLM 推理'}
        for i, record in enumerate(referenced_records(result), 1):
            refs = [record['ref']] if record.get('ref') else record['references']
            text = record.get('text')
            if text is None:
                text = json.dumps(record.get('data', record), ensure_ascii=False)
            if not text.strip():
                continue
            rid = f'{eid}:record:{i}'
            record['evidence_id'] = rid
            self.evidence[rid] = {'kind': 'raw_observation', 'value': None, 'tool': name,
                'stage': stage, 'text': text, 'references': refs,
                'reason': '程序仅核验引用来源，诊断语义由 LLM 判断'}
        self.dedupe[key] = eid
        from .utils.background_noise import policy
        return {'evidence_id': eid, 'data': result, 'input_filter': policy(self.sample).report()}
