"""Generic layer review; the layer comes from the declarative registry."""
from ..rules import LAYERS
from ..utils.check_observations import PARAMETERS, check

NAME = 'check_layer'
DESCRIPTION = '复核指定故障层的 R 支持、CF 必要性签名和 C 对照引用；layer 必须来自规则注册表。'
PARAMETERS = {**PARAMETERS, 'layer': {'type': 'string'}}


def review(registry, arguments, stage):
    layer = arguments.get('layer')
    if layer not in LAYERS:
        raise ValueError('layer 必须来自规则注册表')
    return check(registry, layer, arguments, stage)
