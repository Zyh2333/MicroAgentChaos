"""Layer hypothesis review; no dataset-specific log matching."""
from ..utils.check_observations import PARAMETERS, check
NAME = 'check_model'
DESCRIPTION = '复核外部模型依赖的观测、预期和对照引用；用 rule_ids 标注 rN 预测，返回证据缺口。'


def review(registry, arguments, stage):
    return check(registry, 'F2', arguments, stage)
