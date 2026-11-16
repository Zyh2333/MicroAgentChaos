"""Layer hypothesis review; no dataset-specific log matching."""
from ..utils.check_observations import PARAMETERS, check
NAME = 'check_interaction'
DESCRIPTION = '复核接口交互与响应传递的观测、预期和对照引用；用 rule_ids 标注 rN 预测，返回证据缺口。'


def review(registry, arguments, stage):
    return check(registry, 'F3', arguments, stage)
