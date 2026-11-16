"""Layer hypothesis review; no dataset-specific log matching."""
from ..utils.check_observations import PARAMETERS, check
NAME = 'check_result'
DESCRIPTION = '复核执行控制与结果完整性的观测、预期和对照引用；用 rule_ids 标注 rN 预测，返回证据缺口。'


def review(registry, arguments, stage):
    return check(registry, 'F4', arguments, stage)
