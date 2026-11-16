"""Layer hypothesis review; no dataset-specific log matching."""
from ..utils.check_observations import PARAMETERS, check
NAME = 'check_service'
DESCRIPTION = '复核服务运行环境与服务入口边界的观测、预期和对照引用；用 rule_ids 标注 rN 预测，返回证据缺口。'


def review(registry, arguments, stage):
    return check(registry, 'F1', arguments, stage)
