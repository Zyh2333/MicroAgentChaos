"""Layer hypothesis review; no dataset-specific log matching."""
from ..utils.check_observations import PARAMETERS, check
NAME = 'check_regions'
DESCRIPTION = '复核基础设施与区域层的观测、预期和对照引用；用 rule_ids 标注 rN 预测，返回证据缺口。'


def review(registry, arguments, stage):
    return check(registry, 'F0', arguments, stage)
