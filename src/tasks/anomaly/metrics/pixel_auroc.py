from src.core.registry import METRICS
from .rank_auroc import RankAUROC


@METRICS.register("pixel_auroc")
def build_pixel_auroc(**params):
    return RankAUROC(**params)
