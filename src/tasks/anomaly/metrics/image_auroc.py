from src.core.registry import METRICS
from .rank_auroc import RankAUROC


@METRICS.register("image_auroc")
def build_image_auroc(**params):
    return RankAUROC(**params)
