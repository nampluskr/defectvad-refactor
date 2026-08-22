from torchmetrics.classification import BinaryAUROC

from src.core.registry import METRICS


@METRICS.register("pixel_auroc")
def build_pixel_auroc(**params):
    return BinaryAUROC(**params)
