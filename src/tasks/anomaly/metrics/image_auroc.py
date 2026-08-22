from torchmetrics.classification import BinaryAUROC

from src.core.registry import METRICS


@METRICS.register("image_auroc")
def build_image_auroc(**params):
    return BinaryAUROC(**params)
