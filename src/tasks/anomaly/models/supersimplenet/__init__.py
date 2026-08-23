import timm
import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .anomaly_generator import AnomalyGenerator
from .loss import SSNLoss
from .torch_model import (
    AnomalyMapGenerator,
    FeatureAdapter,
    SegmentationDetectionModule,
    SupersimplenetModel,
    UpscalingFeatureExtractor,
)


def _load_backbone_weights(extractor, weights_path, backbone):
    """Load local backbone weights into the frozen feature extractor.

    Fails loudly on any missing key. Unexpected keys are tolerated only when they
    belong to a top-level submodule the extractor does not have at all -- timm's
    ``features_only`` prunes stages outside ``out_indices`` and drops the
    classifier head. An unexpected key under a submodule the extractor does have
    means a structural mismatch and is rejected.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("model.", "").replace("feature_extractor.", "")
        clean_state_dict[clean_key] = value

    present_top_level = {key.split(".", 1)[0] for key in extractor.state_dict()}
    try:
        missing, unexpected = extractor.load_state_dict(clean_state_dict, strict=False)
    except RuntimeError as exc:
        raise LocalAssetError(
            f"SuperSimpleNet backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"SuperSimpleNet backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"SuperSimpleNet backbone '{backbone}' weights at {weights_path} have unexpected keys: {disallowed}"
        )


@MODELS.register("supersimplenet")
@MODELS.register("supersimplenet_anomaly")
def build_supersimplenet(
    weights_path=None,
    backbone="wide_resnet50_2",
    layers=("layer2", "layer3"),
    perlin_threshold=0.2,
    stop_grad=True,
    adapt_cls_features=False,
    **params,
):
    """No-download SuperSimpleNet model factory.

    Wraps timm model creation with ``pretrained=False`` to prevent network access,
    and loads local weights into the backbone when ``weights_path`` is provided.
    """
    original_create_model = timm.create_model

    def no_download_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = no_download_create_model
    try:
        model = SupersimplenetModel(
            perlin_threshold=perlin_threshold,
            backbone=backbone,
            layers=list(layers),
            stop_grad=stop_grad,
            adapt_cls_features=adapt_cls_features,
        )
    finally:
        timm.create_model = original_create_model

    if weights_path is not None:
        extractor = model.feature_extractor.feature_extractor
        target = getattr(extractor, "feature_extractor", extractor)
        _load_backbone_weights(target, weights_path, backbone)

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    return model


__all__ = [
    "AnomalyGenerator",
    "AnomalyMapGenerator",
    "FeatureAdapter",
    "SSNLoss",
    "SegmentationDetectionModule",
    "SupersimplenetModel",
    "UpscalingFeatureExtractor",
    "build_supersimplenet",
]
