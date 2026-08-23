import timm
import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .anomaly_map import AnomalyMapGenerator
from .feature_extraction import LayerNormFeatureExtractor, get_feature_extractor
from .loss import UFlowLoss
from .torch_model import AffineCouplingSubnet, UflowModel


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
            f"U-Flow backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"U-Flow backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"U-Flow backbone '{backbone}' weights at {weights_path} have unexpected keys: {disallowed}"
        )


@MODELS.register("uflow")
@MODELS.register("uflow_anomaly")
def build_uflow(
    weights_path=None,
    backbone="resnet18",
    input_size=(256, 256),
    flow_steps=4,
    affine_clamp=2.0,
    affine_subnet_channels_ratio=1.0,
    permute_soft=False,
    **params,
):
    """No-download U-Flow model factory.

    Wraps timm model creation with ``pretrained=False`` to prevent network access,
    and loads local weights into the backbone when ``weights_path`` is provided.
    """
    original_create_model = timm.create_model

    def no_download_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = no_download_create_model
    try:
        model = UflowModel(
            input_size=tuple(input_size),
            flow_steps=flow_steps,
            backbone=backbone,
            affine_clamp=affine_clamp,
            affine_subnet_channels_ratio=affine_subnet_channels_ratio,
            permute_soft=permute_soft,
        )
    finally:
        timm.create_model = original_create_model

    if weights_path is not None:
        if hasattr(model.feature_extractor, "feature_extractor"):
            _load_backbone_weights(model.feature_extractor.feature_extractor, weights_path, backbone)

    if hasattr(model.feature_extractor, "feature_extractor"):
        for param in model.feature_extractor.feature_extractor.parameters():
            param.requires_grad = False

    return model


__all__ = [
    "AffineCouplingSubnet",
    "AnomalyMapGenerator",
    "LayerNormFeatureExtractor",
    "UFlowLoss",
    "UflowModel",
    "build_uflow",
    "get_feature_extractor",
]
