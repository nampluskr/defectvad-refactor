import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import FREModel, TiedAE


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
            f"FRE backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"FRE backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"FRE backbone '{backbone}' weights at {weights_path} have unexpected keys: {disallowed}"
        )


@MODELS.register("fre")
@MODELS.register("fre_anomaly")
def build_fre(
    weights_path=None,
    backbone="resnet50",
    layer="layer3",
    input_dim=65536,
    latent_dim=220,
    pooling_kernel_size=2,
    **params,
):
    """No-download FRE model factory.

    ``TimmFeatureExtractor`` only downloads pretrained weights when
    ``pre_trained=True``; construction stays offline via ``pre_trained=False``
    and weights come from ``weights_path`` instead.
    """
    model = FREModel(
        backbone=backbone,
        layer=layer,
        input_dim=input_dim,
        latent_dim=latent_dim,
        pre_trained=False,
        pooling_kernel_size=pooling_kernel_size,
    )

    if weights_path is not None:
        _load_backbone_weights(model.feature_extractor.feature_extractor, weights_path, backbone)

    # Freeze the feature extractor backbone parameters
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False

    return model


__all__ = ["FREModel", "TiedAE", "build_fre"]
