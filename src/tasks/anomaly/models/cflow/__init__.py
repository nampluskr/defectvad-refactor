import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import CflowModel
from .anomaly_map import AnomalyMapGenerator


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
            f"CFLOW backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"CFLOW backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"CFLOW backbone '{backbone}' weights at {weights_path} have unexpected keys: {disallowed}"
        )


@MODELS.register("cflow")
@MODELS.register("cflow_anomaly")
def build_cflow(
    weights_path=None,
    backbone="wide_resnet50_2",
    layers=("layer2", "layer3", "layer4"),
    fiber_batch_size=64,
    decoder="freia-cflow",
    condition_vector=128,
    coupling_blocks=8,
    clamp_alpha=1.9,
    permute_soft=False,
    **params,
):
    """No-download CFLOW model factory.

    ``TimmFeatureExtractor`` only downloads pretrained weights when
    ``pre_trained=True``; construction stays offline via ``pre_trained=False``
    and weights come from ``weights_path`` instead.
    """
    model = CflowModel(
        backbone=backbone,
        layers=list(layers),
        pre_trained=False,
        fiber_batch_size=fiber_batch_size,
        decoder=decoder,
        condition_vector=condition_vector,
        coupling_blocks=coupling_blocks,
        clamp_alpha=clamp_alpha,
        permute_soft=permute_soft,
    )

    if weights_path is not None:
        # model.encoder is the TimmFeatureExtractor wrapper; its inner
        # .feature_extractor is the raw timm module the checkpoint keys match.
        _load_backbone_weights(model.encoder.feature_extractor, weights_path, backbone)

    # The upstream constructor already freezes the encoder; re-assert it here so
    # the freeze provably precedes build_optimizer regardless of load order.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    return model


__all__ = ["CflowModel", "AnomalyMapGenerator", "build_cflow"]
