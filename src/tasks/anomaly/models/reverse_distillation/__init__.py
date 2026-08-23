import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import ReverseDistillationModel
from .anomaly_map import AnomalyMapGenerationMode, AnomalyMapGenerator
from .loss import ReverseDistillationLoss


def _load_backbone_weights(extractor, weights_path, backbone):
    """Load local backbone weights into the timm encoder.

    Fails loudly on any missing key. Unexpected keys are tolerated only when they
    belong to a top-level submodule the extractor does not have at all -- timm's
    ``features_only`` prunes stages outside ``out_indices`` (Reverse Distillation
    reads ``layer1``..``layer3``, so ``layer4`` is absent) and drops the classifier
    head (``fc``). An unexpected key under a submodule the extractor does have
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
        # Shape mismatches surface here (e.g. resnet18 weights for wide_resnet50_2).
        raise LocalAssetError(
            f"Reverse Distillation backbone '{backbone}' weights at {weights_path} do not "
            f"match the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"Reverse Distillation backbone '{backbone}' weights at {weights_path} are "
            f"missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"Reverse Distillation backbone '{backbone}' weights at {weights_path} have "
            f"unexpected keys: {disallowed}"
        )


@MODELS.register("reverse_distillation")
@MODELS.register("reverse_distillation_anomaly")
def build_reverse_distillation(
    weights_path=None,
    backbone="wide_resnet50_2",
    input_size=(256, 256),
    layers=("layer1", "layer2", "layer3"),
    anomaly_map_mode="add",
    **params,
):
    """No-download Reverse Distillation model factory.

    ``input_size`` must match ``data.image_size``: ``AnomalyMapGenerator`` allocates
    the anomaly map at that fixed resolution, so a mismatch silently scores against
    the wrong canvas rather than failing.

    ``anomaly_map_mode`` arrives from config as a plain string and is normalised to
    the enum here. A raw string would also work -- the ``str`` mixin keeps equality
    and hashing consistent, so upstream's ``mode not in {ADD, MULTIPLY}`` set lookup
    accepts it -- but converting fails fast on a typo with a message that names the
    valid values, instead of letting an unchecked string reach the model.
    """
    # pre_trained=False is the only path to timm.create_model here, so construction
    # stays offline. Weights come from weights_path.
    model = ReverseDistillationModel(
        backbone=backbone,
        input_size=tuple(input_size),
        layers=list(layers),
        anomaly_map_mode=AnomalyMapGenerationMode(anomaly_map_mode),
        pre_trained=False,
    )

    if weights_path is not None:
        # model.encoder is the TimmFeatureExtractor wrapper; its inner
        # .feature_extractor is the raw timm module the checkpoint keys match.
        _load_backbone_weights(model.encoder.feature_extractor, weights_path, backbone)

    # anomalib optimises decoder + bottleneck only (configure_optimizers). Freezing
    # the encoder here -- before build_optimizer runs -- makes the trainable set
    # exactly that pair, so the common single-parameter-group builder reproduces
    # anomalib's parameter selection without any change to core/builders.py.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    return model


__all__ = [
    "ReverseDistillationModel",
    "ReverseDistillationLoss",
    "AnomalyMapGenerator",
    "AnomalyMapGenerationMode",
    "build_reverse_distillation",
]
