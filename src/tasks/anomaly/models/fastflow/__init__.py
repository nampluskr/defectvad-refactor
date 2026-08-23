import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import FastflowModel
from .loss import FastflowLoss
from .anomaly_map import AnomalyMapGenerator


def _load_backbone_weights(extractor, weights_path, backbone):
    """Load local backbone weights into the frozen feature extractor.

    Fails loudly on any missing key. Unexpected keys are tolerated only when they
    belong to a top-level submodule the extractor does not have at all -- timm's
    ``features_only`` prunes stages outside ``out_indices`` (e.g. ``layer4``) and
    drops the classifier head (``fc``). An unexpected key under a submodule the
    extractor does have means a structural mismatch and is rejected.
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
        # Shape mismatches surface here (e.g. wide_resnet50_2 weights for resnet18).
        raise LocalAssetError(
            f"FastFlow backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"FastFlow backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"FastFlow backbone '{backbone}' weights at {weights_path} have unexpected keys: {disallowed}"
        )


@MODELS.register("fastflow")
@MODELS.register("fastflow_anomaly")
def build_fastflow(
    weights_path=None,
    input_size=(256, 256),
    backbone="resnet18",
    flow_steps=8,
    conv3x3_only=False,
    hidden_ratio=1.0,
    **params,
):
    """No-download FastFlow model factory.

    ``input_size`` must match ``data.image_size``: the LayerNorm blocks and the
    normalizing-flow blocks are built with fixed spatial dimensions.
    """
    # pre_trained=False is the only path to timm.create_model in this model, so it
    # is sufficient to keep construction offline. Weights come from weights_path.
    model = FastflowModel(
        input_size=tuple(input_size),
        backbone=backbone,
        pre_trained=False,
        flow_steps=flow_steps,
        conv3x3_only=conv3x3_only,
        hidden_ratio=hidden_ratio,
    )

    if weights_path is not None:
        _load_backbone_weights(model.feature_extractor, weights_path, backbone)

    # The upstream constructor already freezes the extractor; re-assert it here so
    # the freeze provably precedes build_optimizer regardless of load order.
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False

    return model


__all__ = ["FastflowModel", "FastflowLoss", "AnomalyMapGenerator", "build_fastflow"]
