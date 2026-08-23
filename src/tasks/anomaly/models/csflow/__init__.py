import torch
import torchvision.models as tv_models

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .anomaly_map import AnomalyMapGenerator, AnomalyMapMode
from .loss import CsFlowLoss
from .torch_model import (
    CrossConvolutions,
    CrossScaleFlow,
    CsFlowModel,
    CsFlowMultiScaleFeatureExtractor,
    ParallelGlowCouplingLayer,
    ParallelPermute,
)


def _load_backbone_weights(extractor, weights_path, backbone):
    """Load local backbone weights into the FX-traced feature extractor.

    ``create_feature_extractor`` wraps the eager EfficientNet module
    without renaming its submodules, so the resulting ``GraphModule`` exposes the
    same state-dict keys as the plain backbone up to the extracted layer.
    Fails loudly on any missing key.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    try:
        missing, unexpected = extractor.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise LocalAssetError(
            f"CS-Flow backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"CS-Flow backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )


@MODELS.register("csflow")
@MODELS.register("csflow_anomaly")
def build_csflow(
    weights_path=None,
    input_size=(256, 256),
    cross_conv_hidden_channels=1024,
    n_coupling_blocks=4,
    clamp=3,
    num_channels=3,
    **params,
):
    """No-download CS-Flow model factory.

    Wraps torchvision efficientnet_b5 creation with ``weights=None`` to prevent network access,
    and loads local weights into the backbone when ``weights_path`` is provided.
    """
    import src.tasks.anomaly.models.csflow.torch_model as csflow_torch_model

    original_b5 = csflow_torch_model.efficientnet_b5

    def no_download_b5(*args, **kwargs):
        kwargs["weights"] = None
        return original_b5(*args, **kwargs)

    csflow_torch_model.efficientnet_b5 = no_download_b5
    try:
        model = CsFlowModel(
            input_size=tuple(input_size),
            cross_conv_hidden_channels=cross_conv_hidden_channels,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
            num_channels=num_channels,
        )
    finally:
        csflow_torch_model.efficientnet_b5 = original_b5

    if weights_path is not None:
        extractor = model.feature_extractor.feature_extractor
        target = getattr(extractor, "feature_extractor", extractor)
        _load_backbone_weights(target, weights_path, "efficientnet_b5")

    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    return model


__all__ = [
    "AnomalyMapGenerator",
    "AnomalyMapMode",
    "CrossConvolutions",
    "CrossScaleFlow",
    "CsFlowLoss",
    "CsFlowModel",
    "CsFlowMultiScaleFeatureExtractor",
    "ParallelGlowCouplingLayer",
    "ParallelPermute",
    "build_csflow",
]
