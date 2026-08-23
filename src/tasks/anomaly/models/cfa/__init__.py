import torch
import torchvision

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import CfaModel


def _load_backbone_weights(feature_extractor, weights_path, backbone):
    """Load local backbone weights into the FX-traced feature extractor.

    ``create_feature_extractor`` wraps the eager ``torchvision.models`` module
    without renaming its submodules, so the resulting ``GraphModule`` exposes the
    same state-dict keys as the plain backbone. Fails loudly on any missing key.
    Unexpected keys are tolerated only when they belong to a top-level submodule
    the extractor does not have at all (e.g. ``fc`` is dropped once FX prunes the
    graph to the requested return nodes). An unexpected key under a submodule the
    extractor does have means a structural mismatch and is rejected.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    present_top_level = {key.split(".", 1)[0] for key in feature_extractor.state_dict()}
    try:
        missing, unexpected = feature_extractor.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise LocalAssetError(
            f"CFA backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"CFA backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"CFA backbone '{backbone}' weights at {weights_path} have unexpected keys: {disallowed}"
        )


@MODELS.register("cfa")
@MODELS.register("cfa_anomaly")
def build_cfa(
    weights_path=None,
    backbone="wide_resnet50_2",
    gamma_c=1,
    gamma_d=1,
    num_nearest_neighbors=3,
    num_hard_negative_features=3,
    radius=1e-5,
    **params,
):
    """No-download CFA model factory.

    ``CfaModel.__init__`` builds its feature extractor through a module-level
    ``get_feature_extractor(backbone, return_nodes)`` helper in ``torch_model.py``
    that calls ``getattr(torchvision.models, backbone)(pretrained=True)`` directly
    -- not through ``TimmFeatureExtractor``. Construction is kept offline by
    temporarily replacing the ``torchvision.models.<backbone>`` constructor with a
    ``pretrained=False`` wrapper for the duration of the call, then restoring it.
    ``get_feature_extractor`` resolves ``torchvision.models.<backbone>`` fresh on
    every call via ``getattr``, so patching the module attribute is enough --
    ``torch_model.py`` need not be touched.
    """
    original_ctor = getattr(torchvision.models, backbone)

    def no_download_ctor(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_ctor(*args, **kwargs)

    setattr(torchvision.models, backbone, no_download_ctor)
    try:
        model = CfaModel(
            backbone=backbone,
            gamma_c=gamma_c,
            gamma_d=gamma_d,
            num_nearest_neighbors=num_nearest_neighbors,
            num_hard_negative_features=num_hard_negative_features,
            radius=radius,
        )
    finally:
        setattr(torchvision.models, backbone, original_ctor)

    if weights_path is not None:
        _load_backbone_weights(model.feature_extractor, weights_path, backbone)

    # forward() only ever runs the backbone under torch.no_grad(), so freezing it
    # here keeps it out of build_optimizer's single parameter group -- matching
    # the frozen-backbone convention every other model in this project uses.
    # Functionally identical to anomalib's own behaviour, which leaves these
    # params trainable but never lets a gradient reach them.
    for param in model.feature_extractor.parameters():
        param.requires_grad_(False)

    return model


__all__ = ["CfaModel", "build_cfa"]
