import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import PatchcoreModel
from .anomaly_map import AnomalyMapGenerator


def _load_backbone_weights(extractor, weights_path, backbone):
    """Load local backbone weights into the timm feature extractor.

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
            f"PatchCore backbone '{backbone}' weights at {weights_path} do not match "
            f"the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"PatchCore backbone '{backbone}' weights at {weights_path} are missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"PatchCore backbone '{backbone}' weights at {weights_path} have unexpected keys: {disallowed}"
        )


@MODELS.register("patchcore")
@MODELS.register("patchcore_anomaly")
def build_patchcore(
    weights_path=None,
    backbone="wide_resnet50_2",
    layers=("layer2", "layer3"),
    num_neighbors=9,
    **params,
):
    """No-download PatchCore model factory.

    PatchCore performs no gradient training: it fills a memory bank with patch
    embeddings and scores by nearest-neighbour search. ``TimmFeatureExtractor``
    already forces ``eval()`` and wraps its forward in ``torch.no_grad()``, so no
    gradient ever reaches the backbone and no instance-level ``train`` binding is
    needed.

    Parameters are deliberately left with ``requires_grad=True``: freezing them
    would leave ``build_optimizer`` with an empty parameter list, which PyTorch
    rejects. Since the forward pass runs under ``no_grad``, the optimizer receives
    no gradients and is a no-op -- the backbone is effectively frozen regardless.
    """
    # pre_trained=False is the only path to timm.create_model here, so construction
    # stays offline. Weights come from weights_path.
    model = PatchcoreModel(
        layers=list(layers),
        backbone=backbone,
        pre_trained=False,
        num_neighbors=num_neighbors,
    )

    if weights_path is not None:
        # model.feature_extractor is the TimmFeatureExtractor wrapper; its inner
        # .feature_extractor is the raw timm module the checkpoint keys match.
        _load_backbone_weights(model.feature_extractor.feature_extractor, weights_path, backbone)

    return model


__all__ = ["PatchcoreModel", "AnomalyMapGenerator", "build_patchcore"]
