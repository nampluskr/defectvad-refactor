import os

from src.core.registry import MODELS
from src.tasks.anomaly.models.components.dinov2.dinov2_loader import DinoV2Loader
from src.tasks.anomaly.models.components.dinov2.local_preflight import require_local_dinov2_weight
from .torch_model import AnomalyDINOModel


@MODELS.register("anomaly_dino")
@MODELS.register("anomaly_dino_anomaly")
def build_anomaly_dino(
    weights_path=None,
    num_neighbours=1,
    encoder_name="dinov2_vit_small_14",
    masking=False,
    coreset_subsampling=False,
    sampling_ratio=0.1,
    **params,
):
    """No-download AnomalyDINO pure-PyTorch model factory.

    ``AnomalyDINOModel.__init__`` builds its encoder via
    ``DinoV2Loader.from_name(encoder_name)``, which constructs a loader with no
    ``cache_dir`` (defaults to ``torch.hub.get_dir()/dinov2`` and downloads on a
    miss). Same fix as ``build_dinomaly``: ``weights_path`` names one specific
    local ``dinov2_vit*_pretrain.pth`` file so the common config validator's
    ``os.path.isfile`` check passes; its directory (``paths.backbone_root``,
    where every DINOv2 variant's file lives flat) is what ``DinoV2Loader``
    actually needs, so we patch its ``__init__`` for construction only.
    """
    if weights_path is None:
        model = AnomalyDINOModel(
            num_neighbours=num_neighbours,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
        )
    else:
        original_init = DinoV2Loader.__init__
        local_cache_dir = os.path.dirname(weights_path)
        require_local_dinov2_weight(local_cache_dir, encoder_name)

        def _local_init(self, cache_dir=None, vit_factory=None):
            original_init(self, cache_dir=local_cache_dir, vit_factory=vit_factory)

        DinoV2Loader.__init__ = _local_init
        try:
            model = AnomalyDINOModel(
                num_neighbours=num_neighbours,
                encoder_name=encoder_name,
                masking=masking,
                coreset_subsampling=coreset_subsampling,
                sampling_ratio=sampling_ratio,
            )
        finally:
            DinoV2Loader.__init__ = original_init

    return model


__all__ = ["AnomalyDINOModel", "build_anomaly_dino"]
