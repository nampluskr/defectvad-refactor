from src.core.registry import MODELS
from .loss import DraemLoss
from .torch_model import (
    DecoderDiscriminative,
    DecoderReconstructive,
    DiscriminativeSubNetwork,
    DraemModel,
    EncoderDiscriminative,
    EncoderReconstructive,
    ReconstructiveSubNetwork,
)


@MODELS.register("draem")
@MODELS.register("draem_anomaly")
def build_draem(sspcab=False, weights_path=None, **params):
    """Factory function for DRAEM model."""
    return DraemModel(sspcab=sspcab)


__all__ = [
    "DecoderDiscriminative",
    "DecoderReconstructive",
    "DiscriminativeSubNetwork",
    "DraemLoss",
    "DraemModel",
    "EncoderDiscriminative",
    "EncoderReconstructive",
    "ReconstructiveSubNetwork",
    "build_draem",
]
