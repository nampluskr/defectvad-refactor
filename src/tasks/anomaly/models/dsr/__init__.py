from pathlib import Path
from src.core.registry import MODELS
from .loss import DsrSecondStageLoss, DsrThirdStageLoss
from .anomaly_generator import DsrAnomalyGenerator
from .torch_model import (
    DiscreteLatentModel,
    DsrModel,
    ImageReconstructionNetwork,
    SubspaceRestrictionModule,
    UpsamplingModule,
)


@MODELS.register("dsr")
@MODELS.register("dsr_anomaly")
def build_dsr(
    latent_anomaly_strength=0.2,
    embedding_dim=128,
    num_embeddings=4096,
    num_hiddens=128,
    num_residual_layers=2,
    num_residual_hiddens=64,
    weights_path=None,
    **params,
):
    """Factory function for DSR model."""
    model = DsrModel(
        latent_anomaly_strength=latent_anomaly_strength,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        num_hiddens=num_hiddens,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
    )
    if weights_path is not None:
        model.load_pretrained_discrete_model_weights(Path(weights_path))
    return model


__all__ = [
    "DiscreteLatentModel",
    "DsrAnomalyGenerator",
    "DsrModel",
    "DsrSecondStageLoss",
    "DsrThirdStageLoss",
    "ImageReconstructionNetwork",
    "SubspaceRestrictionModule",
    "UpsamplingModule",
    "build_dsr",
]
