from src.core.registry import MODELS
from .loss import DiscriminatorLoss, GeneratorLoss
from .torch_model import Decoder, Discriminator, Encoder, GanomalyModel, Generator


@MODELS.register("ganomaly")
@MODELS.register("ganomaly_anomaly")
def build_ganomaly(
    input_size=(256, 256),
    num_input_channels=3,
    n_features=64,
    latent_vec_size=100,
    extra_layers=0,
    add_final_conv_layer=True,
    weights_path=None,
    **params,
):
    """Factory function for GANomaly model."""
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    elif isinstance(input_size, list):
        input_size = tuple(input_size)

    return GanomalyModel(
        input_size=input_size,
        num_input_channels=num_input_channels,
        n_features=n_features,
        latent_vec_size=latent_vec_size,
        extra_layers=extra_layers,
        add_final_conv_layer=add_final_conv_layer,
    )


__all__ = [
    "Decoder",
    "Discriminator",
    "DiscriminatorLoss",
    "Encoder",
    "GanomalyModel",
    "Generator",
    "GeneratorLoss",
    "build_ganomaly",
]
