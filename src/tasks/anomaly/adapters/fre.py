import torch

from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("fre")
class FreAdapter(AnomalyAdapter):
    """FRE (Feature Reconstruction Error) model adapter.

    In the training pass, features are extracted from the pretrained CNN backbone
    and reconstructed using the tied autoencoder. The loss is computed as the MSE
    between original and reconstructed features.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.loss_fn = torch.nn.MSELoss()

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        features_in, features_out, _ = model.get_features(images)
        loss = self.loss_fn(features_in, features_out)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
