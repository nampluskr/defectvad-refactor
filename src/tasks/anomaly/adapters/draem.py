import os
import torch
from torch import nn

from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.components.generators.perlin import PerlinAnomalyGenerator
from src.tasks.anomaly.models.draem.loss import DraemLoss
from .base import AnomalyAdapter


@ADAPTERS.register("draem")
class DraemAdapter(AnomalyAdapter):
    """DRAEM model adapter."""

    def __init__(
        self,
        loss_fn=None,
        metrics=None,
        smooth_sigma=4.0,
        dtd_dir=None,
        enable_sspcab=False,
        sspcab_lambda=0.1,
        beta=(0.1, 1.0),
        **params,
    ):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.dtd_dir = dtd_dir
        self.enable_sspcab = enable_sspcab
        self.sspcab_lambda = sspcab_lambda
        self.beta = tuple(beta) if isinstance(beta, list) else beta
        self.loss_fn = DraemLoss()
        self.augmenter = None
        self.sspcab_activations = {} if enable_sspcab else None
        self.sspcab_loss = nn.MSELoss() if enable_sspcab else None

    def on_fit_start(self, model, loaders, device):
        super().on_fit_start(model, loaders, device)
        self.augmenter = PerlinAnomalyGenerator(
            anomaly_source_path=self.dtd_dir,
            blend_factor=self.beta,
        )
        if self.enable_sspcab:
            def get_activation(name):
                def hook(_, __, output):
                    self.sspcab_activations[name] = output
                return hook

            model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
            model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

    def train_step(self, model, batch, device):
        input_image = batch[0].to(device)
        augmented_image, anomaly_mask = self.augmenter(input_image)
        reconstruction, prediction = model(augmented_image)
        loss = self.loss_fn(input_image, reconstruction, anomaly_mask, prediction)

        if self.enable_sspcab:
            loss = loss + self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"],
            )

        return {"loss": loss, "loss_dict": {"loss": float(loss.item())}}
