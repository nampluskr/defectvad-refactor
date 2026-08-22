import os
import types

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.core.builders import build_dataloader
from src.core.errors import LocalAssetError
from src.core.offline import load_local_weights
from src.core.registry import ADAPTERS
from src.tasks.anomaly.adapters.base import AnomalyAdapter
from src.tasks.anomaly.models.efficientad.torch_model import (
    EfficientAdModel,
    EfficientAdModelSize,
    reduce_tensor_elems,
)


def _freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


@ADAPTERS.register("efficientad")
class EfficientAdAdapter(AnomalyAdapter):
    """Adapter for EfficientAD anomaly detection model."""

    def __init__(
        self,
        loss_fn,
        metrics,
        auxiliary_root=None,
        auxiliary_seed=42,
        smooth_sigma=4.0,
        **params,
    ):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        if auxiliary_root is None:
            raise LocalAssetError(
                "EfficientAdAdapter requires adapter.params.auxiliary_root (ImageNette class directory)."
            )
        self.auxiliary_root = auxiliary_root
        self.auxiliary_seed = auxiliary_seed
        self._auxiliary_loader = None
        self._auxiliary_iterator = None

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        batch_imagenet = self._next_auxiliary_batch(device)
        loss_st, loss_ae, loss_stae = model(batch=images, batch_imagenet=batch_imagenet)
        loss = loss_st + loss_ae + loss_stae
        return {
            "loss": loss,
            "loss_dict": {
                "loss": float(loss.detach()),
                "loss_st": float(loss_st.detach()),
                "loss_ae": float(loss_ae.detach()),
                "loss_stae": float(loss_stae.detach()),
            },
        }

    def _next_auxiliary_batch(self, device):
        try:
            images, _ = next(self._auxiliary_iterator)
        except StopIteration:
            self._auxiliary_iterator = iter(self._auxiliary_loader)
            images, _ = next(self._auxiliary_iterator)
        return images.to(device)

    def on_fit_start(self, model, loaders, device):
        load_local_weights(model.teacher, model.teacher_weights_path, strict=True)

        sample_images, _ = next(iter(loaders["train"]))
        image_size = tuple(sample_images.shape[-2:])
        self._build_auxiliary_loader(image_size, device)

        if not model.is_set(model.mean_std):
            mean_std = self._teacher_channel_mean_std(model, loaders["train"], device)
            model.mean_std.update(mean_std)
            _freeze(model.mean_std)

    def on_validation_start(self, model, loaders, device):
        if not model.is_set(model.quantiles):
            self._calibrate_quantiles(model, loaders["valid"], device)

    def on_fit_end(self, model, loaders, device):
        if not model.is_set(model.quantiles):
            self._calibrate_quantiles(model, loaders["valid"], device)
        super().on_fit_end(model, loaders, device)

    def _calibrate_quantiles(self, model, valid_loader, device):
        devices = [device] if getattr(device, "type", None) == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            quantiles = self._map_norm_quantiles(model, valid_loader, device)
        model.quantiles.update(quantiles)
        _freeze(model.quantiles)

    def _build_auxiliary_loader(self, image_size, device):
        height, width = image_size
        transform = transforms.Compose([
            transforms.Resize((height * 2, width * 2)),
            transforms.RandomGrayscale(p=0.3),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
        ])
        dataset = ImageFolder(self.auxiliary_root, transform=transform)
        config_data = {"batch_size": 1, "num_workers": 0, "drop_last": False}
        self._auxiliary_loader = build_dataloader(
            dataset, config_data, "train", self, self.auxiliary_seed, device
        )
        self._auxiliary_iterator = iter(self._auxiliary_loader)

    @torch.no_grad()
    def _teacher_channel_mean_std(self, model, train_loader, device):
        n = channel_sum = channel_sum_sqr = None
        for batch in train_loader:
            images = batch[0].to(device)
            y = model.teacher(images)
            if n is None:
                num_channels = y.shape[1]
                n = torch.zeros(num_channels, dtype=torch.int64, device=y.device)
                channel_sum = torch.zeros(num_channels, dtype=torch.float32, device=y.device)
                channel_sum_sqr = torch.zeros(num_channels, dtype=torch.float32, device=y.device)
            n += y[:, 0].numel()
            channel_sum += y.sum(dim=[0, 2, 3])
            channel_sum_sqr += (y ** 2).sum(dim=[0, 2, 3])

        channel_mean = channel_sum / n
        channel_std = torch.sqrt(channel_sum_sqr / n - channel_mean ** 2).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]
        return {"mean": channel_mean, "std": channel_std}

    @torch.no_grad()
    def _map_norm_quantiles(self, model, valid_loader, device):
        was_training = model.training
        model.eval()
        maps_st, maps_ae = [], []
        for images, targets in valid_loader:
            images = images.to(device)
            for image, target in zip(images, targets):
                if int(target["label"]) == 0:
                    map_st, map_ae = model.get_maps(image.unsqueeze(0), normalize=False)
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)
        if was_training:
            model.train()

        if not maps_st:
            raise LocalAssetError(
                "EfficientAD quantile calibration needs normal (label == 0) samples in the valid split."
            )

        qa_st, qb_st = self._quantiles_of_maps(maps_st, device)
        qa_ae, qb_ae = self._quantiles_of_maps(maps_ae, device)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _quantiles_of_maps(self, maps, device):
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(device)
        qb = torch.quantile(maps_flat, q=0.995).to(device)
        if not bool(torch.isfinite(qa)) or not bool(torch.isfinite(qb)) or float(qb - qa) <= 0.0:
            raise LocalAssetError(
                f"EfficientAD quantile calibration produced a degenerate range (q0.9={float(qa)}, q0.995={float(qb)})."
            )
        return qa, qb
