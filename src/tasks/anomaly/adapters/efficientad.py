import types

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.core.builders import build_dataloader
from src.core.errors import LocalAssetError
from src.core.offline import load_local_weights
from src.core.registry import ADAPTERS, MODELS
from src.tasks.anomaly.adapter import AnomalyAdapter
from src.tasks.anomaly.upstream.efficient_ad.torch_model import (
    EfficientAdModel,
    EfficientAdModelSize,
    reduce_tensor_elems,
)


def _freeze(module):
    """nn.ParameterDict.update() replaces its entries with brand-new nn.Parameter objects, which
    default to requires_grad=True -- so mean_std/quantiles must be re-frozen every time they are
    updated (Codex A-P4 Minor #1: these calibration buffers must not leak into the optimizer's
    param group, matching anomalib's student+ae-only configure_optimizers). This alone does not
    remove them from an optimizer already built before on_fit_start/on_fit_end run -- see
    build_efficientad, which freezes the same ParameterDicts at construction time, before
    cli/commands.py#build_optimizer ever captures model.parameters()."""
    for parameter in module.parameters():
        parameter.requires_grad = False


@ADAPTERS.register("efficientad")
class EfficientAdAdapter(AnomalyAdapter):
    """SPEC SS4.3/SS4.5: EfficientAdModel (upstream torch_model.py) carries no lifecycle hooks
    of its own -- teacher pretrained-weight loading, the teacher channel statistics, the
    ImageNette auxiliary stream, and the 90%/99.5% quantile calibration all live in anomalib's
    lightning_model.py, which this project never imports (CON-002). This adapter ports each step
    as plain PyTorch, keeping core/engine.py's on_fit_start(model, loaders, device)/on_fit_end
    signature unchanged -- the auxiliary stream is a third loader that only EfficientAD needs, so
    it is built and consumed here instead of growing the common engine (NFR-003/NFR-005)."""

    def __init__(self, loss_fn, metrics, auxiliary_root=None, auxiliary_seed=42, smooth_sigma=4.0,
                 **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        if auxiliary_root is None:
            raise LocalAssetError(
                "EfficientAdAdapter requires adapter.params.auxiliary_root (ImageNette "
                "class-folder directory, e.g. ${paths.dataset_root}/imagenette2/train)."
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
                "loss": loss.item(),
                "loss_st": loss_st.item(),
                "loss_ae": loss_ae.item(),
                "loss_stae": loss_stae.item(),
            },
        }

    def _next_auxiliary_batch(self, device):
        try:
            images, _ = next(self._auxiliary_iterator)
        except StopIteration:
            # Auxiliary stream is shorter than train_loader in general -- cycle it (SPEC SS4.5).
            self._auxiliary_iterator = iter(self._auxiliary_loader)
            images, _ = next(self._auxiliary_iterator)
        return images.to(device)

    def on_fit_start(self, model, loaders, device):
        # Lightning hook mapping (SPEC SS4.5 table), same order as anomalib's on_train_start:
        # 1) teacher weights, 2) auxiliary stream, 3) teacher channel stats (train-only, no
        # test/valid leakage -- loaders here is {"train", "valid"} per PLAN-P5 SS5).
        load_local_weights(model.teacher, model.teacher_weights_path, strict=True)

        sample_images, _ = next(iter(loaders["train"]))
        image_size = tuple(sample_images.shape[-2:])
        self._build_auxiliary_loader(image_size, device)

        if not model.is_set(model.mean_std):
            mean_std = self._teacher_channel_mean_std(model, loaders["train"], device)
            model.mean_std.update(mean_std)
            _freeze(model.mean_std)

    def on_fit_end(self, model, loaders, device):
        # SPEC SS4.3 on_fit_end order constraint: quantile calibration must finish *before*
        # super().on_fit_end() runs compute_thresholds -- forward() uses self.quantiles to scale
        # the anomaly map, so a threshold computed against un-calibrated scores does not match
        # the calibrated scores evaluate/predict will see later.
        quantiles = self._map_norm_quantiles(model, loaders["valid"], device)
        model.quantiles.update(quantiles)
        _freeze(model.quantiles)
        super().on_fit_end(model, loaders, device)

    def _build_auxiliary_loader(self, image_size, device):
        height, width = image_size
        # Mirrors anomalib lightning_model.py#prepare_imagenette_data exactly: no normalization
        # here either -- EfficientAdModel applies imagenet_norm_batch internally (SPEC SS4.5).
        transform = transforms.Compose([
            transforms.Resize((height * 2, width * 2)),
            transforms.RandomGrayscale(p=0.3),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
        ])
        dataset = ImageFolder(self.auxiliary_root, transform=transform)
        # batch_size=1 always (paper requirement, anomalib hardcodes it independent of the main
        # train batch size); num_workers/drop_last are irrelevant knobs for this internal stream.
        config_data = {"batch_size": 1, "num_workers": 0, "drop_last": False}
        self._auxiliary_loader = build_dataloader(
            dataset, config_data, "train", self, self.auxiliary_seed, device)
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
                if int(target["label"]) == 0:  # good/normal images only (SPEC SS4.3)
                    map_st, map_ae = model.get_maps(image.unsqueeze(0), normalize=False)
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)
        if was_training:
            model.train()

        qa_st, qb_st = self._quantiles_of_maps(maps_st, device)
        qa_ae, qb_ae = self._quantiles_of_maps(maps_ae, device)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _quantiles_of_maps(self, maps, device):
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(device)
        qb = torch.quantile(maps_flat, q=0.995).to(device)
        return qa, qb


@MODELS.register("efficientad_anomaly")
def build_efficientad(weights_path, teacher_out_channels=384, model_size="small", padding=False,
                       pad_maps=True, **params):
    """No-download model factory (SPEC SS4.6): unlike STFPM's constructor, upstream
    EfficientAdModel.__init__ never touches pretrained weights or the network, so no timm-style
    monkeypatch is needed here. Teacher weights are loaded later, in
    EfficientAdAdapter.on_fit_start (table row, SPEC SS4.5/SS4.6) -- this factory only stashes
    the local path for that hook to read, and freezes the teacher (upstream never sets
    requires_grad on it; every teacher forward call is already wrapped in torch.no_grad() inside
    torch_model.py, so this has no numeric effect but keeps the optimizer's param group matching
    anomalib's explicit student+ae-only configure_optimizers, SPEC review axis 'frozen 파라미터')."""
    model = EfficientAdModel(
        teacher_out_channels=teacher_out_channels,
        model_size=EfficientAdModelSize(model_size),
        padding=padding,
        pad_maps=pad_maps,
    )
    for parameter in model.teacher.parameters():
        parameter.requires_grad = False
    # Codex A-P4 Minor #1: build_optimizer (cli/commands.py) runs before EfficientAdAdapter's
    # on_fit_start/on_fit_end hooks ever fire, so it captures whatever Parameter objects exist on
    # mean_std/quantiles right now -- freezing them here, not only after their later .update()
    # calls, is what actually keeps them out of the optimizer's param group.
    _freeze(model.mean_std)
    _freeze(model.quantiles)
    # Codex A-P4 Minor #2: unlike STFPM's TimmFeatureExtractor (which overrides train() to stay
    # in eval whenever requires_grad=False, SPEC SS4.4), upstream EfficientAdModel has no such
    # guard -- the outer model.train() call in core/engine.py's _train_epoch would otherwise flip
    # teacher.training back to True every epoch. The current PDN teacher has no BatchNorm/Dropout
    # so this has no numeric effect today, but the invariant ("teacher stays frozen and in eval")
    # should hold structurally, not by accident of architecture. Binding an instance-level
    # override (not a upstream/ file edit, CON-001) mirrors STFPM's own train()-pinning behavior.
    model.teacher.train = types.MethodType(lambda self, mode=True: torch.nn.Module.train(self, False), model.teacher)
    model.teacher.eval()
    model.teacher_weights_path = weights_path
    return model
