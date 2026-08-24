import os

import torch
from torch.nn.init import trunc_normal_

from src.core.registry import MODELS
from src.tasks.anomaly.models.components.dinov2.dinov2_loader import DinoV2Loader
from src.tasks.anomaly.models.components.dinov2.local_preflight import require_local_dinov2_weight
from .torch_model import DinomalyModel


def _initialize_trainable_modules(trainable_modules):
    """Truncated-normal init for Linear, constant init for LayerNorm -- matches
    upstream ``Dinomaly._initialize_trainable_modules`` in ``lightning_model.py``.
    """
    for module in trainable_modules.modules():
        if isinstance(module, torch.nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)


@MODELS.register("dinomaly")
@MODELS.register("dinomaly_anomaly")
def build_dinomaly(
    weights_path=None,
    encoder_name="dinov2reg_vit_base_14",
    bottleneck_dropout=0.2,
    decoder_depth=8,
    target_layers=None,
    fuse_layer_encoder=None,
    fuse_layer_decoder=None,
    remove_class_token=False,
    **params,
):
    """No-download Dinomaly pure-PyTorch model factory.

    ``DinomalyModel.__init__`` builds its DINOv2 encoder via
    ``DinoV2Loader(vit_factory=...).load(encoder_name)`` with no ``cache_dir``
    argument, so it always resolves weights under ``torch.hub.get_dir()/dinov2``
    and downloads on a miss. ``weights_path`` here points at one specific local
    ``dinov2_vit*_[reg4_]pretrain.pth`` file (so the common config validator's
    ``os.path.isfile`` check passes); its *directory* -- ``paths.backbone_root``,
    where every DINOv2 variant's file lives flat -- is what actually matters.
    We patch ``DinoV2Loader.__init__`` for the duration of construction so its
    default ``cache_dir`` becomes that directory, and restore it immediately
    after. ``DinoV2Loader`` itself
    already fails loudly on a structural mismatch (``load_state_dict`` raises on
    shape errors); it only tolerates missing/extra keys the way upstream defines
    for the dinov2 vs. dinov2-reg key-set difference, which we do not alter.

    Only ``bottleneck`` and ``decoder`` are trainable, matching upstream
    ``Dinomaly.__init__`` in ``lightning_model.py`` -- everything else
    (the DINOv2 encoder) stays frozen. The encoder is also unconditionally
    wrapped in ``torch.no_grad()`` inside ``torch_model.py``'s own forward, so
    freezing here only controls what the optimizer selects.
    """
    if weights_path is None:
        model = DinomalyModel(
            encoder_name=encoder_name,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )
    else:
        original_init = DinoV2Loader.__init__

        local_cache_dir = os.path.dirname(weights_path)
        require_local_dinov2_weight(local_cache_dir, encoder_name)

        def _local_init(self, cache_dir=None, vit_factory=None):
            original_init(self, cache_dir=local_cache_dir, vit_factory=vit_factory)

        DinoV2Loader.__init__ = _local_init
        try:
            model = DinomalyModel(
                encoder_name=encoder_name,
                bottleneck_dropout=bottleneck_dropout,
                decoder_depth=decoder_depth,
                target_layers=target_layers,
                fuse_layer_encoder=fuse_layer_encoder,
                fuse_layer_decoder=fuse_layer_decoder,
                remove_class_token=remove_class_token,
            )
        finally:
            DinoV2Loader.__init__ = original_init

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.bottleneck.parameters():
        parameter.requires_grad = True
    for parameter in model.decoder.parameters():
        parameter.requires_grad = True

    trainable_modules = torch.nn.ModuleList([model.bottleneck, model.decoder])
    _initialize_trainable_modules(trainable_modules)

    return model


__all__ = ["DinomalyModel", "build_dinomaly"]
