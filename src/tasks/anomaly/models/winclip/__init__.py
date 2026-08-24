import torch

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import WinClipModel

import open_clip


@MODELS.register("winclip")
@MODELS.register("winclip_anomaly")
def build_winclip(weights_path=None, scales=(2, 3), **params):
    """No-download WinCLIP factory.

    ``WinClipModel.__init__`` hardcodes ``self.pretrained = PRETRAINED`` (the tag
    ``"laion400m_e31"``) and passes it straight to
    ``open_clip.create_model_and_transforms``, which resolves the tag remotely (download or
    HF hub lookup). We force ``pretrained=None`` for the single constructor call — this
    builds the CLIP architecture with random weights and touches no network, mirroring
    ``build_stfpm``'s ``pretrained=False`` substitution — then load the local checkpoint
    into ``model.clip`` (the full open_clip ``CLIP`` submodule) directly afterward.

    ``class_name`` is intentionally left unset here: passing it during construction would
    compute text embeddings from the random-init CLIP before real weights are loaded.
    ``WinclipAdapter.on_validation_start`` calls ``model.setup(class_name, ref_images)``
    once the correct weights are in place, matching upstream's own two-phase
    construct-then-setup design (its Lightning wrapper does the same in its ``setup()``
    hook).
    """
    original_create = open_clip.create_model_and_transforms

    def no_download_create(*args, **kwargs):
        kwargs["pretrained"] = None
        return original_create(*args, **kwargs)

    open_clip.create_model_and_transforms = no_download_create
    try:
        model = WinClipModel(scales=tuple(scales), apply_transform=False)
    finally:
        open_clip.create_model_and_transforms = original_create

    if weights_path is None:
        raise LocalAssetError(
            "WinCLIP requires a local open_clip checkpoint. Set model.params.weights_path "
            "to the local ViT-B-16-plus-240 / laion400m_e31 weights file."
        )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.clip.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise LocalAssetError(
            f"WinCLIP weights at {weights_path} key mismatch: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )

    return model


__all__ = ["WinClipModel", "build_winclip"]
