import types

import torch
import torchvision
from torch import nn

from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import UniNetModel
from .components import UniNetLoss


def _load_teacher_weights(teacher, weights_path, backbone):
    """Load local torchvision weights into a UniNet teacher feature extractor.

    ``Teachers._get_teacher`` wraps the raw torchvision model in
    ``create_feature_extractor``, which keeps the original module hierarchy
    under it. State dict keys therefore match the plain torchvision checkpoint
    as-is -- no key prefix stripping is needed, unlike the timm-wrapped models.
    """
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    present_top_level = {key.split(".", 1)[0] for key in teacher.state_dict()}
    try:
        missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise LocalAssetError(
            f"UniNet teacher backbone '{backbone}' weights at {weights_path} do not "
            f"match the model structure: {exc}"
        ) from exc

    if missing:
        raise LocalAssetError(
            f"UniNet teacher backbone '{backbone}' weights at {weights_path} are "
            f"missing keys: {missing}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"UniNet teacher backbone '{backbone}' weights at {weights_path} have "
            f"unexpected keys: {disallowed}"
        )


@MODELS.register("uninet")
@MODELS.register("uninet_anomaly")
def build_uninet(
    weights_path=None,
    student_backbone="wide_resnet50_2",
    teacher_backbone="wide_resnet50_2",
    temperature=0.1,
    **params,
):
    """No-download UniNet pure-PyTorch model factory.

    ``Teachers._get_teacher`` calls ``getattr(torchvision.models, backbone)(pretrained=True)``,
    which downloads. We swap in a ``pretrained=False`` wrapper for the duration of
    construction only, then inject local weights into both the frozen
    ``source_teacher`` and the trainable ``target_teacher`` (they start from the
    same checkpoint upstream).

    ``source_teacher`` is permanently frozen and forced into eval mode: upstream's
    ``Teachers.__init__`` calls ``.eval()`` once, but that flag does not survive the
    common engine's per-epoch ``model.train()`` (which recurses into all submodules).
    An instance-level ``train`` override, like EfficientAD's teacher, makes eval
    mode stick. ``target_teacher`` stays trainable (upstream optimizes it at a low
    learning rate) and follows the engine's normal train/eval switching.
    """
    original_ctor = getattr(torchvision.models, teacher_backbone)

    def _no_download_ctor(*ctor_args, **ctor_kwargs):
        ctor_kwargs["pretrained"] = False
        return original_ctor(*ctor_args, **ctor_kwargs)

    setattr(torchvision.models, teacher_backbone, _no_download_ctor)
    try:
        loss = UniNetLoss(temperature=temperature)
        model = UniNetModel(student_backbone=student_backbone, teacher_backbone=teacher_backbone, loss=loss)
    finally:
        setattr(torchvision.models, teacher_backbone, original_ctor)

    if weights_path is not None:
        _load_teacher_weights(model.teachers.source_teacher, weights_path, teacher_backbone)
        _load_teacher_weights(model.teachers.target_teacher, weights_path, teacher_backbone)

    for parameter in model.teachers.source_teacher.parameters():
        parameter.requires_grad = False

    model.teachers.source_teacher.train = types.MethodType(
        lambda self, mode=True: nn.Module.train(self, False), model.teachers.source_teacher
    )
    model.teachers.source_teacher.eval()

    return model


__all__ = ["UniNetModel", "UniNetLoss", "build_uninet"]
