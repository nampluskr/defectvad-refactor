import types
import torch
from torch import nn

from src.core.registry import MODELS
from .torch_model import EfficientAdModel, EfficientAdModelSize


def _freeze(module_or_dict):
    if isinstance(module_or_dict, (dict, nn.ParameterDict)):
        for val in module_or_dict.values():
            if isinstance(val, (torch.Tensor, nn.Parameter)):
                val.requires_grad = False
    elif isinstance(module_or_dict, nn.Module):
        for param in module_or_dict.parameters():
            param.requires_grad = False


@MODELS.register("efficientad")
@MODELS.register("efficientad_anomaly")
def build_efficientad(
    weights_path: str = None,
    teacher_out_channels: int = 384,
    model_size: str = "small",
    padding: bool = False,
    pad_maps: bool = True,
    **params,
) -> EfficientAdModel:
    """No-download EfficientAD pure-PyTorch model factory.
    
    Instantiates the EfficientAdModel with frozen teacher network.
    """
    model = EfficientAdModel(
        teacher_out_channels=teacher_out_channels,
        model_size=EfficientAdModelSize(model_size),
        padding=padding,
        pad_maps=pad_maps,
    )
    for parameter in model.teacher.parameters():
        parameter.requires_grad = False

    _freeze(model.mean_std)
    _freeze(model.quantiles)

    model.teacher.train = types.MethodType(
        lambda self, mode=True: nn.Module.train(self, False), model.teacher
    )
    model.teacher.eval()
    model.teacher_weights_path = weights_path
    return model


__all__ = ["EfficientAdModel", "EfficientAdModelSize", "build_efficientad"]
