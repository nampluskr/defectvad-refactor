import torch.nn as nn
from src.core.registry import LOSSES


@LOSSES.register("none")
def build_none_loss(**params):
    return nn.Identity()
