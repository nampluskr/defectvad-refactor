import timm
from src.core.errors import LocalAssetError
from src.core.registry import MODELS
from .torch_model import STFPMModel
from .loss import STFPMLoss
from .anomaly_map import AnomalyMapGenerator


@MODELS.register("stfpm")
@MODELS.register("stfpm_anomaly")
def build_stfpm(weights_path: str = None, layers=("layer1", "layer2", "layer3"), backbone="resnet18", **params):
    """No-download STFPM model factory.
    
    Builds the pure-PyTorch STFPMModel and loads local teacher weights if provided.
    """
    original_create_model = timm.create_model

    def no_download_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = no_download_create_model
    try:
        model = STFPMModel(layers=list(layers), backbone=backbone)
    finally:
        timm.create_model = original_create_model

    if weights_path is not None:
        import torch
        extractor = model.teacher_model.feature_extractor
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        present_top_level = {key.split(".", 1)[0] for key in extractor.state_dict()}
        
        # Filter state dict if wrapped
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("model.", "").replace("feature_extractor.", "")
            clean_state_dict[clean_k] = v

        missing, unexpected = extractor.load_state_dict(clean_state_dict, strict=False)
        disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
        if disallowed:
            raise LocalAssetError(
                f"STFPM teacher weights at {weights_path} have unexpected keys: {disallowed}"
            )

    return model


__all__ = ["STFPMModel", "STFPMLoss", "AnomalyMapGenerator", "build_stfpm"]
