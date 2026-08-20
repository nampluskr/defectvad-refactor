import timm

from src.core.errors import LocalAssetError
from src.core.offline import load_local_weights
from src.core.registry import ADAPTERS, MODELS
from src.tasks.anomaly.adapter import AnomalyAdapter
from src.tasks.anomaly.upstream.stfpm.loss import STFPMLoss
from src.tasks.anomaly.upstream.stfpm.torch_model import STFPMModel


@ADAPTERS.register("stfpm")
class StfpmAdapter(AnomalyAdapter):
    """SPEC SS4.1/SS4.3: anomalib's torch_model.py has no train_step -- the loss lives in
    lightning_model.py, which this project never imports (CON-002). This adapter calls upstream
    STFPMLoss directly against the (teacher_features, student_features) tuple that
    STFPMModel.forward returns while self.training is True."""

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.stfpm_loss = STFPMLoss()

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        teacher_features, student_features = model(images)
        loss = self.stfpm_loss(teacher_features, student_features)
        return {"loss": loss, "loss_dict": {"loss": loss.item()}}


@MODELS.register("stfpm_anomaly")
def build_stfpm(weights_path, layers=("layer1", "layer2", "layer3"), backbone="resnet18", **params):
    """No-download model factory, option A of SPEC SS4.6.

    STFPMModel.__init__ builds the teacher with pre_trained=True, which calls
    timm.create_model(..., pretrained=True) and would try to download on this offline sandbox
    (CON-004). Since upstream torch_model.py cannot be changed to pre_trained=False (CON-001,
    option C rejected), timm.create_model -- the confirmed single entry point (SPEC SS4.6) -- is
    monkeypatched to force pretrained=False for the duration of the upstream constructor call
    only, then restored via try/finally. Both teacher and student come out of the constructor
    randomly initialized; the teacher feature extractor is then loaded from a local checkpoint.
    The student stays randomly initialized by design (pre_trained=False already), and teacher
    freeze is left entirely to the upstream constructor (SPEC SS4.4) -- this factory does not
    re-set requires_grad or .eval() itself.
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

    extractor = model.teacher_model.feature_extractor
    # Which top-level submodules the extractor actually has, before loading anything -- with
    # out_indices confined to `layers`, timm prunes trailing resnet stages it doesn't need
    # (e.g. layer4 when layers stops at layer3) in addition to never building a classifier head
    # (features_only=True). A checkpoint key whose top-level name isn't present here belongs to
    # a submodule that structurally does not exist in this extractor, so it is safe to discard.
    # A key whose top-level name IS present but still comes back unexpected would mean the
    # extractor's internal layout diverges from the checkpoint's -- that must fail loudly instead
    # of being silently dropped (SPEC SS4.6 strict-load condition; P2 confirmed empirically that
    # unexpected keys are not limited to fc./head. -- pruned trailing stages are unexpected too).
    present_top_level = {key.split(".", 1)[0] for key in extractor.state_dict()}
    _, missing, unexpected = load_local_weights(extractor, weights_path, strict=False)
    if missing:
        raise LocalAssetError(
            f"STFPM teacher weights at {weights_path} are missing required keys: {list(missing)}"
        )
    disallowed = [key for key in unexpected if key.split(".", 1)[0] in present_top_level]
    if disallowed:
        raise LocalAssetError(
            f"STFPM teacher weights at {weights_path} have unexpected keys for submodules the "
            f"extractor does have, indicating a structural mismatch: {disallowed}"
        )
    return model
