from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.reverse_distillation.loss import ReverseDistillationLoss
from .base import AnomalyAdapter


@ADAPTERS.register("reverse_distillation")
class ReverseDistillationAdapter(AnomalyAdapter):
    """Reverse Distillation model adapter.

    No lifecycle hooks are needed. The encoder is frozen in the factory and
    ``torch_model.forward`` calls ``self.encoder.eval()`` on every pass, so the
    engine's per-epoch ``model.train()`` cannot put it back into training mode.
    There is no memory bank and no post-training calibration, so only the training
    loss differs from the common anomaly adapter.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.rd_loss = ReverseDistillationLoss()

    def train_step(self, model, batch, device):
        """Cosine-similarity loss between encoder and decoder feature pyramids.

        In training mode the model returns ``(encoder_features, decoder_features)``,
        which anomalib feeds straight into the loss as ``self.loss(*self.model(...))``.
        """
        images = batch[0].to(device)
        encoder_features, decoder_features = model(images)
        loss = self.rd_loss(encoder_features, decoder_features)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
