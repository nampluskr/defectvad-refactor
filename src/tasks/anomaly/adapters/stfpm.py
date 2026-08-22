from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.stfpm.loss import STFPMLoss
from .base import AnomalyAdapter


@ADAPTERS.register("stfpm")
class StfpmAdapter(AnomalyAdapter):
    """STFPM model adapter."""

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.stfpm_loss = STFPMLoss()

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        teacher_features, student_features = model(images)
        loss = self.stfpm_loss(teacher_features, student_features)
        batch_size = images.shape[0]
        if batch_size > 0:
            loss = loss / batch_size
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
