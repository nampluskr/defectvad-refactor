from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.uflow.loss import UFlowLoss
from .base import AnomalyAdapter


@ADAPTERS.register("uflow")
class UflowAdapter(AnomalyAdapter):
    """U-Flow model adapter."""

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.uflow_loss = UFlowLoss()

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        z, ljd = model(images)
        loss = self.uflow_loss(z, ljd)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
