from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.csflow.loss import CsFlowLoss
from .base import AnomalyAdapter


@ADAPTERS.register("csflow")
class CsflowAdapter(AnomalyAdapter):
    """CS-Flow model adapter."""

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.csflow_loss = CsFlowLoss()

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        z_dist, jacobians = model(images)
        loss = self.csflow_loss(z_dist, jacobians)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
