from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.fastflow.loss import FastflowLoss
from .base import AnomalyAdapter


@ADAPTERS.register("fastflow")
class FastflowAdapter(AnomalyAdapter):
    """FastFlow model adapter.

    FastFlow needs no lifecycle hooks: the backbone is frozen and forced to eval()
    inside the model's own forward, and there is no post-training calibration. Only
    the training loss differs from the common anomaly adapter.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.fastflow_loss = FastflowLoss()

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        hidden_variables, jacobians = model(images)
        loss = self.fastflow_loss(hidden_variables, jacobians)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
