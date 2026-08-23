import torch

from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.supersimplenet.loss import SSNLoss
from .base import AnomalyAdapter


@ADAPTERS.register("supersimplenet")
class SupersimplenetAdapter(AnomalyAdapter):
    """SuperSimpleNet model adapter."""

    def __init__(self, loss_fn, metrics, truncation_term=0.5, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.ssn_loss = SSNLoss(truncation_term=truncation_term)

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        targets = batch[1]

        masks = (
            torch.stack([t["mask"] for t in targets]).to(device)
            if targets and "mask" in targets[0]
            else None
        )
        labels = (
            torch.stack([t["label"] for t in targets]).to(device).float()
            if targets and "label" in targets[0]
            else torch.zeros(images.shape[0], dtype=torch.float32, device=device)
        )

        anomaly_map, anomaly_score, target_masks, target_labels = model(
            images=images, masks=masks, labels=labels
        )
        loss = self.ssn_loss(
            pred_map=anomaly_map,
            pred_score=anomaly_score,
            target_mask=target_masks,
            target_label=target_labels,
        )
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}

    def configure_optimizers(self, model, config_optim):
        """Configure AdamW with split LR: adaptor (0.0001) and segdec (0.0002) as upstream."""
        spec = config_optim.get("optimizer", {})
        base_lr = spec.get("params", {}).get("lr", 0.0002)
        weight_decay = spec.get("params", {}).get("weight_decay", 0.00001)
        adaptor_lr = spec.get("params", {}).get("adaptor_lr", base_lr * 0.5)
        segdec_lr = spec.get("params", {}).get("segdec_lr", base_lr)
        return torch.optim.AdamW(
            [
                {"params": model.adaptor.parameters(), "lr": adaptor_lr},
                {"params": model.segdec.parameters(), "lr": segdec_lr, "weight_decay": weight_decay},
            ]
        )

