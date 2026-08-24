import torch

from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("uninet")
class UniNetAdapter(AnomalyAdapter):
    """UniNet model adapter.

    ``source_teacher`` is frozen and eval-locked in the factory, so no lifecycle
    hooks are needed beyond the training loss and the split-LR optimizer.
    """

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        targets = batch[1]

        # MVTec train split yields empty targets ({}) -- every sample is normal,
        # so an all-zero per-pixel mask and a scalar label=0 select the same
        # (entire) set of feature tokens in UniNetLoss, giving an identical loss.
        # We take the cheaper label-only path here.
        if targets and "mask" in targets[0]:
            masks = torch.stack([t["mask"] for t in targets]).to(device)
            labels = torch.stack([t["label"] for t in targets]).to(device)
        else:
            masks = None
            labels = torch.zeros(images.shape[0], dtype=torch.long, device=device)

        loss = model(images=images, masks=masks, labels=labels)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}

    def configure_optimizers(self, model, config_optim):
        """AdamW with split LR: student/bottleneck/dfs at base lr, target_teacher at a
        much lower lr, matching upstream ``configure_optimizers``. ``source_teacher``
        is excluded -- it is frozen in the factory.
        """
        spec = config_optim.get("optimizer", {})
        params = spec.get("params", {})
        base_lr = params.get("lr", 5e-3)
        teacher_lr = params.get("teacher_lr", 1e-6)
        return torch.optim.AdamW(
            [
                {"params": model.student.parameters()},
                {"params": model.bottleneck.parameters()},
                {"params": model.dfs.parameters()},
                {"params": model.teachers.target_teacher.parameters(), "lr": teacher_lr},
            ],
            lr=base_lr,
            betas=tuple(params.get("betas", (0.9, 0.999))),
            weight_decay=params.get("weight_decay", 1e-5),
            eps=params.get("eps", 1e-10),
            amsgrad=params.get("amsgrad", True),
        )
