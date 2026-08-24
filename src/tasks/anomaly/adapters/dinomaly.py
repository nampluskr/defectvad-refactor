import itertools

import torch

from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.dinomaly.components import StableAdamW, WarmCosineScheduler
from .base import AnomalyAdapter


@ADAPTERS.register("dinomaly")
class DinomalyAdapter(AnomalyAdapter):
    """Dinomaly model adapter.

    Upstream's ``WarmCosineScheduler`` steps once per optimizer step (per batch)
    and needs the total step budget (epochs * steps/epoch) up front to build its
    cosine schedule array -- the common engine's ``scheduler.step()`` runs once
    per epoch, which cannot reproduce that. Like ``CflowAdapter``, this adapter
    owns a private ``StableAdamW`` + ``WarmCosineScheduler`` pair over the
    unfrozen bottleneck/decoder parameters and does the real zero_grad/backward/
    step/step itself inside ``train_step``, returning a zero dummy loss (tied
    into the same parameter graph via ``0.0 * parameter``) so the engine's own
    optimizer -- built separately by ``build_optimizer`` over the same trainable
    set -- always sees an exact-zero gradient and its step is a true no-op.

    Known limitation (matches CFLOW): ``--resume`` does not preserve this
    adapter's private optimizer/scheduler state, since only the inert engine
    optimizer is checkpointed.
    """

    def __init__(
        self,
        loss_fn,
        metrics,
        smooth_sigma=0,
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-8,
        base_value=2e-3,
        final_value=2e-4,
        warmup_iters=100,
        **params,
    ):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.optimizer_params = {
            "lr": lr,
            "betas": tuple(betas),
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "eps": eps,
        }
        self.scheduler_params = {
            "base_value": base_value,
            "final_value": final_value,
            "warmup_iters": warmup_iters,
        }
        self.private_optimizer = None
        self.private_scheduler = None
        self.step_count = 0

    def on_fit_start(self, model, loaders, device):
        super().on_fit_start(model, loaders, device)
        trainable_params = itertools.chain(model.bottleneck.parameters(), model.decoder.parameters())
        self.private_optimizer = StableAdamW([{"params": trainable_params}], **self.optimizer_params)
        total_steps = self.total_epochs * len(loaders["train"])
        self.private_scheduler = WarmCosineScheduler(
            self.private_optimizer, total_iters=total_steps, **self.scheduler_params
        )
        self.step_count = 0

    def train_step(self, model, batch, device):
        images = batch[0].to(device)

        self.private_optimizer.zero_grad()
        loss = model(images, global_step=self.step_count)
        loss.backward()
        self.private_optimizer.step()
        self.private_scheduler.step()
        self.step_count += 1

        # Leave no leftover gradient from the private step: the engine's own
        # backward (on the zero dummy loss below) accumulates into .grad rather
        # than replacing it, so a stale nonzero .grad here would leak into the
        # engine's optimizer step (same fix as CflowAdapter).
        self.private_optimizer.zero_grad()

        dummy_loss = sum(
            (0.0 * parameter.sum() for parameter in itertools.chain(model.bottleneck.parameters(), model.decoder.parameters())),
            torch.zeros([], device=device),
        )
        return {"loss": dummy_loss, "loss_dict": {"loss": float(loss.detach())}}
