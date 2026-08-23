import torch

from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("patchcore")
class PatchcoreAdapter(AnomalyAdapter):
    """PatchCore model adapter.

    PatchCore has no loss and no gradient training. The training pass only collects
    patch embeddings into the model's embedding store; the memory bank is then built
    once by coreset subsampling before the first validation.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, coreset_sampling_ratio=0.1, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self._memory_bank_fitted = False

    def train_step(self, model, batch, device):
        """Collect embeddings. Returns a dummy loss to satisfy the engine contract.

        The engine always calls ``loss.backward()``, so the returned tensor must
        carry grad. It is a detached leaf, so backward produces no gradients for
        any parameter -- matching anomalib's own ``training_step``.
        """
        images = batch[0].to(device)
        model(images)
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        return {"loss": loss, "loss_dict": {"loss": 0.0}}

    def on_validation_start(self, model, loaders, device):
        """Build the memory bank by coreset subsampling before the first validation.

        This mirrors anomalib's ``MemoryBankMixin.on_validation_start``. It must run
        before any evaluation, otherwise the model raises on an empty memory bank.
        Runs once: the embedding store is cleared by ``subsample_embedding``.

        KCenterGreedy and SparseRandomProjection consume randomness, so the draw is
        isolated with ``fork_rng`` to keep the training RNG stream reproducible.
        """
        super().on_validation_start(model, loaders, device)
        if self._memory_bank_fitted:
            return
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            model.subsample_embedding(self.coreset_sampling_ratio)
        self._memory_bank_fitted = True
