import torch

from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("dfm")
class DfmAdapter(AnomalyAdapter):
    """DFM model adapter.

    DFM has no loss and no gradient training. The training pass only collects
    pooled backbone features into the model's memory bank; a PCA transform (and,
    for ``score_type="nll"``, a Gaussian model) is then fitted from them once
    before the first validation.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self._fitted = False

    def train_step(self, model, batch, device):
        """Collect features. Returns a dummy loss to satisfy the engine contract.

        The engine always calls ``loss.backward()``, so the returned tensor must
        carry grad. It is a detached leaf, so backward produces no gradients for
        any parameter -- matching anomalib's own ``training_step``.
        """
        images = batch[0].to(device)
        model(images)
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        return {"loss": loss, "loss_dict": {"loss": 0.0}}

    def on_validation_start(self, model, loaders, device):
        """Fit PCA (and Gaussian, for ``nll``) to the collected features.

        This mirrors anomalib's ``MemoryBankMixin.on_validation_start``. It must
        run before any evaluation, otherwise scoring reads an unfitted PCA. Runs
        once: ``DFMModel.fit`` clears the memory bank afterwards.

        The PCA fit uses SVD, which consumes no randomness, so no ``fork_rng``
        guard is needed here.
        """
        super().on_validation_start(model, loaders, device)
        if self._fitted:
            # The configured budget is a single epoch, but guard against extra
            # epochs re-collecting features that would never be used.
            model.memory_bank = []
            return
        model.fit()
        self._fitted = True
