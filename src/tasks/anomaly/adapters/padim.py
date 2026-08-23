import torch

from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("padim")
class PadimAdapter(AnomalyAdapter):
    """PaDiM model adapter.

    PaDiM has no loss and no gradient training. The training pass only collects
    patch embeddings into the model's memory bank; a per-position multivariate
    Gaussian is then fitted from them once before the first validation.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self._gaussian_fitted = False

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
        """Fit the Gaussian to the collected embeddings before the first validation.

        This mirrors anomalib's ``MemoryBankMixin.on_validation_start``. It must run
        before any evaluation, otherwise the anomaly map is computed from an unfitted
        distribution. Runs once: ``PadimModel.fit`` clears the memory bank afterwards.

        Neither the fit nor the Mahalanobis scoring consumes randomness, so no
        ``fork_rng`` guard is needed here. The one random draw in PaDiM is the
        feature-index subsample, taken at construction time and stored in the ``idx``
        buffer, so it is fixed by the run seed and travels with the checkpoint.
        """
        super().on_validation_start(model, loaders, device)
        if self._gaussian_fitted:
            # The configured budget is a single epoch, but guard against extra epochs
            # re-collecting embeddings that would never be used: drop them so the
            # memory bank cannot grow without bound.
            model.memory_bank = []
            return
        model.fit()
        self._gaussian_fitted = True
