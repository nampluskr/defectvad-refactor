import torch

from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("anomaly_dino")
class AnomalyDinoAdapter(AnomalyAdapter):
    """AnomalyDINO model adapter.

    Same shape as ``PatchcoreAdapter``: no loss, no gradient training. The
    training pass only collects normalized DINOv2 patch embeddings into the
    model's embedding store (under ``torch.inference_mode()`` inside
    ``extract_features``, so nothing here carries a real gradient); the memory
    bank is finalized once by ``model.fit()`` -- which also runs the optional
    coreset subsampling baked into the model's own constructor flags -- before
    the first validation.
    """

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
        """Finalize the memory bank via ``model.fit()`` before the first validation.

        Must run before any evaluation, otherwise the model raises on an empty
        memory bank. Runs once: ``fit()`` clears the embedding store.

        Coreset selection (when enabled) consumes randomness, so the draw is
        isolated with ``fork_rng`` to keep the training RNG stream reproducible.
        """
        super().on_validation_start(model, loaders, device)
        if model.memory_bank.numel() > 0:
            return
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            model.fit()
