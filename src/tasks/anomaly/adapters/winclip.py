import torch

from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("winclip")
class WinclipAdapter(AnomalyAdapter):
    """WinCLIP model adapter.

    WinCLIP is zero-/few-shot: it has no loss and no gradient training. Training-time
    setup is limited to collecting the CLIP text embeddings for the prompt ensemble (and,
    for ``k_shot > 0``, visual embeddings from a handful of normal reference images) once
    before the first training epoch -- mirroring anomalib's ``WinClip.setup`` Lightning hook.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, class_name=None, k_shot=0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.class_name = class_name
        self.k_shot = k_shot
        self._is_setup = False

    def train_step(self, model, batch, device):
        """No training happens. Returns a dummy loss to satisfy the engine contract.

        The engine always calls ``loss.backward()``, so the returned tensor must carry
        grad. It is a detached leaf, so backward produces no gradients for any parameter.
        Unlike PatchCore/DFM, the model is not called here: ``WinClipModel.forward``
        requires text embeddings to already be collected (via ``setup``, which only runs
        in ``on_fit_start``), and calling it earlier would raise.
        """
        del model, batch
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        return {"loss": loss, "loss_dict": {"loss": 0.0}}

    def on_fit_start(self, model, loaders, device):
        """Collect text (and, for few-shot, visual) embeddings before the first training epoch.

        Mirrors upstream's ``WinClip.setup()``, a Lightning hook that runs before ``fit()``
        starts. Doing this before any dataloader iteration (rather than at the first
        validation) matters for few-shot: ``loaders["train"]`` is shuffled, so collecting
        reference images only after a full "training" epoch has already drawn from it would
        pick a different -- and non-reproducible relative to upstream -- sample.
        """
        super().on_fit_start(model, loaders, device)
        if self._is_setup:
            return
        class_name = self.class_name if self.class_name is not None else self._infer_class_name(loaders)
        ref_images = self._collect_reference_images(loaders["train"], device) if self.k_shot else None
        model.setup(class_name, ref_images)
        self._is_setup = True

    def _infer_class_name(self, loaders):
        for split in ("valid", "train", "test"):
            dataset = getattr(loaders.get(split), "dataset", None)
            category = getattr(dataset, "category", None)
            if category:
                return category
        return "object"

    def _collect_reference_images(self, train_loader, device):
        """Gather the first ``k_shot`` normal images from the training set.

        Mirrors anomalib's ``WinClip.collect_reference_images``.
        """
        ref_images = torch.empty(0)
        for batch in train_loader:
            images = batch[0][: self.k_shot - ref_images.shape[0]]
            ref_images = torch.cat((ref_images, images))
            if ref_images.shape[0] == self.k_shot:
                break
        return ref_images.to(device)
