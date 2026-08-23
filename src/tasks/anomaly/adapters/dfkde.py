import os

import torch

from src.core.registry import ADAPTERS
from src.tasks.anomaly.postprocess.smoother import best_f1_threshold, to_output_dict
from .base import AnomalyAdapter


@ADAPTERS.register("dfkde")
class DfkdeAdapter(AnomalyAdapter):
    """DFKDE model adapter.

    DFKDE has no loss and no gradient training. The training pass only collects
    pooled backbone features into the model's memory bank; a PCA + Gaussian-KDE
    classifier is then fitted from them once before the first validation.

    Unlike every other model wired to ``AnomalyAdapter``, DFKDE produces an
    image-level anomaly score only -- ``DfkdeModel.forward`` never returns an
    anomaly map (see ``dfkde/torch_model.py``). The common ``eval_step``,
    ``predict_step`` and threshold calibration all assume a pixel-level map
    exists, so this adapter overrides them instead of forcing a fabricated map
    through the shared pixel-metric pipeline.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=0, **params):
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
        """Fit the PCA + Gaussian-KDE classifier to the collected features.

        This mirrors anomalib's ``MemoryBankMixin.on_validation_start``. It must
        run before any evaluation, otherwise scoring reads an unfitted KDE model.
        Runs once: ``DfkdeModel.fit`` clears the memory bank afterwards.

        ``KDEClassifier.fit`` draws a random subset via Python's ``random`` module
        (not ``torch``) only when the memory bank exceeds ``max_training_points``.
        That module has its own seeded stream independent of ``torch``'s, so no
        ``fork_rng`` guard is needed here.
        """
        super().on_validation_start(model, loaders, device)
        if self._fitted:
            # The configured budget is a single epoch, but guard against extra
            # epochs re-collecting features that would never be used.
            model.memory_bank = []
            return
        model.fit()
        self._fitted = True

    def eval_step(self, model, batch, device):
        images = batch[0].to(device)
        targets = batch[1]
        outputs = to_output_dict(model(images))
        labels = torch.stack([t["label"] for t in targets]).to(device)
        return {
            "loss": None,
            "outputs": {"scores": outputs["pred_score"], "labels": labels},
        }

    def update_metrics(self, outputs):
        o = outputs["outputs"]
        if "image_auroc" in self.metrics:
            self.metrics["image_auroc"].update(o["scores"], o["labels"])

    def predict_step(self, model, batch, device):
        images = batch[0].to(device)
        paths = batch[1] if len(batch) > 1 and isinstance(batch[1], (list, tuple)) else None
        stems = batch[2] if len(batch) > 2 and isinstance(batch[2], (list, tuple)) else None

        outputs = to_output_dict(model(images))

        predictions = []
        for i in range(images.shape[0]):
            score = float(outputs["pred_score"][i])
            is_anomalous = (
                bool(score >= self.image_threshold) if self.image_threshold is not None else None
            )
            if stems and i < len(stems) and isinstance(stems[i], str):
                stem = stems[i]
            elif paths and i < len(paths) and isinstance(paths[i], str):
                stem = os.path.splitext(os.path.basename(paths[i]))[0]
            else:
                stem = f"sample_{i:04d}"

            item = {
                "stem": stem,
                "anomaly_score": round(score, 5),
                "is_anomalous": is_anomalous,
                "threshold": self.image_threshold,
            }
            if paths and i < len(paths) and isinstance(paths[i], str):
                item["image_path"] = paths[i]
            predictions.append(item)
        return predictions

    def on_fit_end(self, model, loaders, device):
        """Calibrate the image threshold only -- there is no pixel-level map.

        Reimplements the image-score half of ``postprocess.smoother.compute_thresholds``
        directly instead of calling it, since that helper assumes an anomaly map.
        """
        if "valid" not in loaders:
            return
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for images, targets in loaders["valid"]:
                images = images.to(device)
                outputs = to_output_dict(model(images))
                batch_labels = torch.stack([t["label"] for t in targets]).to(device)
                scores.append(outputs["pred_score"].detach().cpu())
                labels.append(batch_labels.detach().cpu())
        self.image_threshold = best_f1_threshold(torch.cat(scores), torch.cat(labels))
        self.pixel_threshold = None
