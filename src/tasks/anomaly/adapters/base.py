import torch

from src.core.adapter import TaskAdapter
from src.core.registry import ADAPTERS
from src.tasks.anomaly.postprocess.smoother import (
    compute_thresholds,
    smooth_anomaly_map,
    to_output_dict,
)


def anomaly_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


@ADAPTERS.register("anomaly")
class AnomalyAdapter(TaskAdapter):
    """Base Anomaly task adapter."""

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, **params)
        self.smooth_sigma = smooth_sigma
        self.image_threshold = None
        self.pixel_threshold = None
        self._last_maps = None

    def collate_fn(self):
        return anomaly_collate

    def batch_size(self, batch):
        return batch[0].shape[0]

    def _smooth(self, anomaly_map):
        return smooth_anomaly_map(anomaly_map, self.smooth_sigma)

    def train_step(self, model, batch, device):
        raise NotImplementedError("Subclasses must override train_step to call specific model loss.")

    def eval_step(self, model, batch, device):
        images = batch[0].to(device)
        targets = batch[1]
        outputs = to_output_dict(model(images))
        labels = torch.stack([t["label"] for t in targets]).to(device)
        masks = torch.stack([t["mask"] for t in targets]).to(device)
        maps = self._smooth(outputs["anomaly_map"])
        return {
            "loss": None,
            "outputs": {
                "scores": outputs["pred_score"],
                "maps": maps,
                "labels": labels,
                "masks": masks,
            },
        }

    def update_metrics(self, outputs):
        o = outputs["outputs"]
        if "image_auroc" in self.metrics:
            self.metrics["image_auroc"].update(o["scores"], o["labels"])
        if "pixel_auroc" in self.metrics:
            self.metrics["pixel_auroc"].update(o["maps"].flatten(), o["masks"].flatten().long())

    def compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def predict_step(self, model, batch, device):
        images = batch[0].to(device)
        raw_ids = batch[1] if len(batch) > 1 else None
        outputs = to_output_dict(model(images))
        maps = self._smooth(outputs["anomaly_map"])
        self._last_maps = maps.detach().cpu()

        predictions = []
        for i in range(images.shape[0]):
            score = float(outputs["pred_score"][i])
            is_anomalous = (
                bool(score >= self.image_threshold) if self.image_threshold is not None else None
            )
            stem = (
                raw_ids[i]
                if isinstance(raw_ids, (list, tuple)) and isinstance(raw_ids[i], str)
                else f"sample_{i:04d}"
            )
            predictions.append({
                "stem": stem,
                "anomaly_score": score,
                "is_anomalous": is_anomalous,
                "image_threshold": self.image_threshold,
            })
        return predictions

    def on_fit_start(self, model, loaders, device):
        if hasattr(model, "on_fit_start"):
            model.on_fit_start(loaders["train"], device)

    def on_fit_end(self, model, loaders, device):
        if hasattr(model, "on_fit_end"):
            model.on_fit_end(loaders["valid"], device)
        if "valid" in loaders:
            self.image_threshold, self.pixel_threshold = compute_thresholds(
                model, loaders["valid"], device, self.smooth_sigma
            )

    def extra_final_metrics(self):
        return {
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
        }
