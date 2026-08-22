import torch
import torchvision.transforms.v2.functional as F


def smooth_anomaly_map(anomaly_map, sigma):
    """Gaussian smoothing applied to anomaly map."""
    if sigma is None or sigma <= 0:
        return anomaly_map
    kernel_size = max(3, int(2 * round(3 * sigma) + 1))
    smoothed = F.gaussian_blur(anomaly_map.unsqueeze(1), kernel_size=kernel_size, sigma=sigma)
    return smoothed.squeeze(1)


def to_output_dict(outputs):
    """Normalize model output to standard dict format {"pred_score": (B,), "anomaly_map": (B, H, W)}."""
    if hasattr(outputs, "_asdict"):
        outputs = outputs._asdict()
    if outputs.get("pred_score") is not None and outputs["pred_score"].ndim == 2:
        outputs["pred_score"] = outputs["pred_score"].squeeze(1)
    if outputs.get("anomaly_map") is not None and outputs["anomaly_map"].ndim == 4:
        outputs["anomaly_map"] = outputs["anomaly_map"].squeeze(1)
    return outputs


def best_f1_threshold(scores, labels):
    """Compute optimal threshold maximizing F1 score."""
    if scores.numel() == 0 or labels.sum() == 0:
        return 0.0

    sorted_scores, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order].float()

    tp_cum = torch.cumsum(sorted_labels, dim=0)
    predicted_count = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32, device=scores.device)
    precision = tp_cum / predicted_count
    total_positive = sorted_labels.sum()
    recall = tp_cum / total_positive
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    is_last_in_run = torch.ones_like(sorted_scores, dtype=torch.bool)
    is_last_in_run[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    f1 = torch.where(is_last_in_run, f1, torch.full_like(f1, float("-inf")))

    best_index = int(torch.argmax(f1))
    return float(sorted_scores[best_index])


def compute_thresholds(model, valid_loader, device, smooth_sigma):
    """Compute valid-only image and pixel thresholds."""
    model.eval()
    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            outputs = to_output_dict(model(images))
            maps = smooth_anomaly_map(outputs["anomaly_map"], smooth_sigma)
            labels = torch.stack([t["label"] for t in targets]).to(device)
            masks = torch.stack([t["mask"] for t in targets]).to(device)

            image_scores.append(outputs["pred_score"].detach().cpu())
            image_labels.append(labels.detach().cpu())
            pixel_scores.append(maps.flatten().detach().cpu())
            pixel_labels.append(masks.flatten().detach().cpu())

    image_threshold = best_f1_threshold(torch.cat(image_scores), torch.cat(image_labels))
    pixel_threshold = best_f1_threshold(torch.cat(pixel_scores), torch.cat(pixel_labels))
    return image_threshold, pixel_threshold
