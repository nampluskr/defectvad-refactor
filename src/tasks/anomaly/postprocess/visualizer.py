import os
import numpy as np
import torch
from PIL import Image


def normalize_anomaly_map(anomaly_map, min_val=None, max_val=None):
    """Normalize anomaly map to [0, 1] range."""
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    if min_val is None:
        min_val = float(anomaly_map.min())
    if max_val is None:
        max_val = float(anomaly_map.max())

    if max_val - min_val > 1e-6:
        norm_map = (anomaly_map - min_val) / (max_val - min_val)
    else:
        norm_map = np.zeros_like(anomaly_map)
    return np.clip(norm_map, 0.0, 1.0)


def anomaly_map_to_heatmap(anomaly_map):
    """Convert normalized anomaly map (H, W) to RGB heatmap (H, W, 3)."""
    norm_map = normalize_anomaly_map(anomaly_map)
    
    # Fast vectorized pseudo-jet colormap without matplotlib dependency
    # 0.0: Blue (0, 0, 255) -> Cyan -> Green -> Yellow -> Red (255, 0, 0) : 1.0
    r = np.clip(1.5 - np.abs(norm_map * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(norm_map * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(norm_map * 4.0 - 1.0), 0.0, 1.0)

    heatmap = np.stack([r, g, b], axis=-1) * 255.0
    return heatmap.astype(np.uint8)


def overlay_heatmap(image_np, heatmap_np, alpha=0.4):
    """Overlay heatmap onto original RGB image."""
    if image_np.shape[:2] != heatmap_np.shape[:2]:
        h, w = heatmap_np.shape[:2]
        img_pil = Image.fromarray(image_np).resize((w, h), Image.Resampling.BILINEAR)
        image_np = np.array(img_pil)

    overlay = (image_np * (1.0 - alpha) + heatmap_np * alpha).astype(np.uint8)
    return overlay


def save_prediction_visualization(image_path, anomaly_map, output_dir, stem, threshold=None):
    """Save side-by-side visualization of Original, Heatmap, and Overlay."""
    os.makedirs(output_dir, exist_ok=True)

    # Load original image
    orig_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_img)

    # Generate heatmap
    heatmap_np = anomaly_map_to_heatmap(anomaly_map)
    
    # Resize heatmap to match original image dimensions
    heatmap_pil = Image.fromarray(heatmap_np).resize(orig_img.size, Image.Resampling.BILINEAR)
    heatmap_resized_np = np.array(heatmap_pil)

    # Generate overlay
    overlay_np = overlay_heatmap(orig_np, heatmap_resized_np, alpha=0.4)

    # Combine side by side: [Original | Heatmap | Overlay]
    w, h = orig_img.size
    combined = Image.new("RGB", (w * 3, h))
    combined.paste(orig_img, (0, 0))
    combined.paste(heatmap_pil, (w, 0))
    combined.paste(Image.fromarray(overlay_np), (w * 2, 0))

    save_path = os.path.join(output_dir, f"{stem}_vis.png")
    combined.save(save_path)
    return save_path
