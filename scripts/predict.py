import argparse
import glob
import os
import sys
import time
import warnings

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.offline import enable_offline_guard
enable_offline_guard()

warnings.filterwarnings("ignore", module="torchmetrics.*")

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
import yaml

from src.core.builders import build_adapter, build_loss, build_metrics, build_model
from src.core.checkpoint import load_checkpoint
from src.core.config import resolve_config, validate_config
from src.core.context import RunContext
from src.core.logger import setup_logger
from src.tasks.anomaly.postprocess.visualizer import save_prediction_visualization
from src.utils.io import save_json

# Pre-load tasks for registry population
import src.tasks.anomaly


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class PredictImageDataset(Dataset):
    """Dataset for inference on raw images."""

    def __init__(self, image_paths, image_size=(256, 256)):
        self.image_paths = image_paths
        self.transform = T.Compose([
            T.ToImage(),
            T.Resize(image_size, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
        tensor = self.transform(img_rgb)
        stem = os.path.splitext(os.path.basename(path))[0]
        return tensor, path, stem


def collect_image_paths(input_target):
    """Collect image paths from file or directory."""
    if os.path.isfile(input_target):
        ext = os.path.splitext(input_target)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            return [os.path.abspath(input_target)]
        raise ValueError(f"Unsupported image file format: {input_target}")

    if os.path.isdir(input_target):
        found = []
        for root, _, files in os.walk(input_target):
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    found.append(os.path.abspath(os.path.join(root, f)))
        if not found:
            raise FileNotFoundError(f"No valid images found under directory: {input_target}")
        return found

    raise FileNotFoundError(f"Input path not found: {input_target}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CV Prediction / Inference Entry Point",
        allow_abbrev=False,
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument("--model", "-m", type=str, help="Path to model config YAML")
    parser.add_argument("--data", "-d", type=str, help="Path to dataset config YAML (optional)")
    parser.add_argument("--config", "-c", type=str, help="Path to self-contained config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for predictions and visualizations")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--threshold", type=float, default=None, help="Custom anomaly threshold override")
    parser.add_argument("--no-vis", action="store_true", help="Disable saving visualization images")
    parser.add_argument("--device", type=str, help="Override execution device")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    parser.add_argument("--set", action="append", dest="overrides", default=[], help="Dotted key override (KEY=VALUE)")

    args, unknown = parser.parse_known_args()

    data_selectors = {}
    model_selectors = {}

    idx = 0
    while idx < len(unknown):
        item = unknown[idx]
        if item.startswith("--data."):
            key = item[7:]
            if idx + 1 < len(unknown) and not unknown[idx + 1].startswith("--"):
                val = unknown[idx + 1]
                idx += 2
            else:
                val = True
                idx += 1
            data_selectors[key] = val
        elif item.startswith("--model."):
            key = item[8:]
            if idx + 1 < len(unknown) and not unknown[idx + 1].startswith("--"):
                val = unknown[idx + 1]
                idx += 2
            else:
                val = True
                idx += 1
            model_selectors[key] = val
        elif item.startswith("--set="):
            args.overrides.append(item[6:])
            idx += 1
        else:
            raise ValueError(f"Unrecognized argument: {item}")

    return args, data_selectors, model_selectors


def main():
    args, data_selectors, model_selectors = parse_args()

    if not args.model and not args.config and not args.data:
        print("Error: At least one of --model, --data, or --config must be provided.")
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    image_paths = collect_image_paths(args.input)

    cli_overrides = list(args.overrides)
    if args.batch_size is not None:
        cli_overrides.append(f"data.batch_size={args.batch_size}")
    if args.output_dir is not None:
        cli_overrides.append(f"output.root={args.output_dir}")
    if args.seed is not None:
        cli_overrides.append(f"runtime.seed={args.seed}")
    if args.device is not None:
        cli_overrides.append(f"runtime.device={args.device}")

    # Fallback to base anomaly data config if --data is not specified
    data_path = args.data
    if not data_path and not args.config:
        data_path = "configs/anomaly/data/mvtec.yaml"

    config = resolve_config(
        data_path=data_path,
        model_path=args.model,
        config_path=args.config,
        data_selectors=data_selectors,
        model_selectors=model_selectors,
        cli_overrides=cli_overrides,
    )

    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        if os.path.basename(ckpt_dir) == "checkpoints":
            output_dir = os.path.join(os.path.dirname(ckpt_dir), "predictions")
        else:
            output_dir = os.path.join(ckpt_dir, "predictions")

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    if not args.no_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # Setup logger and context
    logger = setup_logger(output_dir, log_file="predict.log")
    ctx = RunContext(config, run_dir=output_dir)
    ctx.setup_seed()
    ctx.start(" ".join(sys.argv))

    logger.info(f"Starting inference on {len(image_paths)} images")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {output_dir}")

    # Build dataset and dataloader
    image_size = config.get("data", {}).get("image_size", [256, 256])
    dataset = PredictImageDataset(image_paths, image_size=tuple(image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    # Build model, loss, adapter
    loss_fn = build_loss(config["loss"])
    metrics = build_metrics([])
    adapter = build_adapter(config["adapter"], loss_fn=loss_fn, metrics=metrics)

    model = build_model(config["model"])
    model.to(ctx.device)
    model.eval()

    # Load checkpoint
    ckpt_data = load_checkpoint(args.checkpoint, model=model, map_location=ctx.device, restore_rng=False)
    if "adapter_state" in ckpt_data and ckpt_data["adapter_state"] is not None:
        if hasattr(adapter, "load_state_dict"):
            adapter.load_state_dict(ckpt_data["adapter_state"])
        elif isinstance(ckpt_data["adapter_state"], dict):
            for k, v in ckpt_data["adapter_state"].items():
                if hasattr(adapter, k):
                    setattr(adapter, k, v)

    # Threshold determination
    threshold = args.threshold
    if threshold is None and getattr(adapter, "image_threshold", None) is not None:
        threshold = adapter.image_threshold
    adapter.image_threshold = threshold

    logger.info(f"Decision threshold: {threshold if threshold is not None else 'None (unspecified)'}")

    # Inference loop
    predictions = []
    start_time = time.perf_counter()
    anomalous_count = 0

    with torch.no_grad():
        for batch_images, batch_paths, batch_stems in loader:
            batch_images = batch_images.to(ctx.device)
            step_predictions = adapter.predict_step(model, (batch_images, batch_stems), ctx.device)
            last_maps = getattr(adapter, "_last_maps", None)

            for i, pred in enumerate(step_predictions):
                img_path = batch_paths[i]
                stem = batch_stems[i]
                score = pred["anomaly_score"]
                is_anom = pred["is_anomalous"]

                if is_anom:
                    anomalous_count += 1

                vis_path = None
                if not args.no_vis and last_maps is not None:
                    vis_path = save_prediction_visualization(
                        image_path=img_path,
                        anomaly_map=last_maps[i],
                        output_dir=vis_dir,
                        stem=stem,
                        threshold=threshold,
                    )

                record = {
                    "image_path": img_path,
                    "stem": stem,
                    "anomaly_score": round(score, 5),
                    "is_anomalous": is_anom,
                    "threshold": threshold,
                    "visualization": vis_path,
                }
                predictions.append(record)

    elapsed = time.perf_counter() - start_time
    predictions_file = os.path.join(output_dir, "predictions.json")
    save_json(predictions, predictions_file)

    ctx.finish()
    logger.info(f"Inference completed in {elapsed:.2f}s ({len(image_paths) / max(elapsed, 1e-4):.1f} img/s)")
    logger.info(f"Detected {anomalous_count}/{len(image_paths)} anomalous images")
    logger.info(f"Saved prediction records to {predictions_file}")
    if not args.no_vis:
        logger.info(f"Saved visualization images to {vis_dir}")


if __name__ == "__main__":
    main()
