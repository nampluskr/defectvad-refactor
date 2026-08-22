import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.offline import enable_offline_guard
enable_offline_guard()

import warnings
warnings.filterwarnings("ignore", module="torchmetrics.*")

import yaml
from src.core.builders import (
    build_adapter,
    build_dataloader,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_transforms,
)
from src.core.checkpoint import load_checkpoint
from src.core.config import resolve_config, validate_config
from src.core.context import RunContext
from src.core.engine import Engine
from src.core.logger import MetricsCsvWriter, setup_logger
from src.utils.io import save_yaml

# Pre-load tasks for registry population
import src.tasks.anomaly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CV Training Entry Point",
        allow_abbrev=False,
    )
    parser.add_argument("--data", "-d", type=str, help="Path to dataset config YAML")
    parser.add_argument("--model", "-m", type=str, help="Path to model config YAML")
    parser.add_argument("--config", "-c", type=str, help="Path to self-contained config YAML")
    parser.add_argument("--epochs", "-e", type=int, help="Override training epochs")
    parser.add_argument("--batch_size", "-b", type=int, help="Override batch size")
    parser.add_argument("--output_dir", "-o", type=str, help="Override output directory")
    parser.add_argument("--run_name", type=str, help="Override run name")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, help="Override execution device")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume")
    parser.add_argument("--print_config", action="store_true", help="Print resolved config and exit")
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

    if not args.data and not args.model and not args.config:
        print("Error: At least one of --data, --model, or --config must be provided.")
        sys.exit(1)

    cli_overrides = list(args.overrides)
    if args.epochs is not None:
        cli_overrides.append(f"train.epochs={args.epochs}")
    if args.batch_size is not None:
        cli_overrides.append(f"data.batch_size={args.batch_size}")
    if args.output_dir is not None:
        cli_overrides.append(f"output.root={args.output_dir}")
    if args.run_name is not None:
        cli_overrides.append(f"output.run_name={args.run_name}")
    if args.seed is not None:
        cli_overrides.append(f"runtime.seed={args.seed}")
    if args.device is not None:
        cli_overrides.append(f"runtime.device={args.device}")

    config = resolve_config(
        data_path=args.data,
        model_path=args.model,
        config_path=args.config,
        data_selectors=data_selectors,
        model_selectors=model_selectors,
        cli_overrides=cli_overrides,
    )

    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return

    validate_config(config)

    # Determine run output directory
    task_name = config.get("meta", {}).get("task_name", "cv_task")
    run_name = config.get("output", {}).get("run_name") or f"{task_name}_{config.get('model', {}).get('name', 'model')}"
    output_root = config.get("output", {}).get("root", "outputs")
    run_dir = os.path.join(output_root, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save resolved config
    save_yaml(config, os.path.join(run_dir, "config.resolved.yaml"))

    # Setup logger and context
    logger = setup_logger(run_dir)
    metric_names = [m["name"] for m in config.get("metrics", [])]
    metrics_writer = MetricsCsvWriter(run_dir, metric_names)
    ctx = RunContext(config, run_dir=run_dir)
    ctx.setup_seed()
    ctx.start(" ".join(sys.argv))

    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Run directory: {run_dir}")

    # Build components
    transforms = build_transforms(config["data"])
    train_dataset = build_dataset(config["data"], split="train", transform=transforms["train"])
    valid_dataset = build_dataset(config["data"], split="valid", transform=transforms["eval"])

    loss_fn = build_loss(config["loss"])
    metrics = build_metrics(config.get("metrics", []))
    adapter = build_adapter(config["adapter"], loss_fn=loss_fn, metrics=metrics)

    train_loader = build_dataloader(
        train_dataset, config["data"], split="train", adapter=adapter, seed=ctx.seed, device=config["runtime"]["device"]
    )
    valid_loader = build_dataloader(
        valid_dataset, config["data"], split="valid", adapter=adapter, seed=ctx.seed, device=config["runtime"]["device"]
    )

    model = build_model(config["model"])
    optimizer = build_optimizer(config["optim"], model)
    scheduler = build_scheduler(config["optim"], optimizer)

    start_epoch = 1
    best_metric = None
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, map_location=ctx.device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_metric = checkpoint.get("best_metric")

    engine = Engine(logger=logger, metrics_writer=metrics_writer, checkpoint_dir=checkpoint_dir)
    best_score = engine.fit(
        model=model,
        adapter=adapter,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        ctx=ctx,
        start_epoch=start_epoch,
        best_metric=best_metric,
    )

    ctx.finish()
    monitor_name = config.get("train", {}).get("monitor", {}).get("metric", "metric")
    logger.info(f"Training completed successfully. Best {monitor_name}: {best_score:.3f}")


if __name__ == "__main__":
    main()
