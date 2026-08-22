import argparse
import os
import sys
import warnings

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.offline import enable_offline_guard
enable_offline_guard()

warnings.filterwarnings("ignore", module="torchmetrics.*")

import yaml
from src.core.builders import (
    build_adapter,
    build_dataloader,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_transforms,
)
from src.core.checkpoint import load_checkpoint
from src.core.config import resolve_config, validate_config
from src.core.context import RunContext
from src.core.engine import Trainer
from src.core.logger import format_result, setup_logger
from src.utils.io import save_json, save_yaml

# Pre-load tasks for registry population
import src.tasks.anomaly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CV Evaluation Entry Point",
        allow_abbrev=False,
    )
    parser.add_argument("--data", "-d", type=str, help="Path to dataset config YAML")
    parser.add_argument("--model", "-m", type=str, help="Path to model config YAML")
    parser.add_argument("--config", "-c", type=str, help="Path to self-contained config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", "-s", type=str, default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument("--batch-size", "-b", type=int, help="Override batch size")
    parser.add_argument("--output-dir", "-o", type=str, help="Override output directory")
    parser.add_argument("--run-name", type=str, help="Override run name")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, help="Override execution device")
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

    if not args.data and not args.model and not args.config:
        print("Error: At least one of --data, --model, or --config must be provided.")
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    cli_overrides = list(args.overrides)
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
    run_name = config.get("output", {}).get("run_name")

    if args.output_dir:
        output_dir = args.output_dir
    elif run_name:
        output_root = config.get("output", {}).get("root", "outputs")
        output_dir = os.path.join(output_root, run_name)
    else:
        # Default: use directory containing the checkpoint's parent or outputs
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        if os.path.basename(ckpt_dir) == "checkpoints":
            output_dir = os.path.dirname(ckpt_dir)
        else:
            output_dir = ckpt_dir

    os.makedirs(output_dir, exist_ok=True)

    # Setup logger and context
    logger = setup_logger(output_dir, log_file=f"evaluate_{args.split}.log")
    ctx = RunContext(config, run_dir=output_dir)
    ctx.setup_seed()
    ctx.start(" ".join(sys.argv))

    logger.info(f"Evaluating split: {args.split}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {output_dir}")

    # Build components
    transforms = build_transforms(config["data"])
    eval_transform = transforms.get("eval", transforms.get("test", transforms.get("train")))
    eval_dataset = build_dataset(config["data"], split=args.split, transform=eval_transform)

    loss_fn = build_loss(config["loss"])
    metrics = build_metrics(config.get("metrics", []))
    adapter = build_adapter(config["adapter"], loss_fn=loss_fn, metrics=metrics)

    eval_loader = build_dataloader(
        eval_dataset,
        config["data"],
        split=args.split,
        adapter=adapter,
        seed=ctx.seed,
        device=config["runtime"]["device"],
        allow_test_split=True,
    )


    model = build_model(config["model"])
    model.to(ctx.device)

    # Load checkpoint
    ckpt_data = load_checkpoint(args.checkpoint, model=model, map_location=ctx.device, restore_rng=False)
    if "adapter_state" in ckpt_data and ckpt_data["adapter_state"] is not None:
        if hasattr(adapter, "load_state_dict"):
            adapter.load_state_dict(ckpt_data["adapter_state"])
        elif isinstance(ckpt_data["adapter_state"], dict):
            for k, v in ckpt_data["adapter_state"].items():
                if hasattr(adapter, k):
                    setattr(adapter, k, v)

    trainer = Trainer(logger=logger)
    results, elapsed = trainer.evaluate(model, adapter, eval_loader, ctx, split=args.split)

    # Save metrics JSON
    metrics_file = os.path.join(output_dir, f"metrics_{args.split}.json")
    save_json(results, metrics_file)

    ctx.finish()
    formatted = format_result(results, sep=", ")
    logger.info(f"Evaluation completed in {elapsed:.2f}s: {formatted}")
    logger.info(f"Saved results to {metrics_file}")


if __name__ == "__main__":
    main()
