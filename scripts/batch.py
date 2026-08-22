import argparse
import os
import sys
import warnings

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.offline import enable_offline_guard
enable_offline_guard()

warnings.filterwarnings("ignore", module="torchmetrics.*")

from src.batch.parser import expand_batch_config
from src.batch.runner import BatchRunner
from src.batch.summary import render_summary_table, save_summary_csv, save_summary_json
from src.core.logger import setup_logger

# Pre-load tasks for registry population
import src.tasks.anomaly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CV Multi-Condition Batch Execution Entry Point",
        allow_abbrev=False,
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to batch manifest YAML")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict", "all"],
        help="Batch execution mode (train, evaluate, predict, all; default: train)",
    )
    parser.add_argument("--only", type=str, default=None, help="Filter cases by name or pattern (comma-separated)")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Override batch output root directory")
    parser.add_argument("--device", type=str, default=None, help="Override execution device (e.g. cuda, cpu)")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Override training epochs for all cases")
    parser.add_argument("--batch_size", "-b", type=int, default=None, help="Override batch size for all cases")
    parser.add_argument("--split", "-s", type=str, default=None, help="Override evaluation split (default: test)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completed case outputs")
    parser.add_argument("--print_cases", action="store_true", help="Print expanded cases list (dry-run) and exit")
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        help="Dotted key override applied to all cases (KEY=VALUE)",
    )

    args, unknown = parser.parse_known_args()

    for item in unknown:
        if item.startswith("--set="):
            args.overrides.append(item[6:])
        else:
            raise ValueError(f"Unrecognized argument: {item}")

    return args


def main():
    args = parse_args()

    cli_overrides = list(args.overrides)
    if args.epochs is not None:
        cli_overrides.append(f"train.epochs={args.epochs}")
    if args.batch_size is not None:
        cli_overrides.append(f"data.batch_size={args.batch_size}")
    if args.device is not None:
        cli_overrides.append(f"runtime.device={args.device}")

    batch_config = expand_batch_config(
        config_or_path=args.config,
        only=args.only,
        output_dir_override=args.output_dir,
        cli_overrides=cli_overrides,
    )

    if args.split:
        for case in batch_config.cases:
            case.split = args.split

    # Dry-run: print cases table
    if args.print_cases:
        print(f"\n[Batch Manifest Cases: {batch_config.name}] (Total: {len(batch_config.cases)} cases)")
        print(f"Output Root: {batch_config.output_root}")
        print(f"Base Data:   {batch_config.base_data}")
        print(f"Base Model:  {batch_config.base_model}\n")
        print(f"{'No.':<4} {'Case ID':<25} {'Run Name':<35} {'Data Selectors':<25} {'Model Selectors':<20}")
        print("-" * 115)
        for idx, case in enumerate(batch_config.cases, 1):
            d_str = ", ".join(f"{k}={v}" for k, v in case.data_selectors.items())
            m_str = ", ".join(f"{k}={v}" for k, v in case.model_selectors.items())
            print(f"{idx:<4} {case.case_id:<25} {case.run_name:<35} {d_str:<25} {m_str:<20}")
        print("-" * 115)
        return

    if not batch_config.cases:
        print(f"Warning: No cases matched batch specification or filter (only='{args.only}').")
        return

    os.makedirs(batch_config.output_root, exist_ok=True)
    logger = setup_logger(batch_config.output_root, log_file=f"batch_{args.mode}.log")

    runner = BatchRunner(
        batch_config=batch_config,
        mode=args.mode,
        overwrite=args.overwrite,
        logger=logger,
    )

    summary = runner.run()

    # Save summary artifacts
    summary_json_path = os.path.join(batch_config.output_root, "summary.json")
    summary_csv_path = os.path.join(batch_config.output_root, "summary.csv")
    save_summary_json(summary, summary_json_path)
    save_summary_csv(summary, summary_csv_path)

    # Render summary table to console and log
    table_str = render_summary_table(summary)
    print("\n" + table_str + "\n")
    logger.info(f"\n{table_str}")
    logger.info(f"Saved summary JSON to {summary_json_path}")
    logger.info(f"Saved summary CSV to {summary_csv_path}")

    if summary.failed_cases > 0:
        logger.warning(f"Batch completed with {summary.failed_cases} failures out of {summary.total_cases} cases.")


if __name__ == "__main__":
    main()
