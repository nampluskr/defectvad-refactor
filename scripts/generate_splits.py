"""Unified CLI entry point to generate dataset train/valid/test split JSON files."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import resolve_config
from src.core.errors import ConfigError, LocalAssetError
from src.core.registry import DATASETS
from src.utils.io import save_json
import src.tasks.anomaly  # Register anomaly datasets and components


def get_dataset_categories(dataset_cls, dataset_root):
    for attr in ("MVTEC_CATEGORIES", "BTAD_CATEGORIES", "VISA_CATEGORIES", "CATEGORIES"):
        if hasattr(dataset_cls, attr):
            return getattr(dataset_cls, attr)

    if os.path.isdir(dataset_root):
        return sorted([
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
        ])
    return []


def main():
    parser = argparse.ArgumentParser(description="Generate dataset train/valid/test split JSON files")
    parser.add_argument("-d", "--data", required=True, help="Path to dataset config YAML (e.g. configs/anomaly/data/mvtec.yaml)")
    parser.add_argument("--category", default="all", help="Category name or 'all'")
    parser.add_argument("--data.category", dest="data_category", default=None, help="Category selector override")
    parser.add_argument("--dataset_root", default=None, help="Override dataset root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--out_dir", default=None, help="Output directory for split JSONs (defaults to directory in config)")
    parser.add_argument("--local_config", default=None, help="Path to local.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="Config overrides (key=value)")
    args = parser.parse_args()

    category_arg = args.data_category or args.category

    # Resolve config to expand placeholders and local paths
    cli_overrides = list(args.overrides)
    if args.dataset_root:
        cli_overrides.append(f"data.root={args.dataset_root}")

    config = resolve_config(
        data_path=args.data,
        local_config_path=args.local_config,
        cli_overrides=cli_overrides,
    )

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("name")
    if not dataset_name:
        raise ConfigError(f"Missing 'data.name' in config {args.data}")

    dataset_cls = DATASETS.get(dataset_name)
    if not hasattr(dataset_cls, "generate_split"):
        raise ConfigError(f"Dataset class '{dataset_cls.__name__}' does not implement 'generate_split' classmethod.")

    dataset_root = data_cfg.get("root")
    if not dataset_root:
        raise ConfigError(f"Missing 'data.root' in config {args.data}")

    split_path_in_cfg = data_cfg.get("split", {}).get("path")
    out_dir = args.out_dir or (os.path.dirname(split_path_in_cfg) if split_path_in_cfg else os.path.join("configs", "anomaly", "splits"))
    os.makedirs(out_dir, exist_ok=True)

    if category_arg == "all":
        categories = get_dataset_categories(dataset_cls, dataset_root)
        if not categories:
            raise LocalAssetError(f"No categories found for dataset '{dataset_name}' under '{dataset_root}'.")
    else:
        categories = [category_arg]

    # Determine output file template from selectors if available
    selector_template = config.get("selectors", {}).get("category", {}).get("data.split.path")

    for cat in categories:
        try:
            split_dict = dataset_cls.generate_split(dataset_root, cat, seed=args.seed)
        except LocalAssetError as exc:
            print(f"Skipping {cat}: {exc}")
            continue

        if selector_template:
            out_filename = os.path.basename(selector_template.replace("{value}", str(cat)))
        else:
            prefix = dataset_name.replace("_anomaly", "").replace("_dataset", "")
            out_filename = f"{prefix}_{cat}.json"

        out_path = os.path.join(out_dir, out_filename)
        save_json(split_dict, out_path)
        print(
            f"wrote {out_path}: train={len(split_dict['train'])} "
            f"valid={len(split_dict['valid'])} test={len(split_dict['test'])}"
        )


if __name__ == "__main__":
    main()
