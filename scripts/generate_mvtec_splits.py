"""Generate configs/splits/mvtec_<category>.json for a local MVTec AD category.

Reproduces the mvtec_bottle.json split brought in from cv_boilerplate (PLAN-P5 SS3.2/3.4,
docs/dev/v0.1/PRD.md SS5.3): train is the full official train/good set, and valid/test are a
40%/60% split of the official test set, stratified per defect_type via
src/data/split.py#generate_ratio_split with runtime.seed (42) as the split seed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import generate_ratio_split
from src.utils.io import save_json

TRAIN_PREFIX = "train_good"


def build_split(category_dir, seed):
    train_dir = os.path.join(category_dir, "train", "good")
    test_dir = os.path.join(category_dir, "test")

    train_ids = sorted(
        f"{TRAIN_PREFIX}/{os.path.splitext(name)[0]}"
        for name in os.listdir(train_dir)
        if name.endswith(".png")
    )

    test_ids = []
    defect_types = []
    for defect_type in sorted(os.listdir(test_dir)):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        for name in sorted(os.listdir(defect_dir)):
            if not name.endswith(".png"):
                continue
            test_ids.append(f"{defect_type}/{os.path.splitext(name)[0]}")
            defect_types.append(defect_type)

    ratio_split = generate_ratio_split(
        test_ids, ratio={"train": 0.0, "valid": 0.4, "test": 0.6}, seed=seed, stratify_by=defect_types
    )
    assert not ratio_split["train"], "test-folder split must not produce a 'train' bucket"

    return {
        "train": train_ids,
        "valid": sorted(ratio_split["valid"]),
        "test": sorted(ratio_split["test"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("--dataset-root", default="/mnt/d/datasets/mvtec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    category_dir = os.path.join(args.dataset_root, args.category)
    split_dict = build_split(category_dir, args.seed)

    out_path = args.out or os.path.join("configs", "splits", f"mvtec_{args.category}.json")
    save_json(split_dict, out_path)
    print(f"wrote {out_path}: train={len(split_dict['train'])} "
          f"valid={len(split_dict['valid'])} test={len(split_dict['test'])}")


if __name__ == "__main__":
    main()
