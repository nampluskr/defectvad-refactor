"""Generate configs/splits/mvtec_<category>.json for local MVTec AD categories."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import generate_ratio_split
from src.utils.io import save_json

TRAIN_PREFIX = "train_good"
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


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
    parser.add_argument("category", nargs="?", default="all", help="MVTec category or 'all'")
    parser.add_argument("--dataset-root", default="/mnt/d/datasets/mvtec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.join("configs", "splits"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    categories = MVTEC_CATEGORIES if args.category == "all" else [args.category]

    for cat in categories:
        cat_dir = os.path.join(args.dataset_root, cat)
        if not os.path.isdir(cat_dir):
            print(f"Skipping {cat}: directory not found at {cat_dir}")
            continue

        split_dict = build_split(cat_dir, args.seed)
        out_path = os.path.join(args.out_dir, f"mvtec_{cat}.json")
        save_json(split_dict, out_path)
        print(f"wrote {out_path}: train={len(split_dict['train'])} valid={len(split_dict['valid'])} test={len(split_dict['test'])}")


if __name__ == "__main__":
    main()
