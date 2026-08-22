"""Generate configs/splits/btad_<category>.json for local BTAD categories."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import generate_ratio_split
from src.utils.io import save_json

TRAIN_PREFIX = "train_ok"
BTAD_CATEGORIES = ["01", "02", "03"]
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def build_split(category_dir, seed):
    train_dir = os.path.join(category_dir, "train", "ok")
    test_ok_dir = os.path.join(category_dir, "test", "ok")
    test_ko_dir = os.path.join(category_dir, "test", "ko")

    train_files = sorted(os.listdir(train_dir)) if os.path.isdir(train_dir) else []
    train_ids = [
        f"{TRAIN_PREFIX}/{os.path.splitext(name)[0]}"
        for name in train_files
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]

    test_ids = []
    defect_types = []

    if os.path.isdir(test_ok_dir):
        for name in sorted(os.listdir(test_ok_dir)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                test_ids.append(f"ok/{os.path.splitext(name)[0]}")
                defect_types.append("ok")

    if os.path.isdir(test_ko_dir):
        for name in sorted(os.listdir(test_ko_dir)):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                test_ids.append(f"ko/{os.path.splitext(name)[0]}")
                defect_types.append("ko")

    ratio_split = generate_ratio_split(
        test_ids,
        ratio={"train": 0.0, "valid": 0.4, "test": 0.6},
        seed=seed,
        stratify_by=defect_types,
    )
    assert not ratio_split["train"], "test-folder split must not produce a 'train' bucket"

    return {
        "train": sorted(train_ids),
        "valid": sorted(ratio_split["valid"]),
        "test": sorted(ratio_split["test"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate BTAD split files")
    parser.add_argument("category", nargs="?", default="all", help="BTAD category (01, 02, 03) or 'all'")
    parser.add_argument("--dataset-root", default="/mnt/d/datasets/btad")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.join("configs", "splits"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    categories = BTAD_CATEGORIES if args.category == "all" else [args.category]

    for cat in categories:
        cat_dir = os.path.join(args.dataset_root, cat)
        if not os.path.isdir(cat_dir):
            print(f"Skipping {cat}: directory not found at {cat_dir}")
            continue

        split_dict = build_split(cat_dir, args.seed)
        out_path = os.path.join(args.out_dir, f"btad_{cat}.json")
        save_json(split_dict, out_path)
        print(f"wrote {out_path}: train={len(split_dict['train'])} valid={len(split_dict['valid'])} test={len(split_dict['test'])}")


if __name__ == "__main__":
    main()
