"""Generate configs/splits/visa_<category>.json from VisA split_csv/1cls.csv."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split import generate_ratio_split
from src.utils.io import save_json

TRAIN_PREFIX = "train_normal"
VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


def extract_stem(image_path):
    filename = os.path.basename(image_path)
    return os.path.splitext(filename)[0]


def build_splits_from_csv(csv_path, seed):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cat_rows = {}
    for r in rows:
        cat = r["object"]
        cat_rows.setdefault(cat, []).append(r)

    results = {}
    for cat, items in cat_rows.items():
        train_ids = []
        test_ids = []
        defect_types = []

        for item in items:
            split_name = item["split"].lower()
            label = item["label"].lower()
            stem = extract_stem(item["image"])

            if split_name == "train":
                train_ids.append(f"{TRAIN_PREFIX}/{stem}")
            else:
                type_prefix = "Normal" if label == "normal" else "Anomaly"
                test_ids.append(f"{type_prefix}/{stem}")
                defect_types.append(label)

        ratio_split = generate_ratio_split(
            test_ids,
            ratio={"train": 0.0, "valid": 0.4, "test": 0.6},
            seed=seed,
            stratify_by=defect_types,
        )

        results[cat] = {
            "train": sorted(train_ids),
            "valid": sorted(ratio_split["valid"]),
            "test": sorted(ratio_split["test"]),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate VisA split files")
    parser.add_argument("category", nargs="?", default="all", help="VisA category or 'all'")
    parser.add_argument("--dataset-root", default="/mnt/d/datasets/visa")
    parser.add_argument("--csv-path", default=None, help="Path to 1cls.csv (defaults to <dataset-root>/split_csv/1cls.csv)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.join("configs", "splits"))
    args = parser.parse_args()

    csv_path = args.csv_path or os.path.join(args.dataset_root, "split_csv", "1cls.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: 1cls.csv not found at {csv_path}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    all_splits = build_splits_from_csv(csv_path, args.seed)

    categories = VISA_CATEGORIES if args.category == "all" else [args.category]

    for cat in categories:
        if cat not in all_splits:
            print(f"Skipping {cat}: not found in CSV")
            continue

        split_dict = all_splits[cat]
        out_path = os.path.join(args.out_dir, f"visa_{cat}.json")
        save_json(split_dict, out_path)
        print(f"wrote {out_path}: train={len(split_dict['train'])} valid={len(split_dict['valid'])} test={len(split_dict['test'])}")


if __name__ == "__main__":
    main()
