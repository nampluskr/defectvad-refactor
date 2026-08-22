import json
import os
import shutil
import sys
import tempfile
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.core.builders import build_model
from src.core.checkpoint import save_checkpoint
from src.core.config import resolve_config
import src.tasks.anomaly


def test_evaluate_script():
    print("--- Testing scripts/evaluate.py end-to-end ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        # Mock dataset structure
        cat_dir = os.path.join(tmp_dir, "mvtec", "bottle")
        os.makedirs(os.path.join(cat_dir, "train", "good"), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "test", "good"), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "test", "broken_large"), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "ground_truth", "broken_large"), exist_ok=True)

        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(os.path.join(cat_dir, "train", "good", "000.png"))
        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(os.path.join(cat_dir, "test", "good", "001.png"))
        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(os.path.join(cat_dir, "test", "good", "002.png"))
        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(os.path.join(cat_dir, "test", "broken_large", "000.png"))
        Image.new("L", (64, 64), color=255).save(os.path.join(cat_dir, "ground_truth", "broken_large", "000_mask.png"))

        # Mock disjoint split
        mock_split = os.path.join(tmp_dir, "mvtec_bottle.json")
        with open(mock_split, "w") as f:
            json.dump({
                "train": ["train_good/000"],
                "valid": ["good/001"],
                "test": ["good/002", "broken_large/000"],
            }, f)



        # Mock local.yaml
        mock_local = os.path.join(tmp_dir, "local.yaml")
        mock_weights = os.path.join(tmp_dir, "resnet18-f37072fd.pth")
        torch.save({}, mock_weights)

        with open(mock_local, "w") as f:
            f.write(f"paths:\n  dataset_root: {tmp_dir}\n  backbone_root: {tmp_dir}\n")

        # Resolve config and build model to save a mock checkpoint
        cfg = resolve_config(
            data_path="configs/anomaly/data/mvtec.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "bottle"},
            model_selectors={"backbone": "resnet18"},
            cli_overrides=[f"data.split.path={mock_split}"],
            local_config_path=mock_local,
        )

        model = build_model(cfg["model"])
        ckpt_path = os.path.join(tmp_dir, "checkpoints", "best.pth")
        save_checkpoint(ckpt_path, model=model, epoch=1, best_metric=1.0, config=cfg)

        # Run evaluate using sys.argv
        out_dir = os.path.join(tmp_dir, "eval_out")
        sys.argv = [
            "scripts/evaluate.py",
            "--data", "configs/anomaly/data/mvtec.yaml",
            "--model", "configs/anomaly/models/stfpm.yaml",
            "--data.category", "bottle",
            "--model.backbone", "resnet18",
            "--checkpoint", ckpt_path,
            "--split", "test",
            "--output_dir", out_dir,
            "--set", f"data.split.path={mock_split}",
            "--set", f"paths.dataset_root={tmp_dir}",
            "--set", f"paths.backbone_root={tmp_dir}",
        ]

        from scripts.evaluate import main
        main()

        # Check outputs
        metrics_file = os.path.join(out_dir, "metrics_test.json")
        log_file = os.path.join(out_dir, "evaluate_test.log")

        assert os.path.isfile(metrics_file), f"Metrics file not found: {metrics_file}"
        assert os.path.isfile(log_file), f"Log file not found: {log_file}"

        with open(metrics_file) as f:
            res = json.load(f)
            print("Evaluated Metrics:", res)
            assert "image_auroc" in res
            assert "pixel_auroc" in res

        print("Evaluate end-to-end test: PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_evaluate_script()
