import json
import os
import shutil
import sys
import tempfile
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.core.builders import build_model
from src.core.checkpoint import save_checkpoint
from src.core.config import resolve_config
import src.tasks.anomaly


def test_predict_script():
    print("--- Testing scripts/predict.py end-to-end ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        # Mock test image directory
        img_dir = os.path.join(tmp_dir, "test_images")
        os.makedirs(img_dir, exist_ok=True)
        img1_path = os.path.join(img_dir, "sample1.png")
        img2_path = os.path.join(img_dir, "sample2.jpg")

        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(img1_path)
        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(img2_path)

        # Mock local.yaml and weights
        mock_local = os.path.join(tmp_dir, "local.yaml")
        mock_weights = os.path.join(tmp_dir, "resnet18-f37072fd.pth")
        torch.save({}, mock_weights)

        with open(mock_local, "w") as f:
            f.write(f"paths:\n  dataset_root: {tmp_dir}\n  backbone_root: {tmp_dir}\n")

        # Mock config and checkpoint
        cfg = resolve_config(
            data_path="configs/anomaly/data/mvtec.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "bottle"},
            model_selectors={"backbone": "resnet18"},
            local_config_path=mock_local,
        )

        model = build_model(cfg["model"])
        ckpt_path = os.path.join(tmp_dir, "checkpoints", "best.pth")
        save_checkpoint(
            ckpt_path,
            model=model,
            epoch=1,
            best_metric=1.0,
            config=cfg,
            adapter_state={"image_threshold": 0.5, "pixel_threshold": 0.5},
        )

        # Run predict on directory
        out_dir = os.path.join(tmp_dir, "predict_out")
        sys.argv = [
            "scripts/predict.py",
            "--model", "configs/anomaly/models/stfpm.yaml",
            "--model.backbone", "resnet18",
            "--checkpoint", ckpt_path,
            "--input", img_dir,
            "--output_dir", out_dir,
            "--set", f"paths.backbone_root={tmp_dir}",
            "--set", f"paths.dataset_root={tmp_dir}",
        ]

        from scripts.predict import main
        main()

        # Check outputs
        pred_json = os.path.join(out_dir, "predictions.json")
        log_file = os.path.join(out_dir, "predict.log")
        vis_file1 = os.path.join(out_dir, "visualizations", "sample1_vis.png")
        vis_file2 = os.path.join(out_dir, "visualizations", "sample2_vis.png")

        assert os.path.isfile(pred_json), f"Predictions file not found: {pred_json}"
        assert os.path.isfile(log_file), f"Log file not found: {log_file}"
        assert os.path.isfile(vis_file1), f"Visualization 1 not found: {vis_file1}"
        assert os.path.isfile(vis_file2), f"Visualization 2 not found: {vis_file2}"

        with open(pred_json) as f:
            records = json.load(f)
            assert len(records) == 2
            print("Predictions record sample:", records[0])
            assert "anomaly_score" in records[0]
            assert "is_anomalous" in records[0]
            assert records[0]["threshold"] == 0.5

        # Test single image input
        out_single = os.path.join(tmp_dir, "predict_single")
        sys.argv = [
            "scripts/predict.py",
            "--model", "configs/anomaly/models/stfpm.yaml",
            "--model.backbone", "resnet18",
            "--checkpoint", ckpt_path,
            "--input", img1_path,
            "--output_dir", out_single,
            "--threshold", "0.2",
            "--set", f"paths.backbone_root={tmp_dir}",
            "--set", f"paths.dataset_root={tmp_dir}",
        ]
        main()

        with open(os.path.join(out_single, "predictions.json")) as f:
            single_records = json.load(f)
            assert len(single_records) == 1
            assert single_records[0]["threshold"] == 0.2

        print("Predict end-to-end test: PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_predict_script()
