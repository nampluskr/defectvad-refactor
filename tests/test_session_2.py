import os
import sys
import shutil
import tempfile
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.core.registry import DATASETS, TRANSFORMS, METRICS, ADAPTERS, MODELS, LOSSES
import src.tasks.anomaly


def test_session_2():
    print("--- 1. Testing Registry Registrations ---")
    assert "mvtec_anomaly" in DATASETS.entries
    assert "anomaly_default" in TRANSFORMS.entries
    assert "image_auroc" in METRICS.entries
    assert "pixel_auroc" in METRICS.entries
    assert "stfpm" in ADAPTERS.entries
    assert "stfpm_anomaly" in MODELS.entries
    assert "none" in LOSSES.entries
    print("Registry checks: PASSED")

    print("--- 2. Testing Transforms ---")
    transform = TRANSFORMS.build("anomaly_default", image_size=[128, 128])
    dummy_img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    # Test train shape (image, {})
    out_img, out_target = transform(dummy_img, {})
    assert out_img.shape == (3, 128, 128)
    assert out_target == {}

    # Test eval shape (image, {"mask": mask, "label": label})
    dummy_mask = torch.zeros((200, 200), dtype=torch.int64)
    out_img, out_target = transform(dummy_img, {"mask": dummy_mask, "label": torch.tensor(0)})
    assert out_img.shape == (3, 128, 128)
    assert out_target["mask"].shape == (128, 128)
    print("Transforms: PASSED")

    print("--- 3. Testing Metrics ---")
    img_metric = METRICS.build("image_auroc")
    pix_metric = METRICS.build("pixel_auroc")
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = torch.tensor([0, 1, 0, 1])
    img_metric.update(scores, labels)
    res = float(img_metric.compute())
    assert res == 1.0
    print("Metrics: PASSED")

    print("--- 4. Testing MVTec Dataset ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create minimal mock MVTec structure
        cat_dir = os.path.join(tmp_dir, "bottle")
        os.makedirs(os.path.join(cat_dir, "train", "good"), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "test", "broken_large"), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "test", "good"), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "ground_truth", "broken_large"), exist_ok=True)

        Image.new("RGB", (64, 64)).save(os.path.join(cat_dir, "train", "good", "000.png"))
        Image.new("RGB", (64, 64)).save(os.path.join(cat_dir, "test", "good", "000.png"))
        Image.new("RGB", (64, 64)).save(os.path.join(cat_dir, "test", "broken_large", "000.png"))
        Image.new("L", (64, 64)).save(os.path.join(cat_dir, "ground_truth", "broken_large", "000_mask.png"))

        mock_split_path = os.path.join(tmp_dir, "split.json")
        import json
        with open(mock_split_path, "w") as f:
            json.dump({
                "train": ["train_good/000"],
                "valid": ["good/000"],
                "test": ["broken_large/000"],
            }, f)

        ds_train = DATASETS.build(
            "mvtec_anomaly",
            root=tmp_dir,
            split="train",
            category="bottle",
            split_path=mock_split_path,
            transform=transform,
        )
        assert len(ds_train) == 1
        img, target = ds_train[0]
        assert img.shape == (3, 128, 128)
        assert target == {}

        ds_test = DATASETS.build(
            "mvtec_anomaly",
            root=tmp_dir,
            split="test",
            category="bottle",
            split_path=mock_split_path,
            transform=transform,
        )
        assert len(ds_test) == 1
        img, target = ds_test[0]
        assert img.shape == (3, 128, 128)
        assert target["label"].item() == 1
        assert target["mask"].shape == (128, 128)
        print("MVTec Dataset: PASSED")

        print("--- 5. Testing STFPM Adapter Step ---")
        stfpm_adapter = ADAPTERS.build(
            "stfpm",
            loss_fn=LOSSES.build("none"),
            metrics={"image_auroc": img_metric, "pixel_auroc": pix_metric},
        )
        model = MODELS.build("stfpm", backbone="resnet18", weights_path=None)
        model.train()

        batch_images = torch.randn(2, 3, 128, 128)
        batch_targets = [{}, {}]
        step_out = stfpm_adapter.train_step(model, (batch_images, batch_targets), device="cpu")
        assert "loss" in step_out
        assert step_out["loss"].item() >= 0.0
        print("STFPM Adapter train_step: PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nSESSION 2 ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_session_2()
