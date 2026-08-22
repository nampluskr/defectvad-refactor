import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image

from src.core.builders import build_adapter, build_dataloader, build_dataset, build_loss, build_metrics, build_model, build_transforms
from src.core.checkpoint import save_checkpoint
from src.core.config import resolve_config
from src.core.registry import DATASETS
from src.data.split import assert_disjoint, load_split_file
import src.tasks.anomaly


def test_btad_and_visa_integration():
    print("--- 1. Testing BTAD and VisA Registry & Split Integrity ---")
    assert "btad_anomaly" in DATASETS
    assert "visa_anomaly" in DATASETS

    # Check BTAD splits
    btad_cats = ["01", "02", "03"]
    for cat in btad_cats:
        split_path = f"configs/splits/btad_{cat}.json"
        assert os.path.isfile(split_path), f"Split file not found: {split_path}"
        split_dict = load_split_file(split_path)
        assert_disjoint(split_dict)
        assert len(split_dict["train"]) > 0
        assert len(split_dict["valid"]) > 0
        assert len(split_dict["test"]) > 0

    # Check VisA splits
    visa_cats = [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]
    for cat in visa_cats:
        split_path = f"configs/splits/visa_{cat}.json"
        assert os.path.isfile(split_path), f"Split file not found: {split_path}"
        split_dict = load_split_file(split_path)
        assert_disjoint(split_dict)
        assert len(split_dict["train"]) > 0
        assert len(split_dict["valid"]) > 0
        assert len(split_dict["test"]) > 0

    print("--- 2. Testing Config Resolution with Selectors ---")
    cfg_btad = resolve_config(
        data_path="configs/anomaly/data/btad.yaml",
        model_path="configs/anomaly/models/stfpm.yaml",
        data_selectors={"category": "02"},
        model_selectors={"backbone": "resnet18"},
    )
    assert cfg_btad["data"]["params"]["category"] == "02"
    assert cfg_btad["data"]["split"]["path"] == "configs/splits/btad_02.json"

    cfg_visa = resolve_config(
        data_path="configs/anomaly/data/visa.yaml",
        model_path="configs/anomaly/models/stfpm.yaml",
        data_selectors={"category": "pcb1"},
        model_selectors={"backbone": "resnet18"},
    )
    assert cfg_visa["data"]["params"]["category"] == "pcb1"
    assert cfg_visa["data"]["split"]["path"] == "configs/splits/visa_pcb1.json"

    print("--- 3. Testing Dataset Contract and DataLoader with Real Assets ---")
    # BTAD 01 dataset loading if asset exists
    if os.path.isdir("/mnt/d/datasets/btad/01"):
        cfg_btad_01 = resolve_config(
            data_path="configs/anomaly/data/btad.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "01"},
        )
        transforms = build_transforms(cfg_btad_01["data"])
        adapter = build_adapter(cfg_btad_01["adapter"], loss_fn=None, metrics={})

        ds_train = build_dataset(cfg_btad_01["data"], split="train", transform=transforms["train"])
        ds_test = build_dataset(cfg_btad_01["data"], split="test", transform=transforms["eval"])

        assert len(ds_train) > 0
        img_train, target_train = ds_train[0]
        assert isinstance(img_train, torch.Tensor)
        assert img_train.shape[0] == 3
        assert isinstance(target_train, dict)

        assert len(ds_test) > 0
        img_test, target_test = ds_test[0]
        assert isinstance(img_test, torch.Tensor)
        assert "label" in target_test
        assert "mask" in target_test
        assert target_test["label"].dtype == torch.long
        assert target_test["mask"].ndim == 2

        loader = build_dataloader(ds_test, cfg_btad_01["data"], split="test", adapter=adapter, seed=42, device="cpu", allow_test_split=True)
        batch = next(iter(loader))
        assert batch[0].shape[0] == min(cfg_btad_01["data"]["batch_size"], len(ds_test))

    # VisA candle dataset loading if asset exists
    if os.path.isdir("/mnt/d/datasets/visa/candle"):
        cfg_visa_candle = resolve_config(
            data_path="configs/anomaly/data/visa.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "candle"},
        )
        transforms = build_transforms(cfg_visa_candle["data"])
        adapter = build_adapter(cfg_visa_candle["adapter"], loss_fn=None, metrics={})

        ds_train = build_dataset(cfg_visa_candle["data"], split="train", transform=transforms["train"])
        ds_test = build_dataset(cfg_visa_candle["data"], split="test", transform=transforms["eval"])

        assert len(ds_train) > 0
        img_train, target_train = ds_train[0]
        assert isinstance(img_train, torch.Tensor)
        assert img_train.shape[0] == 3

        assert len(ds_test) > 0
        img_test, target_test = ds_test[0]
        assert isinstance(img_test, torch.Tensor)
        assert "label" in target_test
        assert "mask" in target_test

        loader = build_dataloader(ds_test, cfg_visa_candle["data"], split="test", adapter=adapter, seed=42, device="cpu", allow_test_split=True)
        batch = next(iter(loader))
        assert batch[0].shape[0] == min(cfg_visa_candle["data"]["batch_size"], len(ds_test))

    print("--- 4. Testing End-to-End Evaluation CLI on Mock Environment ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        # Mock backbone weights
        mock_weights = os.path.join(tmp_dir, "resnet18-f37072fd.pth")
        torch.save({}, mock_weights)

        # Mock local.yaml
        mock_local = os.path.join(tmp_dir, "local.yaml")
        with open(mock_local, "w") as f:
            f.write(f"paths:\n  dataset_root: {tmp_dir}\n  backbone_root: {tmp_dir}\n")

        # Create mock BTAD structure
        mock_btad_dir = os.path.join(tmp_dir, "btad", "01")
        os.makedirs(os.path.join(mock_btad_dir, "train", "ok"), exist_ok=True)
        os.makedirs(os.path.join(mock_btad_dir, "test", "ok"), exist_ok=True)
        os.makedirs(os.path.join(mock_btad_dir, "test", "ko"), exist_ok=True)
        os.makedirs(os.path.join(mock_btad_dir, "ground_truth", "ko"), exist_ok=True)

        Image.new("RGB", (64, 64), color=(50, 50, 50)).save(os.path.join(mock_btad_dir, "train", "ok", "0000.png"))
        Image.new("RGB", (64, 64), color=(60, 60, 60)).save(os.path.join(mock_btad_dir, "test", "ok", "0001.png"))
        Image.new("RGB", (64, 64), color=(70, 70, 70)).save(os.path.join(mock_btad_dir, "test", "ok", "0003.png"))
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(os.path.join(mock_btad_dir, "test", "ko", "0002.png"))
        Image.new("L", (64, 64), color=255).save(os.path.join(mock_btad_dir, "ground_truth", "ko", "0002.png"))

        mock_split_btad = os.path.join(tmp_dir, "mock_btad_01.json")
        with open(mock_split_btad, "w") as f:
            json.dump({"train": ["train_ok/0000"], "valid": ["ok/0001"], "test": ["ok/0003", "ko/0002"]}, f)

        # Build mock model and checkpoint
        cfg = resolve_config(
            data_path="configs/anomaly/data/btad.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "01"},
            model_selectors={"backbone": "resnet18"},
            local_config_path=mock_local,
            cli_overrides=[f"data.split.path={mock_split_btad}"],
        )

        model = build_model(cfg["model"])
        ckpt_path = os.path.join(tmp_dir, "best.pth")
        save_checkpoint(
            ckpt_path,
            model=model,
            epoch=1,
            best_metric=0.99,
            config=cfg,
        )

        # Run evaluate CLI
        eval_out = os.path.join(tmp_dir, "eval_out")
        sys.argv = [
            "scripts/evaluate.py",
            "--data", "configs/anomaly/data/btad.yaml",
            "--data.category", "01",
            "--model", "configs/anomaly/models/stfpm.yaml",
            "--model.backbone", "resnet18",
            "--checkpoint", ckpt_path,
            "--split", "test",
            "--output-dir", eval_out,
            "--set", f"paths.backbone_root={tmp_dir}",
            "--set", f"paths.dataset_root={tmp_dir}",
            "--set", f"data.split.path={mock_split_btad}",
        ]

        from scripts.evaluate import main as eval_main
        eval_main()

        metrics_path = os.path.join(eval_out, "metrics_test.json")
        assert os.path.isfile(metrics_path), f"metrics_test.json not found: {metrics_path}"
        with open(metrics_path) as f:
            metrics_data = json.load(f)
            assert "image_auroc" in metrics_data
            assert "pixel_auroc" in metrics_data

        print("BTAD and VisA evaluation integration test: PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_btad_and_visa_integration()
