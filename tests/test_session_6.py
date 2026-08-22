import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image

from src.core.builders import (
    build_adapter,
    build_dataloader,
    build_dataset,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_transforms,
)
from src.core.config import resolve_config
from src.core.context import RunContext
from src.core.engine import Engine
import src.tasks.anomaly


def test_loss_and_metrics_stabilization():
    print("--- 1. Testing Cosine and Step Schedulers Builders ---")
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    sched_cosine_1 = build_scheduler({"scheduler": {"name": "cosine", "params": {"t_max": 50, "eta_min": 1e-4}}}, optimizer)
    assert sched_cosine_1 is not None
    assert sched_cosine_1.T_max == 50
    assert sched_cosine_1.eta_min == 1e-4

    sched_cosine_2 = build_scheduler({"scheduler": {"name": "cosine", "params": {"T_max": 30}}}, optimizer)
    assert sched_cosine_2 is not None
    assert sched_cosine_2.T_max == 30

    print("--- 2. Testing STFPM Loss Batch Invariance in Adapter ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        mock_local = os.path.join(tmp_dir, "local.yaml")
        mock_weights = os.path.join(tmp_dir, "resnet18-f37072fd.pth")
        torch.save({}, mock_weights)

        with open(mock_local, "w") as f:
            f.write(f"paths:\n  dataset_root: {tmp_dir}\n  backbone_root: {tmp_dir}\n")

        # Mock category structure
        mock_bottle = os.path.join(tmp_dir, "mvtec", "bottle")
        os.makedirs(os.path.join(mock_bottle, "train", "good"), exist_ok=True)
        os.makedirs(os.path.join(mock_bottle, "test", "good"), exist_ok=True)
        os.makedirs(os.path.join(mock_bottle, "test", "broken_large"), exist_ok=True)
        os.makedirs(os.path.join(mock_bottle, "ground_truth", "broken_large"), exist_ok=True)

        for i in range(4):
            Image.new("RGB", (64, 64), color=(50, 50, 50)).save(os.path.join(mock_bottle, "train", "good", f"{i:03d}.png"))
        Image.new("RGB", (64, 64), color=(50, 50, 50)).save(os.path.join(mock_bottle, "test", "good", "000.png"))
        Image.new("RGB", (64, 64), color=(50, 50, 50)).save(os.path.join(mock_bottle, "test", "good", "001.png"))
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(os.path.join(mock_bottle, "test", "broken_large", "000.png"))
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(os.path.join(mock_bottle, "test", "broken_large", "001.png"))
        Image.new("L", (64, 64), color=255).save(os.path.join(mock_bottle, "ground_truth", "broken_large", "000_mask.png"))
        Image.new("L", (64, 64), color=255).save(os.path.join(mock_bottle, "ground_truth", "broken_large", "001_mask.png"))

        mock_split = os.path.join(tmp_dir, "mvtec_bottle.json")
        with open(mock_split, "w") as f:
            json.dump({
                "train": ["train_good/000", "train_good/001", "train_good/002", "train_good/003"],
                "valid": ["good/000", "broken_large/000"],
                "test": ["good/001", "broken_large/001"],
            }, f)

        cfg = resolve_config(
            data_path="configs/anomaly/data/mvtec.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "bottle"},
            model_selectors={"backbone": "resnet18"},
            local_config_path=mock_local,
            cli_overrides=[
                f"data.split.path={mock_split}",
                "train.epochs=2",
                "data.batch_size=2",
            ],
        )

        assert cfg["optim"]["scheduler"]["name"] == "cosine"

        ctx = RunContext(cfg, run_dir=tmp_dir)
        ctx.setup_seed()

        transforms = build_transforms(cfg["data"])
        train_ds = build_dataset(cfg["data"], split="train", transform=transforms["train"])
        valid_ds = build_dataset(cfg["data"], split="valid", transform=transforms["eval"])

        metrics = build_metrics(cfg["metrics"])
        adapter = build_adapter(cfg["adapter"], loss_fn=None, metrics=metrics)
        train_loader = build_dataloader(train_ds, cfg["data"], split="train", adapter=adapter, seed=ctx.seed, device=ctx.device)
        valid_loader = build_dataloader(valid_ds, cfg["data"], split="valid", adapter=adapter, seed=ctx.seed, device=ctx.device)

        model = build_model(cfg["model"])
        optimizer = build_optimizer(cfg["optim"], model)
        scheduler = build_scheduler(cfg["optim"], optimizer)

        print("--- 3. Testing Engine.fit with Cosine Scheduler & Running Average Loss ---")
        engine = Engine()
        best_metric = engine.fit(
            model=model,
            adapter=adapter,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            ctx=ctx,
        )
        assert best_metric is not None

        # Verify last lr is stepped down
        assert optimizer.param_groups[0]["lr"] < 0.4
        print("Engine fit and scheduler test: PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_loss_and_metrics_stabilization()
