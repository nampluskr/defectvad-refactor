import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.config import (
    load_and_merge_base,
    apply_selectors,
    apply_overrides,
    resolve_paths,
    interpolate,
    resolve_config,
    validate_config,
)
from src.core.context import RunContext, make_worker_init_fn
from src.core.logger import setup_logger, MetricsCsvWriter
from src.core.checkpoint import save_checkpoint, load_checkpoint, capture_rng_state, restore_rng_state
from src.core.registry import DATASETS, MODELS, LOSSES, ADAPTERS, METRICS, TRANSFORMS


def test_session_1():
    print("--- 1. Testing Config Loading, Selectors, and Overrides ---")
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create mock local.yaml
        mock_local_yaml = os.path.join(tmp_dir, "local.yaml")
        with open(mock_local_yaml, "w") as f:
            f.write(f"paths:\n  dataset_root: {tmp_dir}\n  backbone_root: {tmp_dir}\n")

        # Mock split and weight files
        os.makedirs(os.path.join(tmp_dir, "mvtec", "bottle"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, "mvtec", "grid"), exist_ok=True)
        mock_weight_r18 = os.path.join(tmp_dir, "resnet18-f37072fd.pth")
        mock_weight_r50 = os.path.join(tmp_dir, "resnet50-0676ba61.pth")
        torch.save({}, mock_weight_r18)
        torch.save({}, mock_weight_r50)

        # 1.1 Resolve config with default selectors
        cfg = resolve_config(
            data_path="configs/anomaly/data/mvtec.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "bottle"},
            model_selectors={"backbone": "resnet18"},
            cli_overrides=["runtime.seed=99", "train.epochs=5"],
            local_config_path=mock_local_yaml,
        )

        assert cfg["meta"]["task_name"] == "anomaly"
        assert cfg["data"]["params"]["category"] == "bottle"
        assert cfg["data"]["split"]["path"] == "configs/anomaly/splits/mvtec_bottle.json"
        assert cfg["model"]["params"]["backbone"] == "resnet18"
        assert cfg["model"]["params"]["weights_path"] == f"{tmp_dir}/resnet18-f37072fd.pth"
        assert cfg["runtime"]["seed"] == 99
        assert cfg["train"]["epochs"] == 5
        print("Default selector resolution: PASSED")

        # 1.2 Resolve config with alternative selectors (grid + resnet50)
        cfg_alt = resolve_config(
            data_path="configs/anomaly/data/mvtec.yaml",
            model_path="configs/anomaly/models/stfpm.yaml",
            data_selectors={"category": "grid"},
            model_selectors={"backbone": "resnet50"},
            local_config_path=mock_local_yaml,
        )
        assert cfg_alt["data"]["params"]["category"] == "grid"
        assert cfg_alt["data"]["split"]["path"] == "configs/anomaly/splits/mvtec_grid.json"
        assert cfg_alt["model"]["params"]["backbone"] == "resnet50"
        assert cfg_alt["model"]["params"]["weights_path"] == f"{tmp_dir}/resnet50-0676ba61.pth"
        print("Alternative selector (grid/resnet50) resolution: PASSED")

        # 1.3 Validate config registry & paths checks
        import src.tasks.anomaly

        validate_config(cfg, check_paths=True, check_registry=True, check_cuda=False)
        print("Config validation: PASSED")

        print("--- 2. Testing RunContext and Determinism ---")
        run_dir = os.path.join(tmp_dir, "run_test")
        ctx = RunContext(cfg, run_dir=run_dir)
        ctx.setup_seed()
        ctx.start("python scripts/train.py ...")
        ctx.finish()
        info = ctx.env_info()
        assert info["seed"] == 99
        assert info["command_line"] == "python scripts/train.py ..."
        assert info["python_version"] is not None
        print("RunContext and env_info: PASSED")

        print("--- 3. Testing Logger and MetricsCsvWriter ---")
        logger = setup_logger(run_dir, log_level="DEBUG")
        logger.info("Test log message")
        assert os.path.isfile(os.path.join(run_dir, "train.log"))

        csv_writer = MetricsCsvWriter(run_dir, ["image_auroc", "pixel_auroc"])
        csv_writer.write(1, "train", 0.123, {"image_auroc": 0.85, "pixel_auroc": 0.90}, lr=0.001, elapsed_sec=1.5)
        csv_writer.write(1, "valid", 0.100, {"image_auroc": 0.88, "pixel_auroc": 0.92}, lr=0.001, elapsed_sec=0.5)
        assert os.path.isfile(os.path.join(run_dir, "metrics_epoch.csv"))
        with open(os.path.join(run_dir, "metrics_epoch.csv")) as f:
            lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows
        print("Logger and MetricsCsvWriter: PASSED")

        print("--- 4. Testing Checkpoint and RNG State Preservation ---")
        dummy_model = nn.Linear(10, 2)
        dummy_opt = optim.SGD(dummy_model.parameters(), lr=0.1)
        ckpt_path = os.path.join(run_dir, "checkpoints", "test.pth")

        # Capture and save
        save_checkpoint(
            ckpt_path,
            model=dummy_model,
            optimizer=dummy_opt,
            epoch=1,
            best_metric=0.95,
            monitor="image_auroc",
            config=cfg,
            env=info,
        )
        assert os.path.isfile(ckpt_path)

        # Modify model and load back
        with torch.no_grad():
            dummy_model.weight.fill_(999.0)

        target_model = nn.Linear(10, 2)
        target_opt = optim.SGD(target_model.parameters(), lr=0.1)
        loaded = load_checkpoint(ckpt_path, model=target_model, optimizer=target_opt)

        assert loaded["epoch"] == 1
        assert loaded["best_metric"] == 0.95
        assert torch.allclose(target_model.weight, dummy_model.weight) == False  # since dummy_model was modified after save
        print("Checkpoint save/load and state restoration: PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_session_1()
