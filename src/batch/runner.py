import copy
import gc
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from src.batch.parser import BatchCase, BatchConfig
from src.batch.summary import BatchSummary, CaseResult, compute_mean_metrics
from src.core.builders import (
    build_adapter,
    build_dataloader,
    build_dataset,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_transforms,
)
from src.core.checkpoint import load_checkpoint
from src.core.config import resolve_config, validate_config
from src.core.context import RunContext
from src.core.engine import Engine
from src.core.logger import MetricsCsvWriter, format_result, setup_logger
from src.utils.io import save_json, save_yaml


class BatchRunner:
    def __init__(
        self,
        batch_config: BatchConfig,
        mode: str = "train",
        overwrite: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = batch_config
        self.mode = mode.lower()
        self.overwrite = overwrite
        self.logger = logger or logging.getLogger("cv_boilerplate.batch")

    def run(self) -> BatchSummary:
        start_time = time.perf_counter()
        total_cases = len(self.config.cases)
        results: List[CaseResult] = []

        self.logger.info(f"=== Starting Batch Execution: {self.config.name} ({total_cases} cases, mode={self.mode}) ===")
        self.logger.info(f"Output Root: {self.config.output_root}")

        for idx, case in enumerate(self.config.cases, 1):
            case_start = time.perf_counter()
            self.logger.info(f"\n[{idx}/{total_cases}] Running Case: {case.case_id} ({case.run_name})")

            case_dir = case.output_dir
            os.makedirs(case_dir, exist_ok=True)

            # Check for existing completed run if not overwrite
            metrics_file = os.path.join(case_dir, f"metrics_{case.split}.json")
            if not self.overwrite and self.mode in ("evaluate", "all") and os.path.isfile(metrics_file):
                self.logger.info(f"Existing metrics found at {metrics_file}, skipping case (overwrite=False)")
                try:
                    import json
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        cached_metrics = json.load(f)
                    results.append(
                        CaseResult(
                            case_id=case.case_id,
                            run_name=case.run_name,
                            status="SKIPPED",
                            mode=self.mode,
                            metrics=cached_metrics,
                            elapsed_sec=0.0,
                            output_dir=case_dir,
                            meta=case.meta,
                        )
                    )
                    continue
                except Exception:
                    pass

            try:
                metrics_out = self._execute_case(case)
                elapsed = time.perf_counter() - case_start
                self.logger.info(f"[{idx}/{total_cases}] Case {case.case_id} SUCCESS in {elapsed:.2f}s")
                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        run_name=case.run_name,
                        status="SUCCESS",
                        mode=self.mode,
                        metrics=metrics_out or {},
                        elapsed_sec=elapsed,
                        output_dir=case_dir,
                        meta=case.meta,
                    )
                )
            except Exception as e:
                elapsed = time.perf_counter() - case_start
                self.logger.error(f"[{idx}/{total_cases}] Case {case.case_id} FAILED in {elapsed:.2f}s: {e}")
                results.append(
                    CaseResult(
                        case_id=case.case_id,
                        run_name=case.run_name,
                        status="FAILED",
                        mode=self.mode,
                        metrics={},
                        elapsed_sec=elapsed,
                        error_msg=str(e),
                        output_dir=case_dir,
                        meta=case.meta,
                    )
                )
            finally:
                # Cleanup GPU and system memory after each case
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        total_elapsed = time.perf_counter() - start_time
        success_count = sum(1 for r in results if r.status == "SUCCESS")
        failed_count = sum(1 for r in results if r.status == "FAILED")
        skipped_count = sum(1 for r in results if r.status == "SKIPPED")
        mean_metrics = compute_mean_metrics(results)

        return BatchSummary(
            batch_name=self.config.name,
            task_name=self.config.task_name,
            mode=self.mode,
            total_cases=total_cases,
            success_cases=success_count,
            failed_cases=failed_count,
            skipped_cases=skipped_count,
            total_elapsed_sec=total_elapsed,
            results=results,
            mean_metrics=mean_metrics,
        )

    def _execute_case(self, case: BatchCase) -> Optional[Dict[str, float]]:
        # Resolve config for the individual case
        config = resolve_config(
            data_path=case.data_path,
            model_path=case.model_path,
            data_selectors=case.data_selectors,
            model_selectors=case.model_selectors,
            cli_overrides=case.extra_overrides,
        )
        validate_config(config)

        case_dir = case.output_dir
        os.makedirs(case_dir, exist_ok=True)
        save_yaml(config, os.path.join(case_dir, "config.resolved.yaml"))

        # Setup context and logger for this case
        case_logger = setup_logger(case_dir, log_file=f"batch_{self.mode}.log")
        ctx = RunContext(config, run_dir=case_dir)
        ctx.setup_seed()
        ctx.start(f"batch {self.mode} {case.case_id}")

        results_metrics = {}

        # 1. Train Mode or Part of "all"
        if self.mode in ("train", "all"):
            case_logger.info(f"Starting training for case {case.case_id}...")
            checkpoint_dir = os.path.join(case_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            transforms = build_transforms(config["data"])
            train_dataset = build_dataset(config["data"], split="train", transform=transforms["train"])
            valid_dataset = build_dataset(config["data"], split="valid", transform=transforms["eval"])

            loss_fn = build_loss(config["loss"])
            metrics = build_metrics(config.get("metrics", []))
            adapter = build_adapter(config["adapter"], loss_fn=loss_fn, metrics=metrics)

            train_loader = build_dataloader(
                train_dataset,
                config["data"],
                split="train",
                adapter=adapter,
                seed=ctx.seed,
                device=config["runtime"]["device"],
            )
            valid_loader = build_dataloader(
                valid_dataset,
                config["data"],
                split="valid",
                adapter=adapter,
                seed=ctx.seed,
                device=config["runtime"]["device"],
            )

            model = build_model(config["model"])
            optimizer = build_optimizer(config["optim"], model)
            scheduler = build_scheduler(config["optim"], optimizer)

            metric_names = [m["name"] for m in config.get("metrics", [])]
            metrics_writer = MetricsCsvWriter(case_dir, metric_names)

            engine = Engine(logger=case_logger, metrics_writer=metrics_writer, checkpoint_dir=checkpoint_dir)
            best_score = engine.fit(
                model=model,
                adapter=adapter,
                train_loader=train_loader,
                valid_loader=valid_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                ctx=ctx,
            )
            monitor_name = config.get("train", {}).get("monitor", {}).get("metric", "val_score")
            case_logger.info(f"Training completed. Best {monitor_name}: {best_score:.4f}")
            results_metrics["best_val_score"] = float(best_score)

        # 2. Evaluate Mode or Part of "all"
        if self.mode in ("evaluate", "all"):
            ckpt_path = case.checkpoint_path
            if not ckpt_path or not os.path.isfile(ckpt_path):
                local_ckpt = os.path.join(case_dir, "checkpoints", "best.pth")
                if os.path.isfile(local_ckpt):
                    ckpt_path = local_ckpt
                else:
                    raise FileNotFoundError(f"Checkpoint not found for evaluation: {ckpt_path or local_ckpt}")

            case_logger.info(f"Evaluating case {case.case_id} on split {case.split} using {ckpt_path}...")
            transforms = build_transforms(config["data"])
            eval_transform = transforms.get("eval", transforms.get("test", transforms.get("train")))
            eval_dataset = build_dataset(config["data"], split=case.split, transform=eval_transform)

            loss_fn = build_loss(config["loss"])
            metrics = build_metrics(config.get("metrics", []))
            adapter = build_adapter(config["adapter"], loss_fn=loss_fn, metrics=metrics)

            eval_loader = build_dataloader(
                eval_dataset,
                config["data"],
                split=case.split,
                adapter=adapter,
                seed=ctx.seed,
                device=config["runtime"]["device"],
                allow_test_split=True,
            )

            model = build_model(config["model"])
            model.to(ctx.device)

            ckpt_data = load_checkpoint(ckpt_path, model=model, map_location=ctx.device, restore_rng=False)
            if "adapter_state" in ckpt_data and ckpt_data["adapter_state"] is not None:
                if hasattr(adapter, "load_state_dict"):
                    adapter.load_state_dict(ckpt_data["adapter_state"])
                elif isinstance(ckpt_data["adapter_state"], dict):
                    for k, v in ckpt_data["adapter_state"].items():
                        if hasattr(adapter, k):
                            setattr(adapter, k, v)

            engine = Engine(logger=case_logger)
            eval_results, elapsed_eval = engine.evaluate(model, adapter, eval_loader, ctx, split=case.split)

            metrics_file = os.path.join(case_dir, f"metrics_{case.split}.json")
            save_json(eval_results, metrics_file)

            formatted = format_result(eval_results, sep=", ")
            case_logger.info(f"Evaluation completed in {elapsed_eval:.2f}s: {formatted}")
            results_metrics.update(eval_results)

        ctx.finish()
        return results_metrics
