import os
import time

import torch
from tqdm import tqdm

from src.core.checkpoint import save_checkpoint
from src.core.logger import format_result


class Trainer:
    def __init__(self, logger=None, metrics_writer=None, checkpoint_dir=None):
        self.logger = logger
        self.metrics_writer = metrics_writer
        self.checkpoint_dir = checkpoint_dir

    def log(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def fit(
        self,
        model,
        adapter,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        ctx,
        start_epoch=1,
        best_metric=None,
    ):
        model.to(ctx.device)
        scaler = torch.amp.GradScaler("cuda", enabled=(ctx.amp and ctx.device.type == "cuda"))
        loaders = {"train": train_loader, "valid": valid_loader}
        monitor_metric = ctx.config["train"]["monitor"]["metric"]
        monitor_mode = ctx.config["train"]["monitor"]["mode"]

        adapter.on_fit_start(model, loaders, ctx.device)
        best_checkpoint_path = (
            os.path.join(self.checkpoint_dir, "best.pth") if self.checkpoint_dir else None
        )

        for epoch in range(start_epoch, ctx.epochs + 1):
            adapter.on_epoch_start(model, epoch)
            loss_avg, train_time = self._train_epoch(
                model, adapter, train_loader, optimizer, scaler, ctx, epoch
            )
            if scheduler is not None:
                scheduler.step()

            adapter.on_validation_start(model, loaders, ctx.device)
            valid_results, valid_time = self.evaluate(
                model, adapter, valid_loader, ctx, epoch=epoch, split="valid"
            )
            adapter.on_epoch_end(model, epoch, valid_results)

            total_time = train_time + valid_time
            lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            valid_str = format_result(valid_results, sep=", ")
            valid_part = f" | {valid_str}" if valid_str else ""
            self.log(
                f"[{epoch:2d}/{ctx.epochs:2d}] "
                f"loss={loss_avg['loss']:.3f}"
                f"{valid_part} | "
                f"lr={lr:.1e} ({total_time:.1f}s)"
            )

            current = valid_results.get(monitor_metric)
            improved = current is not None and (
                best_metric is None
                or (monitor_mode == "max" and current > best_metric)
                or (monitor_mode == "min" and current < best_metric)
            )
            if improved:
                best_metric = current
                if self.checkpoint_dir:
                    save_checkpoint(
                        os.path.join(self.checkpoint_dir, "best.pth"),
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_metric,
                        ctx.config["train"]["monitor"],
                        ctx.config,
                        ctx.env_info(),
                    )
            if self.checkpoint_dir and ctx.config["train"].get("save_last", True):
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, "last.pth"),
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_metric,
                    ctx.config["train"]["monitor"],
                    ctx.config,
                    ctx.env_info(),
                )

        if best_checkpoint_path is not None and os.path.isfile(best_checkpoint_path):
            from src.core.checkpoint import load_checkpoint
            load_checkpoint(best_checkpoint_path, model, map_location=ctx.device, restore_rng=False)

        adapter.on_fit_end(model, loaders, ctx.device)

        if best_checkpoint_path is not None and os.path.isfile(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=ctx.device, weights_only=False)
            checkpoint["model_state"] = model.state_dict()
            torch.save(checkpoint, best_checkpoint_path)

        return best_metric

    def _train_epoch(self, model, adapter, loader, optimizer, scaler, ctx, epoch):
        model.train()
        start = time.perf_counter()
        total_count = 0
        loss_scalar_sum = 0.0
        loss_dict_sum = {}

        pbar = tqdm(loader, desc=f"[{epoch:2d}/{ctx.epochs:2d}]", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(ctx.amp and ctx.device.type == "cuda")):
                step_out = adapter.train_step(model, batch, ctx.device)
            loss = step_out["loss"]
            scaler.scale(loss).backward()
            if ctx.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), ctx.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_count = adapter.batch_size(batch)
            total_count += batch_count
            loss_scalar_sum += float(loss.detach()) * batch_count
            for key, value in step_out.get("loss_dict", {}).items():
                loss_dict_sum[key] = loss_dict_sum.get(key, 0.0) + value * batch_count

            lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
            pbar.set_postfix_str(f"loss={float(loss.detach()):.3f}, lr={lr:.1e}")

        elapsed = time.perf_counter() - start
        loss_avg = {"loss": loss_scalar_sum / max(total_count, 1)}
        loss_avg.update({key: value / max(total_count, 1) for key, value in loss_dict_sum.items()})
        lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
        if self.metrics_writer is not None:
            self.metrics_writer.write(epoch, "train", loss_avg.get("loss", 0.0), {}, lr, elapsed)
        return loss_avg, elapsed

    def evaluate(self, model, adapter, loader, ctx, epoch=None, split="valid"):
        model.eval()
        adapter.reset_metrics()
        start = time.perf_counter()
        total_count = 0
        loss_sum = 0.0
        pbar_desc = f"[{epoch:2d}/{ctx.epochs:2d}] [{split}]" if epoch is not None else f"[{split}]"
        pbar = tqdm(loader, desc=pbar_desc, leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch in pbar:
                outputs = adapter.eval_step(model, batch, ctx.device)
                adapter.update_metrics(outputs)
                batch_count = adapter.batch_size(batch)
                total_count += batch_count
                if outputs.get("loss") is not None:
                    loss_sum += float(outputs["loss"]) * batch_count
        results = adapter.compute_metrics()
        elapsed = time.perf_counter() - start
        loss_avg = loss_sum / max(total_count, 1)
        if epoch is None:
            valid_str = format_result(results, sep=", ")
            self.log(f"[{split}] {valid_str} ({elapsed:.1f}s)")
        if self.metrics_writer is not None:
            lr = 0.0
            self.metrics_writer.write(
                epoch if epoch is not None else 0, split, loss_avg, results, lr, elapsed
            )
        return results, elapsed

    def predict(self, model, adapter, loader, ctx):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                predictions.extend(adapter.predict_step(model, batch, ctx.device))
        return predictions
