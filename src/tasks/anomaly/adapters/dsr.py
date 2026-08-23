from pathlib import Path
import torch

from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.components.generators.perlin import PerlinAnomalyGenerator
from src.tasks.anomaly.models.dsr.anomaly_generator import DsrAnomalyGenerator
from src.tasks.anomaly.models.dsr.loss import DsrSecondStageLoss, DsrThirdStageLoss
from .base import AnomalyAdapter


@ADAPTERS.register("dsr")
class DsrAdapter(AnomalyAdapter):
    """DSR adapter with 2-phase training schedule and dual optimizers."""

    def __init__(
        self,
        loss_fn=None,
        metrics=None,
        smooth_sigma=4.0,
        upsampling_train_ratio=0.7,
        lr=0.0002,
        gradient_clip_val=1.0,
        weights_path=None,
        **params,
    ):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.upsampling_train_ratio = upsampling_train_ratio
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.weights_path = weights_path

        self.quantized_anomaly_generator = DsrAnomalyGenerator()
        self.perlin_generator = PerlinAnomalyGenerator()
        self.second_stage_loss = DsrSecondStageLoss()
        self.third_stage_loss = DsrThirdStageLoss()

        self.optimizer_d = None
        self.scheduler_d = None
        self.optimizer_u = None
        self.second_phase_epoch = 0
        self.current_epoch = 1
        self.phase_switched = False

    def on_fit_start(self, model, loaders, device):
        super().on_fit_start(model, loaders, device)

        if self.weights_path is not None:
            model.load_pretrained_discrete_model_weights(Path(self.weights_path), device=device)

        # Set discrete latent model to eval mode and frozen
        model.discrete_latent_model.eval()
        for p in model.discrete_latent_model.parameters():
            p.requires_grad = False

        total_epochs = getattr(self, "total_epochs", 1)
        self.second_phase_epoch = int(total_epochs * self.upsampling_train_ratio)
        anneal_epoch = max(1, int(0.8 * self.second_phase_epoch))

        self.optimizer_d = torch.optim.Adam(
            params=list(model.image_reconstruction_network.parameters())
            + list(model.subspace_restriction_module_hi.parameters())
            + list(model.subspace_restriction_module_lo.parameters())
            + list(model.anomaly_detection_module.parameters()),
            lr=self.lr,
        )
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.optimizer_d,
            step_size=anneal_epoch,
            gamma=0.1,
        )

        self.optimizer_u = torch.optim.Adam(
            params=model.upsampling_module.parameters(),
            lr=self.lr,
        )

    def on_epoch_start(self, model, epoch):
        self.current_epoch = epoch
        if self.current_epoch > self.second_phase_epoch and not self.phase_switched:
            for p in model.image_reconstruction_network.parameters():
                p.requires_grad = False
            for p in model.subspace_restriction_module_hi.parameters():
                p.requires_grad = False
            for p in model.subspace_restriction_module_lo.parameters():
                p.requires_grad = False
            for p in model.anomaly_detection_module.parameters():
                p.requires_grad = False

            for p in model.upsampling_module.parameters():
                p.requires_grad = True

            self.phase_switched = True

    def train_step(self, model, batch, device):
        if self.current_epoch <= self.second_phase_epoch:
            # Phase 1: Subspace restriction + Anomaly detection
            input_image = batch[0].to(device)
            anomaly_mask = self.quantized_anomaly_generator.augment_batch(input_image)
            model_outputs = model(input_image, anomaly_mask)
            loss = self.second_stage_loss(
                model_outputs["recon_feat_hi"],
                model_outputs["recon_feat_lo"],
                model_outputs["embedding_bot"],
                model_outputs["embedding_top"],
                input_image,
                model_outputs["obj_spec_image"],
                model_outputs["anomaly_map"],
                model_outputs["true_anomaly_map"],
            )
            self.optimizer_d.zero_grad()
            loss.backward()
            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.image_reconstruction_network.parameters())
                    + list(model.subspace_restriction_module_hi.parameters())
                    + list(model.subspace_restriction_module_lo.parameters())
                    + list(model.anomaly_detection_module.parameters()),
                    max_norm=self.gradient_clip_val,
                )
            self.optimizer_d.step()
        else:
            # Phase 2: Upsampling module
            input_image = batch[0].to(device)
            input_image, anomaly_maps = self.perlin_generator(input_image)
            model_outputs = model(input_image)
            loss = self.third_stage_loss(
                model_outputs["anomaly_map"],
                anomaly_maps,
            )
            self.optimizer_u.zero_grad()
            loss.backward()
            if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.upsampling_module.parameters(),
                    max_norm=self.gradient_clip_val,
                )
            self.optimizer_u.step()

        dummy_loss = sum(
            (0.0 * p.sum() for p in model.parameters()),
            torch.zeros([], device=device),
        )
        return {
            "loss": dummy_loss,
            "loss_dict": {"loss": float(loss.item())},
        }

    def on_epoch_end(self, model, epoch, valid_results=None):
        super().on_epoch_end(model, epoch, valid_results)
        if self.current_epoch <= self.second_phase_epoch and self.scheduler_d is not None:
            self.scheduler_d.step()
