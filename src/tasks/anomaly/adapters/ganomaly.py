import torch

from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.ganomaly.loss import DiscriminatorLoss, GeneratorLoss
from .base import AnomalyAdapter


@ADAPTERS.register("ganomaly")
class GanomalyAdapter(AnomalyAdapter):
    """GANomaly adapter with dual optimizer for generator and discriminator."""

    def __init__(
        self,
        loss_fn=None,
        metrics=None,
        smooth_sigma=4.0,
        wadv=1,
        wcon=50,
        wenc=1,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        **params,
    ):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.generator_loss = GeneratorLoss(wadv=self.wadv, wcon=self.wcon, wenc=self.wenc)
        self.discriminator_loss = DiscriminatorLoss()
        self.optimizer_g = None
        self.optimizer_d = None

    def on_fit_start(self, model, loaders, device):
        super().on_fit_start(model, loaders, device)
        self.optimizer_d = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        self.optimizer_g = torch.optim.Adam(
            model.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        padded, fake, latent_i, latent_o = model(images)
        pred_real, _ = model.discriminator(padded)

        # generator update
        pred_fake, _ = model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)

        self.optimizer_g.zero_grad()
        g_loss.backward(retain_graph=True)
        self.optimizer_g.step()

        # discriminator update
        pred_fake, _ = model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)

        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        total_loss = float(g_loss.item() + d_loss.item())
        dummy_loss = sum(
            (0.0 * p.sum() for p in model.parameters()),
            torch.zeros([], device=device),
        )
        return {
            "loss": dummy_loss,
            "loss_dict": {
                "loss": total_loss,
                "generator_loss": float(g_loss.item()),
                "discriminator_loss": float(d_loss.item()),
            },
        }
