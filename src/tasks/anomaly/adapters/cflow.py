import einops
import torch
from torch.nn import functional as F

from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.cflow.utils import get_logp, positional_encoding_2d
from .base import AnomalyAdapter


@ADAPTERS.register("cflow")
class CflowAdapter(AnomalyAdapter):
    """CFLOW model adapter.

    Upstream trains with Lightning's ``automatic_optimization=False``: each fiber
    (a random mini-batch of feature vectors) gets its own zero_grad/backward/step
    call, so one image batch triggers many small decoder updates (~168 Adam steps
    for the default wide_resnet50_2 / 256x256 / batch-size-8 configuration). The
    common engine instead performs exactly one zero_grad/backward/step per
    ``train_step`` call, around whatever "loss" the adapter returns.

    To keep the exact upstream per-fiber optimization schedule, this adapter owns
    a private Adam optimizer over the decoder parameters and does the per-fiber
    zero_grad/backward/step itself inside ``train_step`` -- the real training
    happens here. The "loss" handed back to the engine is a zero tied into the
    decoder graph via ``0.0 * parameter``, so the engine's own optimizer (built
    separately by ``build_optimizer`` over the same decoder parameters) always
    sees an exact-zero gradient: its Adam moment estimates never leave zero and
    its step is a true no-op, so it cannot interfere with the updates already
    applied above.

    Known limitations of this design (adversarial review round 2):
    - ``--resume`` does not preserve this adapter's private Adam momentum: only
      the (always-inert) engine optimizer is checkpointed by src/core/checkpoint.py,
      so a resumed run restarts the decoder optimizer's moments from zero. Model
      weights and RNG state still resume correctly. Fixing this needs an
      adapter-state checkpoint hook in the common engine, which is out of scope
      for a single-model port (NFR-005).
    - ``runtime.amp`` and ``train.grad_clip`` do not govern the real per-fiber
      updates above -- they only apply to the inert dummy loss the engine sees.
      This matches upstream's own ``trainer_arguments`` (``gradient_clip_val=0``,
      i.e. no clipping); enabling AMP for CFLOW in this project's config has no
      effect on the actual decoder training.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, lr=0.0001, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.lr = lr
        self.decoder_optimizer = None

    def on_fit_start(self, model, loaders, device):
        super().on_fit_start(model, loaders, device)
        decoder_params = [p for decoder in model.decoders for p in decoder.parameters()]
        self.decoder_optimizer = torch.optim.Adam(decoder_params, lr=self.lr)

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        activation = model.encoder(images)

        loss_sum = 0.0
        loss_count = 0
        for layer_idx, layer in enumerate(model.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW
            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            embedding_length = batch_size * im_height * im_width

            pos_encoding = einops.repeat(
                positional_encoding_2d(model.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w -> (tile b) c h w",
                tile=batch_size,
            ).to(device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")  # BHWxP
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")  # BHWxC
            # Sampled on CPU (upstream never passes device=), then moved to match
            # c_r/e_r -- keeps this on the same RNG stream anomalib itself uses.
            perm = torch.randperm(embedding_length).to(device)
            decoder = model.decoders[layer_idx]

            fiber_batches = embedding_length // model.fiber_batch_size
            if fiber_batches <= 0:
                msg = "Make sure we have enough fibers, otherwise decrease fiber_batch_size or increase batch size!"
                raise ValueError(msg)

            for batch_num in range(fiber_batches):
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(
                        batch_num * model.fiber_batch_size,
                        (batch_num + 1) * model.fiber_batch_size,
                        device=device,
                    )
                else:
                    idx = torch.arange(batch_num * model.fiber_batch_size, embedding_length, device=device)
                c_p = c_r[perm[idx]]  # NxP
                e_p = e_r[perm[idx]]  # NxC
                self.decoder_optimizer.zero_grad()
                p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                loss = -F.logsigmoid(log_prob)
                loss.mean().backward()
                self.decoder_optimizer.step()
                loss_sum += float(loss.sum())
                loss_count += loss.numel()

        # Leave no leftover gradient from the last fiber: the engine's own
        # backward (on the zero dummy loss below) accumulates into .grad rather
        # than replacing it, so a stale nonzero .grad here would leak into the
        # engine's optimizer step.
        self.decoder_optimizer.zero_grad()

        avg_loss = loss_sum / max(loss_count, 1)
        dummy_loss = sum(
            (0.0 * parameter.sum() for decoder in model.decoders for parameter in decoder.parameters()),
            torch.zeros([], device=device),
        )
        return {"loss": dummy_loss, "loss_dict": {"loss": avg_loss}}
