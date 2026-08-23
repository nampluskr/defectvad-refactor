import torch

from src.core.registry import ADAPTERS
from src.tasks.anomaly.models.cfa.loss import CfaLoss
from .base import AnomalyAdapter


class _ImageBatch:
    """Minimal stand-in for anomalib's ``Batch`` -- only the ``.image`` attribute
    that ``CfaModel.initialize_centroid`` reads is needed."""

    __slots__ = ("image",)

    def __init__(self, image):
        self.image = image


def _as_image_batches(loader):
    """Adapt this project's ``(images, targets)`` batches to the ``.image``
    attribute access ``initialize_centroid`` (copied verbatim from anomalib)
    expects, without touching the model file."""
    for images, _targets in loader:
        yield _ImageBatch(images)


@ADAPTERS.register("cfa")
class CfaAdapter(AnomalyAdapter):
    """CFA model adapter.

    CFA trains a descriptor network by gradient descent, but first needs a
    one-time memory-bank centroid computed from a full pass over the training
    set. anomalib does this in ``Cfa.on_train_start`` (Lightning's "once before
    training" hook); this project's equivalent is ``on_fit_start``, which the
    engine also calls exactly once, before the first epoch.
    """

    def __init__(self, loss_fn, metrics, smooth_sigma=0, **params):
        super().__init__(loss_fn, metrics, smooth_sigma=smooth_sigma, **params)
        self.cfa_loss = None
        self._radius = None

    def on_fit_start(self, model, loaders, device):
        """Initialize the memory-bank centroid before any training step.

        Also builds ``CfaLoss`` here, from the already-constructed model's own
        ``num_nearest_neighbors``/``num_hard_negative_features``/``radius``
        attributes rather than a second copy of those values in adapter config.
        A separate adapter-level copy could silently drift from ``model.params``
        (the anomaly map and the training loss would then optimize different
        hyperspheres); reading them off the model makes that impossible, matching
        anomalib's ``Cfa.__init__``, which passes one argument set to both
        ``CfaModel`` and ``CfaLoss``.

        Runs a full forward pass over the training set (no gradients) to seed
        ``model.memory_bank``. ``CfaModel.forward`` raises if this has not run.
        ``KMeans`` (only invoked when ``gamma_c > 1``, not the default) draws from
        NumPy's global RNG, independent of ``torch``'s seeded stream, so no
        ``fork_rng`` guard is needed here.
        """
        super().on_fit_start(model, loaders, device)
        self._radius = float(model.radius)
        self.cfa_loss = CfaLoss(
            num_nearest_neighbors=model.num_nearest_neighbors,
            num_hard_negative_features=model.num_hard_negative_features,
            radius=self._radius,
        )
        model.initialize_centroid(data_loader=_as_image_batches(loaders["train"]))

    def train_step(self, model, batch, device):
        """Compute the CFA hypersphere loss from the model's training-mode output.

        In training mode ``CfaModel.forward`` returns the raw distance tensor
        (not ``InferenceBatch``); ``CfaLoss`` consumes it directly, matching
        anomalib's ``self.loss(distance)``.

        ``CfaLoss.__init__`` builds ``radius`` as ``torch.ones(1, requires_grad=True)
        * radius`` (``loss.py:58``) -- a *non-leaf* tensor carrying a ``grad_fn``,
        created once and then reused by every call to ``forward``. The first
        ``backward()`` frees that shared node, so the second step dies with
        "Trying to backward through the graph a second time". anomalib works
        around this by overriding ``backward()`` with ``retain_graph=True``
        (``lightning_model.py:233``, with a standing TODO asking why it is needed);
        the common engine here has no such per-model hook and retaining the whole
        graph every step would leak memory.

        Rebuilding the tensor per step gives each step its own node instead. This
        is numerically identical: ``radius`` is a plain attribute, never an
        ``nn.Parameter``, never handed to ``build_optimizer``, so no gradient
        accumulated into it was ever applied.
        """
        self.cfa_loss.radius = torch.ones(1, requires_grad=True) * self._radius
        images = batch[0].to(device)
        distance = model(images)
        loss = self.cfa_loss(distance)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
