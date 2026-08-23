import torch
from torchmetrics import Metric


class RankAUROC(Metric):
    """Binary AUROC computed from ranks, without torchmetrics' input squashing.

    ``torchmetrics.classification.BinaryAUROC`` pushes ``preds`` through a sigmoid
    whenever any value falls outside ``[0, 1]``. AUROC depends only on the ranking,
    so a monotone squash is normally harmless -- but float32 ``sigmoid`` saturates
    to exactly ``1.0`` above roughly ``x = 17``. Once that happens every prediction
    becomes one tie and the result is exactly ``0.5``, regardless of how good the
    model is.

    CS-Flow hits this: its anomaly map is the product of three ``mean(z**2)`` terms
    and lands in the hundreds-to-thousands range, so every pixel collapsed to 1.0
    and ``pixel_auroc`` read exactly ``0.500`` on every epoch. Models whose maps
    stay small never saturated, so their recorded numbers are unaffected by this
    metric -- sigmoid is strictly monotone below saturation and AUROC is invariant
    to it.

    Scores of any magnitude are accepted here and AUROC is computed via the
    Mann-Whitney U statistic with tie-aware average ranks. Accumulated state is
    kept on CPU so a full-resolution pixel metric does not grow GPU memory.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **params):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        self.preds.append(preds.detach().flatten().float().cpu())
        self.target.append(target.detach().flatten().to(torch.uint8).cpu())

    def compute(self):
        if not self.preds:
            return torch.tensor(0.0)

        preds = torch.cat(self.preds)
        target = torch.cat(self.target)

        positive = target == 1
        num_positive = int(positive.sum())
        num_negative = int(preds.numel() - num_positive)
        # Undefined with a single class present; torchmetrics returns 0.0 here too.
        if num_positive == 0 or num_negative == 0:
            return torch.tensor(0.0)

        ranks = self._average_ranks(preds)
        rank_sum = ranks[positive].sum()
        auroc = (rank_sum - num_positive * (num_positive + 1) / 2.0) / (num_positive * num_negative)
        return auroc.float()

    @staticmethod
    def _average_ranks(values):
        """Return 1-based ranks of ``values``, averaging ranks within tied groups."""
        order = torch.argsort(values)
        _, counts = torch.unique_consecutive(values[order], return_counts=True)
        end = torch.cumsum(counts, dim=0)
        start = end - counts
        # Mean of the 1-based positions each tied group occupies.
        group_rank = (start + 1 + end).double() / 2.0

        ranks = torch.empty(values.numel(), dtype=torch.float64)
        ranks[order] = torch.repeat_interleave(group_rank, counts)
        return ranks
