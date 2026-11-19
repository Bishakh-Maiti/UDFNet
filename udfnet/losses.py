"""Training objective of UDFNet (Eqs. 18-21).

L_overall = CE(y, fused) + sum_m CE(y, g^m) + gamma * L_extra   (+ sparsity on BN scales, AsyFFNet)

L_extra is a ranking regularizer that uses each training sample's *historical* unimodal loss
(averaged over past epochs, Eq. 18): if sample i has historically been harder than sample j for
modality m, its fusion weight lambda_i^m should be lower than lambda_j^m.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class LossHistory:
    """Per-sample, per-modality running mean of the training loss (Eq. 18).

    Losses are only recorded from epoch `start_epoch` (L_s in the paper) onward. A sample's mean
    becomes usable once it has been seen at least once in that window.
    """

    def __init__(self, n_samples: int, n_modalities: int, start_epoch: int = 5, device="cpu"):
        self.start_epoch = start_epoch
        self.total = torch.zeros(n_modalities, n_samples, device=device)
        self.count = torch.zeros(n_samples, device=device)

    def to(self, device):
        self.total, self.count = self.total.to(device), self.count.to(device)
        return self

    @torch.no_grad()
    def update(self, epoch: int, idx: torch.Tensor, unimodal_losses: torch.Tensor) -> None:
        """unimodal_losses: (M, B) per-sample losses for the batch."""
        if epoch < self.start_epoch:
            return
        self.total[:, idx] += unimodal_losses.detach()
        self.count[idx] += 1

    def get(self, idx: torch.Tensor):
        count = self.count[idx]
        mean = self.total[:, idx] / count.clamp_min(1)
        return mean, count > 0


def ranking_regularizer(lambdas: torch.Tensor, hist_loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """L_extra: margin ranking loss between each sample and a partner sample in the batch.

    For a pair (i, j) and modality m, with historical losses l_i, l_j:
        loss = max(0, sign(l_i - l_j) * (lambda_i - lambda_j) + |l_i - l_j|)
    i.e. the historically worse sample must receive a lower weight, by a margin |l_i - l_j|.

    ASSUMPTION / DEVIATION: Eq. (19) as printed puts the sign on the lambdas and the margin on
    the weights, which leaves no useful gradient (the sign and the history carry no gradient).
    This is the correctness-ranking form used by QMF (ICML 2023), which the paper builds on.
    """
    b = lambdas.shape[1]
    if b < 2:
        return lambdas.sum() * 0.0
    partner = torch.roll(torch.arange(b, device=lambdas.device), shifts=-1)
    li, lj = lambdas, lambdas[:, partner]
    hi, hj = hist_loss, hist_loss[:, partner]
    pair_valid = (valid & valid[partner]).unsqueeze(0) & (hi != hj)
    if not pair_valid.any():
        return lambdas.sum() * 0.0
    sign = torch.sign(hi - hj)
    loss = F.relu(sign * (li - lj) + (hi - hj).abs())
    return loss[pair_valid].mean()


def udfnet_loss(
    out: Dict[str, torch.Tensor],
    target: torch.Tensor,
    idx: torch.Tensor,
    epoch: int,
    history: LossHistory,
    gamma: float = 0.1,
    sparsity_weight: float = 0.0,
    sparsity_term: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    fused_ce = F.cross_entropy(out["fused"], target)
    uni = out["unimodal"]  # (M, B, K)
    uni_per_sample = torch.stack([F.cross_entropy(g, target, reduction="none") for g in uni], dim=0)
    uni_ce = uni_per_sample.mean(dim=1).sum()

    hist, valid = history.get(idx)
    reg = ranking_regularizer(out["lambdas"], hist, valid)
    history.update(epoch, idx, uni_per_sample)

    total = fused_ce + uni_ce + gamma * reg
    if sparsity_term is not None and sparsity_weight > 0:
        total = total + sparsity_weight * sparsity_term
    return {"total": total, "fused_ce": fused_ce.detach(), "uni_ce": uni_ce.detach(), "reg": reg.detach()}
