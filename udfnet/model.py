"""UDFNet: Uncertainty-aware Dynamic Fusion Network (Wang et al., Information Fusion 2025).

Unofficial reimplementation written from the paper's text (Sections 3.1-3.3, Algorithm 1).
Where the paper leaves a detail open, the choice made here is marked "ASSUMPTION" and is
listed in README.md.

Pipeline (per Fig. 1):
    x_hsi, x_rad
      -> FeatureExtractionPropagation (AsyFFNet-style: per-modality stem conv, shared residual
         blocks with modality-specific BatchNorm and BN-scale-driven channel exchange)
      -> ImprovedCrissCrossAttention (one per modality)
      -> classifier head (one per modality)  -> unimodal logits g^m
      -> energy score u^m = -tau * logsumexp(g^m / tau)               (Eq. 17)
      -> fusion weight lambda^m = alpha^m * u^m + beta^m, alpha<0, beta>=0   (Eq. 16)
      -> fused logits f = sum_m lambda^m * g^m                         (Eq. 2)
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3.1 Feature extraction and propagation (AsyFFNet-style)
# ---------------------------------------------------------------------------


class SharedResidualBlock(nn.Module):
    """Residual block whose convolutions are shared by all modalities.

    Each modality has its own BatchNorm layers ("independent scale factors"). After the second
    BatchNorm, channels whose BN scale |gamma| falls below `exchange_threshold` are treated as
    redundant and replaced by the same channel from the other modality (asymmetric feature
    propagation). As in channel-exchanging networks, each modality is only allowed to exchange
    a disjoint half of the channels, and the L1 sparsity penalty is applied to that half.
    """

    def __init__(self, channels: int, n_modalities: int = 2, exchange_threshold: float = 2e-2):
        super().__init__()
        if n_modalities != 2:
            raise ValueError("Channel exchange is implemented for exactly two modalities.")
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(n_modalities))
        self.bn2 = nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(n_modalities))
        self.exchange_threshold = exchange_threshold
        half = channels // 2
        eligible = torch.zeros(n_modalities, channels, dtype=torch.bool)
        eligible[0, :half] = True  # HSI may receive radar features in the first half
        eligible[1, half:] = True  # radar may receive HSI features in the second half
        self.register_buffer("eligible", eligible, persistent=False)

    def exchange(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for m, x in enumerate(feats):
            other = feats[1 - m]
            gamma = self.bn2[m].weight.detach().abs()
            mask = (gamma < self.exchange_threshold) & self.eligible[m]
            mask = mask.view(1, -1, 1, 1)
            out.append(torch.where(mask, other, x))
        return out

    def sparsity_penalty(self) -> torch.Tensor:
        """L1 on the exchangeable BN scale factors (drives redundant channels toward zero)."""
        terms = [self.bn2[m].weight[self.eligible[m]].abs().sum() for m in range(len(self.bn2))]
        return torch.stack(terms).sum()

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        h = [F.relu(self.bn1[m](self.conv1(x))) for m, x in enumerate(feats)]
        h = [self.bn2[m](self.conv2(x)) for m, x in enumerate(h)]
        h = self.exchange(h)
        return [F.relu(a + b) for a, b in zip(h, feats)]


class FeatureExtractionPropagation(nn.Module):
    """f_FEP in Eq. (8): un-shared stem convs for dimension alignment + shared residual blocks."""

    def __init__(self, in_channels: Sequence[int], channels: int = 64, n_blocks: int = 2,
                 exchange_threshold: float = 2e-2):
        super().__init__()
        self.stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        )
        self.blocks = nn.ModuleList(
            SharedResidualBlock(channels, len(in_channels), exchange_threshold) for _ in range(n_blocks)
        )

    def sparsity_penalty(self) -> torch.Tensor:
        return torch.stack([b.sparsity_penalty() for b in self.blocks]).sum()

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        feats = [stem(x) for stem, x in zip(self.stems, xs)]
        for block in self.blocks:
            feats = block(feats)
        return feats


# ---------------------------------------------------------------------------
# 3.2 Global feature association: improved Criss-Cross Attention
# ---------------------------------------------------------------------------


class ImprovedCrissCrossAttention(nn.Module):
    """Criss-Cross Attention with multi-head attention, learnable positional encodings on Q and K,
    a learnable residual scale eta, and LayerNorm (Eqs. 9-15).

    Every position attends to the 2S-1 positions in its own row and column (its own position is
    counted once).
    """

    def __init__(self, channels: int, patch_size: int, heads: int = 4, reduction: int = 8):
        super().__init__()
        qk_per_head = max(1, channels // reduction // heads)
        self.heads = heads
        self.qk_channels = qk_per_head * heads  # C' (reduced dimension)
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.query = nn.Conv2d(channels, self.qk_channels, 1)
        self.key = nn.Conv2d(channels, self.qk_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.pos_q = nn.Parameter(torch.zeros(1, self.qk_channels, patch_size, patch_size))
        self.pos_k = nn.Parameter(torch.zeros(1, self.qk_channels, patch_size, patch_size))
        nn.init.trunc_normal_(self.pos_q, std=0.02)
        nn.init.trunc_normal_(self.pos_k, std=0.02)
        self.proj = nn.Conv2d(channels, channels, 1)  # "restored to the original dimension"
        self.eta = nn.Parameter(torch.zeros(1))  # ASSUMPTION: initialised at 0, as in CCNet
        self.norm = nn.LayerNorm(channels)
        self.scale = 1.0 / math.sqrt(qk_per_head)  # d_k = C'/h

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention over [column (H) | row (W)] positions, shape (B, h, H, W, H+W)."""
        b, _, height, width = x.shape
        q = (self.query(x) + self.pos_q).view(b, self.heads, -1, height, width)
        k = (self.key(x) + self.pos_k).view(b, self.heads, -1, height, width)
        e_col = torch.einsum("bhdiw,bhdjw->bhiwj", q, k)  # same column, every row j
        e_row = torch.einsum("bhdiw,bhdiv->bhiwv", q, k)  # same row, every column v
        self_mask = torch.eye(height, dtype=torch.bool, device=x.device).view(height, 1, height)
        e_col = e_col.masked_fill(self_mask, float("-inf"))  # count own position only once
        return torch.softmax(torch.cat([e_col, e_row], dim=-1) * self.scale, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, height, width = z.shape
        if (height, width) != tuple(self.pos_q.shape[-2:]):
            raise ValueError(f"Expected {tuple(self.pos_q.shape[-2:])} patches, got {(height, width)}")
        att = self.attention_weights(z)
        a_col, a_row = att[..., :height], att[..., height:]
        v = self.value(z).view(b, self.heads, -1, height, width)
        o = torch.einsum("bhiwj,bhdjw->bhdiw", a_col, v) + torch.einsum("bhiwv,bhdiv->bhdiw", a_row, v)
        o = self.proj(o.reshape(b, c, height, width))
        out = z + self.eta * o
        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# 3.3 Uncertainty-aware dynamic fusion
# ---------------------------------------------------------------------------


class ClassifierHead(nn.Module):
    """ASSUMPTION: global average pooling + dropout + linear layer (the paper does not specify)."""

    def __init__(self, channels: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z.mean(dim=(2, 3)))


def energy_score(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Eq. (17): u = -tau * log sum_k exp(g_k / tau). Lower energy = more confident."""
    return -tau * torch.logsumexp(logits / tau, dim=1)


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class UDFNet(nn.Module):
    """Full UDFNet. Call with a list [x_hsi, x_rad] of (B, C_m, S, S) tensors."""

    def __init__(
        self,
        in_channels: Sequence[int],
        n_classes: int,
        patch_size: int,
        channels: int = 128,
        n_blocks: int = 2,
        heads: int = 4,
        tau: float = 1.0,
        exchange_threshold: float = 2e-2,
        dropout: float = 0.2,
        detach_uncertainty: bool = True,
        alpha_init: float = 0.1,
        beta_init: float = 0.1,
    ):
        super().__init__()
        n_mod = len(in_channels)
        self.tau = tau
        self.detach_uncertainty = detach_uncertainty
        self.fep = FeatureExtractionPropagation(in_channels, channels, n_blocks, exchange_threshold)
        self.cca = nn.ModuleList(ImprovedCrissCrossAttention(channels, patch_size, heads) for _ in range(n_mod))
        self.heads = nn.ModuleList(ClassifierHead(channels, n_classes, dropout) for _ in range(n_mod))
        # alpha = -softplus(a) < 0 and beta = softplus(b) >= 0 enforce the paper's sign constraints.
        self.alpha_raw = nn.Parameter(torch.full((n_mod,), _inverse_softplus(alpha_init)))
        self.beta_raw = nn.Parameter(torch.full((n_mod,), _inverse_softplus(beta_init)))

    @property
    def alpha(self) -> torch.Tensor:
        return -F.softplus(self.alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)

    def sparsity_penalty(self) -> torch.Tensor:
        return self.fep.sparsity_penalty()

    def forward(self, xs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self.fep(xs)
        feats = [cca(z) for cca, z in zip(self.cca, feats)]
        logits = [head(z) for head, z in zip(self.heads, feats)]
        energies = torch.stack([energy_score(g, self.tau) for g in logits], dim=0)  # (M, B)
        u = energies.detach() if self.detach_uncertainty else energies
        # Eq. (16). ASSUMPTION: clamp at 0 so a modality can be silenced but never vote "against".
        lambdas = (self.alpha.view(-1, 1) * u + self.beta.view(-1, 1)).clamp_min(0.0)  # (M, B)
        fused = sum(lam.unsqueeze(1) * g for lam, g in zip(lambdas, logits))  # Eq. (2)
        return {
            "fused": fused,
            "unimodal": torch.stack(logits, dim=0),  # (M, B, K)
            "lambdas": lambdas,
            "energies": energies,
        }
