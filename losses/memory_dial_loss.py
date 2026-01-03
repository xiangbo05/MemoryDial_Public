# losses/memory_dial_loss.py
from __future__ import annotations

"""
Implements the MEMORY DIAL objective from the paper:

Eq. (5):  L_MD(θ; α, τ) = (1-α) * L_std(θ) + α * L_mem(θ; τ)

Notes:
- alpha ∈ [0,1] is the ONLY dial we sweep in main experiments.
- temperature tau is FIXED (tau=0.1 in all main experiments).
- This is NOT equivalent to a single 'effective temperature' CE;
  we combine gradients from two distinct softmax geometries by mixing losses.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

from .standard_ce import standard_ce_loss
from .temperature_ce import temperature_sharpened_ce_loss


@dataclass(frozen=True)
class MemoryDialLossOutput:
    loss: torch.Tensor
    loss_std: torch.Tensor
    loss_mem: torch.Tensor
    token_count: torch.Tensor
    alpha: float
    tau: float


def memory_dial_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    tau: float = 0.1,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> MemoryDialLossOutput:
    """
    MEMORY DIAL objective (Eq. 5).

    Args:
      logits: (B, T, V)
      labels: (B, T)
      alpha: α ∈ [0,1]
      tau: τ (fixed to 0.1 in main experiments)
      ignore_index: label value to ignore
      reduction: "mean" or "sum"

    Returns:
      MemoryDialLossOutput with total loss and components.
    """
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0,1]. Got alpha={alpha}")
    if not (tau > 0.0):
        raise ValueError(f"tau must be > 0. Got tau={tau}")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    out_std = standard_ce_loss(
        logits=logits,
        labels=labels,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    out_mem = temperature_sharpened_ce_loss(
        logits=logits,
        labels=labels,
        tau=tau,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    # Convex combination of losses (Eq. 5)
    a = float(alpha)
    loss = (1.0 - a) * out_std.loss + a * out_mem.loss

    return MemoryDialLossOutput(
        loss=loss,
        loss_std=out_std.loss,
        loss_mem=out_mem.loss,
        token_count=out_std.token_count,  # same mask/count
        alpha=a,
        tau=float(tau),
    )

