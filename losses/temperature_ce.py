# losses/temperature_ce.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    token_count: torch.Tensor


def _shift_for_causal_lm(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B,T,V). Got shape={tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be (B,T). Got shape={tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        raise ValueError(
            f"Batch/time dims must match. logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def temperature_sharpened_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.1,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> LossOutput:
    """
    Temperature-sharpened cross-entropy (Eq. 3-4 in the paper).
    Implements p_theta^(tau)(·|x<t) = softmax(z/tau) and CE w.r.t. the true token.

    IMPORTANT:
      - tau in (0, 1] for "sharpening" as described in the paper.
      - This function ONLY defines L_mem; it does not mix with L_std.

    Args:
      logits: (B, T, V)
      labels: (B, T)
      tau: temperature parameter τ (fixed to 0.1 in main experiments)
      ignore_index: label value to ignore
      reduction: "mean" or "sum"

    Returns:
      LossOutput(loss=..., token_count=...)
    """
    if not (tau > 0.0):
        raise ValueError(f"tau must be > 0. Got tau={tau}")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    shift_logits, shift_labels = _shift_for_causal_lm(logits, labels)

    # Scale logits by tau (sharpen when tau < 1)
    scaled = shift_logits / float(tau)

    B, Tm1, V = scaled.shape
    flat_logits = scaled.view(B * Tm1, V)
    flat_labels = shift_labels.view(B * Tm1)

    valid = (flat_labels != ignore_index)
    token_count = valid.sum().to(dtype=torch.long)

    if token_count.item() == 0:
        zero = flat_logits.sum() * 0.0
        return LossOutput(loss=zero, token_count=token_count)

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    return LossOutput(loss=loss, token_count=token_count)
