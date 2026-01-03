# losses/standard_ce.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    token_count: torch.Tensor


def _shift_for_causal_lm(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shift logits and labels for causal LM:
      - predict token t from prefix < t
      - drop last logit, drop first label
    logits: (B, T, V)
    labels: (B, T)
    returns:
      shift_logits: (B, T-1, V)
      shift_labels: (B, T-1)
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B,T,V). Got shape={tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be (B,T). Got shape={tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        raise ValueError(
            f"Batch/time dims must match. logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )

    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def standard_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> LossOutput:
    """
    Standard autoregressive cross-entropy loss (Eq. 2 in the paper).
    This is the vanilla LM objective with NO alpha and NO temperature scaling.

    Args:
      logits: (B, T, V)
      labels: (B, T) with ignore_index positions masked out
      ignore_index: label value to ignore
      reduction: "mean" (default) returns mean over non-ignored tokens,
                 "sum" returns sum over non-ignored tokens

    Returns:
      LossOutput(loss=..., token_count=...)
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    shift_logits, shift_labels = _shift_for_causal_lm(logits, labels)

    # Flatten
    B, Tm1, V = shift_logits.shape
    flat_logits = shift_logits.view(B * Tm1, V)
    flat_labels = shift_labels.view(B * Tm1)

    # Count valid tokens
    valid = (flat_labels != ignore_index)
    token_count = valid.sum().to(dtype=torch.long)

    # If everything is ignored, return 0 (safe for distributed / fp16)
    if token_count.item() == 0:
        zero = flat_logits.sum() * 0.0
        return LossOutput(loss=zero, token_count=token_count)

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    if reduction == "sum":
        return LossOutput(loss=loss, token_count=token_count)
    else:
        # "mean" in PyTorch already averages over non-ignored tokens.
        return LossOutput(loss=loss, token_count=token_count)
