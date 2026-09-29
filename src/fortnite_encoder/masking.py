from __future__ import annotations

import torch
from torch import BoolTensor, Tensor


def masked_softmax(logits: Tensor, eligible: BoolTensor, dim: int = -1) -> Tensor:
    """Softmax with ``True = eligible`` and exact-zero fully masked rows."""

    if eligible.dtype != torch.bool:
        raise TypeError("eligible mask must have bool dtype")
    finite_eligible = eligible & torch.isfinite(logits)
    floor = torch.finfo(logits.dtype).min
    safe_logits = torch.where(finite_eligible, logits, floor)
    probabilities = torch.softmax(safe_logits, dim=dim)
    probabilities = probabilities * finite_eligible.to(probabilities.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True)
    return torch.where(
        normalizer > 0,
        probabilities / normalizer.clamp_min(torch.finfo(probabilities.dtype).tiny),
        torch.zeros_like(probabilities),
    )


def apply_node_mask(values: Tensor, node_mask: BoolTensor) -> Tensor:
    return torch.where(
        node_mask.unsqueeze(-1), values, torch.zeros_like(values)
    )
