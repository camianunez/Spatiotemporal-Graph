from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import RotationLossConfig
from .contracts import RotationLogits, RotationLossOutput, RotationTargets


def _zero_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _masked_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> tuple[Tensor, int]:
    count = int(mask.sum().item())
    if count == 0:
        return _zero_loss(logits), 0
    return F.cross_entropy(logits[mask], labels[mask]), count


def _masked_binary_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> tuple[Tensor, int]:
    count = int(mask.sum().item())
    if count == 0:
        return _zero_loss(logits), 0
    return (
        F.binary_cross_entropy_with_logits(logits[mask], labels[mask]),
        count,
    )


def compute_rotation_loss(
    logits: RotationLogits,
    targets: RotationTargets,
    config: RotationLossConfig | None = None,
) -> RotationLossOutput:
    """Compute independent masked means and the weighted multi-task total."""

    weights = config or RotationLossConfig()
    position_logits = logits.future_position
    if position_logits.ndim != 5:
        raise ValueError("future position logits must have shape [B,T,N,H,C]")
    expected_target_shape = position_logits.shape[:-1]
    if targets.future_position.shape != expected_target_shape:
        raise ValueError("future position labels do not match position logits")
    if targets.future_position_mask.shape != expected_target_shape:
        raise ValueError("future position mask does not match position logits")
    if logits.query_mask.shape != position_logits.shape[:3]:
        raise ValueError("query_mask does not match the logit query axes")

    device = position_logits.device
    query_mask = logits.query_mask.to(device=device, dtype=torch.bool)
    position_labels = targets.future_position.to(device=device, dtype=torch.long)
    position_mask = targets.future_position_mask.to(
        device=device,
        dtype=torch.bool,
    )
    horizon_losses: list[Tensor] = []
    horizon_counts: list[int] = []
    for horizon_index in range(position_logits.shape[-2]):
        horizon_mask = (
            position_mask[..., horizon_index]
            & query_mask
        )
        horizon_loss, horizon_count = _masked_cross_entropy(
            position_logits[..., horizon_index, :],
            position_labels[..., horizon_index],
            horizon_mask,
        )
        horizon_losses.append(horizon_loss)
        horizon_counts.append(horizon_count)
    position_loss = torch.stack(horizon_losses).sum()

    if logits.zone_entry.ndim != 4 or logits.zone_entry.shape[:3] != query_mask.shape:
        raise ValueError("zone entry logits must have shape [B,T,N,C]")
    if targets.zone_entry.shape != query_mask.shape:
        raise ValueError("zone entry labels do not match the query axes")
    if targets.zone_entry_mask.shape != query_mask.shape:
        raise ValueError("zone entry mask does not match the query axes")
    entry_mask = (
        targets.zone_entry_mask.to(device=device, dtype=torch.bool)
        & query_mask
    )
    entry_loss, entry_count = _masked_cross_entropy(
        logits.zone_entry,
        targets.zone_entry.to(device=device, dtype=torch.long),
        entry_mask,
    )

    survival_loss = _zero_loss(position_logits)
    survival_count = 0
    if logits.survival is not None:
        if targets.survival is None or targets.survival_mask is None:
            raise ValueError("enabled survival logits require survival targets")
        if logits.survival.shape != query_mask.shape:
            raise ValueError("survival logits do not match the query axes")
        if (
            targets.survival.shape != query_mask.shape
            or targets.survival_mask.shape != query_mask.shape
        ):
            raise ValueError("survival targets do not match the query axes")
        survival_mask = (
            targets.survival_mask.to(device=device, dtype=torch.bool)
            & query_mask
        )
        survival_loss, survival_count = _masked_binary_cross_entropy(
            logits.survival,
            targets.survival.to(device=device, dtype=logits.survival.dtype),
            survival_mask,
        )

    placement_loss = _zero_loss(position_logits)
    placement_count = 0
    if logits.placement is not None:
        if targets.placement is None or targets.placement_mask is None:
            raise ValueError("enabled placement logits require placement targets")
        if (
            logits.placement.ndim != 4
            or logits.placement.shape[:3] != query_mask.shape
            or logits.placement.shape[-1] != 5
        ):
            raise ValueError("placement logits must have shape [B,T,N,5]")
        if (
            targets.placement.shape != query_mask.shape
            or targets.placement_mask.shape != query_mask.shape
        ):
            raise ValueError("placement targets do not match the query axes")
        placement_mask = (
            targets.placement_mask.to(device=device, dtype=torch.bool)
            & query_mask
        )
        placement_loss, placement_count = _masked_cross_entropy(
            logits.placement,
            targets.placement.to(device=device, dtype=torch.long),
            placement_mask,
        )

    total = (
        position_loss
        + float(weights.entry_weight) * entry_loss
        + float(weights.survival_weight) * survival_loss
        + float(weights.placement_weight) * placement_loss
    )
    return RotationLossOutput(
        total=total,
        future_position=position_loss,
        future_position_by_horizon=tuple(horizon_losses),
        zone_entry=entry_loss,
        survival=survival_loss,
        placement=placement_loss,
        future_position_valid_counts=tuple(horizon_counts),
        zone_entry_valid_count=entry_count,
        survival_valid_count=survival_count,
        placement_valid_count=placement_count,
    )


rotation_loss = compute_rotation_loss
compute_rotation_losses = compute_rotation_loss
