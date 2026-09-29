from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .planner_contracts import PlannerLogits, PlannerTargets


@dataclass(frozen=True, slots=True)
class PlannerLossConfig:
    """Explicit weights and numerical constants for planner supervision."""

    route_cell_weight: float = 1.0
    route_offset_weight: float = 1.0
    congestion_weight: float = 0.25
    poisson_epsilon: float = 1e-8
    smooth_l1_beta: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "route_cell_weight",
            "route_offset_weight",
            "congestion_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("poisson_epsilon", "smooth_l1_beta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if float(self.poisson_epsilon) != 1e-8:
            raise ValueError("poisson_epsilon is fixed at 1e-8")

    @property
    def cell_weight(self) -> float:
        return self.route_cell_weight

    @property
    def offset_weight(self) -> float:
        return self.route_offset_weight

    @property
    def density_weight(self) -> float:
        return self.congestion_weight


@dataclass(frozen=True, slots=True)
class PlannerLossComponents:
    total: Tensor
    route_cell: Tensor
    route_offset: Tensor
    congestion: Tensor
    route_cell_count: int
    route_offset_count: int
    congestion_horizon_count: int

    @property
    def total_loss(self) -> Tensor:
        return self.total

    @property
    def cell(self) -> Tensor:
        return self.route_cell

    @property
    def offset(self) -> Tensor:
        return self.route_offset

    @property
    def density(self) -> Tensor:
        return self.congestion

    @property
    def route_cell_loss(self) -> Tensor:
        return self.route_cell

    @property
    def route_offset_loss(self) -> Tensor:
        return self.route_offset

    @property
    def congestion_loss(self) -> Tensor:
        return self.congestion

    @property
    def route_cell_valid_count(self) -> int:
        return self.route_cell_count

    @property
    def route_offset_valid_count(self) -> int:
        return self.route_offset_count

    @property
    def congestion_valid_count(self) -> int:
        return self.congestion_horizon_count

    @property
    def cell_valid_count(self) -> int:
        return self.route_cell_count

    @property
    def offset_valid_count(self) -> int:
        return self.route_offset_count

    @property
    def counts(self) -> dict[str, int]:
        return {
            "route_cell": self.route_cell_count,
            "route_offset": self.route_offset_count,
            "congestion_horizon": self.congestion_horizon_count,
        }


@dataclass(frozen=True, slots=True)
class PlannerRouteLossComponents:
    route_cell: Tensor
    route_offset: Tensor
    route_cell_count: int
    route_offset_count: int


def _connected_zero(reference: Tensor) -> Tensor:
    return reference.float().sum() * 0.0


def compute_planner_route_loss(
    logits: PlannerLogits,
    targets: PlannerTargets,
    config: PlannerLossConfig | None = None,
) -> PlannerRouteLossComponents:
    """Compute only route losses; congestion is deliberately untouched."""

    weights = config or PlannerLossConfig()
    q = targets.route_cells.shape[0]
    if tuple(logits.cell_logits.shape) != (q, 12, 1024):
        raise ValueError("cell_logits must have shape [Q,12,1024]")
    if tuple(logits.offsets.shape) != (q, 12, 2):
        raise ValueError("offsets must have shape [Q,12,2]")
    if logits.query_mask.dtype != torch.bool or logits.query_mask.ndim != 3:
        raise ValueError("query_mask must be a bool [B,T,N] tensor")
    if int(logits.query_mask.sum().item()) != q:
        raise ValueError("query_mask true count must equal PlannerTargets Q")
    device = logits.cell_logits.device
    if logits.offsets.device != device:
        raise ValueError("all route logits must share a device")
    route_mask = targets.route_mask.to(device=device, dtype=torch.bool)
    route_cells = targets.route_cells.to(device=device, dtype=torch.int64)
    cell_count = int(route_mask.sum().item())
    if cell_count:
        terms = -F.log_softmax(logits.cell_logits.float(), dim=-1).gather(
            -1, route_cells.unsqueeze(-1)
        ).squeeze(-1)
        route_cell = terms.masked_select(route_mask).sum() / cell_count
        offset_terms = F.smooth_l1_loss(
            logits.offsets.float(),
            targets.route_offsets.to(device=device, dtype=torch.float32),
            reduction="none",
            beta=float(weights.smooth_l1_beta),
        )
        route_offset = offset_terms.masked_select(
            route_mask.unsqueeze(-1)
        ).sum() / (2 * cell_count)
    else:
        route_cell = _connected_zero(logits.cell_logits)
        route_offset = _connected_zero(logits.offsets)
    return PlannerRouteLossComponents(
        route_cell=route_cell,
        route_offset=route_offset,
        route_cell_count=cell_count,
        route_offset_count=cell_count,
    )


def compute_planner_loss(
    logits: PlannerLogits,
    targets: PlannerTargets,
    config: PlannerLossConfig | None = None,
) -> PlannerLossComponents:
    """Compute the three independently normalized planner losses in FP32."""

    weights = config or PlannerLossConfig()
    q = targets.route_cells.shape[0]
    if tuple(logits.cell_logits.shape) != (q, 12, 1024):
        raise ValueError("cell_logits must have shape [Q,12,1024]")
    if tuple(logits.offsets.shape) != (q, 12, 2):
        raise ValueError("offsets must have shape [Q,12,2]")
    if tuple(logits.density_logits.shape) != (q, 12, 32, 32):
        raise ValueError("density_logits must have shape [Q,12,32,32]")
    if tuple(logits.density.shape) != tuple(logits.density_logits.shape):
        raise ValueError("density must have the same shape as density_logits")
    if logits.query_mask.dtype != torch.bool or logits.query_mask.ndim != 3:
        raise ValueError("query_mask must be a bool [B,T,N] tensor")
    if int(logits.query_mask.sum().item()) != q:
        raise ValueError("query_mask true count must equal PlannerTargets Q")
    devices = {
        logits.cell_logits.device,
        logits.offsets.device,
        logits.density_logits.device,
    }
    if len(devices) != 1:
        raise ValueError("all planner logits must share a device")
    device = logits.cell_logits.device

    cell_logits = logits.cell_logits.float()
    route_cells = targets.route_cells.to(device=device, dtype=torch.int64)
    route_mask = targets.route_mask.to(device=device, dtype=torch.bool)
    cell_count = int(route_mask.sum().item())
    if cell_count:
        log_probabilities = F.log_softmax(cell_logits, dim=-1)
        cell_terms = -log_probabilities.gather(
            -1, route_cells.unsqueeze(-1)
        ).squeeze(-1)
        route_cell = cell_terms.masked_select(route_mask).sum() / max(
            cell_count, 1
        )
    else:
        route_cell = _connected_zero(logits.cell_logits)

    predicted_offsets = logits.offsets.float()
    target_offsets = targets.route_offsets.to(
        device=device, dtype=torch.float32
    )
    # A resolved waypoint always supervises both its cell and its exact
    # within-cell offset.  There are no terminal route tokens.
    offset_mask = route_mask
    offset_count = int(offset_mask.sum().item())
    if offset_count:
        offset_terms = F.smooth_l1_loss(
            predicted_offsets,
            target_offsets,
            reduction="none",
            beta=float(weights.smooth_l1_beta),
        )
        route_offset = offset_terms.masked_select(
            offset_mask.unsqueeze(-1)
        ).sum() / max(2 * offset_count, 1)
    else:
        route_offset = _connected_zero(logits.offsets)

    density_logits = logits.density_logits.float()
    congestion_targets = targets.congestion_targets.to(
        device=device, dtype=torch.float32
    )
    congestion_mask = targets.congestion_mask.to(
        device=device, dtype=torch.bool
    )
    congestion_count = int(congestion_mask.sum().item())
    if congestion_count:
        rate = F.softplus(density_logits)
        poisson_terms = rate - congestion_targets * torch.log(
            rate + float(weights.poisson_epsilon)
        )
        congestion = poisson_terms.masked_select(
            congestion_mask[:, :, None, None]
        ).sum() / max(32 * 32 * congestion_count, 1)
    else:
        congestion = _connected_zero(logits.density_logits)

    total = (
        float(weights.route_cell_weight) * route_cell
        + float(weights.route_offset_weight) * route_offset
        + float(weights.congestion_weight) * congestion
    )
    return PlannerLossComponents(
        total=total,
        route_cell=route_cell,
        route_offset=route_offset,
        congestion=congestion,
        route_cell_count=cell_count,
        route_offset_count=offset_count,
        congestion_horizon_count=congestion_count,
    )


planner_loss = compute_planner_loss
compute_planner_losses = compute_planner_loss


__all__ = [
    "PlannerLossComponents",
    "PlannerLossConfig",
    "PlannerRouteLossComponents",
    "compute_planner_loss",
    "compute_planner_losses",
    "compute_planner_route_loss",
    "planner_loss",
]
