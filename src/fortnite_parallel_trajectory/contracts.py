from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .config import (
    D_MODEL,
    NUM_HORIZONS,
    NUM_ROUTE_MODES,
    PARALLEL_TRAJECTORY_ARCHITECTURE_ID,
    PARALLEL_TRAJECTORY_TARGET_SCHEMA_ID,
    ZONE_STATE_WIDTH,
)

if TYPE_CHECKING:
    from fortnite_encoder.planner_supervision import PlannerQuery
    from fortnite_encoder.world_grid import WorldGridProfile


def _move_tensor_fields(
    instance: object, *args: object, **kwargs: object
) -> dict[str, Any]:
    return {
        field.name: (
            value.to(*args, **kwargs) if isinstance(value, Tensor) else value
        )
        for field in fields(instance)
        for value in (getattr(instance, field.name),)
    }


def _require_float_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _require_bool_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have bool dtype")


@dataclass(frozen=True, slots=True)
class ParallelTrajectoryBatch:
    """The complete, target-free input contract for the parallel decoder.

    ``memory_mask`` uses public validity semantics: ``True`` denotes a causal
    memory token that attention is allowed to read.
    """

    z_query: Tensor
    memory: Tensor
    memory_mask: Tensor
    current_xy: Tensor
    zone_state: Tensor
    congestion_tokens: Tensor
    query_mask: Tensor

    def __post_init__(self) -> None:
        for name in (
            "z_query",
            "memory",
            "current_xy",
            "zone_state",
            "congestion_tokens",
        ):
            _require_float_tensor(name, getattr(self, name))
        _require_bool_tensor("memory_mask", self.memory_mask)
        _require_bool_tensor("query_mask", self.query_mask)

        if self.z_query.ndim != 2 or self.z_query.shape[1] != D_MODEL:
            raise ValueError("z_query must have shape [Q,256]")
        q = self.z_query.shape[0]
        if self.memory.ndim != 3 or tuple(self.memory.shape[::2]) != (q, D_MODEL):
            raise ValueError("memory must have shape [Q,M,256]")
        if tuple(self.memory_mask.shape) != (q, self.memory.shape[1]):
            raise ValueError("memory_mask must have shape [Q,M]")
        if tuple(self.current_xy.shape) != (q, 2):
            raise ValueError("current_xy must have shape [Q,2]")
        if tuple(self.zone_state.shape) != (q, ZONE_STATE_WIDTH):
            raise ValueError(f"zone_state must have shape [Q,{ZONE_STATE_WIDTH}]")
        if (
            self.congestion_tokens.ndim != 3
            or self.congestion_tokens.shape[0] != q
            or self.congestion_tokens.shape[2] != D_MODEL
        ):
            raise ValueError("congestion_tokens must have shape [Q,C,256]")
        if tuple(self.query_mask.shape) != (q,):
            raise ValueError("query_mask must have shape [Q]")
        if q and self.memory.shape[1] == 0:
            raise ValueError("memory must contain at least one token")
        if q and self.congestion_tokens.shape[1] == 0:
            raise ValueError("congestion_tokens must contain at least one token")
        if q and bool((self.memory_mask.sum(dim=-1) == 0).any()):
            raise ValueError("each query must have at least one valid memory token")

        devices = {
            self.z_query.device,
            self.memory.device,
            self.memory_mask.device,
            self.current_xy.device,
            self.zone_state.device,
            self.congestion_tokens.device,
            self.query_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("all parallel trajectory inputs must share a device")
        for name in (
            "z_query",
            "memory",
            "current_xy",
            "zone_state",
            "congestion_tokens",
        ):
            value = getattr(self, name)
            if value.numel() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")

    @property
    def query_state(self) -> Tensor:
        return self.z_query

    @property
    def causal_memory(self) -> Tensor:
        return self.memory

    @property
    def causal_memory_mask(self) -> Tensor:
        return self.memory_mask

    @property
    def zone_vector(self) -> Tensor:
        return self.zone_state

    def to(self, *args: object, **kwargs: object) -> ParallelTrajectoryBatch:
        return ParallelTrajectoryBatch(
            **_move_tensor_fields(self, *args, **kwargs)
        )


@dataclass(frozen=True, slots=True)
class ParallelTrajectoryTargets:
    """Current-relative, cell-normalized full-route supervision.

    Invalid horizons carry no semantic target value.  Builders write zeros in
    those slots and ``target_mask`` is the sole authority for loss inclusion.
    A mask may contain holes for transient coordinate absence.
    """

    target_displacements: Tensor
    target_mask: Tensor
    query_mask: Tensor | None = None
    current_xy: Tensor | None = None
    future_xy: Tensor | None = None
    queries: tuple[PlannerQuery, ...] = ()
    world_grid_profile_id: str | None = None
    world_grid_profile_hash: str | None = None
    target_schema_id: str = PARALLEL_TRAJECTORY_TARGET_SCHEMA_ID

    def __post_init__(self) -> None:
        _require_float_tensor("target_displacements", self.target_displacements)
        _require_bool_tensor("target_mask", self.target_mask)
        if (
            self.target_displacements.ndim != 3
            or tuple(self.target_displacements.shape[1:]) != (NUM_HORIZONS, 2)
        ):
            raise ValueError("target_displacements must have shape [Q,12,2]")
        q = self.target_displacements.shape[0]
        if tuple(self.target_mask.shape) != (q, NUM_HORIZONS):
            raise ValueError("target_mask must have shape [Q,12]")
        if self.query_mask is None:
            object.__setattr__(
                self,
                "query_mask",
                torch.ones(q, dtype=torch.bool, device=self.target_mask.device),
            )
        assert self.query_mask is not None
        _require_bool_tensor("query_mask", self.query_mask)
        if tuple(self.query_mask.shape) != (q,):
            raise ValueError("query_mask must have shape [Q]")
        if torch.any(self.target_mask & ~self.query_mask.unsqueeze(-1)):
            raise ValueError("target_mask may not select an ineligible query")

        for name in ("current_xy", "future_xy"):
            value = getattr(self, name)
            if value is None:
                continue
            _require_float_tensor(name, value)
            expected = (q, 2) if name == "current_xy" else (q, NUM_HORIZONS, 2)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {list(expected)}")

        tensors = [
            self.target_displacements,
            self.target_mask,
            self.query_mask,
            *(value for value in (self.current_xy, self.future_xy) if value is not None),
        ]
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all target tensors must share a device")
        valid_displacements = self.target_displacements.masked_select(
            self.target_mask.unsqueeze(-1)
        )
        if valid_displacements.numel() and not bool(
            torch.isfinite(valid_displacements).all()
        ):
            raise ValueError("valid target displacements must be finite")
        if torch.any(
            self.target_displacements.masked_select(
                ~self.target_mask.unsqueeze(-1)
            )
            != 0
        ):
            raise ValueError("masked target displacements must be zero")
        if self.current_xy is not None:
            valid_current = self.current_xy.masked_select(
                self.query_mask.unsqueeze(-1)
            )
            if valid_current.numel() and not bool(torch.isfinite(valid_current).all()):
                raise ValueError("eligible current coordinates must be finite")
        if self.future_xy is not None:
            valid_future = self.future_xy.masked_select(self.target_mask.unsqueeze(-1))
            if valid_future.numel() and not bool(torch.isfinite(valid_future).all()):
                raise ValueError("valid future coordinates must be finite")
        if self.queries and len(self.queries) != q:
            raise ValueError("queries metadata must align with Q")
        if self.target_schema_id != PARALLEL_TRAJECTORY_TARGET_SCHEMA_ID:
            raise ValueError("unsupported parallel trajectory target schema")

    @property
    def normalized_displacements(self) -> Tensor:
        return self.target_displacements

    @property
    def displacements(self) -> Tensor:
        return self.target_displacements

    @property
    def horizon_mask(self) -> Tensor:
        return self.target_mask

    def to(self, *args: object, **kwargs: object) -> ParallelTrajectoryTargets:
        return ParallelTrajectoryTargets(
            **_move_tensor_fields(self, *args, **kwargs)
        )


def normalized_displacements_to_world(
    normalized_displacements: Tensor,
    current_xy: Tensor,
    profile: WorldGridProfile,
) -> tuple[Tensor, Tensor]:
    """Reconstruct world coordinates without clipping and return bounds validity."""

    from fortnite_encoder.world_grid import WorldGridProfile

    _require_float_tensor("normalized_displacements", normalized_displacements)
    _require_float_tensor("current_xy", current_xy)
    if not isinstance(profile, WorldGridProfile):
        raise TypeError("profile must be a WorldGridProfile")
    if normalized_displacements.ndim < 3 or normalized_displacements.shape[-1] != 2:
        raise ValueError("normalized_displacements must end in [H,2]")
    if current_xy.ndim != 2 or current_xy.shape != (normalized_displacements.shape[0], 2):
        raise ValueError("current_xy must have shape [Q,2]")
    if current_xy.device != normalized_displacements.device:
        raise ValueError("current_xy and displacements must share a device")
    if not bool(torch.isfinite(normalized_displacements).all()) or not bool(
        torch.isfinite(current_xy).all()
    ):
        raise ValueError("coordinate-conversion inputs must be finite")

    prefix = (current_xy.shape[0],) + (1,) * (normalized_displacements.ndim - 2) + (2,)
    origin = current_xy.reshape(prefix).to(normalized_displacements.dtype)
    scale = normalized_displacements.new_tensor(
        [profile.cell_width_world_units, profile.cell_height_world_units]
    )
    world = origin + normalized_displacements * scale
    valid = (
        torch.isfinite(world).all(dim=-1)
        & (world[..., 0] >= float(profile.world_x_min))
        & (world[..., 0] <= float(profile.world_x_max))
        & (world[..., 1] >= float(profile.world_y_min))
        & (world[..., 1] <= float(profile.world_y_max))
    )
    return world, valid


@dataclass(frozen=True, slots=True)
class ParallelTrajectoryOutput:
    mode_logits: Tensor
    mode_probabilities: Tensor
    means: Tensor
    scales: Tensor
    correlations: Tensor
    primary_mode_index: Tensor
    primary_route_normalized: Tensor
    marginal_mean_route_normalized: Tensor
    query_mask: Tensor
    horizon_mask: Tensor | None = None
    current_xy: Tensor | None = None

    def __post_init__(self) -> None:
        for name in (
            "mode_logits",
            "mode_probabilities",
            "means",
            "scales",
            "correlations",
            "primary_route_normalized",
            "marginal_mean_route_normalized",
        ):
            _require_float_tensor(name, getattr(self, name))
        if not isinstance(self.primary_mode_index, Tensor):
            raise TypeError("primary_mode_index must be a tensor")
        if self.primary_mode_index.dtype != torch.int64:
            raise TypeError("primary_mode_index must have int64 dtype")
        _require_bool_tensor("query_mask", self.query_mask)

        if self.mode_logits.ndim != 2 or self.mode_logits.shape[1] != NUM_ROUTE_MODES:
            raise ValueError("mode_logits must have shape [Q,5]")
        q = self.mode_logits.shape[0]
        expected = {
            "mode_probabilities": (q, NUM_ROUTE_MODES),
            "means": (q, NUM_ROUTE_MODES, NUM_HORIZONS, 2),
            "scales": (q, NUM_ROUTE_MODES, NUM_HORIZONS, 2),
            "correlations": (q, NUM_ROUTE_MODES, NUM_HORIZONS),
            "primary_mode_index": (q,),
            "primary_route_normalized": (q, NUM_HORIZONS, 2),
            "marginal_mean_route_normalized": (q, NUM_HORIZONS, 2),
            "query_mask": (q,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {list(shape)}")
        if self.horizon_mask is not None:
            _require_bool_tensor("horizon_mask", self.horizon_mask)
            if tuple(self.horizon_mask.shape) != (q, NUM_HORIZONS):
                raise ValueError("horizon_mask must have shape [Q,12]")
        if self.current_xy is not None:
            _require_float_tensor("current_xy", self.current_xy)
            if tuple(self.current_xy.shape) != (q, 2):
                raise ValueError("current_xy must have shape [Q,2]")

        tensors = [
            getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Tensor)
        ]
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all output tensors must share a device")
        for name in (
            "mode_logits",
            "mode_probabilities",
            "means",
            "scales",
            "correlations",
            "primary_route_normalized",
            "marginal_mean_route_normalized",
        ):
            value = getattr(self, name)
            if value.numel() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if self.scales.numel() and bool((self.scales <= 0).any()):
            raise ValueError("scales must be strictly positive")
        if self.correlations.numel() and bool((self.correlations.abs() >= 1).any()):
            raise ValueError("correlations must satisfy abs(rho) < 1")
        if self.scales.numel():
            scales32 = self.scales.float()
            rho32 = self.correlations.float()
            determinants = (
                scales32[..., 0].square()
                * scales32[..., 1].square()
                * (1.0 - rho32.square())
            )
            if not bool(torch.isfinite(determinants).all()) or bool(
                (determinants <= 0).any()
            ):
                raise ValueError(
                    "bivariate covariance determinants must be finite and positive"
                )
        if self.mode_probabilities.numel():
            if bool((self.mode_probabilities < 0).any()):
                raise ValueError("mode probabilities must be nonnegative")
            sums = self.mode_probabilities.float().sum(dim=-1)
            if not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)):
                raise ValueError("mode probabilities must sum to one")
        if self.primary_mode_index.numel() and bool(
            ((self.primary_mode_index < 0) | (self.primary_mode_index >= NUM_ROUTE_MODES)).any()
        ):
            raise ValueError("primary_mode_index is outside [0,5)")

    @property
    def architecture_id(self) -> str:
        return PARALLEL_TRAJECTORY_ARCHITECTURE_ID

    def with_horizon_mask(self, horizon_mask: Tensor | None) -> ParallelTrajectoryOutput:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values["horizon_mask"] = horizon_mask
        return ParallelTrajectoryOutput(**values)

    def all_routes_world(
        self,
        profile: WorldGridProfile,
        current_xy: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        origin = self.current_xy if current_xy is None else current_xy
        if origin is None:
            raise ValueError("current_xy is required for world reconstruction")
        return normalized_displacements_to_world(self.means, origin, profile)

    world_routes = all_routes_world

    def primary_route_world(
        self,
        profile: WorldGridProfile,
        current_xy: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        origin = self.current_xy if current_xy is None else current_xy
        if origin is None:
            raise ValueError("current_xy is required for world reconstruction")
        return normalized_displacements_to_world(
            self.primary_route_normalized, origin, profile
        )

    def marginal_mean_route_world(
        self,
        profile: WorldGridProfile,
        current_xy: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        origin = self.current_xy if current_xy is None else current_xy
        if origin is None:
            raise ValueError("current_xy is required for world reconstruction")
        return normalized_displacements_to_world(
            self.marginal_mean_route_normalized, origin, profile
        )

    def to(self, *args: object, **kwargs: object) -> ParallelTrajectoryOutput:
        return ParallelTrajectoryOutput(
            **_move_tensor_fields(self, *args, **kwargs)
        )


@dataclass(frozen=True, slots=True)
class RouteMixtureLossOutput:
    loss: Tensor
    valid_route_count: int
    valid_horizon_count: int
    per_route_nll: Tensor
    nll_per_valid_horizon: Tensor

    @property
    def raw_loss(self) -> Tensor:
        return self.loss

    @property
    def route_nll(self) -> Tensor:
        return self.per_route_nll

    @property
    def total(self) -> Tensor:
        return self.loss


__all__ = [
    "ParallelTrajectoryBatch",
    "ParallelTrajectoryOutput",
    "ParallelTrajectoryTargets",
    "RouteMixtureLossOutput",
    "normalized_displacements_to_world",
]
