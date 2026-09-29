from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor

from .contracts import EncoderBatch


def _move_dataclass(instance: object, *args: object, **kwargs: object) -> dict[str, object]:
    return {
        field.name: (
            value.to(*args, **kwargs) if isinstance(value, (Tensor, EncoderBatch)) else value
        )
        for field in fields(instance)
        for value in (getattr(instance, field.name),)
    }


@dataclass(frozen=True, slots=True)
class PlannerObservation:
    """Caller-declared information that is observable by the route planner.

    All masks use the encoder's public ``[B,T,N]`` axis order.  Identity,
    placement, supervision, and future-label data are deliberately absent.
    """

    causal_time_mask: BoolTensor
    own_team_mask: BoolTensor
    observable_opponent_mask: BoolTensor
    observable_prior_opponent_mask: BoolTensor
    revealed_target_zone_mask: BoolTensor
    query_mask: BoolTensor

    def __post_init__(self) -> None:
        expected_dimensions = {
            "causal_time_mask": 2,
            "own_team_mask": 2,
            "observable_opponent_mask": 3,
            "observable_prior_opponent_mask": 2,
            "revealed_target_zone_mask": 2,
            "query_mask": 3,
        }
        for name, ndim in expected_dimensions.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
            if value.ndim != ndim:
                raise ValueError(f"{name} must have {ndim} dimensions")

        if self.causal_time_mask.shape[1] > 1 and torch.any(
            self.causal_time_mask[:, 1:] & ~self.causal_time_mask[:, :-1]
        ):
            raise ValueError("causal_time_mask must be a valid prefix")
        if torch.any(self.own_team_mask.sum(dim=-1) != 1):
            raise ValueError("own_team_mask must select exactly one team per batch item")

    def to(self, *args: object, **kwargs: object) -> PlannerObservation:
        return PlannerObservation(**_move_dataclass(self, *args, **kwargs))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PlannerBatch:
    """Sanitized, identifier-free inputs that may be passed to the planner."""

    encoder_batch: EncoderBatch
    own_team_mask: BoolTensor
    revealed_target_zone_mask: BoolTensor
    query_mask: BoolTensor

    @property
    def batch(self) -> EncoderBatch:
        """Readable alias for callers composing an encoder manually."""

        return self.encoder_batch

    @property
    def causal_time_mask(self) -> BoolTensor:
        return self.encoder_batch.time_mask

    def to(self, *args: object, **kwargs: object) -> PlannerBatch:
        return PlannerBatch(**_move_dataclass(self, *args, **kwargs))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PreviousWaypoints:
    """The eleven future-prefix positions ``y1..y11``.

    The planner derives ``y0`` from the sanitized query observation and
    prepends it internally, so these tensors never contain a BOS token.
    """

    cells: LongTensor
    offsets: FloatTensor

    def __post_init__(self) -> None:
        if not isinstance(self.cells, Tensor) or not isinstance(self.offsets, Tensor):
            raise TypeError("cells and offsets must be tensors")
        if self.cells.dtype != torch.int64:
            raise TypeError("cells must have int64 dtype")
        if not self.offsets.is_floating_point():
            raise TypeError("offsets must have a floating-point dtype")
        if self.cells.ndim != 2 or self.cells.shape[1] != 11:
            raise ValueError("cells must have shape [Q,11]")
        if self.offsets.shape != (*self.cells.shape, 2):
            raise ValueError("offsets must have shape [Q,11,2]")
        if self.cells.numel() and (
            int(self.cells.min()) < 0 or int(self.cells.max()) > 1023
        ):
            raise ValueError("cells must contain map cells in [0,1023]")
        if self.offsets.numel() and (
            not bool(torch.isfinite(self.offsets).all())
            or bool((self.offsets.abs() > 1.0).any())
        ):
            raise ValueError("offsets must be finite and bounded by [-1,1]")

    def to(self, *args: object, **kwargs: object) -> PreviousWaypoints:
        return PreviousWaypoints(**_move_dataclass(self, *args, **kwargs))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PlannerLogits:
    density_logits: FloatTensor
    density: FloatTensor
    cell_logits: FloatTensor
    offsets: FloatTensor
    query_mask: BoolTensor

    def to(self, *args: object, **kwargs: object) -> PlannerLogits:
        return PlannerLogits(**_move_dataclass(self, *args, **kwargs))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PlannerTargets:
    """Identifier-free teacher-forcing targets for ``Q`` planner queries.

    The field list is intentionally the complete public target schema.  CPU
    identifiers and target-construction diagnostics live in the supervision
    wrapper rather than travelling with model inputs.
    """

    route_cells: LongTensor
    route_offsets: FloatTensor
    route_mask: BoolTensor
    previous_cells: LongTensor
    previous_offsets: FloatTensor
    congestion_targets: FloatTensor
    congestion_mask: BoolTensor

    def __post_init__(self) -> None:
        tensors = {
            field.name: getattr(self, field.name) for field in fields(self)
        }
        if any(not isinstance(value, Tensor) for value in tensors.values()):
            raise TypeError("all PlannerTargets fields must be tensors")
        if self.route_cells.dtype != torch.int64:
            raise TypeError("route_cells must have int64 dtype")
        if self.previous_cells.dtype != torch.int64:
            raise TypeError("previous_cells must have int64 dtype")
        for name in ("route_offsets", "previous_offsets", "congestion_targets"):
            if getattr(self, name).dtype != torch.float32:
                raise TypeError(f"{name} must have float32 dtype")
        for name in ("route_mask", "congestion_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")

        if self.route_cells.ndim != 2 or self.route_cells.shape[1] != 12:
            raise ValueError("route_cells must have shape [Q,12]")
        q = self.route_cells.shape[0]
        route_shape = (q, 12)
        if tuple(self.route_offsets.shape) != (q, 12, 2):
            raise ValueError("route_offsets must have shape [Q,12,2]")
        if tuple(self.route_mask.shape) != route_shape:
            raise ValueError("route_mask must have shape [Q,12]")
        if tuple(self.previous_cells.shape) != (q, 11):
            raise ValueError("previous_cells must have shape [Q,11]")
        if tuple(self.previous_offsets.shape) != (q, 11, 2):
            raise ValueError("previous_offsets must have shape [Q,11,2]")
        if tuple(self.congestion_targets.shape) != (q, 12, 32, 32):
            raise ValueError(
                "congestion_targets must have shape [Q,12,32,32]"
            )
        if tuple(self.congestion_mask.shape) != route_shape:
            raise ValueError("congestion_mask must have shape [Q,12]")

        if self.route_cells.numel() and (
            int(self.route_cells.min()) < 0
            or int(self.route_cells.max()) > 1023
        ):
            raise ValueError("route_cells must contain map cells in [0,1023]")
        if self.previous_cells.numel() and (
            int(self.previous_cells.min()) < 0
            or int(self.previous_cells.max()) > 1023
        ):
            raise ValueError("previous_cells must contain map cells in [0,1023]")
        if torch.any(self.route_mask[:, 1:] & ~self.route_mask[:, :-1]):
            raise ValueError("route_mask must be a valid prefix")

        for name in ("route_offsets", "previous_offsets"):
            value = getattr(self, name)
            if value.numel() and (
                not bool(torch.isfinite(value).all())
                or bool((value.abs() > 1.0).any())
            ):
                raise ValueError(f"{name} must be finite and bounded by [-1,1]")
        if self.congestion_targets.numel() and (
            not bool(torch.isfinite(self.congestion_targets).all())
            or bool((self.congestion_targets < 0).any())
        ):
            raise ValueError(
                "congestion_targets must be finite and nonnegative"
            )
        if torch.any(
            self.route_offsets.masked_select(~self.route_mask.unsqueeze(-1))
            != 0
        ):
            raise ValueError("masked route offsets must be zero")
        if torch.any(self.route_cells.masked_select(~self.route_mask) != 0):
            raise ValueError("masked route cells must be zero")
        expected_previous_cells = torch.where(
            self.route_mask[:, :-1],
            self.route_cells[:, :-1],
            torch.zeros_like(self.previous_cells),
        )
        expected_previous_offsets = torch.where(
            self.route_mask[:, :-1].unsqueeze(-1),
            self.route_offsets[:, :-1],
            torch.zeros_like(self.previous_offsets),
        )
        if not torch.equal(self.previous_cells, expected_previous_cells):
            raise ValueError("previous_cells must be the shifted valid route prefix")
        if not torch.equal(self.previous_offsets, expected_previous_offsets):
            raise ValueError("previous_offsets must be the shifted valid route prefix")
        if torch.any(
            self.congestion_targets.masked_select(
                ~self.congestion_mask[:, :, None, None]
            )
            != 0
        ):
            raise ValueError("masked congestion targets must be zero")

    @property
    def previous_waypoints(self) -> PreviousWaypoints:
        return PreviousWaypoints(self.previous_cells, self.previous_offsets)

    def to(self, *args: object, **kwargs: object) -> PlannerTargets:
        return PlannerTargets(**_move_dataclass(self, *args, **kwargs))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class GeneratedRoute:
    cells: LongTensor
    offsets: FloatTensor
    waypoints_xy_uu: FloatTensor
    waypoint_mask: BoolTensor
    step_log_probabilities: FloatTensor
    scores: FloatTensor
    query_mask: BoolTensor

    @property
    def valid_waypoint_mask(self) -> BoolTensor:
        return self.waypoint_mask

    @property
    def log_probability(self) -> FloatTensor:
        return self.scores

    def to(self, *args: object, **kwargs: object) -> GeneratedRoute:
        return GeneratedRoute(**_move_dataclass(self, *args, **kwargs))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BeamRoutes:
    cells: LongTensor
    offsets: FloatTensor
    waypoints_xy_uu: FloatTensor
    waypoint_mask: BoolTensor
    step_log_probabilities: FloatTensor
    scores: FloatTensor
    query_mask: BoolTensor

    @property
    def valid_waypoint_mask(self) -> BoolTensor:
        return self.waypoint_mask

    @property
    def log_probabilities(self) -> FloatTensor:
        return self.scores

    def to(self, *args: object, **kwargs: object) -> BeamRoutes:
        return BeamRoutes(**_move_dataclass(self, *args, **kwargs))  # type: ignore[arg-type]
