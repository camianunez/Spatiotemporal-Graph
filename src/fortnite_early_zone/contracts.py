from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any

import torch
from torch import Tensor

from fortnite_encoder.planner_contracts import PlannerObservation
from fortnite_encoder.planner_supervision import (
    PlannerQuery,
    PlannerTargetDiagnostics,
)


MOVEMENT_EPSILON_SCHEMA_VERSION = "early-zone-movement-epsilon:1.0"
MOVEMENT_EPSILON_SCHEMA_ID = MOVEMENT_EPSILON_SCHEMA_VERSION
SOFT_LINEAR_BAND_SCHEMA_VERSION = "early-zone-soft-linear-band:1.0"
SOFT_LINEAR_BAND_MODE = "soft_linear_band"
SOFT_LINEAR_BAND_LOWER_METERS = 1.0
SOFT_LINEAR_BAND_UPPER_METERS = 3.0
EARLY_ZONE_RESIDUAL_TARGET_SCHEMA_VERSION = "1.0"
EARLY_ZONE_RESIDUAL_TARGET_SCHEMA_ID = (
    f"early-zone-residual-targets:{EARLY_ZONE_RESIDUAL_TARGET_SCHEMA_VERSION}"
)
EARLY_ZONE_SOFT_TARGET_SCHEMA_VERSION = "2.0"
EARLY_ZONE_SOFT_TARGET_SCHEMA_ID = (
    f"early-zone-residual-targets:{EARLY_ZONE_SOFT_TARGET_SCHEMA_VERSION}"
)
ROUTE_STEPS = 12
PREVIOUS_ROUTE_STEPS = ROUTE_STEPS - 1
MIXTURE_COMPONENTS = 5
GRID_ROWS = 32
GRID_COLUMNS = 32


def _move_dataclass(instance: object, *args: object, **kwargs: object) -> dict[str, Any]:
    return {
        field.name: (
            value.to(*args, **kwargs) if isinstance(value, Tensor) else value
        )
        for field in fields(instance)
        for value in (getattr(instance, field.name),)
    }


class MovementState(IntEnum):
    HOLD = 0
    MOVE = 1


@dataclass(frozen=True, slots=True)
class MovementEpsilonConfig:
    """Versioned physical threshold used to derive HOLD/MOVE supervision."""

    schema_version: str
    movement_epsilon_meters: float

    def __post_init__(self) -> None:
        if self.schema_version != MOVEMENT_EPSILON_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{MOVEMENT_EPSILON_SCHEMA_VERSION!r}"
            )
        value = self.movement_epsilon_meters
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("movement_epsilon_meters must be finite and positive")
        object.__setattr__(self, "movement_epsilon_meters", float(value))

    @property
    def canonical_id(self) -> str:
        return self.schema_version


@dataclass(frozen=True, slots=True)
class SoftLinearBandConfig:
    """Pinned continuous movement supervision on the physical 1--3 m band."""

    schema_version: str = SOFT_LINEAR_BAND_SCHEMA_VERSION
    mode: str = SOFT_LINEAR_BAND_MODE
    lower_bound_meters: float = SOFT_LINEAR_BAND_LOWER_METERS
    upper_bound_meters: float = SOFT_LINEAR_BAND_UPPER_METERS

    def __post_init__(self) -> None:
        if self.schema_version != SOFT_LINEAR_BAND_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{SOFT_LINEAR_BAND_SCHEMA_VERSION!r}"
            )
        if self.mode != SOFT_LINEAR_BAND_MODE:
            raise ValueError("mode must be 'soft_linear_band'")
        for name, expected in (
            ("lower_bound_meters", SOFT_LINEAR_BAND_LOWER_METERS),
            ("upper_bound_meters", SOFT_LINEAR_BAND_UPPER_METERS),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                raise ValueError(f"{name} is fixed at {expected}")
            object.__setattr__(self, name, float(value))
        if self.lower_bound_meters >= self.upper_bound_meters:
            raise ValueError("soft movement bounds must be strictly increasing")

    def target(self, distance_meters: float) -> float:
        value = float(distance_meters)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("distance_meters must be finite and nonnegative")
        if value <= self.lower_bound_meters:
            return 0.0
        if value >= self.upper_bound_meters:
            return 1.0
        return (value - self.lower_bound_meters) / (
            self.upper_bound_meters - self.lower_bound_meters
        )

    @property
    def canonical_id(self) -> str:
        return self.schema_version


@dataclass(frozen=True, slots=True)
class PreviousResidualRoute:
    """The eleven teacher-forced residual steps preceding decoder steps 2..12."""

    normalized_displacements: Tensor
    movement_states: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.normalized_displacements, Tensor) or not isinstance(
            self.movement_states, Tensor
        ):
            raise TypeError("teacher-forcing values must be tensors")
        if not self.normalized_displacements.is_floating_point():
            raise TypeError("normalized_displacements must be floating point")
        if self.movement_states.dtype != torch.int64:
            raise TypeError("movement_states must have int64 dtype")
        if self.normalized_displacements.ndim != 3 or tuple(
            self.normalized_displacements.shape[1:]
        ) != (PREVIOUS_ROUTE_STEPS, 2):
            raise ValueError(
                "normalized_displacements must have shape [Q,11,2]"
            )
        if tuple(self.movement_states.shape) != (
            self.normalized_displacements.shape[0],
            PREVIOUS_ROUTE_STEPS,
        ):
            raise ValueError("movement_states must have shape [Q,11]")
        if self.normalized_displacements.device != self.movement_states.device:
            raise ValueError("teacher-forcing tensors must share a device")
        if self.normalized_displacements.numel() and not bool(
            torch.isfinite(self.normalized_displacements).all()
        ):
            raise ValueError("normalized_displacements must be finite")
        if self.movement_states.numel() and bool(
            (
                (self.movement_states != int(MovementState.HOLD))
                & (self.movement_states != int(MovementState.MOVE))
            ).any()
        ):
            raise ValueError("movement_states must contain only HOLD or MOVE")
    @property
    def displacements(self) -> Tensor:
        return self.normalized_displacements

    def to(self, *args: object, **kwargs: object) -> PreviousResidualRoute:
        return PreviousResidualRoute(**_move_dataclass(self, *args, **kwargs))


@dataclass(frozen=True, slots=True)
class PreviousSoftResidualRoute:
    """Raw observed displacement prefixes paired with float movement targets."""

    normalized_displacements: Tensor
    movement_targets: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.normalized_displacements, Tensor) or not isinstance(
            self.movement_targets, Tensor
        ):
            raise TypeError("soft teacher-forcing values must be tensors")
        if not self.normalized_displacements.is_floating_point():
            raise TypeError("normalized_displacements must be floating point")
        if not self.movement_targets.is_floating_point():
            raise TypeError("movement_targets must be floating point")
        if self.normalized_displacements.ndim != 3 or tuple(
            self.normalized_displacements.shape[1:]
        ) != (PREVIOUS_ROUTE_STEPS, 2):
            raise ValueError("normalized_displacements must have shape [Q,11,2]")
        if tuple(self.movement_targets.shape) != (
            self.normalized_displacements.shape[0],
            PREVIOUS_ROUTE_STEPS,
        ):
            raise ValueError("movement_targets must have shape [Q,11]")
        if self.normalized_displacements.device != self.movement_targets.device:
            raise ValueError("soft teacher-forcing tensors must share a device")
        if self.normalized_displacements.numel() and not bool(
            torch.isfinite(self.normalized_displacements).all()
        ):
            raise ValueError("normalized_displacements must be finite")
        if self.movement_targets.numel() and (
            not bool(torch.isfinite(self.movement_targets).all())
            or bool(
                ((self.movement_targets < 0.0) | (self.movement_targets > 1.0)).any()
            )
        ):
            raise ValueError("movement_targets must be finite values in [0,1]")

    @property
    def displacements(self) -> Tensor:
        return self.normalized_displacements

    def to(self, *args: object, **kwargs: object) -> PreviousSoftResidualRoute:
        return PreviousSoftResidualRoute(**_move_dataclass(self, *args, **kwargs))


def _require_float_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


@dataclass(frozen=True, slots=True)
class EarlyZoneResidualDecoderOutput:
    movement_logits: Tensor
    movement_probabilities: Tensor
    mixture_logits: Tensor
    mixture_probabilities: Tensor
    component_means: Tensor
    component_scales: Tensor
    component_correlations: Tensor
    expected_displacements: Tensor
    route_valid_mask: Tensor

    def __post_init__(self) -> None:
        float_values = {
            "movement_logits": self.movement_logits,
            "movement_probabilities": self.movement_probabilities,
            "mixture_logits": self.mixture_logits,
            "mixture_probabilities": self.mixture_probabilities,
            "component_means": self.component_means,
            "component_scales": self.component_scales,
            "component_correlations": self.component_correlations,
            "expected_displacements": self.expected_displacements,
        }
        for name, value in float_values.items():
            _require_float_tensor(name, value)
        if not isinstance(self.route_valid_mask, Tensor):
            raise TypeError("route_valid_mask must be a tensor")
        if self.route_valid_mask.dtype != torch.bool:
            raise TypeError("route_valid_mask must have bool dtype")
        if self.movement_logits.ndim != 2 or self.movement_logits.shape[1] != ROUTE_STEPS:
            raise ValueError("movement_logits must have shape [Q,12]")
        q = self.movement_logits.shape[0]
        route_shape = (q, ROUTE_STEPS)
        expected_shapes = {
            "movement_probabilities": route_shape,
            "mixture_logits": (*route_shape, MIXTURE_COMPONENTS),
            "mixture_probabilities": (*route_shape, MIXTURE_COMPONENTS),
            "component_means": (*route_shape, MIXTURE_COMPONENTS, 2),
            "component_scales": (*route_shape, MIXTURE_COMPONENTS, 2),
            "component_correlations": (*route_shape, MIXTURE_COMPONENTS),
            "expected_displacements": (*route_shape, 2),
            "route_valid_mask": route_shape,
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {list(shape)}")
        devices = {value.device for value in float_values.values()}
        devices.add(self.route_valid_mask.device)
        if len(devices) != 1:
            raise ValueError("all decoder outputs must share a device")

        valid = self.route_valid_mask
        for name, value in float_values.items():
            expanded = valid
            while expanded.ndim < value.ndim:
                expanded = expanded.unsqueeze(-1)
            selected = value.masked_select(expanded)
            if selected.numel() and not bool(torch.isfinite(selected).all()):
                raise ValueError(f"valid {name} values must be finite")
            invalid = value.masked_select(~expanded)
            if invalid.numel() and bool((invalid != 0).any()):
                raise ValueError(f"invalid {name} values must be exactly zero")

        if bool(valid.any()):
            movement = self.movement_probabilities.masked_select(valid)
            if bool(((movement < 0.0) | (movement > 1.0)).any()):
                raise ValueError("valid movement probabilities must be in [0,1]")
            mixture_sums = self.mixture_probabilities.sum(dim=-1).masked_select(valid)
            if not bool(
                torch.allclose(
                    mixture_sums,
                    torch.ones_like(mixture_sums),
                    rtol=1e-5,
                    atol=1e-6,
                )
            ):
                raise ValueError("valid mixture probabilities must sum to one")
            expanded_valid = valid.unsqueeze(-1).unsqueeze(-1)
            scales = self.component_scales.masked_select(expanded_valid)
            if bool((scales <= 0.0).any()):
                raise ValueError("valid component scales must be positive")
            rho = self.component_correlations.masked_select(valid.unsqueeze(-1))
            if bool((rho.abs() >= 1.0).any()):
                raise ValueError("valid component correlations must have abs(rho) < 1")
            sx = self.component_scales[..., 0]
            sy = self.component_scales[..., 1]
            determinant = sx.square() * sy.square() * (
                1.0 - self.component_correlations.square()
            )
            if bool(
                (
                    determinant.masked_select(valid.unsqueeze(-1)) <= 0.0
                ).any()
            ):
                raise ValueError("valid covariance determinants must be positive")

    @property
    def movement_probability(self) -> Tensor:
        return self.movement_probabilities

    @property
    def mixture_weights(self) -> Tensor:
        return self.mixture_probabilities

    @property
    def means(self) -> Tensor:
        return self.component_means

    @property
    def scales(self) -> Tensor:
        return self.component_scales

    @property
    def correlations(self) -> Tensor:
        return self.component_correlations

    @property
    def rho(self) -> Tensor:
        return self.component_correlations

    @property
    def expected_normalized_displacements(self) -> Tensor:
        return self.expected_displacements

    @property
    def expected_displacement(self) -> Tensor:
        return self.expected_displacements

    @property
    def move_logits(self) -> Tensor:
        return self.movement_logits

    @property
    def move_probabilities(self) -> Tensor:
        return self.movement_probabilities

    @property
    def mixture_means(self) -> Tensor:
        return self.component_means

    @property
    def mixture_scales(self) -> Tensor:
        return self.component_scales

    @property
    def mixture_correlations(self) -> Tensor:
        return self.component_correlations

    def to(self, *args: object, **kwargs: object) -> EarlyZoneResidualDecoderOutput:
        return EarlyZoneResidualDecoderOutput(
            **_move_dataclass(self, *args, **kwargs)
        )


@dataclass(frozen=True, slots=True)
class EarlyZoneResidualPlannerOutput:
    decoder: EarlyZoneResidualDecoderOutput
    density_logits: Tensor
    density: Tensor
    expected_xy_uu: Tensor
    initial_xy_uu: Tensor
    query_valid_mask: Tensor
    early_zone_mask: Tensor
    unsupported_phase: Tensor
    query_mask: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.decoder, EarlyZoneResidualDecoderOutput):
            raise TypeError("decoder must be an EarlyZoneResidualDecoderOutput")
        q = self.decoder.movement_logits.shape[0]
        for name in ("density_logits", "density", "expected_xy_uu", "initial_xy_uu"):
            _require_float_tensor(name, getattr(self, name))
        if tuple(self.density_logits.shape) != (q, ROUTE_STEPS, GRID_ROWS, GRID_COLUMNS):
            raise ValueError("density_logits must have shape [Q,12,32,32]")
        if self.density.shape != self.density_logits.shape:
            raise ValueError("density must have the same shape as density_logits")
        if tuple(self.expected_xy_uu.shape) != (q, ROUTE_STEPS, 2):
            raise ValueError("expected_xy_uu must have shape [Q,12,2]")
        if tuple(self.initial_xy_uu.shape) != (q, 2):
            raise ValueError("initial_xy_uu must have shape [Q,2]")
        for name in ("query_valid_mask", "early_zone_mask", "unsupported_phase"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.dtype != torch.bool:
                raise TypeError(f"{name} must be a bool tensor")
            if tuple(value.shape) != (q,):
                raise ValueError(f"{name} must have shape [Q]")
        if not isinstance(self.query_mask, Tensor) or self.query_mask.dtype != torch.bool:
            raise TypeError("query_mask must be a bool tensor")
        if self.query_mask.ndim != 3:
            raise ValueError("query_mask must have shape [B,T,N]")
        if int(self.query_mask.sum().item()) != q:
            raise ValueError("query_mask true count must equal Q")
        if not torch.equal(self.unsupported_phase, ~self.early_zone_mask):
            raise ValueError("unsupported_phase must be the complement of early_zone_mask")
        expected_route_mask = (
            self.query_valid_mask & self.early_zone_mask
        ).unsqueeze(-1).expand(-1, ROUTE_STEPS)
        if not torch.equal(self.decoder.route_valid_mask, expected_route_mask):
            raise ValueError("route_valid_mask does not match query/phase eligibility")
        invalid_route = ~self.decoder.route_valid_mask.unsqueeze(-1)
        invalid_xy = self.expected_xy_uu.masked_select(invalid_route)
        if invalid_xy.numel() and bool((invalid_xy != 0).any()):
            raise ValueError("invalid expected_xy_uu values must be exactly zero")
        valid_xy = self.expected_xy_uu.masked_select(
            self.decoder.route_valid_mask.unsqueeze(-1)
        )
        if valid_xy.numel() and not bool(torch.isfinite(valid_xy).all()):
            raise ValueError("valid expected_xy_uu values must be finite")
        eligible = self.query_valid_mask & self.early_zone_mask
        invalid_initial = self.initial_xy_uu.masked_select(~eligible.unsqueeze(-1))
        if invalid_initial.numel() and bool((invalid_initial != 0).any()):
            raise ValueError("invalid initial_xy_uu values must be exactly zero")
        if self.density.numel() and (
            not bool(torch.isfinite(self.density).all())
            or bool((self.density <= 0.0).any())
        ):
            raise ValueError("density must be finite and positive")

    @property
    def congestion_logits(self) -> Tensor:
        return self.density_logits

    @property
    def congestion_density(self) -> Tensor:
        return self.density

    @property
    def route_valid_mask(self) -> Tensor:
        return self.decoder.route_valid_mask

    @property
    def movement_logits(self) -> Tensor:
        return self.decoder.movement_logits

    @property
    def movement_probabilities(self) -> Tensor:
        return self.decoder.movement_probabilities

    @property
    def movement_probability(self) -> Tensor:
        return self.decoder.movement_probabilities

    @property
    def mixture_logits(self) -> Tensor:
        return self.decoder.mixture_logits

    @property
    def mixture_probabilities(self) -> Tensor:
        return self.decoder.mixture_probabilities

    @property
    def mixture_weights(self) -> Tensor:
        return self.decoder.mixture_probabilities

    @property
    def component_means(self) -> Tensor:
        return self.decoder.component_means

    @property
    def component_scales(self) -> Tensor:
        return self.decoder.component_scales

    @property
    def component_correlations(self) -> Tensor:
        return self.decoder.component_correlations

    @property
    def expected_displacements(self) -> Tensor:
        return self.decoder.expected_displacements

    @property
    def expected_normalized_displacements(self) -> Tensor:
        return self.decoder.expected_displacements

    @property
    def expected_waypoints_xy_uu(self) -> Tensor:
        return self.expected_xy_uu

    @property
    def expected_world_xy_uu(self) -> Tensor:
        return self.expected_xy_uu

    def to(self, *args: object, **kwargs: object) -> EarlyZoneResidualPlannerOutput:
        values = _move_dataclass(self, *args, **kwargs)
        values["decoder"] = self.decoder.to(*args, **kwargs)
        return EarlyZoneResidualPlannerOutput(**values)


@dataclass(frozen=True, slots=True)
class EarlyZoneResidualTargets:
    movement_states: Tensor
    route_displacements: Tensor
    route_xy_uu: Tensor
    route_mask: Tensor
    previous_displacements: Tensor
    previous_movement_states: Tensor
    congestion_targets: Tensor
    congestion_mask: Tensor

    def __post_init__(self) -> None:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        if any(not isinstance(value, Tensor) for value in values.values()):
            raise TypeError("all EarlyZoneResidualTargets fields must be tensors")
        if self.movement_states.dtype != torch.int64:
            raise TypeError("movement_states must have int64 dtype")
        if self.previous_movement_states.dtype != torch.int64:
            raise TypeError("previous_movement_states must have int64 dtype")
        for name in (
            "route_displacements",
            "route_xy_uu",
            "previous_displacements",
            "congestion_targets",
        ):
            if getattr(self, name).dtype != torch.float32:
                raise TypeError(f"{name} must have float32 dtype")
        for name in ("route_mask", "congestion_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
        if self.movement_states.ndim != 2 or self.movement_states.shape[1] != ROUTE_STEPS:
            raise ValueError("movement_states must have shape [Q,12]")
        q = self.movement_states.shape[0]
        route_shape = (q, ROUTE_STEPS)
        expected_shapes = {
            "route_displacements": (*route_shape, 2),
            "route_xy_uu": (*route_shape, 2),
            "route_mask": route_shape,
            "previous_displacements": (q, PREVIOUS_ROUTE_STEPS, 2),
            "previous_movement_states": (q, PREVIOUS_ROUTE_STEPS),
            "congestion_targets": (q, ROUTE_STEPS, GRID_ROWS, GRID_COLUMNS),
            "congestion_mask": route_shape,
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {list(shape)}")
        if bool(
            (
                (self.movement_states != int(MovementState.HOLD))
                & (self.movement_states != int(MovementState.MOVE))
            ).any()
        ):
            raise ValueError("movement_states must contain only HOLD or MOVE")
        if torch.any(self.route_mask[:, 1:] & ~self.route_mask[:, :-1]):
            raise ValueError("route_mask must be a valid prefix")
        invalid = ~self.route_mask
        if bool(self.movement_states.masked_select(invalid).any()):
            raise ValueError("invalid movement states must be HOLD")
        for name in ("route_displacements", "route_xy_uu"):
            value = getattr(self, name)
            if value.numel() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
            if bool(value.masked_select(invalid.unsqueeze(-1)).any()):
                raise ValueError(f"invalid {name} values must be zero")
        hold = self.movement_states == int(MovementState.HOLD)
        if bool(self.route_displacements.masked_select(hold.unsqueeze(-1)).any()):
            raise ValueError("HOLD route displacements must be exactly zero")
        expected_previous_displacements = torch.where(
            self.route_mask[:, :-1].unsqueeze(-1),
            self.route_displacements[:, :-1],
            torch.zeros_like(self.previous_displacements),
        )
        expected_previous_states = torch.where(
            self.route_mask[:, :-1],
            self.movement_states[:, :-1],
            torch.zeros_like(self.previous_movement_states),
        )
        if not torch.equal(self.previous_displacements, expected_previous_displacements):
            raise ValueError("previous_displacements must equal the valid route prefix")
        if not torch.equal(self.previous_movement_states, expected_previous_states):
            raise ValueError("previous_movement_states must equal the valid route prefix")
        if self.congestion_targets.numel() and (
            not bool(torch.isfinite(self.congestion_targets).all())
            or bool((self.congestion_targets < 0.0).any())
        ):
            raise ValueError("congestion_targets must be finite and nonnegative")
        if bool(
            self.congestion_targets.masked_select(
                ~self.congestion_mask[:, :, None, None]
            ).any()
        ):
            raise ValueError("masked congestion targets must be zero")

    @property
    def normalized_displacements(self) -> Tensor:
        return self.route_displacements

    @property
    def displacement_targets(self) -> Tensor:
        return self.route_displacements

    @property
    def movement_targets(self) -> Tensor:
        return self.movement_states

    @property
    def path_targets_xy_uu(self) -> Tensor:
        return self.route_xy_uu

    @property
    def raw_route_xy_uu(self) -> Tensor:
        return self.route_xy_uu

    @property
    def previous_normalized_displacements(self) -> Tensor:
        return self.previous_displacements

    @property
    def previous_route(self) -> PreviousResidualRoute:
        return PreviousResidualRoute(
            self.previous_displacements,
            self.previous_movement_states,
        )

    def to(self, *args: object, **kwargs: object) -> EarlyZoneResidualTargets:
        return EarlyZoneResidualTargets(**_move_dataclass(self, *args, **kwargs))


@dataclass(frozen=True, slots=True)
class EarlyZoneResidualTargetMetadata:
    queries: tuple[PlannerQuery, ...]
    world_grid_profile_id: str
    world_grid_profile_hash: str
    movement_epsilon_schema_version: str
    movement_epsilon_meters: float
    target_schema_id: str = EARLY_ZONE_RESIDUAL_TARGET_SCHEMA_ID

    def __post_init__(self) -> None:
        MovementEpsilonConfig(
            self.movement_epsilon_schema_version,
            self.movement_epsilon_meters,
        )
        if self.target_schema_id != EARLY_ZONE_RESIDUAL_TARGET_SCHEMA_ID:
            raise ValueError("target_schema_id is unsupported")

    @property
    def movement_epsilon_schema_id(self) -> str:
        return self.movement_epsilon_schema_version

    @property
    def movement_epsilon_config(self) -> MovementEpsilonConfig:
        return MovementEpsilonConfig(
            self.movement_epsilon_schema_version,
            self.movement_epsilon_meters,
        )


@dataclass(frozen=True, slots=True)
class EarlyZoneResidualSupervision:
    targets: EarlyZoneResidualTargets
    diagnostics: PlannerTargetDiagnostics
    metadata: EarlyZoneResidualTargetMetadata

    def to(self, *args: object, **kwargs: object) -> EarlyZoneResidualSupervision:
        return EarlyZoneResidualSupervision(
            targets=self.targets.to(*args, **kwargs),
            diagnostics=self.diagnostics,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class EarlyZoneSoftTargets:
    """V2 route supervision with continuous movement mass and raw motion."""

    movement_targets: Tensor
    route_displacements: Tensor
    route_xy_uu: Tensor
    route_mask: Tensor
    previous_displacements: Tensor
    previous_movement_targets: Tensor
    congestion_targets: Tensor
    congestion_mask: Tensor

    def __post_init__(self) -> None:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        if any(not isinstance(value, Tensor) for value in values.values()):
            raise TypeError("all EarlyZoneSoftTargets fields must be tensors")
        for name in (
            "movement_targets",
            "route_displacements",
            "route_xy_uu",
            "previous_displacements",
            "previous_movement_targets",
            "congestion_targets",
        ):
            if getattr(self, name).dtype != torch.float32:
                raise TypeError(f"{name} must have float32 dtype")
        for name in ("route_mask", "congestion_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
        if self.movement_targets.ndim != 2 or self.movement_targets.shape[1] != ROUTE_STEPS:
            raise ValueError("movement_targets must have shape [Q,12]")
        q = self.movement_targets.shape[0]
        route_shape = (q, ROUTE_STEPS)
        expected_shapes = {
            "route_displacements": (*route_shape, 2),
            "route_xy_uu": (*route_shape, 2),
            "route_mask": route_shape,
            "previous_displacements": (q, PREVIOUS_ROUTE_STEPS, 2),
            "previous_movement_targets": (q, PREVIOUS_ROUTE_STEPS),
            "congestion_targets": (q, ROUTE_STEPS, GRID_ROWS, GRID_COLUMNS),
            "congestion_mask": route_shape,
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {list(shape)}")
        if torch.any(self.route_mask[:, 1:] & ~self.route_mask[:, :-1]):
            raise ValueError("route_mask must be a valid prefix")
        invalid = ~self.route_mask
        if self.movement_targets.numel() and (
            not bool(torch.isfinite(self.movement_targets).all())
            or bool(
                ((self.movement_targets < 0.0) | (self.movement_targets > 1.0)).any()
            )
        ):
            raise ValueError("movement_targets must be finite values in [0,1]")
        if bool(self.movement_targets.masked_select(invalid).any()):
            raise ValueError("invalid movement targets must be exactly zero")
        for name in (
            "route_displacements",
            "route_xy_uu",
            "previous_displacements",
            "previous_movement_targets",
        ):
            value = getattr(self, name)
            if value.numel() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        for name in ("route_displacements", "route_xy_uu"):
            if bool(getattr(self, name).masked_select(invalid.unsqueeze(-1)).any()):
                raise ValueError(f"invalid {name} values must be zero")
        expected_previous_displacements = torch.where(
            self.route_mask[:, :-1].unsqueeze(-1),
            self.route_displacements[:, :-1],
            torch.zeros_like(self.previous_displacements),
        )
        expected_previous_targets = torch.where(
            self.route_mask[:, :-1],
            self.movement_targets[:, :-1],
            torch.zeros_like(self.previous_movement_targets),
        )
        if not torch.equal(self.previous_displacements, expected_previous_displacements):
            raise ValueError("previous_displacements must equal the raw valid route prefix")
        if not torch.equal(self.previous_movement_targets, expected_previous_targets):
            raise ValueError("previous_movement_targets must equal the valid target prefix")
        if self.congestion_targets.numel() and (
            not bool(torch.isfinite(self.congestion_targets).all())
            or bool((self.congestion_targets < 0.0).any())
        ):
            raise ValueError("congestion_targets must be finite and nonnegative")
        if bool(
            self.congestion_targets.masked_select(
                ~self.congestion_mask[:, :, None, None]
            ).any()
        ):
            raise ValueError("masked congestion targets must be zero")

    @property
    def normalized_displacements(self) -> Tensor:
        return self.route_displacements

    @property
    def displacement_targets(self) -> Tensor:
        return self.route_displacements

    @property
    def path_targets_xy_uu(self) -> Tensor:
        return self.route_xy_uu

    @property
    def raw_route_xy_uu(self) -> Tensor:
        return self.route_xy_uu

    @property
    def previous_normalized_displacements(self) -> Tensor:
        return self.previous_displacements

    @property
    def previous_route(self) -> PreviousSoftResidualRoute:
        return PreviousSoftResidualRoute(
            self.previous_displacements,
            self.previous_movement_targets,
        )

    @property
    def effective_move_mass(self) -> float:
        return float(
            self.movement_targets.masked_select(self.route_mask).double().sum().item()
        )

    @property
    def effective_hold_mass(self) -> float:
        return float(self.route_mask.sum().item()) - self.effective_move_mass

    def to(self, *args: object, **kwargs: object) -> EarlyZoneSoftTargets:
        return EarlyZoneSoftTargets(**_move_dataclass(self, *args, **kwargs))


@dataclass(frozen=True, slots=True)
class EarlyZoneSoftTargetMetadata:
    queries: tuple[PlannerQuery, ...]
    world_grid_profile_id: str
    world_grid_profile_hash: str
    movement_supervision: SoftLinearBandConfig
    target_schema_id: str = EARLY_ZONE_SOFT_TARGET_SCHEMA_ID

    def __post_init__(self) -> None:
        if not isinstance(self.movement_supervision, SoftLinearBandConfig):
            raise TypeError("movement_supervision must be SoftLinearBandConfig")
        if self.target_schema_id != EARLY_ZONE_SOFT_TARGET_SCHEMA_ID:
            raise ValueError("target_schema_id is unsupported")


@dataclass(frozen=True, slots=True)
class EarlyZoneSoftSupervision:
    targets: EarlyZoneSoftTargets
    diagnostics: PlannerTargetDiagnostics
    metadata: EarlyZoneSoftTargetMetadata

    def to(self, *args: object, **kwargs: object) -> EarlyZoneSoftSupervision:
        return EarlyZoneSoftSupervision(
            targets=self.targets.to(*args, **kwargs),
            diagnostics=self.diagnostics,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class GeneratedResidualRoute:
    movement_states: Tensor
    mixture_component_indices: Tensor
    normalized_displacements: Tensor
    waypoints_xy_uu: Tensor
    route_valid_mask: Tensor
    step_log_probabilities: Tensor
    cumulative_log_probabilities: Tensor
    scores: Tensor
    query_valid_mask: Tensor
    early_zone_mask: Tensor
    unsupported_phase: Tensor
    query_mask: Tensor

    @property
    def component_indices(self) -> Tensor:
        return self.mixture_component_indices

    @property
    def displacements(self) -> Tensor:
        return self.normalized_displacements

    @property
    def waypoint_mask(self) -> Tensor:
        return self.route_valid_mask

    @property
    def valid_waypoint_mask(self) -> Tensor:
        return self.route_valid_mask

    @property
    def log_probability(self) -> Tensor:
        return self.scores

    def to(self, *args: object, **kwargs: object) -> GeneratedResidualRoute:
        return GeneratedResidualRoute(**_move_dataclass(self, *args, **kwargs))


@dataclass(frozen=True, slots=True)
class BeamResidualRoutes:
    movement_states: Tensor
    mixture_component_indices: Tensor
    normalized_displacements: Tensor
    waypoints_xy_uu: Tensor
    route_valid_mask: Tensor
    step_log_probabilities: Tensor
    cumulative_log_probabilities: Tensor
    scores: Tensor
    query_valid_mask: Tensor
    early_zone_mask: Tensor
    unsupported_phase: Tensor
    query_mask: Tensor

    @property
    def component_indices(self) -> Tensor:
        return self.mixture_component_indices

    @property
    def displacements(self) -> Tensor:
        return self.normalized_displacements

    @property
    def waypoint_mask(self) -> Tensor:
        return self.route_valid_mask

    @property
    def valid_waypoint_mask(self) -> Tensor:
        return self.route_valid_mask

    @property
    def log_probabilities(self) -> Tensor:
        return self.scores

    def to(self, *args: object, **kwargs: object) -> BeamResidualRoutes:
        return BeamResidualRoutes(**_move_dataclass(self, *args, **kwargs))


@dataclass(frozen=True, slots=True)
class EarlyZoneResidualLoss:
    movement_loss: Tensor
    movement_count: int | float
    displacement_nll: Tensor
    displacement_count: int | float
    path_loss: Tensor
    path_count: int | float
    congestion_loss: Tensor
    congestion_horizon_count: int | float

    @property
    def movement_valid_step_count(self) -> int:
        return self.movement_count

    @property
    def valid_movement_step_count(self) -> int:
        return self.movement_count

    @property
    def displacement_valid_move_step_count(self) -> int:
        return self.displacement_count

    @property
    def move_step_count(self) -> int:
        return self.displacement_count

    @property
    def effective_move_mass(self) -> float:
        return float(self.displacement_count)

    @property
    def effective_hold_mass(self) -> float:
        return float(self.movement_count) - float(self.displacement_count)

    @property
    def path_valid_step_count(self) -> int:
        return self.path_count

    @property
    def congestion_valid_horizon_count(self) -> int:
        return self.congestion_horizon_count


# Readable compatibility aliases for callers that prefer the full feature name.
EarlyZoneGeneratedRoute = GeneratedResidualRoute
EarlyZoneBeamRoutes = BeamResidualRoutes
EarlyZoneResidualLossComponents = EarlyZoneResidualLoss
ResidualDecoderOutput = EarlyZoneResidualDecoderOutput
ResidualPlannerOutput = EarlyZoneResidualPlannerOutput
ResidualTargets = EarlyZoneResidualTargets
EarlyZoneResidualPlannerLogits = EarlyZoneResidualPlannerOutput
EarlyZoneResidualLogits = EarlyZoneResidualDecoderOutput
ResidualGeneratedRoute = GeneratedResidualRoute
ResidualBeamRoutes = BeamResidualRoutes


__all__ = [
    "BeamResidualRoutes",
    "EARLY_ZONE_RESIDUAL_TARGET_SCHEMA_ID",
    "EARLY_ZONE_RESIDUAL_TARGET_SCHEMA_VERSION",
    "EARLY_ZONE_SOFT_TARGET_SCHEMA_ID",
    "EARLY_ZONE_SOFT_TARGET_SCHEMA_VERSION",
    "EarlyZoneBeamRoutes",
    "EarlyZoneGeneratedRoute",
    "EarlyZoneResidualDecoderOutput",
    "EarlyZoneResidualLoss",
    "EarlyZoneResidualLossComponents",
    "EarlyZoneResidualLogits",
    "EarlyZoneResidualPlannerLogits",
    "EarlyZoneResidualPlannerOutput",
    "EarlyZoneResidualSupervision",
    "EarlyZoneResidualTargetMetadata",
    "EarlyZoneResidualTargets",
    "EarlyZoneSoftSupervision",
    "EarlyZoneSoftTargetMetadata",
    "EarlyZoneSoftTargets",
    "GeneratedResidualRoute",
    "MIXTURE_COMPONENTS",
    "MOVEMENT_EPSILON_SCHEMA_ID",
    "MOVEMENT_EPSILON_SCHEMA_VERSION",
    "MovementEpsilonConfig",
    "MovementState",
    "PREVIOUS_ROUTE_STEPS",
    "PreviousResidualRoute",
    "PreviousSoftResidualRoute",
    "ROUTE_STEPS",
    "ResidualDecoderOutput",
    "ResidualBeamRoutes",
    "ResidualGeneratedRoute",
    "ResidualPlannerOutput",
    "ResidualTargets",
    "SOFT_LINEAR_BAND_LOWER_METERS",
    "SOFT_LINEAR_BAND_MODE",
    "SOFT_LINEAR_BAND_SCHEMA_VERSION",
    "SOFT_LINEAR_BAND_UPPER_METERS",
    "SoftLinearBandConfig",
]
