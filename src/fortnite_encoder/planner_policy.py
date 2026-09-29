from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, Sequence

import torch
from torch import Tensor

from .contracts import EncoderBatch
from .planner_contracts import PlannerBatch, PlannerObservation


OBSERVATION_PROVIDER = "own_team_only"
OBSERVATION_PROVIDER_VERSION = "1.0"
OBSERVATION_PROVIDER_ID = (
    f"{OBSERVATION_PROVIDER}:{OBSERVATION_PROVIDER_VERSION}"
)


@dataclass(frozen=True, slots=True)
class PlannerPolicyConfig:
    provider: Literal["own_team_only"] = OBSERVATION_PROVIDER
    version: Literal["1.0"] = OBSERVATION_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.provider != OBSERVATION_PROVIDER:
            raise ValueError("only observation provider 'own_team_only' is supported")
        if self.version != OBSERVATION_PROVIDER_VERSION:
            raise ValueError("only own_team_only observation version '1.0' is supported")

    @property
    def canonical_id(self) -> str:
        return OBSERVATION_PROVIDER_ID

    @property
    def provider_id(self) -> str:
        return self.canonical_id

    @property
    def name(self) -> str:
        return self.provider

    @property
    def provider_version(self) -> str:
        return self.version


def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        formatted = ",".join(str(size) for size in shape)
        raise ValueError(f"{name} must have shape [{formatted}]")


def _validate_encoder_batch_shapes(batch: EncoderBatch) -> tuple[int, int, int, int]:
    xyz = batch.player_xyz_uu
    if xyz.ndim != 5 or xyz.shape[-1] != 3:
        raise ValueError("player_xyz_uu must have shape [B,T,N,P,3]")
    b, t, n, p, _ = xyz.shape
    if min(b, t, n, p) <= 0:
        raise ValueError("encoder batch axes B,T,N,P must be nonempty")

    for name in ("player_alive", "player_coord_mask", "life_index"):
        _require_shape(name, getattr(batch, name), (b, t, n, p))
    _require_shape("player_slot_mask", batch.player_slot_mask, (b, n, p))
    for name in ("current_circle_uu", "target_circle_uu"):
        _require_shape(name, getattr(batch, name), (b, t, 4))
    for name in (
        "zone_mask",
        "zone_phase",
        "match_elapsed_s",
        "players_remaining",
        "teams_remaining",
        "absolute_tick_index",
        "time_mask",
    ):
        _require_shape(name, getattr(batch, name), (b, t))
    _require_shape("phase_times_s", batch.phase_times_s, (b, t, 3))
    _require_shape("team_slot_mask", batch.team_slot_mask, (b, n))
    _require_shape("prior_player_xyz_uu", batch.prior_player_xyz_uu, (b, n, p, 3))
    for name in (
        "prior_player_alive",
        "prior_player_coord_mask",
        "prior_life_index",
    ):
        _require_shape(name, getattr(batch, name), (b, n, p))
    _require_shape("prior_state_available", batch.prior_state_available, (b,))

    boolean_fields = (
        "player_alive",
        "player_coord_mask",
        "player_slot_mask",
        "zone_mask",
        "time_mask",
        "team_slot_mask",
        "prior_player_alive",
        "prior_player_coord_mask",
        "prior_state_available",
    )
    for name in boolean_fields:
        if getattr(batch, name).dtype != torch.bool:
            raise TypeError(f"{name} must have bool dtype")
    return b, t, n, p


def _validate_policy(
    batch: EncoderBatch,
    observation: PlannerObservation,
) -> tuple[int, int, int, int]:
    b, t, n, p = _validate_encoder_batch_shapes(batch)
    expected = {
        "causal_time_mask": (b, t),
        "own_team_mask": (b, n),
        "observable_opponent_mask": (b, t, n),
        "observable_prior_opponent_mask": (b, n),
        "revealed_target_zone_mask": (b, t),
        "query_mask": (b, t, n),
    }
    for name, shape in expected.items():
        _require_shape(name, getattr(observation, name), shape)

    devices = {getattr(batch, field.name).device for field in fields(batch)}
    devices.update(
        getattr(observation, field.name).device for field in fields(observation)
    )
    if len(devices) != 1:
        raise ValueError("all planner policy tensors must be on the same device")

    if torch.any(observation.causal_time_mask & ~batch.time_mask):
        raise ValueError("causal_time_mask may only select valid encoder timesteps")
    if torch.any(observation.own_team_mask & ~batch.team_slot_mask):
        raise ValueError("own_team_mask must select a real team slot")

    real_time_team = batch.time_mask[:, :, None] & batch.team_slot_mask[:, None, :]
    own_at_time = observation.own_team_mask[:, None, :]
    if torch.any(observation.observable_opponent_mask & own_at_time):
        raise ValueError("observable_opponent_mask may not select the own team")
    if torch.any(observation.observable_opponent_mask & ~real_time_team):
        raise ValueError("observable_opponent_mask may only select real timesteps and teams")
    if torch.any(
        observation.observable_prior_opponent_mask & observation.own_team_mask
    ):
        raise ValueError("observable_prior_opponent_mask may not select the own team")
    if torch.any(
        observation.observable_prior_opponent_mask & ~batch.team_slot_mask
    ):
        raise ValueError("observable_prior_opponent_mask may only select real teams")
    if torch.any(observation.revealed_target_zone_mask & ~batch.time_mask):
        raise ValueError("revealed_target_zone_mask may only select valid timesteps")

    roster = batch.player_slot_mask[:, None, :, :]
    living_team = (batch.player_alive & roster).any(dim=-1)
    valid_own_query = (
        observation.causal_time_mask[:, :, None]
        & batch.time_mask[:, :, None]
        & observation.own_team_mask[:, None, :]
        & batch.team_slot_mask[:, None, :]
        & living_team
    )
    if torch.any(observation.query_mask & ~valid_own_query):
        raise ValueError("query_mask must select valid, living own-team nodes")
    return b, t, n, p


def apply_planner_input_policy(
    batch: EncoderBatch,
    observation: PlannerObservation,
) -> PlannerBatch:
    """Copy and sanitize an encoder batch before any encoder computation.

    Own-team state is retained across the causal prefix.  Opponent tensors are
    retained only under explicit observability masks, and all time-varying
    tensors beyond the prefix are zeroed.  Static roster/padding masks retain
    their existing meaning and are copied unchanged.
    """

    _validate_policy(batch, observation)
    causal = observation.causal_time_mask & batch.time_mask
    own = observation.own_team_mask[:, None, :]
    visible_opponent = (
        observation.observable_opponent_mask
        & ~own
        & batch.team_slot_mask[:, None, :]
    )
    visible_team = causal[:, :, None] & (own | visible_opponent)
    visible_player = visible_team.unsqueeze(-1)

    def visible_player_value(value: Tensor) -> Tensor:
        mask = visible_player
        if value.ndim == 5:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, value, torch.zeros_like(value))

    def causal_value(value: Tensor) -> Tensor:
        mask = causal
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, value, torch.zeros_like(value))

    prior_visible_team = (
        (observation.own_team_mask | observation.observable_prior_opponent_mask)
        & batch.team_slot_mask
        & batch.prior_state_available[:, None]
    )
    prior_visible_player = prior_visible_team.unsqueeze(-1)

    def prior_value(value: Tensor) -> Tensor:
        mask = prior_visible_player
        if value.ndim == 4:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, value, torch.zeros_like(value))

    revealed_target = observation.revealed_target_zone_mask & causal
    target_mask = revealed_target
    while target_mask.ndim < batch.target_circle_uu.ndim:
        target_mask = target_mask.unsqueeze(-1)

    sanitized = EncoderBatch(
        player_xyz_uu=visible_player_value(batch.player_xyz_uu),
        player_alive=visible_player_value(batch.player_alive),
        player_coord_mask=visible_player_value(batch.player_coord_mask),
        life_index=visible_player_value(batch.life_index),
        player_slot_mask=batch.player_slot_mask.clone(),
        current_circle_uu=causal_value(batch.current_circle_uu),
        target_circle_uu=torch.where(
            target_mask,
            batch.target_circle_uu,
            torch.zeros_like(batch.target_circle_uu),
        ),
        zone_mask=causal_value(batch.zone_mask),
        zone_phase=causal_value(batch.zone_phase),
        phase_times_s=causal_value(batch.phase_times_s),
        match_elapsed_s=causal_value(batch.match_elapsed_s),
        players_remaining=causal_value(batch.players_remaining),
        teams_remaining=causal_value(batch.teams_remaining),
        absolute_tick_index=causal_value(batch.absolute_tick_index),
        time_mask=causal.clone(),
        team_slot_mask=batch.team_slot_mask.clone(),
        prior_player_xyz_uu=prior_value(batch.prior_player_xyz_uu),
        prior_player_alive=prior_value(batch.prior_player_alive),
        prior_player_coord_mask=prior_value(batch.prior_player_coord_mask),
        prior_life_index=prior_value(batch.prior_life_index),
        prior_state_available=batch.prior_state_available.clone(),
    )
    return PlannerBatch(
        encoder_batch=sanitized,
        own_team_mask=observation.own_team_mask.clone(),
        revealed_target_zone_mask=revealed_target.clone(),
        query_mask=observation.query_mask.clone(),
    )


def build_planner_observation(
    batch: EncoderBatch,
    focal_team_indices: Sequence[int] | Tensor,
    config: PlannerPolicyConfig | None = None,
) -> PlannerObservation:
    """Declare an ``own_team_only:1.0`` observation for causal query slices.

    Every batch item is expected to be a trailing slice ending at its one
    query. Opponent state is never declared observable, including the prior
    motion sidecar. Target-zone geometry becomes visible exactly when the
    already-validated encoder zone sample is active.
    """

    resolved = config or PlannerPolicyConfig()
    if resolved.canonical_id != OBSERVATION_PROVIDER_ID:
        raise ValueError("unsupported planner observation policy")
    b, t, n, _ = _validate_encoder_batch_shapes(batch)
    if isinstance(focal_team_indices, Tensor):
        if focal_team_indices.ndim != 1 or focal_team_indices.shape[0] != b:
            raise ValueError("focal_team_indices must have shape [B]")
        if focal_team_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("focal_team_indices must have integer dtype")
        indices = focal_team_indices.to(device=batch.time_mask.device, dtype=torch.int64)
    else:
        values = tuple(focal_team_indices)
        if len(values) != b or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError("focal_team_indices must contain one integer per batch item")
        indices = torch.tensor(
            values, dtype=torch.int64, device=batch.time_mask.device
        )
    if bool(((indices < 0) | (indices >= n)).any()):
        raise ValueError("focal team index is outside the collated team axis")

    own = torch.zeros((b, n), dtype=torch.bool, device=batch.time_mask.device)
    own[torch.arange(b, device=own.device), indices] = True
    if torch.any(own & ~batch.team_slot_mask):
        raise ValueError("focal team index selects a padded team")
    causal = batch.time_mask.clone()
    lengths = causal.sum(dim=-1)
    if torch.any(lengths <= 0):
        raise ValueError("each planner query slice must have a valid timestep")
    query = torch.zeros((b, t, n), dtype=torch.bool, device=causal.device)
    query[
        torch.arange(b, device=query.device),
        lengths.to(torch.int64) - 1,
        indices,
    ] = True
    observation = PlannerObservation(
        causal_time_mask=causal,
        own_team_mask=own,
        observable_opponent_mask=torch.zeros(
            (b, t, n), dtype=torch.bool, device=causal.device
        ),
        observable_prior_opponent_mask=torch.zeros(
            (b, n), dtype=torch.bool, device=causal.device
        ),
        revealed_target_zone_mask=batch.zone_mask & causal,
        query_mask=query,
    )
    _validate_policy(batch, observation)
    return observation


build_own_team_observation = build_planner_observation


__all__ = [
    "OBSERVATION_PROVIDER",
    "OBSERVATION_PROVIDER_ID",
    "OBSERVATION_PROVIDER_VERSION",
    "PlannerBatch",
    "PlannerObservation",
    "PlannerPolicyConfig",
    "apply_planner_input_policy",
    "build_own_team_observation",
    "build_planner_observation",
]
