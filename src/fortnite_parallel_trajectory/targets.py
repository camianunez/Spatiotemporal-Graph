from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from fortnite_encoder.planner_supervision import (
    PlannerQuery,
    PlannerTargetSource,
    _count_at,
)
from fortnite_encoder.tensorize import SAMPLE_INTERVAL_SECONDS, _FLOAT_TOLERANCE

from .config import NUM_HORIZONS
from .contracts import ParallelTrajectoryTargets


def _living_count(source: PlannerTargetSource, tick: int, team: int) -> int:
    match = source.match
    return int(
        (
            match.player_alive[tick, team]
            & match.player_slot_mask[team]
        ).sum().item()
    )


def _current_centroid(
    source: PlannerTargetSource,
    tick: int,
    team: int,
) -> Tensor | None:
    match = source.match
    roster = match.player_slot_mask[team]
    valid = (
        match.player_alive[tick, team]
        & match.player_coord_mask[tick, team]
        & roster
    )
    if not bool(valid.any()):
        return None
    coordinates = match.player_xyz_uu[tick, team, valid, :2].double()
    if not bool(torch.isfinite(coordinates).all()):
        return None
    centroid = coordinates.mean(dim=0)
    return centroid if source.profile.cell(centroid.tolist()) is not None else None


def _query_is_eligible(
    source: PlannerTargetSource,
    tick: int,
    team: int,
) -> tuple[bool, Tensor | None]:
    match = source.match
    if not bool(match.time_mask[tick]) or not bool(match.team_slot_mask[team]):
        return False, None
    if _living_count(source, tick, team) <= 0:
        return False, None
    phase = int(match.zone_phase[tick].item())
    if not 1 <= phase <= 8:
        return False, None
    centroid = _current_centroid(source, tick, team)
    return centroid is not None, centroid


def _normalize_queries(
    source: PlannerTargetSource,
    queries: Iterable[PlannerQuery | tuple[int, int] | Mapping[str, Any]] | None,
) -> tuple[PlannerQuery, ...]:
    match = source.match
    if queries is None:
        result: list[PlannerQuery] = []
        for tick in range(match.num_timesteps):
            for team, team_id in enumerate(match.team_ids):
                eligible, _ = _query_is_eligible(source, tick, team)
                if eligible:
                    result.append(
                        PlannerQuery(
                            match.session_id,
                            team_id,
                            int(match.absolute_tick_index[tick].item()),
                        )
                    )
        return tuple(result)

    result = []
    for raw in queries:
        if isinstance(raw, PlannerQuery):
            query = raw
        elif isinstance(raw, Mapping):
            query = PlannerQuery(
                session_id=str(raw.get("session_id", match.session_id)),
                team_id=raw["team_id"],
                absolute_tick_index=raw["absolute_tick_index"],
            )
        else:
            values = tuple(raw)
            if len(values) != 2:
                raise ValueError("query tuples must contain (team_id, absolute_tick_index)")
            query = PlannerQuery(match.session_id, values[0], values[1])
        result.append(query)
    if len(set(result)) != len(result):
        raise ValueError("parallel trajectory queries must be unique")
    return tuple(result)


def _resolve_observation_mask(
    count: int,
    observation_query_mask: Sequence[bool] | Tensor | None,
) -> Tensor:
    if observation_query_mask is None:
        return torch.ones(count, dtype=torch.bool)
    if isinstance(observation_query_mask, Tensor):
        if observation_query_mask.dtype != torch.bool:
            raise TypeError("observation_query_mask must have bool dtype")
        if observation_query_mask.device.type != "cpu":
            raise ValueError("full-match target construction must remain on CPU")
        mask = observation_query_mask.clone()
    else:
        values = tuple(observation_query_mask)
        if any(type(value) is not bool for value in values):
            raise TypeError("observation_query_mask must contain booleans")
        mask = torch.tensor(values, dtype=torch.bool)
    if tuple(mask.shape) != (count,):
        raise ValueError("observation_query_mask must have shape [Q]")
    return mask


def _exact_target_tick(times: Tensor, target_time: float) -> int | None:
    matches = torch.isclose(
        times.double(),
        torch.tensor(target_time, dtype=torch.float64),
        rtol=0.0,
        atol=_FLOAT_TOLERANCE,
    ).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return None
    if matches.numel() != 1:
        raise ValueError("full match contains duplicate target timestamps")
    return int(matches.item())


def build_parallel_trajectory_targets(
    source: PlannerTargetSource,
    queries: Iterable[PlannerQuery | tuple[int, int] | Mapping[str, Any]] | None = None,
    *,
    observation_query_mask: Sequence[bool] | Tensor | None = None,
) -> ParallelTrajectoryTargets:
    """Build twelve current-relative targets while the full match is available.

    Explicit queries are retained in caller order and receive ``query_mask=False``
    when current-state eligibility fails.  With ``queries=None``, only eligible
    phase-1-through-8 queries are emitted.  Missing coordinates can create a
    transient mask hole; elimination or phase 9 masks that horizon and the
    complete remaining suffix.
    """

    if not isinstance(source, PlannerTargetSource):
        raise TypeError("source must be a PlannerTargetSource")
    normalized = _normalize_queries(source, queries)
    q = len(normalized)
    policy_mask = _resolve_observation_mask(q, observation_query_mask)
    displacements = torch.zeros((q, NUM_HORIZONS, 2), dtype=torch.float32)
    target_mask = torch.zeros((q, NUM_HORIZONS), dtype=torch.bool)
    query_mask = torch.zeros(q, dtype=torch.bool)
    current_xy = torch.zeros((q, 2), dtype=torch.float32)
    future_xy = torch.zeros((q, NUM_HORIZONS, 2), dtype=torch.float32)

    match = source.match
    team_indices = {team_id: index for index, team_id in enumerate(match.team_ids)}
    tick_indices = {
        int(value): index
        for index, value in enumerate(match.absolute_tick_index.cpu().tolist())
    }
    times = match.match_elapsed_s.cpu().double()
    recording_end = float(times[-1].item())
    width = float(source.profile.cell_width_world_units)
    height = float(source.profile.cell_height_world_units)

    for row, query in enumerate(normalized):
        if query.session_id != match.session_id:
            raise ValueError("parallel trajectory query session does not match source")
        if query.team_id not in team_indices:
            raise ValueError("parallel trajectory query team is not in source")
        if query.absolute_tick_index not in tick_indices:
            raise ValueError("parallel trajectory query tick is not in source")
        team = team_indices[query.team_id]
        tick = tick_indices[query.absolute_tick_index]
        eligible, centroid = _query_is_eligible(source, tick, team)
        eligible = eligible and bool(policy_mask[row])
        query_mask[row] = eligible
        if centroid is None:
            continue
        current_xy[row] = centroid.to(torch.float32)
        if not eligible:
            continue

        query_time = float(times[tick].item())
        terminal = False
        for horizon in range(NUM_HORIZONS):
            if terminal:
                break
            target_time = query_time + (horizon + 1) * SAMPLE_INTERVAL_SECONDS
            if target_time > recording_end + _FLOAT_TOLERANCE:
                break
            target_tick = _exact_target_tick(times, target_time)
            if target_tick is None:
                # A sampling gap inside the recording masks only this horizon.
                continue

            sampled_living = _living_count(source, target_tick, team)
            lifecycle_living = _count_at(source, team, target_time)
            if lifecycle_living is None or lifecycle_living != sampled_living:
                # Later lifecycle state cannot be trusted after an ambiguity.
                terminal = True
                continue
            if sampled_living <= 0:
                terminal = True
                continue
            phase = int(match.zone_phase[target_tick].item())
            if phase >= 9:
                terminal = True
                continue
            if not 1 <= phase <= 8:
                # Missing/invalid phase metadata affects this timestamp only.
                continue

            status = source.centroid_status[target_tick][team]
            if status != "valid":
                # Coordinate absence is not itself evidence of termination.
                continue
            xy = source.centroids_xy_uu[target_tick, team].double()
            if not bool(torch.isfinite(xy).all()):
                continue
            if source.profile.cell(xy.tolist()) is None:
                continue
            future_xy[row, horizon] = xy.to(torch.float32)
            displacements[row, horizon, 0] = float(
                (xy[0] - centroid[0]).item() / width
            )
            displacements[row, horizon, 1] = float(
                (xy[1] - centroid[1]).item() / height
            )
            if not bool(torch.isfinite(displacements[row, horizon]).all()):
                displacements[row, horizon].zero_()
                future_xy[row, horizon].zero_()
                continue
            target_mask[row, horizon] = True

    return ParallelTrajectoryTargets(
        target_displacements=displacements,
        target_mask=target_mask,
        query_mask=query_mask,
        current_xy=current_xy,
        future_xy=future_xy,
        queries=normalized,
        world_grid_profile_id=source.profile.profile_id,
        world_grid_profile_hash=source.profile.profile_hash,
    )


__all__ = ["build_parallel_trajectory_targets"]
