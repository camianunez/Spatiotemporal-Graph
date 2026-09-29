from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

from .config import RotationHeadConfig
from .contracts import (
    CollatedRotationSupervision,
    RotationBatchMetadata,
    RotationSupervision,
    RotationTargetMetadata,
    RotationTargets,
    TensorizedMatch,
)
from .tensorize import (
    ROSTER_SIZE,
    SAMPLE_INTERVAL_SECONDS,
    _FLOAT_TOLERANCE,
    _INPUT_SCHEMAS,
    _coordinate_triple,
    _finite,
    _nonnegative_int,
    _validate_phase_rows,
)
from .world_grid import WorldGridProfile


_TEAM_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=True),
        pa.field("team_id", pa.int32(), nullable=False),
        pa.field("match_time_seconds", pa.float64(), nullable=False),
        pa.field("final_placement", pa.int32(), nullable=True),
        pa.field("x", pa.float64(), nullable=True),
        pa.field("y", pa.float64(), nullable=True),
        pa.field("z", pa.float64(), nullable=True),
    ]
)
_PLAYER_EVENT_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=True),
        pa.field("player_id", pa.string(), nullable=True),
        pa.field("team_id", pa.int32(), nullable=False),
        pa.field("match_time_seconds", pa.float64(), nullable=False),
        pa.field("event_type", pa.string(), nullable=True),
        pa.field("life_index", pa.int32(), nullable=False),
        pa.field("x", pa.float64(), nullable=True),
        pa.field("y", pa.float64(), nullable=True),
        pa.field("z", pa.float64(), nullable=True),
        pa.field("zone_phase", pa.int32(), nullable=True),
        pa.field("zone_basis", pa.string(), nullable=True),
        pa.field("location_quality", pa.string(), nullable=True),
    ]
)
_SUPERVISION_SCHEMAS = {
    "team_samples.parquet": _TEAM_SAMPLE_SCHEMA,
    "player_events.parquet": _PLAYER_EVENT_SCHEMA,
    "zone_phases.parquet": _INPUT_SCHEMAS["zone_phases.parquet"],
}
_LIFECYCLE_EVENT_DELTAS = {"death": -1, "reboot": 1}
_LIFECYCLE_TIMESTAMP_EPSILON_SECONDS = 0.001
_EVENT_TYPES = {
    "dbno",
    "death",
    "reboot",
    "zone_initially_inside",
    "zone_entry",
    "zone_reentry",
}


def _read_supervision_table(directory: Path, name: str) -> pa.Table:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"required supervision input is missing: {path}")
    table = pq.read_table(path)
    if not table.schema.equals(_SUPERVISION_SCHEMAS[name], check_metadata=False):
        raise ValueError(f"{name} has an unexpected Parquet schema")
    return table


def _validate_full_match(match: TensorizedMatch) -> None:
    expected_ticks = torch.arange(match.num_timesteps, dtype=torch.int64)
    if (
        match.num_timesteps == 0
        or not torch.equal(match.absolute_tick_index.cpu(), expected_ticks)
        or not math.isclose(
            float(match.match_elapsed_s[0]),
            0.0,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        )
        or not bool(match.time_mask.all())
        or not bool(match.team_slot_mask.all())
    ):
        raise ValueError(
            "rotation supervision must be generated from a full unsliced match"
        )


def _validate_team_rows(
    rows: list[dict[str, Any]],
    match: TensorizedMatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not rows:
        raise ValueError("team_samples.parquet must not be empty")
    if len(rows) != match.num_timesteps * match.num_teams:
        raise ValueError("team_samples must contain every aligned team at every tick")

    team_index = {team_id: index for index, team_id in enumerate(match.team_ids)}
    time_index = {
        float(time): index
        for index, time in enumerate(match.match_elapsed_s.cpu().tolist())
    }
    centroids = torch.zeros(
        (match.num_timesteps, match.num_teams, 2),
        dtype=torch.float64,
    )
    centroid_mask = torch.zeros(
        (match.num_timesteps, match.num_teams),
        dtype=torch.bool,
    )
    placement = torch.zeros(
        (match.num_timesteps, match.num_teams),
        dtype=torch.int64,
    )
    placement_mask = torch.zeros_like(placement, dtype=torch.bool)
    seen: set[tuple[int, float]] = set()

    for row in rows:
        if row["session_id"] != match.session_id:
            raise ValueError("team sample session_id does not match the encoder match")
        team_id = row["team_id"]
        if (
            isinstance(team_id, bool)
            or not isinstance(team_id, int)
            or team_id not in team_index
        ):
            raise ValueError("team sample team_id is not aligned to TensorizedMatch")
        time_seconds = _finite(row["match_time_seconds"], "match_time_seconds")
        if time_seconds not in time_index:
            raise ValueError("team sample tick is not aligned to TensorizedMatch")
        key = (team_id, time_seconds)
        if key in seen:
            raise ValueError("team_samples.parquet contains duplicate rows")
        seen.add(key)
        tick = time_index[time_seconds]
        team = team_index[team_id]

        has_coordinate, xyz = _coordinate_triple(
            row,
            ("x", "y", "z"),
            "team centroid",
        )
        team_alive = bool(
            (
                match.player_alive[tick, team]
                & match.player_slot_mask[team]
            ).any()
        )
        if not team_alive and has_coordinate:
            raise ValueError("eliminated teams cannot have a centroid")
        if has_coordinate:
            centroids[tick, team] = torch.tensor(xyz[:2], dtype=torch.float64)
            centroid_mask[tick, team] = True

        value = row["final_placement"]
        if (
            value is not None
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        ):
            placement[tick, team] = _placement_bucket(value)
            placement_mask[tick, team] = True

    if len(seen) != match.num_timesteps * match.num_teams:
        raise ValueError("team_samples must align to all encoder teams and ticks")
    return centroids, centroid_mask, placement, placement_mask


def _validate_event_rows(
    rows: list[dict[str, Any]],
    match: TensorizedMatch,
) -> dict[int, list[tuple[float, int]]]:
    team_ids = set(match.team_ids)
    deltas = {team_id: [] for team_id in match.team_ids}
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row["session_id"] != match.session_id:
            raise ValueError("player event session_id does not match the encoder match")
        player_id = row["player_id"]
        if not isinstance(player_id, str) or not player_id:
            raise ValueError("player event player_id must be a nonempty string")
        team_id = row["team_id"]
        if (
            isinstance(team_id, bool)
            or not isinstance(team_id, int)
            or team_id not in team_ids
        ):
            raise ValueError("player event team_id is not aligned to TensorizedMatch")
        time_seconds = _finite(row["match_time_seconds"], "match_time_seconds")
        if time_seconds < 0.0:
            raise ValueError("player event match_time_seconds must be nonnegative")
        event_type = row["event_type"]
        if event_type not in _EVENT_TYPES:
            raise ValueError("player event has an unsupported event_type")
        life_index = _nonnegative_int(row["life_index"], "life_index")
        has_coordinate, _ = _coordinate_triple(
            row,
            ("x", "y", "z"),
            "player event location",
        )
        if event_type == "death" and not has_coordinate:
            raise ValueError("death events require a complete location")
        if event_type in {"dbno", "reboot"} and has_coordinate:
            raise ValueError("dbno and reboot events must not contain coordinates")

        zone_phase = row["zone_phase"]
        if zone_phase is not None:
            _nonnegative_int(zone_phase, "zone_phase")
        for field in ("zone_basis", "location_quality"):
            value = row[field]
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be a nonempty string or null")
        key = (
            player_id,
            time_seconds,
            event_type,
            life_index,
            zone_phase,
            row["zone_basis"],
        )
        if key in seen:
            raise ValueError("player_events.parquet contains duplicate logical rows")
        seen.add(key)
        if event_type in _LIFECYCLE_EVENT_DELTAS:
            deltas[team_id].append(
                (time_seconds, _LIFECYCLE_EVENT_DELTAS[event_type])
            )
    for values in deltas.values():
        values.sort(key=lambda item: (item[0], item[1]))
    _validate_lifecycle_alignment(match, deltas)
    return deltas


def _living_count(match: TensorizedMatch, tick: int, team: int) -> int:
    return int(
        (
            match.player_alive[tick, team]
            & match.player_slot_mask[team]
        ).sum()
    )


def _validate_lifecycle_alignment(
    match: TensorizedMatch,
    deltas: dict[int, list[tuple[float, int]]],
) -> None:
    times = match.match_elapsed_s.cpu().tolist()
    for team, team_id in enumerate(match.team_ids):
        count = _living_count(match, 0, team)
        roster_count = int(match.player_slot_mask[team].sum())
        previous_time = float(times[0])
        events = deltas[team_id]
        event_index = 0
        while (
            event_index < len(events)
            and events[event_index][0]
            <= previous_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
        ):
            event_index += 1
        for tick in range(1, match.num_timesteps):
            current_time = float(times[tick])
            while (
                event_index < len(events)
                and events[event_index][0]
                <= current_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
            ):
                event_time = events[event_index][0]
                net_delta = 0
                while (
                    event_index < len(events)
                    and math.isclose(
                        events[event_index][0],
                        event_time,
                        rel_tol=0.0,
                        abs_tol=_FLOAT_TOLERANCE,
                    )
                ):
                    net_delta += events[event_index][1]
                    event_index += 1
                count += net_delta
                if not 0 <= count <= roster_count:
                    raise ValueError("lifecycle events produce an invalid living count")
            if count != _living_count(match, tick, team):
                raise ValueError(
                    "lifecycle events do not align with sampled living membership"
                )
            previous_time = current_time
        while event_index < len(events):
            event_time = events[event_index][0]
            net_delta = 0
            while (
                event_index < len(events)
                and math.isclose(
                    events[event_index][0],
                    event_time,
                    rel_tol=0.0,
                    abs_tol=_FLOAT_TOLERANCE,
                )
            ):
                net_delta += events[event_index][1]
                event_index += 1
            count += net_delta
            if not 0 <= count <= roster_count:
                raise ValueError("lifecycle events produce an invalid living count")


def _count_at(
    match: TensorizedMatch,
    team: int,
    target_time: float,
    deltas: dict[int, list[tuple[float, int]]],
) -> int:
    times = match.match_elapsed_s.cpu()
    eligible = (times <= target_time + _FLOAT_TOLERANCE).nonzero(
        as_tuple=False
    )
    if eligible.numel() == 0:
        raise ValueError("target time precedes the sampled match")
    tick = int(eligible[-1].item())
    count = _living_count(match, tick, team)
    start_time = float(times[tick])
    for event_time, delta in deltas[match.team_ids[team]]:
        if (
            event_time
            > start_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
            and event_time
            <= target_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
        ):
            count += delta
    return count


def _first_elimination_time(
    match: TensorizedMatch,
    team: int,
    query_time: float,
    deadline: float,
    deltas: dict[int, list[tuple[float, int]]],
) -> float | None:
    count = _count_at(match, team, query_time, deltas)
    events = deltas[match.team_ids[team]]
    event_index = 0
    while event_index < len(events):
        event_time = events[event_index][0]
        if (
            event_time
            <= query_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
        ):
            event_index += 1
            continue
        if math.isfinite(deadline) and event_time >= deadline - _FLOAT_TOLERANCE:
            break
        net_delta = 0
        while (
            event_index < len(events)
            and math.isclose(
                events[event_index][0],
                event_time,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            )
        ):
            net_delta += events[event_index][1]
            event_index += 1
        count += net_delta
        if count == 0:
            return event_time
    return None


def _placement_bucket(placement: int) -> int:
    if placement == 1:
        return 0
    if placement <= 5:
        return 1
    if placement <= 10:
        return 2
    if placement <= 25:
        return 3
    return 4


def _map_cell(
    world_xy: Tensor,
    profile: WorldGridProfile,
    config: RotationHeadConfig,
) -> int | None:
    if (
        profile.grid_rows != config.grid_height
        or profile.grid_columns != config.grid_width
    ):
        raise ValueError("WorldGridProfile dimensions do not match head configuration")
    return profile.cell((float(world_xy[0]), float(world_xy[1])))


def _entry_crossing_fraction(
    start: Tensor,
    end: Tensor,
    center: Tensor,
    radius: float,
) -> float | None:
    offset = start - center
    direction = end - start
    a = float(torch.dot(direction, direction))
    if a <= 0.0:
        return None
    b = 2.0 * float(torch.dot(offset, direction))
    c = float(torch.dot(offset, offset)) - radius * radius
    if c <= 0.0:
        return None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = math.sqrt(max(discriminant, 0.0))
    for fraction in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
        if -_FLOAT_TOLERANCE <= fraction <= 1.0 + _FLOAT_TOLERANCE:
            point = offset + min(max(fraction, 0.0), 1.0) * direction
            if float(torch.dot(point, direction)) < 0.0:
                return min(max(fraction, 0.0), 1.0)
    return None


def _has_lifecycle_discontinuity(
    team_id: int,
    start_time: float,
    end_time: float,
    deltas: dict[int, list[tuple[float, int]]],
) -> bool:
    return any(
        event_time
        > start_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
        and event_time
        <= end_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
        for event_time, _ in deltas[team_id]
    )


def _entry_label(
    match: TensorizedMatch,
    tick: int,
    team: int,
    centroids: Tensor,
    centroid_mask: Tensor,
    phases: dict[int, dict[str, Any]],
    deltas: dict[int, list[tuple[float, int]]],
    config: RotationHeadConfig,
) -> tuple[int, bool]:
    if (
        not _living_count(match, tick, team)
        or not bool(centroid_mask[tick, team])
        or not bool(match.zone_mask[tick])
    ):
        return 0, False
    circle = match.target_circle_uu[tick].to(dtype=torch.float64)
    radius = float(circle[3])
    phase = int(match.zone_phase[tick])
    if radius <= 0.0 or phase not in phases:
        return 0, False

    center = circle[:2]
    query_centroid = centroids[tick, team]
    if float(torch.linalg.vector_norm(query_centroid - center)) <= radius:
        return config.already_inside_index, True

    next_phase = phases.get(phase + 1)
    deadline = (
        float(next_phase["activation_time_seconds"])
        if next_phase is not None
        else math.inf
    )
    times = match.match_elapsed_s.cpu().tolist()
    query_time = float(times[tick])
    elimination_time = _first_elimination_time(
        match,
        team,
        query_time,
        deadline,
        deltas,
    )
    for start_tick in range(tick, match.num_timesteps - 1):
        start_time = float(times[start_tick])
        end_time = float(times[start_tick + 1])
        if start_time >= deadline - _FLOAT_TOLERANCE:
            break
        if (
            elimination_time is not None
            and start_time >= elimination_time - _FLOAT_TOLERANCE
        ):
            break
        if not (
            bool(centroid_mask[start_tick, team])
            and bool(centroid_mask[start_tick + 1, team])
        ):
            continue
        roster = match.player_slot_mask[team]
        if not torch.equal(
            match.player_alive[start_tick, team][roster],
            match.player_alive[start_tick + 1, team][roster],
        ):
            continue
        if not torch.equal(
            match.life_index[start_tick, team][roster],
            match.life_index[start_tick + 1, team][roster],
        ):
            continue
        if _has_lifecycle_discontinuity(
            match.team_ids[team],
            start_time,
            end_time,
            deltas,
        ):
            continue
        fraction = _entry_crossing_fraction(
            centroids[start_tick, team],
            centroids[start_tick + 1, team],
            center,
            radius,
        )
        if fraction is None:
            continue
        crossing_time = start_time + fraction * (end_time - start_time)
        if crossing_time >= deadline - _FLOAT_TOLERANCE:
            continue
        if (
            elimination_time is not None
            and crossing_time >= elimination_time - _FLOAT_TOLERANCE
        ):
            continue
        crossing = (
            centroids[start_tick, team]
            + fraction
            * (centroids[start_tick + 1, team] - centroids[start_tick, team])
        )
        angle = math.atan2(
            float(crossing[1] - center[1]),
            float(crossing[0] - center[0]),
        ) % (2.0 * math.pi)
        angle_bin = int(
            math.floor(angle * config.entry_angle_bins / (2.0 * math.pi))
        )
        return min(angle_bin, config.entry_angle_bins - 1), True

    if elimination_time is not None:
        return config.eliminated_before_entry_index, True
    return config.no_valid_entry_index, True


def _build_targets(
    match: TensorizedMatch,
    centroids: Tensor,
    centroid_mask: Tensor,
    placement: Tensor,
    placement_data_mask: Tensor,
    phases: dict[int, dict[str, Any]],
    deltas: dict[int, list[tuple[float, int]]],
    profile: WorldGridProfile,
    config: RotationHeadConfig,
) -> RotationTargets:
    t, n, h = match.num_timesteps, match.num_teams, config.num_horizons
    future_position = torch.zeros((t, n, h), dtype=torch.int64)
    future_position_mask = torch.zeros((t, n, h), dtype=torch.bool)
    zone_entry = torch.zeros((t, n), dtype=torch.int64)
    zone_entry_mask = torch.zeros((t, n), dtype=torch.bool)
    survival = (
        torch.zeros((t, n), dtype=torch.float32)
        if config.enable_survival
        else None
    )
    survival_mask = (
        torch.zeros((t, n), dtype=torch.bool)
        if config.enable_survival
        else None
    )
    placement_targets = placement.clone() if config.enable_placement else None
    placement_mask = (
        torch.zeros((t, n), dtype=torch.bool)
        if config.enable_placement
        else None
    )
    times = match.match_elapsed_s.cpu().tolist()
    phase_activations = sorted(
        float(row["activation_time_seconds"]) for row in phases.values()
    )

    for tick in range(t):
        query_time = float(times[tick])
        for team in range(n):
            query_alive = _living_count(match, tick, team) > 0
            if not query_alive:
                continue

            for horizon_index, horizon in enumerate(config.horizons_seconds):
                target_time = query_time + float(horizon)
                living_at_target = _count_at(match, team, target_time, deltas)
                if living_at_target == 0:
                    future_position[tick, team, horizon_index] = (
                        config.eliminated_position_index
                    )
                    future_position_mask[tick, team, horizon_index] = True
                    continue
                target_tick = tick + int(horizon / SAMPLE_INTERVAL_SECONDS)
                if target_tick >= t:
                    continue
                if not bool(centroid_mask[target_tick, team]):
                    continue
                cell = _map_cell(
                    centroids[target_tick, team],
                    profile,
                    config,
                )
                if cell is not None:
                    future_position[tick, team, horizon_index] = cell
                    future_position_mask[tick, team, horizon_index] = True

            entry, entry_valid = _entry_label(
                match,
                tick,
                team,
                centroids,
                centroid_mask,
                phases,
                deltas,
                config,
            )
            zone_entry[tick, team] = entry
            zone_entry_mask[tick, team] = entry_valid

            if survival is not None and survival_mask is not None:
                next_activation = next(
                    (
                        activation
                        for activation in phase_activations
                        if activation > query_time + _FLOAT_TOLERANCE
                    ),
                    None,
                )
                if next_activation is not None:
                    survival[tick, team] = float(
                        _count_at(match, team, next_activation, deltas) > 0
                    )
                    survival_mask[tick, team] = True

            if placement_mask is not None:
                placement_mask[tick, team] = placement_data_mask[tick, team]

    return RotationTargets(
        future_position=future_position,
        future_position_mask=future_position_mask,
        zone_entry=zone_entry,
        zone_entry_mask=zone_entry_mask,
        survival=survival,
        survival_mask=survival_mask,
        placement=placement_targets,
        placement_mask=placement_mask,
    )


def load_rotation_supervision(
    path: str | Path,
    match: TensorizedMatch,
    profile: WorldGridProfile,
    config: RotationHeadConfig | None = None,
) -> RotationSupervision:
    """Load target-bearing v2 tables and build full-match labels explicitly."""

    if not isinstance(profile, WorldGridProfile):
        raise TypeError("profile must be a WorldGridProfile")
    resolved_config = config or RotationHeadConfig()
    _validate_full_match(match)
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"match session directory does not exist: {directory}")
    tables = {
        name: _read_supervision_table(directory, name)
        for name in (
            "team_samples.parquet",
            "player_events.parquet",
            "zone_phases.parquet",
        )
    }
    team_rows = tables["team_samples.parquet"].to_pylist()
    event_rows = tables["player_events.parquet"].to_pylist()
    phase_rows = tables["zone_phases.parquet"].to_pylist()
    for row in phase_rows:
        if row["session_id"] != match.session_id:
            raise ValueError("zone phase session_id does not match the encoder match")
    phases = _validate_phase_rows(phase_rows)
    centroids, centroid_mask, placement, placement_mask = _validate_team_rows(
        team_rows,
        match,
    )
    deltas = _validate_event_rows(event_rows, match)
    targets = _build_targets(
        match,
        centroids,
        centroid_mask,
        placement,
        placement_mask,
        phases,
        deltas,
        profile,
        resolved_config,
    )
    return RotationSupervision(
        targets=targets,
        absolute_tick_index=match.absolute_tick_index.clone(),
        metadata=RotationTargetMetadata(
            session_id=match.session_id,
            team_ids=match.team_ids,
            world_grid_profile_id=profile.profile_id,
            world_grid_profile_hash=profile.profile_hash,
        ),
    )


def _slice_targets(targets: RotationTargets, start: int, stop: int) -> RotationTargets:
    return RotationTargets(
        future_position=targets.future_position[start:stop].clone(),
        future_position_mask=targets.future_position_mask[start:stop].clone(),
        zone_entry=targets.zone_entry[start:stop].clone(),
        zone_entry_mask=targets.zone_entry_mask[start:stop].clone(),
        survival=(
            targets.survival[start:stop].clone()
            if targets.survival is not None
            else None
        ),
        survival_mask=(
            targets.survival_mask[start:stop].clone()
            if targets.survival_mask is not None
            else None
        ),
        placement=(
            targets.placement[start:stop].clone()
            if targets.placement is not None
            else None
        ),
        placement_mask=(
            targets.placement_mask[start:stop].clone()
            if targets.placement_mask is not None
            else None
        ),
    )


def slice_rotation_supervision(
    supervision: RotationSupervision,
    start_tick: int,
    length: int,
) -> RotationSupervision:
    if isinstance(start_tick, bool) or not isinstance(start_tick, int):
        raise TypeError("start_tick must be an integer absolute tick")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    matches = (
        supervision.absolute_tick_index == start_tick
    ).nonzero(as_tuple=False)
    if matches.numel() != 1:
        raise ValueError("start_tick is not present in this supervision")
    start = int(matches.item())
    stop = start + length
    if stop > supervision.num_timesteps:
        raise ValueError("requested window extends beyond the supervision")
    return RotationSupervision(
        targets=_slice_targets(supervision.targets, start, stop),
        absolute_tick_index=supervision.absolute_tick_index[start:stop].clone(),
        metadata=supervision.metadata,
    )


def collate_rotation_supervision(
    supervision: Iterable[RotationSupervision],
) -> CollatedRotationSupervision:
    items = tuple(supervision)
    if not items:
        raise ValueError("at least one rotation supervision item is required")
    horizon_count = items[0].targets.future_position.shape[-1]
    if any(item.targets.future_position.shape[-1] != horizon_count for item in items):
        raise ValueError("all supervision items must use the same horizons")
    survival_enabled = items[0].targets.survival is not None
    placement_enabled = items[0].targets.placement is not None
    if any(
        (item.targets.survival is not None) != survival_enabled
        or (item.targets.placement is not None) != placement_enabled
        for item in items
    ):
        raise ValueError("all supervision items must enable the same auxiliary heads")

    max_t = max(item.num_timesteps for item in items)
    max_n = max(item.num_teams for item in items)
    b = len(items)
    future_position = torch.zeros(
        (b, max_t, max_n, horizon_count),
        dtype=torch.int64,
    )
    future_position_mask = torch.zeros_like(future_position, dtype=torch.bool)
    zone_entry = torch.zeros((b, max_t, max_n), dtype=torch.int64)
    zone_entry_mask = torch.zeros_like(zone_entry, dtype=torch.bool)
    survival = (
        torch.zeros((b, max_t, max_n), dtype=torch.float32)
        if survival_enabled
        else None
    )
    survival_mask = (
        torch.zeros((b, max_t, max_n), dtype=torch.bool)
        if survival_enabled
        else None
    )
    placement = (
        torch.zeros((b, max_t, max_n), dtype=torch.int64)
        if placement_enabled
        else None
    )
    placement_mask = (
        torch.zeros((b, max_t, max_n), dtype=torch.bool)
        if placement_enabled
        else None
    )

    for batch, item in enumerate(items):
        t, n = item.num_timesteps, item.num_teams
        targets = item.targets
        if targets.future_position.shape[:2] != (t, n):
            raise ValueError("rotation supervision has invalid target shapes")
        future_position[batch, :t, :n] = targets.future_position
        future_position_mask[batch, :t, :n] = targets.future_position_mask
        zone_entry[batch, :t, :n] = targets.zone_entry
        zone_entry_mask[batch, :t, :n] = targets.zone_entry_mask
        if survival is not None and survival_mask is not None:
            assert targets.survival is not None
            assert targets.survival_mask is not None
            survival[batch, :t, :n] = targets.survival
            survival_mask[batch, :t, :n] = targets.survival_mask
        if placement is not None and placement_mask is not None:
            assert targets.placement is not None
            assert targets.placement_mask is not None
            placement[batch, :t, :n] = targets.placement
            placement_mask[batch, :t, :n] = targets.placement_mask

    return CollatedRotationSupervision(
        targets=RotationTargets(
            future_position=future_position,
            future_position_mask=future_position_mask,
            zone_entry=zone_entry,
            zone_entry_mask=zone_entry_mask,
            survival=survival,
            survival_mask=survival_mask,
            placement=placement,
            placement_mask=placement_mask,
        ),
        metadata=RotationBatchMetadata(
            session_ids=tuple(item.metadata.session_id for item in items),
            team_ids=tuple(item.metadata.team_ids for item in items),
            world_grid_profile_ids=tuple(
                item.metadata.world_grid_profile_id for item in items
            ),
            world_grid_profile_hashes=tuple(
                item.metadata.world_grid_profile_hash for item in items
            ),
        ),
    )


load_supervision = load_rotation_supervision
slice_supervision = slice_rotation_supervision
collate_supervision = collate_rotation_supervision
load_rotation_targets = load_rotation_supervision
slice_rotation_targets = slice_rotation_supervision
collate_rotation_targets = collate_rotation_supervision
