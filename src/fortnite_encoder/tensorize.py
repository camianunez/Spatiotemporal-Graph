from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .contracts import (
    BatchMetadata,
    CollatedEncoderInput,
    EncoderBatch,
    TensorizedMatch,
)


ROSTER_SIZE = 2
SAMPLE_INTERVAL_SECONDS = 5.0
_FLOAT_TOLERANCE = 1e-6

_MATCH_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=True),
        pa.field("match_time_seconds", pa.float64(), nullable=False),
        pa.field("players_remaining", pa.int32(), nullable=False),
        pa.field("teams_remaining", pa.int32(), nullable=False),
    ]
)
_PLAYER_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=True),
        pa.field("player_id", pa.string(), nullable=True),
        pa.field("team_id", pa.int32(), nullable=False),
        pa.field("match_time_seconds", pa.float64(), nullable=False),
        pa.field("alive", pa.bool_(), nullable=False),
        pa.field("life_index", pa.int32(), nullable=False),
        pa.field("x", pa.float64(), nullable=True),
        pa.field("y", pa.float64(), nullable=True),
        pa.field("z", pa.float64(), nullable=True),
        pa.field("target_zone_offset_x", pa.float64(), nullable=True),
        pa.field("target_zone_offset_y", pa.float64(), nullable=True),
        pa.field("target_zone_inside", pa.bool_(), nullable=True),
        pa.field(
            "target_zone_edge_distance_normalized", pa.float64(), nullable=True
        ),
        pa.field("current_boundary_offset_x", pa.float64(), nullable=True),
        pa.field("current_boundary_offset_y", pa.float64(), nullable=True),
        pa.field("current_boundary_inside", pa.bool_(), nullable=True),
        pa.field(
            "current_boundary_edge_distance_normalized",
            pa.float64(),
            nullable=True,
        ),
    ]
)
_ZONE_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=True),
        pa.field("match_time_seconds", pa.float64(), nullable=False),
        pa.field("zone_phase", pa.int32(), nullable=True),
        pa.field("time_until_closure_seconds", pa.float64(), nullable=True),
        pa.field("current_boundary_x", pa.float64(), nullable=True),
        pa.field("current_boundary_y", pa.float64(), nullable=True),
        pa.field("current_boundary_z", pa.float64(), nullable=True),
        pa.field("current_boundary_radius", pa.float64(), nullable=True),
        pa.field("target_x", pa.float64(), nullable=True),
        pa.field("target_y", pa.float64(), nullable=True),
        pa.field("target_z", pa.float64(), nullable=True),
        pa.field("target_radius", pa.float64(), nullable=True),
    ]
)
_ZONE_PHASE_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=True),
        pa.field("zone_phase", pa.int32(), nullable=False),
        pa.field("activation_time_seconds", pa.float64(), nullable=False),
        pa.field("shrink_start_time_seconds", pa.float64(), nullable=False),
        pa.field("closure_time_seconds", pa.float64(), nullable=False),
        pa.field("source_x", pa.float64(), nullable=False),
        pa.field("source_y", pa.float64(), nullable=False),
        pa.field("source_z", pa.float64(), nullable=False),
        pa.field("source_radius", pa.float64(), nullable=False),
        pa.field("target_x", pa.float64(), nullable=False),
        pa.field("target_y", pa.float64(), nullable=False),
        pa.field("target_z", pa.float64(), nullable=False),
        pa.field("target_radius", pa.float64(), nullable=False),
    ]
)
_INPUT_SCHEMAS = {
    "match_samples.parquet": _MATCH_SCHEMA,
    "player_samples.parquet": _PLAYER_SCHEMA,
    "zone_samples.parquet": _ZONE_SAMPLE_SCHEMA,
    "zone_phases.parquet": _ZONE_PHASE_SCHEMA,
}


def _read_table(directory: Path, name: str) -> pa.Table:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"required encoder input is missing: {path}")
    table = pq.read_table(path)
    if not table.schema.equals(_INPUT_SCHEMAS[name], check_metadata=False):
        raise ValueError(f"{name} has an unexpected Parquet schema")
    return table


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _session_id(tables: Iterable[pa.Table]) -> str:
    ids: set[str] = set()
    for table in tables:
        for value in table.column("session_id").to_pylist():
            if not isinstance(value, str) or not value:
                raise ValueError("session_id values must be nonempty strings")
            ids.add(value)
    if len(ids) != 1:
        raise ValueError("all encoder input rows must have the same session_id")
    return next(iter(ids))


def _validate_match_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[float, int]]:
    if not rows:
        raise ValueError("match_samples.parquet must not be empty")
    if len({row["match_time_seconds"] for row in rows}) != len(rows):
        raise ValueError("match_samples.parquet contains duplicate ticks")
    rows.sort(key=lambda row: row["match_time_seconds"])
    for index, row in enumerate(rows):
        actual = _finite(row["match_time_seconds"], "match_time_seconds")
        expected = index * SAMPLE_INTERVAL_SECONDS
        if not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE
        ):
            raise ValueError(
                "match samples must use the exact five-second grid beginning at zero"
            )
        _nonnegative_int(row["players_remaining"], "players_remaining")
        _nonnegative_int(row["teams_remaining"], "teams_remaining")
    return rows, {
        float(row["match_time_seconds"]): index for index, row in enumerate(rows)
    }


def _coordinate_triple(
    row: dict[str, Any], names: tuple[str, str, str], label: str
) -> tuple[bool, tuple[float, float, float]]:
    values = tuple(row[name] for name in names)
    if all(value is None for value in values):
        return False, (0.0, 0.0, 0.0)
    if any(value is None for value in values):
        raise ValueError(f"{label} must be a complete XYZ triple or entirely null")
    return True, tuple(_finite(value, f"{label}.{name}") for name, value in zip(names, values))  # type: ignore[return-value]


def _validate_unused_player_values(row: dict[str, Any]) -> None:
    for name in (
        "target_zone_offset_x",
        "target_zone_offset_y",
        "target_zone_edge_distance_normalized",
        "current_boundary_offset_x",
        "current_boundary_offset_y",
        "current_boundary_edge_distance_normalized",
    ):
        if row[name] is not None:
            _finite(row[name], name)
    for name in ("target_zone_inside", "current_boundary_inside"):
        if row[name] is not None and not isinstance(row[name], bool):
            raise ValueError(f"{name} must be boolean or null")


def _validate_phase_rows(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not rows:
        raise ValueError("zone_phases.parquet must not be empty")
    by_phase: dict[int, dict[str, Any]] = {}
    numeric_names = (
        "activation_time_seconds",
        "shrink_start_time_seconds",
        "closure_time_seconds",
        "source_x",
        "source_y",
        "source_z",
        "source_radius",
        "target_x",
        "target_y",
        "target_z",
        "target_radius",
    )
    for row in rows:
        phase = _nonnegative_int(row["zone_phase"], "zone_phase")
        if phase == 0 or phase in by_phase:
            raise ValueError("zone phases must be unique positive integers")
        for name in numeric_names:
            _finite(row[name], name)
        activation = float(row["activation_time_seconds"])
        shrink = float(row["shrink_start_time_seconds"])
        closure = float(row["closure_time_seconds"])
        if not activation <= shrink < closure:
            raise ValueError("zone phase times must satisfy activation <= shrink < closure")
        if row["source_radius"] < 0 or row["target_radius"] < 0:
            raise ValueError("zone radii must be nonnegative")
        by_phase[phase] = row

    ordered = sorted(by_phase)
    if ordered != list(range(1, len(ordered) + 1)):
        raise ValueError("zone phases must be consecutively numbered from one")
    for previous, current in zip(ordered, ordered[1:]):
        if not math.isclose(
            float(by_phase[previous]["closure_time_seconds"]),
            float(by_phase[current]["activation_time_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            raise ValueError("each zone phase must activate at the previous closure")
    return by_phase


def _active_phase_at(
    time_seconds: float, phases: dict[int, dict[str, Any]]
) -> int:
    active = [
        phase
        for phase, row in phases.items()
        if float(row["activation_time_seconds"]) <= time_seconds + _FLOAT_TOLERANCE
    ]
    return max(active, default=0)


def _validate_circle_sample(
    row: dict[str, Any],
    phase_row: dict[str, Any],
    time_seconds: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    names = (
        "current_boundary_x",
        "current_boundary_y",
        "current_boundary_z",
        "current_boundary_radius",
        "target_x",
        "target_y",
        "target_z",
        "target_radius",
    )
    if any(row[name] is None for name in names):
        raise ValueError("active zone rows require complete current and target circles")
    values = tuple(_finite(row[name], name) for name in names)
    current = values[:4]
    target = values[4:]
    if current[3] < 0 or target[3] < 0:
        raise ValueError("zone radii must be nonnegative")
    expected_target = tuple(
        float(phase_row[name])
        for name in ("target_x", "target_y", "target_z", "target_radius")
    )
    if any(
        not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-3)
        for actual, expected in zip(target, expected_target)
    ):
        raise ValueError("zone sample target does not match its active phase")
    closure = float(phase_row["closure_time_seconds"])
    countdown = row["time_until_closure_seconds"]
    if countdown is None or not math.isclose(
        _finite(countdown, "time_until_closure_seconds"),
        max(closure - time_seconds, 0.0),
        rel_tol=1e-7,
        abs_tol=1e-3,
    ):
        raise ValueError("zone closure countdown is inconsistent with its active phase")
    return current, target


def load_match_session(path: str | Path) -> TensorizedMatch:
    """Load and validate one v2 session without opening target-bearing tables."""

    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"match session directory does not exist: {directory}")
    tables = {
        name: _read_table(directory, name)
        for name in (
            "match_samples.parquet",
            "player_samples.parquet",
            "zone_samples.parquet",
            "zone_phases.parquet",
        )
    }
    session_id = _session_id(tables.values())
    match_rows, time_to_index = _validate_match_rows(
        tables["match_samples.parquet"].to_pylist()
    )
    player_rows = tables["player_samples.parquet"].to_pylist()
    zone_rows = tables["zone_samples.parquet"].to_pylist()
    phases = _validate_phase_rows(tables["zone_phases.parquet"].to_pylist())

    if not player_rows:
        raise ValueError("player_samples.parquet must not be empty")
    player_team: dict[str, int] = {}
    seen_player_ticks: set[tuple[str, float]] = set()
    for row in player_rows:
        player_id = row["player_id"]
        if not isinstance(player_id, str) or not player_id:
            raise ValueError("player_id values must be nonempty strings")
        team_id = row["team_id"]
        if isinstance(team_id, bool) or not isinstance(team_id, int):
            raise ValueError("team_id must be an integer")
        if player_id in player_team and player_team[player_id] != team_id:
            raise ValueError("a player_id cannot change teams within a match")
        player_team[player_id] = team_id
        time_seconds = _finite(row["match_time_seconds"], "match_time_seconds")
        if time_seconds not in time_to_index:
            raise ValueError("player sample tick does not join to match_samples")
        key = (player_id, time_seconds)
        if key in seen_player_ticks:
            raise ValueError("player_samples.parquet contains duplicate rows")
        seen_player_ticks.add(key)
        _validate_unused_player_values(row)

    expected_player_ticks = {
        (player_id, time_seconds)
        for player_id in player_team
        for time_seconds in time_to_index
    }
    if seen_player_ticks != expected_player_ticks:
        raise ValueError("every roster player must have exactly one row at every tick")

    players_by_team: dict[int, list[str]] = {}
    for player_id, team_id in player_team.items():
        players_by_team.setdefault(team_id, []).append(player_id)
    if any(len(players) > ROSTER_SIZE for players in players_by_team.values()):
        raise ValueError("a team exceeds configured duo capacity")
    team_ids = tuple(sorted(players_by_team))
    if not team_ids:
        raise ValueError("the match roster must contain at least one team")
    for players in players_by_team.values():
        players.sort()
    team_index = {team_id: index for index, team_id in enumerate(team_ids)}
    player_slot = {
        player_id: (team_index[team_id], slot)
        for team_id, players in players_by_team.items()
        for slot, player_id in enumerate(players)
    }

    t = len(match_rows)
    n = len(team_ids)
    player_xyz = torch.zeros((t, n, ROSTER_SIZE, 3), dtype=torch.float32)
    player_alive = torch.zeros((t, n, ROSTER_SIZE), dtype=torch.bool)
    player_coord = torch.zeros((t, n, ROSTER_SIZE), dtype=torch.bool)
    life_index = torch.zeros((t, n, ROSTER_SIZE), dtype=torch.int64)
    player_slot_mask = torch.zeros((n, ROSTER_SIZE), dtype=torch.bool)
    for team_id, players in players_by_team.items():
        player_slot_mask[team_index[team_id], : len(players)] = True

    rows_by_player: dict[str, list[dict[str, Any]]] = {
        player_id: [] for player_id in player_team
    }
    for row in player_rows:
        rows_by_player[row["player_id"]].append(row)
        tick = time_to_index[float(row["match_time_seconds"])]
        team, slot = player_slot[row["player_id"]]
        alive = row["alive"]
        if not isinstance(alive, bool):
            raise ValueError("alive must be boolean")
        life = _nonnegative_int(row["life_index"], "life_index")
        has_coord, xyz = _coordinate_triple(row, ("x", "y", "z"), "player position")
        if not alive and has_coord:
            raise ValueError("dead player coordinates must be null")
        player_alive[tick, team, slot] = alive
        player_coord[tick, team, slot] = has_coord
        life_index[tick, team, slot] = life
        if has_coord:
            player_xyz[tick, team, slot] = torch.tensor(xyz, dtype=torch.float32)

    for player_id, rows in rows_by_player.items():
        rows.sort(key=lambda row: row["match_time_seconds"])
        previous: dict[str, Any] | None = None
        for row in rows:
            if previous is not None:
                old_life = int(previous["life_index"])
                new_life = int(row["life_index"])
                if new_life < old_life:
                    raise ValueError("life_index cannot decrease")
                if not previous["alive"] and row["alive"] and new_life == old_life:
                    raise ValueError("a dead-to-alive transition must increment life_index")
                if new_life > old_life and not row["alive"]:
                    raise ValueError("a new life must begin alive")
            previous = row

    for tick, row in enumerate(match_rows):
        alive = player_alive[tick] & player_slot_mask
        derived_players = int(alive.sum())
        derived_teams = int(alive.any(dim=-1).sum())
        if (
            row["players_remaining"] != derived_players
            or row["teams_remaining"] != derived_teams
        ):
            raise ValueError("match survival counts must be lifecycle-derived")

    if len(zone_rows) != t:
        raise ValueError("zone_samples must contain exactly one row per match tick")
    if len({row["match_time_seconds"] for row in zone_rows}) != len(zone_rows):
        raise ValueError("zone_samples.parquet contains duplicate ticks")
    zone_by_time = {
        _finite(row["match_time_seconds"], "match_time_seconds"): row
        for row in zone_rows
    }
    if set(zone_by_time) != set(time_to_index):
        raise ValueError("zone samples must join completely to match samples")

    current_circle = torch.zeros((t, 4), dtype=torch.float32)
    target_circle = torch.zeros((t, 4), dtype=torch.float32)
    zone_mask = torch.zeros(t, dtype=torch.bool)
    zone_phase = torch.zeros(t, dtype=torch.int64)
    phase_times = torch.zeros((t, 3), dtype=torch.float32)
    for time_seconds, tick in time_to_index.items():
        row = zone_by_time[time_seconds]
        expected_phase = _active_phase_at(time_seconds, phases)
        actual_phase = row["zone_phase"]
        if expected_phase == 0:
            if actual_phase is not None:
                raise ValueError("zone_phase must be null before phase one activates")
            nullable_names = (
                "time_until_closure_seconds",
                "current_boundary_x",
                "current_boundary_y",
                "current_boundary_z",
                "current_boundary_radius",
                "target_x",
                "target_y",
                "target_z",
                "target_radius",
            )
            if any(row[name] is not None for name in nullable_names):
                raise ValueError("inactive zone rows cannot expose circle geometry")
            continue
        if actual_phase != expected_phase:
            raise ValueError("zone sample does not identify the currently active phase")
        phase_row = phases[expected_phase]
        current, target = _validate_circle_sample(row, phase_row, time_seconds)
        zone_mask[tick] = True
        zone_phase[tick] = expected_phase
        current_circle[tick] = torch.tensor(current, dtype=torch.float32)
        target_circle[tick] = torch.tensor(target, dtype=torch.float32)
        phase_times[tick] = torch.tensor(
            (
                phase_row["activation_time_seconds"],
                phase_row["shrink_start_time_seconds"],
                phase_row["closure_time_seconds"],
            ),
            dtype=torch.float32,
        )

    return TensorizedMatch(
        session_id=session_id,
        team_ids=team_ids,
        player_xyz_uu=player_xyz,
        player_alive=player_alive,
        player_coord_mask=player_coord,
        life_index=life_index,
        player_slot_mask=player_slot_mask,
        current_circle_uu=current_circle,
        target_circle_uu=target_circle,
        zone_mask=zone_mask,
        zone_phase=zone_phase,
        phase_times_s=phase_times,
        match_elapsed_s=torch.tensor(
            [row["match_time_seconds"] for row in match_rows], dtype=torch.float32
        ),
        players_remaining=torch.tensor(
            [row["players_remaining"] for row in match_rows], dtype=torch.int64
        ),
        teams_remaining=torch.tensor(
            [row["teams_remaining"] for row in match_rows], dtype=torch.int64
        ),
        absolute_tick_index=torch.arange(t, dtype=torch.int64),
        time_mask=torch.ones(t, dtype=torch.bool),
        team_slot_mask=torch.ones(n, dtype=torch.bool),
        prior_player_xyz_uu=torch.zeros((n, ROSTER_SIZE, 3), dtype=torch.float32),
        prior_player_alive=torch.zeros((n, ROSTER_SIZE), dtype=torch.bool),
        prior_player_coord_mask=torch.zeros((n, ROSTER_SIZE), dtype=torch.bool),
        prior_life_index=torch.zeros((n, ROSTER_SIZE), dtype=torch.int64),
        prior_state_available=torch.tensor(False, dtype=torch.bool),
    )


def slice_window(
    match: TensorizedMatch, start_tick: int, length: int
) -> TensorizedMatch:
    """Take a contiguous causal window while retaining one motion-only sidecar."""

    if isinstance(start_tick, bool) or not isinstance(start_tick, int):
        raise TypeError("start_tick must be an integer absolute tick")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    matches = (match.absolute_tick_index == start_tick).nonzero(as_tuple=False)
    if matches.numel() != 1:
        raise ValueError("start_tick is not present in this tensorized match")
    start = int(matches.item())
    stop = start + length
    if stop > match.num_timesteps:
        raise ValueError("requested window extends beyond the tensorized match")
    expected = torch.arange(start_tick, start_tick + length, dtype=torch.int64)
    if not torch.equal(match.absolute_tick_index[start:stop].cpu(), expected):
        raise ValueError("requested window is not contiguous in absolute match ticks")

    if start > 0:
        prior_xyz = match.player_xyz_uu[start - 1].clone()
        prior_alive = match.player_alive[start - 1].clone()
        prior_coord = match.player_coord_mask[start - 1].clone()
        prior_life = match.life_index[start - 1].clone()
        prior_available = torch.tensor(True, dtype=torch.bool)
    else:
        prior_xyz = match.prior_player_xyz_uu.clone()
        prior_alive = match.prior_player_alive.clone()
        prior_coord = match.prior_player_coord_mask.clone()
        prior_life = match.prior_life_index.clone()
        prior_available = match.prior_state_available.clone()

    return TensorizedMatch(
        session_id=match.session_id,
        team_ids=match.team_ids,
        player_xyz_uu=match.player_xyz_uu[start:stop].clone(),
        player_alive=match.player_alive[start:stop].clone(),
        player_coord_mask=match.player_coord_mask[start:stop].clone(),
        life_index=match.life_index[start:stop].clone(),
        player_slot_mask=match.player_slot_mask.clone(),
        current_circle_uu=match.current_circle_uu[start:stop].clone(),
        target_circle_uu=match.target_circle_uu[start:stop].clone(),
        zone_mask=match.zone_mask[start:stop].clone(),
        zone_phase=match.zone_phase[start:stop].clone(),
        phase_times_s=match.phase_times_s[start:stop].clone(),
        match_elapsed_s=match.match_elapsed_s[start:stop].clone(),
        players_remaining=match.players_remaining[start:stop].clone(),
        teams_remaining=match.teams_remaining[start:stop].clone(),
        absolute_tick_index=match.absolute_tick_index[start:stop].clone(),
        time_mask=match.time_mask[start:stop].clone(),
        team_slot_mask=match.team_slot_mask.clone(),
        prior_player_xyz_uu=prior_xyz,
        prior_player_alive=prior_alive,
        prior_player_coord_mask=prior_coord,
        prior_life_index=prior_life,
        prior_state_available=prior_available,
    )


def collate_encoder_inputs(
    matches: Iterable[TensorizedMatch],
) -> CollatedEncoderInput:
    items = tuple(matches)
    if not items:
        raise ValueError("at least one tensorized match is required")
    max_t = max(match.num_timesteps for match in items)
    max_n = max(match.num_teams for match in items)
    b = len(items)

    player_xyz = torch.zeros((b, max_t, max_n, ROSTER_SIZE, 3), dtype=torch.float32)
    player_alive = torch.zeros((b, max_t, max_n, ROSTER_SIZE), dtype=torch.bool)
    player_coord = torch.zeros_like(player_alive)
    life_index = torch.zeros((b, max_t, max_n, ROSTER_SIZE), dtype=torch.int64)
    player_slot = torch.zeros((b, max_n, ROSTER_SIZE), dtype=torch.bool)
    current_circle = torch.zeros((b, max_t, 4), dtype=torch.float32)
    target_circle = torch.zeros((b, max_t, 4), dtype=torch.float32)
    zone_mask = torch.zeros((b, max_t), dtype=torch.bool)
    zone_phase = torch.zeros((b, max_t), dtype=torch.int64)
    phase_times = torch.zeros((b, max_t, 3), dtype=torch.float32)
    elapsed = torch.zeros((b, max_t), dtype=torch.float32)
    players_remaining = torch.zeros((b, max_t), dtype=torch.int64)
    teams_remaining = torch.zeros((b, max_t), dtype=torch.int64)
    absolute_tick = torch.zeros((b, max_t), dtype=torch.int64)
    time_mask = torch.zeros((b, max_t), dtype=torch.bool)
    team_slot = torch.zeros((b, max_n), dtype=torch.bool)
    prior_xyz = torch.zeros((b, max_n, ROSTER_SIZE, 3), dtype=torch.float32)
    prior_alive = torch.zeros((b, max_n, ROSTER_SIZE), dtype=torch.bool)
    prior_coord = torch.zeros_like(prior_alive)
    prior_life = torch.zeros((b, max_n, ROSTER_SIZE), dtype=torch.int64)
    prior_available = torch.zeros(b, dtype=torch.bool)

    for batch_index, match in enumerate(items):
        t, n = match.num_timesteps, match.num_teams
        if match.player_xyz_uu.shape != (t, n, ROSTER_SIZE, 3):
            raise ValueError("tensorized match has an invalid player tensor shape")
        player_xyz[batch_index, :t, :n] = match.player_xyz_uu
        player_alive[batch_index, :t, :n] = match.player_alive
        player_coord[batch_index, :t, :n] = match.player_coord_mask
        life_index[batch_index, :t, :n] = match.life_index
        player_slot[batch_index, :n] = match.player_slot_mask
        current_circle[batch_index, :t] = match.current_circle_uu
        target_circle[batch_index, :t] = match.target_circle_uu
        zone_mask[batch_index, :t] = match.zone_mask
        zone_phase[batch_index, :t] = match.zone_phase
        phase_times[batch_index, :t] = match.phase_times_s
        elapsed[batch_index, :t] = match.match_elapsed_s
        players_remaining[batch_index, :t] = match.players_remaining
        teams_remaining[batch_index, :t] = match.teams_remaining
        absolute_tick[batch_index, :t] = match.absolute_tick_index
        time_mask[batch_index, :t] = match.time_mask
        team_slot[batch_index, :n] = match.team_slot_mask
        prior_xyz[batch_index, :n] = match.prior_player_xyz_uu
        prior_alive[batch_index, :n] = match.prior_player_alive
        prior_coord[batch_index, :n] = match.prior_player_coord_mask
        prior_life[batch_index, :n] = match.prior_life_index
        prior_available[batch_index] = match.prior_state_available

    batch = EncoderBatch(
        player_xyz_uu=player_xyz,
        player_alive=player_alive,
        player_coord_mask=player_coord,
        life_index=life_index,
        player_slot_mask=player_slot,
        current_circle_uu=current_circle,
        target_circle_uu=target_circle,
        zone_mask=zone_mask,
        zone_phase=zone_phase,
        phase_times_s=phase_times,
        match_elapsed_s=elapsed,
        players_remaining=players_remaining,
        teams_remaining=teams_remaining,
        absolute_tick_index=absolute_tick,
        time_mask=time_mask,
        team_slot_mask=team_slot,
        prior_player_xyz_uu=prior_xyz,
        prior_player_alive=prior_alive,
        prior_player_coord_mask=prior_coord,
        prior_life_index=prior_life,
        prior_state_available=prior_available,
    )
    metadata = BatchMetadata(
        session_ids=tuple(match.session_id for match in items),
        team_ids=tuple(match.team_ids for match in items),
    )
    return CollatedEncoderInput(batch=batch, metadata=metadata)

