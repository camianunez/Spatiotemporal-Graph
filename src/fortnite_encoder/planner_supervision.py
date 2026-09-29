from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

from .contracts import TensorizedMatch
from .planner_contracts import PlannerTargets
from .supervision import (
    _PLAYER_EVENT_SCHEMA,
    _TEAM_SAMPLE_SCHEMA,
    _validate_event_rows,
    _validate_full_match,
)
from .tensorize import SAMPLE_INTERVAL_SECONDS, _FLOAT_TOLERANCE, _finite
from .world_grid import (
    CELL_HEIGHT_WORLD_UNITS,
    CELL_WIDTH_WORLD_UNITS,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
    WORLD_Y_MIN,
    WorldGridProfile,
)


PLANNER_TARGET_SCHEMA_VERSION = "2.0"
PLANNER_TARGET_SCHEMA_ID = f"planner-targets:{PLANNER_TARGET_SCHEMA_VERSION}"
ROUTE_STEPS = 12
GRID_ROWS = 32
GRID_COLUMNS = 32
GAUSSIAN_RADIUS_CELLS = 3
GAUSSIAN_SIGMA_CELLS = 1.0
_LIFECYCLE_TIMESTAMP_EPSILON_SECONDS = 0.001
_TEAM_TARGET_COLUMNS = (
    "session_id",
    "team_id",
    "match_time_seconds",
    "x",
    "y",
    "z",
)


@dataclass(frozen=True, slots=True, order=True)
class PlannerQuery:
    """CPU-only identity of one focal-team query in a full match."""

    session_id: str
    team_id: int
    absolute_tick_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a nonempty string")
        if isinstance(self.team_id, bool) or not isinstance(self.team_id, int):
            raise TypeError("team_id must be an integer")
        if (
            isinstance(self.absolute_tick_index, bool)
            or not isinstance(self.absolute_tick_index, int)
            or self.absolute_tick_index < 0
        ):
            raise ValueError("absolute_tick_index must be nonnegative")


@dataclass(frozen=True, slots=True)
class PlannerTargetMetadata:
    queries: tuple[PlannerQuery, ...]
    world_grid_profile_id: str
    world_grid_profile_hash: str
    target_schema_id: str = PLANNER_TARGET_SCHEMA_ID


@dataclass(frozen=True, slots=True)
class PlannerTargetDiagnostics:
    """Deterministic aggregate counts emitted by target construction."""

    regular_waypoints: int = 0
    elimination_censors: int = 0
    match_end_censors: int = 0
    focal_team_exclusions: int = 0
    valid_congestion_horizons: int = 0
    masked_congestion_horizons: int = 0
    first_unavailable_timestamp: int = 0
    first_lifecycle_ambiguity: int = 0
    first_missing_centroid: int = 0
    first_nonfinite_centroid: int = 0
    first_out_of_bounds_centroid: int = 0
    first_unresolved_state: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"diagnostic counter {item.name} must be nonnegative"
                )

    @property
    def regular_waypoint_count(self) -> int:
        return self.regular_waypoints

    @property
    def elimination_censor_count(self) -> int:
        return self.elimination_censors

    @property
    def match_end_censor_count(self) -> int:
        return self.match_end_censors

    @property
    def focal_team_exclusion_count(self) -> int:
        return self.focal_team_exclusions

    @property
    def valid_congestion_horizon_count(self) -> int:
        return self.valid_congestion_horizons

    @property
    def masked_congestion_horizon_count(self) -> int:
        return self.masked_congestion_horizons

    @property
    def first_route_masking_reasons(self) -> dict[str, int]:
        return {
            "elimination": self.elimination_censors,
            "match_end": self.match_end_censors,
            "unavailable_timestamp": self.first_unavailable_timestamp,
            "lifecycle_ambiguity": self.first_lifecycle_ambiguity,
            "missing_centroid": self.first_missing_centroid,
            "nonfinite_centroid": self.first_nonfinite_centroid,
            "out_of_bounds_centroid": self.first_out_of_bounds_centroid,
            "unresolved_state": self.first_unresolved_state,
        }

    @property
    def first_route_truncation_reasons(self) -> dict[str, int]:
        return self.first_route_masking_reasons

    def to_dict(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["first_route_masking_reasons"] = (
            self.first_route_masking_reasons
        )
        return result

    def __add__(
        self, other: PlannerTargetDiagnostics
    ) -> PlannerTargetDiagnostics:
        if not isinstance(other, PlannerTargetDiagnostics):
            return NotImplemented
        return PlannerTargetDiagnostics(
            **{
                item.name: getattr(self, item.name) + getattr(other, item.name)
                for item in fields(self)
            }
        )


@dataclass(frozen=True, slots=True)
class PlannerSupervision:
    targets: PlannerTargets
    diagnostics: PlannerTargetDiagnostics
    metadata: PlannerTargetMetadata

    def to(self, *args: object, **kwargs: object) -> PlannerSupervision:
        return PlannerSupervision(
            targets=self.targets.to(*args, **kwargs),
            diagnostics=self.diagnostics,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class PlannerTargetSource:
    """Validated full-match, CPU-only source used to derive planner labels."""

    match: TensorizedMatch
    profile: WorldGridProfile
    centroids_xy_uu: Tensor
    centroid_status: tuple[tuple[str, ...], ...]
    lifecycle_deltas: Mapping[int, tuple[tuple[float, int], ...]]
    lifecycle_event_times: tuple[float, ...]
    inferred_match_end_seconds: float
    lifecycle_complete: bool = True

    def __post_init__(self) -> None:
        expected = (self.match.num_timesteps, self.match.num_teams, 2)
        if tuple(self.centroids_xy_uu.shape) != expected:
            raise ValueError("centroids_xy_uu must have shape [T,N,2]")
        if self.centroids_xy_uu.device.type != "cpu":
            raise ValueError("planner target sources must remain on CPU")
        if len(self.centroid_status) != self.match.num_timesteps or any(
            len(row) != self.match.num_teams for row in self.centroid_status
        ):
            raise ValueError("centroid_status must align with [T,N]")


def _require_parquet_schema(
    path: Path,
    expected: pa.Schema,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required planner target input is missing: {path}")
    try:
        actual = pq.ParquetFile(path).schema_arrow
    except Exception as exc:
        raise ValueError(f"cannot inspect {label}: {exc}") from exc
    if not actual.equals(expected, check_metadata=False):
        raise ValueError(f"{label} has an unexpected Parquet schema")


def _living_count(match: TensorizedMatch, tick: int, team: int) -> int:
    return int(
        (
            match.player_alive[tick, team]
            & match.player_slot_mask[team]
        ).sum().item()
    )


def _read_centroid_source(
    directory: Path,
    match: TensorizedMatch,
) -> tuple[Tensor, tuple[tuple[str, ...], ...]]:
    """Read centroid columns only; placement is inspected in schema, not loaded."""

    path = directory / "team_samples.parquet"
    _require_parquet_schema(path, _TEAM_SAMPLE_SCHEMA, "team_samples.parquet")
    table = pq.read_table(path, columns=list(_TEAM_TARGET_COLUMNS))
    expected_selected = pa.schema(
        [_TEAM_SAMPLE_SCHEMA.field(name) for name in _TEAM_TARGET_COLUMNS]
    )
    if not table.schema.equals(expected_selected, check_metadata=False):
        raise ValueError("selected team centroid columns have an unexpected schema")
    rows = table.to_pylist()
    expected_count = match.num_timesteps * match.num_teams
    if len(rows) != expected_count:
        raise ValueError("team_samples must contain every aligned team at every tick")

    team_index = {team_id: index for index, team_id in enumerate(match.team_ids)}
    time_index = {
        float(value): index
        for index, value in enumerate(match.match_elapsed_s.cpu().tolist())
    }
    centroids = torch.zeros(
        (match.num_timesteps, match.num_teams, 2), dtype=torch.float64
    )
    statuses: list[list[str]] = [
        ["missing_centroid"] * match.num_teams
        for _ in range(match.num_timesteps)
    ]
    seen: set[tuple[int, float]] = set()
    for row in rows:
        if row["session_id"] != match.session_id:
            raise ValueError("team sample session_id does not match encoder match")
        team_id = row["team_id"]
        if (
            isinstance(team_id, bool)
            or not isinstance(team_id, int)
            or team_id not in team_index
        ):
            raise ValueError("team sample team_id is not aligned to encoder match")
        time_seconds = _finite(row["match_time_seconds"], "match_time_seconds")
        if time_seconds not in time_index:
            raise ValueError("team sample tick is not aligned to encoder match")
        key = (team_id, time_seconds)
        if key in seen:
            raise ValueError("team_samples.parquet contains duplicate rows")
        seen.add(key)
        tick = time_index[time_seconds]
        team = team_index[team_id]
        values = (row["x"], row["y"], row["z"])
        present = tuple(value is not None for value in values)
        status = "valid"
        if not any(present):
            status = "missing_centroid"
        elif not all(present):
            raise ValueError(
                "team centroid must be a complete XYZ triple or entirely null"
            )
        elif any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            status = "nonfinite_centroid"
        else:
            centroids[tick, team] = torch.tensor(
                (float(values[0]), float(values[1])), dtype=torch.float64
            )

        alive = _living_count(match, tick, team) > 0
        if not alive and status == "valid":
            raise ValueError("eliminated teams cannot have a centroid")
        statuses[tick][team] = status

    if len(seen) != expected_count:
        raise ValueError("team samples do not align to every encoder team and tick")
    return centroids, tuple(tuple(row) for row in statuses)


def _read_lifecycle_source(
    directory: Path,
    match: TensorizedMatch,
) -> tuple[
    dict[int, tuple[tuple[float, int], ...]],
    tuple[float, ...],
]:
    path = directory / "player_events.parquet"
    _require_parquet_schema(path, _PLAYER_EVENT_SCHEMA, "player_events.parquet")
    table = pq.read_table(path)
    rows = table.to_pylist()
    validated = _validate_event_rows(rows, match)
    deltas = {
        team_id: tuple(values) for team_id, values in validated.items()
    }
    event_times = tuple(
        sorted(
            {
                float(row["match_time_seconds"])
                for row in rows
                if row["event_type"] in {"death", "reboot"}
            }
        )
    )
    return deltas, event_times


def load_planner_target_source(
    path: str | Path,
    match: TensorizedMatch,
    profile: WorldGridProfile,
) -> PlannerTargetSource:
    """Load validated full-match target sources without placement or neighbors."""

    if not isinstance(match, TensorizedMatch):
        raise TypeError("match must be a TensorizedMatch")
    if not isinstance(profile, WorldGridProfile):
        raise TypeError("profile must be a WorldGridProfile")
    if (
        profile.grid_rows != GRID_ROWS
        or profile.grid_columns != GRID_COLUMNS
        or profile.cell_ordering != "row_major"
        or not profile.inclusive_maximum_bounds
    ):
        raise ValueError("planner supervision requires the canonical inclusive 32x32 grid")
    _validate_full_match(match)
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"match session directory does not exist: {directory}")
    centroids, statuses = _read_centroid_source(directory, match)
    deltas, event_times = _read_lifecycle_source(directory, match)
    final_sample = float(match.match_elapsed_s[-1].item())
    inferred_end = max((final_sample, *event_times))
    return PlannerTargetSource(
        match=match,
        profile=profile,
        centroids_xy_uu=centroids,
        centroid_status=statuses,
        lifecycle_deltas=deltas,
        lifecycle_event_times=event_times,
        inferred_match_end_seconds=inferred_end,
    )


def eligible_planner_queries(
    source_or_match: PlannerTargetSource | TensorizedMatch,
) -> tuple[PlannerQuery, ...]:
    match = (
        source_or_match.match
        if isinstance(source_or_match, PlannerTargetSource)
        else source_or_match
    )
    if not isinstance(match, TensorizedMatch):
        raise TypeError("source_or_match must contain a TensorizedMatch")
    profile = (
        source_or_match.profile
        if isinstance(source_or_match, PlannerTargetSource)
        else None
    )
    result: list[PlannerQuery] = []
    for tick in range(match.num_timesteps):
        absolute_tick = int(match.absolute_tick_index[tick].item())
        for team, team_id in enumerate(match.team_ids):
            y0, _ = _query_y0_cell_offset(match, tick, team, profile)
            if _living_count(match, tick, team) > 0 and y0 is not None:
                result.append(
                    PlannerQuery(match.session_id, team_id, absolute_tick)
                )
    return tuple(result)


def sample_planner_queries(
    source_or_match: PlannerTargetSource | TensorizedMatch,
    count: int,
    *,
    seed: int,
) -> tuple[PlannerQuery, ...]:
    """Uniformly sample eligible query triples without replacement."""

    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    eligible = eligible_planner_queries(source_or_match)
    if not eligible:
        return ()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(eligible), generator=generator).tolist()
    return tuple(eligible[index] for index in order[: min(count, len(order))])


def _count_at(
    source: PlannerTargetSource,
    team: int,
    target_time: float,
) -> int | None:
    match = source.match
    times = match.match_elapsed_s.cpu()
    eligible = (times <= target_time + _FLOAT_TOLERANCE).nonzero(
        as_tuple=False
    )
    if eligible.numel() == 0:
        return None
    tick = int(eligible[-1].item())
    count = _living_count(match, tick, team)
    start_time = float(times[tick].item())
    try:
        events = source.lifecycle_deltas[match.team_ids[team]]
    except KeyError:
        return None
    roster_count = int(match.player_slot_mask[team].sum().item())
    event_index = 0
    while event_index < len(events):
        event_time = events[event_index][0]
        if (
            event_time <= start_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
            or event_time > target_time + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
        ):
            event_index += 1
            continue
        net = 0
        while event_index < len(events) and math.isclose(
            events[event_index][0],
            event_time,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        ):
            net += events[event_index][1]
            event_index += 1
        count += net
        if not 0 <= count <= roster_count:
            return None
    return count


def _cell_and_offset(
    xy: Tensor,
    profile: WorldGridProfile,
) -> tuple[int, tuple[float, float]] | None:
    x, y = float(xy[0]), float(xy[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    cell = profile.cell((x, y))
    if cell is None:
        return None
    row, column = divmod(cell, profile.grid_columns)
    fraction_x = (
        x - (profile.world_x_min + column * profile.cell_width_world_units)
    ) / profile.cell_width_world_units
    fraction_y = (
        y - (profile.world_y_min + row * profile.cell_height_world_units)
    ) / profile.cell_height_world_units
    offset_x = 2.0 * fraction_x - 1.0
    offset_y = 2.0 * fraction_y - 1.0
    if not (
        -1.0 <= offset_x <= 1.0 and -1.0 <= offset_y <= 1.0
    ):
        raise ValueError("world-grid cell mapping produced an invalid offset")
    return cell, (offset_x, offset_y)


def _query_y0_cell_offset(
    match: TensorizedMatch,
    tick: int,
    team: int,
    profile: WorldGridProfile | None,
) -> tuple[tuple[int, tuple[float, float]] | None, str | None]:
    """Derive the exact query-time waypoint from valid living focal players."""

    roster = match.player_slot_mask[team]
    living = match.player_alive[tick, team] & roster
    valid = living & match.player_coord_mask[tick, team]
    if not bool(valid.any()):
        return None, "missing_query_coordinate"
    coordinates = match.player_xyz_uu[tick, team, valid, :2].double()
    if not bool(torch.isfinite(coordinates).all()):
        return None, "nonfinite_query_coordinate"
    centroid = coordinates.mean(dim=0)
    if profile is not None:
        mapped = _cell_and_offset(centroid, profile)
    else:
        x, y = float(centroid[0]), float(centroid[1])
        if not (
            WORLD_X_MIN <= x <= WORLD_X_MAX
            and WORLD_Y_MIN <= y <= WORLD_Y_MAX
        ):
            mapped = None
        else:
            column = min(31, math.floor((x - WORLD_X_MIN) / CELL_WIDTH_WORLD_UNITS))
            row = min(31, math.floor((y - WORLD_Y_MIN) / CELL_HEIGHT_WORLD_UNITS))
            fraction_x = (
                x - (WORLD_X_MIN + column * CELL_WIDTH_WORLD_UNITS)
            ) / CELL_WIDTH_WORLD_UNITS
            fraction_y = (
                y - (WORLD_Y_MIN + row * CELL_HEIGHT_WORLD_UNITS)
            ) / CELL_HEIGHT_WORLD_UNITS
            mapped = (
                row * GRID_COLUMNS + column,
                (2.0 * fraction_x - 1.0, 2.0 * fraction_y - 1.0),
            )
    if mapped is None:
        return None, "out_of_bounds_query_coordinate"
    return mapped, None


def planner_query_y0(
    source: PlannerTargetSource,
    query: PlannerQuery,
) -> tuple[int, tuple[float, float]]:
    """Return a sampled query's required map-cell/offset ``y0``."""

    if not isinstance(source, PlannerTargetSource):
        raise TypeError("source must be a PlannerTargetSource")
    if not isinstance(query, PlannerQuery):
        raise TypeError("query must be a PlannerQuery")
    if query.session_id != source.match.session_id:
        raise ValueError("planner query session does not match target source")
    try:
        team = source.match.team_ids.index(query.team_id)
    except ValueError as exc:
        raise ValueError("planner query team is not in target source") from exc
    matches = (
        source.match.absolute_tick_index == query.absolute_tick_index
    ).nonzero(as_tuple=False)
    if matches.numel() != 1:
        raise ValueError("planner query tick is not in target source")
    mapped, reason = _query_y0_cell_offset(
        source.match, int(matches.item()), team, source.profile
    )
    if mapped is None:
        raise ValueError(f"planner query has invalid y0: {reason}")
    return mapped


def _coordinate_failure_reason(xy: Tensor) -> str:
    return (
        "nonfinite_centroid"
        if not bool(torch.isfinite(xy).all())
        else "out_of_bounds_centroid"
    )


def world_xy_to_planner_cell_offset(
    world_xy: Sequence[float] | Tensor,
    profile: WorldGridProfile,
) -> tuple[int, tuple[float, float]] | None:
    """Public canonical coordinate mapper used by planner supervision."""

    if isinstance(world_xy, Tensor):
        values = world_xy.detach().cpu().to(torch.float64).reshape(-1)
    else:
        values = torch.tensor(tuple(world_xy), dtype=torch.float64)
    if values.numel() != 2:
        raise ValueError("world_xy must contain exactly two coordinates")
    return _cell_and_offset(values, profile)


def _gaussian_for_cell(
    cell: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    row, column = divmod(cell, GRID_COLUMNS)
    result = torch.zeros((GRID_ROWS, GRID_COLUMNS), dtype=dtype)
    weighted: list[tuple[int, int, float]] = []
    for delta_row in range(-GAUSSIAN_RADIUS_CELLS, GAUSSIAN_RADIUS_CELLS + 1):
        target_row = row + delta_row
        if not 0 <= target_row < GRID_ROWS:
            continue
        for delta_column in range(
            -GAUSSIAN_RADIUS_CELLS, GAUSSIAN_RADIUS_CELLS + 1
        ):
            target_column = column + delta_column
            if not 0 <= target_column < GRID_COLUMNS:
                continue
            squared_distance = delta_row * delta_row + delta_column * delta_column
            weight = math.exp(
                -squared_distance / (2.0 * GAUSSIAN_SIGMA_CELLS**2)
            )
            weighted.append((target_row, target_column, weight))
    normalization = sum(item[2] for item in weighted)
    if not normalization > 0.0:
        raise RuntimeError("discrete Gaussian has no support")
    for target_row, target_column, weight in weighted:
        result[target_row, target_column] = weight / normalization
    return result


def planner_gaussian_kernel(cell: int) -> Tensor:
    """Return the independently normalized radius-3, sigma-1 cell kernel."""

    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < 1024:
        raise ValueError("cell must be an integer in [0,1023]")
    return _gaussian_for_cell(cell).to(torch.float32)


def _normalize_queries(
    source: PlannerTargetSource,
    queries: Iterable[PlannerQuery | tuple[int, int] | Mapping[str, Any]] | None,
) -> tuple[PlannerQuery, ...]:
    if queries is None:
        return eligible_planner_queries(source)
    result: list[PlannerQuery] = []
    for raw in queries:
        if isinstance(raw, PlannerQuery):
            query = raw
        elif isinstance(raw, Mapping):
            query = PlannerQuery(
                session_id=str(raw.get("session_id", source.match.session_id)),
                team_id=raw["team_id"],
                absolute_tick_index=raw["absolute_tick_index"],
            )
        else:
            values = tuple(raw)
            if len(values) != 2:
                raise ValueError("query tuples must contain (team_id, absolute_tick_index)")
            query = PlannerQuery(source.match.session_id, values[0], values[1])
        result.append(query)
    if len(set(result)) != len(result):
        raise ValueError("planner queries must be unique")
    return tuple(result)


def _increment_first_reason(counters: dict[str, int], reason: str) -> None:
    key = {
        "elimination": "elimination_censors",
        "match_end": "match_end_censors",
        "unavailable_timestamp": "first_unavailable_timestamp",
        "lifecycle_ambiguity": "first_lifecycle_ambiguity",
        "missing_centroid": "first_missing_centroid",
        "nonfinite_centroid": "first_nonfinite_centroid",
        "out_of_bounds_centroid": "first_out_of_bounds_centroid",
        "unresolved_state": "first_unresolved_state",
    }.get(reason)
    if key is None:
        raise RuntimeError(f"unknown target diagnostic reason {reason!r}")
    counters[key] += 1


def build_planner_targets(
    source: PlannerTargetSource,
    queries: Iterable[PlannerQuery | tuple[int, int] | Mapping[str, Any]] | None = None,
) -> PlannerSupervision:
    """Build route and congestion targets from a validated full match.

    Query tuples use ``(team_id, absolute_tick_index)``.  The complete future
    source remains available throughout this function; callers may create a
    causal encoder slice only after it returns.
    """

    if not isinstance(source, PlannerTargetSource):
        raise TypeError("source must be a PlannerTargetSource")
    normalized = _normalize_queries(source, queries)
    q = len(normalized)
    route_cells = torch.zeros((q, ROUTE_STEPS), dtype=torch.int64)
    route_offsets = torch.zeros((q, ROUTE_STEPS, 2), dtype=torch.float32)
    route_mask = torch.zeros((q, ROUTE_STEPS), dtype=torch.bool)
    congestion = torch.zeros(
        (q, ROUTE_STEPS, GRID_ROWS, GRID_COLUMNS), dtype=torch.float32
    )
    congestion_mask = torch.zeros_like(route_mask)

    counter_names = [item.name for item in fields(PlannerTargetDiagnostics)]
    counters = {name: 0 for name in counter_names}
    match = source.match
    team_index = {team_id: index for index, team_id in enumerate(match.team_ids)}
    tick_index = {
        int(value): index
        for index, value in enumerate(match.absolute_tick_index.cpu().tolist())
    }
    gaussian_cache: dict[int, Tensor] = {}

    for query_index, query in enumerate(normalized):
        if query.session_id != match.session_id:
            raise ValueError("planner query session does not match target source")
        if query.team_id not in team_index:
            raise ValueError("planner query team is not in target source")
        if query.absolute_tick_index not in tick_index:
            raise ValueError("planner query tick is not in target source")
        team = team_index[query.team_id]
        tick = tick_index[query.absolute_tick_index]
        if _living_count(match, tick, team) <= 0:
            raise ValueError("planner query focal team must be living")
        query_y0, query_y0_reason = _query_y0_cell_offset(
            match, tick, team, source.profile
        )
        if query_y0 is None:
            raise ValueError(
                f"planner query has invalid y0: {query_y0_reason}"
            )
        query_time = float(match.match_elapsed_s[tick].item())

        route_resolved = True
        for horizon in range(ROUTE_STEPS):
            target_tick = tick + horizon + 1
            target_time = query_time + (horizon + 1) * SAMPLE_INTERVAL_SECONDS
            if route_resolved:
                if target_tick < match.num_timesteps:
                    sampled_count = _living_count(match, target_tick, team)
                    lifecycle_count = _count_at(source, team, target_time)
                    if lifecycle_count is None or lifecycle_count != sampled_count:
                        _increment_first_reason(counters, "lifecycle_ambiguity")
                        route_resolved = False
                    elif sampled_count == 0:
                        _increment_first_reason(counters, "elimination")
                        route_resolved = False
                    else:
                        status = source.centroid_status[target_tick][team]
                        if status != "valid":
                            _increment_first_reason(counters, status)
                            route_resolved = False
                        else:
                            mapped = _cell_and_offset(
                                source.centroids_xy_uu[target_tick, team],
                                source.profile,
                            )
                            if mapped is None:
                                _increment_first_reason(
                                    counters,
                                    _coordinate_failure_reason(
                                        source.centroids_xy_uu[target_tick, team]
                                    ),
                                )
                                route_resolved = False
                            else:
                                cell, waypoint_offset = mapped
                                route_cells[query_index, horizon] = cell
                                route_offsets[query_index, horizon] = torch.tensor(
                                    waypoint_offset, dtype=torch.float32
                                )
                                route_mask[query_index, horizon] = True
                                counters["regular_waypoints"] += 1
                else:
                    living = _count_at(source, team, target_time)
                    if living is None:
                        _increment_first_reason(counters, "lifecycle_ambiguity")
                        route_resolved = False
                    elif living == 0:
                        _increment_first_reason(counters, "elimination")
                        route_resolved = False
                    elif (
                        source.lifecycle_complete
                        and target_time
                        >= source.inferred_match_end_seconds
                        - _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
                        and not any(
                            event_time
                            > target_time
                            + _LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
                            for event_time in source.lifecycle_event_times
                        )
                    ):
                        _increment_first_reason(counters, "match_end")
                        route_resolved = False
                    elif target_time < source.inferred_match_end_seconds:
                        _increment_first_reason(counters, "unavailable_timestamp")
                        route_resolved = False
                    else:
                        _increment_first_reason(counters, "unresolved_state")
                        route_resolved = False

            # Congestion is independent of route resolution and focal outcome.
            if target_tick >= match.num_timesteps:
                counters["masked_congestion_horizons"] += 1
                continue
            counters["focal_team_exclusions"] += 1
            horizon_density = torch.zeros(
                (GRID_ROWS, GRID_COLUMNS), dtype=torch.float64
            )
            valid_horizon = True
            for opponent in range(match.num_teams):
                if opponent == team:
                    continue
                living_mass = _living_count(match, target_tick, opponent)
                if living_mass == 0:
                    continue
                status = source.centroid_status[target_tick][opponent]
                if status != "valid":
                    valid_horizon = False
                    break
                mapped = _cell_and_offset(
                    source.centroids_xy_uu[target_tick, opponent],
                    source.profile,
                )
                if mapped is None:
                    valid_horizon = False
                    break
                cell, _ = mapped
                kernel = gaussian_cache.get(cell)
                if kernel is None:
                    kernel = _gaussian_for_cell(cell)
                    gaussian_cache[cell] = kernel
                horizon_density += float(living_mass) * kernel
            if valid_horizon:
                congestion[query_index, horizon] = horizon_density.to(torch.float32)
                congestion_mask[query_index, horizon] = True
                counters["valid_congestion_horizons"] += 1
            else:
                counters["masked_congestion_horizons"] += 1

    previous_cells = torch.zeros((q, ROUTE_STEPS - 1), dtype=torch.int64)
    previous_offsets = torch.zeros((q, ROUTE_STEPS - 1, 2), dtype=torch.float32)
    # Only resolved future waypoints become teacher inputs.  Masked suffixes
    # retain the neutral map-cell/offset values already allocated above.
    previous_valid = route_mask[:, :-1]
    previous_cells[previous_valid] = route_cells[:, :-1][previous_valid]
    previous_offsets[previous_valid] = route_offsets[:, :-1][
        previous_valid
    ]
    targets = PlannerTargets(
        route_cells=route_cells,
        route_offsets=route_offsets,
        route_mask=route_mask,
        previous_cells=previous_cells,
        previous_offsets=previous_offsets,
        congestion_targets=congestion,
        congestion_mask=congestion_mask,
    )
    diagnostics = PlannerTargetDiagnostics(**counters)
    return PlannerSupervision(
        targets=targets,
        diagnostics=diagnostics,
        metadata=PlannerTargetMetadata(
            queries=normalized,
            world_grid_profile_id=source.profile.profile_id,
            world_grid_profile_hash=source.profile.profile_hash,
        ),
    )


def load_planner_supervision(
    path: str | Path,
    match: TensorizedMatch,
    profile: WorldGridProfile,
    queries: Iterable[PlannerQuery | tuple[int, int] | Mapping[str, Any]] | None = None,
) -> PlannerSupervision:
    source = load_planner_target_source(path, match, profile)
    return build_planner_targets(source, queries)


def load_planner_targets(
    path: str | Path,
    match: TensorizedMatch,
    profile: WorldGridProfile,
    queries: Iterable[PlannerQuery | tuple[int, int] | Mapping[str, Any]] | None = None,
) -> PlannerSupervision:
    """Readable target-loader alias retaining diagnostics and CPU metadata."""

    return load_planner_supervision(path, match, profile, queries)


def load_planner_target_tensors(
    path: str | Path,
    match: TensorizedMatch,
    profile: WorldGridProfile,
    queries: Iterable[PlannerQuery | tuple[int, int] | Mapping[str, Any]] | None = None,
) -> PlannerTargets:
    return load_planner_supervision(path, match, profile, queries).targets


def slice_planner_targets(
    targets: PlannerTargets,
    start: int,
    stop: int | None = None,
) -> PlannerTargets:
    if type(start) is not int or start < 0:
        raise ValueError("start must be a nonnegative integer")
    resolved_stop = start + 1 if stop is None else stop
    if type(resolved_stop) is not int or not start <= resolved_stop <= targets.route_cells.shape[0]:
        raise ValueError("invalid PlannerTargets slice")
    values = {
        item.name: getattr(targets, item.name)[start:resolved_stop].clone()
        for item in fields(targets)
    }
    return PlannerTargets(**values)


def collate_planner_targets(
    targets: Iterable[PlannerTargets],
) -> PlannerTargets:
    items = tuple(targets)
    if not items:
        raise ValueError("at least one PlannerTargets item is required")
    return PlannerTargets(
        **{
            item.name: torch.cat(
                [getattr(value, item.name) for value in items], dim=0
            )
            for item in fields(PlannerTargets)
        }
    )


build_planner_supervision = build_planner_targets


__all__ = [
    "GAUSSIAN_RADIUS_CELLS",
    "GAUSSIAN_SIGMA_CELLS",
    "PLANNER_TARGET_SCHEMA_ID",
    "PLANNER_TARGET_SCHEMA_VERSION",
    "PlannerQuery",
    "PlannerSupervision",
    "PlannerTargetDiagnostics",
    "PlannerTargetMetadata",
    "PlannerTargetSource",
    "build_planner_supervision",
    "build_planner_targets",
    "collate_planner_targets",
    "eligible_planner_queries",
    "load_planner_supervision",
    "load_planner_target_source",
    "load_planner_target_tensors",
    "load_planner_targets",
    "sample_planner_queries",
    "slice_planner_targets",
    "planner_gaussian_kernel",
    "planner_query_y0",
    "world_xy_to_planner_cell_offset",
]
