from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, replace
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fortnite_encoder.config import RotationHeadConfig
from fortnite_encoder.contracts import TensorizedMatch
from fortnite_encoder.supervision import (
    _PLAYER_EVENT_SCHEMA,
    load_rotation_supervision,
)
from fortnite_encoder.tensorize import _INPUT_SCHEMAS, load_match_session
from fortnite_encoder.training import (
    TrainingConfigurationError,
    _FullSession,
    _SessionRecord,
)
from fortnite_encoder.world_grid import WorldGridProfile


LEGACY_FALSE_POSITIVE = "a new life must begin alive"
LIFECYCLE_TIMESTAMP_EPSILON_SECONDS = 0.001
_LIFECYCLE_EVENT_ORDER = {"dbno": 0, "death": 1, "reboot": 2}


class LifecycleContextError(ValueError):
    """A sampled lifecycle cannot be reconciled with its exact event stream."""


def _finite_time(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LifecycleContextError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LifecycleContextError(f"{field} must be finite")
    return result


def _life_index(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LifecycleContextError(f"{field} must be a nonnegative integer")
    return value


def _event_sort_key(row: Mapping[str, Any]) -> tuple[float, int]:
    return (
        float(row["match_time_seconds"]),
        _LIFECYCLE_EVENT_ORDER[str(row["event_type"])],
    )


def _apply_lifecycle_event(
    *,
    alive: bool,
    life_index: int,
    event: Mapping[str, Any],
) -> tuple[bool, int]:
    event_type = str(event["event_type"])
    if event_type == "death":
        alive = False
    elif event_type == "reboot":
        if not alive:
            life_index += 1
        alive = True
    elif event_type != "dbno":
        raise AssertionError(f"unexpected lifecycle event {event_type!r}")
    event_life = _life_index(event["life_index"], "event life_index")
    if event_life != life_index:
        raise LifecycleContextError(
            "lifecycle event life_index does not match event-derived state"
        )
    return alive, life_index


def validate_lifecycle_context(
    player_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate sampled player state against exact Dataset v2 lifecycle events.

    The first sample for each player is treated as observed full-session context.
    Every later sample is replayed from that prior sample through all exact death,
    reboot, and DBNO events in the intervening five-second interval.  This is the
    Dataset v2 contract: a reboot and a subsequent death may both occur between
    two samples, so the later sample can be dead with a higher life index.
    """

    if not player_rows:
        raise LifecycleContextError("player_samples.parquet must not be empty")

    samples_by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    team_by_player: dict[str, int] = {}
    session_ids: set[str] = set()
    for row in player_rows:
        session_id = row.get("session_id")
        player_id = row.get("player_id")
        team_id = row.get("team_id")
        if not isinstance(session_id, str) or not session_id:
            raise LifecycleContextError("sample session_id must be nonempty")
        if not isinstance(player_id, str) or not player_id:
            raise LifecycleContextError("sample player_id must be nonempty")
        if isinstance(team_id, bool) or not isinstance(team_id, int):
            raise LifecycleContextError("sample team_id must be an integer")
        if player_id in team_by_player and team_by_player[player_id] != team_id:
            raise LifecycleContextError("a player cannot change teams")
        if not isinstance(row.get("alive"), bool):
            raise LifecycleContextError("sample alive must be boolean")
        _finite_time(row.get("match_time_seconds"), "sample match_time_seconds")
        _life_index(row.get("life_index"), "sample life_index")
        session_ids.add(session_id)
        team_by_player[player_id] = team_id
        samples_by_player[player_id].append(row)
    if len(session_ids) != 1:
        raise LifecycleContextError("sample rows contain mixed session IDs")
    session_id = next(iter(session_ids))

    events_by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    lifecycle_event_count = 0
    for row in event_rows:
        event_type = row.get("event_type")
        if event_type not in _LIFECYCLE_EVENT_ORDER:
            continue
        event_session = row.get("session_id")
        player_id = row.get("player_id")
        team_id = row.get("team_id")
        if event_session != session_id:
            raise LifecycleContextError("lifecycle event session_id is not aligned")
        if not isinstance(player_id, str) or player_id not in samples_by_player:
            raise LifecycleContextError("lifecycle event player is not in the roster")
        if team_id != team_by_player[player_id]:
            raise LifecycleContextError("lifecycle event team is not aligned")
        event_time = _finite_time(
            row.get("match_time_seconds"), "event match_time_seconds"
        )
        if event_time < 0.0:
            raise LifecycleContextError("lifecycle event time must be nonnegative")
        _life_index(row.get("life_index"), "event life_index")
        events_by_player[player_id].append(row)
        lifecycle_event_count += 1

    collapsed: list[dict[str, Any]] = []
    for player_id, samples in samples_by_player.items():
        samples.sort(key=lambda row: float(row["match_time_seconds"]))
        seen_times: set[float] = set()
        for row in samples:
            sample_time = float(row["match_time_seconds"])
            if sample_time in seen_times:
                raise LifecycleContextError("player samples contain a duplicate tick")
            seen_times.add(sample_time)

        events = sorted(events_by_player[player_id], key=_event_sort_key)
        event_index = 0
        first_time = float(samples[0]["match_time_seconds"])
        while (
            event_index < len(events)
            and float(events[event_index]["match_time_seconds"])
            <= first_time + LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
        ):
            event_index += 1

        previous = samples[0]
        alive = bool(previous["alive"])
        life = int(previous["life_index"])
        for current in samples[1:]:
            previous_time = float(previous["match_time_seconds"])
            current_time = float(current["match_time_seconds"])
            if current_time <= previous_time:
                raise LifecycleContextError("player sample times must increase")
            previous_life = int(previous["life_index"])
            current_life = int(current["life_index"])
            if current_life < previous_life:
                raise LifecycleContextError("life_index cannot decrease")

            interval_events: list[Mapping[str, Any]] = []
            while (
                event_index < len(events)
                and float(events[event_index]["match_time_seconds"])
                <= current_time + LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
            ):
                event = events[event_index]
                event_time = float(event["match_time_seconds"])
                if event_time > previous_time + LIFECYCLE_TIMESTAMP_EPSILON_SECONDS:
                    alive, life = _apply_lifecycle_event(
                        alive=alive,
                        life_index=life,
                        event=event,
                    )
                    interval_events.append(event)
                event_index += 1

            if alive != bool(current["alive"]) or life != current_life:
                raise LifecycleContextError(
                    "exact lifecycle events do not produce the sampled alive/life state"
                )
            if current_life > previous_life and not bool(current["alive"]):
                collapsed.append(
                    {
                        "session_id": session_id,
                        "player_id": player_id,
                        "team_id": team_by_player[player_id],
                        "previous_sample_time_seconds": previous_time,
                        "previous_alive": bool(previous["alive"]),
                        "previous_life_index": previous_life,
                        "current_sample_time_seconds": current_time,
                        "current_alive": bool(current["alive"]),
                        "current_life_index": current_life,
                        "events": [dict(event) for event in interval_events],
                    }
                )
            previous = current

        # Validate life indices on exact lifecycle events after the last complete
        # sample too; these events are valid full-session context even though no
        # later five-second grid row exists.
        last_time = float(samples[-1]["match_time_seconds"])
        while event_index < len(events):
            event = events[event_index]
            if (
                float(event["match_time_seconds"])
                > last_time + LIFECYCLE_TIMESTAMP_EPSILON_SECONDS
            ):
                alive, life = _apply_lifecycle_event(
                    alive=alive,
                    life_index=life,
                    event=event,
                )
            event_index += 1

    return {
        "schema_version": "fncs-lifecycle-context-validation:1.0",
        "session_id": session_id,
        "player_count": len(samples_by_player),
        "sample_row_count": len(player_rows),
        "lifecycle_event_count": lifecycle_event_count,
        "collapsed_transition_count": len(collapsed),
        "collapsed_transitions": collapsed,
        "status": "valid",
    }


def _normalized_sample_rows(
    player_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Delay dead-only life-index visibility for the pinned legacy validator.

    Only the disposable shadow table sees this representation.  The returned
    TensorizedMatch has its raw Dataset v2 life-index tensor restored before it
    can be sliced, collated, hashed, or consumed by the model.
    """

    visible_by_key: dict[tuple[str, float], int] = {}
    by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in player_rows:
        by_player[str(row["player_id"])].append(row)
    changed = 0
    for player_id, rows in by_player.items():
        rows.sort(key=lambda row: float(row["match_time_seconds"]))
        visible_life: int | None = None
        for row in rows:
            raw_life = int(row["life_index"])
            if visible_life is None or bool(row["alive"]):
                visible_life = raw_life
            elif raw_life <= visible_life:
                visible_life = raw_life
            if visible_life != raw_life:
                changed += 1
            visible_by_key[(player_id, float(row["match_time_seconds"]))] = visible_life

    normalized: list[dict[str, Any]] = []
    for raw in player_rows:
        row = dict(raw)
        row["life_index"] = visible_by_key[
            (str(row["player_id"]), float(row["match_time_seconds"]))
        ]
        normalized.append(row)
    return normalized, changed


def _raw_life_tensor(
    player_rows: Sequence[Mapping[str, Any]],
    match: TensorizedMatch,
) -> torch.Tensor:
    team_index = {team_id: index for index, team_id in enumerate(match.team_ids)}
    players_by_team: dict[int, list[str]] = defaultdict(list)
    for row in player_rows:
        team_id = int(row["team_id"])
        player_id = str(row["player_id"])
        if player_id not in players_by_team[team_id]:
            players_by_team[team_id].append(player_id)
    for players in players_by_team.values():
        players.sort()
    slot_by_player = {
        player_id: (team_index[team_id], slot)
        for team_id, players in players_by_team.items()
        for slot, player_id in enumerate(players)
    }
    time_index = {
        float(value): index
        for index, value in enumerate(match.match_elapsed_s.cpu().tolist())
    }
    result = torch.zeros_like(match.life_index)
    for row in player_rows:
        tick = time_index[float(row["match_time_seconds"])]
        team, slot = slot_by_player[str(row["player_id"])]
        result[tick, team, slot] = int(row["life_index"])
    return result


def load_match_session_with_lifecycle_context(
    path: str | Path,
) -> tuple[TensorizedMatch, dict[str, Any]]:
    """Recover only the event-proven five-second lifecycle false positive."""

    directory = Path(path)
    player_path = directory / "player_samples.parquet"
    event_path = directory / "player_events.parquet"
    player_table = pq.read_table(player_path)
    if not player_table.schema.equals(
        _INPUT_SCHEMAS["player_samples.parquet"], check_metadata=False
    ):
        raise LifecycleContextError("player_samples.parquet has an unexpected schema")
    event_table = pq.read_table(event_path)
    if not event_table.schema.equals(_PLAYER_EVENT_SCHEMA, check_metadata=False):
        raise LifecycleContextError("player_events.parquet has an unexpected schema")
    player_rows = player_table.to_pylist()
    event_rows = event_table.to_pylist()
    context = validate_lifecycle_context(player_rows, event_rows)
    if context["collapsed_transition_count"] <= 0:
        raise LifecycleContextError(
            "legacy lifecycle failure is not an event-proven collapsed transition"
        )

    normalized_rows, changed_rows = _normalized_sample_rows(player_rows)
    with tempfile.TemporaryDirectory(prefix="fncs-lifecycle-shadow-") as temporary:
        shadow = Path(temporary)
        for name in (
            "match_samples.parquet",
            "zone_samples.parquet",
            "zone_phases.parquet",
        ):
            shutil.copyfile(directory / name, shadow / name)
        pq.write_table(
            pa.Table.from_pylist(normalized_rows, schema=player_table.schema),
            shadow / "player_samples.parquet",
        )
        legacy_match = load_match_session(shadow)

    raw_life = _raw_life_tensor(player_rows, legacy_match)
    match = replace(legacy_match, life_index=raw_life)
    for field in fields(match):
        if field.name == "life_index":
            continue
        left = getattr(legacy_match, field.name)
        right = getattr(match, field.name)
        if isinstance(left, torch.Tensor):
            if not torch.equal(left, right):
                raise AssertionError(f"recovery changed tensor field {field.name}")
        elif left != right:
            raise AssertionError(f"recovery changed metadata field {field.name}")
    context = {
        **context,
        "legacy_exception": LEGACY_FALSE_POSITIVE,
        "shadow_rows_with_delayed_life_index": changed_rows,
        "returned_tensor_uses_raw_life_index": True,
        "non_life_tensor_fields_changed": [],
        "status": "recovered",
    }
    return match, context


def recover_full_session(
    record: _SessionRecord,
    profile: WorldGridProfile,
    head_config: RotationHeadConfig,
) -> tuple[_FullSession, dict[str, Any]]:
    match, report = load_match_session_with_lifecycle_context(record.path)
    if match.session_id != record.session_id:
        raise TrainingConfigurationError(
            f"session directory {record.session_id} contains {match.session_id}"
        )
    supervision = load_rotation_supervision(
        record.path,
        match,
        profile,
        head_config,
    )
    if (
        supervision.metadata.world_grid_profile_id != profile.profile_id
        or supervision.metadata.world_grid_profile_hash != profile.profile_hash
        or record.world_grid_profile_id != profile.profile_id
        or record.world_grid_profile_hash != profile.profile_hash
        or not profile.supports(record.build, record.world_identity)
    ):
        raise TrainingConfigurationError(
            f"session {record.session_id} is bound to an unexpected world-grid profile"
        )
    return _FullSession(match=match, supervision=supervision), report
