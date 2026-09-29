from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fncs_encoder_training.lifecycle import (
    LifecycleContextError,
    _normalized_sample_rows,
    load_match_session_with_lifecycle_context,
    validate_lifecycle_context,
)
from fortnite_encoder import load_match_session, slice_window
from fortnite_encoder.supervision import _PLAYER_EVENT_SCHEMA


def _sample(
    time_seconds: float,
    *,
    alive: bool,
    life_index: int,
) -> dict[str, Any]:
    return {
        "session_id": "session",
        "player_id": "player",
        "team_id": 1,
        "match_time_seconds": time_seconds,
        "alive": alive,
        "life_index": life_index,
    }


def _event(
    time_seconds: float,
    event_type: str,
    life_index: int,
) -> dict[str, Any]:
    return {
        "session_id": "session",
        "player_id": "player",
        "team_id": 1,
        "match_time_seconds": time_seconds,
        "event_type": event_type,
        "life_index": life_index,
    }


@pytest.mark.parametrize(
    ("samples", "events"),
    [
        ([_sample(5.0, alive=True, life_index=0)], []),
        ([_sample(5.0, alive=False, life_index=0)], []),
        (
            [
                _sample(0.0, alive=True, life_index=0),
                _sample(5.0, alive=False, life_index=0),
            ],
            [_event(3.0, "death", 0)],
        ),
        (
            [
                _sample(0.0, alive=False, life_index=0),
                _sample(5.0, alive=True, life_index=1),
            ],
            [_event(3.0, "reboot", 1)],
        ),
    ],
    ids=("window-start-alive", "window-start-dead", "death-inside", "reboot-inside"),
)
def test_event_context_accepts_valid_window_lifecycle_states(
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    report = validate_lifecycle_context(samples, events)
    assert report["status"] == "valid"
    assert report["collapsed_transition_count"] == 0


def test_event_context_rejects_unexplained_dead_life_increment() -> None:
    samples = [
        _sample(0.0, alive=False, life_index=0),
        _sample(5.0, alive=False, life_index=1),
    ]
    with pytest.raises(
        LifecycleContextError,
        match="exact lifecycle events do not produce",
    ):
        validate_lifecycle_context(samples, [])


def test_exact_redacted_real_fixture_is_an_event_proven_collapsed_transition() -> None:
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "fncs_lifecycle_collapsed_reboot_death.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    report = validate_lifecycle_context(
        fixture["player_samples"], fixture["player_events"]
    )
    assert report["collapsed_transition_count"] == 1
    transition = report["collapsed_transitions"][0]
    assert [event["event_type"] for event in transition["events"]] == [
        "reboot",
        "death",
    ]
    assert transition["current_alive"] is False
    assert transition["current_life_index"] == 2
    normalized, changed = _normalized_sample_rows(fixture["player_samples"])
    assert changed == 1
    assert [row["life_index"] for row in normalized] == [1, 1]


def _write_collapsed_session(directory: Path) -> None:
    player_path = directory / "player_samples.parquet"
    player_table = pq.read_table(player_path)
    player_rows = player_table.to_pylist()
    for row in player_rows:
        if row["player_id"] != "bravo":
            continue
        tick = int(row["match_time_seconds"] // 5)
        if tick == 0:
            continue
        row["alive"] = False
        row["life_index"] = 0 if tick == 1 else 1
        for name in (
            "x",
            "y",
            "z",
            "target_zone_offset_x",
            "target_zone_offset_y",
            "target_zone_inside",
            "target_zone_edge_distance_normalized",
            "current_boundary_offset_x",
            "current_boundary_offset_y",
            "current_boundary_inside",
            "current_boundary_edge_distance_normalized",
        ):
            row[name] = None
    pq.write_table(
        pa.Table.from_pylist(player_rows, schema=player_table.schema), player_path
    )

    match_path = directory / "match_samples.parquet"
    match_table = pq.read_table(match_path)
    match_rows = match_table.to_pylist()
    by_time: dict[float, list[dict[str, Any]]] = {}
    for row in player_rows:
        by_time.setdefault(float(row["match_time_seconds"]), []).append(row)
    for row in match_rows:
        at_tick = by_time[float(row["match_time_seconds"])]
        row["players_remaining"] = sum(bool(item["alive"]) for item in at_tick)
        row["teams_remaining"] = len(
            {item["team_id"] for item in at_tick if item["alive"]}
        )
    pq.write_table(
        pa.Table.from_pylist(match_rows, schema=match_table.schema), match_path
    )

    event_rows = [
        {
            "session_id": directory.name,
            "player_id": "bravo",
            "team_id": 10,
            "match_time_seconds": 4.0,
            "event_type": "death",
            "life_index": 0,
            "x": 10_000.0,
            "y": 20_000.0,
            "z": 0.0,
            "zone_phase": None,
            "zone_basis": None,
            "location_quality": "replay_event",
        },
        {
            "session_id": directory.name,
            "player_id": "bravo",
            "team_id": 10,
            "match_time_seconds": 5.8,
            "event_type": "reboot",
            "life_index": 1,
            "x": None,
            "y": None,
            "z": None,
            "zone_phase": None,
            "zone_basis": None,
            "location_quality": None,
        },
        {
            "session_id": directory.name,
            "player_id": "bravo",
            "team_id": 10,
            "match_time_seconds": 9.8,
            "event_type": "death",
            "life_index": 1,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "zone_phase": None,
            "zone_basis": None,
            "location_quality": "replay_event",
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(event_rows, schema=_PLAYER_EVENT_SCHEMA),
        directory / "player_events.parquet",
    )


def test_recovered_full_match_and_windows_preserve_raw_lifecycle(
    session_factory: Callable[[str], Path],
) -> None:
    directory = session_factory("collapsed-real-shape")
    _write_collapsed_session(directory)
    with pytest.raises(ValueError, match="a new life must begin alive"):
        load_match_session(directory)

    match, report = load_match_session_with_lifecycle_context(directory)
    assert report["collapsed_transition_count"] == 1
    assert report["returned_tensor_uses_raw_life_index"] is True
    assert report["non_life_tensor_fields_changed"] == []

    # Team 10 sorts first and "bravo" sorts into roster slot 1.
    assert bool(match.player_alive[0, 0, 1])
    assert not bool(match.player_alive[1, 0, 1])
    assert not bool(match.player_alive[2, 0, 1])
    assert int(match.life_index[1, 0, 1]) == 0
    assert int(match.life_index[2, 0, 1]) == 1
    assert not bool(match.player_coord_mask[1, 0, 1])
    assert not bool(match.player_coord_mask[2, 0, 1])

    # The life transition is at the first visible tick.  The prior sidecar
    # retains the dead life-0 state instead of inventing a window-local life.
    at_transition = slice_window(match, start_tick=2, length=2)
    assert int(at_transition.absolute_tick_index[0]) == 2
    assert bool(at_transition.prior_state_available)
    assert not bool(at_transition.prior_player_alive[0, 1])
    assert int(at_transition.prior_life_index[0, 1]) == 0
    assert int(at_transition.life_index[0, 0, 1]) == 1
    assert torch.equal(at_transition.life_index, match.life_index[2:4])
    assert torch.equal(at_transition.player_alive, match.player_alive[2:4])
    assert torch.equal(
        at_transition.player_coord_mask, match.player_coord_mask[2:4]
    )

    # The same transition immediately precedes this next window; slicing again
    # cannot reinterpret its first row as a lifecycle transition.
    after_transition = slice_window(match, start_tick=3, length=1)
    assert int(after_transition.prior_life_index[0, 1]) == 1
    assert int(after_transition.life_index[0, 0, 1]) == 1
    assert not bool(after_transition.player_alive[0, 0, 1])

    alive_start = slice_window(match, start_tick=0, length=2)
    dead_start_with_death_inside = slice_window(match, start_tick=1, length=2)
    assert bool(alive_start.player_alive[0, 0, 1])
    assert not bool(alive_start.player_alive[1, 0, 1])
    assert not bool(dead_start_with_death_inside.player_alive[0, 0, 1])
