from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from fortnite_encoder import (
    collate_encoder_inputs,
    load_match_session,
    slice_window,
)


def test_loads_only_four_non_target_tables(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    real_read = pq.read_table

    def recording_read(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", recording_read)
    match = load_match_session(session_dir)
    assert set(opened) == {
        "match_samples.parquet",
        "player_samples.parquet",
        "zone_samples.parquet",
        "zone_phases.parquet",
    }
    assert match.team_ids == (10, 20)
    assert match.player_xyz_uu.shape == (4, 2, 2, 3)
    assert match.absolute_tick_index.tolist() == [0, 1, 2, 3]
    assert not hasattr(match, "final_placement")


def test_window_uses_absolute_ticks_and_prior_motion_sidecar(
    tensorized_match,
) -> None:
    window = slice_window(tensorized_match, start_tick=2, length=2)
    assert window.absolute_tick_index.tolist() == [2, 3]
    assert window.match_elapsed_s.tolist() == [10.0, 15.0]
    assert bool(window.prior_state_available)
    assert torch.equal(
        window.prior_player_xyz_uu, tensorized_match.player_xyz_uu[1]
    )
    assert window.num_timesteps == 2


def test_collation_pads_time_and_keeps_ids_cpu_side(tensorized_match) -> None:
    window = slice_window(tensorized_match, start_tick=1, length=2)
    collated = collate_encoder_inputs([tensorized_match, window])
    assert collated.batch.player_xyz_uu.shape == (2, 4, 2, 2, 3)
    assert collated.batch.time_mask.tolist() == [
        [True, True, True, True],
        [True, True, False, False],
    ]
    assert collated.metadata.session_ids == (
        tensorized_match.session_id,
        tensorized_match.session_id,
    )
    assert collated.metadata.team_ids == ((10, 20), (10, 20))
    assert not hasattr(collated.batch, "team_ids")
    assert not hasattr(collated.batch, "player_ids")


def test_rejects_bad_cadence(session_dir: Path) -> None:
    path = session_dir / "match_samples.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[2]["match_time_seconds"] = 11.0
    pq.write_table(table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="five-second grid"):
        load_match_session(session_dir)


def test_rejects_incomplete_tick_join(session_dir: Path) -> None:
    path = session_dir / "player_samples.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 1), path)
    with pytest.raises(ValueError, match="exactly one row"):
        load_match_session(session_dir)


def test_rejects_lifecycle_count_mismatch(session_dir: Path) -> None:
    path = session_dir / "match_samples.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[2]["players_remaining"] = 4
    pq.write_table(table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="lifecycle-derived"):
        load_match_session(session_dir)


def test_rejects_schema_drift(session_dir: Path) -> None:
    path = session_dir / "match_samples.parquet"
    table = pq.read_table(path)
    pq.write_table(table.drop(["teams_remaining"]), path)
    with pytest.raises(ValueError, match="unexpected Parquet schema"):
        load_match_session(session_dir)


def test_window_bounds_are_rejected(tensorized_match) -> None:
    with pytest.raises(ValueError, match="not present"):
        slice_window(tensorized_match, 99, 1)
    with pytest.raises(ValueError, match="beyond"):
        slice_window(tensorized_match, 3, 2)
    with pytest.raises(ValueError, match="positive"):
        slice_window(tensorized_match, 0, 0)


def test_collation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        collate_encoder_inputs([])

