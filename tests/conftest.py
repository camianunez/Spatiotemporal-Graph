from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "src"))

from fortnite_encoder import (  # noqa: E402
    EncoderConfig,
    collate_encoder_inputs,
    load_match_session,
)
from fortnite_encoder.tensorize import _INPUT_SCHEMAS  # noqa: E402


def _match_rows(session_id: str) -> list[dict[str, Any]]:
    return [
        {
            "session_id": session_id,
            "match_time_seconds": float(tick * 5),
            "players_remaining": count,
            "teams_remaining": 2,
        }
        for tick, count in enumerate((4, 4, 3, 4))
    ]


def _player_rows(session_id: str) -> list[dict[str, Any]]:
    positions: dict[str, list[tuple[bool, int, tuple[float, float, float] | None]]] = {
        "alpha": [
            (True, 0, (0.0, 0.0, 0.0)),
            (True, 0, (10_000.0, 0.0, 0.0)),
            (True, 0, (20_000.0, 0.0, 0.0)),
            (True, 0, (30_000.0, 0.0, 0.0)),
        ],
        "bravo": [
            (True, 0, (0.0, 20_000.0, 0.0)),
            (True, 0, (10_000.0, 20_000.0, 0.0)),
            (False, 0, None),
            (True, 1, None),
        ],
        "charlie": [
            (True, 0, (100_000.0, 0.0, 0.0)),
            (True, 0, (90_000.0, 0.0, 0.0)),
            (True, 0, (80_000.0, 0.0, 0.0)),
            (True, 0, (70_000.0, 0.0, 0.0)),
        ],
        "delta": [
            (True, 0, (100_000.0, 20_000.0, 0.0)),
            (True, 0, (90_000.0, 20_000.0, 0.0)),
            (True, 0, (80_000.0, 20_000.0, 0.0)),
            (True, 0, (70_000.0, 20_000.0, 0.0)),
        ],
    }
    teams = {"alpha": 10, "bravo": 10, "charlie": 20, "delta": 20}
    rows: list[dict[str, Any]] = []
    for tick in range(4):
        for player_id in ("alpha", "bravo", "charlie", "delta"):
            alive, life_index, xyz = positions[player_id][tick]
            x, y, z = xyz if xyz is not None else (None, None, None)
            rows.append(
                {
                    "session_id": session_id,
                    "player_id": player_id,
                    "team_id": teams[player_id],
                    "match_time_seconds": float(tick * 5),
                    "alive": alive,
                    "life_index": life_index,
                    "x": x,
                    "y": y,
                    "z": z,
                    "target_zone_offset_x": None,
                    "target_zone_offset_y": None,
                    "target_zone_inside": None,
                    "target_zone_edge_distance_normalized": None,
                    "current_boundary_offset_x": None,
                    "current_boundary_offset_y": None,
                    "current_boundary_inside": None,
                    "current_boundary_edge_distance_normalized": None,
                }
            )
    return rows


def _phase_rows(session_id: str) -> list[dict[str, Any]]:
    return [
        {
            "session_id": session_id,
            "zone_phase": 1,
            "activation_time_seconds": 5.0,
            "shrink_start_time_seconds": 10.0,
            "closure_time_seconds": 20.0,
            "source_x": 0.0,
            "source_y": 0.0,
            "source_z": 0.0,
            "source_radius": 100_000.0,
            "target_x": 20_000.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_radius": 50_000.0,
        }
    ]


def _zone_rows(session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "session_id": session_id,
            "match_time_seconds": 0.0,
            "zone_phase": None,
            "time_until_closure_seconds": None,
            "current_boundary_x": None,
            "current_boundary_y": None,
            "current_boundary_z": None,
            "current_boundary_radius": None,
            "target_x": None,
            "target_y": None,
            "target_z": None,
            "target_radius": None,
        }
    ]
    current = (
        (0.0, 0.0, 0.0, 100_000.0),
        (0.0, 0.0, 0.0, 100_000.0),
        (10_000.0, 0.0, 0.0, 75_000.0),
    )
    for tick, circle in zip((1, 2, 3), current):
        rows.append(
            {
                "session_id": session_id,
                "match_time_seconds": float(tick * 5),
                "zone_phase": 1,
                "time_until_closure_seconds": float(20 - tick * 5),
                "current_boundary_x": circle[0],
                "current_boundary_y": circle[1],
                "current_boundary_z": circle[2],
                "current_boundary_radius": circle[3],
                "target_x": 20_000.0,
                "target_y": 0.0,
                "target_z": 0.0,
                "target_radius": 50_000.0,
            }
        )
    return rows


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = _INPUT_SCHEMAS[path.name]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


@pytest.fixture
def session_factory(
    tmp_path: Path,
) -> Callable[[str], Path]:
    def create(name: str = "synthetic-session") -> Path:
        directory = tmp_path / name
        directory.mkdir()
        session_id = name
        tables = {
            "match_samples.parquet": _match_rows(session_id),
            "player_samples.parquet": _player_rows(session_id),
            "zone_samples.parquet": _zone_rows(session_id),
            "zone_phases.parquet": _phase_rows(session_id),
        }
        for file_name, rows in tables.items():
            _write_table(directory / file_name, rows)
        # These deliberately invalid sentinels prove the tensorizer never opens them.
        for file_name in (
            "team_samples.parquet",
            "player_events.parquet",
            "centroid_neighbors.parquet",
        ):
            (directory / file_name).write_bytes(b"not parquet")
        return directory

    return create


@pytest.fixture
def session_dir(session_factory: Callable[[str], Path]) -> Path:
    return session_factory()


@pytest.fixture
def tensorized_match(session_dir: Path):
    return load_match_session(session_dir)


@pytest.fixture
def encoder_batch(tensorized_match):
    return collate_encoder_inputs([tensorized_match]).batch


@pytest.fixture
def small_config() -> EncoderConfig:
    return EncoderConfig(
        d_model=32,
        n_heads=4,
        player_embedding_dim=16,
        edge_hidden_dim=16,
        ffn_dim=64,
        spatial_layers=1,
        temporal_layers=1,
        dropout=0.0,
        max_timesteps=32,
        max_zone_phase=4,
    )


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    torch.manual_seed(7)

