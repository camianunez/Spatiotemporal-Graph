from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fncs_encoder_training.tensorize_compatibility import (
    LEGACY_ZONE_FALSE_POSITIVE,
    TensorizationCompatibilityError,
    load_match_session_with_compatibility_context,
    validate_zone_timing_context,
)
from fortnite_encoder import load_match_session, slice_window


def _phase_rows(session_id: str, activation: float) -> list[dict[str, Any]]:
    return [
        {
            "session_id": session_id,
            "zone_phase": 1,
            "activation_time_seconds": 5.0,
            "shrink_start_time_seconds": 7.0,
            "closure_time_seconds": activation,
            "source_x": 0.0,
            "source_y": 0.0,
            "source_z": 0.0,
            "source_radius": 100_000.0,
            "target_x": 20_000.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_radius": 50_000.0,
        },
        {
            "session_id": session_id,
            "zone_phase": 2,
            "activation_time_seconds": activation,
            "shrink_start_time_seconds": 15.0,
            "closure_time_seconds": 20.0,
            "source_x": 20_000.0,
            "source_y": 0.0,
            "source_z": 0.0,
            "source_radius": 50_000.0,
            "target_x": 30_000.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_radius": 25_000.0,
        },
    ]


def _zone_rows(session_id: str, activation: float) -> list[dict[str, Any]]:
    return [
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
        },
        {
            "session_id": session_id,
            "match_time_seconds": 5.0,
            "zone_phase": 1,
            "time_until_closure_seconds": activation - 5.0,
            "current_boundary_x": 0.0,
            "current_boundary_y": 0.0,
            "current_boundary_z": 0.0,
            "current_boundary_radius": 100_000.0,
            "target_x": 20_000.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_radius": 50_000.0,
        },
        {
            "session_id": session_id,
            "match_time_seconds": 10.0,
            "zone_phase": 2,
            "time_until_closure_seconds": 10.0,
            "current_boundary_x": 20_000.0,
            "current_boundary_y": 0.0,
            "current_boundary_z": 0.0,
            "current_boundary_radius": 50_000.0,
            "target_x": 30_000.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_radius": 25_000.0,
        },
        {
            "session_id": session_id,
            "match_time_seconds": 15.0,
            "zone_phase": 2,
            "time_until_closure_seconds": 5.0,
            "current_boundary_x": 20_000.0,
            "current_boundary_y": 0.0,
            "current_boundary_z": 0.0,
            "current_boundary_radius": 50_000.0,
            "target_x": 30_000.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_radius": 25_000.0,
        },
    ]


def _write_zone_timing_session(directory: Path, activation: float) -> None:
    phase_path = directory / "zone_phases.parquet"
    zone_path = directory / "zone_samples.parquet"
    phase_schema = pq.read_table(phase_path).schema
    zone_schema = pq.read_table(zone_path).schema
    pq.write_table(
        pa.Table.from_pylist(
            _phase_rows(directory.name, activation),
            schema=phase_schema,
        ),
        phase_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            _zone_rows(directory.name, activation),
            schema=zone_schema,
        ),
        zone_path,
    )


def test_producer_epsilon_zone_transition_is_recovered_with_raw_phase_times(
    session_factory: Callable[[str], Path],
) -> None:
    directory = session_factory("zone-producer-epsilon")
    activation = 10.0002
    _write_zone_timing_session(directory, activation)

    with pytest.raises(ValueError, match=LEGACY_ZONE_FALSE_POSITIVE):
        load_match_session(directory)

    match, report = load_match_session_with_compatibility_context(
        directory,
        legacy_exception=LEGACY_ZONE_FALSE_POSITIVE,
    )
    assert report["zone_timing"]["strict_mismatch_count"] == 1
    mismatch = report["zone_timing"]["strict_mismatches"][0]
    assert mismatch["match_time_seconds"] == 10.0
    assert mismatch["sample_zone_phase"] == 2
    assert mismatch["pinned_tensorizer_phase"] == 1
    assert mismatch["producer_phase"] == 2
    assert mismatch["activation_minus_sample_seconds"] == pytest.approx(0.0002)
    assert report["returned_tensor_uses_raw_phase_times"] is True
    assert report["non_restored_tensor_fields_changed"] == []

    assert match.zone_phase.tolist() == [0, 1, 2, 2]
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 7.0, activation],
            [activation, 15.0, 20.0],
            [activation, 15.0, 20.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(match.phase_times_s, expected)
    window = slice_window(match, start_tick=2, length=2)
    assert torch.equal(window.phase_times_s, match.phase_times_s[2:4])
    assert window.zone_phase.tolist() == [2, 2]


def test_zone_transition_outside_producer_epsilon_is_rejected(
    session_factory: Callable[[str], Path],
) -> None:
    directory = session_factory("zone-outside-producer-epsilon")
    _write_zone_timing_session(directory, 10.002)
    with pytest.raises(ValueError, match=LEGACY_ZONE_FALSE_POSITIVE):
        load_match_session(directory)
    with pytest.raises(
        TensorizationCompatibilityError,
        match="producer timestamp contract",
    ):
        load_match_session_with_compatibility_context(
            directory,
            legacy_exception=LEGACY_ZONE_FALSE_POSITIVE,
        )


def test_zone_transition_with_inconsistent_target_is_rejected(
    session_factory: Callable[[str], Path],
) -> None:
    directory = session_factory("zone-inconsistent-target")
    _write_zone_timing_session(directory, 10.0002)
    zone_path = directory / "zone_samples.parquet"
    table = pq.read_table(zone_path)
    rows = table.to_pylist()
    rows[2]["target_x"] = 31_000.0
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), zone_path)
    with pytest.raises(
        TensorizationCompatibilityError,
        match="geometry/countdown is invalid",
    ):
        load_match_session_with_compatibility_context(
            directory,
            legacy_exception=LEGACY_ZONE_FALSE_POSITIVE,
        )


def test_exact_real_transition_values_match_only_the_producer_contract() -> None:
    session_id = "6e3992aceb1f490bb4962a4c95f6d4a8"
    phases = [
        {
            "session_id": session_id,
            "zone_phase": 11,
            "activation_time_seconds": 1243.9989700317383,
            "shrink_start_time_seconds": 1245.0001907348633,
            "closure_time_seconds": 1295.0001907348633,
            "source_x": -47643.95,
            "source_y": -23362.9,
            "source_z": 0.0,
            "source_radius": 1100.0,
            "target_x": -46032.36,
            "target_y": -30534.04,
            "target_z": 0.0,
            "target_radius": 1000.0,
        },
        {
            "session_id": session_id,
            "zone_phase": 12,
            "activation_time_seconds": 1295.0001907348633,
            "shrink_start_time_seconds": 1296.0048294067383,
            "closure_time_seconds": 1386.0048294067383,
            "source_x": -46032.36,
            "source_y": -30534.04,
            "source_z": 0.0,
            "source_radius": 1000.0,
            "target_x": -54149.74,
            "target_y": -37194.22,
            "target_z": 0.0,
            "target_radius": 0.0,
        },
    ]
    # The validator requires phases numbered from one, so retain exact phases
    # 1-10 only as timing-compatible placeholders; the edge evidence is exact.
    placeholders: list[dict[str, Any]] = []
    for phase in range(1, 11):
        activation = float((phase - 1) * 100)
        placeholders.append(
            {
                **phases[0],
                "zone_phase": phase,
                "activation_time_seconds": activation,
                "shrink_start_time_seconds": activation + 1.0,
                "closure_time_seconds": activation + 99.0,
            }
        )
    sample = {
        "session_id": session_id,
        "match_time_seconds": 1295.0,
        "zone_phase": 12,
        "time_until_closure_seconds": 91.00482940673828,
        "current_boundary_x": -46032.36,
        "current_boundary_y": -30534.04,
        "current_boundary_z": 0.0,
        "current_boundary_radius": 1000.0,
        "target_x": -54149.74,
        "target_y": -37194.22,
        "target_z": 0.0,
        "target_radius": 0.0,
    }
    report = validate_zone_timing_context(placeholders + phases, [sample])
    assert report["strict_mismatch_count"] == 1
    assert report["strict_mismatches"][0][
        "activation_minus_sample_seconds"
    ] == pytest.approx(0.00019073486328125)
