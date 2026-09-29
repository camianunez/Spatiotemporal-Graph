from __future__ import annotations

import json
import math

import pytest

from fortnite_encoder import WorldGridProfile, WorldGridProfileError
from fortnite_encoder.world_grid import _payload_hash


def profile_dict(**changes):
    raw = {
        "schema_version": "2.0",
        "profile_id": "model-world-grid-v1",
        "profile_kind": "engineered_model_world_partition",
        "authoritative_terrain_bounds": False,
        "compatible_builds": ["41.10", "41.20"],
        "world_identity": "/Hera_Map/Maps/Hera_V2_Terrain",
        "world_x_min": -131_072,
        "world_x_max": 131_072,
        "world_y_min": -131_072,
        "world_y_max": 131_072,
        "grid_rows": 32,
        "grid_columns": 32,
        "column_axis": "x",
        "row_axis": "y",
        "column_direction": "increasing",
        "row_direction": "increasing",
        "inclusive_maximum_bounds": True,
        "cell_ordering": "row_major",
        "lattice_origin_x": 0,
        "lattice_origin_y": 0,
        "lattice_spacing_world_units": 512,
        "cell_width_world_units": 8_192,
        "cell_height_world_units": 8_192,
        "world_unit_scale": {
            "meters_per_world_unit": 0.01,
            "semantic_source": "Epic Unreal Units",
            "source_url": (
                "https://dev.epicgames.com/documentation/fortnite/unreal-units"
            ),
        },
        "publication_audit": {
            "report_id": "model-world-grid-v1-publication-audit",
            "audit_sha256": "1" * 64,
            "coordinate_audit_sha256": "2" * 64,
        },
        "evidence": [
            {
                "kind": "engineered_definition",
                "subject": "model_world_partition",
                "source": "model-world-grid-v1",
                "semantics": "fixed non-authoritative model partition",
            },
            {
                "kind": "replay_metadata",
                "subject": "reported_metadata_containment",
                "source": "FortPoiManager",
                "semantics": "reported extents are contained",
            },
            {
                "kind": "coverage_audit",
                "subject": "model_visible_coordinate_coverage",
                "source": "accepted dataset-v2",
                "semantics": "coverage only; coordinates do not define bounds",
            },
        ],
        "profile_hash": "",
    }
    raw.update(changes)
    raw["profile_hash"] = _payload_hash(raw, "profile_hash")
    return raw


def test_profile_json_round_trip_and_deterministic_hash(tmp_path) -> None:
    raw = profile_dict()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    first = WorldGridProfile.load(path)
    second = WorldGridProfile.from_dict(first.to_dict())
    assert first == second
    assert first.profile_hash == second.compute_hash() == raw["profile_hash"]
    assert first.supports("41.10", raw["world_identity"])
    assert not first.supports("41.00", raw["world_identity"])
    assert not first.supports("41.10", "/Different/World")
    with pytest.raises(WorldGridProfileError, match="run configuration"):
        WorldGridProfile.load(path, expected_hash="0" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "1.0", "engineered profile value"),
        ("profile_id", "other", "engineered profile value"),
        ("profile_kind", "terrain", "engineered profile value"),
        ("authoritative_terrain_bounds", True, "engineered profile value"),
        ("world_x_min", -131_071, "engineered profile value"),
        ("world_x_max", 131_073, "engineered profile value"),
        ("grid_rows", 31, "engineered profile value"),
        ("grid_columns", 64, "engineered profile value"),
        ("column_axis", "y", "engineered profile value"),
        ("row_direction", "decreasing", "engineered profile value"),
        ("inclusive_maximum_bounds", False, "engineered profile value"),
        ("lattice_spacing_world_units", 1024, "engineered profile value"),
        ("cell_width_world_units", 4096, "engineered profile value"),
        ("compatible_builds", [], "engineered profile value"),
    ],
)
def test_profile_rejects_noncanonical_contract(field, value, message) -> None:
    with pytest.raises(WorldGridProfileError, match=message):
        WorldGridProfile.from_dict(profile_dict(**{field: value}))


def test_unknown_fields_hash_tampering_and_old_schema_are_rejected() -> None:
    raw = profile_dict()
    raw["image_sha256"] = "forbidden"
    with pytest.raises(WorldGridProfileError, match="unknown"):
        WorldGridProfile.from_dict(raw)
    raw = profile_dict()
    raw["profile_hash"] = "0" * 64
    with pytest.raises(WorldGridProfileError, match="does not match"):
        WorldGridProfile.from_dict(raw)
    raw = profile_dict(schema_version="1.0")
    with pytest.raises(WorldGridProfileError, match="engineered profile"):
        WorldGridProfile.from_dict(raw)


def test_metadata_derived_bound_claim_cannot_replace_engineered_evidence() -> None:
    evidence = [
        {
            "kind": "replay_metadata",
            "subject": "model_world_partition",
            "source": "observed player extrema",
            "semantics": "metadata-derived terrain bounds",
        },
        {
            "kind": "replay_metadata",
            "subject": "reported_metadata_containment",
            "source": "FortPoiManager",
            "semantics": "contained",
        },
        {
            "kind": "coverage_audit",
            "subject": "model_visible_coordinate_coverage",
            "source": "dataset",
            "semantics": "coverage",
        },
    ]
    with pytest.raises(WorldGridProfileError, match="engineered_definition"):
        WorldGridProfile.from_dict(profile_dict(evidence=evidence))


def test_exact_bounds_cells_and_out_of_envelope_values() -> None:
    profile = WorldGridProfile.from_dict(profile_dict())
    assert profile.cell((-131_072, -131_072)) == 0
    assert profile.cell((131_072, -131_072)) == 31
    assert profile.cell((-131_072, 131_072)) == 31 * 32
    assert profile.cell((131_072, 131_072)) == 1023
    assert profile.cell((0, 0)) == 16 * 32 + 16
    assert profile.cell((-131_072.000001, 0)) is None
    assert profile.cell((131_072.000001, 0)) is None
    assert profile.cell((0, -131_072.000001)) is None
    assert profile.cell((0, 131_072.000001)) is None
    with pytest.raises(WorldGridProfileError, match="finite"):
        profile.cell((math.nan, 0))
    with pytest.raises(WorldGridProfileError, match="exactly two"):
        profile.cell((0,))


def test_every_8192_unit_boundary_is_row_major_and_deterministic() -> None:
    profile = WorldGridProfile.from_dict(profile_dict())
    for boundary in range(33):
        x = profile.world_x_min + boundary * 8_192
        expected_column = min(boundary, 31)
        assert profile.cell((x, profile.world_y_min)) == expected_column
        y = profile.world_y_min + boundary * 8_192
        expected_row = min(boundary, 31)
        assert profile.cell((profile.world_x_min, y)) == expected_row * 32
