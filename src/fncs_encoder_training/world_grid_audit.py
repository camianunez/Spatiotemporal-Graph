from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from fortnite_encoder.validate_world_grid import (
    _SOURCE_NAMES,
    _bits,
    _compact_array,
    _finite_pair,
    _observations,
)
from fortnite_encoder.world_grid import WorldGridProfile

from .common import canonical_json, file_sha256, sha256_bytes, utc_now


def _write_jsonl_line(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write((canonical_json(value) + "\n").encode("utf-8"))


def audit_world_grid(
    *,
    dataset_root: str | Path,
    inventory: Mapping[str, Any],
    profile_path: str | Path,
    expected_profile_raw_sha256: str,
    expected_profile_hash: str,
    publication_audit_path: str | Path,
    expected_audit_raw_sha256: str,
    expected_audit_payload_sha256: str,
    expected_coordinate_audit_sha256: str,
    out_of_bounds_path: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    report_path = root / "ingestion-report.json"
    profile_source = Path(profile_path).resolve()
    audit_source = Path(publication_audit_path).resolve()
    if file_sha256(profile_source) != expected_profile_raw_sha256:
        raise ValueError("world-grid profile raw SHA-256 changed")
    if file_sha256(audit_source) != expected_audit_raw_sha256:
        raise ValueError("world-grid publication audit raw SHA-256 changed")

    # This is the authoritative independent-corpus compatibility check.  It
    # verifies the publication audit itself, then every accepted report item's
    # build, changelist, world identity, map geometry, and binding.
    profile = WorldGridProfile.load(
        profile_source,
        expected_hash=expected_profile_hash,
        audit_path=audit_source,
        ingestion_report_path=report_path,
        audit_binding_mode="independent",
    )
    if profile.publication_audit.audit_sha256 != expected_audit_payload_sha256:
        raise ValueError("world-grid publication audit payload hash changed")
    if (
        profile.publication_audit.coordinate_audit_sha256
        != expected_coordinate_audit_sha256
    ):
        raise ValueError("world-grid coordinate audit hash changed")

    sessions = inventory.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 1150:
        raise ValueError("world-grid audit requires all 1,150 accepted sessions")
    builds = Counter(str(item["build"]) for item in sessions)
    changelists = Counter(str(item["changelist"]) for item in sessions)
    worlds = Counter(str(item["world_identity"]) for item in sessions)
    for item in sessions:
        if not profile.supports(str(item["build"]), str(item["world_identity"])):
            raise ValueError(
                f"world-grid profile does not support session {item['game_session_id']}"
            )

    counters: dict[str, dict[str, Any]] = {
        source: {
            "coordinate_pairs": 0,
            "finite_coordinates": 0,
            "unavailable_coordinate_pairs": 0,
            "out_of_bounds_coordinates": 0,
        }
        for source in _SOURCE_NAMES
    }
    assignment_digest = hashlib.sha256()
    output_path = Path(out_of_bounds_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    out_of_bounds_total = 0
    try:
        with os.fdopen(handle, "wb") as stream:
            for session_index, session in enumerate(sessions, start=1):
                session_id = session["game_session_id"]
                directory = root / session["directory"]
                for source, table_name, record_key, raw_x, raw_y in _observations(
                    directory, session_id
                ):
                    counter = counters[source]
                    counter["coordinate_pairs"] += 1
                    pair = _finite_pair(raw_x, raw_y)
                    cell: int | None = None
                    if pair is None:
                        counter["unavailable_coordinate_pairs"] += 1
                    else:
                        counter["finite_coordinates"] += 1
                        cell = profile.cell(pair)
                        if cell is None:
                            counter["out_of_bounds_coordinates"] += 1
                            out_of_bounds_total += 1
                            _write_jsonl_line(
                                stream,
                                {
                                    "game_session_id": session_id,
                                    "source": source,
                                    "table": table_name,
                                    "record_key": record_key,
                                    "x": pair[0],
                                    "y": pair[1],
                                    "cell": None,
                                    "target_policy": (
                                        "masked_without_clamping"
                                        if source
                                        in {
                                            "eligible_player_centroid",
                                            "zone_phase_source",
                                            "zone_phase_target",
                                            "zone_sample_current",
                                            "zone_sample_target",
                                        }
                                        else "recorded_input_coordinate"
                                    ),
                                },
                            )
                    line = _compact_array(
                        [
                            session_id,
                            table_name,
                            source,
                            record_key,
                            _bits(raw_x),
                            _bits(raw_y),
                            "null" if cell is None else str(cell),
                        ]
                    )
                    assignment_digest.update((line + "\n").encode("utf-8"))
                if progress is not None:
                    progress(session_index, len(sessions))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    coordinate_sources: dict[str, Any] = {}
    for source, values in counters.items():
        finite = values["finite_coordinates"]
        out_of_bounds = values["out_of_bounds_coordinates"]
        coordinate_sources[source] = {
            **values,
            "out_of_bounds_percentage": (
                0.0 if finite == 0 else 100.0 * out_of_bounds / finite
            ),
        }
    result: dict[str, Any] = {
        "schema_version": "fncs-world-grid-compatibility:1.0",
        "created_utc": utc_now(),
        "status": "compatible",
        "binding_mode": "independent_dataset",
        "dataset_root": str(root),
        "accepted_session_count": len(sessions),
        "all_accepted_sessions_audited": True,
        "ingestion_report_sha256": inventory["ingestion_report_sha256"],
        "dataset_validation_sha256": inventory["dataset_validation_sha256"],
        "profile_path": str(profile_source),
        "profile_raw_sha256": file_sha256(profile_source),
        "profile_id": profile.profile_id,
        "profile_hash": profile.profile_hash,
        "profile_kind": profile.profile_kind,
        "authoritative_terrain_bounds": profile.authoritative_terrain_bounds,
        "publication_audit_path": str(audit_source),
        "publication_audit_raw_sha256": file_sha256(audit_source),
        "publication_audit_sha256": profile.publication_audit.audit_sha256,
        "coordinate_audit_sha256": profile.publication_audit.coordinate_audit_sha256,
        "compatible_builds": list(profile.compatible_builds),
        "observed_build_counts": dict(sorted(builds.items())),
        "observed_changelist_counts": dict(sorted(changelists.items())),
        "observed_world_identity_counts": dict(sorted(worlds.items())),
        "coordinate_scale": {
            "meters_per_world_unit": profile.world_unit_scale.meters_per_world_unit,
            "semantic_source": profile.world_unit_scale.semantic_source,
        },
        "bounds": {
            "x_min": profile.world_x_min,
            "x_max": profile.world_x_max,
            "y_min": profile.world_y_min,
            "y_max": profile.world_y_max,
            "inclusive": True,
        },
        "cell_mapping": {
            "grid_rows": profile.grid_rows,
            "grid_columns": profile.grid_columns,
            "cell_width_world_units": profile.cell_width_world_units,
            "cell_height_world_units": profile.cell_height_world_units,
            "ordering": "row_major",
            "out_of_bounds_cell": None,
        },
        "coordinate_sources": coordinate_sources,
        "cell_assignments_sha256": assignment_digest.hexdigest(),
        "out_of_bounds_coordinate_count": out_of_bounds_total,
        "out_of_bounds_coordinates_path": str(output_path.resolve()),
        "out_of_bounds_coordinates_raw_sha256": file_sha256(output_path),
        "out_of_bounds_recording_complete": True,
        "target_policy": "out-of-bounds targets are masked without clamping",
        "profile_refit_or_modification_performed": False,
        "compatibility_checks": {
            "build": True,
            "changelist": True,
            "map_identity": True,
            "map_grid_geometry": True,
            "coordinate_scale": True,
            "bounds": True,
            "cell_mapping": True,
            "profile_hash": True,
            "publication_audit": True,
        },
        "world_grid_validation_audit_sha256": "",
    }
    deterministic = dict(result)
    deterministic.pop("created_utc")
    deterministic.pop("world_grid_validation_audit_sha256")
    result["world_grid_validation_audit_sha256"] = sha256_bytes(
        canonical_json(deterministic).encode("utf-8")
    )
    return result
