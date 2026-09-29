from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import pyarrow.parquet as pq

from .world_grid import (
    WorldGridProfile,
    WorldGridProfileError,
    _canonical_json,
    _payload_hash,
    ingestion_binding_from_report,
)


_SOURCE_NAMES = (
    "player_position",
    "player_event_position",
    "eligible_player_centroid",
    "zone_phase_source",
    "zone_phase_target",
    "zone_sample_current",
    "zone_sample_target",
)


def _compact_array(values: list[str]) -> str:
    encoded = json.dumps(
        values,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (
        encoded.replace('\\"', "\\u0022")
        .replace("&", "\\u0026")
        .replace("<", "\\u003C")
        .replace(">", "\\u003E")
    )


def _bits(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("coordinate and time values must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("coordinate and time values must be finite")
    return struct.pack(">d", number).hex()


def _integer(value: Any, location: str) -> str:
    if type(value) is not int:
        raise ValueError(f"{location} must be an integer")
    return str(value)


def _nullable_integer(value: Any, location: str) -> str:
    return "null" if value is None else _integer(value, location)


def _finite_pair(x: Any, y: Any) -> tuple[float, float] | None:
    if x is None and y is None:
        return None
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
    ):
        raise ValueError("coordinate pair must be both null or both finite")
    return float(x), float(y)


def _observations(
    directory: Path,
    session_id: str,
) -> list[tuple[str, str, str, Any, Any]]:
    output: list[tuple[str, str, str, Any, Any]] = []

    def rows(table_name: str, required: set[str]) -> list[dict[str, Any]]:
        table = pq.read_table(directory / table_name)
        if not required <= set(table.column_names):
            raise ValueError(f"{table_name} has an unexpected schema")
        values = table.to_pylist()
        if any(row.get("session_id") != session_id for row in values):
            raise ValueError(
                f"{table_name} contains a row outside session {session_id}"
            )
        return values

    for row in rows(
        "player_samples.parquet",
        {
            "session_id",
            "player_id",
            "team_id",
            "match_time_seconds",
            "alive",
            "life_index",
            "x",
            "y",
        },
    ):
        if type(row["alive"]) is not bool:
            raise ValueError("player sample alive must be boolean")
        key = _compact_array(
            [
                str(row["player_id"]),
                _integer(row["team_id"], "player team_id"),
                _bits(row["match_time_seconds"]),
                "true" if row["alive"] else "false",
                _integer(row["life_index"], "player life_index"),
            ]
        )
        output.append(
            (
                "player_position",
                "player_samples.parquet",
                key,
                row["x"],
                row["y"],
            )
        )

    for row in rows(
        "player_events.parquet",
        {
            "session_id",
            "player_id",
            "team_id",
            "match_time_seconds",
            "event_type",
            "life_index",
            "x",
            "y",
            "zone_phase",
            "zone_basis",
            "location_quality",
        },
    ):
        key = _compact_array(
            [
                str(row["player_id"]),
                _integer(row["team_id"], "event team_id"),
                _bits(row["match_time_seconds"]),
                str(row["event_type"]),
                _integer(row["life_index"], "event life_index"),
                _nullable_integer(row["zone_phase"], "event zone_phase"),
                "null" if row["zone_basis"] is None else str(row["zone_basis"]),
                (
                    "null"
                    if row["location_quality"] is None
                    else str(row["location_quality"])
                ),
            ]
        )
        output.append(
            (
                "player_event_position",
                "player_events.parquet",
                key,
                row["x"],
                row["y"],
            )
        )

    for row in rows(
        "team_samples.parquet",
        {
            "session_id",
            "team_id",
            "match_time_seconds",
            "final_placement",
            "x",
            "y",
        },
    ):
        key = _compact_array(
            [
                _integer(row["team_id"], "team sample team_id"),
                _bits(row["match_time_seconds"]),
                _nullable_integer(
                    row["final_placement"], "team sample final_placement"
                ),
            ]
        )
        output.append(
            (
                "eligible_player_centroid",
                "team_samples.parquet",
                key,
                row["x"],
                row["y"],
            )
        )

    for row in rows(
        "zone_phases.parquet",
        {
            "session_id",
            "zone_phase",
            "activation_time_seconds",
            "shrink_start_time_seconds",
            "closure_time_seconds",
            "source_x",
            "source_y",
            "target_x",
            "target_y",
        },
    ):
        key = _compact_array(
            [
                _integer(row["zone_phase"], "zone phase"),
                _bits(row["activation_time_seconds"]),
                _bits(row["shrink_start_time_seconds"]),
                _bits(row["closure_time_seconds"]),
            ]
        )
        output.extend(
            (
                (
                    "zone_phase_source",
                    "zone_phases.parquet",
                    key,
                    row["source_x"],
                    row["source_y"],
                ),
                (
                    "zone_phase_target",
                    "zone_phases.parquet",
                    key,
                    row["target_x"],
                    row["target_y"],
                ),
            )
        )

    for row in rows(
        "zone_samples.parquet",
        {
            "session_id",
            "match_time_seconds",
            "zone_phase",
            "current_boundary_x",
            "current_boundary_y",
            "target_x",
            "target_y",
        },
    ):
        key = _compact_array(
            [
                _bits(row["match_time_seconds"]),
                _nullable_integer(row["zone_phase"], "zone sample phase"),
            ]
        )
        output.extend(
            (
                (
                    "zone_sample_current",
                    "zone_samples.parquet",
                    key,
                    row["current_boundary_x"],
                    row["current_boundary_y"],
                ),
                (
                    "zone_sample_target",
                    "zone_samples.parquet",
                    key,
                    row["target_x"],
                    row["target_y"],
                ),
            )
        )
    return sorted(
        output,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            _bits(item[3]),
            _bits(item[4]),
        ),
    )


def _session_directories(
    dataset_root: Path,
    binding: dict[str, Any],
) -> list[tuple[dict[str, Any], Path]]:
    output: list[tuple[dict[str, Any], Path]] = []
    expected_by_partition: dict[str, set[str]] = {
        "attested": set(),
        "unattested": set(),
    }
    for session in binding["accepted_sessions"]:
        disposition = session["disposition"]
        partition = disposition.removeprefix("included_")
        expected_by_partition[partition].add(session["session_id"])
        path = dataset_root / partition / session["session_id"]
        if not path.is_dir():
            raise FileNotFoundError(f"included dataset session is missing: {path}")
        output.append((session, path))
    for partition, expected in expected_by_partition.items():
        root = dataset_root / partition
        actual = (
            {item.name for item in root.iterdir() if item.is_dir()}
            if root.is_dir()
            else set()
        )
        if actual != expected:
            raise ValueError(
                f"dataset partition {partition} does not exactly match "
                "included ingestion sessions"
            )
    return sorted(output, key=lambda item: item[0]["session_id"])


def validate_dataset_world_grid(
    dataset_root: str | Path,
    profile_path: str | Path,
    audit_path: str | Path,
    *,
    binding_mode: Literal["canonical", "independent"] = "canonical",
) -> dict[str, Any]:
    if binding_mode not in {"canonical", "independent"}:
        raise ValueError("binding_mode must be canonical or independent")
    root = Path(dataset_root)
    report_path = root / "ingestion-report.json"
    profile = WorldGridProfile.load(
        profile_path,
        audit_path=audit_path,
        ingestion_report_path=report_path,
        audit_binding_mode=binding_mode,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    binding = ingestion_binding_from_report(report)
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    canonical_coordinate_audit = audit["coordinate_audit"]
    canonical_sources = canonical_coordinate_audit["coordinate_sources"]

    counters: dict[str, dict[str, Any]] = {
        source: {
            "coordinate_pairs": 0,
            "finite_coordinates": 0,
            "unavailable_coordinate_pairs": 0,
            "out_of_bounds_coordinates": 0,
            "out_of_bounds_examples": [],
        }
        for source in _SOURCE_NAMES
    }
    assignment_digest = hashlib.sha256()

    for session, directory in _session_directories(root, binding):
        session_id = session["session_id"]
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
                    if len(counter["out_of_bounds_examples"]) < 32:
                        counter["out_of_bounds_examples"].append(
                            {
                                "session_id": session_id,
                                "source": source,
                                "table": table_name,
                                "record_key": record_key,
                                "x": pair[0],
                                "y": pair[1],
                            }
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

    summaries: dict[str, dict[str, Any]] = {}
    for source in _SOURCE_NAMES:
        counter = counters[source]
        finite = counter["finite_coordinates"]
        out_of_bounds = counter["out_of_bounds_coordinates"]
        summaries[source] = {
            **counter,
            "out_of_bounds_percentage": (
                0.0 if finite == 0 else 100.0 * out_of_bounds / finite
            ),
        }
    assignment_hash = assignment_digest.hexdigest()

    if binding_mode == "canonical":
        count_fields = (
            "coordinate_pairs",
            "finite_coordinates",
            "unavailable_coordinate_pairs",
            "out_of_bounds_coordinates",
        )
        for source in _SOURCE_NAMES:
            expected = canonical_sources[source]
            actual = summaries[source]
            if any(actual[field] != expected[field] for field in count_fields):
                raise ValueError(
                    f"{source} counts do not match canonical coordinate audit"
                )
        if assignment_hash != canonical_coordinate_audit[
            "cell_assignments_sha256"
        ]:
            raise ValueError(
                "cell-assignment digest does not match canonical coordinate audit"
            )

    ingestion_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()

    result = {
        "schema_version": "2.0",
        "status": "matches_publication_audit",
        "profile_id": profile.profile_id,
        "profile_hash": profile.profile_hash,
        "publication_audit_id": profile.publication_audit.report_id,
        "publication_audit_sha256": profile.publication_audit.audit_sha256,
        "coordinate_audit_sha256": (
            profile.publication_audit.coordinate_audit_sha256
        ),
        "ingestion_binding_sha256": binding["ingestion_binding_sha256"],
        "accepted_session_count": binding["accepted_session_count"],
        "accepted_session_counts_by_build": binding[
            "accepted_session_counts_by_build"
        ],
        "coordinate_sources": summaries,
        "cell_assignments_sha256": assignment_hash,
        "canonical_counts_match": True,
        "canonical_digest_match": True,
        "session_order_invariant": True,
        "input_row_order_invariant": True,
        "split_membership_used": False,
        "observed_extrema_used_for_bounds": False,
        "report_hash": "",
    }
    if binding_mode == "independent":
        result.update(
            {
                "status": "compatible_independent_dataset",
                "binding_mode": "independent",
                "publication_ingestion_binding_sha256": (
                    canonical_coordinate_audit["dataset_binding"][
                        "ingestion_binding_sha256"
                    ]
                ),
                "ingestion_report_sha256": ingestion_digest,
                "canonical_counts_match": None,
                "canonical_digest_match": None,
                "dataset_compatibility_verified": True,
            }
        )
    result["report_hash"] = _payload_hash(result, "report_hash")
    return result


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all accepted dataset-v2 absolute XY streams against "
            "the published model-world partition without training"
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--binding-mode",
        choices=("canonical", "independent"),
        default="canonical",
        help=(
            "Use independent to verify the published contract against a "
            "different report-backed corpus without requiring canonical counts."
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = validate_dataset_world_grid(
            args.dataset_root,
            args.profile,
            args.audit,
            binding_mode=args.binding_mode,
        )
    except (OSError, ValueError, WorldGridProfileError) as exc:
        parser.error(str(exc))
    _write_atomic(Path(args.output), result)
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
