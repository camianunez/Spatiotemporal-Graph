from __future__ import annotations

import hashlib
import json
import mmap
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from fortnite_encoder.supervision import _SUPERVISION_SCHEMAS
from fortnite_encoder.tensorize import _INPUT_SCHEMAS

from .common import atomic_write_json, canonical_json, file_sha256, sha256_bytes, utc_now


DATASET_SCHEMA_VERSION = "2.0.0"
REPORT_SCHEMA_VERSION = "2.0"
EXPECTED_ACCEPTED = 1150
EXPECTED_REJECTED = 1357
TABLE_NAMES = (
    "centroid_neighbors.parquet",
    "match_samples.parquet",
    "player_events.parquet",
    "player_samples.parquet",
    "team_samples.parquet",
    "zone_phases.parquet",
    "zone_samples.parquet",
)
EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "match_samples.parquet": (
        "session_id",
        "match_time_seconds",
        "players_remaining",
        "teams_remaining",
    ),
    "player_samples.parquet": (
        "session_id",
        "player_id",
        "team_id",
        "match_time_seconds",
        "alive",
        "life_index",
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
    ),
    "team_samples.parquet": (
        "session_id",
        "team_id",
        "match_time_seconds",
        "final_placement",
        "x",
        "y",
        "z",
    ),
    "player_events.parquet": (
        "session_id",
        "player_id",
        "team_id",
        "match_time_seconds",
        "event_type",
        "life_index",
        "x",
        "y",
        "z",
        "zone_phase",
        "zone_basis",
        "location_quality",
    ),
    "zone_phases.parquet": (
        "session_id",
        "zone_phase",
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
    ),
    "zone_samples.parquet": (
        "session_id",
        "match_time_seconds",
        "zone_phase",
        "time_until_closure_seconds",
        "current_boundary_x",
        "current_boundary_y",
        "current_boundary_z",
        "current_boundary_radius",
        "target_x",
        "target_y",
        "target_z",
        "target_radius",
    ),
    "centroid_neighbors.parquet": (
        "session_id",
        "match_time_seconds",
        "source_team_id",
        "neighbor_team_id",
        "neighbor_rank",
        "distance_xy",
        "delta_x",
        "delta_y",
    ),
}
CENTROID_NEIGHBOR_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=True),
        pa.field("match_time_seconds", pa.float64(), nullable=False),
        pa.field("source_team_id", pa.int32(), nullable=False),
        pa.field("neighbor_team_id", pa.int32(), nullable=False),
        pa.field("neighbor_rank", pa.int32(), nullable=False),
        pa.field("distance_xy", pa.float64(), nullable=False),
        pa.field("delta_x", pa.float64(), nullable=False),
        pa.field("delta_y", pa.float64(), nullable=False),
    ]
)
EXPECTED_SCHEMAS = {
    **_INPUT_SCHEMAS,
    "team_samples.parquet": _SUPERVISION_SCHEMAS["team_samples.parquet"],
    "player_events.parquet": _SUPERVISION_SCHEMAS["player_events.parquet"],
    "centroid_neighbors.parquet": CENTROID_NEIGHBOR_SCHEMA,
}


def _header_value(prefix: bytes, name: str) -> Any:
    marker = (f'"{name}"').encode("utf-8")
    offset = prefix.find(marker)
    if offset < 0:
        raise ValueError(f"ingestion report header is missing {name}")
    colon = prefix.find(b":", offset + len(marker))
    if colon < 0:
        raise ValueError(f"ingestion report header field {name} is malformed")
    decoder = json.JSONDecoder()
    text = prefix[colon + 1 :].decode("utf-8")
    value, _ = decoder.raw_decode(text.lstrip())
    return value


def ingestion_report_header(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as stream:
        prefix = stream.read(1024 * 1024)
    items_offset = prefix.find(b'"items"')
    if items_offset < 0:
        raise ValueError("ingestion report does not contain an items array")
    header = prefix[:items_offset]
    return {
        name: _header_value(header, name)
        for name in (
            "schema_version",
            "dataset_schema_version",
            "raw_file_count",
            "unique_payload_count",
            "replay_count",
        )
    }


def iter_ingestion_items(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream top-level ingestion ``items`` without loading the 600+ MiB report."""

    source = Path(path)
    with source.open("rb") as stream, mmap.mmap(
        stream.fileno(), length=0, access=mmap.ACCESS_READ
    ) as mapped:
        marker = mapped.find(b'"items"')
        if marker < 0:
            raise ValueError("ingestion report does not contain items")
        array_start = mapped.find(b"[", marker)
        if array_start < 0:
            raise ValueError("ingestion report items is not an array")
        index = array_start + 1
        size = len(mapped)
        while True:
            while index < size and mapped[index] in b" \t\r\n,":
                index += 1
            if index >= size:
                raise ValueError("unterminated ingestion report items array")
            if mapped[index] == ord("]"):
                return
            if mapped[index] != ord("{"):
                raise ValueError(f"ingestion report item at byte {index} is not an object")
            start = index
            depth = 0
            in_string = False
            escaped = False
            while index < size:
                value = mapped[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif value == ord("\\"):
                        escaped = True
                    elif value == ord('"'):
                        in_string = False
                else:
                    if value == ord('"'):
                        in_string = True
                    elif value == ord("{"):
                        depth += 1
                    elif value == ord("}"):
                        depth -= 1
                        if depth == 0:
                            raw = mapped[start : index + 1]
                            item = json.loads(raw)
                            if not isinstance(item, dict):
                                raise ValueError("ingestion report item is not an object")
                            yield item
                            index += 1
                            break
                index += 1
            else:
                raise ValueError("unterminated ingestion report item")


def _issue_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            str(item["code"])
            for item in value
            if isinstance(item, dict) and isinstance(item.get("code"), str)
        }
    )


def scan_allowlist(report_path: str | Path) -> dict[str, Any]:
    header = ingestion_report_header(report_path)
    if header["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("ingestion report schema_version must be 2.0")
    if header["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError("ingestion report dataset schema must be 2.0.0")
    accepted: list[dict[str, Any]] = []
    rejected = 0
    report_session_ids: set[str] = set()
    duplicate_report_session_ids: list[str] = []
    rejected_reasons: Counter[str] = Counter()
    for raw in iter_ingestion_items(report_path):
        session_id = raw.get("session_id")
        if isinstance(session_id, str) and session_id:
            if session_id in report_session_ids:
                duplicate_report_session_ids.append(session_id)
            report_session_ids.add(session_id)
        disposition = raw.get("disposition")
        if disposition in {"included_attested", "included_unattested"}:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("included report item has no canonical session_id")
            replay_sha256 = raw.get("replay_sha256")
            if (
                not isinstance(replay_sha256, str)
                or len(replay_sha256) != 64
                or any(character not in "0123456789abcdef" for character in replay_sha256)
            ):
                raise ValueError(f"included session {session_id} has an invalid replay hash")
            build_profile = raw.get("build_profile")
            if not isinstance(build_profile, dict):
                raise ValueError(f"included session {session_id} has no build profile")
            output_artifacts = raw.get("output_artifacts")
            if not isinstance(output_artifacts, list):
                raise ValueError(f"included session {session_id} has no output artifacts")
            accepted.append(
                {
                    "game_session_id": session_id,
                    "source_replay_sha256": replay_sha256,
                    "source_replay_relative_path": raw.get("replay_relative_path"),
                    "data_status": raw.get("data_status"),
                    "provenance_status": raw.get("provenance_status"),
                    "disposition": disposition,
                    "trust_partition": disposition.removeprefix("included_"),
                    "build": build_profile.get("build"),
                    "changelist": build_profile.get("changelist"),
                    "branch": build_profile.get("branch"),
                    "world_identity": raw.get("world_identity"),
                    "map_grid": raw.get("map_grid"),
                    "data_issue_codes": _issue_codes(raw.get("data_issues")),
                    "provenance_issue_codes": _issue_codes(raw.get("provenance_issues")),
                    "output_artifacts": output_artifacts,
                    "stratification_fields": {
                        name: raw.get(name)
                        for name in ("region", "division", "event", "week")
                    },
                }
            )
        elif disposition == "rejected":
            rejected += 1
            for issue in raw.get("data_issues", []):
                if isinstance(issue, dict) and isinstance(issue.get("code"), str):
                    rejected_reasons[issue["code"]] += 1
        else:
            raise ValueError(f"unsupported ingestion disposition {disposition!r}")
    if duplicate_report_session_ids:
        raise ValueError(
            "ingestion report duplicates session IDs: "
            + ", ".join(sorted(duplicate_report_session_ids)[:8])
        )
    if len(accepted) != EXPECTED_ACCEPTED or rejected != EXPECTED_REJECTED:
        raise ValueError(
            f"ingestion allowlist counts are {len(accepted)}/{rejected}, expected "
            f"{EXPECTED_ACCEPTED}/{EXPECTED_REJECTED}"
        )
    if len({item["game_session_id"] for item in accepted}) != EXPECTED_ACCEPTED:
        raise ValueError("accepted canonical game_session_id values are not unique")
    if len({item["source_replay_sha256"] for item in accepted}) != EXPECTED_ACCEPTED:
        raise ValueError("accepted source replay hashes are not unique")
    if any(item["data_status"] != "accepted_with_warnings" for item in accepted):
        raise ValueError("all accepted pilot sessions must preserve accepted_with_warnings")
    if any(item["provenance_status"] != "incomplete" for item in accepted):
        raise ValueError("all accepted pilot sessions must preserve incomplete provenance")
    return {
        "header": header,
        "accepted": sorted(accepted, key=lambda item: item["game_session_id"]),
        "rejected_count": rejected,
        "rejected_reason_occurrences": dict(sorted(rejected_reasons.items())),
    }


def _validation_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("passed") is not True
        or raw.get("dataset_schema_version") != DATASET_SCHEMA_VERSION
        or not isinstance(raw.get("sessions"), list)
    ):
        raise ValueError("source dataset-validation.json is not a passing v2 report")
    summary = raw.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("accepted_canonical_sessions") != EXPECTED_ACCEPTED
        or summary.get("rejected_sessions") != EXPECTED_REJECTED
    ):
        raise ValueError("source dataset-validation counts do not match the task")
    by_session: dict[str, Any] = {}
    for item in raw["sessions"]:
        if not isinstance(item, dict) or not isinstance(item.get("game_session_id"), str):
            raise ValueError("source validation contains an invalid session")
        session_id = item["game_session_id"]
        if session_id in by_session:
            raise ValueError(f"source validation duplicates session {session_id}")
        by_session[session_id] = item
    return raw, by_session


def _safe_dataset_path(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"dataset artifact escapes the dataset root: {relative}")
    return candidate


def _schema_signature(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def validate_dataset(
    dataset_root: str | Path,
    validation_path: str | Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    report_path = root / "ingestion-report.json"
    source_validation_path = Path(validation_path).resolve()
    report_hash_before = file_sha256(report_path)
    validation_hash_before = file_sha256(source_validation_path)
    allowlist = scan_allowlist(report_path)
    source_validation, validation_by_session = _validation_report(
        source_validation_path
    )
    accepted_ids = {item["game_session_id"] for item in allowlist["accepted"]}
    if set(validation_by_session) != accepted_ids:
        raise ValueError("source validation and ingestion allowlist session sets differ")

    actual_directories: dict[str, str] = {}
    transient_paths: list[str] = []
    for trust in ("attested", "unattested"):
        partition = root / trust
        if not partition.is_dir():
            continue
        for child in partition.iterdir():
            if child.is_dir():
                if child.name.startswith(".") or child.name.endswith((".staging", ".backup")):
                    transient_paths.append(str(child))
                    continue
                if child.name in actual_directories:
                    raise ValueError(f"session directory {child.name} exists in two partitions")
                actual_directories[child.name] = trust
    if transient_paths:
        raise ValueError(f"transient dataset directories remain: {transient_paths[:8]}")
    if set(actual_directories) != accepted_ids:
        missing = sorted(accepted_ids - set(actual_directories))
        unbound = sorted(set(actual_directories) - accepted_ids)
        raise ValueError(f"dataset directories are not exactly report-bound; missing={missing[:8]}, unbound={unbound[:8]}")

    sessions: list[dict[str, Any]] = []
    table_totals: Counter[str] = Counter()
    total_bytes = 0
    for index, item in enumerate(allowlist["accepted"], start=1):
        session_id = item["game_session_id"]
        trust = item["trust_partition"]
        if actual_directories[session_id] != trust:
            raise ValueError(f"session {session_id} is published in the wrong trust partition")
        directory = root / trust / session_id
        actual_names = sorted(path.name for path in directory.glob("*.parquet") if path.is_file())
        if tuple(actual_names) != TABLE_NAMES:
            raise ValueError(f"session {session_id} does not contain exactly the seven v2 tables")
        report_tables = {
            Path(raw["path"]).name: raw
            for raw in item["output_artifacts"]
            if isinstance(raw, dict) and isinstance(raw.get("path"), str)
        }
        validation_item = validation_by_session[session_id]
        if validation_item.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(f"session {session_id} is not schema 2.0.0")
        validation_tables = {
            Path(raw["path"]).name: raw
            for raw in validation_item.get("tables", [])
            if isinstance(raw, dict) and isinstance(raw.get("path"), str)
        }
        if set(report_tables) != set(TABLE_NAMES) or set(validation_tables) != set(TABLE_NAMES):
            raise ValueError(f"session {session_id} table manifests are not exact")
        tables: list[dict[str, Any]] = []
        for table_name in TABLE_NAMES:
            report_table = report_tables[table_name]
            validation_table = validation_tables[table_name]
            if any(
                report_table.get(name) != validation_table.get(name)
                for name in ("path", "byte_size", "sha256", "row_count")
            ):
                raise ValueError(f"session {session_id} {table_name} report bindings differ")
            expected_relative = f"{trust}/{session_id}/{table_name}"
            if report_table["path"].replace("\\", "/") != expected_relative:
                raise ValueError(f"session {session_id} {table_name} has a noncanonical path")
            table_path = _safe_dataset_path(root, report_table["path"])
            stat = table_path.stat()
            digest = file_sha256(table_path)
            parquet = pq.ParquetFile(table_path)
            schema = parquet.schema_arrow
            if list(schema.names) != list(EXPECTED_COLUMNS[table_name]):
                raise ValueError(f"session {session_id} {table_name} has unexpected columns")
            if not schema.equals(EXPECTED_SCHEMAS[table_name], check_metadata=False):
                raise ValueError(f"session {session_id} {table_name} has an unexpected schema")
            if (
                stat.st_size != report_table["byte_size"]
                or digest != report_table["sha256"]
                or parquet.metadata.num_rows != report_table["row_count"]
            ):
                raise ValueError(f"session {session_id} {table_name} changed after ingestion")
            total_bytes += stat.st_size
            table_totals[table_name] += parquet.metadata.num_rows
            tables.append(
                {
                    "path": expected_relative,
                    "byte_size": stat.st_size,
                    "row_count": parquet.metadata.num_rows,
                    "sha256": digest,
                    "schema": _schema_signature(schema),
                }
            )
        sessions.append(
            {
                **{key: value for key, value in item.items() if key != "output_artifacts"},
                "directory": f"{trust}/{session_id}",
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "tables": tables,
                "reopened_and_validated": True,
            }
        )
        if progress is not None:
            progress(index, EXPECTED_ACCEPTED)

    report_hash_after = file_sha256(report_path)
    validation_hash_after = file_sha256(source_validation_path)
    if report_hash_before != report_hash_after or validation_hash_before != validation_hash_after:
        raise ValueError("dataset control artifacts changed during validation")
    result: dict[str, Any] = {
        "schema_version": "fncs-dataset-validation:1.0",
        "created_utc": utc_now(),
        "dataset_root": str(root),
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "ingestion_report_path": str(report_path),
        "ingestion_report_sha256": report_hash_after,
        "source_dataset_validation_path": str(source_validation_path),
        "source_dataset_validation_raw_sha256": validation_hash_after,
        "source_dataset_validation_passed": source_validation.get("passed") is True,
        "accepted_canonical_sessions": len(sessions),
        "accepted_with_warning_sessions": sum(
            item["data_status"] == "accepted_with_warnings" for item in sessions
        ),
        "unattested_sessions": sum(
            item["trust_partition"] == "unattested" for item in sessions
        ),
        "rejected_replay_payloads": allowlist["rejected_count"],
        "duplicate_game_session_ids": [],
        "unbound_directories": [],
        "validated_table_count": len(sessions) * len(TABLE_NAMES),
        "total_parquet_bytes": total_bytes,
        "table_row_counts": dict(sorted(table_totals.items())),
        "rejected_reason_occurrences": allowlist["rejected_reason_occurrences"],
        "checks": {
            "ingestion_allowlist_is_authoritative": True,
            "exact_1150_accepted": len(sessions) == EXPECTED_ACCEPTED,
            "exact_1357_rejected": allowlist["rejected_count"] == EXPECTED_REJECTED,
            "all_sessions_reopened": len(sessions) == EXPECTED_ACCEPTED,
            "exactly_seven_tables_per_session": True,
            "all_table_bytes_rehashed": True,
            "all_parquet_footers_reopened": True,
            "all_schemas_exact": True,
            "all_row_counts_exact": True,
            "no_unbound_directories": True,
            "no_duplicate_game_session_ids": True,
            "dataset_not_modified": True,
        },
        "sessions": sessions,
        "dataset_validation_sha256": "",
    }
    deterministic = dict(result)
    deterministic.pop("created_utc")
    deterministic.pop("dataset_validation_sha256")
    result["dataset_validation_sha256"] = sha256_bytes(
        canonical_json(deterministic).encode("utf-8")
    )
    return result


def build_split_manifest(
    inventory: Mapping[str, Any],
    *,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> dict[str, Any]:
    sessions = inventory.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != train_count + validation_count + test_count:
        raise ValueError("dataset inventory does not match requested split counts")
    if inventory.get("accepted_canonical_sessions") != len(sessions):
        raise ValueError("dataset inventory accepted count is inconsistent")
    scored: list[tuple[str, dict[str, Any]]] = []
    for item in sessions:
        session_id = item["game_session_id"]
        score = hashlib.sha256(
            b"fncs-fresh-encoder-split-v1\0"
            + str(seed).encode("ascii")
            + b"\0"
            + session_id.encode("utf-8")
        ).hexdigest()
        scored.append((score, item))
    scored.sort(key=lambda pair: (pair[0], pair[1]["game_session_id"]))
    boundaries = (train_count, train_count + validation_count)
    partitions = {
        "train": scored[: boundaries[0]],
        "validation": scored[boundaries[0] : boundaries[1]],
        "test": scored[boundaries[1] :],
    }
    assignments: list[dict[str, Any]] = []
    ordered: dict[str, list[str]] = {}
    replay_hashes: dict[str, list[str]] = {}
    for partition, rows in partitions.items():
        ordered[partition] = []
        replay_hashes[partition] = []
        for order, (score, item) in enumerate(rows):
            session_id = item["game_session_id"]
            replay_hash = item["source_replay_sha256"]
            ordered[partition].append(session_id)
            replay_hashes[partition].append(replay_hash)
            assignments.append(
                {
                    "partition": partition,
                    "partition_order": order,
                    "game_session_id": session_id,
                    "source_replay_sha256": replay_hash,
                    "hash_score_sha256": score,
                    "data_status": item["data_status"],
                    "provenance_status": item["provenance_status"],
                    "trust_partition": item["trust_partition"],
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": "fncs-canonical-session-split:1.0",
        "created_utc": utc_now(),
        "immutable": True,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "split_algorithm": "ascending_sha256(seed, canonical_game_session_id)",
        "split_algorithm_domain": "fncs-fresh-encoder-split-v1",
        "seed": seed,
        "stratification": {
            "used": False,
            "reason": (
                "Reliable region, division, event, and week provenance is unavailable: "
                "all 1,150 accepted sessions are unattested with incomplete provenance."
            ),
            "fields_considered": ["region", "division", "event", "week"],
        },
        "counts": {name: len(rows) for name, rows in partitions.items()},
        "ordered_session_ids": ordered,
        "ordered_source_replay_sha256": replay_hashes,
        "assignments": assignments,
        "ingestion_report_sha256": inventory["ingestion_report_sha256"],
        "dataset_validation_sha256": inventory["dataset_validation_sha256"],
        "split_manifest_sha256": "",
    }
    deterministic = dict(manifest)
    deterministic.pop("created_utc")
    deterministic.pop("split_manifest_sha256")
    manifest["split_manifest_sha256"] = sha256_bytes(
        canonical_json(deterministic).encode("utf-8")
    )
    expected = {"train": train_count, "validation": validation_count, "test": test_count}
    if manifest["counts"] != expected:
        raise AssertionError("split implementation did not produce exact counts")
    return manifest


def write_inventory(path: str | Path, inventory: Mapping[str, Any]) -> None:
    atomic_write_json(path, inventory)


def write_split_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(path, manifest)


def load_bound_inventory(path: str | Path, expected_hash: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("dataset_validation_sha256") != expected_hash:
        raise ValueError("dataset inventory payload hash mismatch")
    deterministic = dict(value)
    deterministic.pop("created_utc", None)
    recorded = deterministic.pop("dataset_validation_sha256", None)
    actual = sha256_bytes(canonical_json(deterministic).encode("utf-8"))
    if recorded != actual:
        raise ValueError("dataset inventory is not self-consistent")
    return value


def load_bound_split(path: str | Path, expected_hash: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("split_manifest_sha256") != expected_hash:
        raise ValueError("split manifest payload hash mismatch")
    deterministic = dict(value)
    deterministic.pop("created_utc", None)
    recorded = deterministic.pop("split_manifest_sha256", None)
    actual = sha256_bytes(canonical_json(deterministic).encode("utf-8"))
    if recorded != actual:
        raise ValueError("split manifest is not self-consistent")
    return value
