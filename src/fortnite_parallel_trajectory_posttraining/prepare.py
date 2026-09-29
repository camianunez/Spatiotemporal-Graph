from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fncs_encoder_training.common import append_jsonl
from fortnite_encoder.planner_supervision import eligible_planner_queries
from fortnite_parallel_trajectory.targets import build_parallel_trajectory_targets
from fortnite_parallel_trajectory_fncs_training.data import (
    FNCSParallelTrajectorySessionRepository,
)

from .bindings import load_lineage_setups
from .common import (
    ANALYSIS_DIRECTORY,
    HORIZONS_SECONDS,
    SCALE_SESSION_COUNTS,
    SEED,
    current_utc,
    file_record,
    file_sha256,
    make_file_read_only,
    payload_sha256,
    read_json,
    read_jsonl,
    replace_json,
    require,
    self_seal,
    verify_self_seal,
    write_new_json,
)
from .contracts import SCALE_CONTRACT_PATH, VALIDATION_CONTRACT_PATH
from .scale import exposure_summary


PROGRESS_PATH = ANALYSIS_DIRECTORY / "training_subset_census_progress.jsonl"
ACCESS_LEDGER_PATH = ANALYSIS_DIRECTORY / "data_access_training_subset_census.json"
FINAL_ACCESS_PATH = ANALYSIS_DIRECTORY / "training_subset_census_access_final.json"
MANIFEST_PATH = ANALYSIS_DIRECTORY / "nested_subset_manifest.json"


def _shard_progress_path(shard_index: int, shard_count: int) -> Path:
    return ANALYSIS_DIRECTORY / f"training_subset_census_shard_{shard_index}_of_{shard_count}.jsonl"


def _shard_ledger_path(shard_index: int, shard_count: int) -> Path:
    return ANALYSIS_DIRECTORY / f"data_access_training_subset_census_shard_{shard_index}_of_{shard_count}.json"


def _score(session_id: str) -> str:
    payload = str(SEED).encode("utf-8") + b"\0" + session_id.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_training_ids(session_ids: Sequence[str]) -> tuple[str, ...]:
    require(len(session_ids) == 920 and len(set(session_ids)) == 920, "training universe changed")
    return tuple(sorted(session_ids, key=lambda session_id: (_score(session_id), session_id)))


def _inventory_rows(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = inventory.get("sessions")
    require(isinstance(rows, list), "dataset inventory has no session rows")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        require(isinstance(raw, Mapping), "dataset inventory session row is invalid")
        session_id = raw.get("game_session_id")
        require(isinstance(session_id, str) and session_id not in result, "inventory session ID is invalid")
        result[session_id] = raw
    require(len(result) == 1150, "dataset inventory session count changed")
    return result


def _metadata_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    tables = raw.get("tables")
    require(isinstance(tables, list) and len(tables) == 7, "inventory table set changed")
    table_rows: list[dict[str, Any]] = []
    for table in sorted(tables, key=lambda value: str(value.get("path"))):
        require(isinstance(table, Mapping), "inventory table row is invalid")
        table_rows.append(
            {
                "path": table.get("path"),
                "sha256": table.get("sha256"),
                "byte_size": table.get("byte_size"),
                "row_count": table.get("row_count"),
            }
        )
    return {
        "source_replay_relative_path": raw.get("source_replay_relative_path"),
        "source_replay_sha256": raw.get("source_replay_sha256"),
        "build": raw.get("build"),
        "branch": raw.get("branch"),
        "changelist": raw.get("changelist"),
        "world_identity": raw.get("world_identity"),
        "ingestion_disposition": raw.get("disposition"),
        "data_status": raw.get("data_status"),
        "data_issue_codes": raw.get("data_issue_codes"),
        "provenance_status": raw.get("provenance_status"),
        "provenance_issue_codes": raw.get("provenance_issue_codes"),
        "trust_partition": raw.get("trust_partition"),
        "reopened_and_validated": raw.get("reopened_and_validated"),
        "tables": table_rows,
        "table_records_sha256": payload_sha256(table_rows),
    }


def _census_row(
    *, session_id: str, order_index: int, source: Any, inventory: Mapping[str, Any]
) -> dict[str, Any]:
    match = source.match
    tick_positions = {
        int(value): index
        for index, value in enumerate(match.absolute_tick_index.cpu().tolist())
    }
    planner_eligible = eligible_planner_queries(source)
    candidates = tuple(
        query
        for query in planner_eligible
        if 1
        <= int(match.zone_phase[tick_positions[query.absolute_tick_index]].item())
        <= 8
    )
    targets = build_parallel_trajectory_targets(source, candidates)
    eligible_mask = targets.query_mask & targets.target_mask.any(dim=-1)
    valid_target_counts = {
        str(horizon): int((targets.target_mask[:, index] & eligible_mask).sum().item())
        for index, horizon in enumerate(HORIZONS_SECONDS)
    }
    return {
        "schema_version": "fncs-decoder-training-subset-census-row:1.0",
        "order_index_zero_based": order_index,
        "session_id": session_id,
        "selection_score_sha256": _score(session_id),
        "all_planner_eligible_query_count": len(planner_eligible),
        "phase_1_through_8_candidate_window_count": len(candidates),
        "eligible_route_query_count": int(eligible_mask.sum().item()),
        "valid_target_count_by_horizon_seconds": valid_target_counts,
        "valid_target_count_total": sum(valid_target_counts.values()),
        "source": _metadata_payload(inventory),
    }


def _load_progress(ordered_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not PROGRESS_PATH.is_file():
        return []
    rows = read_jsonl(PROGRESS_PATH, "training subset census progress")
    require(len(rows) <= len(ordered_ids), "census progress exceeds training universe")
    for index, row in enumerate(rows):
        require(
            row.get("order_index_zero_based") == index
            and row.get("session_id") == ordered_ids[index]
            and row.get("selection_score_sha256") == _score(ordered_ids[index]),
            f"census progress diverged at row {index}",
        )
    return rows


def _verify_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    validation = read_json(VALIDATION_CONTRACT_PATH, "validation contract")
    scale = read_json(SCALE_CONTRACT_PATH, "scale contract")
    verify_self_seal(validation, "contract_payload_sha256")
    verify_self_seal(scale, "contract_payload_sha256")
    require(
        scale.get("subset_construction", {}).get("seed") == SEED,
        "scale subset seed changed",
    )
    return validation, scale


def build_census_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    require(type(shard_count) is int and 2 <= shard_count <= 8, "shard count must be 2..8")
    require(type(shard_index) is int and 0 <= shard_index < shard_count, "shard index is invalid")
    _verify_contracts()
    setup, _, _, _ = load_lineage_setups()
    ordered = _ordered_training_ids(tuple(setup.split_session_ids["train"]))
    canonical_rows = _load_progress(ordered)
    canonical_indices = {
        int(row["order_index_zero_based"]) for row in canonical_rows
    }
    progress_path = _shard_progress_path(shard_index, shard_count)
    externally_completed = set(canonical_indices)
    for other_path in ANALYSIS_DIRECTORY.glob("training_subset_census_shard_*_of_*.jsonl"):
        if other_path == progress_path:
            continue
        for other_row in read_jsonl(other_path, f"prior census shard {other_path.name}"):
            other_index = other_row.get("order_index_zero_based")
            require(
                type(other_index) is int
                and 0 <= other_index < len(ordered)
                and other_row.get("session_id") == ordered[other_index]
                and other_row.get("selection_score_sha256") == _score(ordered[other_index]),
                f"prior shard row changed in {other_path.name}",
            )
            externally_completed.add(other_index)
    assigned = [
        index
        for index in range(len(ordered))
        if index % shard_count == shard_index and index not in externally_completed
    ]
    ledger_path = _shard_ledger_path(shard_index, shard_count)
    rows = read_jsonl(progress_path, f"census shard {shard_index}") if progress_path.is_file() else []
    require(len(rows) <= len(assigned), "shard progress exceeds its assignment")
    for position, row in enumerate(rows):
        index = assigned[position]
        require(
            row.get("order_index_zero_based") == index
            and row.get("session_id") == ordered[index]
            and row.get("selection_score_sha256") == _score(ordered[index]),
            f"shard {shard_index} diverged at local row {position}",
        )
    inventory_by_id = _inventory_rows(setup.inventory)
    repository = FNCSParallelTrajectorySessionRepository(
        paths=setup.session_paths,
        partitions=setup.partition_by_session,
        active_partitions=("train",),
        test_session_ids=setup.split_session_ids["test"],
        profile=setup.profile,
        ledger_path=ledger_path,
        split_manifest_sha256=setup.config.expected_split_manifest_sha256,
        phase=f"training_subset_census_shard_{shard_index}_of_{shard_count}",
        resume_ledger=ledger_path.is_file(),
    )
    for position in range(len(rows), len(assigned)):
        index = assigned[position]
        session_id = ordered[index]
        source = repository.get(session_id)
        row = _census_row(
            session_id=session_id,
            order_index=index,
            source=source,
            inventory=inventory_by_id[session_id],
        )
        append_jsonl(progress_path, row)
        rows.append(row)
        repository._cache.pop(session_id, None)
        if (position + 1) % 5 == 0 or position + 1 == len(assigned):
            repository.assert_no_test_access()
            print(
                json.dumps(
                    {
                        "event": "training_subset_census_shard",
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                        "completed_assigned_sessions": position + 1,
                        "total_assigned_sessions": len(assigned),
                        "test_attempts": 0,
                        "test_opens": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            gc.collect()
    repository.assert_no_test_access()
    return {
        "status": "shard_complete",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "row_count": len(rows),
        "progress": file_record(progress_path),
        "ledger": file_record(ledger_path),
    }


def assemble_census_shards(shard_count: int) -> dict[str, Any]:
    require(type(shard_count) is int and 2 <= shard_count <= 8, "shard count must be 2..8")
    _verify_contracts()
    setup, _, _, _ = load_lineage_setups()
    ordered = _ordered_training_ids(tuple(setup.split_session_ids["train"]))
    canonical = _load_progress(ordered)
    by_index = {int(row["order_index_zero_based"]): row for row in canonical}
    expected_paths = [_shard_progress_path(index, shard_count) for index in range(shard_count)]
    require(all(path.is_file() for path in expected_paths), "one or more final census shards are missing")
    all_paths = sorted(ANALYSIS_DIRECTORY.glob("training_subset_census_shard_*_of_*.jsonl"))
    for path in all_paths:
        for row in read_jsonl(path, f"census shard {path.name}"):
            index = row.get("order_index_zero_based")
            require(
                type(index) is int
                and 0 <= index < len(ordered)
                and row.get("session_id") == ordered[index]
                and row.get("selection_score_sha256") == _score(ordered[index]),
                f"shard row {index} changed",
            )
            prior = by_index.get(index)
            require(prior is None or prior == row, f"conflicting duplicate census row {index}")
            by_index[index] = row
    require(set(by_index) == set(range(920)), "census shards have gaps")
    for index in range(len(canonical), 920):
        append_jsonl(PROGRESS_PATH, by_index[index])
    completed = _load_progress(ordered)
    require(len(completed) == 920, "assembled census is incomplete")
    return {
        "status": "assembled",
        "row_count": len(completed),
        "progress": file_record(PROGRESS_PATH),
    }


def build_nested_subset_manifest() -> dict[str, Any]:
    validation_contract, scale_contract = _verify_contracts()
    if MANIFEST_PATH.is_file():
        manifest = read_json(MANIFEST_PATH, "nested subset manifest")
        verify_self_seal(manifest, "manifest_payload_sha256")
        return {"status": "already_complete", "manifest": file_record(MANIFEST_PATH)}
    setup, _, _, _ = load_lineage_setups()
    train_ids = tuple(setup.split_session_ids["train"])
    validation_ids = set(setup.split_session_ids["validation"])
    test_ids = set(setup.split_session_ids["test"])
    ordered = _ordered_training_ids(train_ids)
    require(not set(ordered) & validation_ids, "training universe overlaps validation")
    require(not set(ordered) & test_ids, "training universe overlaps test")
    inventory_by_id = _inventory_rows(setup.inventory)
    rows = _load_progress(ordered)
    if len(rows) < len(ordered):
        repository = FNCSParallelTrajectorySessionRepository(
            paths=setup.session_paths,
            partitions=setup.partition_by_session,
            active_partitions=("train",),
            test_session_ids=setup.split_session_ids["test"],
            profile=setup.profile,
            ledger_path=ACCESS_LEDGER_PATH,
            split_manifest_sha256=setup.config.expected_split_manifest_sha256,
            phase="training_subset_census",
            resume_ledger=ACCESS_LEDGER_PATH.is_file(),
        )
        for index in range(len(rows), len(ordered)):
            session_id = ordered[index]
            source = repository.get(session_id)
            row = _census_row(
                session_id=session_id,
                order_index=index,
                source=source,
                inventory=inventory_by_id[session_id],
            )
            append_jsonl(PROGRESS_PATH, row)
            rows.append(row)
            repository._cache.pop(session_id, None)
            if (index + 1) % 10 == 0 or index + 1 == len(ordered):
                repository.assert_no_test_access()
                print(
                    json.dumps(
                        {
                            "event": "training_subset_census",
                            "completed_sessions": index + 1,
                            "total_sessions": len(ordered),
                            "test_attempts": 0,
                            "test_opens": 0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                gc.collect()
    require(len(rows) == 920, "training subset census is incomplete")
    ledger_paths = [ACCESS_LEDGER_PATH] if ACCESS_LEDGER_PATH.is_file() else []
    ledger_paths.extend(sorted(ANALYSIS_DIRECTORY.glob("data_access_training_subset_census_shard_*_of_*.json")))
    require(ledger_paths, "training census has no access ledger")
    access_reports = [read_json(path, f"training census ledger {path.name}") for path in ledger_paths]
    require(
        all(
            access["zero_test_attempts"] is True
            and access["zero_test_opens"] is True
            and access["zero_validation_attempts"] is True
            and access["zero_validation_opens"] is True
            for access in access_reports
        ),
        "training census crossed a partition boundary",
    )
    request_counts: Counter[str] = Counter()
    attempt_counts: Counter[str] = Counter()
    open_counts: Counter[str] = Counter()
    for access in access_reports:
        request_counts.update(access["session_request_counts"])
        attempt_counts.update(access["dataset_attempt_counts"])
        open_counts.update(access["dataset_open_counts"])
    final_access = {
        "schema_version": "fncs-decoder-training-subset-census-access-final:1.0",
        "created_utc": current_utc(),
        "progress_row_count": len(rows),
        "progress": file_record(PROGRESS_PATH),
        "ledgers": [file_record(path) for path in ledger_paths],
        "session_request_counts": dict(sorted(request_counts.items())),
        "dataset_attempt_counts": dict(sorted(attempt_counts.items())),
        "dataset_open_counts": dict(sorted(open_counts.items())),
        "test_session_request_attempt_count": sum(int(value["test_session_request_attempt_count"]) for value in access_reports),
        "test_session_parquet_open_attempt_count": sum(int(value["test_session_parquet_open_attempt_count"]) for value in access_reports),
        "test_session_parquet_open_count": sum(int(value["test_session_parquet_open_count"]) for value in access_reports),
        "zero_test_attempts": True,
        "zero_test_opens": True,
        "zero_validation_attempts": True,
        "zero_validation_opens": True,
    }
    replace_json(FINAL_ACCESS_PATH, final_access)
    subsets: dict[str, Any] = {}
    for size in SCALE_SESSION_COUNTS:
        selected_rows = rows[:size]
        session_ids = [row["session_id"] for row in selected_rows]
        horizon_counts = {
            str(horizon): sum(
                int(row["valid_target_count_by_horizon_seconds"][str(horizon)])
                for row in selected_rows
            )
            for horizon in HORIZONS_SECONDS
        }
        data_status = Counter(row["source"]["data_status"] for row in selected_rows)
        provenance_status = Counter(
            row["source"]["provenance_status"] for row in selected_rows
        )
        builds = Counter(row["source"]["build"] for row in selected_rows)
        subsets[str(size)] = {
            "name": f"S_{size}",
            "unique_session_count": len(set(session_ids)),
            "session_ids": session_ids,
            "session_ids_sha256": payload_sha256(session_ids),
            "selection_score_sha256s": [
                row["selection_score_sha256"] for row in selected_rows
            ],
            "is_prefix_of_next_subset": size != SCALE_SESSION_COUNTS[-1],
            "phase_1_through_8_candidate_window_count": sum(
                int(row["phase_1_through_8_candidate_window_count"])
                for row in selected_rows
            ),
            "eligible_route_query_count": sum(
                int(row["eligible_route_query_count"]) for row in selected_rows
            ),
            "valid_target_count_by_horizon_seconds": horizon_counts,
            "valid_target_count_total": sum(horizon_counts.values()),
            "build_counts": dict(sorted(builds.items())),
            "ingestion_data_status_counts": dict(sorted(data_status.items())),
            "provenance_status_counts": dict(sorted(provenance_status.items())),
            "effective_exposure": exposure_summary(session_ids, 1840),
            "session_census": selected_rows,
        }
    for smaller, larger in zip(SCALE_SESSION_COUNTS, SCALE_SESSION_COUNTS[1:]):
        require(
            subsets[str(larger)]["session_ids"][:smaller]
            == subsets[str(smaller)]["session_ids"],
            f"S_{smaller} is not a prefix of S_{larger}",
        )
    manifest = self_seal(
        {
            "schema_version": "fncs-decoder-nested-training-subsets:1.0",
            "created_utc": current_utc(),
            "selection_frozen_before_scale_training": True,
            "selection_uses_no_validation_or_outcome_information": True,
            "seed": SEED,
            "score_algorithm": "SHA256(UTF8(decimal seed) || NUL || UTF8(canonical session ID))",
            "ordering": "ascending score bytes, then ascending canonical session ID",
            "training_universe_count": len(ordered),
            "training_universe_session_ids_sha256": payload_sha256(sorted(ordered)),
            "ordered_training_session_ids_sha256": payload_sha256(list(ordered)),
            "validation_overlap_count": len(set(ordered) & validation_ids),
            "test_overlap_count": len(set(ordered) & test_ids),
            "nested_prefixes_verified": True,
            "census_rows_sha256": payload_sha256(rows),
            "census_artifacts": {
                "progress": file_record(PROGRESS_PATH),
                "access_final": file_record(FINAL_ACCESS_PATH),
            },
            "source_bindings": {
                "validation_contract_sha256": file_sha256(VALIDATION_CONTRACT_PATH),
                "validation_contract_payload_sha256": validation_contract[
                    "contract_payload_sha256"
                ],
                "scale_contract_sha256": file_sha256(SCALE_CONTRACT_PATH),
                "scale_contract_payload_sha256": scale_contract["contract_payload_sha256"],
                "dataset_inventory_raw_sha256": setup.config.expected_dataset_inventory_sha256,
                "ingestion_report_raw_sha256": setup.config.expected_ingestion_report_sha256,
                "split_manifest_payload_sha256": setup.config.expected_split_manifest_sha256,
                "target_builder_sha256": file_sha256(
                    Path(__file__).resolve().parents[1]
                    / "fortnite_parallel_trajectory"
                    / "targets.py"
                ),
                "census_builder_sha256": file_sha256(Path(__file__)),
            },
            "subsets": subsets,
        },
        "manifest_payload_sha256",
    )
    write_new_json(MANIFEST_PATH, manifest)
    make_file_read_only(MANIFEST_PATH)
    return {"status": "completed", "manifest": file_record(MANIFEST_PATH)}


def status_report() -> dict[str, Any]:
    rows = read_jsonl(PROGRESS_PATH) if PROGRESS_PATH.is_file() else []
    result: dict[str, Any] = {
        "completed_census_sessions": len(rows),
        "total_census_sessions": 920,
        "manifest_status": "complete" if MANIFEST_PATH.is_file() else "pending",
    }
    if MANIFEST_PATH.is_file():
        manifest = read_json(MANIFEST_PATH)
        verify_self_seal(manifest, "manifest_payload_sha256")
        result["manifest"] = file_record(MANIFEST_PATH)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build locked nested scale subsets and census.")
    parser.add_argument("command", choices=("build", "status", "shard", "assemble"))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build_nested_subset_manifest()
    elif args.command == "status":
        result = status_report()
    elif args.command == "shard":
        require(args.shard_index is not None, "--shard-index is required for shard")
        result = build_census_shard(args.shard_index, args.shard_count)
    else:
        result = assemble_census_shards(args.shard_count)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MANIFEST_PATH", "build_nested_subset_manifest", "status_report"]
