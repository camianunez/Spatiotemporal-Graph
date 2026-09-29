from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields, is_dataclass, replace
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from fncs_encoder_training.artifacts import environment_report
from fncs_encoder_training.common import (
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    file_sha256,
    rng_state,
    seed_everything,
    source_tree_manifest,
    tensor_state_sha256,
    utc_now,
)
from fortnite_encoder.planner_supervision import PlannerQuery
from fortnite_parallel_trajectory_training.data import (
    ParallelTrajectoryQueryExample,
    collate_parallel_trajectory_query_examples,
    prepare_parallel_trajectory_query_examples,
    session_batches,
    training_session_order,
)
from fortnite_parallel_trajectory_training.metrics import (
    ParallelTrajectoryMetricAccumulator,
)
from fortnite_parallel_trajectory_training.training import (
    evaluate_validation,
    resolve_precision,
    run_training_batch,
    sample_training_batch_queries,
)
from fortnite_parallel_trajectory_fncs_training.binding import (
    VerifiedSetup,
    assert_frozen_encoder,
    build_frozen_model,
    downstream_state,
    verify_setup,
)
from fortnite_parallel_trajectory_fncs_training.checkpoint import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from fortnite_parallel_trajectory_fncs_training.config import (
    FROZEN_FNCS_CHECKPOINT_SCHEMA,
    FrozenFNCSTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
)
from fortnite_parallel_trajectory_fncs_training.data import (
    FNCSParallelTrajectorySessionRepository,
)
from fortnite_parallel_trajectory_fncs_training.optimization import (
    PiecewiseLinearScheduler,
    build_optimizer,
)
from fortnite_parallel_trajectory_fncs_training.training import (
    _append_frozen_proof,
    _resolved_config,
    _runtime_payload,
    _verify_preflight,
    _write_latest,
    validate_resolved_resume_contract,
    write_artifact_manifest,
)


PARENT_NAME = "parallel-trajectory-decoder-v1-fncs-frozen-encoder-run-1"
CHILD_PREFIX = f"{PARENT_NAME}-resume-"
EXPECTED_STEP = 1840
EXPECTED_NEXT_STEP = 1841
EXPECTED_ENDPOINT = 9200
EXPECTED_CHECKPOINT_SHA256 = (
    "5826b65803887a399c33cd26bf4f2111423af0fef1631e2423eab2f056e542f9"
)
EXPECTED_PARENT_INVENTORY_SHA256 = (
    "b8779132aca2f6dc3cab6854c8efa7ff5bfa3db642cf6b4fc962e2588b2acb2e"
)
EXPECTED_CONFIG_SHA256 = (
    "b13aac4b834f31e1ff609ae95aa86d683bebf5bac8d17fa396e7decd085b1785"
)
OLD_TRAINING_SOURCE_SHA256 = (
    "1f477ba9f91bf652367ea903d35af06b0a5466cc09f511252ae18f4f069882bd"
)
NEW_TRAINING_SOURCE_SHA256 = (
    "50c0a0bf1ffa0f02323b4594670cbfecdb4918afd7196b93226a6f14835f4177"
)
OLD_TRAINING_PY_SHA256 = (
    "78036feeccb7907ee55c9344d167025533049189f73d0fe3d4cfeb39ad04fe90"
)
NEW_TRAINING_PY_SHA256 = (
    "b49e5e9c855ff4c1883b5191e65c2f3368874feb07e97f91dea2d22443f5f9c4"
)
BEST_CHECKPOINT_SHA256 = (
    "fc77f9828a2ae2c5408284569ae71f2cce860b4a1c65b1948a4a4e072da7e0b3"
)
ENCODER_STATE_SHA256 = (
    "a70edc067a0b4e284b93b4a60ae472d0f2d9b4bc226143feecd841a505a9489f"
)
CANONICAL_ENCODER_SHA256 = (
    "dc8d2dc2e9bfce5ed9cacd557190dd33ca2d8af5d6225ae54464234d8cc3992a"
)
EXECUTION_ENCODER_SHA256 = (
    "71c48713546c60aaf7573eacd199f8972c2fd012c9375c2296d89cff2d78cfb5"
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingCompatibilityError(f"cannot read {label}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TrainingCompatibilityError(f"{label} must be a JSON object")
    return raw


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrainingCompatibilityError(f"cannot read {label}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingCompatibilityError(
                f"{label} line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise TrainingCompatibilityError(
                f"{label} line {line_number} is not an object"
            )
        records.append(value)
    return records


def _finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    return False


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        item = value.detach().cpu().contiguous()
        digest.update(b"torch\0")
        digest.update(str(item.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_json(list(item.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        item = np.ascontiguousarray(value)
        digest.update(b"numpy\0")
        digest.update(str(item.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_json(list(item.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        return
    if isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            _update_digest(digest, item)
        return
    if isinstance(value, float) and not math.isfinite(value):
        digest.update(b"float\0")
        digest.update(
            ("nan" if math.isnan(value) else "+inf" if value > 0 else "-inf").encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    )
    digest.update(b"\0")


def structure_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _clone_structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_structure(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_structure(item) for item in value)
    if isinstance(value, list):
        return [_clone_structure(item) for item in value]
    return copy.deepcopy(value)


def _structure_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_structure_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_structure_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _all_tensor_values_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(
            torch.isfinite(value).all()
        )
    if isinstance(value, np.ndarray):
        return value.dtype.kind not in "fc" or bool(np.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_tensor_values_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_tensor_values_finite(item) for item in value)
    return True


def _path_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    source = root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"inventory root is missing: {source}")
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(source).as_posix(),
    ):
        rows.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return rows, digest.hexdigest()


def _pid_alive(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist.exe", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout.strip()
    return result.returncode == 0 and output.startswith('"') and f'"{pid}"' in output


def _is_read_only(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_READONLY", 1))


def _object_payload(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError(f"expected dataclass payload, got {type(value).__name__}")
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _training_package_root() -> Path:
    import fortnite_parallel_trajectory_fncs_training.training as training_module

    return Path(training_module.__file__).resolve().parent


def _recovery_source_manifest() -> dict[str, Any]:
    return source_tree_manifest(Path(__file__).resolve().parent)


def _build_setups(
    config_path: Path, parent_resolved: Mapping[str, Any]
) -> tuple[VerifiedSetup, VerifiedSetup, dict[str, Any], dict[str, Any]]:
    base_config = FrozenFNCSTrainingConfig.from_json(config_path)
    if base_config.configuration_raw_sha256 != EXPECTED_CONFIG_SHA256:
        raise TrainingCompatibilityError("canonical configuration hash changed")
    old_manifest = parent_resolved.get("training_source_manifest")
    if not isinstance(old_manifest, Mapping):
        raise TrainingCompatibilityError("parent training source manifest is missing")
    old_manifest = dict(old_manifest)
    current_manifest = source_tree_manifest(_training_package_root())
    if (
        old_manifest.get("source_tree_sha256") != OLD_TRAINING_SOURCE_SHA256
        or current_manifest.get("source_tree_sha256") != NEW_TRAINING_SOURCE_SHA256
        or base_config.expected_training_source_digest != OLD_TRAINING_SOURCE_SHA256
    ):
        raise TrainingCompatibilityError("training source transition hash changed")
    transition_config = replace(
        base_config, expected_training_source_digest=NEW_TRAINING_SOURCE_SHA256
    )
    verified = verify_setup(transition_config)
    current_setup = replace(verified, config=base_config)
    checkpoint_setup = replace(
        current_setup, training_source_manifest=old_manifest
    )
    _verify_preflight(checkpoint_setup)
    return current_setup, checkpoint_setup, old_manifest, current_manifest


def _source_transition_report(
    audit_dir: Path,
    old_manifest: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    old_entries = {row["path"]: row["sha256"] for row in old_manifest["entries"]}
    new_entries = {row["path"]: row["sha256"] for row in current_manifest["entries"]}
    missing = sorted(set(old_entries) - set(new_entries))
    added = sorted(set(new_entries) - set(old_entries))
    changed = [
        {
            "path": name,
            "old_sha256": old_entries[name],
            "new_sha256": new_entries[name],
        }
        for name in sorted(set(old_entries) & set(new_entries))
        if old_entries[name] != new_entries[name]
    ]
    old_training = audit_dir / "source-before" / "training.py"
    current_training = _training_package_root() / "training.py"
    if (
        missing
        or added
        or changed
        != [
            {
                "path": "training.py",
                "old_sha256": OLD_TRAINING_PY_SHA256,
                "new_sha256": NEW_TRAINING_PY_SHA256,
            }
        ]
        or file_sha256(old_training) != OLD_TRAINING_PY_SHA256
        or file_sha256(current_training) != NEW_TRAINING_PY_SHA256
    ):
        raise TrainingCompatibilityError(
            "source transition is not limited to the resume validator"
        )
    old_lines = old_training.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = current_training.read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    unified = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="training.py@parent",
            tofile="training.py@recovery",
        )
    )
    return {
        "old_training_source_manifest": dict(old_manifest),
        "new_training_source_manifest": dict(current_manifest),
        "missing_files": missing,
        "added_files": added,
        "changed_files": changed,
        "unified_diff": unified,
        "model_execution_changed": False,
        "losses_changed": False,
        "optimizer_updates_changed": False,
        "scheduler_changed": False,
        "sampling_changed": False,
        "checkpoint_payload_changed": False,
        "focused_tests": {
            "path": "ml/tests/test_parallel_trajectory_fncs_resume_compatibility.py",
            "passed": 13,
            "failed": 0,
        },
    }


def _common_bindings(
    setup: VerifiedSetup,
    *,
    parent_inventory_sha256: str,
    recovery_source_sha256: str,
) -> dict[str, Any]:
    config = setup.config
    return {
        "step1840_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "parent_inventory_sha256": parent_inventory_sha256,
        "encoder": {
            "canonical_best_pt_sha256": CANONICAL_ENCODER_SHA256,
            "execution_encoder_only_best_pt_sha256": EXECUTION_ENCODER_SHA256,
            "frozen_encoder_state_sha256": ENCODER_STATE_SHA256,
        },
        "configuration_sha256": config.configuration_raw_sha256,
        "dataset": {
            "inventory_raw_sha256": config.expected_dataset_inventory_sha256,
            "ingestion_report_raw_sha256": config.expected_ingestion_report_sha256,
            "dataset_validation_raw_sha256": (
                config.expected_dataset_validation_raw_sha256
            ),
            "dataset_validation_sha256": config.expected_dataset_validation_sha256,
        },
        "split": {
            "raw_sha256": config.expected_split_manifest_raw_sha256,
            "canonical_sha256": config.expected_split_manifest_sha256,
            "counts": dict(config.expected_split_counts),
        },
        "source": {
            "old_training_source_sha256": OLD_TRAINING_SOURCE_SHA256,
            "new_training_source_sha256": NEW_TRAINING_SOURCE_SHA256,
            "old_training_py_sha256": OLD_TRAINING_PY_SHA256,
            "new_training_py_sha256": NEW_TRAINING_PY_SHA256,
            "recovery_source_sha256": recovery_source_sha256,
        },
    }


def _optimizer_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    groups = raw.get("param_groups")
    states = raw.get("state")
    if not isinstance(groups, list) or not isinstance(states, Mapping):
        raise TrainingCompatibilityError("optimizer state is malformed")
    steps: list[int] = []
    moment_entries = 0
    for value in states.values():
        if not isinstance(value, Mapping):
            raise TrainingCompatibilityError("optimizer parameter state is malformed")
        step = value.get("step")
        if isinstance(step, torch.Tensor) and step.numel() == 1:
            steps.append(int(step.item()))
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            steps.append(int(step))
        if "exp_avg" in value and "exp_avg_sq" in value:
            moment_entries += 1
    return {
        "parameter_group_count": len(groups),
        "state_entry_count": len(states),
        "parameters_with_first_and_second_moments": moment_entries,
        "recorded_step_values": sorted(set(steps)),
        "all_state_tensors_finite": _all_tensor_values_finite(raw),
        "state_sha256": structure_sha256(raw),
    }


def _strict_checkpoint_report(
    checkpoint_path: Path,
    setup: VerifiedSetup,
    *,
    label: str,
) -> dict[str, Any]:
    model, initial = build_frozen_model(setup, device="cpu")
    optimizer, optimizer_contract = build_optimizer(model, setup.config)
    scheduler = PiecewiseLinearScheduler(optimizer, setup.config)
    metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
    state, best = load_checkpoint(
        checkpoint_path,
        setup=setup,
        model=model,
        optimizer=optimizer,
        optimizer_contract=optimizer_contract,
        scheduler=scheduler,
        initial_downstream_parameters=initial,
        metrics=metrics,
        restore_rng=False,
    )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frozen = assert_frozen_encoder(model, setup, stage=f"{label}_strict_reload")
    rng = raw["rng_state"]
    rng_complete = (
        isinstance(rng, Mapping)
        and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        and isinstance(rng["python"], tuple)
        and isinstance(rng["numpy"], tuple)
        and isinstance(rng["torch_cpu"], torch.Tensor)
        and isinstance(rng["torch_cuda"], list)
        and len(rng["torch_cuda"]) == torch.cuda.device_count()
    )
    report = {
        "path": str(checkpoint_path.resolve()),
        "raw_sha256": file_sha256(checkpoint_path),
        "strict_application_load": True,
        "checkpoint_schema_version": raw["checkpoint_schema_version"],
        "checkpoint_type": raw["type"],
        "top_level_keys": sorted(raw),
        "training_state": state.to_dict(),
        "best_validation_route_nll": best,
        "downstream_state_sha256": tensor_state_sha256(downstream_state(model)),
        "embedded_downstream_state_sha256": raw["downstream_state_sha256"],
        "downstream_tensor_count": len(raw["downstream_state"]),
        "optimizer": _optimizer_report(raw["optimizer_state"]),
        "scheduler_state": dict(raw["scheduler_state"]),
        "scheduler_step_index": scheduler.step_index,
        "scheduler_upcoming_update": scheduler.upcoming_update,
        "learning_rates_for_upcoming_update": scheduler.get_last_lr(),
        "rng_state_complete": rng_complete,
        "rng_state_sha256": structure_sha256(rng),
        "metric_accumulator_state_sha256": structure_sha256(
            raw["metric_accumulator_state"]
        ),
        "sampler_position": {
            "epoch": state.current_epoch,
            "next_batch_index": state.next_batch_index,
            "derivation": "seed_epoch_batch_and_sorted_split_ids",
        },
        "frozen_encoder": frozen,
        "data_access_zero_test_attempts": raw["data_access"].get(
            "zero_test_attempts"
        ),
        "data_access_zero_test_opens": raw["data_access"].get("zero_test_opens"),
        "test_partition_evaluated": raw["test_partition_evaluated"],
        "all_downstream_tensors_finite": _all_tensor_values_finite(
            raw["downstream_state"]
        ),
        "all_optimizer_tensors_finite": _all_tensor_values_finite(
            raw["optimizer_state"]
        ),
    }
    del raw, model, optimizer, scheduler, metrics
    return report


def _metrics_report(parent: Path) -> dict[str, Any]:
    records = _read_jsonl(parent / "metrics.jsonl", "parent metrics")
    optimizer_records = [row for row in records if row.get("type") == "optimizer_step"]
    durable = [row for row in optimizer_records if int(row["optimizer_step"]) <= 1840]
    post = [row for row in optimizer_records if int(row["optimizer_step"]) > 1840]
    validations = [
        row
        for row in records
        if row.get("type") == "validation_epoch"
        and int(row["optimizer_step"]) <= 1840
    ]
    if [row["optimizer_step"] for row in durable] != list(range(1, 1841)):
        raise TrainingCompatibilityError(
            "parent optimizer metrics are not exactly contiguous through step 1840"
        )
    if not records or not all(_finite_json(row) for row in records):
        raise TrainingCompatibilityError("parent metrics contain nonfinite values")
    incumbent = min(validations, key=lambda row: float(row["selection_value"]))
    return {
        "record_count": len(records),
        "all_records_finite": True,
        "durable_optimizer_step_count": len(durable),
        "durable_optimizer_steps_exactly_1_through_1840": True,
        "durable_last_record": durable[-1],
        "validation_records_through_step1840": validations,
        "actual_incumbent": {
            "completed_epoch": incumbent["completed_epoch"],
            "optimizer_step": incumbent["optimizer_step"],
            "selection_metric": incumbent["selection_metric"],
            "selection_value": incumbent["selection_value"],
        },
        "excluded_nondurable_parent_records": {
            "optimizer_step_count": len(post),
            "first_optimizer_step": post[0]["optimizer_step"] if post else None,
            "last_optimizer_step": post[-1]["optimizer_step"] if post else None,
            "incorporated_into_child": False,
        },
    }


def _parent_access_report(parent: Path) -> dict[str, Any]:
    access = _read_json(parent / "data_access.json", "parent data-access ledger")
    sealed = set(access.get("sealed_test_session_ids", []))
    events = access.get("events")
    if not isinstance(events, list):
        raise TrainingCompatibilityError("parent access events are missing")
    test_matches = sum(
        1
        for event in events
        if event.get("partition") == "test" or event.get("session_id") in sealed
    )
    report = {
        "sealed_test_session_count": access.get("sealed_test_session_count"),
        "sealed_test_session_ids_unique": len(sealed),
        "test_event_count": test_matches,
        "test_session_request_attempt_count": access.get(
            "test_session_request_attempt_count"
        ),
        "test_session_parquet_open_attempt_count": access.get(
            "test_session_parquet_open_attempt_count"
        ),
        "test_session_parquet_open_count": access.get(
            "test_session_parquet_open_count"
        ),
        "test_partition_evaluated": access.get("test_partition_evaluated"),
        "zero_test_attempts": access.get("zero_test_attempts"),
        "zero_test_opens": access.get("zero_test_opens"),
        "dataset_attempt_counts": access.get("dataset_attempt_counts"),
        "dataset_open_counts": access.get("dataset_open_counts"),
    }
    if not (
        report["sealed_test_session_count"] == 115
        and report["sealed_test_session_ids_unique"] == 115
        and test_matches == 0
        and report["test_session_request_attempt_count"] == 0
        and report["test_session_parquet_open_attempt_count"] == 0
        and report["test_session_parquet_open_count"] == 0
        and report["test_partition_evaluated"] is False
        and report["zero_test_attempts"] is True
        and report["zero_test_opens"] is True
    ):
        raise TrainingCompatibilityError("parent records forbidden test access")
    return report


def _query_payload(queries: Sequence[PlannerQuery]) -> list[dict[str, Any]]:
    return [asdict(query) for query in queries]


def _derive_examples(
    *,
    setup: VerifiedSetup,
    repository: FNCSParallelTrajectorySessionRepository,
    epoch: int,
    batch_index: int,
) -> tuple[
    tuple[PlannerQuery, ...],
    tuple[ParallelTrajectoryQueryExample, ...],
    dict[str, Any],
]:
    config = setup.config
    order = training_session_order(
        setup.split_session_ids["train"], seed=config.seed, epoch=epoch
    )
    batches = session_batches(order, config.batch_size)
    if len(batches) != config.updates_per_epoch:
        raise TrainingCompatibilityError("deterministic training batch count changed")
    session_ids = batches[batch_index]
    sources = {session_id: repository.get(session_id) for session_id in session_ids}
    queries = sample_training_batch_queries(
        sources,
        session_ids,
        seed=config.seed,
        epoch=epoch,
        session_batch_index=batch_index,
        queries_per_session=config.queries_per_session,
    )
    examples = prepare_parallel_trajectory_query_examples(
        sources, queries, context_length_ticks=config.context_length_ticks
    )
    expected = len(session_ids) * config.queries_per_session
    if len(queries) != expected or len(examples) != expected:
        raise TrainingCompatibilityError("deterministic recovery query count changed")
    batch, observation, targets = collate_parallel_trajectory_query_examples(examples)
    input_payload = {
        "encoder_batch": _object_payload(batch),
        "planner_observation": _object_payload(observation),
    }
    target_payload = {
        "target_displacements": targets.target_displacements,
        "current_xy": targets.current_xy,
        "future_xy": targets.future_xy,
    }
    mask_payload = {
        "target_mask": targets.target_mask,
        "query_mask": targets.query_mask,
    }
    identity = _query_payload(queries)
    hashes = {
        "ordered_session_ids_sha256": structure_sha256(order),
        "batch_session_ids": list(session_ids),
        "batch_session_ids_sha256": structure_sha256(session_ids),
        "query_identities": identity,
        "query_identities_sha256": structure_sha256(identity),
        "input_sha256": structure_sha256(input_payload),
        "target_sha256": structure_sha256(target_payload),
        "mask_sha256": structure_sha256(mask_payload),
    }
    return queries, examples, hashes


def _run_restored_update(
    *,
    checkpoint_setup: VerifiedSetup,
    checkpoint_path: Path,
    examples: Sequence[ParallelTrajectoryQueryExample],
    input_hashes: Mapping[str, Any],
    spec: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = checkpoint_setup.config
    seed_everything(config.seed)
    model, initial = build_frozen_model(checkpoint_setup, device=spec.device)
    optimizer, optimizer_contract = build_optimizer(model, config)
    scheduler = PiecewiseLinearScheduler(optimizer, config)
    metrics = ParallelTrajectoryMetricAccumulator(checkpoint_setup.profile)
    state, best = load_checkpoint(
        checkpoint_path,
        setup=checkpoint_setup,
        model=model,
        optimizer=optimizer,
        optimizer_contract=optimizer_contract,
        scheduler=scheduler,
        initial_downstream_parameters=initial,
        metrics=metrics,
        restore_rng=True,
    )
    if state.optimizer_step != EXPECTED_STEP or scheduler.step_index != EXPECTED_STEP:
        raise TrainingCompatibilityError("restored checkpoint is not at step 1840")
    model.train()
    frozen_before = assert_frozen_encoder(
        model, checkpoint_setup, stage="deterministic_restore_before_step1841"
    )
    encoder_outputs: list[Any] = []
    gradients: dict[str, torch.Tensor | None] = {}

    def capture_encoder(_module: Any, _inputs: Any, output: Any) -> None:
        encoder_outputs.append(_clone_structure(_object_payload(output)))

    encoder_handle = model.encoder.register_forward_hook(capture_encoder)
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_clip(
        parameters: Iterable[torch.Tensor], *args: Any, **kwargs: Any
    ) -> Any:
        values = list(parameters)
        for name, parameter in model.named_parameters():
            gradients[name] = (
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
            )
        return original_clip(values, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = capture_clip
    try:
        result = run_training_batch(
            model=model,
            examples=examples,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,  # type: ignore[arg-type]
            spec=spec,
            metrics=metrics,
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        encoder_handle.remove()
    state_after = replace(
        state,
        optimizer_step=state.optimizer_step + 1,
        next_batch_index=state.next_batch_index + 1,
        first_optimizer_step_completed=True,
    )
    frozen_after = assert_frozen_encoder(
        model, checkpoint_setup, stage="deterministic_restore_after_step1841"
    )
    downstream = downstream_state(model)
    optimizer_state = _clone_structure(optimizer.state_dict())
    scheduler_state = _clone_structure(scheduler.state_dict())
    next_rng = _clone_structure(rng_state())
    metric_state = _clone_structure(metrics.state_dict())
    encoder_gradients = {
        name: value for name, value in gradients.items() if name.startswith("encoder.")
    }
    downstream_gradients = {
        name: value for name, value in gradients.items() if not name.startswith("encoder.")
    }
    encoder_nonzero = [
        name
        for name, value in encoder_gradients.items()
        if value is not None and bool(torch.count_nonzero(value).item())
    ]
    downstream_nonzero = [
        name
        for name, value in downstream_gradients.items()
        if value is not None and bool(torch.count_nonzero(value).item())
    ]
    snapshot = {
        "input_hashes": dict(input_hashes),
        "loss_and_schedule": result.to_dict(),
        "gradients_before_clipping": gradients,
        "downstream_state": downstream,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": next_rng,
        "sampler_state": {
            "current_epoch": state_after.current_epoch,
            "next_batch_index": state_after.next_batch_index,
            "optimizer_step": state_after.optimizer_step,
            "ordered_session_ids_sha256": input_hashes[
                "ordered_session_ids_sha256"
            ],
        },
        "metric_accumulator_state": metric_state,
        "encoder_outputs": encoder_outputs,
        "frozen_encoder_after": frozen_after,
    }
    summary = {
        "training_state_before": state.to_dict(),
        "training_state_after": state_after.to_dict(),
        "best_validation_route_nll": best,
        "input_hashes": dict(input_hashes),
        "loss_and_schedule": result.to_dict(),
        "loss_sha256": structure_sha256(result.to_dict()),
        "gradients_before_clipping_sha256": structure_sha256(gradients),
        "all_gradients_finite": _all_tensor_values_finite(gradients),
        "downstream_parameters_with_nonzero_gradient": len(downstream_nonzero),
        "encoder_parameters_with_nonzero_gradient": len(encoder_nonzero),
        "downstream_state_sha256": tensor_state_sha256(downstream),
        "downstream_tensors_all_finite": _all_tensor_values_finite(downstream),
        "optimizer_state_sha256": structure_sha256(optimizer_state),
        "optimizer_tensors_all_finite": _all_tensor_values_finite(optimizer_state),
        "scheduler_state_sha256": structure_sha256(scheduler_state),
        "rng_state_sha256": structure_sha256(next_rng),
        "sampler_state_sha256": structure_sha256(snapshot["sampler_state"]),
        "metric_accumulator_state_sha256": structure_sha256(metric_state),
        "encoder_outputs_sha256": structure_sha256(encoder_outputs),
        "frozen_encoder_before": frozen_before,
        "frozen_encoder_after": frozen_after,
    }
    del model, optimizer, scheduler, metrics
    torch.cuda.empty_cache()
    return snapshot, summary


def _deterministic_restore_comparison(
    *,
    current_setup: VerifiedSetup,
    checkpoint_setup: VerifiedSetup,
    checkpoint_path: Path,
    access_path: Path,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if access_path.exists():
        raise FileExistsError(f"deterministic access ledger already exists: {access_path}")
    spec = resolve_precision()
    repository = FNCSParallelTrajectorySessionRepository(
        paths=current_setup.session_paths,
        partitions=current_setup.partition_by_session,
        active_partitions=("train",),
        test_session_ids=current_setup.split_session_ids["test"],
        profile=current_setup.profile,
        ledger_path=access_path,
        split_manifest_sha256=current_setup.config.expected_split_manifest_sha256,
        phase="deterministic_restore_step_1841",
        resume_ledger=False,
    )
    _, examples_a, hashes_a = _derive_examples(
        setup=current_setup, repository=repository, epoch=5, batch_index=0
    )
    first, first_summary = _run_restored_update(
        checkpoint_setup=checkpoint_setup,
        checkpoint_path=checkpoint_path,
        examples=examples_a,
        input_hashes=hashes_a,
        spec=spec,
    )
    _, examples_b, hashes_b = _derive_examples(
        setup=current_setup, repository=repository, epoch=5, batch_index=0
    )
    second, second_summary = _run_restored_update(
        checkpoint_setup=checkpoint_setup,
        checkpoint_path=checkpoint_path,
        examples=examples_b,
        input_hashes=hashes_b,
        spec=spec,
    )
    comparisons = {
        "batch_and_query_identities": hashes_a["query_identities"]
        == hashes_b["query_identities"]
        and hashes_a["batch_session_ids"] == hashes_b["batch_session_ids"],
        "input_hash": hashes_a["input_sha256"] == hashes_b["input_sha256"],
        "target_hash": hashes_a["target_sha256"] == hashes_b["target_sha256"],
        "mask_hash": hashes_a["mask_sha256"] == hashes_b["mask_sha256"],
        "losses_learning_rates_and_gradient_norm": _structure_equal(
            first["loss_and_schedule"], second["loss_and_schedule"]
        ),
        "gradients_before_clipping": _structure_equal(
            first["gradients_before_clipping"],
            second["gradients_before_clipping"],
        ),
        "post_update_downstream_state": _structure_equal(
            first["downstream_state"], second["downstream_state"]
        ),
        "post_update_optimizer_state": _structure_equal(
            first["optimizer_state"], second["optimizer_state"]
        ),
        "post_update_scheduler_state": _structure_equal(
            first["scheduler_state"], second["scheduler_state"]
        ),
        "post_update_rng_state": _structure_equal(
            first["rng_state"], second["rng_state"]
        ),
        "post_update_sampler_state": _structure_equal(
            first["sampler_state"], second["sampler_state"]
        ),
        "metric_accumulator_state": _structure_equal(
            first["metric_accumulator_state"],
            second["metric_accumulator_state"],
        ),
        "frozen_encoder_outputs": _structure_equal(
            first["encoder_outputs"], second["encoder_outputs"]
        ),
        "frozen_encoder_tensor_state": _structure_equal(
            first["frozen_encoder_after"], second["frozen_encoder_after"]
        ),
    }
    access = repository.report()
    ledger = _read_json(access_path, "deterministic data-access ledger")
    ledger["recovery_bindings"] = dict(bindings)
    atomic_write_json(access_path, ledger)
    gates = {
        **{f"exact_{name}": value for name, value in comparisons.items()},
        "first_update_is_step1841": first_summary["training_state_after"]
        == {
            "completed_epochs": 4,
            "optimizer_step": 1841,
            "current_epoch": 5,
            "next_batch_index": 1,
            "first_optimizer_step_completed": True,
        },
        "finite_losses_gradients_parameters_optimizer": all(
            (
                first_summary["all_gradients_finite"],
                first_summary["downstream_tensors_all_finite"],
                first_summary["optimizer_tensors_all_finite"],
                _finite_json(first_summary["loss_and_schedule"]),
            )
        ),
        "nonzero_downstream_gradients": first_summary[
            "downstream_parameters_with_nonzero_gradient"
        ]
        > 0,
        "zero_encoder_gradients": first_summary[
            "encoder_parameters_with_nonzero_gradient"
        ]
        == 0,
        "encoder_hash_unchanged": first_summary["frozen_encoder_after"][
            "encoder_state_sha256"
        ]
        == ENCODER_STATE_SHA256,
        "training_only_repository": access.get("allowed_partitions") == ["train"],
        "zero_validation_access": access.get("validation_session_request_count") == 0
        and access.get("validation_dataset_attempt_count") == 0
        and access.get("validation_dataset_open_count") == 0,
        "zero_forbidden_test_access": access.get("zero_test_attempts") is True
        and access.get("zero_test_opens") is True,
    }
    report = {
        "schema_version": "fncs-frozen-decoder-deterministic-restore:1.0",
        "created_utc": utc_now(),
        "status": "passed" if all(gates.values()) else "failed",
        "bindings": dict(bindings),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "restoration_count": 2,
        "validation_executions": 0,
        "test_executions": 0,
        "comparisons": comparisons,
        "gates": gates,
        "trial_1": first_summary,
        "trial_2": second_summary,
        "data_access": {
            "ledger_path": str(access_path.resolve()),
            "ledger_sha256": file_sha256(access_path),
            "report": access,
        },
    }
    return report


def _environment_payload() -> dict[str, Any]:
    workspace = Path(__file__).resolve().parents[3]
    payload = environment_report(workspace)
    payload.update(
        {
            "recovery_created_utc": utc_now(),
            "complete_command_line": [sys.executable, *sys.argv],
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "recovery_source_manifest": _recovery_source_manifest(),
        }
    )
    return payload


def _resolved_compatibility(
    *,
    current_setup: VerifiedSetup,
    checkpoint_setup: VerifiedSetup,
    parent_resolved: Mapping[str, Any],
    checkpoint_path: Path,
    child_run: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything(current_setup.config.seed)
    spec = resolve_precision()
    environment = {
        **environment_report(Path(__file__).resolve().parents[3]),
        "precision": "bf16",
        "autocast_dtype": "torch.bfloat16",
        "complete_command_line": [sys.executable, *sys.argv],
    }
    model, initial = build_frozen_model(checkpoint_setup, device="cpu")
    optimizer, optimizer_contract = build_optimizer(model, current_setup.config)
    scheduler = PiecewiseLinearScheduler(optimizer, current_setup.config)
    resolved = _resolved_config(
        current_setup,
        initial_downstream_parameters=initial,
        optimizer_contract=optimizer_contract.to_dict(),
        scheduler_contract=scheduler.contract(),
        environment=environment,
    )
    resolved["physical_run_directory"] = str(child_run.resolve())
    resolved["resume_control"] = {
        "mode": "cross_directory_exact_continuation",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "next_optimizer_step": EXPECTED_NEXT_STEP,
    }
    validate_resolved_resume_contract(
        parent_resolved,
        resolved,
        approved_source_transition=(
            OLD_TRAINING_SOURCE_SHA256,
            NEW_TRAINING_SOURCE_SHA256,
        ),
    )
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return resolved, environment


def _audit_manifest(audit_dir: Path, bindings: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in audit_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(audit_dir).as_posix(),
    ):
        if path.name == "artifact_manifest.json" or path.name.endswith(".tmp"):
            continue
        rows.append(
            {
                "path": path.relative_to(audit_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "read_only": _is_read_only(path),
            }
        )
    return {
        "schema_version": "fncs-frozen-decoder-recovery-artifact-manifest:1.0",
        "created_utc": utc_now(),
        "bindings": dict(bindings),
        "manifest_self_hash_excluded": True,
        "artifact_count_excluding_manifest": len(rows),
        "artifacts": rows,
    }


def run_audit(
    *,
    config_path: str | Path,
    parent_run: str | Path,
    checkpoint_copy: str | Path,
    audit_directory: str | Path,
    child_run: str | Path,
) -> dict[str, Any]:
    config_source = Path(config_path).resolve()
    parent = Path(parent_run).resolve()
    checkpoint = Path(checkpoint_copy).resolve()
    audit_dir = Path(audit_directory).resolve()
    child = Path(child_run).resolve()
    if parent.name != PARENT_NAME:
        raise TrainingConfigurationError("parent run name changed")
    if not child.name.startswith(CHILD_PREFIX):
        raise TrainingConfigurationError("child run name is outside the recovery lineage")
    if not audit_dir.is_dir():
        raise FileNotFoundError(f"recovery audit directory is missing: {audit_dir}")
    required_outputs = (
        "superseding_authorization.json",
        "parent_inventory.json",
        "step1840_checkpoint_binding.json",
        "resume_compatibility.json",
        "source_difference_report.json",
        "deterministic_restore_comparison.json",
        "recovery_decision.json",
        "data_access_ledger.json",
        "environment.json",
        "artifact_manifest.json",
    )
    existing = [name for name in required_outputs if (audit_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite recovery audit artifacts: {existing}")
    if child.exists():
        raise FileExistsError(f"child run already exists: {child}")
    inventory_before, inventory_sha256 = _path_inventory(parent)
    if inventory_sha256 != EXPECTED_PARENT_INVENTORY_SHA256:
        raise TrainingCompatibilityError("parent inventory changed before recovery")
    if (
        checkpoint == parent / "last.pt"
        or file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.stat().st_size != (parent / "last.pt").stat().st_size
        or not _is_read_only(checkpoint)
    ):
        raise TrainingCompatibilityError("read-only checkpoint copy binding failed")
    parent_runtime = _read_json(parent / "runtime.json", "parent runtime")
    recorded_pid = int(parent_runtime["process_id"])
    if _pid_alive(recorded_pid):
        raise TrainingCompatibilityError("recorded parent writer PID is active")
    parent_resolved = _read_json(parent / "resolved_config.json", "parent resolved config")
    current_setup, checkpoint_setup, old_manifest, current_manifest = _build_setups(
        config_source, parent_resolved
    )
    recovery_source = _recovery_source_manifest()
    bindings = _common_bindings(
        current_setup,
        parent_inventory_sha256=inventory_sha256,
        recovery_source_sha256=recovery_source["source_tree_sha256"],
    )
    source_report = _source_transition_report(
        audit_dir, old_manifest, current_manifest
    )
    source_report.update(
        {
            "schema_version": "fncs-frozen-decoder-source-difference:1.0",
            "created_utc": utc_now(),
            "status": "approved_resume_validation_only_transition",
            "bindings": bindings,
            "recovery_source_manifest": recovery_source,
        }
    )
    resolved, environment = _resolved_compatibility(
        current_setup=current_setup,
        checkpoint_setup=checkpoint_setup,
        parent_resolved=parent_resolved,
        checkpoint_path=checkpoint,
        child_run=child,
    )
    last_report = _strict_checkpoint_report(
        checkpoint, checkpoint_setup, label="step1840_copy"
    )
    best_report = _strict_checkpoint_report(
        parent / "best.pt", checkpoint_setup, label="parent_best"
    )
    metrics = _metrics_report(parent)
    access = _parent_access_report(parent)
    expected_last_state = {
        "completed_epochs": 4,
        "optimizer_step": 1840,
        "current_epoch": 5,
        "next_batch_index": 0,
        "first_optimizer_step_completed": True,
    }
    expected_best_state = {
        "completed_epochs": 3,
        "optimizer_step": 1380,
        "current_epoch": 4,
        "next_batch_index": 0,
        "first_optimizer_step_completed": True,
    }
    checkpoint_gates = {
        "raw_sha256_exact": last_report["raw_sha256"]
        == EXPECTED_CHECKPOINT_SHA256,
        "strict_deserialization_and_application_reload": last_report[
            "strict_application_load"
        ],
        "training_state_exact_step1840": last_report["training_state"]
        == expected_last_state,
        "next_update_is_1841": last_report["scheduler_upcoming_update"] == 1841,
        "downstream_hash_exact": last_report["downstream_state_sha256"]
        == last_report["embedded_downstream_state_sha256"],
        "optimizer_moments_complete": last_report["optimizer"][
            "state_entry_count"
        ]
        == 135
        and last_report["optimizer"][
            "parameters_with_first_and_second_moments"
        ]
        == 135
        and last_report["optimizer"]["recorded_step_values"] == [1840],
        "scheduler_exact_step1840": last_report["scheduler_step_index"] == 1840,
        "rng_complete": last_report["rng_state_complete"],
        "sampler_cursor_present": last_report["sampler_position"]
        == {"epoch": 5, "next_batch_index": 0, "derivation": "seed_epoch_batch_and_sorted_split_ids"},
        "finite_downstream_optimizer": last_report[
            "all_downstream_tensors_finite"
        ]
        and last_report["all_optimizer_tensors_finite"],
        "frozen_encoder_exact": last_report["frozen_encoder"][
            "encoder_state_sha256"
        ]
        == ENCODER_STATE_SHA256
        and last_report["frozen_encoder"]["requires_grad_parameter_count"] == 0
        and last_report["frozen_encoder"]["parameters_with_gradient"] == 0,
        "zero_forbidden_access": last_report["data_access_zero_test_attempts"]
        is True
        and last_report["data_access_zero_test_opens"] is True
        and last_report["test_partition_evaluated"] is False,
        "best_checkpoint_raw_hash": best_report["raw_sha256"]
        == BEST_CHECKPOINT_SHA256,
        "best_checkpoint_state": best_report["training_state"]
        == expected_best_state,
        "best_metric_matches_history": best_report["best_validation_route_nll"]
        == metrics["actual_incumbent"]["selection_value"]
        == 11.071055390944947,
        "parent_metrics_finite_contiguous": metrics["all_records_finite"]
        and metrics["durable_optimizer_steps_exactly_1_through_1840"],
        "parent_test_access_zero": access["zero_test_attempts"] is True
        and access["zero_test_opens"] is True
        and access["test_event_count"] == 0,
    }
    if not all(checkpoint_gates.values()):
        raise TrainingCompatibilityError("step-1840 checkpoint audit gate failed")
    authorization = {
        "schema_version": "fncs-frozen-decoder-superseding-authorization:1.0",
        "created_utc": utc_now(),
        "status": "authorized",
        "bindings": bindings,
        "supersedes_only": {
            "halted_audit": str(
                parent.parents[1]
                / "audit"
                / f"{PARENT_NAME}-recovery-20260827T142628Z-halted"
            ),
            "obsolete_requirement": "step460_checkpoint_and_downstream_hash_77d378967d",
            "preserve_halted_audit_unchanged": True,
        },
        "canonical_parent": str(parent),
        "canonical_checkpoint_copy": str(checkpoint),
        "expected_optimizer_step": EXPECTED_STEP,
        "next_optimizer_step": EXPECTED_NEXT_STEP,
        "endpoint_optimizer_step": EXPECTED_ENDPOINT,
        "remaining_updates": EXPECTED_ENDPOINT - EXPECTED_STEP,
        "child_run": str(child),
        "approved_source_transition": {
            "scope": "resume_contract_validation_only",
            "old_source_sha256": OLD_TRAINING_SOURCE_SHA256,
            "new_source_sha256": NEW_TRAINING_SOURCE_SHA256,
            "changed_files": source_report["changed_files"],
        },
    }
    parent_inventory = {
        "schema_version": "fncs-frozen-decoder-parent-inventory:1.0",
        "created_utc": utc_now(),
        "status": "complete_observed_parent_inventory",
        "bindings": bindings,
        "parent_run": str(parent),
        "inventory_hash_contract": (
            "sorted UTF8(relative POSIX path) NUL decimal_size NUL "
            "lowercase_raw_sha256 LF"
        ),
        "file_count": len(inventory_before),
        "total_size_bytes": sum(row["size_bytes"] for row in inventory_before),
        "files": inventory_before,
        "external_logs": [
            {
                "path": str(parent.with_name(parent.name + suffix)),
                "size_bytes": parent.with_name(parent.name + suffix).stat().st_size,
                "sha256": file_sha256(parent.with_name(parent.name + suffix)),
            }
            for suffix in (".stdout.log", ".stderr.log")
        ],
    }
    checkpoint_binding = {
        "schema_version": "fncs-frozen-decoder-step1840-binding:1.0",
        "created_utc": utc_now(),
        "status": "passed",
        "bindings": bindings,
        "process_audit": {
            "recorded_parent_runtime_pid": recorded_pid,
            "recorded_pid_active": False,
            "matching_trainer_or_launcher_active": False,
            "external_elevated_command_line_scan_match_count": 0,
        },
        "copy": {
            "path": str(checkpoint),
            "separate_from_parent": checkpoint != parent / "last.pt",
            "read_only": _is_read_only(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
            "matches_parent": file_sha256(checkpoint)
            == file_sha256(parent / "last.pt"),
        },
        "last_checkpoint": last_report,
        "best_checkpoint": best_report,
        "metrics": metrics,
        "parent_data_access": access,
        "gates": checkpoint_gates,
    }
    compatibility = {
        "schema_version": "fncs-frozen-decoder-resume-compatibility:1.0",
        "created_utc": utc_now(),
        "status": "passed",
        "bindings": bindings,
        "existing_bug": {
            "old_validator_removed_only_top_level_created_utc": True,
            "old_validator_rejected_json_list_vs_runtime_tuple": True,
            "old_validator_rejected_pid_launch_time_and_resume_command": True,
            "old_path_required_checkpoint_inside_physical_run_directory": True,
        },
        "repair": {
            "volatile_fields_allowed": [
                "PID and parent PID",
                "launch timestamp",
                "physical child run directory",
                "launcher metadata",
                "explicit resume control argument",
            ],
            "all_other_fields_compared_by_canonical_json": True,
            "approved_source_transition": [
                OLD_TRAINING_SOURCE_SHA256,
                NEW_TRAINING_SOURCE_SHA256,
            ],
            "focused_tests_passed": 13,
        },
        "resolved_contract_validation_passed": True,
        "current_resolved_contract_sha256": hashlib.sha256(
            canonical_json(resolved).encode("utf-8")
        ).hexdigest(),
        "parent_resolved_config_sha256": file_sha256(
            parent / "resolved_config.json"
        ),
        "numerical_contract_unchanged": True,
    }
    environment_record = {
        "schema_version": "fncs-frozen-decoder-recovery-environment:1.0",
        "created_utc": utc_now(),
        "status": "recorded",
        "bindings": bindings,
        "environment": environment,
        "audit_process": _environment_payload(),
    }
    atomic_write_json(audit_dir / "superseding_authorization.json", authorization)
    atomic_write_json(audit_dir / "parent_inventory.json", parent_inventory)
    atomic_write_json(audit_dir / "step1840_checkpoint_binding.json", checkpoint_binding)
    atomic_write_json(audit_dir / "resume_compatibility.json", compatibility)
    atomic_write_json(audit_dir / "source_difference_report.json", source_report)
    atomic_write_json(audit_dir / "environment.json", environment_record)
    deterministic = _deterministic_restore_comparison(
        current_setup=current_setup,
        checkpoint_setup=checkpoint_setup,
        checkpoint_path=checkpoint,
        access_path=audit_dir / "data_access_ledger.json",
        bindings=bindings,
    )
    atomic_write_json(
        audit_dir / "deterministic_restore_comparison.json", deterministic
    )
    inventory_after, inventory_after_sha256 = _path_inventory(parent)
    parent_unchanged = (
        inventory_after_sha256 == inventory_sha256
        and inventory_after == inventory_before
        and file_sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256
    )
    decision_gates = {
        "no_parent_process_active": True,
        "step1840_checkpoint_gate_passed": all(checkpoint_gates.values()),
        "read_only_copy_hash_stable": file_sha256(checkpoint)
        == EXPECTED_CHECKPOINT_SHA256,
        "parent_inventory_unchanged": parent_unchanged,
        "resume_compatibility_passed": True,
        "source_transition_limited_to_resume_validator": source_report[
            "changed_files"
        ]
        == [
            {
                "path": "training.py",
                "old_sha256": OLD_TRAINING_PY_SHA256,
                "new_sha256": NEW_TRAINING_PY_SHA256,
            }
        ],
        "deterministic_restore_gate_passed": deterministic["status"] == "passed",
        "no_recovery_validation_runs": deterministic["validation_executions"] == 0,
        "forbidden_test_access_zero": deterministic["gates"][
            "zero_forbidden_test_access"
        ],
        "child_path_unused": not child.exists(),
    }
    decision = {
        "schema_version": "fncs-frozen-decoder-recovery-decision:1.0",
        "created_utc": utc_now(),
        "status": "authorized" if all(decision_gates.values()) else "halted",
        "decision": "launch_exact_child_continuation"
        if all(decision_gates.values())
        else "do_not_launch",
        "bindings": bindings,
        "parent_run": str(parent),
        "checkpoint_copy": str(checkpoint),
        "child_run": str(child),
        "parent_inventory_before_sha256": inventory_sha256,
        "parent_inventory_after_sha256": inventory_after_sha256,
        "gates": decision_gates,
        "next_optimizer_step": EXPECTED_NEXT_STEP,
        "endpoint_optimizer_step": EXPECTED_ENDPOINT,
        "remaining_updates": EXPECTED_ENDPOINT - EXPECTED_STEP,
        "incumbent": metrics["actual_incumbent"],
        "validation_policy": "original_epoch_boundaries_only",
        "parent_files_modified": 0,
        "forbidden_test_attempts": 0,
    }
    atomic_write_json(audit_dir / "recovery_decision.json", decision)
    atomic_write_json(
        audit_dir / "artifact_manifest.json", _audit_manifest(audit_dir, bindings)
    )
    if decision["status"] != "authorized":
        raise TrainingCompatibilityError("recovery audit did not authorize launch")
    return decision


def _validate_prelaunch(
    *,
    audit_dir: Path,
    parent: Path,
    checkpoint: Path,
    child: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = _read_json(audit_dir / "recovery_decision.json", "recovery decision")
    manifest = _read_json(audit_dir / "artifact_manifest.json", "audit manifest")
    if (
        decision.get("status") != "authorized"
        or decision.get("decision") != "launch_exact_child_continuation"
        or Path(decision.get("parent_run", "")).resolve() != parent
        or Path(decision.get("checkpoint_copy", "")).resolve() != checkpoint
        or Path(decision.get("child_run", "")).resolve() != child
    ):
        raise TrainingCompatibilityError("recovery decision does not authorize launch")
    for row in manifest.get("artifacts", []):
        path = audit_dir / row["path"]
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise TrainingCompatibilityError(
                f"recovery audit artifact changed before launch: {row['path']}"
            )
    inventory, inventory_sha256 = _path_inventory(parent)
    del inventory
    if (
        inventory_sha256 != EXPECTED_PARENT_INVENTORY_SHA256
        or file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256
        or not _is_read_only(checkpoint)
    ):
        raise TrainingCompatibilityError("parent or checkpoint copy changed prelaunch")
    return decision, dict(decision["bindings"])


def _save_last(
    *,
    run_directory: Path,
    setup: VerifiedSetup,
    model: Any,
    optimizer: Any,
    optimizer_contract: Any,
    scheduler: Any,
    state: TrainingState,
    initial_downstream: Mapping[str, Any],
    best_validation_route_nll: float | None,
    metrics: ParallelTrajectoryMetricAccumulator,
    data_access: Mapping[str, Any],
) -> str:
    save_checkpoint(
        run_directory / "last.pt",
        setup=setup,
        model=model,
        optimizer=optimizer,
        optimizer_contract=optimizer_contract,
        scheduler=scheduler,
        state=state,
        initial_downstream_parameters=initial_downstream,
        best_validation_route_nll=best_validation_route_nll,
        metrics=metrics,
        data_access=data_access,
    )
    _write_latest(run_directory, state, "last.pt")
    return file_sha256(run_directory / "last.pt")


def _allowed_bootstrap_files(child: Path) -> None:
    if not child.is_dir():
        raise FileNotFoundError(f"precreated child directory is missing: {child}")
    allowed = {"stdout.log", "stderr.log"}
    unexpected = sorted(item.name for item in child.iterdir() if item.name not in allowed)
    if unexpected:
        raise FileExistsError(
            f"refusing non-unused child directory; unexpected artifacts: {unexpected}"
        )


def run_continuation(
    *,
    config_path: str | Path,
    parent_run: str | Path,
    checkpoint_copy: str | Path,
    audit_directory: str | Path,
    child_run: str | Path,
) -> dict[str, Any]:
    config_source = Path(config_path).resolve()
    parent = Path(parent_run).resolve()
    checkpoint = Path(checkpoint_copy).resolve()
    audit_dir = Path(audit_directory).resolve()
    child = Path(child_run).resolve()
    if parent.name != PARENT_NAME or not child.name.startswith(CHILD_PREFIX):
        raise TrainingConfigurationError("recovery lineage path changed")
    _allowed_bootstrap_files(child)
    decision, bindings = _validate_prelaunch(
        audit_dir=audit_dir, parent=parent, checkpoint=checkpoint, child=child
    )
    parent_resolved = _read_json(parent / "resolved_config.json", "parent resolved config")
    current_setup, checkpoint_setup, _, _ = _build_setups(
        config_source, parent_resolved
    )
    seed_everything(current_setup.config.seed)
    spec = resolve_precision()
    worker_started = time.perf_counter()
    environment = {
        **environment_report(Path(__file__).resolve().parents[3]),
        "precision": "bf16",
        "autocast_dtype": "torch.bfloat16",
        "complete_command_line": [sys.executable, *sys.argv],
    }
    model, initial_downstream = build_frozen_model(
        checkpoint_setup, device=spec.device
    )
    optimizer, optimizer_contract = build_optimizer(model, current_setup.config)
    scheduler = PiecewiseLinearScheduler(optimizer, current_setup.config)
    epoch_metrics = ParallelTrajectoryMetricAccumulator(current_setup.profile)
    state, best_validation_route_nll = load_checkpoint(
        checkpoint,
        setup=checkpoint_setup,
        model=model,
        optimizer=optimizer,
        optimizer_contract=optimizer_contract,
        scheduler=scheduler,
        initial_downstream_parameters=initial_downstream,
        metrics=epoch_metrics,
        restore_rng=True,
    )
    if state != TrainingState(
        completed_epochs=4,
        optimizer_step=1840,
        current_epoch=5,
        next_batch_index=0,
        first_optimizer_step_completed=True,
    ) or scheduler.step_index != EXPECTED_STEP:
        raise TrainingCompatibilityError("prelaunch restoration is not exact step 1840")
    resolved = _resolved_config(
        current_setup,
        initial_downstream_parameters=initial_downstream,
        optimizer_contract=optimizer_contract.to_dict(),
        scheduler_contract=scheduler.contract(),
        environment=environment,
    )
    resolved["physical_run_directory"] = str(child)
    resolved["resume_control"] = {
        "mode": "cross_directory_exact_continuation",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "next_optimizer_step": EXPECTED_NEXT_STEP,
    }
    validate_resolved_resume_contract(
        parent_resolved,
        resolved,
        approved_source_transition=(
            OLD_TRAINING_SOURCE_SHA256,
            NEW_TRAINING_SOURCE_SHA256,
        ),
    )
    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    atomic_write_json(child / "environment.json", environment)
    atomic_write_json(child / "encoder_binding.json", current_setup.encoder_binding.report)
    atomic_write_json(child / "initial_downstream_parameters.json", initial_downstream)
    atomic_write_json(child / "resolved_config.json", resolved)
    atomic_write_bytes(
        child / "split_manifest.json", current_setup.config.split_manifest.read_bytes()
    )
    atomic_write_bytes(child / "metrics.jsonl", b"")
    atomic_write_json(child / "data_access.json", raw_checkpoint["data_access"])
    shutil.copy2(parent / "best.pt", child / "best.pt")
    shutil.copy2(parent / "best.json", child / "best.json")
    if (
        file_sha256(child / "best.pt") != BEST_CHECKPOINT_SHA256
        or file_sha256(child / "best.pt") != file_sha256(parent / "best.pt")
    ):
        raise TrainingCompatibilityError("carried-forward best checkpoint changed")
    _append_frozen_proof(
        child / "frozen_encoder_proof.json",
        setup=current_setup,
        model=model,
        stage="after_exact_step1840_restore_before_child_training",
        epoch=state.current_epoch,
        optimizer_step=state.optimizer_step,
    )
    recovery_parent = {
        "schema_version": "fncs-frozen-decoder-recovery-parent:1.0",
        "created_utc": utc_now(),
        "bindings": bindings,
        "parent_run": str(parent),
        "parent_checkpoint": str(parent / "last.pt"),
        "verified_read_only_checkpoint_copy": str(checkpoint),
        "parent_state": state.to_dict(),
        "next_optimizer_step": EXPECTED_NEXT_STEP,
        "endpoint_optimizer_step": EXPECTED_ENDPOINT,
        "recovery_audit": str(audit_dir),
        "recovery_decision_sha256": file_sha256(
            audit_dir / "recovery_decision.json"
        ),
        "metrics_policy": (
            "parent records through step 1840 plus child records beginning at 1841; "
            "parent nondurable records after 1840 are excluded"
        ),
        "incumbent_best": decision["incumbent"],
        "incumbent_best_checkpoint_sha256": BEST_CHECKPOINT_SHA256,
    }
    atomic_write_json(child / "recovery_parent.json", recovery_parent)
    atomic_write_json(
        child / "launch.json",
        {
            "schema_version": "fncs-frozen-decoder-recovery-launch:1.0",
            "created_utc": utc_now(),
            "status": "worker_started",
            "bindings": bindings,
            "launcher_pid": os.getppid(),
            "worker_pid": os.getpid(),
            "command_line": [sys.executable, *sys.argv],
            "parent_run": str(parent),
            "child_run": str(child),
            "resume_checkpoint": str(checkpoint),
            "expected_first_update": EXPECTED_NEXT_STEP,
        },
    )
    repository = FNCSParallelTrajectorySessionRepository(
        paths=current_setup.session_paths,
        partitions=current_setup.partition_by_session,
        active_partitions=("train", "validation"),
        test_session_ids=current_setup.split_session_ids["test"],
        profile=current_setup.profile,
        ledger_path=child / "data_access.json",
        split_manifest_sha256=current_setup.config.expected_split_manifest_sha256,
        phase="full_training",
        resume_ledger=True,
    )
    atomic_write_json(
        child / "runtime.json",
        _runtime_payload(
            state=state,
            run_directory=child,
            data_access=repository.report(),
            status="running",
        ),
    )
    write_artifact_manifest(child, status="exact_step1840_restored")
    config = current_setup.config
    train_ids = current_setup.split_session_ids["train"]
    validation_ids = current_setup.split_session_ids["validation"]
    validation_queries: tuple[PlannerQuery, ...] | None = None
    first_recovery_step_saved = False
    try:
        while state.optimizer_step < config.max_optimizer_steps:
            epoch = state.current_epoch
            if epoch > config.max_epochs:
                raise TrainingCompatibilityError(
                    "epoch counter exceeded the 20-epoch contract"
                )
            ordered = training_session_order(train_ids, seed=config.seed, epoch=epoch)
            batches = session_batches(ordered, config.batch_size)
            if len(batches) != config.updates_per_epoch:
                raise TrainingCompatibilityError(
                    "training-loader length changed from 460 updates per epoch"
                )
            if state.next_batch_index > len(batches):
                raise TrainingCompatibilityError("resume sampler position is invalid")
            model.train()
            assert_frozen_encoder(model, current_setup, stage=f"epoch_{epoch}_train_mode")
            for batch_index in range(state.next_batch_index, len(batches)):
                update_started = time.perf_counter()
                session_ids = batches[batch_index]
                sources = {
                    session_id: repository.get(session_id) for session_id in session_ids
                }
                queries = sample_training_batch_queries(
                    sources,
                    session_ids,
                    seed=config.seed,
                    epoch=epoch,
                    session_batch_index=batch_index,
                    queries_per_session=config.queries_per_session,
                )
                expected_queries = config.queries_per_session * len(session_ids)
                if len(queries) != expected_queries:
                    raise TrainingConfigurationError(
                        f"training batch has {len(queries)} queries, expected {expected_queries}"
                    )
                examples = prepare_parallel_trajectory_query_examples(
                    sources,
                    queries,
                    context_length_ticks=config.context_length_ticks,
                )
                if len(examples) != expected_queries:
                    raise TrainingCompatibilityError(
                        "sampled training query lost target supervision"
                    )
                batch_hashes: dict[str, Any] | None = None
                if not first_recovery_step_saved:
                    batch, observation, targets = (
                        collate_parallel_trajectory_query_examples(examples)
                    )
                    batch_hashes = {
                        "batch_session_ids": list(session_ids),
                        "query_identities": _query_payload(queries),
                        "query_identities_sha256": structure_sha256(
                            _query_payload(queries)
                        ),
                        "input_sha256": structure_sha256(
                            {
                                "encoder_batch": _object_payload(batch),
                                "planner_observation": _object_payload(observation),
                            }
                        ),
                        "target_sha256": structure_sha256(
                            {
                                "target_displacements": targets.target_displacements,
                                "current_xy": targets.current_xy,
                                "future_xy": targets.future_xy,
                            }
                        ),
                        "mask_sha256": structure_sha256(
                            {
                                "target_mask": targets.target_mask,
                                "query_mask": targets.query_mask,
                            }
                        ),
                    }
                result = run_training_batch(
                    model=model,
                    examples=examples,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,  # type: ignore[arg-type]
                    spec=spec,
                    metrics=epoch_metrics,
                )
                frozen = assert_frozen_encoder(
                    model,
                    current_setup,
                    stage=f"optimizer_step_{state.optimizer_step + 1}",
                )
                state = replace(
                    state,
                    optimizer_step=state.optimizer_step + 1,
                    next_batch_index=batch_index + 1,
                    first_optimizer_step_completed=True,
                )
                record = {
                    "type": "optimizer_step",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "optimizer_step": state.optimizer_step,
                    "frozen_encoder": True,
                    **result.to_dict(),
                }
                append_jsonl(child / "metrics.jsonl", record)
                if not first_recovery_step_saved:
                    if state.optimizer_step != EXPECTED_NEXT_STEP or batch_index != 0:
                        raise TrainingCompatibilityError(
                            "first child optimizer update was not exact step 1841"
                        )
                    repository.assert_no_test_access()
                    last_sha256 = _save_last(
                        run_directory=child,
                        setup=current_setup,
                        model=model,
                        optimizer=optimizer,
                        optimizer_contract=optimizer_contract,
                        scheduler=scheduler,
                        state=state,
                        initial_downstream=initial_downstream,
                        best_validation_route_nll=best_validation_route_nll,
                        metrics=epoch_metrics,
                        data_access=repository.report(),
                    )
                    _append_frozen_proof(
                        child / "frozen_encoder_proof.json",
                        setup=current_setup,
                        model=model,
                        stage="after_first_recovery_optimizer_step",
                        epoch=epoch,
                        optimizer_step=state.optimizer_step,
                    )
                    first_durable = {
                        "schema_version": "fncs-frozen-decoder-first-recovery-step:1.0",
                        "created_utc": utc_now(),
                        "bindings": bindings,
                        "optimizer_step": state.optimizer_step,
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "training_state": state.to_dict(),
                        "checkpoint": "last.pt",
                        "checkpoint_sha256": last_sha256,
                        "metric_record_sha256": hashlib.sha256(
                            canonical_json(record).encode("utf-8")
                        ).hexdigest(),
                        "batch": batch_hashes,
                        "loss_and_schedule": result.to_dict(),
                        "scheduler_step_index": scheduler.step_index,
                        "next_scheduler_update": scheduler.upcoming_update,
                        "optimizer_state_sha256": structure_sha256(
                            optimizer.state_dict()
                        ),
                        "frozen_encoder": frozen,
                        "data_access": {
                            "zero_test_attempts": repository.report()[
                                "zero_test_attempts"
                            ],
                            "zero_test_opens": repository.report()["zero_test_opens"],
                            "test_session_request_attempt_count": repository.report()[
                                "test_session_request_attempt_count"
                            ],
                            "test_session_parquet_open_attempt_count": repository.report()[
                                "test_session_parquet_open_attempt_count"
                            ],
                            "test_session_parquet_open_count": repository.report()[
                                "test_session_parquet_open_count"
                            ],
                        },
                        "update_elapsed_seconds": time.perf_counter() - update_started,
                        "worker_elapsed_seconds": time.perf_counter() - worker_started,
                    }
                    atomic_write_json(child / "first_durable_step.json", first_durable)
                    first_recovery_step_saved = True
                    write_artifact_manifest(
                        child, status="first_recovery_optimizer_step_durable"
                    )
                repository.assert_no_test_access()
                atomic_write_json(
                    child / "runtime.json",
                    _runtime_payload(
                        state=state,
                        run_directory=child,
                        data_access=repository.report(),
                        status="running",
                    ),
                )
                print(
                    json.dumps(
                        {
                            "event": "optimizer_step",
                            "epoch": epoch,
                            "optimizer_step": state.optimizer_step,
                            "route_nll": result.route_nll,
                            "encoder_state_sha256": ENCODER_STATE_SHA256,
                            "test_attempts": 0,
                            "test_opens": 0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if state.optimizer_step >= config.max_optimizer_steps:
                    break
            if state.next_batch_index != len(batches):
                continue
            train_report = epoch_metrics.metrics()
            append_jsonl(
                child / "metrics.jsonl",
                {
                    "type": "training_epoch",
                    "completed_epoch": epoch,
                    "optimizer_step": state.optimizer_step,
                    "metrics": train_report,
                },
            )
            if epoch % config.validation_every_epochs != 0:
                raise TrainingCompatibilityError(
                    "validation frequency diverged from every epoch"
                )
            validation_started = time.perf_counter()
            validation_metrics, validation_queries = evaluate_validation(
                model=model,
                repository=repository,  # type: ignore[arg-type]
                validation_session_ids=validation_ids,
                config=config,  # type: ignore[arg-type]
                spec=spec,
                validation_queries=validation_queries,
            )
            validation_seconds = time.perf_counter() - validation_started
            selection_value = float(validation_metrics["route_nll"])
            if not math.isfinite(selection_value):
                raise TrainingCompatibilityError(
                    "validation route mixture NLL is nonfinite"
                )
            improved = (
                best_validation_route_nll is None
                or selection_value < best_validation_route_nll
            )
            if improved:
                best_validation_route_nll = selection_value
            state = replace(
                state,
                completed_epochs=epoch,
                current_epoch=epoch + 1,
                next_batch_index=0,
            )
            _append_frozen_proof(
                child / "frozen_encoder_proof.json",
                setup=current_setup,
                model=model,
                stage="after_epoch",
                epoch=epoch,
                optimizer_step=state.optimizer_step,
            )
            append_jsonl(
                child / "metrics.jsonl",
                {
                    "type": "validation_epoch",
                    "completed_epoch": epoch,
                    "optimizer_step": state.optimizer_step,
                    "metrics": validation_metrics,
                    "selection_metric": config.checkpoint_selection_metric,
                    "selection_value": selection_value,
                    "improved": improved,
                    "validation_seconds": validation_seconds,
                    "query_count": len(validation_queries),
                    "test_information_used": False,
                },
            )
            epoch_metrics = ParallelTrajectoryMetricAccumulator(current_setup.profile)
            if improved:
                save_checkpoint(
                    child / "best.pt",
                    setup=current_setup,
                    model=model,
                    optimizer=optimizer,
                    optimizer_contract=optimizer_contract,
                    scheduler=scheduler,
                    state=state,
                    initial_downstream_parameters=initial_downstream,
                    best_validation_route_nll=best_validation_route_nll,
                    metrics=epoch_metrics,
                    data_access=repository.report(),
                )
                atomic_write_json(
                    child / "best.json",
                    {
                        "checkpoint": "best.pt",
                        "checkpoint_sha256": file_sha256(child / "best.pt"),
                        "completed_epoch": epoch,
                        "optimizer_step": state.optimizer_step,
                        "selection_metric": config.checkpoint_selection_metric,
                        "value": selection_value,
                        "oracle_metrics_used_for_selection": False,
                        "test_information_used_for_selection": False,
                    },
                )
            _save_last(
                run_directory=child,
                setup=current_setup,
                model=model,
                optimizer=optimizer,
                optimizer_contract=optimizer_contract,
                scheduler=scheduler,
                state=state,
                initial_downstream=initial_downstream,
                best_validation_route_nll=best_validation_route_nll,
                metrics=epoch_metrics,
                data_access=repository.report(),
            )
            repository.assert_no_test_access()
            atomic_write_json(
                child / "runtime.json",
                _runtime_payload(
                    state=state,
                    run_directory=child,
                    data_access=repository.report(),
                    status="running",
                ),
            )
            write_artifact_manifest(child, status=f"completed_epoch_{epoch}")
            model.train()
            assert_frozen_encoder(
                model, current_setup, stage=f"after_epoch_{epoch}_return_to_train"
            )
        _append_frozen_proof(
            child / "frozen_encoder_proof.json",
            setup=current_setup,
            model=model,
            stage="after_training",
            epoch=state.completed_epochs,
            optimizer_step=state.optimizer_step,
        )
        repository.assert_no_test_access()
        final_access = repository.report()
        summary = {
            "schema_version": "fncs-frozen-decoder-final-summary:1.0",
            "created_utc": utc_now(),
            "status": "completed",
            "checkpoint_schema_version": FROZEN_FNCS_CHECKPOINT_SCHEMA,
            "run_directory": str(child),
            "optimizer_steps": state.optimizer_step,
            "completed_epochs": state.completed_epochs,
            "best_validation_route_nll": best_validation_route_nll,
            "selection_metric": config.checkpoint_selection_metric,
            "canonical_best_pt_sha256": CANONICAL_ENCODER_SHA256,
            "execution_encoder_only_best_pt_sha256": EXECUTION_ENCODER_SHA256,
            "encoder_state_sha256": ENCODER_STATE_SHA256,
            "encoder_unchanged": True,
            "data_access": final_access,
            "test_partition_evaluated": False,
            "scientific_caveat": config.scientific_caveat,
        }
        atomic_write_json(child / "final_summary.json", summary)
        atomic_write_json(
            child / "runtime.json",
            _runtime_payload(
                state=state,
                run_directory=child,
                data_access=final_access,
                status="completed",
            ),
        )
        write_artifact_manifest(child, status="completed")
        return summary
    except Exception as exc:
        access = repository.report()
        atomic_write_json(
            child / "failure.json",
            {
                "schema_version": "fncs-frozen-decoder-failure:1.0",
                "created_utc": utc_now(),
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:8000],
                "training_state": state.to_dict(),
                "data_access": access,
            },
        )
        atomic_write_json(
            child / "runtime.json",
            _runtime_payload(
                state=state,
                run_directory=child,
                data_access=access,
                status="failed",
            ),
        )
        write_artifact_manifest(child, status="failed")
        raise


def verify_launch(
    *,
    parent_run: str | Path,
    checkpoint_copy: str | Path,
    audit_directory: str | Path,
    child_run: str | Path,
) -> dict[str, Any]:
    parent = Path(parent_run).resolve()
    checkpoint = Path(checkpoint_copy).resolve()
    audit_dir = Path(audit_directory).resolve()
    child = Path(child_run).resolve()
    first = _read_json(child / "first_durable_step.json", "first durable step")
    launch = _read_json(child / "launch.json", "launch record")
    runtime = _read_json(child / "runtime.json", "child runtime")
    access = _read_json(child / "data_access.json", "child data access")
    metrics = _read_jsonl(child / "metrics.jsonl", "child metrics")
    optimizer_records = [row for row in metrics if row.get("type") == "optimizer_step"]
    first_record = optimizer_records[0] if optimizer_records else None
    last_raw = torch.load(child / "last.pt", map_location="cpu", weights_only=False)
    worker_pid = int(launch["worker_pid"])
    launcher_pid = int(launch["launcher_pid"])
    update_seconds = float(first["update_elapsed_seconds"])
    updates_per_second = 1.0 / update_seconds
    remaining = EXPECTED_ENDPOINT - int(runtime["optimizer_step"])
    gates = {
        "launcher_pid_active": _pid_alive(launcher_pid),
        "worker_pid_active": _pid_alive(worker_pid),
        "first_executed_update_is_1841": first.get("optimizer_step") == 1841
        and first_record is not None
        and first_record.get("optimizer_step") == 1841,
        "child_metrics_begin_at_1841": first_record is not None
        and first_record.get("optimizer_step") == 1841,
        "no_duplicate_1840": all(
            row.get("optimizer_step") != 1840 for row in optimizer_records
        ),
        "durable_checkpoint_at_least_1841": last_raw["training_state"][
            "optimizer_step"
        ]
        >= 1841,
        "scheduler_continued": first["loss_and_schedule"]["scheduler_record"][0][
            "stage"
        ]
        == "linear_decay"
        and first["loss_and_schedule"]["learning_rates_used"]
        == [0.0002699673913043478] * 4,
        "optimizer_moments_restored": first.get("optimizer_state_sha256") is not None,
        "encoder_unchanged": first["frozen_encoder"]["encoder_state_sha256"]
        == ENCODER_STATE_SHA256
        and first["frozen_encoder"]["requires_grad_parameter_count"] == 0,
        "stderr_empty": (child / "stderr.log").stat().st_size == 0,
        "forbidden_test_access_zero": access.get("zero_test_attempts") is True
        and access.get("zero_test_opens") is True
        and access.get("test_session_request_attempt_count") == 0
        and access.get("test_session_parquet_open_attempt_count") == 0
        and access.get("test_session_parquet_open_count") == 0,
        "parent_inventory_unchanged": _path_inventory(parent)[1]
        == EXPECTED_PARENT_INVENTORY_SHA256,
        "checkpoint_copy_hash_stable": file_sha256(checkpoint)
        == EXPECTED_CHECKPOINT_SHA256,
    }
    report = {
        "schema_version": "fncs-frozen-decoder-launch-verification:1.0",
        "created_utc": utc_now(),
        "status": "passed" if all(gates.values()) else "failed",
        "bindings": first["bindings"],
        "parent_run": str(parent),
        "child_run": str(child),
        "launcher_pid": launcher_pid,
        "worker_pid": worker_pid,
        "runtime": runtime,
        "first_durable_step": first,
        "current_last_checkpoint_sha256": file_sha256(child / "last.pt"),
        "current_optimizer_step": runtime["optimizer_step"],
        "remaining_updates": remaining,
        "incumbent": _read_json(parent / "best.json", "parent best metadata"),
        "observed_first_update_seconds": update_seconds,
        "observed_updates_per_second": updates_per_second,
        "eta_seconds_from_first_resumed_update": remaining / updates_per_second,
        "stderr_size_bytes": (child / "stderr.log").stat().st_size,
        "gates": gates,
    }
    atomic_write_json(child / "launch_verification.json", report)
    write_artifact_manifest(child, status="launch_verified")
    decision = _read_json(audit_dir / "recovery_decision.json", "recovery decision")
    decision["launch_milestone"] = {
        "verified_utc": report["created_utc"],
        "status": report["status"],
        "child_launch_verification": str(child / "launch_verification.json"),
        "child_launch_verification_sha256": file_sha256(
            child / "launch_verification.json"
        ),
        "launcher_pid": launcher_pid,
        "worker_pid": worker_pid,
        "first_optimizer_step": 1841,
        "current_optimizer_step": runtime["optimizer_step"],
    }
    atomic_write_json(audit_dir / "recovery_decision.json", decision)
    atomic_write_json(
        audit_dir / "artifact_manifest.json",
        _audit_manifest(audit_dir, decision["bindings"]),
    )
    if report["status"] != "passed":
        raise TrainingCompatibilityError("launch milestone verification failed")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and resume the frozen-encoder decoder at exact step 1840."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "continue", "verify-launch"):
        command = subparsers.add_parser(name)
        command.add_argument("--parent-run", type=Path, required=True)
        command.add_argument("--checkpoint-copy", type=Path, required=True)
        command.add_argument("--audit-directory", type=Path, required=True)
        command.add_argument("--child-run", type=Path, required=True)
        if name != "verify-launch":
            command.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "parent_run": arguments.parent_run,
        "checkpoint_copy": arguments.checkpoint_copy,
        "audit_directory": arguments.audit_directory,
        "child_run": arguments.child_run,
    }
    if arguments.command == "audit":
        result = run_audit(config_path=arguments.config, **common)
    elif arguments.command == "continue":
        result = run_continuation(config_path=arguments.config, **common)
    else:
        result = verify_launch(**common)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


__all__ = [
    "build_parser",
    "main",
    "run_audit",
    "run_continuation",
    "structure_sha256",
    "verify_launch",
]
