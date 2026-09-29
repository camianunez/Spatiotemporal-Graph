from __future__ import annotations

import argparse
import ast
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import torch
from torch import Tensor

from fortnite_encoder.config import EncoderConfig, RotationHeadConfig, RotationLossConfig
from fortnite_encoder.losses import compute_rotation_loss
from fortnite_encoder.rotation import RotationModel
from fortnite_encoder.training import (
    LossMetricAccumulator,
    PrecisionSpec,
    TrainingConfigurationError,
    WindowRequest,
    _SessionRecord,
    _SessionRepository,
    _autocast_context,
    _batch_requests,
    _move_batch,
    _validate_finite_logits_and_loss,
    collate_training_windows,
    resolve_precision,
    validation_window_requests,
)
from fortnite_encoder.world_grid import WorldGridProfile

from .artifacts import environment_report
from .common import (
    atomic_write_json,
    canonical_json,
    file_sha256,
    rendered_json,
    seed_everything,
    sha256_bytes,
    source_tree_manifest,
    tensor_bytes,
    tensor_state_sha256,
    utc_now,
)
from .config import FreshEncoderConfig, load_config
from .data_access import AuditedSessionRepository, TRAINING_READ_SEQUENCE
from .dataset import load_bound_inventory, load_bound_split
from .runtime import CHECKPOINT_SCHEMA_VERSION, _load_resolved, _records


REPORT_DATE = "20260824"
REPORT_SUFFIX = f"-test-evaluation-{REPORT_DATE}"
LOCKED_CHECKPOINT_NAME = "best.pt"
LOCKED_CHECKPOINT_SHA256 = (
    "dc8d2dc2e9bfce5ed9cacd557190dd33ca2d8af5d6225ae54464234d8cc3992a"
)
LOCKED_COMPLETED_EPOCH = 19
LOCKED_OPTIMIZER_STEP = 4370
LOCKED_VALIDATION_FUTURE_POSITION = 7.842962951838928
VALIDATION_ABSOLUTE_TOLERANCE = 1e-9
VALIDATION_RELATIVE_TOLERANCE = 1e-9
EXPECTED_SPLIT_COUNTS = {"train": 920, "validation": 115, "test": 115}
EXPECTED_ACCEPTED_SESSIONS = 1150
HORIZONS = (15, 30, 60)
POSITION_HORIZON_NAMES = tuple(f"future_position_{value}s" for value in HORIZONS)
REQUIRED_ARTIFACTS = frozenset(
    {
        "checkpoint_selection.json",
        "evaluation_contract.json",
        "validation_reproduction.json",
        "test_evaluation.json",
        "generalization_gap.json",
        "data_access_ledger.json",
        "environment.json",
        "evaluator_source_manifest.json",
        "final_evaluation_summary.json",
        "artifact_manifest.json",
    }
)
CHECKPOINT_KEYS = frozenset(
    {
        "amp_scaler_state",
        "checkpoint_schema_version",
        "compatibility",
        "created_utc",
        "data_access",
        "epoch_metric_state",
        "initial_parameters",
        "metrics_jsonl",
        "model_state",
        "optimizer_contract",
        "optimizer_state",
        "precision",
        "resolved_training_config",
        "rng_state",
        "sampler_state",
        "scheduler_state",
        "test_partition_evaluated",
        "training_state",
        "type",
    }
)
UNIQUE_EVALUATION_TABLES = tuple(dict.fromkeys(TRAINING_READ_SEQUENCE))


class EvaluationContractError(RuntimeError):
    """Raised before or during evaluation when a locked condition changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationContractError(message)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvaluationContractError(f"cannot read {label}: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{label} must be a JSON object")
    return value


def _payload_digest(value: Mapping[str, Any]) -> str:
    deterministic = dict(value)
    deterministic.pop("created_utc", None)
    deterministic.pop("payload_sha256", None)
    return sha256_bytes(canonical_json(deterministic).encode("utf-8"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output.setdefault("created_utc", utc_now())
    output["self_seal_contract"] = (
        "SHA-256 of canonical JSON after removing created_utc and payload_sha256"
    )
    output["payload_sha256"] = _payload_digest(output)
    return output


def _verify_seal(value: Mapping[str, Any], label: str) -> None:
    _require(
        value.get("self_seal_contract")
        == "SHA-256 of canonical JSON after removing created_utc and payload_sha256",
        f"{label} self-seal contract changed",
    )
    recorded = value.get("payload_sha256")
    _require(
        isinstance(recorded, str) and recorded == _payload_digest(value),
        f"{label} payload self-hash changed",
    )


def _write_sealed(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    output = _seal(value)
    atomic_write_json(path, output)
    reread = _read_json(path, Path(path).name)
    _verify_seal(reread, Path(path).name)
    _require(reread == output, f"atomic round-trip changed {Path(path).name}")
    return output


def _metric_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output["metric_payload_self_seal_contract"] = (
        "SHA-256 of canonical JSON after removing metric_payload_sha256"
    )
    deterministic = dict(output)
    deterministic.pop("metric_payload_sha256", None)
    output["metric_payload_sha256"] = sha256_bytes(
        canonical_json(deterministic).encode("utf-8")
    )
    return output


def _verify_metric_payload(value: Mapping[str, Any], label: str) -> None:
    expected = dict(value)
    recorded = expected.pop("metric_payload_sha256", None)
    _require(
        recorded == sha256_bytes(canonical_json(expected).encode("utf-8")),
        f"{label} metric payload self-hash changed",
    )


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _manifest_entries_match(
    manifest: Mapping[str, Any], source_root: Path, expected_digest: str, label: str
) -> None:
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and entries, f"{label} entries are missing")
    aggregate = hashlib.sha256()
    seen: set[str] = set()
    for raw in entries:
        _require(isinstance(raw, dict), f"{label} entry is not an object")
        relative = raw.get("path")
        digest = raw.get("sha256")
        _require(
            isinstance(relative, str)
            and isinstance(digest, str)
            and relative not in seen,
            f"{label} entry is malformed",
        )
        seen.add(relative)
        path = source_root / Path(relative)
        _require(path.is_file(), f"{label} source is missing: {relative}")
        actual = file_sha256(path)
        _require(actual == digest, f"{label} source hash changed: {relative}")
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(actual))
    _require(
        manifest.get("file_count") == len(entries), f"{label} file count changed"
    )
    _require(
        manifest.get("source_tree_sha256") == expected_digest,
        f"{label} recorded source-tree digest changed",
    )
    _require(
        aggregate.hexdigest() == expected_digest,
        f"{label} recomputed source-tree digest changed",
    )


def _verify_checkpoint_selection(
    output_directory: Path, run_directory: Path, config_path: Path
) -> dict[str, Any]:
    path = output_directory / "checkpoint_selection.json"
    _require(path.is_file(), "checkpoint_selection.json must be published first")
    selection = _read_json(path, "checkpoint selection lock")
    _verify_seal(selection, "checkpoint selection lock")
    _require(
        selection.get("schema_version")
        == "fncs-encoder-checkpoint-selection-lock:1.0",
        "checkpoint selection schema changed",
    )
    selected = selection.get("selection")
    bindings = selection.get("bindings")
    _require(isinstance(selected, dict), "checkpoint selection record is missing")
    _require(isinstance(bindings, dict), "checkpoint selection bindings are missing")
    expected_selection = {
        "checkpoint": (
            "runs/encoder-fncs-pilot-20260817-fresh-run-3/best.pt"
        ),
        "checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
        "completed_epoch": LOCKED_COMPLETED_EPOCH,
        "optimizer_step": LOCKED_OPTIMIZER_STEP,
        "metric": "future_position",
        "direction": "minimize",
        "validation_value": LOCKED_VALIDATION_FUTURE_POSITION,
        "selected_using_validation_only": True,
        "test_data_used_for_selection": False,
    }
    _require(selected == expected_selection, "locked checkpoint selection changed")
    checks = {
        "canonical_config_raw_sha256": config_path,
        "resolved_training_config_raw_sha256": (
            run_directory / "resolved_training_config.json"
        ),
        "split_manifest_raw_sha256": run_directory / "split_manifest.json",
        "dataset_inventory_raw_sha256": run_directory / "dataset_inventory.json",
        "world_grid_compatibility_audit_raw_sha256": (
            run_directory / "world_grid_compatibility_audit.json"
        ),
        "protected_encoder_source_manifest_raw_sha256": (
            run_directory / "encoder_source_manifest.json"
        ),
        "training_orchestration_source_manifest_raw_sha256": (
            run_directory / "training_orchestration_source_manifest.json"
        ),
        "initial_parameters_raw_sha256": run_directory / "initial_parameters.json",
        "final_training_summary_raw_sha256": run_directory / "final_summary.json",
    }
    for name, source in checks.items():
        _require(source.is_file(), f"checkpoint binding source is missing: {source}")
        _require(
            bindings.get(name) == file_sha256(source),
            f"checkpoint selection binding changed: {name}",
        )
    canonical_path = run_directory / "canonical_config.json"
    _require(
        bindings.get("canonical_config_raw_sha256") == file_sha256(canonical_path),
        "persisted canonical configuration differs from launch configuration",
    )
    _require(
        file_sha256(run_directory / LOCKED_CHECKPOINT_NAME) == LOCKED_CHECKPOINT_SHA256,
        "locked best.pt SHA-256 changed",
    )
    return selection


def _verify_static_bindings(
    *,
    config: FreshEncoderConfig,
    config_path: Path,
    run_directory: Path,
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], WorldGridProfile]:
    bindings = selection["bindings"]
    resolved = _load_resolved(run_directory / "resolved_training_config.json")
    _require(
        resolved.get("resolved_training_config_sha256")
        == bindings["resolved_training_config_sha256"],
        "resolved configuration canonical hash changed",
    )
    _require(
        file_sha256(config_path) == bindings["canonical_config_raw_sha256"],
        "launch configuration raw hash changed",
    )
    inventory = load_bound_inventory(
        run_directory / "dataset_inventory.json",
        bindings["dataset_validation_sha256"],
    )
    split = load_bound_split(
        run_directory / "split_manifest.json", bindings["split_manifest_sha256"]
    )
    _require(split.get("counts") == EXPECTED_SPLIT_COUNTS, "split counts changed")
    _require(
        inventory.get("accepted_canonical_sessions") == EXPECTED_ACCEPTED_SESSIONS,
        "accepted session census changed",
    )
    sessions = inventory.get("sessions")
    _require(
        isinstance(sessions, list) and len(sessions) == EXPECTED_ACCEPTED_SESSIONS,
        "dataset inventory session census changed",
    )
    _require(
        all(item.get("data_status") == "accepted_with_warnings" for item in sessions),
        "scientific caveat changed: not every accepted session has warnings",
    )
    _require(
        all(
            item.get("trust_partition") == "unattested"
            and item.get("provenance_status") == "incomplete"
            for item in sessions
        ),
        "scientific caveat changed: accepted sessions are no longer all unattested",
    )
    raw_sources = {
        "ingestion_report_sha256": Path(config.dataset_root) / "ingestion-report.json",
        "source_dataset_validation_raw_sha256": Path(config.dataset_validation_path),
        "world_grid_profile_raw_sha256": Path(config.world_grid_profile),
        "world_grid_publication_audit_raw_sha256": Path(
            config.world_grid_publication_audit
        ),
    }
    for name, source in raw_sources.items():
        _require(source.is_file(), f"bound source is missing: {source}")
        _require(
            bindings.get(name) == file_sha256(source), f"raw binding changed: {name}"
        )
    _require(
        inventory.get("ingestion_report_sha256") == bindings["ingestion_report_sha256"],
        "ingestion report binding changed",
    )
    _require(
        inventory.get("source_dataset_validation_raw_sha256")
        == bindings["source_dataset_validation_raw_sha256"],
        "dataset-validation raw binding changed",
    )
    _require(
        inventory.get("dataset_validation_sha256")
        == bindings["dataset_validation_sha256"],
        "dataset-validation payload binding changed",
    )
    world_audit = _read_json(
        run_directory / "world_grid_compatibility_audit.json",
        "world-grid compatibility audit",
    )
    deterministic_world = dict(world_audit)
    deterministic_world.pop("created_utc", None)
    recorded_world_hash = deterministic_world.pop(
        "world_grid_validation_audit_sha256", None
    )
    actual_world_hash = sha256_bytes(
        canonical_json(deterministic_world).encode("utf-8")
    )
    _require(
        world_audit.get("status") == "compatible"
        and recorded_world_hash == bindings["world_grid_validation_audit_sha256"]
        and actual_world_hash == recorded_world_hash,
        "world-grid compatibility audit changed",
    )
    world_bindings = {
        "profile_raw_sha256": "world_grid_profile_raw_sha256",
        "profile_hash": "world_grid_profile_sha256",
        "publication_audit_raw_sha256": "world_grid_publication_audit_raw_sha256",
        "publication_audit_sha256": "world_grid_publication_audit_sha256",
        "coordinate_audit_sha256": "world_grid_coordinate_audit_sha256",
    }
    for audit_name, selection_name in world_bindings.items():
        _require(
            world_audit.get(audit_name) == bindings.get(selection_name),
            f"world-grid binding changed: {audit_name}",
        )
    out_of_bounds_path = run_directory / "world_grid_out_of_bounds.jsonl"
    _require(
        file_sha256(out_of_bounds_path)
        == world_audit.get("out_of_bounds_coordinates_raw_sha256"),
        "world-grid out-of-bounds audit changed",
    )
    profile = WorldGridProfile.load(
        config.world_grid_profile,
        expected_hash=bindings["world_grid_profile_sha256"],
    )

    encoder_manifest_path = run_directory / "encoder_source_manifest.json"
    encoder_manifest = _read_json(encoder_manifest_path, "protected encoder manifest")
    _manifest_entries_match(
        encoder_manifest,
        Path(resolved["workspace"]) / "ml" / "src" / "fortnite_encoder",
        bindings["protected_encoder_source_sha256"],
        "protected encoder",
    )
    current_encoder = source_tree_manifest(
        Path(resolved["workspace"]) / "ml" / "src" / "fortnite_encoder"
    )
    _require(
        current_encoder["source_tree_sha256"]
        == bindings["protected_encoder_source_sha256"],
        "protected encoder source tree changed",
    )
    training_manifest = _read_json(
        run_directory / "training_orchestration_source_manifest.json",
        "training orchestration source manifest",
    )
    _manifest_entries_match(
        training_manifest,
        Path(resolved["workspace"]) / "ml" / "src" / "fncs_encoder_training",
        bindings["training_orchestration_source_sha256"],
        "training orchestration",
    )
    initial = _read_json(run_directory / "initial_parameters.json", "initial parameters")
    _require(
        initial.get("complete_initial_parameter_sha256")
        == bindings["initial_parameter_sha256"],
        "initial-parameter hash changed",
    )
    final_summary = _read_json(
        run_directory / "final_summary.json", "final training summary"
    )
    _require(final_summary.get("status") == "completed", "training is not completed")
    _require(
        final_summary.get("best_completed_epoch") == LOCKED_COMPLETED_EPOCH
        and final_summary.get("best_optimizer_step") == LOCKED_OPTIMIZER_STEP
        and final_summary.get("best_metric") == LOCKED_VALIDATION_FUTURE_POSITION,
        "final training checkpoint selection changed",
    )
    _require(
        final_summary.get("test_file_attempts") == 0
        and final_summary.get("test_file_opens") == 0
        and final_summary.get("test_partition_evaluated") is False,
        "training artifacts report prior test access",
    )
    training_ledger = _read_json(
        run_directory / "data_access_ledger.json", "training data-access ledger"
    )
    _require(
        training_ledger.get("test_session_parquet_attempt_count") == 0
        and training_ledger.get("test_session_parquet_open_count") == 0,
        "training ledger reports prior test access",
    )
    return resolved, inventory, split, profile


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _static_safety_audit(source_path: Path) -> dict[str, Any]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    forbidden_calls: list[str] = []
    training_calls: list[str] = []
    optimizer_calls: list[str] = []
    scheduler_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name.endswith(".backward") or name == "torch.autograd.backward":
            forbidden_calls.append(name)
        if name.endswith("run_training") or name.endswith("run_rotation_training"):
            training_calls.append(name)
        final = name.rsplit(".", 1)[-1]
        if final in {
            "Adam",
            "AdamW",
            "SGD",
            "Optimizer",
            "build_adamw_optimizer",
        }:
            optimizer_calls.append(name)
        if final in {
            "PiecewiseLinearScheduler",
            "WarmupCosineScheduler",
            "LRScheduler",
        }:
            scheduler_calls.append(name)
    _require(not forbidden_calls, "evaluator contains a backward call")
    _require(not training_calls, "evaluator contains a training entry-point call")
    _require(not optimizer_calls, "evaluator constructs an optimizer")
    _require(not scheduler_calls, "evaluator constructs a scheduler")
    return {
        "ast_parsed": True,
        "backward_call_count": len(forbidden_calls),
        "training_entry_point_call_count": len(training_calls),
        "optimizer_construction_call_count": len(optimizer_calls),
        "scheduler_construction_call_count": len(scheduler_calls),
        "passed": True,
    }


def _evaluator_source_manifest(workspace: Path) -> dict[str, Any]:
    evaluator = Path(__file__).resolve()
    focused_test = workspace / "ml" / "tests" / "test_fncs_encoder_evaluate.py"
    _require(focused_test.is_file(), "focused evaluator test source is missing")
    entry = {
        "path": _relative(evaluator, workspace),
        "raw_sha256": file_sha256(evaluator),
        "size_bytes": evaluator.stat().st_size,
    }
    test_entry = {
        "path": _relative(focused_test, workspace),
        "raw_sha256": file_sha256(focused_test),
        "size_bytes": focused_test.stat().st_size,
    }
    digest = hashlib.sha256()
    digest.update(entry["path"].encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(entry["raw_sha256"]))
    return {
        "schema_version": "fncs-encoder-evaluator-source-manifest:1.0",
        "evaluator_entry_point": "python -m fncs_encoder_training.evaluate",
        "hash_contract": "UTF8(relative POSIX path) || NUL || raw SHA-256 bytes",
        "evaluator_sources": [entry],
        "qualification_test_sources": [test_entry],
        "evaluator_source_sha256": digest.hexdigest(),
        "static_safety_audit": _static_safety_audit(evaluator),
    }


class EvaluationAccessLedger:
    """Durable split-aware access state for the one-shot evaluator."""

    SCHEMA = "fncs-encoder-final-evaluation-data-access:1.0"

    def __init__(
        self,
        *,
        path: Path,
        partition_by_session: Mapping[str, str],
        test_session_ids: Sequence[str],
        split_manifest_sha256: str,
    ) -> None:
        if path.exists():
            prior = _read_json(path, "existing evaluation data-access ledger")
            if int(prior.get("test_evaluation_passes_started", 0)) > 0:
                raise EvaluationContractError(
                    "a test pass was already started; automatic retry is forbidden"
                )
            raise EvaluationContractError(
                "an incomplete evaluation ledger already exists; explicit recovery is required"
            )
        self.path = path
        self.partition_by_session = dict(partition_by_session)
        self.test_session_ids = tuple(test_session_ids)
        self.test_session_set = frozenset(test_session_ids)
        self.split_manifest_sha256 = split_manifest_sha256
        self.created_utc = utc_now()
        self.phase = "qualification"
        self.allowed_partitions: tuple[str, ...] = ()
        self.validation_passes_started = 0
        self.validation_passes_completed = 0
        self.test_evaluation_passes_started = 0
        self.test_evaluation_passes_completed = 0
        self.test_authorized = False
        self.authorization_contract_payload_sha256: str | None = None
        self.authorization_contract_raw_sha256: str | None = None
        self.test_partition_unsealed = False
        self.status = "qualification"
        self.file_attempt_counts: Counter[str] = Counter()
        self.file_open_counts: Counter[str] = Counter()
        self.operation_counts: Counter[str] = Counter()
        self.attempted_session_order: list[str] = []
        self.opened_session_order: list[str] = []
        self.attempted_session_ids: set[str] = set()
        self.opened_session_ids: set[str] = set()
        self.derived_artifact_reads: list[dict[str, Any]] = []
        self.persist()

    def _partition(self, session_id: str) -> str:
        partition = self.partition_by_session.get(session_id)
        if partition not in {"train", "validation", "test"}:
            raise EvaluationContractError(
                f"data access attempted for unbound session: {session_id}"
            )
        return partition

    def _check_access(self, session_id: str) -> str:
        partition = self._partition(session_id)
        if partition not in self.allowed_partitions:
            raise EvaluationContractError(
                f"partition {partition} is not active during {self.phase}"
            )
        if partition == "test":
            if (
                not self.test_authorized
                or self.phase != "one_shot_test_evaluation"
                or self.test_evaluation_passes_started != 1
                or self.test_evaluation_passes_completed != 0
                or session_id not in self.test_session_set
            ):
                raise EvaluationContractError(
                    f"unauthorized one-shot test access rejected: {session_id}"
                )
        return partition

    def attempt(self, session_id: str, table_name: str, operation: str) -> None:
        partition = self._check_access(session_id)
        self.file_attempt_counts[f"{partition}:{table_name}"] += 1
        self.operation_counts[f"{self.phase}:{operation}:attempt"] += 1
        if session_id not in self.attempted_session_ids:
            self.attempted_session_ids.add(session_id)
            self.attempted_session_order.append(session_id)
        if partition == "test":
            # Conservatively unseal before the authorized I/O call. This ensures
            # a mid-read exception cannot leave the ledger claiming the split is sealed.
            self.test_partition_unsealed = True
        if partition == "test":
            self.persist()

    def opened(self, session_id: str, table_name: str, operation: str) -> None:
        partition = self._check_access(session_id)
        self.file_open_counts[f"{partition}:{table_name}"] += 1
        self.operation_counts[f"{self.phase}:{operation}:open"] += 1
        if session_id not in self.opened_session_ids:
            self.opened_session_ids.add(session_id)
            self.opened_session_order.append(session_id)
        if partition == "test":
            self.test_partition_unsealed = True
        if partition == "test":
            self.persist()

    def start_validation_pass(self, number: int) -> None:
        _require(
            number == self.validation_passes_started + 1 and number in {1, 2},
            "validation pass order changed",
        )
        _require(not self.test_authorized, "validation cannot run after test authorization")
        self.validation_passes_started = number
        self.phase = f"validation_reproduction_pass_{number}"
        self.allowed_partitions = ("validation",)
        self.status = "validation_reproduction"
        self.persist()

    def complete_validation_pass(self, number: int) -> None:
        _require(
            self.validation_passes_started == number
            and self.validation_passes_completed == number - 1,
            "validation completion order changed",
        )
        self.validation_passes_completed = number
        self.persist()

    def authorize_test(self, contract_path: Path, contract: Mapping[str, Any]) -> None:
        _require(
            self.validation_passes_started == 2
            and self.validation_passes_completed == 2,
            "test authorization requires exactly two completed validation passes",
        )
        _require(self.test_attempts == 0 and self.test_opens == 0, "test was accessed early")
        _require(contract.get("authorization", {}).get("authorized") is True, "contract is not authorized")
        _require(
            tuple(contract.get("authorization", {}).get("ordered_test_session_ids", ()))
            == self.test_session_ids,
            "contract test authorization list differs from split manifest",
        )
        self.test_authorized = True
        self.authorization_contract_payload_sha256 = str(contract["payload_sha256"])
        self.authorization_contract_raw_sha256 = file_sha256(contract_path)
        self.phase = "authorized_waiting_for_test"
        self.allowed_partitions = ()
        self.status = "authorized"
        self.persist()

    def start_test_pass(self) -> None:
        _require(self.test_authorized, "test pass is not authorized")
        _require(
            self.test_evaluation_passes_started == 0
            and self.test_evaluation_passes_completed == 0
            and self.test_attempts == 0
            and self.test_opens == 0,
            "one-shot test evaluation cannot be retried",
        )
        self.test_evaluation_passes_started = 1
        self.phase = "one_shot_test_evaluation"
        self.allowed_partitions = ("test",)
        self.status = "test_running"
        self.persist()

    def complete_test_pass(self) -> None:
        _require(
            self.test_evaluation_passes_started == 1
            and self.test_evaluation_passes_completed == 0,
            "test pass completion state changed",
        )
        _require(self.test_opens > 0, "test pass completed without opening test data")
        self.test_evaluation_passes_completed = 1
        self.phase = "completed"
        self.allowed_partitions = ()
        self.status = "completed"
        self.test_partition_unsealed = True
        self.persist()

    def record_derived_artifact_read(self, path: Path, purpose: str) -> None:
        _require(
            self.phase == "one_shot_test_evaluation",
            "test-derived artifact read occurred outside the authorized pass",
        )
        self.derived_artifact_reads.append(
            {
                "path": path.name,
                "purpose": purpose,
                "raw_sha256": file_sha256(path),
            }
        )
        self.persist()

    def fail_test(self, reason: str) -> None:
        _require(
            self.test_evaluation_passes_started == 1,
            "cannot record a test failure before a test pass starts",
        )
        self.phase = "failed_after_test_access"
        self.allowed_partitions = ()
        self.status = "failed_requires_explicit_recovery_decision"
        if self.test_attempts > 0 or self.test_opens > 0:
            self.test_partition_unsealed = True
        self.operation_counts[f"failure:{reason}"] += 1
        self.persist()

    @property
    def test_attempts(self) -> int:
        return sum(
            count
            for name, count in self.file_attempt_counts.items()
            if name.startswith("test:")
        )

    @property
    def test_opens(self) -> int:
        return sum(
            count
            for name, count in self.file_open_counts.items()
            if name.startswith("test:")
        )

    def report(self) -> dict[str, Any]:
        partition_attempts = {
            partition: sum(
                count
                for name, count in self.file_attempt_counts.items()
                if name.startswith(f"{partition}:")
            )
            for partition in ("train", "validation", "test")
        }
        partition_opens = {
            partition: sum(
                count
                for name, count in self.file_open_counts.items()
                if name.startswith(f"{partition}:")
            )
            for partition in ("train", "validation", "test")
        }
        return {
            "schema_version": self.SCHEMA,
            "created_utc": self.created_utc,
            "updated_utc": utc_now(),
            "status": self.status,
            "phase": self.phase,
            "allowed_partitions": list(self.allowed_partitions),
            "split_manifest_sha256": self.split_manifest_sha256,
            "ordered_authorized_test_session_ids": list(self.test_session_ids),
            "test_authorized": self.test_authorized,
            "authorization_contract_payload_sha256": self.authorization_contract_payload_sha256,
            "authorization_contract_raw_sha256": self.authorization_contract_raw_sha256,
            "validation_passes_started": self.validation_passes_started,
            "validation_passes_completed": self.validation_passes_completed,
            "test_evaluation_passes_started": self.test_evaluation_passes_started,
            "test_evaluation_passes_completed": self.test_evaluation_passes_completed,
            "automatic_test_retry_permitted": False,
            "test_partition_unsealed": self.test_partition_unsealed,
            "partition_attempt_counts": partition_attempts,
            "partition_open_counts": partition_opens,
            "test_session_parquet_attempt_count": self.test_attempts,
            "test_session_parquet_open_count": self.test_opens,
            "file_attempt_counts": dict(sorted(self.file_attempt_counts.items())),
            "file_open_counts": dict(sorted(self.file_open_counts.items())),
            "operation_counts": dict(sorted(self.operation_counts.items())),
            "unique_attempted_session_ids_in_first_attempt_order": list(
                self.attempted_session_order
            ),
            "unique_opened_session_ids_in_first_open_order": list(
                self.opened_session_order
            ),
            "derived_artifact_reads": list(self.derived_artifact_reads),
        }

    def persist(self) -> None:
        atomic_write_json(self.path, _seal(self.report()))


class EvaluationSessionRepository(AuditedSessionRepository):
    """Training-identical repository with evaluation authorization and hash checks."""

    def __init__(
        self,
        records: Sequence[_SessionRecord],
        profile: WorldGridProfile,
        head_config: RotationHeadConfig,
        ledger: EvaluationAccessLedger,
        bound_tables: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> None:
        super().__init__(records, profile, head_config, 0, False, ledger)  # type: ignore[arg-type]
        self.evaluation_ledger = ledger
        self.bound_tables = dict(bound_tables)
        self.verified_files: set[tuple[str, str]] = set()

    def verify_bound_file(self, session_id: str, table_name: str) -> None:
        key = (session_id, table_name)
        if key in self.verified_files:
            return
        binding = self.bound_tables.get(key)
        _require(binding is not None, f"inventory table binding is missing: {key}")
        record = self.records.get(session_id)
        _require(record is not None, f"repository session is unbound: {session_id}")
        path = record.path / table_name
        expected_relative = Path(str(binding["path"]))
        _require(
            path.resolve() == (record.path.parents[1] / expected_relative).resolve(),
            f"inventory table path changed: {session_id}/{table_name}",
        )
        self.evaluation_ledger.attempt(session_id, table_name, "sha256_verification")
        actual_size = path.stat().st_size
        actual_hash = file_sha256(path)
        self.evaluation_ledger.opened(session_id, table_name, "sha256_verification")
        _require(
            actual_size == int(binding["byte_size"]),
            f"table size changed: {session_id}/{table_name}",
        )
        _require(
            actual_hash == binding["sha256"],
            f"table SHA-256 changed: {session_id}/{table_name}",
        )
        self.verified_files.add(key)

    def _full(self, session_id: str):  # type: ignore[no-untyped-def]
        if session_id not in self.cache:
            for table_name in UNIQUE_EVALUATION_TABLES:
                self.verify_bound_file(session_id, table_name)
        return super()._full(session_id)

    def windows(self, requests: Sequence[WindowRequest]):  # type: ignore[no-untyped-def]
        for request in requests:
            partition = self.evaluation_ledger._partition(request.session_id)
            _require(
                partition in self.evaluation_ledger.allowed_partitions,
                f"partition {partition} is inactive during evaluation",
            )
            if partition == "test":
                _require(
                    request.session_id in self.evaluation_ledger.test_session_set
                    and self.evaluation_ledger.test_authorized,
                    f"test window request is unauthorized: {request.session_id}",
                )
        return _SessionRepository.windows(self, requests)


def _bound_table_map(inventory: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for session in inventory["sessions"]:
        session_id = session["game_session_id"]
        tables = session.get("tables")
        _require(isinstance(tables, list), f"inventory tables missing for {session_id}")
        for table in tables:
            name = Path(str(table["path"])).name
            key = (session_id, name)
            _require(key not in output, f"duplicate inventory table binding: {key}")
            output[key] = dict(table)
    return output


def _audited_session_lengths(
    *,
    records: Sequence[_SessionRecord],
    session_ids: Sequence[str],
    encoder_config: EncoderConfig,
    repository: EvaluationSessionRepository,
    ledger: EvaluationAccessLedger,
) -> dict[str, int]:
    by_id = {record.session_id: record for record in records}
    output: dict[str, int] = {}
    for session_id in sorted(session_ids):
        _require(session_id in by_id, f"session record is missing: {session_id}")
        repository.verify_bound_file(session_id, "match_samples.parquet")
        path = by_id[session_id].path / "match_samples.parquet"
        ledger.attempt(session_id, "match_samples.parquet", "session_length_footer")
        try:
            count = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:
            raise EvaluationContractError(
                f"cannot inspect match table {path}: {exc}"
            ) from exc
        ledger.opened(session_id, "match_samples.parquet", "session_length_footer")
        _require(count > 0, f"session has no sampled ticks: {session_id}")
        _require(
            count <= encoder_config.max_timesteps,
            f"session exceeds max_timesteps: {session_id}",
        )
        output[session_id] = count
    return output


class EvaluationMetrics:
    def __init__(self, loss_config: RotationLossConfig, position_classes: int) -> None:
        self.losses = LossMetricAccumulator(loss_config)
        self.position_classes = position_classes
        self.eliminated_index = position_classes - 1
        self.position_regular_correct = [0, 0, 0]
        self.position_regular_count = [0, 0, 0]
        self.position_eliminated_correct = [0, 0, 0]
        self.position_eliminated_count = [0, 0, 0]
        self.entry_correct = 0
        self.entry_count = 0
        self.survival_correct = 0
        self.survival_count = 0
        self.placement_correct = 0
        self.placement_count = 0
        self.placement_confusion = torch.zeros((5, 5), dtype=torch.int64)
        self.mask_counts: dict[str, dict[str, int]] = {
            **{
                name: {
                    "target_mask_true": 0,
                    "effective_valid": 0,
                    "target_mask_true_outside_query": 0,
                    "invalid_at_valid_query": 0,
                }
                for name in POSITION_HORIZON_NAMES
            },
            **{
                name: {
                    "target_mask_true": 0,
                    "effective_valid": 0,
                    "target_mask_true_outside_query": 0,
                    "invalid_at_valid_query": 0,
                }
                for name in ("zone_entry", "survival", "placement")
            },
        }
        self.query_count = 0
        self.padded_query_slot_count = 0
        self.nonfinite_logits = Counter()
        self.nonfinite_losses = Counter()

    @staticmethod
    def _accuracy(correct: int, count: int) -> dict[str, Any]:
        return {
            "value": correct / count if count else None,
            "correct": correct,
            "count": count,
        }

    def _mask_update(self, name: str, raw_mask: Tensor, query_mask: Tensor) -> Tensor:
        mask = raw_mask.to(device=query_mask.device, dtype=torch.bool)
        effective = mask & query_mask
        counter = self.mask_counts[name]
        counter["target_mask_true"] += int(mask.sum().item())
        counter["effective_valid"] += int(effective.sum().item())
        counter["target_mask_true_outside_query"] += int((mask & ~query_mask).sum().item())
        counter["invalid_at_valid_query"] += int((~mask & query_mask).sum().item())
        return effective

    def update(self, output: Any, targets: Any, loss: Any) -> None:
        self.losses.update(loss)
        logits = output.logits
        query = logits.query_mask.to(dtype=torch.bool)
        self.query_count += int(query.sum().item())
        self.padded_query_slot_count += query.numel() - int(query.sum().item())
        position_predictions = logits.future_position.argmax(dim=-1)
        for index, name in enumerate(POSITION_HORIZON_NAMES):
            effective = self._mask_update(
                name, targets.future_position_mask[..., index], query
            )
            labels = targets.future_position[..., index].to(
                device=query.device, dtype=torch.long
            )
            predictions = position_predictions[..., index]
            eliminated = effective & (labels == self.eliminated_index)
            regular = effective & ~eliminated
            self.position_regular_correct[index] += int(
                ((predictions == labels) & regular).sum().item()
            )
            self.position_regular_count[index] += int(regular.sum().item())
            self.position_eliminated_correct[index] += int(
                ((predictions == labels) & eliminated).sum().item()
            )
            self.position_eliminated_count[index] += int(eliminated.sum().item())

        entry_mask = self._mask_update("zone_entry", targets.zone_entry_mask, query)
        entry_labels = targets.zone_entry.to(device=query.device, dtype=torch.long)
        entry_predictions = logits.zone_entry.argmax(dim=-1)
        self.entry_correct += int(
            ((entry_predictions == entry_labels) & entry_mask).sum().item()
        )
        self.entry_count += int(entry_mask.sum().item())

        _require(logits.survival is not None, "survival logits are unavailable")
        _require(
            targets.survival is not None and targets.survival_mask is not None,
            "survival targets are unavailable",
        )
        survival_mask = self._mask_update("survival", targets.survival_mask, query)
        survival_labels = targets.survival.to(device=query.device)
        survival_predictions = logits.survival >= 0.0
        self.survival_correct += int(
            ((survival_predictions == (survival_labels >= 0.5)) & survival_mask)
            .sum()
            .item()
        )
        self.survival_count += int(survival_mask.sum().item())

        _require(logits.placement is not None, "placement logits are unavailable")
        _require(
            targets.placement is not None and targets.placement_mask is not None,
            "placement targets are unavailable",
        )
        placement_mask = self._mask_update("placement", targets.placement_mask, query)
        placement_labels = targets.placement.to(device=query.device, dtype=torch.long)
        placement_predictions = logits.placement.argmax(dim=-1)
        self.placement_correct += int(
            ((placement_predictions == placement_labels) & placement_mask).sum().item()
        )
        count = int(placement_mask.sum().item())
        self.placement_count += count
        if count:
            flat = (
                placement_labels[placement_mask] * 5
                + placement_predictions[placement_mask]
            )
            self.placement_confusion += torch.bincount(
                flat.detach().cpu(), minlength=25
            ).reshape(5, 5)

        named_logits = {
            "future_position": logits.future_position,
            "zone_entry": logits.zone_entry,
            "survival": logits.survival,
            "placement": logits.placement,
        }
        for name, tensor in named_logits.items():
            if tensor is not None:
                self.nonfinite_logits[name] += int((~torch.isfinite(tensor)).sum().item())
        named_losses = {
            "total": loss.total,
            "future_position": loss.future_position,
            "zone_entry": loss.zone_entry,
            "survival": loss.survival,
            "placement": loss.placement,
            **{
                POSITION_HORIZON_NAMES[index]: value
                for index, value in enumerate(loss.future_position_by_horizon)
            },
        }
        for name, tensor in named_losses.items():
            self.nonfinite_losses[name] += int((~torch.isfinite(tensor)).sum().item())

    def result(self) -> dict[str, Any]:
        loss_metrics = self.losses.metrics()
        placement_f1: list[float] = []
        per_class: list[dict[str, Any]] = []
        for class_index in range(5):
            true_positive = int(self.placement_confusion[class_index, class_index])
            false_positive = int(self.placement_confusion[:, class_index].sum()) - true_positive
            false_negative = int(self.placement_confusion[class_index, :].sum()) - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            value = 0.0 if denominator == 0 else 2 * true_positive / denominator
            placement_f1.append(value)
            per_class.append(
                {
                    "class_index": class_index,
                    "f1": value,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "support": int(self.placement_confusion[class_index, :].sum()),
                }
            )
        nonfinite_logits = {
            name: int(self.nonfinite_logits[name])
            for name in ("future_position", "zone_entry", "survival", "placement")
        }
        nonfinite_losses = {
            name: int(self.nonfinite_losses[name])
            for name in (
                "total",
                "future_position",
                *POSITION_HORIZON_NAMES,
                "zone_entry",
                "survival",
                "placement",
            )
        }
        return {
            "losses": {
                "future_position": loss_metrics["future_position"],
                "future_position_15s": loss_metrics["future_position_15s"],
                "future_position_30s": loss_metrics["future_position_30s"],
                "future_position_60s": loss_metrics["future_position_60s"],
                "zone_entry": loss_metrics["zone_entry"],
                "survival_bce_with_logits": loss_metrics["survival"],
                "placement": loss_metrics["placement"],
                "total_combined": loss_metrics["total"],
            },
            "accuracies": {
                "future_position": {
                    f"{HORIZONS[index]}s": {
                        "regular_cell_top1": self._accuracy(
                            self.position_regular_correct[index],
                            self.position_regular_count[index],
                        ),
                        "eliminated_class": self._accuracy(
                            self.position_eliminated_correct[index],
                            self.position_eliminated_count[index],
                        ),
                    }
                    for index in range(3)
                },
                "zone_entry_top1": self._accuracy(
                    self.entry_correct, self.entry_count
                ),
                "survival_threshold_0_5": self._accuracy(
                    self.survival_correct, self.survival_count
                ),
                "placement_top1": self._accuracy(
                    self.placement_correct, self.placement_count
                ),
            },
            "placement_macro_f1": {
                "value": sum(placement_f1) / 5,
                "zero_division_policy": 0.0,
                "classes_in_macro_average": [0, 1, 2, 3, 4],
                "per_class": per_class,
                "confusion_matrix_rows_truth_columns_prediction": (
                    self.placement_confusion.tolist()
                ),
            },
            "valid_target_counts": dict(loss_metrics["counts"]),
            "mask_counts": self.mask_counts,
            "query_count": self.query_count,
            "padded_or_inactive_query_slot_count": self.padded_query_slot_count,
            "nonfinite": {
                "logit_counts": nonfinite_logits,
                "loss_counts": nonfinite_losses,
                "total_count": sum(nonfinite_logits.values())
                + sum(nonfinite_losses.values()),
            },
        }


def _model_state_digest(model: RotationModel) -> str:
    return tensor_state_sha256(
        {name: tensor.detach() for name, tensor in model.state_dict().items()}
    )


def _load_locked_model(
    *,
    run_directory: Path,
    resolved: Mapping[str, Any],
    selection: Mapping[str, Any],
    device: torch.device,
) -> tuple[RotationModel, dict[str, Any]]:
    checkpoint_path = run_directory / LOCKED_CHECKPOINT_NAME
    _require(checkpoint_path.name == "best.pt", "only best.pt may be evaluated")
    _require(file_sha256(checkpoint_path) == LOCKED_CHECKPOINT_SHA256, "best.pt changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), "best.pt is not a checkpoint mapping")
    _require(set(checkpoint) == CHECKPOINT_KEYS, "best.pt checkpoint schema fields changed")
    _require(
        checkpoint.get("checkpoint_schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "best.pt checkpoint schema version changed",
    )
    _require(checkpoint.get("type") == "RotationModelTrainingOnlySupervision", "checkpoint model type changed")
    _require(checkpoint.get("precision") == "bf16", "checkpoint precision changed")
    _require(checkpoint.get("test_partition_evaluated") is False, "checkpoint reports test evaluation")
    checkpoint_access = checkpoint.get("data_access")
    _require(isinstance(checkpoint_access, dict), "checkpoint access record is missing")
    _require(
        checkpoint_access.get("test_session_parquet_attempt_count") == 0
        and checkpoint_access.get("test_session_parquet_open_count") == 0,
        "checkpoint reports prior test access",
    )
    state = checkpoint.get("training_state")
    _require(isinstance(state, dict), "checkpoint training state is missing")
    _require(
        state.get("completed_epochs") == LOCKED_COMPLETED_EPOCH
        and state.get("current_epoch") == LOCKED_COMPLETED_EPOCH
        and state.get("optimizer_step") == LOCKED_OPTIMIZER_STEP
        and state.get("best_completed_epoch") == LOCKED_COMPLETED_EPOCH
        and state.get("best_optimizer_step") == LOCKED_OPTIMIZER_STEP
        and state.get("best_metric") == LOCKED_VALIDATION_FUTURE_POSITION,
        "best.pt selection state changed",
    )
    _require(
        canonical_json(checkpoint.get("resolved_training_config"))
        == canonical_json(resolved),
        "checkpoint resolved configuration changed",
    )
    _require(
        checkpoint.get("compatibility", {}).get("bindings") == resolved.get("bindings"),
        "checkpoint compatibility bindings changed",
    )
    compatibility = checkpoint["compatibility"]
    _require(
        compatibility.get("resolved_training_config_sha256")
        == resolved["resolved_training_config_sha256"]
        and compatibility.get("encoder_config") == resolved["encoder_config"]
        and compatibility.get("head_config") == resolved["head_config"]
        and compatibility.get("loss_config") == resolved["loss_config"]
        and compatibility.get("optimizer") == resolved["optimizer_contract"]
        and compatibility.get("scheduler") == resolved["scheduler"]
        and compatibility.get("initial_parameter_sha256")
        == resolved["initial_parameters"]["complete_initial_parameter_sha256"]
        and compatibility.get("precision") == "bf16"
        and compatibility.get("device_type") == "cuda"
        and compatibility.get("sampler") == resolved["sampler"]
        and compatibility.get("checkpoint_selection_metric") == "future_position"
        and compatibility.get("test_partition_policy")
        == "sealed_zero_attempts_zero_opens",
        "checkpoint compatibility contract changed",
    )
    _require(
        checkpoint.get("optimizer_contract") == resolved["optimizer_contract"],
        "checkpoint optimizer contract changed",
    )
    _require(
        checkpoint.get("initial_parameters") == resolved["initial_parameters"],
        "checkpoint initial-parameter binding changed",
    )
    _require(checkpoint.get("amp_scaler_state") is None, "unexpected AMP scaler state")

    encoder_config = EncoderConfig(**resolved["encoder_config"])
    head_values = dict(resolved["head_config"])
    head_values["horizons_seconds"] = tuple(head_values["horizons_seconds"])
    head_config = RotationHeadConfig(**head_values)
    _require(head_config.horizons_seconds == HORIZONS, "head horizons changed")
    _require(head_config.enable_survival, "survival head is disabled")
    _require(head_config.enable_placement, "placement head is disabled")
    model = RotationModel(encoder_config=encoder_config, head_config=head_config)
    expected_state = model.state_dict()
    model_state = checkpoint.get("model_state")
    _require(isinstance(model_state, Mapping), "checkpoint model_state is missing")
    _require(set(model_state) == set(expected_state), "checkpoint model tensor names changed")
    for name, expected in expected_state.items():
        actual = model_state[name]
        _require(isinstance(actual, Tensor), f"checkpoint state is not a tensor: {name}")
        _require(actual.shape == expected.shape, f"checkpoint tensor shape changed: {name}")
        _require(actual.dtype == expected.dtype, f"checkpoint tensor dtype changed: {name}")
    incompatibility = model.load_state_dict(model_state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "strict checkpoint load was incomplete",
    )
    checkpoint_state_sha256 = tensor_state_sha256(model_state)

    export_path = run_directory / "encoder-only-best.pt"
    export_metadata = _read_json(
        run_directory / "encoder-only-best.json", "encoder-only best metadata"
    )
    _require(
        file_sha256(export_path) == export_metadata.get("raw_sha256"),
        "encoder-only best raw hash changed",
    )
    export = torch.load(export_path, map_location="cpu", weights_only=False)
    _require(isinstance(export, dict), "encoder-only best export is not a mapping")
    _require(
        export.get("checkpoint_schema_version")
        == "fncs-encoder-only-checkpoint:1.0",
        "encoder-only checkpoint schema changed",
    )
    encoder_state = export.get("encoder_state")
    _require(isinstance(encoder_state, Mapping), "encoder-only state is missing")
    expected_encoder_names = {
        name.removeprefix("encoder.")
        for name in model_state
        if name.startswith("encoder.")
    }
    _require(
        set(encoder_state) == expected_encoder_names,
        "encoder-only tensor names differ from best.pt",
    )
    bit_identical = 0
    for name in sorted(expected_encoder_names):
        full = model_state[f"encoder.{name}"]
        exported = encoder_state[name]
        _require(isinstance(exported, Tensor), f"exported encoder value is not a tensor: {name}")
        _require(exported.shape == full.shape, f"exported encoder shape differs: {name}")
        _require(exported.dtype == full.dtype, f"exported encoder dtype differs: {name}")
        _require(torch.equal(exported, full), f"exported encoder tensor differs: {name}")
        _require(
            tensor_bytes(exported) == tensor_bytes(full),
            f"exported encoder bytes differ: {name}",
        )
        bit_identical += 1
    encoder_state_hash = tensor_state_sha256(encoder_state)
    _require(
        encoder_state_hash == export.get("encoder_state_sha256")
        == export_metadata.get("encoder_state_sha256"),
        "encoder-only tensor-state hash changed",
    )
    _require(
        export.get("training_state", {}).get("optimizer_step")
        == LOCKED_OPTIMIZER_STEP,
        "encoder-only export optimizer step changed",
    )

    model.to(device)
    model.eval()
    _require(not model.training, "model did not enter evaluation mode")
    dropout_modules = [
        module for module in model.modules() if isinstance(module, torch.nn.Dropout)
    ]
    _require(
        dropout_modules and all(not module.training for module in dropout_modules),
        "dropout was not disabled by evaluation mode",
    )
    return model, {
        "strict_full_checkpoint_load": True,
        "loaded_checkpoint": "best.pt",
        "loaded_checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
        "last_pt_loaded": False,
        "checkpoint_model_tensor_count": len(model_state),
        "checkpoint_model_state_sha256": checkpoint_state_sha256,
        "encoder_only_export_loaded_for_identity_check_only": True,
        "encoder_only_export_used_for_evaluation": False,
        "encoder_only_raw_sha256": file_sha256(export_path),
        "encoder_tensor_count": bit_identical,
        "encoder_state_sha256": encoder_state_hash,
        "every_encoder_tensor_bit_identical": True,
        "dropout_module_count": len(dropout_modules),
        "dropout_disabled": True,
        "model_eval": True,
        "checkpoint_training_state": {
            "completed_epoch": state["completed_epochs"],
            "optimizer_step": state["optimizer_step"],
            "validation_future_position": state["best_metric"],
            "validation_target_counts": dict(
                state["last_validation_metrics"]["counts"]
            ),
        },
        "selection_payload_sha256": selection["payload_sha256"],
    }


def _evaluate_partition(
    *,
    partition: str,
    model: RotationModel,
    records: Sequence[_SessionRecord],
    session_ids: Sequence[str],
    inventory: Mapping[str, Any],
    profile: WorldGridProfile,
    head_config: RotationHeadConfig,
    loss_config: RotationLossConfig,
    encoder_config: EncoderConfig,
    config: FreshEncoderConfig,
    spec: PrecisionSpec,
    ledger: EvaluationAccessLedger,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    _require(partition in {"validation", "test"}, "unsupported evaluation partition")
    _require(len(session_ids) == 115, f"{partition} session count changed")
    _require(len(set(session_ids)) == len(session_ids), f"{partition} session IDs repeat")
    active = frozenset(session_ids)
    repository = EvaluationSessionRepository(
        tuple(record for record in records if record.session_id in active),
        profile,
        head_config,
        ledger,
        _bound_table_map(inventory),
    )
    lengths = _audited_session_lengths(
        records=records,
        session_ids=session_ids,
        encoder_config=encoder_config,
        repository=repository,
        ledger=ledger,
    )
    requests = validation_window_requests(
        lengths,
        session_ids,
        window_length_ticks=config.window_length_ticks,
    )
    _require(
        tuple(sorted(session_ids))
        == tuple(dict.fromkeys(request.session_id for request in requests)),
        f"{partition} request session order changed",
    )
    request_manifest = [
        {
            "session_id": request.session_id,
            "start_tick": request.start_tick,
            "length": request.length,
        }
        for request in requests
    ]
    request_manifest_sha256 = sha256_bytes(
        canonical_json(request_manifest).encode("utf-8")
    )
    model.eval()
    _require(not model.training, "model left evaluation mode")
    _require(
        all(
            not module.training
            for module in model.modules()
            if isinstance(module, torch.nn.Dropout)
        ),
        "dropout is active during evaluation",
    )
    state_before = _model_state_digest(model)
    metric_accumulator = EvaluationMetrics(loss_config, head_config.num_position_classes)
    order_digest = hashlib.sha256()
    forward_calls = 0
    gradient_enabled_batches = 0
    loss_requires_gradient_batches = 0
    started = time.perf_counter()
    if spec.device.type == "cuda":
        torch.cuda.synchronize(spec.device)
    with torch.inference_mode():
        _require(torch.is_inference_mode_enabled(), "inference mode did not activate")
        for request_batch in _batch_requests(requests, config.batch_size):
            items = repository.windows(request_batch)
            encoder_cpu, supervision_cpu = collate_training_windows(
                items,
                expected_profile_id=profile.profile_id,
                expected_profile_hash=profile.profile_hash,
                head_config=head_config,
            )
            order_digest.update(
                canonical_json(
                    {
                        "requests": [
                            {
                                "session_id": request.session_id,
                                "start_tick": request.start_tick,
                                "length": request.length,
                            }
                            for request in request_batch
                        ],
                        "session_ids": list(encoder_cpu.metadata.session_ids),
                        "team_ids": [list(value) for value in encoder_cpu.metadata.team_ids],
                    }
                ).encode("utf-8")
            )
            order_digest.update(tensor_bytes(encoder_cpu.batch.absolute_tick_index))
            encoder, supervision = _move_batch(
                encoder_cpu, supervision_cpu, spec, config.pin_memory
            )
            if torch.is_grad_enabled():
                gradient_enabled_batches += 1
            with _autocast_context(spec):
                output = model(encoder.batch)
                forward_calls += 1
                loss = compute_rotation_loss(
                    output.logits, supervision.targets, loss_config
                )
            if loss.total.requires_grad:
                loss_requires_gradient_batches += 1
            _validate_finite_logits_and_loss(
                output, loss, encoder.metadata.session_ids
            )
            metric_accumulator.update(output, supervision.targets, loss)
            order_digest.update(tensor_bytes(output.logits.query_mask))
    if spec.device.type == "cuda":
        torch.cuda.synchronize(spec.device)
    elapsed = time.perf_counter() - started
    state_after = _model_state_digest(model)
    _require(state_before == state_after, f"model state changed during {partition}")
    _require(gradient_enabled_batches == 0, "gradients were enabled during inference")
    _require(
        loss_requires_gradient_batches == 0,
        "evaluation losses unexpectedly require gradients",
    )
    metrics = metric_accumulator.result()
    _require(metrics["nonfinite"]["total_count"] == 0, "nonfinite evaluation value")
    payload = _metric_payload(
        {
            "schema_version": "fncs-encoder-deterministic-metric-payload:1.0",
            "partition": partition,
            "session_count": len(session_ids),
            "ordered_session_ids": list(sorted(session_ids)),
            "ordered_session_ids_sha256": sha256_bytes(
                canonical_json(list(sorted(session_ids))).encode("utf-8")
            ),
            "window_count": len(requests),
            "evaluated_timestep_count": sum(request.length for request in requests),
            "request_manifest_sha256": request_manifest_sha256,
            "batch_count": forward_calls,
            "batch_size": config.batch_size,
            "ordering": {
                "session": "lexicographic game_session_id",
                "window": "ascending non-overlapping start_tick within session",
                "batch": "sequential contiguous request groups without shuffling",
                "timestep": "ascending absolute_tick_index",
                "team": "tensorized canonical team_id order",
                "observed_order_sha256": order_digest.hexdigest(),
            },
            "metrics": metrics,
            "execution_contract": {
                "model_eval": True,
                "torch_inference_mode": True,
                "autocast_device_type": "cuda",
                "autocast_dtype": "torch.bfloat16",
                "dropout_disabled": True,
                "shuffle": False,
                "stochastic_augmentation": False,
                "backward_calls": 0,
                "training_calls": 0,
                "optimizer_constructions": 0,
                "scheduler_constructions": 0,
                "gradient_enabled_batches": gradient_enabled_batches,
                "loss_requires_gradient_batches": loss_requires_gradient_batches,
                "model_state_sha256_before": state_before,
                "model_state_sha256_after": state_after,
                "parameters_unchanged": True,
            },
        }
    )
    _verify_metric_payload(payload, partition)
    recovery = {
        "lifecycle_recovery_session_count": len(repository.lifecycle_recoveries),
        "lifecycle_recovery_session_ids": sorted(repository.lifecycle_recoveries),
        "zone_timing_recovery_session_count": len(repository.zone_timing_recoveries),
        "zone_timing_recovery_session_ids": sorted(repository.zone_timing_recoveries),
        "input_file_sha256_verification_count": len(repository.verified_files),
        "all_consumed_input_files_hash_verified": True,
    }
    del repository
    gc.collect()
    return payload, elapsed, recovery


def _pytest_gate(
    *, workspace: Path, output_directory: Path, test_files: Sequence[str], label: str
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str((workspace / "ml" / "src").resolve())
    with tempfile.TemporaryDirectory(
        prefix=f"fncs-{label}-", dir=output_directory.parent
    ) as temporary:
        command = [
            sys.executable,
            "-m",
            "pytest",
            *test_files,
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            temporary,
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=workspace / "ml",
            env=environment,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
    stdout = completed.stdout
    stderr = completed.stderr
    evidence = {
        "label": label,
        "command": command,
        "working_directory": str((workspace / "ml").resolve()),
        "test_files": list(test_files),
        "test_file_count": len(test_files),
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_raw_sha256": sha256_bytes(stdout),
        "stderr_raw_sha256": sha256_bytes(stderr),
        "stdout_tail": stdout.decode("utf-8", errors="replace").splitlines()[-3:],
        "stderr_tail": stderr.decode("utf-8", errors="replace").splitlines()[-3:],
        "passed": completed.returncode == 0,
    }
    _require(completed.returncode == 0, f"{label} failed")
    return evidence


def _qualification_tests(workspace: Path, output_directory: Path) -> dict[str, Any]:
    focused = _pytest_gate(
        workspace=workspace,
        output_directory=output_directory,
        test_files=("tests/test_fncs_encoder_evaluate.py",),
        label="focused_evaluator_tests",
    )
    regression_files = (
        "tests/test_features_and_edges.py",
        "tests/test_model.py",
        "tests/test_rotation_heads.py",
        "tests/test_spatial.py",
        "tests/test_supervision.py",
        "tests/test_temporal.py",
        "tests/test_tensorize.py",
        "tests/test_training.py",
        "tests/test_validate_world_grid.py",
        "tests/test_world_grid.py",
        "tests/test_fncs_encoder_training.py",
    )
    regression = _pytest_gate(
        workspace=workspace,
        output_directory=output_directory,
        test_files=regression_files,
        label="existing_encoder_and_training_regression_suite",
    )
    return {"focused": focused, "regression": regression, "all_passed": True}


def _predeclared_metrics() -> dict[str, Any]:
    return {
        "primary": {
            "name": "future_position",
            "definition": (
                "sum of the independent masked mean cross-entropies at 15, 30, "
                "and 60 seconds, identical to validation/checkpoint selection"
            ),
            "direction": "lower_is_better",
        },
        "secondary": [
            "total_combined_loss",
            "future_position_loss_15s",
            "future_position_loss_30s",
            "future_position_loss_60s",
            "per_horizon_regular_cell_top1_accuracy",
            "per_horizon_eliminated_class_accuracy",
            "zone_entry_loss",
            "zone_entry_top1_accuracy",
            "survival_bce_with_logits_loss",
            "survival_threshold_0_5_accuracy",
            "placement_loss",
            "placement_top1_accuracy",
            "placement_macro_f1",
            "valid_target_and_mask_counts_for_every_head_and_horizon",
            "validation_to_test_absolute_and_relative_generalization_gaps",
        ],
        "reductions": {
            "future_position": (
                "for each horizon, torch cross_entropy mean over target mask AND "
                "query mask; sum the three count-weighted corpus horizon means"
            ),
            "zone_entry": "cross-entropy mean over target mask AND query mask",
            "survival": "binary-cross-entropy-with-logits mean over target mask AND query mask",
            "placement": "cross-entropy mean over target mask AND query mask",
            "total": (
                "future_position + 1.0*zone_entry + 1.0*survival + 1.0*placement"
            ),
            "regular_cell_top1_accuracy": (
                "argmax accuracy among valid future-position targets excluding class 1024"
            ),
            "eliminated_class_accuracy": (
                "argmax equals class 1024 among valid targets whose truth is class 1024"
            ),
            "survival_accuracy": "logit >= 0.0, equivalent to sigmoid probability >= 0.5",
            "placement_macro_f1": (
                "unweighted mean of class F1 for all five fixed placement buckets; "
                "zero for a class with zero 2TP+FP+FN denominator"
            ),
            "generalization_gap": (
                "test minus validation, absolute magnitude, and absolute magnitude "
                "divided by abs(validation) when validation is nonzero"
            ),
        },
        "confidence_intervals": {
            "reported": False,
            "reason": "confidence intervals were not part of the requested predeclared metric set",
            "individual_window_bootstrap_used": False,
        },
        "metric_availability_policy": (
            "If a fixed head has zero valid targets, emit null with its zero count and an "
            "explanation; never change the target or metric contract."
        ),
    }


def _make_contract(
    *,
    selection: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    split: Mapping[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    test_ids = list(split["ordered_session_ids"]["test"])
    _require(len(test_ids) == 115 and len(set(test_ids)) == 115, "test list changed")
    return {
        "schema_version": "fncs-encoder-final-evaluation-contract:1.0",
        "status": "authorized",
        "scientific_boundary": {
            "locked_one_shot_test_evaluation": True,
            "checkpoint_selection_on_validation_only": True,
            "checkpoint_comparison_on_test_forbidden": True,
            "threshold_mask_weight_temperature_or_metric_tuning_after_test_forbidden": True,
            "architecture_head_target_or_world_grid_change_forbidden": True,
            "trajectory_decoder_training_or_evaluation_forbidden": True,
            "exploratory_test_analysis_after_publication_forbidden": True,
            "automatic_retry_after_any_test_access_forbidden": True,
        },
        "checkpoint": {
            "path": str((run_directory / "best.pt").resolve()),
            "sha256": LOCKED_CHECKPOINT_SHA256,
            "completed_epoch": LOCKED_COMPLETED_EPOCH,
            "optimizer_step": LOCKED_OPTIMIZER_STEP,
            "validation_selection_metric": "future_position",
            "validation_selection_value": LOCKED_VALIDATION_FUTURE_POSITION,
            "full_checkpoint_with_supervision_heads_required": True,
            "last_pt_forbidden": True,
        },
        "bindings": dict(selection["bindings"]),
        "evaluator": {
            "entry_point": "python -m fncs_encoder_training.evaluate",
            "source_sha256": source_manifest["evaluator_source_sha256"],
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "model_eval": True,
            "torch_inference_mode": True,
            "precision": "bf16",
            "device": "cuda:0",
            "cuda_runtime": "13.0",
            "dropout_disabled": True,
            "shuffle": False,
            "stochastic_augmentation": False,
            "backward_pass": False,
            "optimizer_or_scheduler_construction": False,
        },
        "ordering": {
            "authorized_session_list": "exact order stored in sealed split manifest",
            "evaluation_session_traversal": "lexicographic game_session_id",
            "window": "ascending contiguous non-overlapping 64-tick windows",
            "batch": "sequential contiguous groups of four requests",
            "team": "tensorized canonical team_id order",
            "timestep": "ascending absolute_tick_index",
        },
        "metrics": _predeclared_metrics(),
        "validation_gates": {
            "validation_reproduction_payload_sha256": validation["payload_sha256"],
            "all_passed": validation["all_validation_gates_passed"],
            "two_byte_identical_metric_payloads": validation[
                "byte_identical_metric_payloads"
            ],
            "future_position_reproduced": validation["future_position_gate"]["passed"],
            "mask_and_target_counts_reproduced": validation["count_gate"]["passed"],
            "finite_logits_and_losses": validation["finite_gate"]["passed"],
            "zero_training_or_optimizer_construction": validation["inference_only_gate"]["passed"],
            "zero_test_attempts_and_opens": validation["test_access_gate"]["passed"],
        },
        "authorization": {
            "authorized": True,
            "authorized_before_first_test_file_access": True,
            "ordered_test_session_ids": test_ids,
            "ordered_test_session_ids_sha256": sha256_bytes(
                canonical_json(test_ids).encode("utf-8")
            ),
            "session_count": 115,
            "exactly_matches_sealed_split_manifest": True,
            "reject_any_session_outside_list": True,
        },
        "decision_rule": {
            "binary_acceptance_threshold_predeclared": False,
            "reason": (
                "No scientifically justified binary generalization threshold was supplied; "
                "the conclusion is restricted to the observed locked validation-to-test gaps."
            ),
        },
    }


def _out_of_bounds_summary(
    *,
    path: Path,
    session_ids: Sequence[str],
    expected_raw_sha256: str,
    ledger: EvaluationAccessLedger,
) -> dict[str, Any]:
    _require(file_sha256(path) == expected_raw_sha256, "out-of-bounds audit changed")
    ledger.record_derived_artifact_read(path, "predeclared test out-of-bounds counts")
    selected = frozenset(session_ids)
    sources: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    total = 0
    target_relevant = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationContractError(
                    f"invalid out-of-bounds audit line {line_number}"
                ) from exc
            if row.get("game_session_id") not in selected:
                continue
            source = str(row.get("source"))
            policy = str(row.get("target_policy"))
            sources[source] += 1
            policies[policy] += 1
            total += 1
            if source == "eligible_player_centroid":
                target_relevant += 1
    return {
        "source_artifact": path.name,
        "source_artifact_raw_sha256": expected_raw_sha256,
        "session_count": len(selected),
        "out_of_bounds_coordinate_record_count": total,
        "eligible_player_centroid_out_of_bounds_record_count": target_relevant,
        "counts_by_coordinate_source": dict(sorted(sources.items())),
        "counts_by_policy": dict(sorted(policies.items())),
        "interpretation": (
            "These are independently audited source-coordinate records for the locked "
            "sessions. Target masking counts in the metric payload are the authoritative "
            "query-level mask counts; source records are not multiplied by queries/horizons."
        ),
    }


def _flatten_comparable_metrics(payload: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = payload["metrics"]
    losses = metrics["losses"]
    accuracies = metrics["accuracies"]
    output: dict[str, float | None] = {
        f"loss.{name}": value for name, value in losses.items()
    }
    for horizon in ("15s", "30s", "60s"):
        position = accuracies["future_position"][horizon]
        output[f"accuracy.future_position.{horizon}.regular_cell_top1"] = position[
            "regular_cell_top1"
        ]["value"]
        output[f"accuracy.future_position.{horizon}.eliminated_class"] = position[
            "eliminated_class"
        ]["value"]
    output["accuracy.zone_entry_top1"] = accuracies["zone_entry_top1"]["value"]
    output["accuracy.survival_threshold_0_5"] = accuracies[
        "survival_threshold_0_5"
    ]["value"]
    output["accuracy.placement_top1"] = accuracies["placement_top1"]["value"]
    output["f1.placement_macro"] = metrics["placement_macro_f1"]["value"]
    return output


def _generalization_gap(
    validation_payload: Mapping[str, Any], test_payload: Mapping[str, Any]
) -> dict[str, Any]:
    validation = _flatten_comparable_metrics(validation_payload)
    test = _flatten_comparable_metrics(test_payload)
    _require(set(validation) == set(test), "validation/test metric schemas differ")
    gaps: dict[str, Any] = {}
    for name in sorted(validation):
        left = validation[name]
        right = test[name]
        if left is None or right is None:
            gaps[name] = {
                "validation": left,
                "test": right,
                "test_minus_validation": None,
                "absolute_gap": None,
                "relative_absolute_gap": None,
                "available": False,
                "explanation": "metric has zero valid targets in at least one partition",
            }
            continue
        signed = right - left
        absolute = abs(signed)
        gaps[name] = {
            "validation": left,
            "test": right,
            "test_minus_validation": signed,
            "absolute_gap": absolute,
            "relative_absolute_gap": absolute / abs(left) if left != 0.0 else None,
            "available": True,
        }
    return {
        "schema_version": "fncs-encoder-generalization-gap:1.0",
        "validation_metric_payload_sha256": validation_payload[
            "metric_payload_sha256"
        ],
        "test_metric_payload_sha256": test_payload["metric_payload_sha256"],
        "gap_contract": (
            "test minus validation; absolute magnitude; absolute magnitude / abs(validation)"
        ),
        "metrics": gaps,
        "primary": gaps["loss.future_position"],
    }


def _artifact_manifest(output_directory: Path) -> dict[str, Any]:
    observed = {path.name for path in output_directory.iterdir() if path.is_file()}
    expected_before_manifest = REQUIRED_ARTIFACTS - {"artifact_manifest.json"}
    _require(
        observed == expected_before_manifest,
        f"evaluation artifact set differs before manifest: {sorted(observed)}",
    )
    artifacts: list[dict[str, Any]] = []
    for name in sorted(expected_before_manifest):
        path = output_directory / name
        parsed = _read_json(path, name)
        _verify_seal(parsed, name)
        artifacts.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "raw_sha256": file_sha256(path),
                "payload_sha256": parsed["payload_sha256"],
            }
        )
    return {
        "schema_version": "fncs-encoder-final-evaluation-artifact-manifest:1.0",
        "status": "complete",
        "run_directory": str(output_directory.resolve()),
        "artifact_count_excluding_manifest": len(artifacts),
        "manifest_self_hash_excluded_to_avoid_recursion": True,
        "all_artifacts_self_sealed": True,
        "artifacts": artifacts,
    }


def _verify_artifact_manifest(output_directory: Path) -> dict[str, Any]:
    manifest_path = output_directory / "artifact_manifest.json"
    manifest = _read_json(manifest_path, "artifact manifest")
    _verify_seal(manifest, "artifact manifest")
    observed = {path.name for path in output_directory.iterdir() if path.is_file()}
    _require(observed == REQUIRED_ARTIFACTS, "final evaluation artifact set changed")
    for entry in manifest["artifacts"]:
        path = output_directory / entry["path"]
        _require(path.is_file(), f"manifest artifact is missing: {entry['path']}")
        _require(path.stat().st_size == entry["size_bytes"], f"artifact size changed: {entry['path']}")
        _require(file_sha256(path) == entry["raw_sha256"], f"artifact hash changed: {entry['path']}")
        parsed = _read_json(path, entry["path"])
        _verify_seal(parsed, entry["path"])
        _require(parsed["payload_sha256"] == entry["payload_sha256"], f"artifact payload hash changed: {entry['path']}")
    return manifest


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
        elif path.is_dir():
            os.chmod(path, stat.S_IREAD | stat.S_IEXEC)
    os.chmod(root, stat.S_IREAD | stat.S_IEXEC)


def _failure_report(
    *, output_directory: Path, ledger: EvaluationAccessLedger, exc: BaseException
) -> None:
    try:
        ledger.fail_test(type(exc).__name__)
        _write_sealed(
            output_directory / "test_failure.json",
            {
                "schema_version": "fncs-encoder-one-shot-test-failure:1.0",
                "status": "failed_requires_explicit_recovery_decision",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "test_file_attempts": ledger.test_attempts,
                "test_file_opens": ledger.test_opens,
                "test_partition_unsealed": ledger.test_partition_unsealed,
                "automatic_retry_permitted": False,
            },
        )
    except Exception:
        # Preserve the original failure. The ledger persists before every test I/O,
        # so even a secondary reporting failure leaves an unsealed marker.
        pass


def run_evaluation(
    config_path: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    config_source = Path(config_path).resolve()
    config = load_config(config_source)
    run_directory = Path(config.run_directory).resolve()
    workspace = run_directory.parents[1]
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else run_directory.with_name(run_directory.name + REPORT_SUFFIX)
    )
    _require(output.is_dir(), "evaluation directory with checkpoint selection is missing")
    observed_initial = {path.name for path in output.iterdir() if path.is_file()}
    _require(
        observed_initial == {"checkpoint_selection.json"},
        "evaluation directory must contain only the prepublished checkpoint selection",
    )
    selection = _verify_checkpoint_selection(output, run_directory, config_source)
    resolved, inventory, split, profile = _verify_static_bindings(
        config=config,
        config_path=config_source,
        run_directory=run_directory,
        selection=selection,
    )
    records, partition_by_session = _records(
        inventory,
        split,
        dataset_root=Path(config.dataset_root),
        profile=profile,
    )
    validation_ids = tuple(split["ordered_session_ids"]["validation"])
    test_ids = tuple(split["ordered_session_ids"]["test"])
    _require(len(validation_ids) == 115 and len(test_ids) == 115, "split population changed")
    _require(set(validation_ids).isdisjoint(test_ids), "validation and test overlap")
    ledger = EvaluationAccessLedger(
        path=output / "data_access_ledger.json",
        partition_by_session=partition_by_session,
        test_session_ids=test_ids,
        split_manifest_sha256=selection["bindings"]["split_manifest_sha256"],
    )

    tests = _qualification_tests(workspace, output)
    source_manifest = _write_sealed(
        output / "evaluator_source_manifest.json",
        _evaluator_source_manifest(workspace),
    )
    seed_everything(config.seed)
    environment = environment_report(workspace)
    _require(environment.get("torch") == torch.__version__, "environment torch version changed")
    _require(environment.get("cuda_runtime") == "13.0", "CUDA 13.0 is required")
    _require(environment.get("cuda_available") is True, "CUDA is unavailable")
    _require(environment.get("bf16_supported") is True, "CUDA BF16 is unavailable")
    _require(environment.get("gpu_model") == "NVIDIA GeForce RTX 4090", "RTX 4090 is required")
    environment.update(
        {
            "evaluation_device": "cuda:0",
            "evaluation_precision": "bf16",
            "deterministic_algorithms_required": True,
            "tf32_disabled": True,
        }
    )
    environment_artifact = _write_sealed(output / "environment.json", environment)
    spec = resolve_precision(config.device, config.precision)  # type: ignore[arg-type]
    _require(spec.device == torch.device("cuda:0"), "evaluation device changed")
    _require(spec.precision == "bf16", "evaluation precision changed")
    model, checkpoint_verification = _load_locked_model(
        run_directory=run_directory,
        resolved=resolved,
        selection=selection,
        device=spec.device,
    )
    encoder_config = EncoderConfig(**resolved["encoder_config"])
    head_values = dict(resolved["head_config"])
    head_values["horizons_seconds"] = tuple(head_values["horizons_seconds"])
    head_config = RotationHeadConfig(**head_values)
    loss_config = RotationLossConfig(**resolved["loss_config"])

    validation_passes: list[dict[str, Any]] = []
    validation_times: list[float] = []
    validation_recoveries: list[dict[str, Any]] = []
    for pass_number in (1, 2):
        ledger.start_validation_pass(pass_number)
        payload, elapsed, recovery = _evaluate_partition(
            partition="validation",
            model=model,
            records=records,
            session_ids=validation_ids,
            inventory=inventory,
            profile=profile,
            head_config=head_config,
            loss_config=loss_config,
            encoder_config=encoder_config,
            config=config,
            spec=spec,
            ledger=ledger,
        )
        ledger.complete_validation_pass(pass_number)
        validation_passes.append(payload)
        validation_times.append(elapsed)
        validation_recoveries.append(recovery)
    first_bytes = rendered_json(validation_passes[0])
    second_bytes = rendered_json(validation_passes[1])
    _require(first_bytes == second_bytes, "validation metric payloads are not byte-identical")
    validation_payload = validation_passes[0]
    actual_future = validation_payload["metrics"]["losses"]["future_position"]
    _require(isinstance(actual_future, float), "validation future_position is unavailable")
    future_passed = math.isclose(
        actual_future,
        LOCKED_VALIDATION_FUTURE_POSITION,
        rel_tol=VALIDATION_RELATIVE_TOLERANCE,
        abs_tol=VALIDATION_ABSOLUTE_TOLERANCE,
    )
    _require(future_passed, "validation future_position was not reproduced")
    checkpoint_counts = checkpoint_verification["checkpoint_training_state"][
        "validation_target_counts"
    ]
    expected_counts = {name: int(value) for name, value in checkpoint_counts.items()}
    actual_counts = {
        name: int(value)
        for name, value in validation_payload["metrics"]["valid_target_counts"].items()
    }
    _require(actual_counts == expected_counts, "validation target/mask counts changed")
    finite_passed = validation_payload["metrics"]["nonfinite"]["total_count"] == 0
    inference_contract = validation_payload["execution_contract"]
    inference_passed = (
        inference_contract["backward_calls"] == 0
        and inference_contract["training_calls"] == 0
        and inference_contract["optimizer_constructions"] == 0
        and inference_contract["scheduler_constructions"] == 0
        and inference_contract["parameters_unchanged"] is True
        and source_manifest["static_safety_audit"]["passed"] is True
    )
    ledger_snapshot = ledger.report()
    test_access_passed = (
        ledger.test_attempts == 0
        and ledger.test_opens == 0
        and ledger_snapshot["partition_attempt_counts"]["test"] == 0
        and ledger_snapshot["partition_open_counts"]["test"] == 0
    )
    _require(finite_passed, "validation logits or losses are nonfinite")
    _require(inference_passed, "inference-only validation gate failed")
    _require(test_access_passed, "test access occurred before authorization")
    validation_artifact = _write_sealed(
        output / "validation_reproduction.json",
        {
            "schema_version": "fncs-encoder-validation-reproduction:1.0",
            "status": "passed",
            "all_validation_gates_passed": True,
            "qualification_tests": tests,
            "checkpoint_verification": checkpoint_verification,
            "validation_metric_payload": validation_payload,
            "pass_1_metric_payload_raw_sha256": sha256_bytes(first_bytes),
            "pass_2_metric_payload_raw_sha256": sha256_bytes(second_bytes),
            "byte_identical_metric_payloads": True,
            "validation_pass_elapsed_seconds": validation_times,
            "validation_input_recovery": validation_recoveries,
            "future_position_gate": {
                "expected": LOCKED_VALIDATION_FUTURE_POSITION,
                "actual": actual_future,
                "absolute_tolerance": VALIDATION_ABSOLUTE_TOLERANCE,
                "relative_tolerance": VALIDATION_RELATIVE_TOLERANCE,
                "absolute_error": abs(actual_future - LOCKED_VALIDATION_FUTURE_POSITION),
                "passed": future_passed,
            },
            "count_gate": {
                "expected": expected_counts,
                "actual": actual_counts,
                "passed": True,
            },
            "finite_gate": {
                "nonfinite": validation_payload["metrics"]["nonfinite"],
                "passed": finite_passed,
            },
            "inference_only_gate": {
                "execution_contract": inference_contract,
                "source_static_safety_audit": source_manifest["static_safety_audit"],
                "passed": inference_passed,
            },
            "test_access_gate": {
                "test_attempts": ledger.test_attempts,
                "test_opens": ledger.test_opens,
                "passed": test_access_passed,
            },
        },
    )

    contract = _write_sealed(
        output / "evaluation_contract.json",
        _make_contract(
            selection=selection,
            source_manifest=source_manifest,
            validation=validation_artifact,
            split=split,
            run_directory=run_directory,
        ),
    )
    _require(contract["authorization"]["authorized"] is True, "contract is not authorized")
    _require(
        tuple(contract["authorization"]["ordered_test_session_ids"]) == test_ids,
        "contract authorization list differs from the sealed split manifest",
    )
    # Recheck every static binding and the finalized evaluator immediately before
    # authorization. No test-session Parquet file has been attempted or opened.
    _verify_static_bindings(
        config=config,
        config_path=config_source,
        run_directory=run_directory,
        selection=selection,
    )
    _require(
        file_sha256(Path(__file__).resolve())
        == source_manifest["evaluator_sources"][0]["raw_sha256"],
        "evaluator source changed after contract sealing",
    )
    _require(file_sha256(run_directory / "best.pt") == LOCKED_CHECKPOINT_SHA256, "best.pt changed before test")
    _require(ledger.test_attempts == 0 and ledger.test_opens == 0, "test was accessed before authorization")
    ledger.authorize_test(output / "evaluation_contract.json", contract)
    ledger.start_test_pass()

    try:
        test_payload, test_elapsed, test_recovery = _evaluate_partition(
            partition="test",
            model=model,
            records=records,
            session_ids=test_ids,
            inventory=inventory,
            profile=profile,
            head_config=head_config,
            loss_config=loss_config,
            encoder_config=encoder_config,
            config=config,
            spec=spec,
            ledger=ledger,
        )
        world_audit = _read_json(
            run_directory / "world_grid_compatibility_audit.json",
            "world-grid compatibility audit",
        )
        out_of_bounds = _out_of_bounds_summary(
            path=run_directory / "world_grid_out_of_bounds.jsonl",
            session_ids=test_ids,
            expected_raw_sha256=world_audit["out_of_bounds_coordinates_raw_sha256"],
            ledger=ledger,
        )
        ledger.complete_test_pass()
        _require(
            test_payload["session_count"] == 115
            and test_payload["metrics"]["nonfinite"]["total_count"] == 0,
            "test evaluation census or numerical gate failed",
        )
        test_artifact = _write_sealed(
            output / "test_evaluation.json",
            {
                "schema_version": "fncs-encoder-one-shot-test-evaluation:1.0",
                "status": "completed",
                "one_shot_pass_number": 1,
                "automatic_retry_occurred": False,
                "metric_payload": test_payload,
                "evaluation_elapsed_seconds": test_elapsed,
                "input_recovery": test_recovery,
                "out_of_bounds": out_of_bounds,
                "checkpoint": {
                    "path": str((run_directory / "best.pt").resolve()),
                    "sha256": LOCKED_CHECKPOINT_SHA256,
                    "completed_epoch": LOCKED_COMPLETED_EPOCH,
                    "optimizer_step": LOCKED_OPTIMIZER_STEP,
                },
                "contract_payload_sha256": contract["payload_sha256"],
                "evaluator_source_sha256": source_manifest["evaluator_source_sha256"],
                "test_access": {
                    "attempts": ledger.test_attempts,
                    "opens": ledger.test_opens,
                    "passes_started": ledger.test_evaluation_passes_started,
                    "passes_completed": ledger.test_evaluation_passes_completed,
                    "partition_unsealed": ledger.test_partition_unsealed,
                },
                "training": {
                    "training_calls": 0,
                    "backward_calls": 0,
                    "optimizer_constructions": 0,
                    "scheduler_constructions": 0,
                    "parameters_unchanged": True,
                },
                "trajectory_decoder": {
                    "trained": False,
                    "evaluated": False,
                    "present": False,
                },
            },
        )
        gap_artifact = _write_sealed(
            output / "generalization_gap.json",
            _generalization_gap(validation_payload, test_payload),
        )
        # Persist the final self-sealed ledger after every test access and before
        # any summary claims are constructed.
        ledger.persist()
        final_ledger = _read_json(output / "data_access_ledger.json", "final data-access ledger")
        _verify_seal(final_ledger, "final data-access ledger")
        _require(
            final_ledger["test_evaluation_passes_started"] == 1
            and final_ledger["test_evaluation_passes_completed"] == 1
            and final_ledger["test_partition_unsealed"] is True,
            "final test access ledger is incomplete",
        )
        primary_gap = gap_artifact["primary"]
        direction = (
            "lower"
            if primary_gap["test_minus_validation"] < 0
            else "higher"
            if primary_gap["test_minus_validation"] > 0
            else "equal"
        )
        conclusion = (
            "The selected checkpoint's locked test future_position loss was "
            f"{test_payload['metrics']['losses']['future_position']!r}, versus "
            f"{validation_payload['metrics']['losses']['future_position']!r} on validation "
            f"({direction} by {primary_gap['absolute_gap']!r}). This is a descriptive "
            "one-shot generalization result; no binary acceptance threshold was predeclared."
        )
        final_summary = _write_sealed(
            output / "final_evaluation_summary.json",
            {
                "schema_version": "fncs-encoder-final-evaluation-summary:1.0",
                "status": "completed",
                "scientific_scope": (
                    "Generalization of the validation-selected encoder checkpoint to the "
                    "previously sealed FNCS test partition under the predeclared supervision objectives."
                ),
                "conclusion": conclusion,
                "primary_metric": {
                    "name": "future_position",
                    "validation": validation_payload["metrics"]["losses"]["future_position"],
                    "test": test_payload["metrics"]["losses"]["future_position"],
                    "test_minus_validation": primary_gap["test_minus_validation"],
                    "absolute_gap": primary_gap["absolute_gap"],
                    "relative_absolute_gap": primary_gap["relative_absolute_gap"],
                },
                "test_census": {
                    "sessions": test_payload["session_count"],
                    "windows": test_payload["window_count"],
                    "queries": test_payload["metrics"]["query_count"],
                    "timesteps": test_payload["evaluated_timestep_count"],
                    "valid_targets": test_payload["metrics"]["valid_target_counts"],
                },
                "all_primary_and_secondary_metrics": test_payload["metrics"],
                "nonfinite_value_count": test_payload["metrics"]["nonfinite"]["total_count"],
                "masking_counts": test_payload["metrics"]["mask_counts"],
                "out_of_bounds_counts": out_of_bounds,
                "bindings": {
                    "checkpoint_selection_payload_sha256": selection["payload_sha256"],
                    "evaluation_contract_payload_sha256": contract["payload_sha256"],
                    "validation_reproduction_payload_sha256": validation_artifact["payload_sha256"],
                    "test_evaluation_payload_sha256": test_artifact["payload_sha256"],
                    "generalization_gap_payload_sha256": gap_artifact["payload_sha256"],
                    "environment_payload_sha256": environment_artifact["payload_sha256"],
                    "evaluator_source_sha256": source_manifest["evaluator_source_sha256"],
                    "checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
                    **dict(selection["bindings"]),
                },
                "test_access": {
                    "attempts": ledger.test_attempts,
                    "opens": ledger.test_opens,
                    "exactly_one_pass_started": True,
                    "exactly_one_pass_completed": True,
                    "automatic_retry_occurred": False,
                },
                "invariants": {
                    "no_training_occurred": True,
                    "no_backward_pass_occurred": True,
                    "no_optimizer_or_scheduler_constructed": True,
                    "model_parameters_unchanged": True,
                    "full_best_checkpoint_with_heads_used": True,
                    "last_checkpoint_not_loaded": True,
                    "test_partition_now_permanently_unsealed": True,
                    "trajectory_decoder_trained_or_evaluated": False,
                },
                "scientific_caveat": (
                    "All 1,150 source sessions were accepted with warnings and remain "
                    "unattested with incomplete provenance."
                ),
                "forbidden_claims": {
                    "optimal_rotation_forecasting_established": False,
                    "decoder_quality_established": False,
                    "causal_decision_quality_established": False,
                    "production_readiness_established": False,
                    "reason": (
                        "Those claims require a separately trained and evaluated decoder; "
                        "decoder development must not reopen this test split."
                    ),
                },
            },
        )
        manifest = _write_sealed(
            output / "artifact_manifest.json", _artifact_manifest(output)
        )
        _verify_artifact_manifest(output)
        _make_read_only(output)
        return {
            "status": "completed",
            "output_directory": str(output),
            "primary_test_future_position": test_payload["metrics"]["losses"]["future_position"],
            "primary_validation_future_position": validation_payload["metrics"]["losses"]["future_position"],
            "artifact_manifest_payload_sha256": manifest["payload_sha256"],
            "final_summary_payload_sha256": final_summary["payload_sha256"],
            "test_partition_unsealed": True,
        }
    except BaseException as exc:
        _failure_report(output_directory=output, ledger=ledger, exc=exc)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked one-shot held-out evaluation of the FNCS encoder"
    )
    parser.add_argument("--config", required=True, help="fresh encoder run configuration")
    parser.add_argument(
        "--output-directory",
        help="override the fixed report directory (intended only for isolated qualification tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_evaluation(args.config, output_directory=args.output_directory)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
