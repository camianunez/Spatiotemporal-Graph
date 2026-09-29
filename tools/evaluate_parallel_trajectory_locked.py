from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch import Tensor


sys.dont_write_bytecode = True

ML_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_ROOT.parent
SOURCE_ROOT = ML_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from fortnite_encoder.contracts import EncoderBatch  # noqa: E402
from fortnite_encoder.planner_contracts import PlannerBatch, PlannerObservation  # noqa: E402
from fortnite_encoder.planner_policy import (  # noqa: E402
    OBSERVATION_PROVIDER_ID,
    apply_planner_input_policy,
)
from fortnite_encoder.planner_supervision import (  # noqa: E402
    PlannerQuery,
    PlannerTargetSource,
    load_planner_target_source,
)
from fortnite_encoder.tensorize import load_match_session  # noqa: E402
from fortnite_parallel_trajectory.config import (  # noqa: E402
    NUM_HORIZONS,
    NUM_ROUTE_MODES,
)
from fortnite_parallel_trajectory.contracts import (  # noqa: E402
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
)
from fortnite_parallel_trajectory.losses import route_mixture_nll  # noqa: E402
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel  # noqa: E402
from fortnite_parallel_trajectory_training.checkpoint import (  # noqa: E402
    TrainingState,
    load_training_checkpoint,
)
from fortnite_parallel_trajectory_training.data import (  # noqa: E402
    collate_parallel_trajectory_query_examples,
    eligible_parallel_trajectory_queries,
    prepare_parallel_trajectory_query_examples,
    stable_seed,
    uniform_sample,
)
from fortnite_parallel_trajectory_training.metrics import (  # noqa: E402
    ParallelTrajectoryMetricAccumulator,
)
from fortnite_parallel_trajectory_training.optimization import (  # noqa: E402
    PiecewiseLinearScheduler,
    build_adamw_optimizer,
)
from fortnite_parallel_trajectory_training.provenance import (  # noqa: E402
    VerifiedTrainingSetup,
    canonical_json,
    file_sha256,
    initialize_model,
    module_state_digest,
    package_source_digest,
    tensor_state_digest,
    verify_training_setup,
)
from fortnite_parallel_trajectory_training.training import (  # noqa: E402
    PrecisionSpec,
    environment_metadata,
    resolve_precision,
    seed_everything,
)


SCHEMA_VERSION = "parallel-trajectory-locked-evaluation:1.0"
CHECKPOINT_SELECTION_SCHEMA = "parallel-trajectory-checkpoint-selection:1.0"
CONTRACT_SCHEMA = "parallel-trajectory-evaluation-contract:1.0"
ACCESS_LEDGER_SCHEMA = "parallel-trajectory-evaluation-access-ledger:1.0"
VALIDATION_SCHEMA = "parallel-trajectory-validation-evaluation:1.0"
TEST_SCHEMA = "parallel-trajectory-test-evaluation:1.0"
SUMMARY_SCHEMA = "parallel-trajectory-final-evaluation-summary:1.0"
MANIFEST_SCHEMA = "parallel-trajectory-evaluation-artifact-manifest:1.0"

CONFIG_PATH = REPOSITORY_ROOT / "configs" / "parallel-trajectory-training-v1-20260813.json"
PARENT_RUN = REPOSITORY_ROOT / "runs" / "parallel-trajectory-decoder-v1-20260813-run-1"
RUN_DIRECTORY = REPOSITORY_ROOT / "runs" / "parallel-trajectory-decoder-v1-20260813-run-1-resume-1"
RECOVERY_AUDIT = REPOSITORY_ROOT / "audit" / "parallel-trajectory-recovery-20260815T181048-0400"
DEFAULT_EVALUATION_DIRECTORY = (
    REPOSITORY_ROOT
    / "runs"
    / "parallel-trajectory-decoder-v1-20260813-run-1-evaluation-20260816"
)

EVALUATION_SEED = 20260804
QUERY_SAMPLER_VERSION = "parallel-trajectory-validation-query-sample-v1"
QUERIES_PER_SESSION = 32
MAX_QUERIES_PER_FORWARD = 4
HORIZONS_SECONDS = tuple(range(5, 61, 5))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
PREDICTIVE_SAMPLES = 512
PREDICTIVE_LEVELS = (0.50, 0.80, 0.90)
MIN_GROUP_ROUTES = 20
MIN_GROUP_SESSIONS = 3

STATIONARY_PATH_LENGTH_METERS = 10.0
STATIONARY_MAX_RADIUS_METERS = 10.0
ZERO_MOTION_PATH_LENGTH_METERS = 1.0
NEAR_DUPLICATE_DISTANCE_METERS = 1.0
EXPLOSIVE_SPEED_METERS_PER_SECOND = 100.0

# These gates are fixed before validation is opened.  They are deliberately
# minimal non-degeneracy/calibration requirements rather than tuned targets.
MAX_SELECTED_MODE_FREQUENCY = 0.98
MIN_EFFECTIVE_MODE_COUNT = 1.25
MIN_PAIRWISE_MODE_DISTANCE_METERS = 1.0
MAX_DUPLICATE_MODE_QUERY_RATE = 0.95
MAX_ABSOLUTE_COVERAGE_ERROR = 0.20
MAX_PRIMARY_OUT_OF_BOUNDS_ROUTE_RATE = 0.01
MAX_PRIMARY_ZERO_MOTION_ROUTE_RATE = 0.95
MAX_PRIMARY_EXPLOSIVE_MOTION_ROUTE_RATE = 0.01
MAX_PRIMARY_NEAR_DUPLICATE_ROUTE_RATE = 0.95


class LockedEvaluationError(RuntimeError):
    """Raised whenever a locked integrity, leakage, or evaluation gate fails."""


@dataclasses.dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: ParallelTrajectoryModel
    state: TrainingState
    best_validation_route_nll: float
    transfer_proof: Mapping[str, Any]
    optimizer_contract: Mapping[str, Any]
    scheduler_contract: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class PartitionArrays:
    partition: str
    query_manifest: tuple[dict[str, Any], ...]
    session_ids: np.ndarray
    phases: np.ndarray
    current_xy_uu: np.ndarray
    truth_xy_uu: np.ndarray
    target_displacements: np.ndarray
    target_mask: np.ndarray
    primary_xy_uu: np.ndarray
    secondary_xy_uu: np.ndarray
    static_xy_uu: np.ndarray
    constant_velocity_xy_uu: np.ndarray
    mode_xy_uu: np.ndarray
    mode_logits: np.ndarray
    mode_probabilities: np.ndarray
    means_normalized: np.ndarray
    scales_normalized: np.ndarray
    correlations: np.ndarray
    route_nll: np.ndarray
    constant_velocity_fallbacks: tuple[str | None, ...]
    causal_input_hashes: tuple[str, ...]


def canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise ValueError(f"unsealed payload must omit {field}")
    result = dict(payload)
    result[field] = payload_sha256(result)
    return result


def verify_self_seal(payload: Mapping[str, Any], field: str) -> None:
    actual = payload.get(field)
    if not isinstance(actual, str) or len(actual) != 64:
        raise LockedEvaluationError(f"{field} is missing or invalid")
    unsealed = dict(payload)
    del unsealed[field]
    if payload_sha256(unsealed) != actual:
        raise LockedEvaluationError(f"{field} mismatch")


def _atomic_write(path: Path, data: bytes, *, require_absent: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if require_absent and path.exists():
        raise LockedEvaluationError(f"refusing to overwrite immutable artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_new_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value, pretty=True), require_absent=True)


def replace_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value, pretty=True), require_absent=False)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockedEvaluationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise LockedEvaluationError(f"{label} must be a JSON object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise LockedEvaluationError(f"cannot read {label}: {exc}") from exc
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LockedEvaluationError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numbers_finite(item) for item in value)
    return False


def _tensor_finite(value: Tensor) -> bool:
    return bool(
        (not value.is_floating_point() and not value.is_complex())
        or torch.isfinite(value).all()
    )


def _tensor_content_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_json(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    ).encode("utf-8")
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _output_sha256(output: ParallelTrajectoryOutput) -> str:
    payload = []
    for field in dataclasses.fields(output):
        value = getattr(output, field.name)
        if isinstance(value, Tensor):
            payload.append(
                {
                    "name": field.name,
                    "sha256": _tensor_content_sha256(value),
                }
            )
        else:
            payload.append({"name": field.name, "value": value})
    return payload_sha256(payload)


def _checkpoint_tensor_report(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise LockedEvaluationError(f"cannot inspect checkpoint {path}: {exc}") from exc
    require(isinstance(checkpoint, Mapping), f"checkpoint is not a mapping: {path}")
    model_state = checkpoint.get("model_state")
    optimizer_state = checkpoint.get("optimizer_state")
    require(isinstance(model_state, Mapping), "checkpoint model_state is missing")
    tensor_rows: list[dict[str, Any]] = []
    for name, tensor in sorted(model_state.items()):
        require(isinstance(name, str) and isinstance(tensor, Tensor), "invalid model state")
        require(_tensor_finite(tensor), f"nonfinite model tensor {name} in {path.name}")
        tensor_rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _tensor_content_sha256(tensor),
            }
        )

    optimizer_tensors: list[Tensor] = []

    def collect(value: Any) -> None:
        if isinstance(value, Tensor):
            optimizer_tensors.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(optimizer_state)
    require(
        all(_tensor_finite(value) for value in optimizer_tensors),
        f"nonfinite optimizer tensor in {path.name}",
    )
    return {
        "path": relative(path),
        "file_sha256": file_sha256(path),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "training_state": checkpoint.get("training_state"),
        "best_validation_route_nll": checkpoint.get("best_validation_route_nll"),
        "model_tensor_count": len(tensor_rows),
        "model_state_sha256": payload_sha256(tensor_rows),
        "all_model_tensors_finite": True,
        "optimizer_tensor_count": len(optimizer_tensors),
        "all_optimizer_tensors_finite": True,
        "bindings_sha256": payload_sha256(checkpoint.get("bindings")),
        "transfer_proof_sha256": payload_sha256(checkpoint.get("transfer_proof")),
    }


def _partition_map(setup: VerifiedTrainingSetup) -> dict[str, str]:
    result: dict[str, str] = {}
    for partition, values in setup.split_session_ids.items():
        for session_id in values:
            require(session_id not in result, "session occurs in multiple partitions")
            result[session_id] = partition
    require(set(result) == set(setup.session_paths), "split and session paths differ")
    return result


def load_selected_checkpoint(
    setup: VerifiedTrainingSetup,
    spec: PrecisionSpec,
    *,
    path: Path | None = None,
) -> LoadedCheckpoint:
    selected = RUN_DIRECTORY / "best.pt" if path is None else path
    model, transfer_proof = initialize_model(setup, device=None)
    optimizer, optimizer_contract = build_adamw_optimizer(model, setup.config)
    scheduler = PiecewiseLinearScheduler(optimizer, setup.config)
    state, best, amp_state = load_training_checkpoint(
        selected,
        setup=setup,
        expected_transfer_proof=transfer_proof,
        model=model,
        optimizer=optimizer,
        optimizer_contract=optimizer_contract,
        scheduler=scheduler,
        metrics=None,
        restore_rng=False,
    )
    require(amp_state is None, "BF16 checkpoint unexpectedly has AMP scaler state")
    require(best is not None and math.isfinite(best), "checkpoint best metric is invalid")
    require(
        all(_tensor_finite(value) for value in model.state_dict().values()),
        "strictly loaded model contains nonfinite tensors",
    )
    model.to(spec.device)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        state=state,
        best_validation_route_nll=float(best),
        transfer_proof=transfer_proof.to_dict(),
        optimizer_contract=optimizer_contract.to_dict(),
        scheduler_contract=scheduler.contract(),
    )


def _move_inputs(
    batch: EncoderBatch,
    observation: PlannerObservation,
    spec: PrecisionSpec,
) -> tuple[EncoderBatch, PlannerObservation]:
    return batch.to(spec.device), observation.to(spec.device)


def _query_dict(query: PlannerQuery) -> dict[str, Any]:
    return {
        "session_id": query.session_id,
        "team_id": query.team_id,
        "absolute_tick_index": query.absolute_tick_index,
    }


def _query_manifest_sha256(queries: Sequence[PlannerQuery]) -> str:
    return payload_sha256([_query_dict(query) for query in queries])


def _initial_ledger(setup: VerifiedTrainingSetup) -> dict[str, Any]:
    return {
        "schema_version": ACCESS_LEDGER_SCHEMA,
        "split_manifest_sha256": setup.split_manifest_sha256,
        "partition_session_ids": {
            name: list(values) for name, values in setup.split_session_ids.items()
        },
        "events": [],
        "counts": {
            name: {"attempts": 0, "opens": 0}
            for name in ("train", "validation", "test")
        },
        "test_authorized": False,
        "test_evaluator_invocation_count": 0,
        "validation_evaluator_invocation_count": 0,
        "validation_contract_precedes_all_validation_access": True,
        "test_access_preconditions_satisfied": None,
    }


def _load_ledger(evaluation_directory: Path) -> dict[str, Any]:
    ledger = read_json(evaluation_directory / "data_access_ledger.json", "access ledger")
    require(ledger.get("schema_version") == ACCESS_LEDGER_SCHEMA, "ledger schema changed")
    require(isinstance(ledger.get("events"), list), "ledger events are malformed")
    require(isinstance(ledger.get("counts"), dict), "ledger counts are malformed")
    return ledger


def _record_access(
    evaluation_directory: Path,
    *,
    partition: str,
    session_id: str,
    action: str,
    purpose: str,
    invocation: int,
) -> None:
    require(partition in {"train", "validation", "test"}, "invalid partition")
    require(action in {"attempt", "open"}, "invalid access action")
    ledger = _load_ledger(evaluation_directory)
    expected_ids = ledger["partition_session_ids"].get(partition)
    require(isinstance(expected_ids, list) and session_id in expected_ids, "split violation")
    if partition == "validation":
        contract_path = evaluation_directory / "evaluation_contract.json"
        require(contract_path.is_file(), "validation attempted before contract publication")
    if partition == "test":
        require(ledger.get("test_authorized") is True, "sealed test access is not authorized")
        require(
            int(ledger.get("test_evaluator_invocation_count", 0)) == 1,
            "test evaluation invocation count is not exactly one",
        )
    events = ledger["events"]
    events.append(
        {
            "sequence": len(events) + 1,
            "partition": partition,
            "session_id": session_id,
            "action": action,
            "purpose": purpose,
            "evaluator_invocation": invocation,
        }
    )
    counts = ledger["counts"][partition]
    counts["attempts" if action == "attempt" else "opens"] += 1
    replace_json(evaluation_directory / "data_access_ledger.json", ledger)


def _open_source(
    setup: VerifiedTrainingSetup,
    evaluation_directory: Path,
    *,
    partition: str,
    session_id: str,
    purpose: str,
    invocation: int,
) -> PlannerTargetSource:
    require(session_id in setup.split_session_ids[partition], "session split violation")
    _record_access(
        evaluation_directory,
        partition=partition,
        session_id=session_id,
        action="attempt",
        purpose=purpose,
        invocation=invocation,
    )
    path = setup.session_paths[session_id]
    match = load_match_session(path)
    require(match.session_id == session_id, "opened session identity changed")
    source = load_planner_target_source(path, match, setup.profile)
    _record_access(
        evaluation_directory,
        partition=partition,
        session_id=session_id,
        action="open",
        purpose=purpose,
        invocation=invocation,
    )
    return source


def _fixed_queries(
    sources: Mapping[str, PlannerTargetSource],
    session_ids: Sequence[str],
) -> tuple[PlannerQuery, ...]:
    selected: list[PlannerQuery] = []
    for session_id in sorted(session_ids):
        eligible = eligible_parallel_trajectory_queries(sources[session_id])
        sample = uniform_sample(
            eligible,
            QUERIES_PER_SESSION,
            seed=stable_seed(
                EVALUATION_SEED,
                QUERY_SAMPLER_VERSION,
                session_id,
            ),
        )
        require(
            len(sample) == QUERIES_PER_SESSION,
            f"session {session_id} has insufficient eligible queries",
        )
        selected.extend(sample)
    queries = tuple(selected)
    require(len(set(queries)) == len(queries), "query census contains duplicates")
    return queries


def _training_determinism_check(
    setup: VerifiedTrainingSetup,
    spec: PrecisionSpec,
    evaluation_directory: Path,
) -> dict[str, Any]:
    session_id = sorted(setup.split_session_ids["train"])[0]
    source = _open_source(
        setup,
        evaluation_directory,
        partition="train",
        session_id=session_id,
        purpose="selected_checkpoint_fixed_training_only_batch",
        invocation=1,
    )
    eligible = eligible_parallel_trajectory_queries(source)
    queries = uniform_sample(
        eligible,
        4,
        seed=stable_seed(
            EVALUATION_SEED,
            "parallel-trajectory-checkpoint-determinism-batch-v1",
            session_id,
        ),
    )
    require(len(queries) == 4, "fixed training session has insufficient queries")
    examples = prepare_parallel_trajectory_query_examples(
        {session_id: source},
        queries,
        context_length_ticks=setup.config.context_length_ticks,
    )
    require(len(examples) == len(queries), "fixed training query preparation changed")
    for example in examples:
        require(
            int(example.match.absolute_tick_index[-1].item())
            == example.query.absolute_tick_index,
            "causal training slice extends beyond its query",
        )
    batch, observation, _targets = collate_parallel_trajectory_query_examples(examples)
    batch, observation = _move_inputs(batch, observation, spec)

    output_hashes: list[str] = []
    model_state_hashes: list[str] = []
    for _repeat in range(2):
        loaded = load_selected_checkpoint(setup, spec)
        model_state_hashes.append(module_state_digest(loaded.model))
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=spec.autocast_dtype, enabled=True
        ):
            # Targets are intentionally absent from the model call.
            output = loaded.model(batch, observation, None)
        require(bool(output.query_mask.all()), "training-only inference lost a query")
        output_hashes.append(_output_sha256(output))
        del loaded, output
        torch.cuda.empty_cache()
    require(len(set(model_state_hashes)) == 1, "strict reload model states differ")
    require(len(set(output_hashes)) == 1, "repeated inference is not bit-identical")
    return {
        "partition": "train",
        "session_id": session_id,
        "query_manifest": [_query_dict(query) for query in queries],
        "query_manifest_sha256": _query_manifest_sha256(queries),
        "query_count": len(queries),
        "context_length_ticks": setup.config.context_length_ticks,
        "causal_slice_last_tick_equals_query_tick": True,
        "future_inputs_passed_to_model": False,
        "strict_reload_count": 2,
        "strict_reload_model_state_sha256": model_state_hashes[0],
        "repeated_output_sha256": output_hashes[0],
        "bit_identical_repeated_inference": True,
    }


def _training_integrity_and_selection(
    setup: VerifiedTrainingSetup,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime = read_json(RUN_DIRECTORY / "runtime.json", "final runtime")
    summary = read_json(RUN_DIRECTORY / "final_summary.json", "training summary")
    latest = read_json(RUN_DIRECTORY / "latest.json", "last checkpoint metadata")
    best = read_json(RUN_DIRECTORY / "best.json", "best checkpoint metadata")
    parent_latest = read_json(PARENT_RUN / "latest.json", "parent checkpoint metadata")
    recovery_parent = read_json(RUN_DIRECTORY / "recovery_parent.json", "recovery parent")
    access_lineage = read_json(RUN_DIRECTORY / "data_access_lineage.json", "access lineage")
    continuation_access = read_json(RUN_DIRECTORY / "data_access.json", "continuation access")
    parent_access = read_json(PARENT_RUN / "data_access.json", "parent access")

    require(runtime.get("status") == "completed", "training runtime is not completed")
    require(summary.get("status") == "completed", "training summary is not completed")
    for payload in (runtime, summary, latest):
        require(int(payload.get("optimizer_step", payload.get("optimizer_steps", -1))) == 860, "optimizer step count is not 860")
        require(int(payload.get("completed_epochs", -1)) == 20, "completed epoch count is not 20")
    require(int(latest.get("current_epoch", -1)) == 21, "final epoch cursor is not 21")
    require(int(latest.get("next_batch_index", -1)) == 0, "final batch cursor is not zero")
    require(int(parent_latest.get("optimizer_step", -1)) == 43, "recovery parent is not committed at step 43")
    require(
        recovery_parent.get("parent_checkpoint_sha256") == parent_latest.get("checkpoint_sha256"),
        "recovery parent checkpoint binding changed",
    )

    stderr_paths = (
        PARENT_RUN.with_suffix(".stderr.log"),
        RUN_DIRECTORY.with_suffix(".stderr.log"),
    )
    # with_suffix does not model the repository's sibling-log naming for a directory.
    stderr_paths = (
        REPOSITORY_ROOT / "runs" / f"{PARENT_RUN.name}.stderr.log",
        REPOSITORY_ROOT / "runs" / f"{RUN_DIRECTORY.name}.stderr.log",
    )
    for path in stderr_paths:
        require(path.is_file(), f"training stderr is missing: {path}")
        require(path.stat().st_size == 0, f"training stderr is nonempty: {path}")

    for payload in (parent_access, continuation_access):
        require(payload.get("test_session_parquet_open_attempt_count") == 0, "training test attempt detected")
        require(payload.get("test_session_parquet_open_count") == 0, "training test open detected")
        require(payload.get("zero_test_session_parquet_attempts") is True, "training test-attempt seal failed")
        require(payload.get("zero_test_session_parquet_opens") is True, "training test-open seal failed")
    require(access_lineage["historical_parent"]["test_attempt_count"] == 0, "historical test attempt detected")
    require(access_lineage["historical_parent"]["test_open_count"] == 0, "historical test open detected")

    parent_records = read_jsonl(PARENT_RUN / "metrics.jsonl", "parent metrics")
    continuation_records = read_jsonl(RUN_DIRECTORY / "metrics.jsonl", "continuation metrics")
    require(parent_records and continuation_records, "training metrics are empty")
    require(
        all(_all_numbers_finite(row) for row in parent_records + continuation_records),
        "a recorded training numeric value is nonfinite",
    )
    parent_steps = [
        int(row["optimizer_step"])
        for row in parent_records
        if row.get("type") == "optimizer_step"
    ]
    continuation_steps = [
        int(row["optimizer_step"])
        for row in continuation_records
        if row.get("type") == "optimizer_step"
    ]
    require(parent_steps == list(range(1, 54)), "parent recorded optimizer steps changed")
    require(continuation_steps == list(range(44, 861)), "continuation optimizer steps are not contiguous 44..860")

    selection_rows = [
        row
        for row in parent_records + continuation_records
        if row.get("type") == "validation_epoch"
        and 1 <= int(row.get("completed_epoch", -1)) <= 20
    ]
    by_epoch: dict[int, dict[str, Any]] = {}
    for row in selection_rows:
        epoch = int(row["completed_epoch"])
        require(epoch not in by_epoch, f"duplicate validation record for epoch {epoch}")
        require(row.get("selection_metric") == "validation_route_mixture_nll", "selection metric changed")
        value = float(row["selection_value"])
        require(math.isfinite(value), "validation selection value is nonfinite")
        require(value == float(row["metrics"]["route_nll"]), "selection value does not reproduce route NLL")
        by_epoch[epoch] = row
    require(sorted(by_epoch) == list(range(1, 21)), "validation candidates do not cover epochs 1..20")
    minimum_epoch = min(by_epoch, key=lambda epoch: float(by_epoch[epoch]["selection_value"]))
    minimum_value = float(by_epoch[minimum_epoch]["selection_value"])
    require(minimum_epoch == int(best["completed_epoch"]), "best.json epoch is not the criterion minimum")
    require(minimum_value == float(best["value"]), "best.json metric is not reproducible")
    require(best.get("selection_metric") == "validation_route_mixture_nll", "best.json selection metric changed")
    require(best.get("oracle_metrics_used_for_selection") is False, "oracle metric was used for selection")
    require(file_sha256(RUN_DIRECTORY / "best.pt") == best["checkpoint_sha256"], "best checkpoint hash changed")
    require(file_sha256(RUN_DIRECTORY / "last.pt") == latest["checkpoint_sha256"], "last checkpoint hash changed")

    checkpoint_reports = [
        _checkpoint_tensor_report(RUN_DIRECTORY / "best.pt"),
        _checkpoint_tensor_report(RUN_DIRECTORY / "last.pt"),
    ]
    require(
        checkpoint_reports[0]["training_state"]
        == {
            "completed_epochs": 20,
            "optimizer_step": 860,
            "current_epoch": 21,
            "next_batch_index": 0,
            "first_optimizer_step_completed": True,
        },
        "selected checkpoint training state is not final",
    )
    require(
        checkpoint_reports[0]["model_state_sha256"]
        == checkpoint_reports[1]["model_state_sha256"],
        "best and last final model tensors differ",
    )

    candidates = [
        {
            "completed_epoch": epoch,
            "optimizer_step": int(by_epoch[epoch]["optimizer_step"]),
            "selection_metric": "validation_route_mixture_nll",
            "selection_value": float(by_epoch[epoch]["selection_value"]),
            "improved_at_time": bool(by_epoch[epoch]["improved"]),
            "selected": epoch == minimum_epoch,
        }
        for epoch in range(1, 21)
    ]
    integrity = {
        "status": "passed",
        "exact_resume_lineage": True,
        "parent_run_directory": relative(PARENT_RUN),
        "continuation_run_directory": relative(RUN_DIRECTORY),
        "expected_optimizer_steps": 860,
        "observed_optimizer_steps": 860,
        "expected_epochs": 20,
        "observed_epochs": 20,
        "parent_committed_optimizer_step": 43,
        "continuation_step_range": [44, 860],
        "historical_uncommitted_parent_step_range": [44, 53],
        "all_recorded_numeric_values_finite": True,
        "parent_metric_record_count": len(parent_records),
        "continuation_metric_record_count": len(continuation_records),
        "stderr_files": [
            {"path": relative(path), "sha256": file_sha256(path), "bytes": 0}
            for path in stderr_paths
        ],
        "training_test_access_attempt_count": 0,
        "training_test_access_open_count": 0,
        "checkpoint_tensor_reports": checkpoint_reports,
        "provenance_verification": {
            "verify_training_setup_passed": True,
            "architecture_id": setup.checkpoint_bindings()["architecture_id"],
            "architecture_package_digest": setup.architecture_package_digest,
            "training_package_digest": setup.training_package_digest,
            "configuration_raw_sha256": file_sha256(CONFIG_PATH),
            "configuration_payload_sha256": payload_sha256(setup.config.to_dict()),
            "split_manifest_sha256": setup.split_manifest_sha256,
            "split_manifest_raw_sha256": file_sha256(Path(setup.config.split_manifest_path)),
            "world_grid_profile_hash": setup.profile.profile_hash,
            "world_grid_profile_raw_sha256": file_sha256(Path(setup.config.world_grid_profile_path)),
            "encoder_lineage_report_sha256": setup.lineage_report_sha256,
            "source_checkpoint_sha256": setup.transfer.source_checkpoint_sha256,
            "transferred_tensor_manifest_sha256": setup.transfer.transferred_tensor_manifest_sha256,
            "checkpoint_bindings_sha256": payload_sha256(setup.checkpoint_bindings()),
        },
    }
    return integrity, candidates


def _environment_payload(spec: PrecisionSpec) -> dict[str, Any]:
    base = environment_metadata(spec)
    try:
        pytest_version = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        pytest_version = None
    return {
        "schema_version": "parallel-trajectory-evaluation-environment:1.0",
        **base,
        "pytest": pytest_version,
        "evaluator_source_path": relative(Path(__file__)),
        "evaluator_source_sha256": file_sha256(Path(__file__)),
        "platform_release": platform.release(),
    }


def _evaluation_contract(
    setup: VerifiedTrainingSetup,
    evaluation_directory: Path,
    checkpoint_selection_sha256: str,
) -> dict[str, Any]:
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "evaluation_directory": relative(evaluation_directory),
        "selected_checkpoint": {
            "path": relative(RUN_DIRECTORY / "best.pt"),
            "sha256": file_sha256(RUN_DIRECTORY / "best.pt"),
            "checkpoint_selection_file_sha256": checkpoint_selection_sha256,
        },
        "frozen_sources": {
            "evaluator_path": relative(Path(__file__)),
            "evaluator_sha256": file_sha256(Path(__file__)),
            "architecture_package_digest": setup.architecture_package_digest,
            "training_package_digest": setup.training_package_digest,
            "model_source_sha256": file_sha256(
                SOURCE_ROOT / "fortnite_parallel_trajectory" / "model.py"
            ),
            "input_policy_source_sha256": file_sha256(
                SOURCE_ROOT / "fortnite_encoder" / "planner_policy.py"
            ),
            "target_source_sha256": file_sha256(
                SOURCE_ROOT / "fortnite_parallel_trajectory" / "targets.py"
            ),
        },
        "partitions": {
            "execution_order": ["validation", "test_if_authorized"],
            "validation_session_ids": list(setup.split_session_ids["validation"]),
            "validation_session_count": 11,
            "test_session_ids": list(setup.split_session_ids["test"]),
            "test_session_count": 10,
            "test_initially_sealed": True,
            "test_evaluation_maximum_invocations": 1,
        },
        "query_census": {
            "seed": EVALUATION_SEED,
            "sampler_version": QUERY_SAMPLER_VERSION,
            "algorithm": (
                "For each lexicographically sorted session, enumerate eligible "
                "phase-1..8 parallel-trajectory queries in canonical order and take "
                "the first 32 indices of torch.randperm seeded by stable_seed(seed, "
                "sampler_version, session_id), without replacement."
            ),
            "queries_per_session": QUERIES_PER_SESSION,
            "validation_query_count": 11 * QUERIES_PER_SESSION,
            "test_query_count_if_authorized": 10 * QUERIES_PER_SESSION,
            "manifest_rule": "ordered exact identities are published with each partition result",
        },
        "forecast_definitions": {
            "horizons_seconds": list(HORIZONS_SECONDS),
            "primary": "k*=argmax_k pi[r,k] with first-index tie break; y_hat[r,h]=mu[r,k*,h]",
            "secondary": "y_bar[r,h]=sum_k pi[r,k]*mu[r,k,h]",
            "static": "repeat the current query-time own-team centroid at every horizon",
            "constant_velocity": (
                "Use the query-time centroid and latest earlier valid causal centroid "
                "with identical living-player membership and life indices; divide by "
                "actual elapsed seconds and extrapolate from the query-time centroid. "
                "Fall back to static when the earlier observation is unavailable."
            ),
            "oracle": "best-of-five minADE/minFDE diagnostic only",
            "no_coordinate_clipping": True,
        },
        "free_running_and_leakage_policy": {
            "observation_provider_id": OBSERVATION_PROVIDER_ID,
            "causal_context_length_ticks": setup.config.context_length_ticks,
            "full_match_used_only_to_construct_scoring_targets_before_causal_slice": True,
            "model_forward_targets_argument": None,
            "future_waypoints_passed_to_model": False,
            "future_coordinates_passed_to_model": False,
            "teacher_forcing": False,
            "constant_velocity_post_query_access": False,
        },
        "numerics": {
            "model_device": "cuda:0",
            "model_precision": "bf16 autocast",
            "likelihood_precision": "torch FP32, matching frozen route_mixture_nll",
            "geometry_and_aggregation_precision": "IEEE-754 binary64",
            "canonical_json": "UTF-8, sorted keys, compact separators, no NaN, trailing LF",
            "timestamps_in_canonical_partition_payloads": False,
        },
        "accuracy_metrics": {
            "distance_units": "meters",
            "ade": (
                "For each route, mean Euclidean error over its valid horizons; "
                "then equal-weight mean over valid routes."
            ),
            "fde": (
                "For each route, Euclidean error at that route's last valid horizon; "
                "then equal-weight mean over valid routes."
            ),
            "horizon_ade": (
                "At horizon H, restrict to routes valid exactly at H, compute each "
                "route's mean error over valid horizons through H, then route-equal mean."
            ),
            "horizon_fde": "At horizon H, route-equal mean exact-H Euclidean error.",
            "global_horizon_step_weighting_forbidden": True,
            "zone_phase_groups": list(range(1, 9)),
            "stationary_regime": {
                "definition": (
                    "Ground-truth cumulative valid-segment path length <=10m and "
                    "maximum valid radius from query position <=10m."
                ),
                "path_length_threshold_meters": STATIONARY_PATH_LENGTH_METERS,
                "maximum_radius_threshold_meters": STATIONARY_MAX_RADIUS_METERS,
            },
            "group_support": {
                "minimum_routes": MIN_GROUP_ROUTES,
                "minimum_sessions": MIN_GROUP_SESSIONS,
            },
        },
        "paired_bootstrap": {
            "cluster": "session_id",
            "paired_unit": "query route",
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "seed_rule": "stable_seed(EVALUATION_SEED,'session-clustered-bootstrap-v1',comparison_label)",
            "rng": "numpy PCG64",
            "interval": "2.5th and 97.5th linear empirical quantiles",
            "difference": "model minus baseline; negative favors model",
            "improvement_supported": "upper confidence bound is strictly below zero",
        },
        "route_coherence": {
            "segments": "query position followed by twelve 5-second forecast points",
            "speed": "segment Euclidean distance / 5 seconds",
            "acceleration": "Euclidean difference of consecutive velocity vectors / 5 seconds",
            "turning_angle": "angle between consecutive segments when both exceed 0.1m",
            "stationary_path_length_threshold_meters": STATIONARY_PATH_LENGTH_METERS,
            "zero_motion_path_length_threshold_meters": ZERO_MOTION_PATH_LENGTH_METERS,
            "explosive_speed_threshold_meters_per_second": EXPLOSIVE_SPEED_METERS_PER_SECOND,
            "cross_query_near_duplicate_rms_threshold_meters": NEAR_DUPLICATE_DISTANCE_METERS,
            "bounds": "locked model-world inclusive rectangle; no clamping",
        },
        "mixture_diagnostics": {
            "mode_count": NUM_ROUTE_MODES,
            "entropy": "-sum_k pi_k log(pi_k) in nats",
            "effective_mode_count": "exp(entropy) per query, then mean",
            "pairwise_distance": "mean over all ten mode pairs and all twelve horizon distances",
            "duplicate_mode": "pairwise complete-route RMS distance <=1m",
            "one_mode_dominates": f"maximum selected-mode frequency >= {MAX_SELECTED_MODE_FREQUENCY}",
            "oracle_use": "diagnostic only; forbidden for selection and superiority",
        },
        "probabilistic_calibration": {
            "predictive_samples_per_query_per_set": PREDICTIVE_SAMPLES,
            "sampling_seed_rule": "stable_seed(EVALUATION_SEED,'predictive-calibration-v1',partition,query_identity)",
            "coherent_route_mode_sampling": True,
            "predictive_region": (
                "Sample-based marginal highest-density region at each horizon: "
                "threshold the exact mixture log density of deterministic samples at "
                "the (1-nominal) empirical quantile and test target density against it."
            ),
            "coverage_levels": list(PREDICTIVE_LEVELS),
            "coverage_aggregation": "mean per-route valid-horizon coverage, then route-equal mean",
            "energy_score": (
                "Two independent deterministic coherent route sample sets; mean ||X-y|| "
                "minus 0.5*mean ||X-X'||. Route vector norms are divided by sqrt(valid horizons)."
            ),
            "route_nll": "frozen route_mixture_nll: sum horizon log density within mode, then one mode logsumexp",
            "per_horizon_nll": "horizon marginal mixture NLL, route-equal at each horizon",
        },
        "validation_eligibility_gate": {
            "integrity_leakage_collapse_reproducibility_must_pass": True,
            "primary_must_have_strictly_lower_ade_and_fde_than_each_baseline": True,
            "all_four_overall_primary_minus_baseline_ci_upper_bounds_must_be_below_zero": True,
            "maximum_selected_mode_frequency": MAX_SELECTED_MODE_FREQUENCY,
            "minimum_mean_effective_mode_count": MIN_EFFECTIVE_MODE_COUNT,
            "minimum_mean_pairwise_mode_distance_meters": MIN_PAIRWISE_MODE_DISTANCE_METERS,
            "maximum_duplicate_mode_query_rate": MAX_DUPLICATE_MODE_QUERY_RATE,
            "maximum_absolute_overall_coverage_error_each_level": MAX_ABSOLUTE_COVERAGE_ERROR,
            "maximum_primary_out_of_bounds_route_rate": MAX_PRIMARY_OUT_OF_BOUNDS_ROUTE_RATE,
            "maximum_primary_zero_motion_route_rate": MAX_PRIMARY_ZERO_MOTION_ROUTE_RATE,
            "maximum_primary_explosive_motion_route_rate": MAX_PRIMARY_EXPLOSIVE_MOTION_ROUTE_RATE,
            "maximum_primary_near_duplicate_route_rate": MAX_PRIMARY_NEAR_DUPLICATE_ROUTE_RATE,
            "nonfinite_output_rate": 0.0,
        },
        "test_claim_gate": {
            "requires_validation_eligibility": True,
            "test_evaluator_runs_exactly_once": True,
            "superiority_requires_strictly_lower_primary_ade_and_fde_than_both_baselines": True,
            "superiority_requires_all_four_paired_ci_upper_bounds_below_zero": True,
            "post_test_tuning_selection_or_calibration_forbidden": True,
        },
    }
    return self_seal(contract, "contract_payload_sha256")


def prepare(evaluation_directory: Path) -> dict[str, Any]:
    evaluation_directory = evaluation_directory.resolve()
    require(not evaluation_directory.exists(), f"evaluation directory already exists: {evaluation_directory}")
    setup = verify_training_setup(CONFIG_PATH)
    spec = resolve_precision()
    seed_everything(EVALUATION_SEED)
    integrity, candidates = _training_integrity_and_selection(setup)

    evaluation_directory.mkdir(parents=True, exist_ok=False)
    write_new_json(evaluation_directory / "data_access_ledger.json", _initial_ledger(setup))
    write_new_json(evaluation_directory / "environment.json", _environment_payload(spec))
    determinism = _training_determinism_check(setup, spec, evaluation_directory)
    loaded = load_selected_checkpoint(setup, spec)
    require(loaded.state.optimizer_step == 860, "strictly loaded checkpoint is not final")
    require(loaded.state.completed_epochs == 20, "strictly loaded checkpoint is not epoch 20")
    require(
        loaded.best_validation_route_nll == float(candidates[-1]["selection_value"]),
        "strictly loaded checkpoint best metric changed",
    )
    del loaded
    torch.cuda.empty_cache()

    checkpoint_payload = self_seal(
        {
            "schema_version": CHECKPOINT_SELECTION_SCHEMA,
            "training_integrity": integrity,
            "selection": {
                "criterion_source": relative(RUN_DIRECTORY / "resolved_config.json"),
                "selection_metric": "validation_route_mixture_nll",
                "direction": "minimize",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selected_checkpoint_path": relative(RUN_DIRECTORY / "best.pt"),
                "selected_checkpoint_sha256": file_sha256(RUN_DIRECTORY / "best.pt"),
                "selected_completed_epoch": 20,
                "selected_optimizer_step": 860,
                "selected_metric_value": float(candidates[-1]["selection_value"]),
                "test_results_used_for_selection": False,
                "oracle_metrics_used_for_selection": False,
                "last_checkpoint_selected_because_newest": False,
                "last_checkpoint_model_weights_bit_identical_to_selected": True,
            },
            "strict_reload_and_training_only_determinism": determinism,
            "locked_hashes": {
                "architecture_id": setup.checkpoint_bindings()["architecture_id"],
                "architecture_config_sha256": payload_sha256(setup.architecture_config),
                "configuration_raw_sha256": file_sha256(CONFIG_PATH),
                "configuration_payload_sha256": payload_sha256(setup.config.to_dict()),
                "architecture_package_digest": setup.architecture_package_digest,
                "training_package_digest": setup.training_package_digest,
                "split_manifest_sha256": setup.split_manifest_sha256,
                "world_grid_profile_hash": setup.profile.profile_hash,
                "world_grid_profile_raw_sha256": file_sha256(Path(setup.config.world_grid_profile_path)),
                "encoder_lineage_report_sha256": setup.lineage_report_sha256,
                "legacy_encoder_digest": setup.config.legacy_encoder_digest,
                "current_encoder_digest": setup.config.current_encoder_digest,
                "source_checkpoint_sha256": setup.transfer.source_checkpoint_sha256,
                "transferred_tensor_manifest_sha256": setup.transfer.transferred_tensor_manifest_sha256,
                "checkpoint_bindings_sha256": payload_sha256(setup.checkpoint_bindings()),
            },
            "test_access_at_checkpoint_lock": {
                "training_attempts": 0,
                "training_opens": 0,
                "evaluation_attempts": 0,
                "evaluation_opens": 0,
                "remains_sealed": True,
            },
        },
        "checkpoint_selection_payload_sha256",
    )
    checkpoint_path = evaluation_directory / "checkpoint_selection.json"
    write_new_json(checkpoint_path, checkpoint_payload)
    checkpoint_file_hash = file_sha256(checkpoint_path)
    contract = _evaluation_contract(setup, evaluation_directory, checkpoint_file_hash)
    write_new_json(evaluation_directory / "evaluation_contract.json", contract)

    return {
        "status": "prepared",
        "evaluation_directory": str(evaluation_directory),
        "checkpoint_selection_sha256": checkpoint_file_hash,
        "evaluation_contract_sha256": file_sha256(evaluation_directory / "evaluation_contract.json"),
        "validation_access_attempts": 0,
        "test_access_attempts": 0,
    }


def _query_indices(batch: PlannerBatch) -> list[tuple[int, int, int]]:
    indices = batch.query_mask.nonzero(as_tuple=False).cpu().tolist()
    batch_size = int(batch.encoder_batch.player_xyz_uu.shape[0])
    require(len(indices) == batch_size, "each evaluation item must have one query")
    require([row[0] for row in indices] == list(range(batch_size)), "query order changed")
    return [(int(b), int(t), int(n)) for b, t, n in indices]


def _valid_living_centroid(
    batch: PlannerBatch,
    batch_index: int,
    time_index: int,
    team_index: int,
) -> Tensor | None:
    encoder = batch.encoder_batch
    roster = encoder.player_slot_mask[batch_index, team_index]
    valid = (
        encoder.player_alive[batch_index, time_index, team_index]
        & encoder.player_coord_mask[batch_index, time_index, team_index]
        & roster
    )
    if not bool(valid.any()):
        return None
    coordinates = encoder.player_xyz_uu[
        batch_index, time_index, team_index, valid, :2
    ].detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(coordinates).all()):
        return None
    return coordinates.mean(dim=0)


def _same_lifecycle(
    batch: PlannerBatch,
    batch_index: int,
    earlier_time: int,
    query_time: int,
    team_index: int,
) -> bool:
    encoder = batch.encoder_batch
    roster = encoder.player_slot_mask[batch_index, team_index]
    earlier_living = encoder.player_alive[batch_index, earlier_time, team_index] & roster
    query_living = encoder.player_alive[batch_index, query_time, team_index] & roster
    return bool(
        torch.equal(earlier_living, query_living)
        and torch.equal(
            encoder.life_index[batch_index, earlier_time, team_index, roster],
            encoder.life_index[batch_index, query_time, team_index, roster],
        )
    )


def derive_locked_baselines(
    planner_batch: PlannerBatch,
) -> tuple[Tensor, Tensor, Tensor, tuple[str | None, ...]]:
    """Build static/CV forecasts using only the sanitized causal batch."""

    encoder = planner_batch.encoder_batch
    require(encoder.player_xyz_uu.device.type == "cpu", "baseline inputs must be CPU")
    horizons = torch.tensor(HORIZONS_SECONDS, dtype=torch.float64)
    current_rows: list[Tensor] = []
    static_rows: list[Tensor] = []
    velocity_rows: list[Tensor] = []
    fallbacks: list[str | None] = []
    for batch_index, query_time, team_index in _query_indices(planner_batch):
        current = _valid_living_centroid(
            planner_batch, batch_index, query_time, team_index
        )
        require(current is not None, "eligible query lacks a current centroid")
        current_rows.append(current)
        static = current.repeat(NUM_HORIZONS, 1)
        static_rows.append(static)

        earlier_xy: Tensor | None = None
        earlier_seconds: float | None = None
        for earlier_time in range(query_time - 1, -1, -1):
            if not bool(encoder.time_mask[batch_index, earlier_time]):
                break
            if not _same_lifecycle(
                planner_batch,
                batch_index,
                earlier_time,
                query_time,
                team_index,
            ):
                break
            candidate = _valid_living_centroid(
                planner_batch, batch_index, earlier_time, team_index
            )
            if candidate is not None:
                earlier_xy = candidate
                earlier_seconds = float(
                    encoder.match_elapsed_s[batch_index, earlier_time].item()
                )
                break
        query_seconds = float(
            encoder.match_elapsed_s[batch_index, query_time].item()
        )
        velocity: Tensor | None = None
        if earlier_xy is not None and earlier_seconds is not None:
            elapsed = query_seconds - earlier_seconds
            if math.isfinite(elapsed) and elapsed > 0.0:
                candidate_velocity = (current - earlier_xy) / elapsed
                if bool(torch.isfinite(candidate_velocity).all()):
                    velocity = candidate_velocity
        if velocity is None:
            velocity_rows.append(static.clone())
            fallbacks.append("second_causal_observation_unavailable")
        else:
            velocity_rows.append(current + horizons[:, None] * velocity)
            fallbacks.append(None)

    return (
        torch.stack(current_rows).to(torch.float64),
        torch.stack(static_rows).to(torch.float64),
        torch.stack(velocity_rows).to(torch.float64),
        tuple(fallbacks),
    )


def _causal_input_sha256(batch: EncoderBatch, observation: PlannerObservation) -> str:
    rows: list[dict[str, Any]] = []
    for owner, value in (("encoder", batch), ("observation", observation)):
        for field in dataclasses.fields(value):
            tensor = getattr(value, field.name)
            require(isinstance(tensor, Tensor), "input contract contains a non-tensor field")
            rows.append(
                {
                    "owner": owner,
                    "name": field.name,
                    "sha256": _tensor_content_sha256(tensor),
                }
            )
    return payload_sha256(rows)


def _begin_partition_invocation(
    evaluation_directory: Path,
    partition: str,
) -> int:
    ledger = _load_ledger(evaluation_directory)
    if partition == "validation":
        count = int(ledger.get("validation_evaluator_invocation_count", 0)) + 1
        require(count <= 2, "validation evaluator may run exactly twice, not more")
        ledger["validation_evaluator_invocation_count"] = count
    elif partition == "test":
        count = int(ledger.get("test_evaluator_invocation_count", 0)) + 1
        require(count == 1, "test evaluator may run exactly once")
        require(ledger.get("test_authorized") is True, "test evaluation is not authorized")
        ledger["test_evaluator_invocation_count"] = count
    else:
        raise LockedEvaluationError("only validation and test are evaluator partitions")
    replace_json(evaluation_directory / "data_access_ledger.json", ledger)
    return count


def _verify_frozen_preconditions(
    setup: VerifiedTrainingSetup,
    evaluation_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = evaluation_directory / "checkpoint_selection.json"
    contract_path = evaluation_directory / "evaluation_contract.json"
    selection = read_json(checkpoint_path, "checkpoint selection")
    contract = read_json(contract_path, "evaluation contract")
    verify_self_seal(selection, "checkpoint_selection_payload_sha256")
    verify_self_seal(contract, "contract_payload_sha256")
    require(
        contract["selected_checkpoint"]["checkpoint_selection_file_sha256"]
        == file_sha256(checkpoint_path),
        "checkpoint selection changed after contract lock",
    )
    require(
        contract["frozen_sources"]["evaluator_sha256"] == file_sha256(Path(__file__)),
        "evaluator source changed after contract lock",
    )
    require(
        contract["selected_checkpoint"]["sha256"]
        == file_sha256(RUN_DIRECTORY / "best.pt"),
        "selected checkpoint changed after contract lock",
    )
    require(
        setup.architecture_package_digest
        == contract["frozen_sources"]["architecture_package_digest"],
        "architecture package changed after contract lock",
    )
    require(
        setup.training_package_digest
        == contract["frozen_sources"]["training_package_digest"],
        "training package changed after contract lock",
    )
    ledger = _load_ledger(evaluation_directory)
    require(ledger["counts"]["test"] == {"attempts": 0, "opens": 0}, "test access occurred before authorization")
    return selection, contract


def evaluate_partition_arrays(
    setup: VerifiedTrainingSetup,
    spec: PrecisionSpec,
    evaluation_directory: Path,
    *,
    partition: str,
) -> PartitionArrays:
    require(partition in {"validation", "test"}, "invalid evaluation partition")
    _verify_frozen_preconditions(setup, evaluation_directory)
    invocation = _begin_partition_invocation(evaluation_directory, partition)
    purpose = f"locked_{partition}_evaluation_repeat_{invocation}"
    session_ids = tuple(setup.split_session_ids[partition])
    sources = {
        session_id: _open_source(
            setup,
            evaluation_directory,
            partition=partition,
            session_id=session_id,
            purpose=purpose,
            invocation=invocation,
        )
        for session_id in sorted(session_ids)
    }
    queries = _fixed_queries(sources, session_ids)
    expected = len(session_ids) * QUERIES_PER_SESSION
    require(len(queries) == expected, "partition query census count changed")

    seed_everything(EVALUATION_SEED)
    loaded = load_selected_checkpoint(setup, spec)
    model = loaded.model
    width = float(setup.profile.cell_width_world_units)
    height = float(setup.profile.cell_height_world_units)
    displacement_scale = np.asarray([width, height], dtype=np.float64)

    manifests: list[dict[str, Any]] = []
    session_rows: list[str] = []
    phases: list[int] = []
    current_chunks: list[np.ndarray] = []
    truth_chunks: list[np.ndarray] = []
    target_displacement_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    primary_chunks: list[np.ndarray] = []
    secondary_chunks: list[np.ndarray] = []
    static_chunks: list[np.ndarray] = []
    velocity_chunks: list[np.ndarray] = []
    mode_world_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    mean_chunks: list[np.ndarray] = []
    scale_chunks: list[np.ndarray] = []
    correlation_chunks: list[np.ndarray] = []
    route_nll_chunks: list[np.ndarray] = []
    fallbacks: list[str | None] = []
    causal_hashes: list[str] = []

    for start in range(0, len(queries), MAX_QUERIES_PER_FORWARD):
        chunk_queries = queries[start : start + MAX_QUERIES_PER_FORWARD]
        examples = prepare_parallel_trajectory_query_examples(
            sources,
            chunk_queries,
            context_length_ticks=setup.config.context_length_ticks,
        )
        require(len(examples) == len(chunk_queries), "evaluation query lost supervision")
        for example in examples:
            require(
                int(example.match.absolute_tick_index[-1].item())
                == example.query.absolute_tick_index,
                "causal input extends beyond query time",
            )
        batch_cpu, observation_cpu, targets_cpu = (
            collate_parallel_trajectory_query_examples(examples)
        )
        causal_hashes.append(_causal_input_sha256(batch_cpu, observation_cpu))
        sanitized = apply_planner_input_policy(batch_cpu, observation_cpu)
        current, static, velocity, chunk_fallbacks = derive_locked_baselines(sanitized)
        require(targets_cpu.current_xy is not None, "target current coordinates are missing")
        require(targets_cpu.future_xy is not None, "target future coordinates are missing")
        require(
            torch.equal(current.to(torch.float32), targets_cpu.current_xy),
            "baseline and target current centroids differ",
        )
        batch_gpu, observation_gpu = _move_inputs(batch_cpu, observation_cpu, spec)
        targets_gpu = targets_cpu.to(spec.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=spec.autocast_dtype, enabled=True
        ):
            # Free-running by contract: no targets or future waypoints enter the model.
            output = model(batch_gpu, observation_gpu, None)
        require(torch.equal(output.query_mask, targets_gpu.query_mask), "output/target query masks differ")
        require(bool(output.query_mask.all()), "evaluation output contains an ineligible query")
        loss = route_mixture_nll(output, targets_gpu)
        require(loss.valid_route_count == len(chunk_queries), "route NLL lost a query")
        for name in ("mode_logits", "mode_probabilities", "means", "scales", "correlations"):
            require(bool(torch.isfinite(getattr(output, name)).all()), f"nonfinite output {name}")

        means = output.means.detach().float().cpu().numpy().astype(np.float64)
        probabilities = (
            output.mode_probabilities.detach().float().cpu().numpy().astype(np.float64)
        )
        logits = output.mode_logits.detach().float().cpu().numpy().astype(np.float64)
        scales = output.scales.detach().float().cpu().numpy().astype(np.float64)
        correlations = (
            output.correlations.detach().float().cpu().numpy().astype(np.float64)
        )
        current_np = current.numpy().astype(np.float64)
        mode_world = current_np[:, None, None, :] + means * displacement_scale
        selected_indices = np.argmax(probabilities, axis=1)
        rows = np.arange(len(chunk_queries))
        primary_world = mode_world[rows, selected_indices]
        secondary_normalized = np.sum(
            probabilities[:, :, None, None] * means, axis=1
        )
        secondary_world = current_np[:, None, :] + secondary_normalized * displacement_scale

        manifests.extend(_query_dict(query) for query in chunk_queries)
        session_rows.extend(query.session_id for query in chunk_queries)
        phases.extend(int(example.query_phase) for example in examples)
        current_chunks.append(current_np)
        truth_chunks.append(targets_cpu.future_xy.numpy().astype(np.float64))
        target_displacement_chunks.append(
            targets_cpu.target_displacements.numpy().astype(np.float64)
        )
        mask_chunks.append(targets_cpu.target_mask.numpy().astype(np.bool_))
        primary_chunks.append(primary_world)
        secondary_chunks.append(secondary_world)
        static_chunks.append(static.numpy().astype(np.float64))
        velocity_chunks.append(velocity.numpy().astype(np.float64))
        mode_world_chunks.append(mode_world)
        logits_chunks.append(logits)
        probability_chunks.append(probabilities)
        mean_chunks.append(means)
        scale_chunks.append(scales)
        correlation_chunks.append(correlations)
        route_nll_chunks.append(
            loss.per_route_nll.detach().float().cpu().numpy().astype(np.float64)
        )
        fallbacks.extend(chunk_fallbacks)

    del model, loaded
    torch.cuda.empty_cache()
    result = PartitionArrays(
        partition=partition,
        query_manifest=tuple(manifests),
        session_ids=np.asarray(session_rows, dtype=str),
        phases=np.asarray(phases, dtype=np.int64),
        current_xy_uu=np.concatenate(current_chunks, axis=0),
        truth_xy_uu=np.concatenate(truth_chunks, axis=0),
        target_displacements=np.concatenate(target_displacement_chunks, axis=0),
        target_mask=np.concatenate(mask_chunks, axis=0),
        primary_xy_uu=np.concatenate(primary_chunks, axis=0),
        secondary_xy_uu=np.concatenate(secondary_chunks, axis=0),
        static_xy_uu=np.concatenate(static_chunks, axis=0),
        constant_velocity_xy_uu=np.concatenate(velocity_chunks, axis=0),
        mode_xy_uu=np.concatenate(mode_world_chunks, axis=0),
        mode_logits=np.concatenate(logits_chunks, axis=0),
        mode_probabilities=np.concatenate(probability_chunks, axis=0),
        means_normalized=np.concatenate(mean_chunks, axis=0),
        scales_normalized=np.concatenate(scale_chunks, axis=0),
        correlations=np.concatenate(correlation_chunks, axis=0),
        route_nll=np.concatenate(route_nll_chunks, axis=0),
        constant_velocity_fallbacks=tuple(fallbacks),
        causal_input_hashes=tuple(causal_hashes),
    )
    require(result.target_mask.any(axis=1).all(), "a census route has no valid horizon")
    require(np.isfinite(result.route_nll).all(), "partition route NLL is nonfinite")
    return result


def distribution_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(array.max()),
    }


def _route_error_values(
    predicted_xy_uu: np.ndarray,
    truth_xy_uu: np.ndarray,
    mask: np.ndarray,
    meters_per_world_unit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = np.linalg.norm(predicted_xy_uu - truth_xy_uu, axis=-1) * meters_per_world_unit
    q = mask.shape[0]
    ade = np.full(q, np.nan, dtype=np.float64)
    fde = np.full(q, np.nan, dtype=np.float64)
    for row in range(q):
        valid = np.flatnonzero(mask[row])
        if valid.size:
            ade[row] = float(errors[row, valid].mean())
            fde[row] = float(errors[row, valid[-1]])
    return errors, ade, fde


def clustered_paired_bootstrap(
    differences: np.ndarray,
    session_ids: np.ndarray,
    *,
    label: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    sessions = np.asarray(session_ids, dtype=str)
    valid = np.isfinite(values)
    values = values[valid]
    sessions = sessions[valid]
    unique = np.unique(sessions)
    require(values.size > 0, f"bootstrap {label} has no paired routes")
    require(unique.size > 0, f"bootstrap {label} has no sessions")
    sums = np.asarray([values[sessions == name].sum() for name in unique], dtype=np.float64)
    counts = np.asarray([np.sum(sessions == name) for name in unique], dtype=np.float64)
    rng = np.random.Generator(
        np.random.PCG64(
            stable_seed(
                EVALUATION_SEED,
                "session-clustered-bootstrap-v1",
                label,
            )
        )
    )
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)))
    multiplicities = np.zeros((resamples, len(unique)), dtype=np.float64)
    for column in range(len(unique)):
        multiplicities[:, column] = np.sum(draws == column, axis=1)
    replicate = (multiplicities @ sums) / (multiplicities @ counts)
    lower, upper = np.quantile(replicate, [0.025, 0.975], method="linear")
    return {
        "difference_definition": "model_minus_baseline_meters",
        "point_difference_meters": float(values.mean()),
        "confidence": BOOTSTRAP_CONFIDENCE,
        "lower_meters": float(lower),
        "upper_meters": float(upper),
        "improvement_supported": bool(upper < 0.0),
        "resamples": int(resamples),
        "session_cluster_count": int(unique.size),
        "paired_route_count": int(values.size),
        "seed": stable_seed(
            EVALUATION_SEED,
            "session-clustered-bootstrap-v1",
            label,
        ),
    }


def _truth_regimes(
    arrays: PartitionArrays,
    meters_per_world_unit: float,
) -> np.ndarray:
    result: list[str] = []
    for row in range(len(arrays.query_manifest)):
        valid = np.flatnonzero(arrays.target_mask[row])
        points = np.concatenate(
            (
                arrays.current_xy_uu[row][None, :],
                arrays.truth_xy_uu[row, valid],
            ),
            axis=0,
        )
        path_length = float(
            np.linalg.norm(np.diff(points, axis=0), axis=-1).sum()
            * meters_per_world_unit
        )
        maximum_radius = float(
            np.linalg.norm(
                arrays.truth_xy_uu[row, valid] - arrays.current_xy_uu[row], axis=-1
            ).max()
            * meters_per_world_unit
        )
        result.append(
            "stationary"
            if path_length <= STATIONARY_PATH_LENGTH_METERS
            and maximum_radius <= STATIONARY_MAX_RADIUS_METERS
            else "moving"
        )
    return np.asarray(result, dtype=str)


def _method_accuracy(
    predicted: np.ndarray,
    arrays: PartitionArrays,
    meters_per_world_unit: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    errors, ade, fde = _route_error_values(
        predicted,
        arrays.truth_xy_uu,
        arrays.target_mask,
        meters_per_world_unit,
    )
    valid_routes = np.isfinite(ade)
    horizons: dict[str, Any] = {}
    for horizon_index, seconds in enumerate(HORIZONS_SECONDS):
        eligible = arrays.target_mask[:, horizon_index]
        cumulative = np.full(len(ade), np.nan, dtype=np.float64)
        for row in np.flatnonzero(eligible):
            earlier = arrays.target_mask[row, : horizon_index + 1]
            cumulative[row] = float(errors[row, : horizon_index + 1][earlier].mean())
        horizons[str(seconds)] = {
            "ade_meters": float(np.nanmean(cumulative)) if np.any(eligible) else None,
            "fde_meters": float(errors[eligible, horizon_index].mean()) if np.any(eligible) else None,
            "valid_route_count": int(eligible.sum()),
            "valid_horizon_count_for_ade": int(
                arrays.target_mask[eligible, : horizon_index + 1].sum()
            ),
        }
    return (
        {
            "overall": {
                "ade_meters": float(np.nanmean(ade)),
                "fde_meters": float(np.nanmean(fde)),
                "valid_route_count": int(valid_routes.sum()),
                "valid_horizon_count": int(arrays.target_mask[valid_routes].sum()),
            },
            "by_horizon_seconds": horizons,
        },
        errors,
        ade,
        fde,
    )


def _comparison_block(
    model_name: str,
    baseline_name: str,
    model_errors: np.ndarray,
    model_ade: np.ndarray,
    model_fde: np.ndarray,
    baseline_errors: np.ndarray,
    baseline_ade: np.ndarray,
    baseline_fde: np.ndarray,
    arrays: PartitionArrays,
    *,
    prefix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ade": clustered_paired_bootstrap(
            model_ade - baseline_ade,
            arrays.session_ids,
            label=f"{prefix}:{model_name}:{baseline_name}:overall:ade",
        ),
        "fde": clustered_paired_bootstrap(
            model_fde - baseline_fde,
            arrays.session_ids,
            label=f"{prefix}:{model_name}:{baseline_name}:overall:fde",
        ),
        "by_horizon_seconds": {},
    }
    for index, seconds in enumerate(HORIZONS_SECONDS):
        eligible = arrays.target_mask[:, index]
        model_cumulative = np.full(len(model_ade), np.nan, dtype=np.float64)
        baseline_cumulative = np.full(len(model_ade), np.nan, dtype=np.float64)
        for row in np.flatnonzero(eligible):
            earlier = arrays.target_mask[row, : index + 1]
            model_cumulative[row] = model_errors[row, : index + 1][earlier].mean()
            baseline_cumulative[row] = baseline_errors[row, : index + 1][earlier].mean()
        if np.any(eligible):
            result["by_horizon_seconds"][str(seconds)] = {
                "ade": clustered_paired_bootstrap(
                    model_cumulative[eligible] - baseline_cumulative[eligible],
                    arrays.session_ids[eligible],
                    label=f"{prefix}:{model_name}:{baseline_name}:{seconds}s:ade",
                ),
                "fde": clustered_paired_bootstrap(
                    model_errors[eligible, index] - baseline_errors[eligible, index],
                    arrays.session_ids[eligible],
                    label=f"{prefix}:{model_name}:{baseline_name}:{seconds}s:fde",
                ),
            }
        else:
            result["by_horizon_seconds"][str(seconds)] = None
    return result


def _group_accuracy(
    selector: np.ndarray,
    method_values: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    arrays: PartitionArrays,
    *,
    label: str,
) -> dict[str, Any]:
    route_count = int(selector.sum())
    session_count = int(np.unique(arrays.session_ids[selector]).size)
    supported = route_count >= MIN_GROUP_ROUTES and session_count >= MIN_GROUP_SESSIONS
    result: dict[str, Any] = {
        "route_count": route_count,
        "session_count": session_count,
        "supported": supported,
        "methods": {},
        "paired_primary_minus_baseline": None,
    }
    if not route_count:
        return result
    for name, (_errors, ade, fde) in method_values.items():
        result["methods"][name] = {
            "ade_meters": float(np.nanmean(ade[selector])),
            "fde_meters": float(np.nanmean(fde[selector])),
            "valid_route_count": int(np.isfinite(ade[selector]).sum()),
            "valid_horizon_count": int(arrays.target_mask[selector].sum()),
        }
    if supported:
        primary = method_values["primary"]
        comparisons: dict[str, Any] = {}
        for baseline in ("static", "constant_velocity"):
            candidate = method_values[baseline]
            comparisons[baseline] = {
                "ade": clustered_paired_bootstrap(
                    primary[1][selector] - candidate[1][selector],
                    arrays.session_ids[selector],
                    label=f"{arrays.partition}:{label}:primary:{baseline}:ade",
                ),
                "fde": clustered_paired_bootstrap(
                    primary[2][selector] - candidate[2][selector],
                    arrays.session_ids[selector],
                    label=f"{arrays.partition}:{label}:primary:{baseline}:fde",
                ),
            }
        result["paired_primary_minus_baseline"] = comparisons
    return result


def accuracy_report(
    arrays: PartitionArrays,
    meters_per_world_unit: float,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    predictions = {
        "primary": arrays.primary_xy_uu,
        "secondary": arrays.secondary_xy_uu,
        "static": arrays.static_xy_uu,
        "constant_velocity": arrays.constant_velocity_xy_uu,
    }
    methods: dict[str, Any] = {}
    values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, prediction in predictions.items():
        report, errors, ade, fde = _method_accuracy(
            prediction, arrays, meters_per_world_unit
        )
        methods[name] = report
        values[name] = (errors, ade, fde)

    paired: dict[str, Any] = {}
    for model_name in ("primary", "secondary"):
        paired[model_name] = {}
        for baseline_name in ("static", "constant_velocity"):
            paired[model_name][baseline_name] = _comparison_block(
                model_name,
                baseline_name,
                *values[model_name],
                *values[baseline_name],
                arrays,
                prefix=arrays.partition,
            )

    regimes = _truth_regimes(arrays, meters_per_world_unit)
    phase_groups = {
        str(phase): _group_accuracy(
            arrays.phases == phase,
            values,
            arrays,
            label=f"phase-{phase}",
        )
        for phase in range(1, 9)
    }
    regime_groups = {
        name: _group_accuracy(
            regimes == name,
            values,
            arrays,
            label=f"regime-{name}",
        )
        for name in ("stationary", "moving")
    }
    return (
        {
            "aggregation": "route_equal",
            "methods": methods,
            "paired_model_minus_baseline": paired,
            "by_zone_phase": phase_groups,
            "by_stationary_moving_regime": regime_groups,
            "constant_velocity": {
                "fallback_count": sum(value is not None for value in arrays.constant_velocity_fallbacks),
                "fallback_rate": float(
                    np.mean([value is not None for value in arrays.constant_velocity_fallbacks])
                ),
                "fallback_reason_counts": {
                    "second_causal_observation_unavailable": sum(
                        value == "second_causal_observation_unavailable"
                        for value in arrays.constant_velocity_fallbacks
                    )
                },
                "post_query_coordinates_used": False,
            },
        },
        values,
    )


def _cross_query_duplicate_rates(relative_routes_m: np.ndarray) -> dict[str, Any]:
    q = relative_routes_m.shape[0]
    if q < 2:
        return {
            "exact_duplicate_route_rate": 0.0,
            "near_identical_route_rate": 0.0,
            "exact_duplicate_route_count": 0,
            "near_identical_route_count": 0,
        }
    deltas = relative_routes_m[:, None, :, :] - relative_routes_m[None, :, :, :]
    rms = np.sqrt(np.mean(np.sum(deltas * deltas, axis=-1), axis=-1))
    np.fill_diagonal(rms, np.inf)
    exact = np.any(rms <= 1e-9, axis=1)
    near = np.any(rms <= NEAR_DUPLICATE_DISTANCE_METERS, axis=1)
    return {
        "exact_duplicate_route_rate": float(exact.mean()),
        "near_identical_route_rate": float(near.mean()),
        "exact_duplicate_route_count": int(exact.sum()),
        "near_identical_route_count": int(near.sum()),
    }


def route_coherence_for_method(
    routes_xy_uu: np.ndarray,
    current_xy_uu: np.ndarray,
    setup: VerifiedTrainingSetup,
) -> dict[str, Any]:
    meters_per_uu = float(setup.profile.world_unit_scale.meters_per_world_unit)
    routes = np.asarray(routes_xy_uu, dtype=np.float64)
    current = np.asarray(current_xy_uu, dtype=np.float64)
    q = routes.shape[0]
    finite_points = np.isfinite(routes).all(axis=-1)
    finite_routes = finite_points.all(axis=-1)
    points_m = np.concatenate((current[:, None, :], routes), axis=1) * meters_per_uu
    segments = np.diff(points_m, axis=1)
    segment_lengths = np.linalg.norm(segments, axis=-1)
    velocity_vectors = segments / 5.0
    speeds = segment_lengths / 5.0
    accelerations = np.linalg.norm(np.diff(velocity_vectors, axis=1), axis=-1) / 5.0
    turning_angles: list[float] = []
    for row in range(q):
        for left, right in zip(segments[row, :-1], segments[row, 1:]):
            left_norm = float(np.linalg.norm(left))
            right_norm = float(np.linalg.norm(right))
            if left_norm <= 0.1 or right_norm <= 0.1:
                continue
            cosine = float(np.dot(left, right) / (left_norm * right_norm))
            turning_angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    net = np.linalg.norm(points_m[:, -1] - points_m[:, 0], axis=-1)
    path = segment_lengths.sum(axis=-1)
    mean_speed = speeds.mean(axis=-1)
    maximum_speed = speeds.max(axis=-1)
    x = routes[..., 0]
    y = routes[..., 1]
    in_bounds = (
        finite_points
        & (x >= float(setup.profile.world_x_min))
        & (x <= float(setup.profile.world_x_max))
        & (y >= float(setup.profile.world_y_min))
        & (y <= float(setup.profile.world_y_max))
    )
    out_of_bounds_points = ~in_bounds
    out_of_bounds_routes = out_of_bounds_points.any(axis=-1)
    zero_motion = path <= ZERO_MOTION_PATH_LENGTH_METERS
    stationary = path <= STATIONARY_PATH_LENGTH_METERS
    explosive = maximum_speed > EXPLOSIVE_SPEED_METERS_PER_SECOND
    duplicates = _cross_query_duplicate_rates(
        (routes - current[:, None, :]) * meters_per_uu
    )
    return {
        "route_count": q,
        "net_displacement_meters": distribution_summary(net),
        "cumulative_path_length_meters": distribution_summary(path),
        "mean_five_second_speed_meters_per_second": distribution_summary(mean_speed),
        "maximum_five_second_speed_meters_per_second": distribution_summary(maximum_speed),
        "five_second_speed_distribution_meters_per_second": distribution_summary(speeds.ravel()),
        "acceleration_distribution_meters_per_second_squared": distribution_summary(accelerations.ravel()),
        "turning_angle_distribution_degrees": distribution_summary(turning_angles),
        "stationary_route_count": int(stationary.sum()),
        "stationary_route_rate": float(stationary.mean()),
        "zero_motion_route_count": int(zero_motion.sum()),
        "zero_motion_route_rate": float(zero_motion.mean()),
        "explosive_motion_route_count": int(explosive.sum()),
        "explosive_motion_route_rate": float(explosive.mean()),
        "out_of_bounds_point_count": int(out_of_bounds_points.sum()),
        "out_of_bounds_point_rate": float(out_of_bounds_points.mean()),
        "out_of_bounds_route_count": int(out_of_bounds_routes.sum()),
        "out_of_bounds_route_rate": float(out_of_bounds_routes.mean()),
        "nonfinite_point_count": int((~finite_points).sum()),
        "nonfinite_output_rate": float((~finite_routes).mean()),
        **duplicates,
        "collapse_indicators": {
            "zero_motion_collapse": bool(
                float(zero_motion.mean()) > MAX_PRIMARY_ZERO_MOTION_ROUTE_RATE
            ),
            "explosive_motion_collapse": bool(
                float(explosive.mean()) > MAX_PRIMARY_EXPLOSIVE_MOTION_ROUTE_RATE
            ),
            "near_duplicate_collapse": bool(
                duplicates["near_identical_route_rate"]
                > MAX_PRIMARY_NEAR_DUPLICATE_ROUTE_RATE
            ),
        },
    }


def route_coherence_report(
    arrays: PartitionArrays,
    setup: VerifiedTrainingSetup,
) -> dict[str, Any]:
    methods = {
        "primary": route_coherence_for_method(
            arrays.primary_xy_uu, arrays.current_xy_uu, setup
        ),
        "secondary": route_coherence_for_method(
            arrays.secondary_xy_uu, arrays.current_xy_uu, setup
        ),
        "static": route_coherence_for_method(
            arrays.static_xy_uu, arrays.current_xy_uu, setup
        ),
        "constant_velocity": route_coherence_for_method(
            arrays.constant_velocity_xy_uu, arrays.current_xy_uu, setup
        ),
    }
    mode_reports = {
        str(mode): route_coherence_for_method(
            arrays.mode_xy_uu[:, mode], arrays.current_xy_uu, setup
        )
        for mode in range(NUM_ROUTE_MODES)
    }
    primary = methods["primary"]
    gate = {
        "nonfinite_outputs_zero": primary["nonfinite_output_rate"] == 0.0,
        "out_of_bounds_rate_acceptable": (
            primary["out_of_bounds_route_rate"]
            <= MAX_PRIMARY_OUT_OF_BOUNDS_ROUTE_RATE
        ),
        "zero_motion_rate_acceptable": (
            primary["zero_motion_route_rate"]
            <= MAX_PRIMARY_ZERO_MOTION_ROUTE_RATE
        ),
        "explosive_motion_rate_acceptable": (
            primary["explosive_motion_route_rate"]
            <= MAX_PRIMARY_EXPLOSIVE_MOTION_ROUTE_RATE
        ),
        "near_duplicate_rate_acceptable": (
            primary["near_identical_route_rate"]
            <= MAX_PRIMARY_NEAR_DUPLICATE_ROUTE_RATE
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "methods": methods,
        "individual_modes": mode_reports,
        "primary_coherence_gate": gate,
    }


def mixture_report(
    arrays: PartitionArrays,
    setup: VerifiedTrainingSetup,
    accuracy_values: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    probabilities = arrays.mode_probabilities
    selected = np.argmax(probabilities, axis=1)
    counts = np.bincount(selected, minlength=NUM_ROUTE_MODES)
    frequencies = counts / len(selected)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, None)), axis=1)
    effective = np.exp(entropy)
    meters_per_uu = float(setup.profile.world_unit_scale.meters_per_world_unit)
    pairs = tuple(
        (left, right)
        for left in range(NUM_ROUTE_MODES)
        for right in range(left + 1, NUM_ROUTE_MODES)
    )
    pair_distances = np.empty((len(selected), len(pairs)), dtype=np.float64)
    duplicate_pairs = np.zeros((len(selected), len(pairs)), dtype=np.bool_)
    for pair_index, (left, right) in enumerate(pairs):
        delta_m = (
            arrays.mode_xy_uu[:, left] - arrays.mode_xy_uu[:, right]
        ) * meters_per_uu
        per_horizon = np.linalg.norm(delta_m, axis=-1)
        pair_distances[:, pair_index] = per_horizon.mean(axis=-1)
        rms = np.sqrt(np.mean(per_horizon * per_horizon, axis=-1))
        duplicate_pairs[:, pair_index] = rms <= NEAR_DUPLICATE_DISTANCE_METERS
    duplicate_query = duplicate_pairs.any(axis=1)

    mode_errors = (
        np.linalg.norm(
            arrays.mode_xy_uu - arrays.truth_xy_uu[:, None, :, :], axis=-1
        )
        * meters_per_uu
    )
    oracle_ade: list[float] = []
    oracle_fde: list[float] = []
    for row in range(len(selected)):
        valid = np.flatnonzero(arrays.target_mask[row])
        oracle_ade.append(float(mode_errors[row][:, valid].mean(axis=1).min()))
        oracle_fde.append(float(mode_errors[row][:, valid[-1]].min()))

    maximum_frequency = float(frequencies.max())
    mean_pairwise = float(pair_distances.mean())
    duplicate_query_rate = float(duplicate_query.mean())
    gate = {
        "no_single_mode_nearly_always_selected": maximum_frequency < MAX_SELECTED_MODE_FREQUENCY,
        "effective_mode_count_usable": float(effective.mean()) >= MIN_EFFECTIVE_MODE_COUNT,
        "pairwise_distance_usable": mean_pairwise >= MIN_PAIRWISE_MODE_DISTANCE_METERS,
        "duplicate_mode_query_rate_acceptable": duplicate_query_rate <= MAX_DUPLICATE_MODE_QUERY_RATE,
    }
    gate["passed"] = all(gate.values())
    return {
        "selected_mode_counts": [int(value) for value in counts],
        "selected_mode_frequencies": [float(value) for value in frequencies],
        "maximum_selected_mode_frequency": maximum_frequency,
        "one_mode_dominates_nearly_every_query": maximum_frequency >= MAX_SELECTED_MODE_FREQUENCY,
        "mean_mode_probabilities": [float(value) for value in probabilities.mean(axis=0)],
        "entropy_nats": distribution_summary(entropy),
        "mean_entropy_nats": float(entropy.mean()),
        "effective_mode_count": distribution_summary(effective),
        "mean_effective_mode_count": float(effective.mean()),
        "pairwise_complete_mode_trajectory_distance_meters": distribution_summary(pair_distances.ravel()),
        "mean_pairwise_complete_mode_trajectory_distance_meters": mean_pairwise,
        "duplicate_mode_pair_rate": float(duplicate_pairs.mean()),
        "per_query_duplicate_mode_rate": duplicate_query_rate,
        "queries_with_duplicate_modes": int(duplicate_query.sum()),
        "oracle_best_of_five_diagnostic_only": {
            "minade_meters": float(np.mean(oracle_ade)),
            "minfde_meters": float(np.mean(oracle_fde)),
            "used_for_checkpoint_selection": False,
            "used_for_superiority_claim": False,
        },
        "primary_minus_oracle_diagnostic_gap_meters": {
            "ade": float(np.nanmean(accuracy_values["primary"][1]) - np.mean(oracle_ade)),
            "fde": float(np.nanmean(accuracy_values["primary"][2]) - np.mean(oracle_fde)),
        },
        "diversity_gate": gate,
    }


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _component_log_density(
    points: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    correlations: np.ndarray,
) -> np.ndarray:
    """Return log densities broadcast over [..., K, H, 2] inputs."""

    residual = points - means
    x = residual[..., 0] / scales[..., 0]
    y = residual[..., 1] / scales[..., 1]
    one_minus = 1.0 - correlations * correlations
    quadratic = (x * x - 2.0 * correlations * x * y + y * y) / one_minus
    return (
        -math.log(2.0 * math.pi)
        - np.log(scales[..., 0])
        - np.log(scales[..., 1])
        - 0.5 * np.log(one_minus)
        - 0.5 * quadratic
    )


def _sample_coherent_routes(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    correlations: np.ndarray,
    count: int,
) -> np.ndarray:
    mode = rng.choice(NUM_ROUTE_MODES, size=count, replace=True, p=probabilities)
    selected_means = means[mode]
    selected_scales = scales[mode]
    selected_correlations = correlations[mode]
    standard = rng.standard_normal((count, NUM_HORIZONS, 2))
    x = selected_means[..., 0] + selected_scales[..., 0] * standard[..., 0]
    orthogonal = np.sqrt(np.maximum(1.0 - selected_correlations**2, 0.0))
    y = selected_means[..., 1] + selected_scales[..., 1] * (
        selected_correlations * standard[..., 0]
        + orthogonal * standard[..., 1]
    )
    return np.stack((x, y), axis=-1)


def calibration_report(
    arrays: PartitionArrays,
    setup: VerifiedTrainingSetup,
) -> dict[str, Any]:
    q = len(arrays.query_manifest)
    cell_scale_m = np.asarray(
        [
            float(setup.profile.cell_width_world_units)
            * float(setup.profile.world_unit_scale.meters_per_world_unit),
            float(setup.profile.cell_height_world_units)
            * float(setup.profile.world_unit_scale.meters_per_world_unit),
        ],
        dtype=np.float64,
    )
    per_horizon_nll: list[list[float]] = [[] for _ in HORIZONS_SECONDS]
    per_horizon_energy: list[list[float]] = [[] for _ in HORIZONS_SECONDS]
    per_horizon_coverage: dict[float, list[list[float]]] = {
        level: [[] for _ in HORIZONS_SECONDS] for level in PREDICTIVE_LEVELS
    }
    route_energy_scores: list[float] = []
    route_mean_horizon_nll: list[float] = []
    route_coverages: dict[float, list[float]] = {
        level: [] for level in PREDICTIVE_LEVELS
    }

    for row in range(q):
        probabilities = arrays.mode_probabilities[row]
        probabilities = probabilities / probabilities.sum()
        log_weights = np.log(np.clip(probabilities, 1e-300, None))
        means = arrays.means_normalized[row]
        scales = arrays.scales_normalized[row]
        correlations = arrays.correlations[row]
        target = arrays.target_displacements[row]
        mask = arrays.target_mask[row]
        target_component = _component_log_density(
            target[None, :, :], means, scales, correlations
        )
        target_log_density = _logsumexp(
            log_weights[:, None] + target_component, axis=0
        )
        valid = np.flatnonzero(mask)
        route_mean_horizon_nll.append(float((-target_log_density[valid]).mean()))
        for horizon in valid:
            per_horizon_nll[horizon].append(float(-target_log_density[horizon]))

        query = arrays.query_manifest[row]
        rng = np.random.Generator(
            np.random.PCG64(
                stable_seed(
                    EVALUATION_SEED,
                    "predictive-calibration-v1",
                    arrays.partition,
                    query,
                )
            )
        )
        sample_a = _sample_coherent_routes(
            rng, probabilities, means, scales, correlations, PREDICTIVE_SAMPLES
        )
        sample_b = _sample_coherent_routes(
            rng, probabilities, means, scales, correlations, PREDICTIVE_SAMPLES
        )
        sample_a_component = _component_log_density(
            sample_a[:, None, :, :],
            means[None, :, :, :],
            scales[None, :, :, :],
            correlations[None, :, :],
        )
        sample_a_log_density = _logsumexp(
            log_weights[None, :, None] + sample_a_component, axis=1
        )
        for level in PREDICTIVE_LEVELS:
            indicators: list[float] = []
            for horizon in valid:
                threshold = float(
                    np.quantile(
                        sample_a_log_density[:, horizon],
                        1.0 - level,
                        method="lower",
                    )
                )
                covered = float(target_log_density[horizon] >= threshold)
                indicators.append(covered)
                per_horizon_coverage[level][horizon].append(covered)
            route_coverages[level].append(float(np.mean(indicators)))

        target_m = target * cell_scale_m
        sample_a_m = sample_a * cell_scale_m
        sample_b_m = sample_b * cell_scale_m
        for horizon in valid:
            energy = float(
                np.linalg.norm(sample_a_m[:, horizon] - target_m[horizon], axis=-1).mean()
                - 0.5
                * np.linalg.norm(
                    sample_a_m[:, horizon] - sample_b_m[:, horizon], axis=-1
                ).mean()
            )
            per_horizon_energy[horizon].append(energy)
        normalization = math.sqrt(float(len(valid)))
        first = np.linalg.norm(
            (sample_a_m[:, valid] - target_m[valid]).reshape(PREDICTIVE_SAMPLES, -1),
            axis=-1,
        ).mean() / normalization
        second = np.linalg.norm(
            (sample_a_m[:, valid] - sample_b_m[:, valid]).reshape(PREDICTIVE_SAMPLES, -1),
            axis=-1,
        ).mean() / normalization
        route_energy_scores.append(float(first - 0.5 * second))

    coverage: dict[str, Any] = {}
    for level in PREDICTIVE_LEVELS:
        observed = float(np.mean(route_coverages[level]))
        coverage[str(level)] = {
            "nominal": level,
            "observed_route_equal": observed,
            "coverage_error": observed - level,
            "absolute_coverage_error": abs(observed - level),
            "valid_route_count": len(route_coverages[level]),
        }
    horizon_rows: dict[str, Any] = {}
    for horizon, seconds in enumerate(HORIZONS_SECONDS):
        horizon_rows[str(seconds)] = {
            "valid_route_count": len(per_horizon_nll[horizon]),
            "mixture_nll": (
                float(np.mean(per_horizon_nll[horizon]))
                if per_horizon_nll[horizon]
                else None
            ),
            "energy_score_meters": (
                float(np.mean(per_horizon_energy[horizon]))
                if per_horizon_energy[horizon]
                else None
            ),
            "coverage": {
                str(level): {
                    "nominal": level,
                    "observed": (
                        float(np.mean(per_horizon_coverage[level][horizon]))
                        if per_horizon_coverage[level][horizon]
                        else None
                    ),
                    "coverage_error": (
                        float(np.mean(per_horizon_coverage[level][horizon]) - level)
                        if per_horizon_coverage[level][horizon]
                        else None
                    ),
                }
                for level in PREDICTIVE_LEVELS
            },
        }
    calibration_gate = {
        f"absolute_coverage_error_at_{level}": (
            coverage[str(level)]["absolute_coverage_error"]
            <= MAX_ABSOLUTE_COVERAGE_ERROR
        )
        for level in PREDICTIVE_LEVELS
    }
    calibration_gate["all_scores_finite"] = bool(
        np.isfinite(arrays.route_nll).all()
        and np.isfinite(route_mean_horizon_nll).all()
        and np.isfinite(route_energy_scores).all()
    )
    calibration_gate["passed"] = all(calibration_gate.values())
    return {
        "route_level_mixture_nll": float(arrays.route_nll.mean()),
        "route_level_mixture_nll_distribution": distribution_summary(arrays.route_nll),
        "route_equal_mean_per_horizon_mixture_nll": float(
            np.mean(route_mean_horizon_nll)
        ),
        "predictive_region_coverage": coverage,
        "deterministic_sample_based_energy_score_meters": float(
            np.mean(route_energy_scores)
        ),
        "energy_score_distribution_meters": distribution_summary(route_energy_scores),
        "by_horizon_seconds": horizon_rows,
        "predictive_samples_per_set": PREDICTIVE_SAMPLES,
        "calibration_gate": calibration_gate,
    }


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def build_partition_core(
    arrays: PartitionArrays,
    setup: VerifiedTrainingSetup,
) -> dict[str, Any]:
    meters_per_uu = float(setup.profile.world_unit_scale.meters_per_world_unit)
    accuracy, accuracy_values = accuracy_report(arrays, meters_per_uu)
    coherence = route_coherence_report(arrays, setup)
    mixture = mixture_report(arrays, setup, accuracy_values)
    calibration = calibration_report(arrays, setup)
    outputs = (
        arrays.primary_xy_uu,
        arrays.secondary_xy_uu,
        arrays.mode_xy_uu,
        arrays.mode_logits,
        arrays.mode_probabilities,
        arrays.scales_normalized,
        arrays.correlations,
    )
    nonfinite_count = sum(int(np.size(value) - np.isfinite(value).sum()) for value in outputs)
    query_manifest = list(arrays.query_manifest)
    payload = {
        "schema_version": "parallel-trajectory-partition-canonical-payload:1.0",
        "partition": arrays.partition,
        "checkpoint": {
            "path": relative(RUN_DIRECTORY / "best.pt"),
            "sha256": file_sha256(RUN_DIRECTORY / "best.pt"),
        },
        "query_census": {
            "ordered_manifest": query_manifest,
            "ordered_manifest_sha256": payload_sha256(query_manifest),
            "query_count": len(query_manifest),
            "session_count": int(np.unique(arrays.session_ids).size),
            "queries_per_session": QUERIES_PER_SESSION,
            "sampler_version": QUERY_SAMPLER_VERSION,
            "seed": EVALUATION_SEED,
        },
        "free_running_integrity": {
            "targets_passed_to_model": False,
            "teacher_forcing": False,
            "future_waypoints_passed_to_model": False,
            "post_query_coordinates_used_by_constant_velocity": False,
            "causal_input_chunk_sha256": list(arrays.causal_input_hashes),
            "causal_input_chunk_count": len(arrays.causal_input_hashes),
            "sanitizer": OBSERVATION_PROVIDER_ID,
            "split_membership_verified": True,
        },
        "valid_counts": {
            "route_count": int(arrays.target_mask.any(axis=1).sum()),
            "horizon_count": int(arrays.target_mask.sum()),
            "valid_routes_by_horizon_seconds": {
                str(seconds): int(arrays.target_mask[:, index].sum())
                for index, seconds in enumerate(HORIZONS_SECONDS)
            },
        },
        "point_forecast_accuracy": accuracy,
        "route_coherence": coherence,
        "mixture_behavior": mixture,
        "probabilistic_calibration": calibration,
        "numerical_integrity": {
            "nonfinite_value_count": nonfinite_count,
            "nonfinite_output_rate": 0.0 if nonfinite_count == 0 else 1.0,
            "all_outputs_finite": nonfinite_count == 0,
            "array_sha256": {
                "primary_xy_uu": _ndarray_sha256(arrays.primary_xy_uu),
                "secondary_xy_uu": _ndarray_sha256(arrays.secondary_xy_uu),
                "mode_xy_uu": _ndarray_sha256(arrays.mode_xy_uu),
                "mode_logits": _ndarray_sha256(arrays.mode_logits),
                "mode_probabilities": _ndarray_sha256(arrays.mode_probabilities),
                "scales_normalized": _ndarray_sha256(arrays.scales_normalized),
                "correlations": _ndarray_sha256(arrays.correlations),
                "target_mask": _ndarray_sha256(arrays.target_mask),
                "truth_xy_uu": _ndarray_sha256(arrays.truth_xy_uu),
            },
        },
        "oracle_metrics_are_diagnostic_only": True,
    }
    require(nonfinite_count == 0, "nonfinite model output detected")
    return self_seal(payload, "canonical_payload_sha256")


def validation_eligibility_gate(
    core: Mapping[str, Any],
    *,
    reproducible: bool,
) -> dict[str, Any]:
    accuracy = core["point_forecast_accuracy"]
    methods = accuracy["methods"]
    primary = methods["primary"]["overall"]
    strict_wins: dict[str, bool] = {}
    ci_support: dict[str, bool] = {}
    for baseline in ("static", "constant_velocity"):
        baseline_metrics = methods[baseline]["overall"]
        for metric in ("ade", "fde"):
            strict_wins[f"primary_{metric}_lower_than_{baseline}"] = bool(
                primary[f"{metric}_meters"]
                < baseline_metrics[f"{metric}_meters"]
            )
            ci = accuracy["paired_model_minus_baseline"]["primary"][baseline][metric]
            ci_support[f"primary_{metric}_ci_supports_improvement_over_{baseline}"] = bool(
                ci["upper_meters"] < 0.0
            )
    checks = {
        "training_integrity_passed": True,
        "split_and_leakage_integrity_passed": bool(
            core["free_running_integrity"]["targets_passed_to_model"] is False
            and core["free_running_integrity"]["future_waypoints_passed_to_model"] is False
            and core["free_running_integrity"]["split_membership_verified"] is True
        ),
        "all_outputs_finite": bool(core["numerical_integrity"]["all_outputs_finite"]),
        "validation_reproducible": reproducible,
        "route_coherence_passed": bool(
            core["route_coherence"]["primary_coherence_gate"]["passed"]
        ),
        "mixture_diversity_passed": bool(
            core["mixture_behavior"]["diversity_gate"]["passed"]
        ),
        "probabilistic_calibration_passed": bool(
            core["probabilistic_calibration"]["calibration_gate"]["passed"]
        ),
        **strict_wins,
        **ci_support,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "eligible_for_sealed_test": not failures,
        "checks": checks,
        "failed_checks": failures,
        "empirical_claim_if_test_remains_sealed": (
            None if not failures else "unsupported_on_validation"
        ),
    }


def test_superiority_gate(core: Mapping[str, Any]) -> dict[str, Any]:
    accuracy = core["point_forecast_accuracy"]
    methods = accuracy["methods"]
    primary = methods["primary"]["overall"]
    checks: dict[str, bool] = {}
    for baseline in ("static", "constant_velocity"):
        candidate = methods[baseline]["overall"]
        for metric in ("ade", "fde"):
            checks[f"primary_{metric}_lower_than_{baseline}"] = bool(
                primary[f"{metric}_meters"] < candidate[f"{metric}_meters"]
            )
            ci = accuracy["paired_model_minus_baseline"]["primary"][baseline][metric]
            checks[f"primary_{metric}_ci_supports_improvement_over_{baseline}"] = bool(
                ci["upper_meters"] < 0.0
            )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "forecasting_superiority_supported": not failures,
        "checks": checks,
        "failed_checks": failures,
        "classification": "supported" if not failures else "unsupported_or_mixed",
    }


def _authorize_test(
    evaluation_directory: Path,
    validation_report: Mapping[str, Any],
) -> None:
    ledger = _load_ledger(evaluation_directory)
    require(ledger["counts"]["test"] == {"attempts": 0, "opens": 0}, "test was already accessed")
    require(ledger["validation_evaluator_invocation_count"] == 2, "validation did not run exactly twice")
    require(
        validation_report["eligibility_gate"]["eligible_for_sealed_test"] is True,
        "validation gate did not authorize test",
    )
    ledger["test_authorized"] = True
    ledger["test_access_preconditions_satisfied"] = {
        "validation_evaluator_invocation_count": 2,
        "validation_payload_byte_identical": True,
        "validation_eligible": True,
        "validation_report_payload_sha256": validation_report[
            "validation_report_payload_sha256"
        ],
        "selected_checkpoint_sha256": file_sha256(RUN_DIRECTORY / "best.pt"),
        "evaluator_source_sha256": file_sha256(Path(__file__)),
        "evaluation_contract_file_sha256": file_sha256(
            evaluation_directory / "evaluation_contract.json"
        ),
    }
    replace_json(evaluation_directory / "data_access_ledger.json", ledger)


def _key_performance(core: Mapping[str, Any]) -> dict[str, Any]:
    methods = core["point_forecast_accuracy"]["methods"]
    return {
        name: {
            "ade_meters": methods[name]["overall"]["ade_meters"],
            "fde_meters": methods[name]["overall"]["fde_meters"],
        }
        for name in ("primary", "secondary", "static", "constant_velocity")
    }


def _build_final_summary(
    evaluation_directory: Path,
    selection: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validation_core = validation_report["canonical_evaluation"]
    validation_eligible = bool(
        validation_report["eligibility_gate"]["eligible_for_sealed_test"]
    )
    ledger = _load_ledger(evaluation_directory)
    if test_report is None:
        claim = "unsupported_on_validation"
        test_status: dict[str, Any] = {
            "authorized": False,
            "performed": False,
            "reason": "validation eligibility gate failed",
        }
    else:
        claim = test_report["superiority_gate"]["classification"]
        test_status = {
            "authorized": True,
            "performed": True,
            "performance": _key_performance(test_report["canonical_evaluation"]),
            "superiority_gate": test_report["superiority_gate"],
        }
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "evaluation_directory": relative(evaluation_directory),
        "selected_checkpoint": {
            "path": selection["selection"]["selected_checkpoint_path"],
            "sha256": selection["selection"]["selected_checkpoint_sha256"],
            "selection_metric": selection["selection"]["selection_metric"],
            "selection_value": selection["selection"]["selected_metric_value"],
            "completed_epoch": selection["selection"]["selected_completed_epoch"],
            "optimizer_step": selection["selection"]["selected_optimizer_step"],
        },
        "training_integrity": {
            "status": selection["training_integrity"]["status"],
            "exact_resume_lineage": selection["training_integrity"]["exact_resume_lineage"],
            "optimizer_steps": selection["training_integrity"]["observed_optimizer_steps"],
            "completed_epochs": selection["training_integrity"]["observed_epochs"],
            "stderr_empty": True,
            "all_recorded_values_and_checkpoint_tensors_finite": True,
            "training_test_access_attempts": 0,
            "training_test_access_opens": 0,
            "provenance_bindings_exact": True,
        },
        "validation_performance": {
            "eligible_for_test": validation_eligible,
            "performance": _key_performance(validation_core),
            "eligibility_gate": validation_report["eligibility_gate"],
            "canonical_payload_sha256": validation_core[
                "canonical_payload_sha256"
            ],
        },
        "test_performance": test_status,
        "point_forecast_accuracy": {
            "validation_primary": _key_performance(validation_core)["primary"],
            "test_primary": (
                None
                if test_report is None
                else _key_performance(test_report["canonical_evaluation"])["primary"]
            ),
            "route_equal_aggregation": True,
        },
        "probabilistic_calibration": {
            "validation": {
                "route_level_mixture_nll": validation_core[
                    "probabilistic_calibration"
                ]["route_level_mixture_nll"],
                "coverage": validation_core["probabilistic_calibration"][
                    "predictive_region_coverage"
                ],
                "energy_score_meters": validation_core[
                    "probabilistic_calibration"
                ]["deterministic_sample_based_energy_score_meters"],
                "gate": validation_core["probabilistic_calibration"][
                    "calibration_gate"
                ],
            },
            "test": (
                None
                if test_report is None
                else {
                    "route_level_mixture_nll": test_report["canonical_evaluation"][
                        "probabilistic_calibration"
                    ]["route_level_mixture_nll"],
                    "coverage": test_report["canonical_evaluation"][
                        "probabilistic_calibration"
                    ]["predictive_region_coverage"],
                    "energy_score_meters": test_report["canonical_evaluation"][
                        "probabilistic_calibration"
                    ]["deterministic_sample_based_energy_score_meters"],
                }
            ),
        },
        "mode_diversity": {
            "validation": validation_core["mixture_behavior"],
            "test": (
                None
                if test_report is None
                else test_report["canonical_evaluation"]["mixture_behavior"]
            ),
        },
        "route_coherence": {
            "validation_primary": validation_core["route_coherence"]["methods"][
                "primary"
            ],
            "validation_gate": validation_core["route_coherence"][
                "primary_coherence_gate"
            ],
            "test_primary": (
                None
                if test_report is None
                else test_report["canonical_evaluation"]["route_coherence"][
                    "methods"
                ]["primary"]
            ),
        },
        "forecasting_superiority_over_static_and_constant_velocity": {
            "classification": claim,
            "supported": claim == "supported",
            "test_required_for_supported_claim": True,
        },
        "data_access": {
            "validation_evaluator_invocations": ledger[
                "validation_evaluator_invocation_count"
            ],
            "test_evaluator_invocations": ledger["test_evaluator_invocation_count"],
            "counts": ledger["counts"],
            "test_authorized": ledger["test_authorized"],
        },
        "historical_artifacts_modified": False,
        "training_resumed_or_retrained_by_evaluator": False,
        "sealed": True,
    }
    return self_seal(payload, "final_summary_payload_sha256")


def _write_manifest(evaluation_directory: Path) -> dict[str, Any]:
    names = [
        "checkpoint_selection.json",
        "evaluation_contract.json",
        "validation_evaluation.json",
        "data_access_ledger.json",
        "environment.json",
        "final_evaluation_summary.json",
    ]
    if (evaluation_directory / "test_evaluation.json").is_file():
        names.append("test_evaluation.json")
    artifacts = [
        {
            "path": relative(evaluation_directory / name),
            "sha256": file_sha256(evaluation_directory / name),
            "bytes": (evaluation_directory / name).stat().st_size,
        }
        for name in sorted(names)
    ]
    payload = self_seal(
        {
            "schema_version": MANIFEST_SCHEMA,
            "evaluation_directory": relative(evaluation_directory),
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
            "external_frozen_inputs": [
                {
                    "path": relative(RUN_DIRECTORY / "best.pt"),
                    "sha256": file_sha256(RUN_DIRECTORY / "best.pt"),
                },
                {
                    "path": relative(CONFIG_PATH),
                    "sha256": file_sha256(CONFIG_PATH),
                },
                {
                    "path": relative(Path(__file__)),
                    "sha256": file_sha256(Path(__file__)),
                },
            ],
            "historical_artifacts_preserved": True,
            "manifest_self_hash_excludes_manifest_payload_sha256_field": True,
        },
        "manifest_payload_sha256",
    )
    write_new_json(evaluation_directory / "artifact_manifest.json", payload)
    return payload


def _make_immutable(evaluation_directory: Path) -> None:
    for path in evaluation_directory.iterdir():
        if path.is_file():
            path.chmod(stat.S_IREAD)


def run_locked_evaluation(evaluation_directory: Path) -> dict[str, Any]:
    evaluation_directory = evaluation_directory.resolve()
    require(evaluation_directory.is_dir(), "prepared evaluation directory is missing")
    require(
        not (evaluation_directory / "artifact_manifest.json").exists(),
        "evaluation directory is already finalized and immutable",
    )
    setup = verify_training_setup(CONFIG_PATH)
    spec = resolve_precision()
    selection, _contract = _verify_frozen_preconditions(setup, evaluation_directory)
    ledger = _load_ledger(evaluation_directory)
    require(ledger["validation_evaluator_invocation_count"] == 0, "validation was already invoked")
    require(ledger["test_evaluator_invocation_count"] == 0, "test was already invoked")

    print(json.dumps({"stage": "validation", "repeat": 1, "status": "starting"}), flush=True)
    validation_arrays_1 = evaluate_partition_arrays(
        setup, spec, evaluation_directory, partition="validation"
    )
    validation_core_1 = build_partition_core(validation_arrays_1, setup)
    validation_bytes_1 = canonical_bytes(validation_core_1)
    print(json.dumps({"stage": "validation", "repeat": 1, "status": "completed", "payload_sha256": hashlib.sha256(validation_bytes_1).hexdigest()}), flush=True)

    print(json.dumps({"stage": "validation", "repeat": 2, "status": "starting"}), flush=True)
    validation_arrays_2 = evaluate_partition_arrays(
        setup, spec, evaluation_directory, partition="validation"
    )
    validation_core_2 = build_partition_core(validation_arrays_2, setup)
    validation_bytes_2 = canonical_bytes(validation_core_2)
    reproducible = validation_bytes_1 == validation_bytes_2
    require(reproducible, "validation canonical payloads are not byte-identical")
    run_hash = hashlib.sha256(validation_bytes_1).hexdigest()
    eligibility = validation_eligibility_gate(
        validation_core_1, reproducible=reproducible
    )
    validation_report = self_seal(
        {
            "schema_version": VALIDATION_SCHEMA,
            "canonical_evaluation": validation_core_1,
            "reproducibility": {
                "evaluator_invocations": 2,
                "run_1_canonical_payload_sha256": run_hash,
                "run_2_canonical_payload_sha256": hashlib.sha256(validation_bytes_2).hexdigest(),
                "byte_identical": True,
            },
            "eligibility_gate": eligibility,
            "test_access_before_gate": {"attempts": 0, "opens": 0},
        },
        "validation_report_payload_sha256",
    )
    write_new_json(
        evaluation_directory / "validation_evaluation.json", validation_report
    )
    print(json.dumps({"stage": "validation", "repeat": 2, "status": "completed", "payload_sha256": run_hash, "eligible_for_test": eligibility["eligible_for_sealed_test"]}), flush=True)

    test_report: dict[str, Any] | None = None
    if eligibility["eligible_for_sealed_test"]:
        _authorize_test(evaluation_directory, validation_report)
        print(json.dumps({"stage": "test", "invocation": 1, "status": "starting"}), flush=True)
        test_arrays = evaluate_partition_arrays(
            setup, spec, evaluation_directory, partition="test"
        )
        test_core = build_partition_core(test_arrays, setup)
        superiority = test_superiority_gate(test_core)
        test_report = self_seal(
            {
                "schema_version": TEST_SCHEMA,
                "canonical_evaluation": test_core,
                "authorization": {
                    "validation_report_file_sha256": file_sha256(
                        evaluation_directory / "validation_evaluation.json"
                    ),
                    "validation_report_payload_sha256": validation_report[
                        "validation_report_payload_sha256"
                    ],
                    "validation_gate_passed": True,
                    "test_evaluator_invocation": 1,
                    "post_test_tuning_performed": False,
                },
                "superiority_gate": superiority,
            },
            "test_report_payload_sha256",
        )
        write_new_json(evaluation_directory / "test_evaluation.json", test_report)
        print(json.dumps({"stage": "test", "invocation": 1, "status": "completed", "superiority": superiority["classification"]}), flush=True)

    summary = _build_final_summary(
        evaluation_directory, selection, validation_report, test_report
    )
    write_new_json(
        evaluation_directory / "final_evaluation_summary.json", summary
    )
    manifest = _write_manifest(evaluation_directory)
    _make_immutable(evaluation_directory)
    return {
        "status": "completed",
        "evaluation_directory": str(evaluation_directory),
        "validation_eligible_for_test": eligibility["eligible_for_sealed_test"],
        "test_performed": test_report is not None,
        "claim": summary[
            "forecasting_superiority_over_static_and_constant_velocity"
        ]["classification"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
    }


def verify_final(evaluation_directory: Path) -> dict[str, Any]:
    evaluation_directory = evaluation_directory.resolve()
    manifest = read_json(evaluation_directory / "artifact_manifest.json", "artifact manifest")
    verify_self_seal(manifest, "manifest_payload_sha256")
    for artifact in manifest["artifacts"]:
        path = REPOSITORY_ROOT / artifact["path"]
        require(path.is_file(), f"manifest artifact is missing: {path}")
        require(file_sha256(path) == artifact["sha256"], f"artifact hash mismatch: {path}")
        require(path.stat().st_size == artifact["bytes"], f"artifact size mismatch: {path}")
    selection = read_json(evaluation_directory / "checkpoint_selection.json", "checkpoint selection")
    contract = read_json(evaluation_directory / "evaluation_contract.json", "evaluation contract")
    validation = read_json(evaluation_directory / "validation_evaluation.json", "validation evaluation")
    summary = read_json(evaluation_directory / "final_evaluation_summary.json", "final summary")
    verify_self_seal(selection, "checkpoint_selection_payload_sha256")
    verify_self_seal(contract, "contract_payload_sha256")
    verify_self_seal(validation, "validation_report_payload_sha256")
    verify_self_seal(summary, "final_summary_payload_sha256")
    test_path = evaluation_directory / "test_evaluation.json"
    if test_path.exists():
        verify_self_seal(read_json(test_path, "test evaluation"), "test_report_payload_sha256")
    ledger = _load_ledger(evaluation_directory)
    require(ledger["validation_evaluator_invocation_count"] == 2, "validation invocation count is not two")
    require(ledger["counts"]["validation"] == {"attempts": 22, "opens": 22}, "validation access counts are not 22/22")
    if test_path.exists():
        require(ledger["test_evaluator_invocation_count"] == 1, "test invocation count is not one")
        require(ledger["counts"]["test"] == {"attempts": 10, "opens": 10}, "test access counts are not 10/10")
    else:
        require(ledger["test_evaluator_invocation_count"] == 0, "sealed test evaluator was invoked")
        require(ledger["counts"]["test"] == {"attempts": 0, "opens": 0}, "sealed test was accessed")
    return {
        "status": "verified",
        "evaluation_directory": str(evaluation_directory),
        "artifact_count": manifest["artifact_count"],
        "test_evaluation_present": test_path.exists(),
        "claim": summary[
            "forecasting_superiority_over_static_and_constant_velocity"
        ]["classification"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked parallel-trajectory evaluation")
    parser.add_argument(
        "command", choices=("prepare", "run", "verify"), help="staged operation"
    )
    parser.add_argument(
        "--evaluation-directory",
        type=Path,
        default=DEFAULT_EVALUATION_DIRECTORY,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.evaluation_directory)
    elif args.command == "run":
        result = run_locked_evaluation(args.evaluation_directory)
    else:
        result = verify_final(args.evaluation_directory)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
