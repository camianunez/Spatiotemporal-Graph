from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from fortnite_encoder.config import EncoderConfig, RotationHeadConfig, RotationLossConfig
from fortnite_encoder.losses import compute_rotation_loss
from fortnite_encoder.rotation import RotationModel
from fortnite_encoder.training import (
    LossMetricAccumulator,
    PrecisionSpec,
    TrainingConfigurationError,
    TrainingNumericalError,
    _SessionRecord,
    _autocast_context,
    _batch_requests,
    _gradient_norm_and_finite,
    _loss_has_supervision,
    _move_batch,
    _validate_finite_logits_and_loss,
    collate_training_windows,
    resolve_precision,
    training_window_requests,
    validation_window_requests,
)
from fortnite_encoder.world_grid import WorldGridProfile

from .artifacts import (
    checkpoint_tensor_report,
    export_encoder_only,
    parameter_report,
    write_artifact_manifest,
)
from .common import (
    append_jsonl,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    file_sha256,
    restore_rng_state,
    rendered_json,
    rng_state,
    seed_everything,
    sha256_bytes,
    source_tree_manifest,
    utc_now,
)
from .config import FreshEncoderConfig, load_config
from .data_access import (
    AuditedSessionRepository,
    DataAccessLedger,
    audited_session_lengths,
)
from .dataset import load_bound_inventory, load_bound_split
from .optimization import (
    OptimizerContract,
    PiecewiseLinearScheduler,
    ScheduleContract,
    build_adamw_optimizer,
)


CHECKPOINT_SCHEMA_VERSION = "fncs-fresh-rotation-training-checkpoint:1.0"


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a durable marker without a check-then-overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(rendered_json(value))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # fdopen owns the descriptor once entered.  A failed marker remains a
        # visible immutable launch failure and is never silently overwritten.
        raise


@dataclass(slots=True)
class TrainingState:
    completed_epochs: int = 0
    current_epoch: int = 0
    next_batch_index: int = 0
    optimizer_step: int = 0
    data_microbatches: int = 0
    supervised_microbatches: int = 0
    accumulation_count: int = 0
    best_metric: float | None = None
    best_optimizer_step: int | None = None
    best_completed_epoch: int | None = None
    non_improvements: int = 0
    stopped_early: bool = False
    first_optimizer_step_completed: bool = False
    last_validation_metrics: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingState:
        declared = {item.name for item in dataclasses.fields(cls)}
        if set(value) != declared:
            raise ValueError("checkpoint training-state fields changed")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_resolved(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("resolved training configuration must be an object")
    recorded = value.get("resolved_training_config_sha256")
    deterministic = dict(value)
    deterministic.pop("created_utc", None)
    deterministic.pop("resolved_training_config_sha256", None)
    from .common import sha256_bytes

    actual = sha256_bytes(canonical_json(deterministic).encode("utf-8"))
    if recorded != actual:
        raise ValueError("resolved training configuration self-hash changed")
    return value


def _runtime_payload(
    *,
    state: TrainingState,
    run_directory: Path,
    status: str,
    ledger: DataAccessLedger,
    spec: PrecisionSpec,
) -> dict[str, Any]:
    access = ledger.report()
    return {
        "schema_version": "fncs-encoder-runtime:1.0",
        "updated_utc": utc_now(),
        "status": status,
        "process_id": os.getpid(),
        "device": str(spec.device),
        "precision": spec.precision,
        "cuda_bf16": spec.device.type == "cuda" and spec.precision == "bf16",
        **state.to_dict(),
        "last_checkpoint_exists": (run_directory / "last.pt").is_file(),
        "last_checkpoint_sha256": (
            file_sha256(run_directory / "last.pt")
            if (run_directory / "last.pt").is_file()
            else None
        ),
        "best_checkpoint_exists": (run_directory / "best.pt").is_file(),
        "test_session_parquet_attempt_count": access[
            "test_session_parquet_attempt_count"
        ],
        "test_session_parquet_open_count": access[
            "test_session_parquet_open_count"
        ],
        "zero_test_file_attempts": access["zero_test_file_attempts"],
        "zero_test_file_opens": access["zero_test_file_opens"],
        "test_partition_evaluated": False,
    }


def _loss_components(loss: Any, weights: RotationLossConfig) -> dict[str, Any]:
    raw = {
        "position": float(loss.future_position.detach().cpu()),
        "entry": float(loss.zone_entry.detach().cpu()),
        "survival": float(loss.survival.detach().cpu()),
        "placement": float(loss.placement.detach().cpu()),
        "regularization": 0.0,
    }
    weighted = {
        "position": raw["position"],
        "entry": float(weights.entry_weight) * raw["entry"],
        "survival": float(weights.survival_weight) * raw["survival"],
        "placement": float(weights.placement_weight) * raw["placement"],
        "regularization": 0.0,
    }
    total = sum(weighted.values())
    actual_total = float(loss.total.detach().cpu())
    if not math.isclose(total, actual_total, rel_tol=1e-5, abs_tol=1e-5):
        raise TrainingNumericalError("raw and weighted loss decomposition changed")
    return {
        "formula": (
            "L_total = L_position + lambda_entry * L_entry + "
            "lambda_survival * L_survival + lambda_placement * L_placement + "
            "L_regularization"
        ),
        "raw": raw,
        "weighted": weighted,
        "total": actual_total,
        "counts": {
            "future_position_by_horizon": list(loss.future_position_valid_counts),
            "zone_entry": loss.zone_entry_valid_count,
            "survival": loss.survival_valid_count,
            "placement": loss.placement_valid_count,
        },
    }


def _encoder_gradient_report(model: RotationModel) -> dict[str, Any]:
    squared = torch.zeros((), dtype=torch.float64)
    parameter_count = 0
    nonzero_count = 0
    for parameter in model.encoder.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        parameter_count += 1
        detached = gradient.detach()
        if not bool(torch.isfinite(detached).all()):
            raise TrainingNumericalError("encoder gradient is non-finite")
        if bool((detached != 0).any()):
            nonzero_count += 1
        squared += detached.double().square().sum().cpu()
    norm = float(torch.sqrt(squared))
    return {
        "parameters_with_gradient": parameter_count,
        "parameters_with_nonzero_gradient": nonzero_count,
        "finite": math.isfinite(norm),
        "nonzero": norm > 0.0 and nonzero_count > 0,
        "gradient_norm": norm,
    }


def _records(
    inventory: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    dataset_root: Path,
    profile: WorldGridProfile,
) -> tuple[tuple[_SessionRecord, ...], dict[str, str]]:
    by_id = {item["game_session_id"]: item for item in inventory["sessions"]}
    partition_by_session: dict[str, str] = {}
    records: list[_SessionRecord] = []
    for assignment in split["assignments"]:
        session_id = assignment["game_session_id"]
        item = by_id.get(session_id)
        if item is None:
            raise ValueError(f"split contains session absent from inventory: {session_id}")
        if assignment["source_replay_sha256"] != item["source_replay_sha256"]:
            raise ValueError(f"split replay hash changed for {session_id}")
        partition_by_session[session_id] = assignment["partition"]
        records.append(
            _SessionRecord(
                session_id=session_id,
                trust_partition=item["trust_partition"],
                path=dataset_root / item["directory"],
                replay_sha256=item["source_replay_sha256"],
                provenance_status=item["provenance_status"],
                build=str(item["build"]),
                world_identity=str(item["world_identity"]),
                world_grid_profile_id=profile.profile_id,
                world_grid_profile_hash=profile.profile_hash,
            )
        )
    if len(records) != 1150 or len(partition_by_session) != 1150:
        raise ValueError("runtime records are not exactly the 1,150 canonical sessions")
    return tuple(records), partition_by_session


def _compatibility_contract(
    *,
    resolved: Mapping[str, Any],
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    initial_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "resolved_training_config_sha256": resolved["resolved_training_config_sha256"],
        "encoder_config": resolved["encoder_config"],
        "head_config": resolved["head_config"],
        "loss_config": resolved["loss_config"],
        "bindings": resolved["bindings"],
        "optimizer": optimizer_contract.to_dict(),
        "scheduler": scheduler.contract(),
        "initial_parameter_sha256": initial_parameters[
            "complete_initial_parameter_sha256"
        ],
        "precision": "bf16",
        "device_type": "cuda",
        "sampler": resolved["sampler"],
        "checkpoint_selection_metric": resolved["checkpoint_selection_metric"],
        "test_partition_policy": "sealed_zero_attempts_zero_opens",
    }


def _checkpoint_payload(
    *,
    model: RotationModel,
    optimizer: torch.optim.AdamW,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    state: TrainingState,
    epoch_accumulator: LossMetricAccumulator,
    resolved: Mapping[str, Any],
    initial_parameters: Mapping[str, Any],
    metrics_path: Path,
    ledger: DataAccessLedger,
) -> dict[str, Any]:
    access = ledger.report()
    if not access["zero_test_file_attempts"] or not access["zero_test_file_opens"]:
        raise TrainingConfigurationError("sealed test access detected before checkpoint")
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "type": "RotationModelTrainingOnlySupervision",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_contract": optimizer_contract.to_dict(),
        "scheduler_state": scheduler.state_dict(),
        "training_state": state.to_dict(),
        "epoch_metric_state": epoch_accumulator.state_dict(),
        "sampler_state": {
            "seed": resolved["seed"],
            "current_epoch": state.current_epoch,
            "next_batch_index": state.next_batch_index,
            "algorithm": resolved["sampler"],
        },
        "rng_state": rng_state(),
        "amp_scaler_state": None,
        "precision": "bf16",
        "compatibility": _compatibility_contract(
            resolved=resolved,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            initial_parameters=initial_parameters,
        ),
        "resolved_training_config": dict(resolved),
        "initial_parameters": dict(initial_parameters),
        "metrics_jsonl": metrics_path.read_bytes() if metrics_path.is_file() else b"",
        "data_access": access,
        "test_partition_evaluated": False,
    }


def _load_checkpoint(
    path: Path,
    *,
    model: RotationModel,
    optimizer: torch.optim.AdamW,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    resolved: Mapping[str, Any],
    initial_parameters: Mapping[str, Any],
    metrics_path: Path,
    epoch_accumulator: LossMetricAccumulator,
) -> TrainingState:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("resume checkpoint schema changed")
    expected = _compatibility_contract(
        resolved=resolved,
        optimizer_contract=optimizer_contract,
        scheduler=scheduler,
        initial_parameters=initial_parameters,
    )
    if canonical_json(checkpoint.get("compatibility")) != canonical_json(expected):
        raise ValueError("resume checkpoint compatibility contract mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    state = TrainingState.from_dict(checkpoint["training_state"])
    if state.optimizer_step != scheduler.step_index:
        raise ValueError("optimizer and scheduler resume counters differ")
    if state.accumulation_count != 0:
        raise ValueError("the fixed accumulation=1 contract cannot resume pending gradients")
    metrics = checkpoint.get("metrics_jsonl")
    if not isinstance(metrics, bytes):
        raise ValueError("checkpoint metrics history is missing")
    atomic_write_bytes(metrics_path, metrics)
    epoch_metric_state = checkpoint.get("epoch_metric_state")
    if not isinstance(epoch_metric_state, Mapping):
        raise ValueError("checkpoint partial-epoch metric state is missing")
    epoch_accumulator.load_state_dict(epoch_metric_state)
    restore_rng_state(checkpoint["rng_state"])
    return state


def _evaluate(
    *,
    model: RotationModel,
    repository: AuditedSessionRepository,
    requests: Sequence[Any],
    config: FreshEncoderConfig,
    profile: WorldGridProfile,
    head_config: RotationHeadConfig,
    loss_config: RotationLossConfig,
    spec: PrecisionSpec,
) -> tuple[dict[str, Any], float]:
    accumulator = LossMetricAccumulator(loss_config)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for request_batch in _batch_requests(requests, config.batch_size):
            items = repository.windows(request_batch)
            encoder, supervision = collate_training_windows(
                items,
                expected_profile_id=profile.profile_id,
                expected_profile_hash=profile.profile_hash,
                head_config=head_config,
            )
            encoder, supervision = _move_batch(
                encoder, supervision, spec, config.pin_memory
            )
            with _autocast_context(spec):
                output = model(encoder.batch)
                loss = compute_rotation_loss(output.logits, supervision.targets, loss_config)
            _validate_finite_logits_and_loss(output, loss, encoder.metadata.session_ids)
            accumulator.update(loss)
    return accumulator.metrics(), time.perf_counter() - started


def run_training(
    config_path: str | Path,
    *,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    run_directory = Path(config.run_directory)
    if not run_directory.is_dir():
        raise ValueError("preflight-authorized run directory does not exist")
    resolved = _load_resolved(run_directory / "resolved_training_config.json")
    if file_sha256(run_directory / "canonical_config.json") != resolved["canonical_config_raw_sha256"]:
        raise ValueError("canonical run configuration changed after preflight")
    if file_sha256(config_path) != resolved["canonical_config_raw_sha256"]:
        raise ValueError("launch configuration differs from preflight configuration")
    if resolved.get("full_training_authorized") is not True:
        raise ValueError("preflight did not authorize full training")

    marker = run_directory / "training_started.json"
    if resume is None:
        if marker.exists():
            raise ValueError("refusing to overwrite or restart an existing immutable run")
    elif not marker.is_file():
        raise ValueError("resume requested for a run that never started")

    bindings = resolved["bindings"]
    inventory = load_bound_inventory(
        run_directory / "dataset_inventory.json",
        bindings["dataset_validation_sha256"],
    )
    split = load_bound_split(
        run_directory / "split_manifest.json", bindings["split_manifest_sha256"]
    )
    source = source_tree_manifest(
        Path(resolved["workspace"]) / "ml" / "src" / "fortnite_encoder"
    )
    if source["source_tree_sha256"] != bindings["encoder_source_digest"]:
        raise ValueError("protected encoder source digest changed after preflight")
    training_source = source_tree_manifest(
        Path(resolved["workspace"]) / "ml" / "src" / "fncs_encoder_training"
    )
    if (
        training_source["source_tree_sha256"]
        != bindings["training_orchestration_source_digest"]
    ):
        raise ValueError("training orchestration source changed after preflight")
    world_audit = json.loads(
        (run_directory / "world_grid_compatibility_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        world_audit.get("status") != "compatible"
        or world_audit.get("world_grid_validation_audit_sha256")
        != bindings["world_grid_validation_audit_sha256"]
    ):
        raise ValueError("world-grid compatibility binding changed")
    deterministic_world_audit = dict(world_audit)
    deterministic_world_audit.pop("created_utc", None)
    recorded_world_audit_hash = deterministic_world_audit.pop(
        "world_grid_validation_audit_sha256", None
    )
    actual_world_audit_hash = sha256_bytes(
        canonical_json(deterministic_world_audit).encode("utf-8")
    )
    if recorded_world_audit_hash != actual_world_audit_hash:
        raise ValueError("world-grid compatibility audit self-hash changed")
    if file_sha256(run_directory / "world_grid_out_of_bounds.jsonl") != world_audit.get(
        "out_of_bounds_coordinates_raw_sha256"
    ):
        raise ValueError("world-grid out-of-bounds record changed after preflight")
    if (
        file_sha256(config.world_grid_profile)
        != bindings["world_grid_profile_raw_sha256"]
        or file_sha256(config.world_grid_publication_audit)
        != bindings["world_grid_publication_audit_raw_sha256"]
        or file_sha256(Path(config.dataset_root) / "ingestion-report.json")
        != bindings["ingestion_report_sha256"]
        or file_sha256(config.dataset_validation_path)
        != bindings["source_dataset_validation_raw_sha256"]
    ):
        raise ValueError("a raw dataset or world-grid authority changed after preflight")
    profile = WorldGridProfile.load(
        config.world_grid_profile,
        expected_hash=config.expected_world_grid_profile_hash,
    )

    encoder_config = EncoderConfig(**resolved["encoder_config"])
    head_values = dict(resolved["head_config"])
    head_values["horizons_seconds"] = tuple(head_values["horizons_seconds"])
    head_config = RotationHeadConfig(**head_values)
    loss_config = RotationLossConfig(**resolved["loss_config"])
    spec = resolve_precision(config.device, config.precision)  # type: ignore[arg-type]
    if (
        spec.device != torch.device("cuda:0")
        or spec.precision != "bf16"
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("production requires the RTX 4090 CUDA BF16 environment")

    seed_everything(config.seed)
    model = RotationModel(
        encoder_config=encoder_config,
        head_config=head_config,
    ).to(spec.device)
    initial_parameters = parameter_report(model, seed=config.seed)
    if (
        initial_parameters["complete_initial_parameter_sha256"]
        != resolved["initial_parameters"]["complete_initial_parameter_sha256"]
    ):
        raise ValueError("production model did not reconstruct from the original seed")
    optimizer, optimizer_contract = build_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        betas=config.adamw_betas,
        epsilon=config.adamw_epsilon,
        weight_decay=config.weight_decay,
    )
    if canonical_json(optimizer_contract.to_dict()) != canonical_json(
        resolved["optimizer_contract"]
    ):
        raise ValueError("production AdamW parameter groups changed after preflight")
    schedule_contract = ScheduleContract(**resolved["schedule_contract_constructor"])
    scheduler = PiecewiseLinearScheduler(optimizer, schedule_contract)
    if canonical_json(scheduler.contract()) != canonical_json(resolved["scheduler"]):
        raise ValueError("production scheduler changed after preflight")

    records, partition_by_session = _records(
        inventory, split, dataset_root=Path(config.dataset_root), profile=profile
    )
    train_ids = tuple(split["ordered_session_ids"]["train"])
    validation_ids = tuple(split["ordered_session_ids"]["validation"])
    test_ids = tuple(split["ordered_session_ids"]["test"])
    if (len(train_ids), len(validation_ids), len(test_ids)) != (920, 115, 115):
        raise ValueError("runtime split counts changed")
    ledger = DataAccessLedger(
        path=run_directory / "data_access_ledger.json",
        partition_by_session=partition_by_session,
        test_session_ids=test_ids,
        split_manifest_sha256=bindings["split_manifest_sha256"],
        phase="full_training",
        allowed_partitions=("train", "validation"),
    )
    ledger.set_phase("full_training", ("train", "validation"))
    lengths = audited_session_lengths(
        records,
        encoder_config,
        active_session_ids=(*train_ids, *validation_ids),
        ledger=ledger,
    )
    active = set(train_ids) | set(validation_ids)
    repository = AuditedSessionRepository(
        tuple(record for record in records if record.session_id in active),
        profile,
        head_config,
        config.num_workers,
        False,
        ledger,
    )
    validation_requests = validation_window_requests(
        lengths, validation_ids, window_length_ticks=config.window_length_ticks
    )
    metrics_path = run_directory / "metrics.jsonl"
    epoch_accumulator = LossMetricAccumulator(loss_config)
    if resume is None:
        _write_exclusive_json(
            marker,
            {
                "schema_version": "fncs-encoder-training-start:1.0",
                "started_utc": utc_now(),
                "process_id": os.getpid(),
                "fresh": True,
                "resume": False,
                "resolved_training_config_sha256": resolved[
                    "resolved_training_config_sha256"
                ],
                "initial_parameter_sha256": initial_parameters[
                    "complete_initial_parameter_sha256"
                ],
            },
        )
        atomic_write_bytes(metrics_path, b"")
        state = TrainingState()
    else:
        resume_path = run_directory / "last.pt" if str(resume) == "last" else Path(resume)
        state = _load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            resolved=resolved,
            initial_parameters=initial_parameters,
            metrics_path=metrics_path,
            epoch_accumulator=epoch_accumulator,
        )
    optimizer.zero_grad(set_to_none=True)

    feature_contract = json.loads(
        (run_directory / "feature_contract.json").read_text(encoding="utf-8")
    )
    provenance = {
        "run_directory": str(run_directory.resolve()),
        "seed": config.seed,
        "authoritative_run": config.authoritative_run_directory,
        "authoritative_checkpoint_inspected_only": config.authoritative_checkpoint,
        "fresh_encoder": True,
        "trajectory_decoder_present": False,
    }

    def save_checkpoint(path: Path, *, export_encoder: bool) -> None:
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            state=state,
            epoch_accumulator=epoch_accumulator,
            resolved=resolved,
            initial_parameters=initial_parameters,
            metrics_path=metrics_path,
            ledger=ledger,
        )
        atomic_torch_save(path, payload)
        metadata = {
            "checkpoint": path.name,
            "checkpoint_raw_sha256": file_sha256(path),
            **state.to_dict(),
        }
        if path.name == "last.pt":
            atomic_write_json(run_directory / "latest.json", metadata)
            atomic_write_json(
                run_directory / "checkpoint_tensor_manifest.json",
                checkpoint_tensor_report(path),
            )
        if export_encoder:
            encoder_name = (
                "encoder-only-best.pt" if path.name == "best.pt" else "encoder-only-last.pt"
            )
            export = export_encoder_only(
                run_directory / encoder_name,
                model=model,
                encoder_config=resolved["encoder_config"],
                feature_contract=feature_contract,
                bindings=bindings,
                training_state=state.to_dict(),
                training_run_provenance=provenance,
            )
            atomic_write_json(
                run_directory / encoder_name.replace(".pt", ".json"), export
            )

    def persist_runtime(status: str) -> None:
        ledger.persist()
        atomic_write_json(
            run_directory / "runtime_state.json",
            _runtime_payload(
                state=state,
                run_directory=run_directory,
                status=status,
                ledger=ledger,
                spec=spec,
            ),
        )

    if resume is None:
        save_checkpoint(run_directory / "last.pt", export_encoder=True)
    persist_runtime("running")

    try:
        while (
            state.current_epoch < config.epochs
            and not state.stopped_early
            and state.optimizer_step < schedule_contract.total_optimizer_steps
        ):
            requests = training_window_requests(
                lengths,
                train_ids,
                seed=config.seed,
                epoch=state.current_epoch,
                window_length_ticks=config.window_length_ticks,
                window_stride_ticks=config.window_stride_ticks,
            )
            batches = _batch_requests(requests, config.batch_size)
            if len(batches) != resolved["microbatches_per_epoch"]:
                raise ValueError("training loader length changed")
            if state.next_batch_index > len(batches):
                raise ValueError("resume sampler position exceeds the deterministic epoch")
            model.train()
            for batch_index in range(state.next_batch_index, len(batches)):
                request_batch = batches[batch_index]
                items = repository.windows(request_batch)
                encoder, supervision = collate_training_windows(
                    items,
                    expected_profile_id=profile.profile_id,
                    expected_profile_hash=profile.profile_hash,
                    head_config=head_config,
                )
                encoder, supervision = _move_batch(
                    encoder, supervision, spec, config.pin_memory
                )
                with _autocast_context(spec):
                    output = model(encoder.batch)
                    loss = compute_rotation_loss(
                        output.logits, supervision.targets, loss_config
                    )
                _validate_finite_logits_and_loss(
                    output, loss, encoder.metadata.session_ids
                )
                if not _loss_has_supervision(loss):
                    raise TrainingConfigurationError(
                        "the fixed FNCS schedule encountered a zero-supervision batch"
                    )
                components = _loss_components(loss, loss_config)
                (loss.total / config.accumulation_steps).backward()
                state.accumulation_count += 1
                state.supervised_microbatches += 1
                epoch_accumulator.update(loss)
                if state.accumulation_count != config.accumulation_steps:
                    raise ValueError("only accumulation_steps=1 is authorized")
                pre_clip_norm = _gradient_norm_and_finite(
                    model, encoder.metadata.session_ids
                )
                encoder_gradients = _encoder_gradient_report(model)
                if not encoder_gradients["finite"] or not encoder_gradients["nonzero"]:
                    raise TrainingNumericalError("encoder gradients must be finite and nonzero")
                scheduler_record = scheduler.step_record()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer_started = time.perf_counter()
                optimizer.step()
                optimizer_seconds = time.perf_counter() - optimizer_started
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.accumulation_count = 0
                state.optimizer_step += 1
                state.data_microbatches += 1
                state.next_batch_index = batch_index + 1
                state.first_optimizer_step_completed = True
                append_jsonl(
                    metrics_path,
                    {
                        "type": "optimizer_step",
                        "recorded_utc": utc_now(),
                        "epoch": state.current_epoch,
                        "batch_index": batch_index,
                        "optimizer_step": state.optimizer_step,
                        "session_ids": list(encoder.metadata.session_ids),
                        "loss_components": components,
                        "scheduler": scheduler_record,
                        "pre_clip_gradient_norm": pre_clip_norm,
                        "encoder_gradients": encoder_gradients,
                        "gradient_clip_norm": config.gradient_clip_norm,
                        "gradient_clipping_applied": True,
                        "optimizer_step_seconds": optimizer_seconds,
                    },
                )
                access = ledger.report()
                if not access["zero_test_file_attempts"] or not access["zero_test_file_opens"]:
                    raise TrainingConfigurationError("sealed test access was detected")
                durable = (
                    state.optimizer_step == 1
                    or state.optimizer_step % config.checkpoint_every_optimizer_steps == 0
                )
                if durable:
                    save_checkpoint(run_directory / "last.pt", export_encoder=True)
                    periodic = run_directory / f"checkpoint-step-{state.optimizer_step:08d}.pt"
                    if state.optimizer_step % config.checkpoint_every_optimizer_steps == 0:
                        save_checkpoint(periodic, export_encoder=False)
                persist_runtime("running")
                print(
                    json.dumps(
                        {
                            "event": "optimizer_step",
                            "optimizer_step": state.optimizer_step,
                            "epoch": state.current_epoch,
                            "loss": components["total"],
                            "checkpoint_durable": durable,
                            "test_file_attempts": 0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if state.next_batch_index != len(batches):
                raise ValueError("epoch ended before every deterministic batch was consumed")
            completed_epoch = state.current_epoch + 1
            append_jsonl(
                metrics_path,
                {
                    "type": "training_epoch",
                    "recorded_utc": utc_now(),
                    "completed_epoch": completed_epoch,
                    "optimizer_step": state.optimizer_step,
                    "metrics": epoch_accumulator.metrics(),
                },
            )
            state.completed_epochs = completed_epoch
            state.current_epoch = completed_epoch
            state.next_batch_index = 0
            epoch_accumulator = LossMetricAccumulator(loss_config)

            if completed_epoch % config.validation_every_epochs == 0:
                ledger.set_phase("routine_validation", ("train", "validation"))
                metrics, elapsed = _evaluate(
                    model=model,
                    repository=repository,
                    requests=validation_requests,
                    config=config,
                    profile=profile,
                    head_config=head_config,
                    loss_config=loss_config,
                    spec=spec,
                )
                current = metrics.get(config.checkpoint_selection_metric)
                if not isinstance(current, (int, float)) or isinstance(current, bool):
                    raise ValueError("checkpoint selection metric is unavailable")
                current = float(current)
                improved = state.best_metric is None or current < (
                    state.best_metric - config.early_stopping_min_delta
                )
                if improved:
                    state.best_metric = current
                    state.best_optimizer_step = state.optimizer_step
                    state.best_completed_epoch = completed_epoch
                    state.non_improvements = 0
                else:
                    state.non_improvements += 1
                    if state.non_improvements >= config.early_stopping_patience:
                        state.stopped_early = True
                state.last_validation_metrics = metrics
                append_jsonl(
                    metrics_path,
                    {
                        "type": "validation",
                        "recorded_utc": utc_now(),
                        "completed_epoch": completed_epoch,
                        "optimizer_step": state.optimizer_step,
                        "metrics": metrics,
                        "selection_metric": config.checkpoint_selection_metric,
                        "selection_value": current,
                        "improved": improved,
                        "validation_seconds": elapsed,
                        "test_partition_evaluated": False,
                    },
                )
                atomic_write_json(
                    run_directory / "checkpoint_selection_report.json",
                    {
                        "schema_version": "fncs-checkpoint-selection:1.0",
                        "metric": config.checkpoint_selection_metric,
                        "direction": "minimize",
                        "fixed_before_training": True,
                        "current_validation_value": current,
                        "best_validation_value": state.best_metric,
                        "best_optimizer_step": state.best_optimizer_step,
                        "best_completed_epoch": state.best_completed_epoch,
                        "validation_split_only": True,
                        "test_partition_evaluated": False,
                    },
                )
                if improved:
                    save_checkpoint(run_directory / "best.pt", export_encoder=True)
                    atomic_write_json(
                        run_directory / "best.json",
                        {
                            "checkpoint": "best.pt",
                            "checkpoint_raw_sha256": file_sha256(
                                run_directory / "best.pt"
                            ),
                            "metric": config.checkpoint_selection_metric,
                            "value": current,
                            "optimizer_step": state.optimizer_step,
                            "completed_epoch": completed_epoch,
                        },
                    )
                ledger.set_phase("full_training", ("train", "validation"))
            save_checkpoint(run_directory / "last.pt", export_encoder=True)
            persist_runtime("running")

        expected_steps = resolved["total_optimizer_steps"]
        status = "early_stopped" if state.stopped_early else "completed"
        if not state.stopped_early and state.optimizer_step != expected_steps:
            raise ValueError(
                f"completed run has {state.optimizer_step} steps, expected {expected_steps}"
            )
        save_checkpoint(run_directory / "last.pt", export_encoder=True)
        summary = {
            "schema_version": "fncs-encoder-final-summary:1.0",
            "created_utc": utc_now(),
            "status": status,
            **state.to_dict(),
            "total_optimizer_steps_contract": expected_steps,
            "scheduler": scheduler.state_dict(),
            "best_metric_name": config.checkpoint_selection_metric,
            "split_counts": split["counts"],
            "bindings": bindings,
            "fresh_initialization": True,
            "trajectory_decoder_trained": False,
            "test_partition_evaluated": False,
            "test_file_attempts": ledger.report()["test_session_parquet_attempt_count"],
            "test_file_opens": ledger.report()["test_session_parquet_open_count"],
        }
        atomic_write_json(run_directory / "final_summary.json", summary)
        persist_runtime(status)
        write_artifact_manifest(run_directory, status=status)
        return summary
    except Exception as exc:
        optimizer.zero_grad(set_to_none=True)
        atomic_write_json(
            run_directory / "failure.json",
            {
                "schema_version": "fncs-encoder-failure:1.0",
                "failed_utc": utc_now(),
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
                "training_state": state.to_dict(),
                "test_partition_evaluated": False,
            },
        )
        persist_runtime("failed")
        write_artifact_manifest(run_directory, status="failed")
        raise
    finally:
        repository.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fresh FNCS spatiotemporal encoder without a trajectory decoder."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_training(arguments.config, resume=arguments.resume)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
