from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from fortnite_encoder.config import EncoderConfig, RotationHeadConfig, RotationLossConfig
from fortnite_encoder.losses import compute_rotation_loss
from fortnite_encoder.rotation import RotationModel
from fortnite_encoder.training import (
    _autocast_context,
    _gradient_norm_and_finite,
    _loss_has_supervision,
    _move_batch,
    _validate_finite_logits_and_loss,
    collate_training_windows,
    resolve_precision,
    training_window_requests,
)
from fortnite_encoder.world_grid import WorldGridProfile

from .artifacts import (
    environment_report,
    parameter_report,
    write_artifact_manifest,
)
from .authority import feature_contract, verify_authorities
from .common import (
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    file_sha256,
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
from .dataset import (
    build_split_manifest,
    validate_dataset,
    write_inventory,
    write_split_manifest,
)
from .optimization import (
    PiecewiseLinearScheduler,
    ScheduleContract,
    build_adamw_optimizer,
)
from .runtime import _encoder_gradient_report, _loss_components, _records
from .world_grid_audit import audit_world_grid


PREFLIGHT_SCHEMA_VERSION = "fncs-fresh-encoder-preflight:1.0"
RESOLVED_SCHEMA_VERSION = "fncs-fresh-encoder-resolved-training:1.0"

CORE_TEST_FILES = (
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


def _record_gate(
    gates: list[dict[str, Any]],
    number: int | str,
    name: str,
    evidence: Mapping[str, Any],
) -> None:
    row = {
        "number": number,
        "name": name,
        "status": "passed",
        "completed_utc": utc_now(),
        "evidence": dict(evidence),
    }
    gates.append(row)
    print(json.dumps({"event": "preflight_gate", **row}, sort_keys=True), flush=True)


def _test_gate(workspace: Path, preflight_directory: Path) -> dict[str, Any]:
    ml_root = workspace / "ml"
    missing = [name for name in CORE_TEST_FILES if not (ml_root / name).is_file()]
    if missing:
        raise ValueError(f"required encoder/training tests are missing: {missing}")
    test_temp = preflight_directory / "pytest-temporary"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *CORE_TEST_FILES,
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(test_temp),
    ]
    environment = dict(os.environ)
    source_root = str((workspace / "ml" / "src").resolve())
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=ml_root,
        env=environment,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    stdout_path = preflight_directory / "pytest.stdout.log"
    stderr_path = preflight_directory / "pytest.stderr.log"
    atomic_write_bytes(stdout_path, process.stdout)
    atomic_write_bytes(stderr_path, process.stderr)
    shutil.rmtree(test_temp, ignore_errors=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"encoder/training test gate failed with exit code {process.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )
    return {
        "command": command,
        "working_directory": str(ml_root),
        "test_file_count": len(CORE_TEST_FILES),
        "test_files": list(CORE_TEST_FILES),
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "stdout_raw_sha256": file_sha256(stdout_path),
        "stderr_raw_sha256": file_sha256(stderr_path),
    }


def _tensor_output(value: Any, prefix: str = "output") -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    if isinstance(value, Tensor):
        output[prefix] = value.detach()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            output.update(_tensor_output(getattr(value, field.name), f"{prefix}.{field.name}"))
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            output.update(_tensor_output(value[key], f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            output.update(_tensor_output(item, f"{prefix}.{index}"))
    return output


def _objective_definition(
    authority: Mapping[str, Any],
    config: FreshEncoderConfig,
) -> dict[str, Any]:
    return {
        "schema_version": "fncs-encoder-objective:1.0",
        "fixed_before_training": True,
        "model": "RotationModel",
        "training_only_auxiliary_heads": [
            "future_position",
            "zone_entry",
            "survival",
            "placement",
        ],
        "formula": (
            "L_total = L_position + lambda_entry * L_entry + "
            "lambda_survival * L_survival + lambda_placement * L_placement + "
            "L_regularization"
        ),
        "weights": {
            "position": 1.0,
            "entry": config.entry_loss_weight,
            "survival": config.survival_loss_weight,
            "placement": config.placement_loss_weight,
            "regularization": config.regularization_loss_weight,
        },
        "regularization": {
            "explicit_loss": 0.0,
            "adamw_weight_decay_is_optimizer_level_only": True,
        },
        "reductions": {
            "future_position": (
                "independent masked mean cross entropy for each 15/30/60-second "
                "horizon, then sum the three horizon means"
            ),
            "zone_entry": "masked mean cross entropy",
            "survival": "masked mean binary cross entropy with logits",
            "placement": "masked mean cross entropy",
        },
        "mask_contract": (
            "each target-valid mask is intersected with the RotationModel query mask; "
            "out-of-bounds position targets are invalid/masked and never clamped"
        ),
        "class_contract": authority["class_contract"],
        "head_config": authority["head_config"],
        "loss_config": authority["loss_config"],
        "raw_and_weighted_components_logged_each_optimizer_step": True,
        "trajectory_decoder_present": False,
        "planner_or_congestion_loss_present": False,
    }


def _progress(label: str):  # type: ignore[no-untyped-def]
    last = 0

    def report(current: int, total: int) -> None:
        nonlocal last
        if current == total or current - last >= 25:
            last = current
            print(
                json.dumps(
                    {"event": label, "completed": current, "total": total},
                    sort_keys=True,
                ),
                flush=True,
            )

    return report


def _smoke_preflight(
    *,
    config: FreshEncoderConfig,
    inventory: Mapping[str, Any],
    split: Mapping[str, Any],
    authority: Mapping[str, Any],
    schedule_contract: ScheduleContract,
    ledger_path: Path,
    disposable_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    encoder_config = EncoderConfig(**authority["encoder_config"])
    head_values = dict(authority["head_config"])
    head_values["horizons_seconds"] = tuple(head_values["horizons_seconds"])
    head_config = RotationHeadConfig(**head_values)
    loss_config = RotationLossConfig(**authority["loss_config"])
    spec = resolve_precision(config.device, config.precision)  # type: ignore[arg-type]
    if (
        spec.device != torch.device("cuda:0")
        or spec.precision != "bf16"
        or spec.scaler_enabled
        or not torch.cuda.is_bf16_supported()
        or "RTX 4090" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("preflight requires cuda:0 BF16 on the RTX 4090")
    profile = WorldGridProfile.load(
        config.world_grid_profile,
        expected_hash=config.expected_world_grid_profile_hash,
    )
    records, partition_by_session = _records(
        inventory,
        split,
        dataset_root=Path(config.dataset_root),
        profile=profile,
    )
    train_ids = tuple(split["ordered_session_ids"]["train"])
    validation_ids = tuple(split["ordered_session_ids"]["validation"])
    test_ids = tuple(split["ordered_session_ids"]["test"])
    ledger = DataAccessLedger(
        path=ledger_path,
        partition_by_session=partition_by_session,
        test_session_ids=test_ids,
        split_manifest_sha256=split["split_manifest_sha256"],
        phase="disposable_smoke",
        allowed_partitions=("train",),
    )
    lengths = audited_session_lengths(
        records,
        encoder_config,
        active_session_ids=train_ids,
        ledger=ledger,
    )
    requests = training_window_requests(
        lengths,
        train_ids,
        seed=config.seed,
        epoch=0,
        window_length_ticks=config.window_length_ticks,
        window_stride_ticks=config.window_stride_ticks,
    )
    request_batch = requests[: config.batch_size]
    if len(request_batch) != config.batch_size:
        raise ValueError("training split cannot form the authoritative smoke batch")
    active = {request.session_id for request in request_batch}
    repository = AuditedSessionRepository(
        tuple(record for record in records if record.session_id in active),
        profile,
        head_config,
        config.num_workers,
        False,
        ledger,
    )
    model: RotationModel | None = None
    reloaded_model: RotationModel | None = None
    reconstructed: RotationModel | None = None
    try:
        items = repository.windows(request_batch)
        encoder, supervision = collate_training_windows(
            items,
            expected_profile_id=profile.profile_id,
            expected_profile_hash=profile.profile_hash,
            head_config=head_config,
        )
        encoder, supervision = _move_batch(
            encoder,
            supervision,
            spec,
            config.pin_memory,
        )

        seed_everything(config.seed)
        model = RotationModel(
            encoder_config=encoder_config,
            head_config=head_config,
        ).to(spec.device)
        initial_parameters = parameter_report(model, seed=config.seed)
        if (
            initial_parameters["encoder_initial_parameter_sha256"]
            == authority["authoritative_encoder_state_sha256"]
            or initial_parameters["training_head_initial_parameter_sha256"]
            == authority["authoritative_training_head_state_sha256"]
        ):
            raise ValueError("fresh initialization unexpectedly equals prior checkpoint tensors")
        optimizer, optimizer_contract = build_adamw_optimizer(
            model,
            learning_rate=config.learning_rate,
            betas=config.adamw_betas,
            epsilon=config.adamw_epsilon,
            weight_decay=config.weight_decay,
        )
        scheduler = PiecewiseLinearScheduler(optimizer, schedule_contract)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(spec):
            output = model(encoder.batch)
            loss = compute_rotation_loss(output.logits, supervision.targets, loss_config)
        _validate_finite_logits_and_loss(output, loss, encoder.metadata.session_ids)
        if not _loss_has_supervision(loss):
            raise ValueError("disposable smoke batch has no supervision")
        components = _loss_components(loss, loss_config)
        counts = components["counts"]
        if (
            any(int(value) <= 0 for value in counts["future_position_by_horizon"])
            or int(counts["zone_entry"]) <= 0
            or int(counts["survival"]) <= 0
            or int(counts["placement"]) <= 0
        ):
            raise ValueError("disposable smoke batch does not exercise every enabled loss")
        loss.total.backward()
        complete_gradient_norm = _gradient_norm_and_finite(
            model, encoder.metadata.session_ids
        )
        encoder_gradients_before = _encoder_gradient_report(model)
        if not encoder_gradients_before["finite"] or not encoder_gradients_before["nonzero"]:
            raise ValueError("encoder smoke gradients are not finite and nonzero")
        scheduler_before = scheduler.step_record()
        clip_return = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        encoder_gradients_after = _encoder_gradient_report(model)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler.step_index != 1:
            raise ValueError("disposable scheduler did not complete exactly one step")

        disposable = {
            "schema_version": "fncs-disposable-smoke-checkpoint:1.0",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_contract": optimizer_contract.to_dict(),
            "scheduler_state": scheduler.state_dict(),
            "initial_parameters": initial_parameters,
            "optimizer_step": 1,
        }
        atomic_torch_save(disposable_path, disposable)
        disposable_hash = file_sha256(disposable_path)

        model.eval()
        with torch.no_grad(), _autocast_context(spec):
            expected_output = _tensor_output(model(encoder.batch))
        loaded = torch.load(disposable_path, map_location="cpu", weights_only=False)
        reloaded_model = RotationModel(
            encoder_config=encoder_config,
            head_config=head_config,
        ).to(spec.device)
        reload_optimizer, reload_contract = build_adamw_optimizer(
            reloaded_model,
            learning_rate=config.learning_rate,
            betas=config.adamw_betas,
            epsilon=config.adamw_epsilon,
            weight_decay=config.weight_decay,
        )
        if canonical_json(reload_contract.to_dict()) != canonical_json(
            optimizer_contract.to_dict()
        ):
            raise ValueError("optimizer parameter groups changed during strict reload")
        # Construct the scheduler while the optimizer still carries its fixed
        # peak LRs.  Loading optimizer state next restores the current warmup
        # LR, while the scheduler retains the immutable peak-LR contract.
        reload_scheduler = PiecewiseLinearScheduler(reload_optimizer, schedule_contract)
        incompatible = reloaded_model.load_state_dict(loaded["model_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("strict reload returned incompatible model keys")
        reload_optimizer.load_state_dict(loaded["optimizer_state"])
        reload_scheduler.load_state_dict(loaded["scheduler_state"])
        reloaded_model.eval()
        with torch.no_grad(), _autocast_context(spec):
            actual_output = _tensor_output(reloaded_model(encoder.batch))
        if set(expected_output) != set(actual_output):
            raise ValueError("strict reload output tensor structure changed")
        differing = [
            name
            for name in expected_output
            if not torch.equal(expected_output[name], actual_output[name])
        ]
        if differing:
            raise ValueError(f"strict reload is not bit-identical: {differing[:8]}")
        compared_tensor_count = len(expected_output)

        access = ledger.report()
        if (
            access["validation_session_parquet_attempt_count"] != 0
            or access["validation_session_parquet_open_count"] != 0
            or not access["zero_test_file_attempts"]
            or not access["zero_test_file_opens"]
        ):
            raise ValueError("disposable smoke accessed validation or sealed test Parquet")
        ledger.persist()
        disposable_path.unlink()
        if disposable_path.exists():
            raise ValueError("disposable smoke checkpoint could not be removed")

        del loaded, reload_optimizer, reload_scheduler, expected_output, actual_output
        del model, reloaded_model, optimizer, scheduler, output, loss, encoder, supervision
        model = None
        reloaded_model = None
        gc.collect()
        torch.cuda.empty_cache()
        seed_everything(config.seed)
        reconstructed = RotationModel(
            encoder_config=encoder_config,
            head_config=head_config,
        ).to(spec.device)
        reconstructed_parameters = parameter_report(reconstructed, seed=config.seed)
        if (
            reconstructed_parameters["complete_initial_parameter_sha256"]
            != initial_parameters["complete_initial_parameter_sha256"]
        ):
            raise ValueError("production seed does not reconstruct initial model exactly")
        smoke_report = {
            "schema_version": "fncs-encoder-disposable-smoke:1.0",
            "created_utc": utc_now(),
            "status": "passed",
            "device": str(spec.device),
            "gpu_model": torch.cuda.get_device_name(0),
            "precision": spec.precision,
            "autocast_dtype": str(spec.autocast_dtype),
            "scaler_enabled": spec.scaler_enabled,
            "batch_size": len(request_batch),
            "training_session_ids": list(encoder_id for encoder_id in (
                request.session_id for request in request_batch
            )),
            "window_requests": [dataclasses.asdict(request) for request in request_batch],
            "loss_components": components,
            "all_enabled_losses_finite": True,
            "all_enabled_losses_supervised": True,
            "complete_pre_clip_gradient_norm": complete_gradient_norm,
            "encoder_gradients_before_clip": encoder_gradients_before,
            "gradient_clip_norm": config.gradient_clip_norm,
            "clip_grad_norm_return": float(clip_return.detach().cpu()),
            "encoder_gradients_after_clip": encoder_gradients_after,
            "optimizer_step_completed": True,
            "scheduler_step_completed": True,
            "scheduler_before_step": scheduler_before,
            "scheduler_after_step_index": 1,
            "disposable_checkpoint_raw_sha256": disposable_hash,
            "strict_model_reload": True,
            "strict_optimizer_reload": True,
            "strict_scheduler_reload": True,
            "bit_identical_evaluation_outputs": True,
            "compared_evaluation_tensor_count": compared_tensor_count,
            "disposable_checkpoint_removed": True,
            "validation_file_attempts": access["validation_session_parquet_attempt_count"],
            "validation_file_opens": access["validation_session_parquet_open_count"],
            "test_file_attempts": access["test_session_parquet_attempt_count"],
            "test_file_opens": access["test_session_parquet_open_count"],
            "production_model_reconstructed_from_original_seed": True,
            "reconstructed_initial_parameter_sha256": reconstructed_parameters[
                "complete_initial_parameter_sha256"
            ],
        }
        return (
            smoke_report,
            initial_parameters,
            optimizer_contract.to_dict(),
            access,
        )
    finally:
        repository.close()
        for item in (model, reloaded_model, reconstructed):
            del item
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolved_config(
    *,
    config: FreshEncoderConfig,
    config_path: Path,
    workspace: Path,
    authority: Mapping[str, Any],
    training_source: Mapping[str, Any],
    inventory: Mapping[str, Any],
    split: Mapping[str, Any],
    world_audit: Mapping[str, Any],
    initial_parameters: Mapping[str, Any],
    optimizer_contract: Mapping[str, Any],
    schedule_contract: ScheduleContract,
    environment: Mapping[str, Any],
    objective: Mapping[str, Any],
    preflight_report_sha256: str,
    preflight_directory: Path,
) -> dict[str, Any]:
    microbatches = math.ceil(config.train_sessions / config.batch_size)
    total_steps = microbatches * config.epochs // config.accumulation_steps
    if total_steps != schedule_contract.total_optimizer_steps:
        raise AssertionError("resolved loader and scheduler totals differ")
    result: dict[str, Any] = {
        "schema_version": RESOLVED_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "workspace": str(workspace),
        "run_directory": config.run_directory,
        "preflight_directory": str(preflight_directory),
        "canonical_config_path": str(config_path),
        "canonical_config_raw_sha256": file_sha256(config_path),
        "dataset_root": config.dataset_root,
        "dataset_schema_version": "2.0.0",
        "seed": config.seed,
        "device": config.device,
        "precision": config.precision,
        "fresh_initialization": True,
        "existing_checkpoint_weights_loaded": False,
        "authoritative_checkpoint_inspection_only": True,
        "encoder_config": authority["encoder_config"],
        "head_config": authority["head_config"],
        "loss_config": authority["loss_config"],
        "effective_encoder_dropouts": {
            "feature_projection": 0.1,
            "head_hidden": 0.1,
            "spatial_attention": 0.1,
            "spatial_residual_and_ffn": 0.1,
            "temporal_attention": 0.1,
            "temporal_residual_and_ffn": 0.1,
        },
        "train_session_count": config.train_sessions,
        "validation_session_count": config.validation_sessions,
        "test_session_count": config.test_sessions,
        "batch_size": config.batch_size,
        "accumulation_steps": config.accumulation_steps,
        "epochs": config.epochs,
        "microbatches_per_epoch": microbatches,
        "optimizer_updates_per_epoch": microbatches // config.accumulation_steps,
        "total_optimizer_steps": total_steps,
        "optimizer_contract": dict(optimizer_contract),
        "schedule_contract_constructor": dataclasses.asdict(schedule_contract),
        "scheduler": {
            **schedule_contract.to_dict(),
            "peak_lrs": [config.learning_rate, config.learning_rate],
        },
        "window_length_ticks": config.window_length_ticks,
        "window_stride_ticks": config.window_stride_ticks,
        "sampler": {
            "algorithm": "fortnite_encoder.training.training_window_requests",
            "one_window_per_canonical_training_session_per_epoch": True,
            "session_order": "deterministic SHA256-derived epoch shuffle",
            "window_start": "deterministic SHA256-derived uniform stride-aligned choice",
            "seed": config.seed,
        },
        "validation_every_epochs": config.validation_every_epochs,
        "checkpoint_selection_metric": config.checkpoint_selection_metric,
        "checkpoint_selection_direction": "minimize",
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "checkpoint_every_optimizer_steps": config.checkpoint_every_optimizer_steps,
        "objective": dict(objective),
        "initial_parameters": dict(initial_parameters),
        "bindings": {
            "ingestion_report_sha256": inventory["ingestion_report_sha256"],
            "source_dataset_validation_raw_sha256": inventory[
                "source_dataset_validation_raw_sha256"
            ],
            "dataset_validation_sha256": inventory["dataset_validation_sha256"],
            "split_manifest_sha256": split["split_manifest_sha256"],
            "encoder_source_digest": authority["source_manifest"]["source_tree_sha256"],
            "training_orchestration_source_digest": training_source[
                "source_tree_sha256"
            ],
            "world_grid_profile_hash": world_audit["profile_hash"],
            "world_grid_profile_raw_sha256": world_audit["profile_raw_sha256"],
            "world_grid_publication_audit_sha256": world_audit[
                "publication_audit_sha256"
            ],
            "world_grid_publication_audit_raw_sha256": world_audit[
                "publication_audit_raw_sha256"
            ],
            "world_grid_coordinate_audit_sha256": world_audit[
                "coordinate_audit_sha256"
            ],
            "world_grid_validation_audit_sha256": world_audit[
                "world_grid_validation_audit_sha256"
            ],
            "preflight_report_raw_sha256": preflight_report_sha256,
        },
        "environment": {
            "python_executable": environment["python_executable"],
            "torch": environment["torch"],
            "cuda_runtime": environment["cuda_runtime"],
            "gpu_model": environment["gpu_model"],
            "bf16_supported": environment["bf16_supported"],
            "git": environment["git"],
        },
        "determinism": {
            "python_numpy_torch_seed": config.seed,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "tf32": False,
            "cublas_workspace_config": ":4096:8",
        },
        "test_partition_policy": {
            "sealed": True,
            "preflight_attempts": 0,
            "preflight_opens": 0,
            "routine_evaluation": False,
        },
        "forbidden_components": [
            "fortnite_early_zone",
            "fortnite_parallel_trajectory",
            "autoregressive_route_decoder",
            "congestion_planner",
            "beam_search",
        ],
        "trajectory_decoder_present": False,
        "full_training_authorized": True,
        "resolved_training_config_sha256": "",
    }
    deterministic = dict(result)
    deterministic.pop("created_utc")
    deterministic.pop("resolved_training_config_sha256")
    result["resolved_training_config_sha256"] = sha256_bytes(
        canonical_json(deterministic).encode("utf-8")
    )
    return result


def _publish_run(
    *,
    config: FreshEncoderConfig,
    config_path: Path,
    preflight_directory: Path,
    resolved: Mapping[str, Any],
) -> Path:
    run_directory = Path(config.run_directory)
    if run_directory.exists():
        raise FileExistsError(f"immutable run already exists: {run_directory}")
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{run_directory.name}.staging-",
            dir=run_directory.parent,
        )
    )
    names = (
        "authoritative_hyperparameters.json",
        "data_access_ledger.json",
        "dataset_inventory.json",
        "encoder_source_manifest.json",
        "training_orchestration_source_manifest.json",
        "environment.json",
        "feature_contract.json",
        "initial_parameters.json",
        "objective_definition.json",
        "optimizer_definition.json",
        "parameter_group_manifest.json",
        "preflight_report.json",
        "pytest.stderr.log",
        "pytest.stdout.log",
        "resume_contract.json",
        "scheduler_definition.json",
        "source_difference_report.json",
        "split_manifest.json",
        "world_grid_compatibility_audit.json",
        "world_grid_out_of_bounds.jsonl",
    )
    try:
        atomic_write_bytes(staging / "canonical_config.json", config_path.read_bytes())
        atomic_write_json(staging / "resolved_training_config.json", resolved)
        for name in names:
            source = preflight_directory / name
            if not source.is_file():
                raise FileNotFoundError(f"preflight artifact is missing: {source}")
            shutil.copy2(source, staging / name)
        atomic_write_json(
            staging / "checkpoint_selection_report.json",
            {
                "schema_version": "fncs-checkpoint-selection:1.0",
                "metric": config.checkpoint_selection_metric,
                "direction": "minimize",
                "fixed_before_training": True,
                "validation_split_only": True,
                "best_checkpoint": None,
                "test_partition_evaluated": False,
            },
        )
        atomic_write_json(
            staging / "launch_contract.json",
            {
                "schema_version": "fncs-encoder-launch-contract:1.0",
                "created_utc": utc_now(),
                "command": [
                    sys.executable,
                    "-m",
                    "fncs_encoder_training.runtime",
                    "--config",
                    str(config_path),
                ],
                "working_directory": str(Path(resolved["workspace"])),
                "background_required": True,
                "stdout": str(run_directory / "training.stdout.log"),
                "stderr": str(run_directory / "training.stderr.log"),
                "cuda_visible_device": 0,
                "precision": "bf16",
            },
        )
        staging.rename(run_directory)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    write_artifact_manifest(run_directory, status="preflight_authorized")
    return run_directory


def run_preflight(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    workspace = Path(__file__).resolve().parents[3]
    preflight_directory = Path(config.preflight_directory)
    run_directory = Path(config.run_directory)
    if preflight_directory.exists():
        raise FileExistsError(
            f"immutable preflight directory already exists: {preflight_directory}"
        )
    if run_directory.exists():
        raise FileExistsError(f"immutable run directory already exists: {run_directory}")
    preflight_directory.mkdir(parents=True, exist_ok=False)
    gates: list[dict[str, Any]] = []
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        test_evidence = _test_gate(workspace, preflight_directory)
        _record_gate(gates, 1, "existing_encoder_and_training_tests", test_evidence)

        authority = verify_authorities(config, workspace)
        training_source = source_tree_manifest(
            workspace / "ml" / "src" / "fncs_encoder_training"
        )
        atomic_write_json(
            preflight_directory / "authoritative_hyperparameters.json", authority
        )
        atomic_write_json(
            preflight_directory / "encoder_source_manifest.json",
            authority["source_manifest"],
        )
        atomic_write_json(
            preflight_directory / "source_difference_report.json",
            authority["source_difference"],
        )
        atomic_write_json(
            preflight_directory / "training_orchestration_source_manifest.json",
            training_source,
        )
        _record_gate(
            gates,
            2,
            "immutable_encoder_source_digest",
            {
                "source_digest": authority["source_manifest"]["source_tree_sha256"],
                "source_files": authority["source_manifest"]["file_count"],
                "status": authority["source_difference"]["status"],
                "training_orchestration_source_digest": training_source[
                    "source_tree_sha256"
                ],
            },
        )

        inventory = validate_dataset(
            config.dataset_root,
            config.dataset_validation_path,
            progress=_progress("dataset_validation_progress"),
        )
        write_inventory(preflight_directory / "dataset_inventory.json", inventory)
        _record_gate(
            gates,
            "dataset",
            "authoritative_allowlist_and_all_session_revalidation",
            {
                "accepted": inventory["accepted_canonical_sessions"],
                "rejected": inventory["rejected_replay_payloads"],
                "validated_tables": inventory["validated_table_count"],
                "ingestion_report_sha256": inventory["ingestion_report_sha256"],
                "dataset_validation_sha256": inventory["dataset_validation_sha256"],
            },
        )

        world_audit = audit_world_grid(
            dataset_root=config.dataset_root,
            inventory=inventory,
            profile_path=config.world_grid_profile,
            expected_profile_raw_sha256=config.expected_world_grid_profile_raw_sha256,
            expected_profile_hash=config.expected_world_grid_profile_hash,
            publication_audit_path=config.world_grid_publication_audit,
            expected_audit_raw_sha256=config.expected_world_grid_audit_raw_sha256,
            expected_audit_payload_sha256=config.expected_world_grid_audit_payload_sha256,
            expected_coordinate_audit_sha256=config.expected_world_grid_coordinate_audit_sha256,
            out_of_bounds_path=preflight_directory / "world_grid_out_of_bounds.jsonl",
            progress=_progress("world_grid_audit_progress"),
        )
        atomic_write_json(
            preflight_directory / "world_grid_compatibility_audit.json", world_audit
        )
        _record_gate(
            gates,
            3,
            "new_dataset_bound_world_grid_audit",
            {
                "status": world_audit["status"],
                "sessions": world_audit["accepted_session_count"],
                "out_of_bounds_coordinates": world_audit[
                    "out_of_bounds_coordinate_count"
                ],
                "validation_audit_sha256": world_audit[
                    "world_grid_validation_audit_sha256"
                ],
            },
        )

        split = build_split_manifest(
            inventory,
            seed=config.seed,
            train_count=config.train_sessions,
            validation_count=config.validation_sessions,
            test_count=config.test_sessions,
        )
        write_split_manifest(preflight_directory / "split_manifest.json", split)
        _record_gate(
            gates,
            4,
            "immutable_canonical_session_split",
            {
                "counts": split["counts"],
                "seed": split["seed"],
                "algorithm": split["split_algorithm"],
                "stratification": split["stratification"],
                "split_manifest_sha256": split["split_manifest_sha256"],
            },
        )

        schedule_contract = ScheduleContract.derive(
            math.ceil(config.train_sessions / config.batch_size)
            * config.epochs
            // config.accumulation_steps,
            warmup_fraction=config.scheduler_warmup_fraction,
            constant_fraction=config.scheduler_constant_fraction,
            linear_decay_fraction=config.scheduler_decay_fraction,
            final_lr_ratio=config.scheduler_final_lr_ratio,
        )
        smoke, initial, optimizer_contract, access = _smoke_preflight(
            config=config,
            inventory=inventory,
            split=split,
            authority=authority,
            schedule_contract=schedule_contract,
            ledger_path=preflight_directory / "data_access_ledger.json",
            disposable_path=preflight_directory / "disposable-smoke.pt",
        )
        atomic_write_json(preflight_directory / "disposable_smoke_report.json", smoke)
        _record_gate(
            gates,
            5,
            "one_training_only_batch_loaded",
            {"session_ids": smoke["training_session_ids"], "batch_size": smoke["batch_size"]},
        )
        _record_gate(
            gates,
            6,
            "cuda_bf16_forward_and_backward",
            {"gpu": smoke["gpu_model"], "precision": smoke["precision"]},
        )
        _record_gate(
            gates,
            7,
            "every_enabled_loss_finite",
            {"counts": smoke["loss_components"]["counts"]},
        )
        _record_gate(
            gates,
            8,
            "finite_nonzero_encoder_gradients",
            smoke["encoder_gradients_before_clip"],
        )
        _record_gate(
            gates,
            9,
            "configured_gradient_clipping_applied",
            {
                "clip_norm": smoke["gradient_clip_norm"],
                "pre_clip_norm": smoke["clip_grad_norm_return"],
            },
        )
        _record_gate(
            gates,
            10,
            "one_optimizer_and_scheduler_step",
            {"optimizer_step": 1, "scheduler_step": 1},
        )
        _record_gate(
            gates,
            11,
            "disposable_checkpoint_strict_reload",
            {"checkpoint_sha256": smoke["disposable_checkpoint_raw_sha256"]},
        )
        _record_gate(
            gates,
            12,
            "bit_identical_eval_outputs_after_reload",
            {"passed": smoke["bit_identical_evaluation_outputs"]},
        )
        _record_gate(
            gates,
            13,
            "zero_validation_and_test_access_during_smoke",
            {
                "validation_attempts": smoke["validation_file_attempts"],
                "validation_opens": smoke["validation_file_opens"],
                "test_attempts": smoke["test_file_attempts"],
                "test_opens": smoke["test_file_opens"],
            },
        )
        _record_gate(
            gates,
            14,
            "disposable_smoke_artifact_removed",
            {"removed": smoke["disposable_checkpoint_removed"]},
        )
        _record_gate(
            gates,
            15,
            "production_model_reconstructed_from_original_seed",
            {
                "initial_parameter_sha256": initial[
                    "complete_initial_parameter_sha256"
                ],
                "reconstructed": True,
            },
        )

        objective = _objective_definition(authority, config)
        feature = feature_contract(authority)
        environment = environment_report(workspace)
        atomic_write_json(preflight_directory / "canonical_config.json", json.loads(config_path.read_text(encoding="utf-8")))
        atomic_write_json(preflight_directory / "feature_contract.json", feature)
        atomic_write_json(preflight_directory / "objective_definition.json", objective)
        atomic_write_json(preflight_directory / "initial_parameters.json", initial)
        atomic_write_json(preflight_directory / "environment.json", environment)
        atomic_write_json(
            preflight_directory / "optimizer_definition.json", optimizer_contract
        )
        atomic_write_json(
            preflight_directory / "parameter_group_manifest.json",
            {
                "schema_version": "fncs-optimizer-parameter-groups:1.0",
                "groups": optimizer_contract["group_parameter_names"],
            },
        )
        atomic_write_json(
            preflight_directory / "scheduler_definition.json",
            schedule_contract.to_dict(),
        )
        atomic_write_json(
            preflight_directory / "resume_contract.json",
            {
                "schema_version": "fncs-encoder-resume-contract:1.0",
                "checkpoint": "last.pt",
                "strict_model_state": True,
                "state": [
                    "complete RotationModel tensors",
                    "AdamW state and exact parameter groups",
                    "piecewise scheduler state and update index",
                    "epoch and next deterministic batch index",
                    "optimizer and microbatch counters",
                    "Python, NumPy, CPU Torch, and all CUDA RNG states",
                    "sampler seed/epoch/batch position",
                    "metrics JSONL durable prefix",
                    "data-access ledger",
                ],
                "pending_accumulation_supported": False,
                "reason": "fixed accumulation_steps is exactly one",
                "compatibility_bindings_required": True,
                "test_partition_evaluated": False,
            },
        )

        preflight_report = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "status": "passed",
            "full_training_authorized": True,
            "elapsed_seconds": time.perf_counter() - started,
            "gates": gates,
            "all_required_gates_passed": len(gates) == 16,
            "source_digest": authority["source_manifest"]["source_tree_sha256"],
            "training_orchestration_source_digest": training_source[
                "source_tree_sha256"
            ],
            "split_counts": split["counts"],
            "scheduler": schedule_contract.to_dict(),
            "initial_parameter_sha256": initial["complete_initial_parameter_sha256"],
            "bindings": {
                "ingestion_report_sha256": inventory["ingestion_report_sha256"],
                "dataset_validation_sha256": inventory["dataset_validation_sha256"],
                "split_manifest_sha256": split["split_manifest_sha256"],
                "world_grid_validation_audit_sha256": world_audit[
                    "world_grid_validation_audit_sha256"
                ],
            },
            "data_access": access,
            "trajectory_decoder_present": False,
        }
        atomic_write_json(preflight_directory / "preflight_report.json", preflight_report)
        preflight_report_raw_sha256 = file_sha256(
            preflight_directory / "preflight_report.json"
        )
        resolved = _resolved_config(
            config=config,
            config_path=config_path,
            workspace=workspace,
            authority=authority,
            training_source=training_source,
            inventory=inventory,
            split=split,
            world_audit=world_audit,
            initial_parameters=initial,
            optimizer_contract=optimizer_contract,
            schedule_contract=schedule_contract,
            environment=environment,
            objective=objective,
            preflight_report_sha256=preflight_report_raw_sha256,
            preflight_directory=preflight_directory,
        )
        atomic_write_json(preflight_directory / "resolved_training_config.json", resolved)
        write_artifact_manifest(preflight_directory, status="passed")
        published = _publish_run(
            config=config,
            config_path=config_path,
            preflight_directory=preflight_directory,
            resolved=resolved,
        )
        result = {
            "status": "passed",
            "full_training_authorized": True,
            "preflight_directory": str(preflight_directory),
            "run_directory": str(published),
            "resolved_training_config_sha256": resolved[
                "resolved_training_config_sha256"
            ],
            "total_optimizer_steps": resolved["total_optimizer_steps"],
            "scheduler": resolved["scheduler"],
            "split_counts": split["counts"],
            "bindings": resolved["bindings"],
        }
        print(json.dumps({"event": "preflight_complete", **result}, sort_keys=True), flush=True)
        return result
    except Exception as exc:
        failure = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "created_utc": utc_now(),
            "status": "failed",
            "full_training_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
            "passed_gates": gates,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:4000],
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(preflight_directory / "preflight_failure.json", failure)
        write_artifact_manifest(preflight_directory, status="failed")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every gate for the fresh FNCS encoder training run."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_preflight(arguments.config)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
