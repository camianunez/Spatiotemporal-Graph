from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Literal, Mapping, Sequence

import torch

from fncs_encoder_training.artifacts import environment_report
from fncs_encoder_training.common import (
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    file_sha256,
    seed_everything,
    utc_now,
)
from fortnite_encoder.planner_supervision import PlannerQuery
from fortnite_parallel_trajectory_training.data import (
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

from .binding import (
    VerifiedSetup,
    assert_frozen_encoder,
    build_frozen_model,
    verify_setup,
)
from .checkpoint import TrainingState, load_checkpoint, save_checkpoint
from .config import (
    FROZEN_FNCS_CHECKPOINT_SCHEMA,
    FrozenFNCSTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
)
from .data import FNCSParallelTrajectorySessionRepository
from .optimization import PiecewiseLinearScheduler, build_optimizer


def _artifact_manifest(run_directory: Path, *, status: str) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_directory.iterdir(), key=lambda item: item.name):
        if (
            not path.is_file()
            or path.name == "artifact_manifest.json"
            or path.name.endswith(".tmp")
        ):
            continue
        artifacts.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "raw_sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": "fncs-frozen-decoder-artifact-manifest:1.0",
        "created_utc": utc_now(),
        "status": status,
        "run_directory": str(run_directory),
        "artifact_count_excluding_manifest": len(artifacts),
        "manifest_self_hash_excluded_to_avoid_recursion": True,
        "artifacts": artifacts,
    }


def write_artifact_manifest(run_directory: Path, *, status: str) -> None:
    atomic_write_json(
        run_directory / "artifact_manifest.json",
        _artifact_manifest(run_directory, status=status),
    )


def _runtime_payload(
    *,
    state: TrainingState,
    run_directory: Path,
    data_access: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    last = run_directory / "last.pt"
    best = run_directory / "best.pt"
    return {
        "schema_version": "fncs-frozen-decoder-runtime:1.0",
        "updated_utc": utc_now(),
        "status": status,
        "process_id": os.getpid(),
        "process_is_current_writer": True,
        "completed_epochs": state.completed_epochs,
        "current_epoch": state.current_epoch,
        "next_batch_index": state.next_batch_index,
        "optimizer_step": state.optimizer_step,
        "first_optimizer_step_completed": state.first_optimizer_step_completed,
        "last_checkpoint_exists": last.is_file(),
        "last_checkpoint_sha256": file_sha256(last) if last.is_file() else None,
        "best_checkpoint_exists": best.is_file(),
        "test_session_request_attempt_count": data_access[
            "test_session_request_attempt_count"
        ],
        "test_session_parquet_open_attempt_count": data_access[
            "test_session_parquet_open_attempt_count"
        ],
        "test_session_parquet_open_count": data_access[
            "test_session_parquet_open_count"
        ],
        "zero_test_attempts": data_access["zero_test_attempts"],
        "zero_test_opens": data_access["zero_test_opens"],
        "validation_dataset_open_count": data_access[
            "validation_dataset_open_count"
        ],
    }


def _resolved_config(
    setup: VerifiedSetup,
    *,
    initial_downstream_parameters: Mapping[str, Any],
    optimizer_contract: Mapping[str, Any],
    scheduler_contract: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "fncs-frozen-decoder-resolved-training:1.0",
        "created_utc": utc_now(),
        "configuration_path": str(setup.config.configuration_path),
        "configuration_raw_sha256": setup.config.configuration_raw_sha256,
        "training_configuration": setup.config.to_dict(),
        "checkpoint_bindings": setup.checkpoint_bindings(),
        "encoder_binding": dict(setup.encoder_binding.report),
        "architecture_source_manifest": dict(setup.architecture_source_manifest),
        "training_source_manifest": dict(setup.training_source_manifest),
        "dataset_bindings": dict(setup.dataset_bindings),
        "world_grid_bindings": dict(setup.world_grid_bindings),
        "prior_decoder_training_contract": dict(setup.prior_decoder_contract),
        "initial_downstream_parameters": dict(initial_downstream_parameters),
        "optimizer": dict(optimizer_contract),
        "scheduler": dict(scheduler_contract),
        "environment": dict(environment),
        "command_line": [sys.executable, *sys.argv],
        "frozen_trainable_boundary": {
            "encoder_requires_grad": False,
            "encoder_mode": "eval",
            "encoder_dropout": "disabled",
            "encoder_optimizer_group": False,
            "trainable": [
                "congestion_predictor",
                "congestion_tokenizer",
                "parallel_trajectory_decoder_v1",
                "decoder_output_projections",
            ],
        },
        "causality": {
            "input_policy": "existing leakage-safe planner input policy",
            "input_window": "trailing causal slice ending at query time",
            "future_coordinates_used_only_for_targets": True,
            "future_zone_geometry_in_inputs": False,
            "normalization_fit_from_data": False,
            "out_of_envelope_targets": "masked_without_clamping",
        },
        "objective": {
            "primary": "FP32 route-level five-mode mixture negative log-likelihood",
            "route_weighting": "equal_across_eligible_routes",
            "empty_mask": "differentiable_zero",
            "congestion_auxiliary_objective": "none",
        },
        "validation": {
            "frequency_epochs": setup.config.validation_every_epochs,
            "queries_per_session": setup.config.validation_queries_per_session,
            "selection_metric": setup.config.checkpoint_selection_metric,
            "test_information_used": False,
        },
        "test_partition": {
            "session_count": 115,
            "explicitly_denylisted": True,
            "decoder_test_split_constructed": False,
            "future_final_test": "newly_collected_sessions_only_outside_this_goal",
        },
        "scientific_caveat": setup.config.scientific_caveat,
    }


def _resume_contract_projection(
    raw: Mapping[str, Any],
    *,
    approved_source_transition: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Return the numerical resume contract with launch-only fields removed."""
    projected = json.loads(json.dumps(raw, sort_keys=True))
    if not isinstance(projected, dict):
        raise TrainingCompatibilityError("resolved training contract must be a mapping")
    projected.pop("created_utc", None)
    projected.pop("command_line", None)
    projected.pop("physical_run_directory", None)
    projected.pop("resume_control", None)
    environment = projected.get("environment")
    if isinstance(environment, dict):
        for name in (
            "created_utc",
            "process_id",
            "parent_process_id",
            "complete_command_line",
            "launch_timestamp",
            "launcher",
        ):
            environment.pop(name, None)
        # Source/configuration hashes below bind executable semantics. Repository
        # status is process-launch provenance and can change as recovery evidence
        # is written without changing numerical execution.
        environment.pop("git", None)
    if approved_source_transition is not None:
        previous, current = approved_source_transition
        source = projected.get("training_source_manifest")
        bindings = projected.get("checkpoint_bindings")
        if (
            not isinstance(source, dict)
            or source.get("source_tree_sha256") not in {previous, current}
            or not isinstance(bindings, dict)
            or bindings.get("training_source_digest") not in {previous, current}
        ):
            raise TrainingCompatibilityError(
                "resolved training source is outside the approved transition"
            )
        projected["training_source_manifest"] = {
            "approved_source_transition": [previous, current]
        }
        bindings["training_source_digest"] = {
            "approved_source_transition": [previous, current]
        }
    return projected


def _resume_contract_difference_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        differences: list[str] = []
        for name in sorted(set(left) | set(right)):
            path = f"{prefix}.{name}" if prefix else name
            if name not in left or name not in right:
                differences.append(path)
            else:
                differences.extend(
                    _resume_contract_difference_paths(left[name], right[name], path)
                )
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix or "<root>"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _resume_contract_difference_paths(
                    left_item, right_item, f"{prefix}[{index}]"
                )
            )
        return differences
    return [] if left == right else [prefix or "<root>"]


def validate_resolved_resume_contract(
    persisted: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    approved_source_transition: tuple[str, str] | None = None,
) -> None:
    persisted_contract = _resume_contract_projection(
        persisted, approved_source_transition=approved_source_transition
    )
    current_contract = _resume_contract_projection(
        current, approved_source_transition=approved_source_transition
    )
    if canonical_json(persisted_contract) != canonical_json(current_contract):
        differences = _resume_contract_difference_paths(
            persisted_contract, current_contract
        )
        raise TrainingCompatibilityError(
            "resolved training contract changed: " + ", ".join(differences[:32])
        )


def _append_frozen_proof(
    path: Path,
    *,
    setup: VerifiedSetup,
    model: torch.nn.Module,
    stage: str,
    epoch: int | None = None,
    optimizer_step: int | None = None,
) -> dict[str, Any]:
    snapshot = assert_frozen_encoder(model, setup, stage=stage)  # type: ignore[arg-type]
    snapshot["epoch"] = epoch
    snapshot["optimizer_step"] = optimizer_step
    if path.is_file():
        try:
            proof = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingCompatibilityError(
                f"cannot read frozen-encoder proof: {exc}"
            ) from exc
        if (
            not isinstance(proof, dict)
            or proof.get("schema_version")
            != "fncs-frozen-encoder-proof:1.0"
            or proof.get("canonical_encoder_state_sha256")
            != setup.config.expected_encoder_state_sha256
            or not isinstance(proof.get("snapshots"), list)
        ):
            raise TrainingCompatibilityError("frozen-encoder proof changed")
    else:
        proof = {
            "schema_version": "fncs-frozen-encoder-proof:1.0",
            "canonical_best_pt_sha256": (
                setup.config.expected_canonical_checkpoint_sha256
            ),
            "execution_encoder_only_best_pt_sha256": (
                setup.config.expected_execution_encoder_checkpoint_sha256
            ),
            "canonical_encoder_state_sha256": (
                setup.config.expected_encoder_state_sha256
            ),
            "encoder_tensor_count": setup.config.expected_encoder_tensor_count,
            "hash_frequency": "before_training_after_every_epoch_after_training",
            "all_snapshots_must_match": True,
            "snapshots": [],
        }
    proof["snapshots"].append(snapshot)
    proof["updated_utc"] = utc_now()
    proof["all_recorded_snapshots_match"] = all(
        row.get("matches_canonical") is True
        and row.get("encoder_state_sha256")
        == setup.config.expected_encoder_state_sha256
        for row in proof["snapshots"]
    )
    atomic_write_json(path, proof)
    return snapshot


def _verify_preflight(setup: VerifiedSetup) -> Mapping[str, Any]:
    path = setup.config.preflight_directory / "preflight_report.json"
    if not path.is_file():
        raise TrainingConfigurationError(
            f"passed preflight report is required before training: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingCompatibilityError(f"cannot read preflight report: {exc}") from exc
    if (
        not isinstance(report, dict)
        or report.get("status") != "passed"
        or report.get("full_training_authorized") is not True
        or report.get("configuration_raw_sha256")
        != setup.config.configuration_raw_sha256
        or report.get("training_source_digest")
        != setup.training_source_manifest["source_tree_sha256"]
        or report.get("architecture_source_digest")
        != setup.architecture_source_manifest["source_tree_sha256"]
    ):
        raise TrainingCompatibilityError("preflight authorization is stale or invalid")
    access = report.get("data_access")
    if (
        not isinstance(access, Mapping)
        or access.get("zero_validation_attempts") is not True
        or access.get("zero_validation_opens") is not True
        or access.get("zero_test_attempts") is not True
        or access.get("zero_test_opens") is not True
    ):
        raise TrainingCompatibilityError(
            "preflight used validation or encoder-test data"
        )
    encoder_binding = report.get("encoder_binding")
    real_step = report.get("real_bf16_optimizer_step")
    gradients = real_step.get("gradient_report") if isinstance(real_step, Mapping) else None
    causality = report.get("causal_future_mutation_invariance")
    determinism = report.get("encoder_determinism")
    reload_report = report.get("checkpoint_reload")
    overfit = report.get("tiny_train_only_overfit")
    test_gates = report.get("test_gates")
    if (
        not isinstance(encoder_binding, Mapping)
        or encoder_binding.get("strict") is not True
        or encoder_binding.get(
            "every_encoder_tensor_name_shape_dtype_and_byte_identical"
        )
        is not True
        or encoder_binding.get("auxiliary_prediction_heads_loaded_into_decoder_model")
        is not False
        or not isinstance(real_step, Mapping)
        or real_step.get("cuda_bf16_forward") is not True
        or real_step.get("gradient_clipping_verified") is not True
        or real_step.get("downstream_parameters_updated") is not True
        or not isinstance(gradients, Mapping)
        or gradients.get("encoder", {}).get("parameters_with_gradient") != 0
        or any(
            gradients.get(owner, {}).get("parameters_with_nonzero_gradient", 0)
            <= 0
            for owner in (
                "decoder",
                "congestion_predictor",
                "congestion_tokenizer",
            )
        )
        or not isinstance(causality, Mapping)
        or causality.get("decoder_inference_bit_identical") is not True
        or not isinstance(determinism, Mapping)
        or determinism.get("identical_repeated_input") is not True
        or not isinstance(reload_report, Mapping)
        or reload_report.get("evaluation_outputs_bit_identical") is not True
        or not isinstance(overfit, Mapping)
        or overfit.get("stationary_collapse") is not False
        or overfit.get("explosive_collapse") is not False
        or overfit.get("single_mode_collapse") is not False
        or not isinstance(test_gates, list)
        or not test_gates
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in test_gates
        )
    ):
        raise TrainingCompatibilityError(
            "preflight authorization is missing one or more mandatory gates"
        )
    return report


def _write_latest(
    run_directory: Path, state: TrainingState, checkpoint_name: str
) -> None:
    checkpoint = run_directory / checkpoint_name
    atomic_write_json(
        run_directory / "latest.json",
        {
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": file_sha256(checkpoint),
            **state.to_dict(),
        },
    )


def run_training(
    config_or_path: FrozenFNCSTrainingConfig | str | Path,
    *,
    resume: Literal["last"] | None = None,
) -> dict[str, Any]:
    setup = verify_setup(config_or_path)
    _verify_preflight(setup)
    config = setup.config
    run_directory = config.run_directory
    if resume is None:
        if run_directory.exists():
            raise TrainingConfigurationError(
                f"refusing to overwrite run directory: {run_directory}"
            )
        run_directory.mkdir(parents=True, exist_ok=False)
    elif not run_directory.is_dir():
        raise TrainingConfigurationError("resume run directory is missing")

    seed_everything(config.seed)
    spec = resolve_precision()
    environment = {
        **environment_report(Path(__file__).resolve().parents[3]),
        "precision": "bf16",
        "autocast_dtype": "torch.bfloat16",
        "complete_command_line": [sys.executable, *sys.argv],
    }
    model, initial_downstream = build_frozen_model(setup, device=spec.device)
    optimizer, optimizer_contract = build_optimizer(model, config)
    scheduler = PiecewiseLinearScheduler(optimizer, config)
    resolved = _resolved_config(
        setup,
        initial_downstream_parameters=initial_downstream,
        optimizer_contract=optimizer_contract.to_dict(),
        scheduler_contract=scheduler.contract(),
        environment=environment,
    )
    metrics_path = run_directory / "metrics.jsonl"
    proof_path = run_directory / "frozen_encoder_proof.json"
    if resume is None:
        atomic_write_json(run_directory / "environment.json", environment)
        atomic_write_json(run_directory / "encoder_binding.json", setup.encoder_binding.report)
        atomic_write_json(run_directory / "initial_downstream_parameters.json", initial_downstream)
        atomic_write_json(run_directory / "resolved_config.json", resolved)
        atomic_write_bytes(
            run_directory / "split_manifest.json",
            config.split_manifest.read_bytes(),
        )
        atomic_write_bytes(metrics_path, b"")
    else:
        persisted = json.loads(
            (run_directory / "resolved_config.json").read_text(encoding="utf-8")
        )
        validate_resolved_resume_contract(persisted, resolved)

    repository = FNCSParallelTrajectorySessionRepository(
        paths=setup.session_paths,
        partitions=setup.partition_by_session,
        active_partitions=("train", "validation"),
        test_session_ids=setup.split_session_ids["test"],
        profile=setup.profile,
        ledger_path=run_directory / "data_access.json",
        split_manifest_sha256=config.expected_split_manifest_sha256,
        phase="full_training",
        resume_ledger=resume is not None,
    )
    state = TrainingState()
    best_validation_route_nll: float | None = None
    epoch_metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
    if resume is None:
        _append_frozen_proof(
            proof_path,
            setup=setup,
            model=model,
            stage="before_training",
            epoch=0,
            optimizer_step=0,
        )
    else:
        state, best_validation_route_nll = load_checkpoint(
            run_directory / "last.pt",
            setup=setup,
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            initial_downstream_parameters=initial_downstream,
            metrics=epoch_metrics,
            restore_rng=True,
        )

    def save_last() -> None:
        repository.assert_no_test_access()
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
            metrics=epoch_metrics,
            data_access=repository.report(),
        )
        _write_latest(run_directory, state, "last.pt")

    if resume is None:
        save_last()
    atomic_write_json(
        run_directory / "runtime.json",
        _runtime_payload(
            state=state,
            run_directory=run_directory,
            data_access=repository.report(),
            status="running",
        ),
    )
    write_artifact_manifest(run_directory, status="initialized")

    train_ids = setup.split_session_ids["train"]
    validation_ids = setup.split_session_ids["validation"]
    validation_queries: tuple[PlannerQuery, ...] | None = None
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
            assert_frozen_encoder(model, setup, stage=f"epoch_{epoch}_train_mode")
            for batch_index in range(state.next_batch_index, len(batches)):
                session_ids = batches[batch_index]
                sources = {
                    session_id: repository.get(session_id)
                    for session_id in session_ids
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
                        f"training batch has {len(queries)} queries, "
                        f"expected {expected_queries}"
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
                result = run_training_batch(
                    model=model,
                    examples=examples,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,  # type: ignore[arg-type]
                    spec=spec,
                    metrics=epoch_metrics,
                )
                assert_frozen_encoder(
                    model,
                    setup,
                    stage=f"optimizer_step_{state.optimizer_step + 1}",
                )
                state = replace(
                    state,
                    optimizer_step=state.optimizer_step + 1,
                    next_batch_index=batch_index + 1,
                    first_optimizer_step_completed=True,
                )
                append_jsonl(
                    metrics_path,
                    {
                        "type": "optimizer_step",
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "optimizer_step": state.optimizer_step,
                        "frozen_encoder": True,
                        **result.to_dict(),
                    },
                )
                if state.optimizer_step == 1:
                    _append_frozen_proof(
                        proof_path,
                        setup=setup,
                        model=model,
                        stage="after_first_optimizer_step",
                        epoch=epoch,
                        optimizer_step=state.optimizer_step,
                    )
                    save_last()
                    write_artifact_manifest(
                        run_directory, status="first_optimizer_step_durable"
                    )
                repository.assert_no_test_access()
                atomic_write_json(
                    run_directory / "runtime.json",
                    _runtime_payload(
                        state=state,
                        run_directory=run_directory,
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
                            "encoder_state_sha256": (
                                config.expected_encoder_state_sha256
                            ),
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
                metrics_path,
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
            started = time.perf_counter()
            validation_metrics, validation_queries = evaluate_validation(
                model=model,
                repository=repository,  # type: ignore[arg-type]
                validation_session_ids=validation_ids,
                config=config,  # type: ignore[arg-type]
                spec=spec,
                validation_queries=validation_queries,
            )
            validation_seconds = time.perf_counter() - started
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
                proof_path,
                setup=setup,
                model=model,
                stage="after_epoch",
                epoch=epoch,
                optimizer_step=state.optimizer_step,
            )
            append_jsonl(
                metrics_path,
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
            epoch_metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
            if improved:
                save_checkpoint(
                    run_directory / "best.pt",
                    setup=setup,
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
                    run_directory / "best.json",
                    {
                        "checkpoint": "best.pt",
                        "checkpoint_sha256": file_sha256(
                            run_directory / "best.pt"
                        ),
                        "completed_epoch": epoch,
                        "optimizer_step": state.optimizer_step,
                        "selection_metric": config.checkpoint_selection_metric,
                        "value": selection_value,
                        "oracle_metrics_used_for_selection": False,
                        "test_information_used_for_selection": False,
                    },
                )
            save_last()
            repository.assert_no_test_access()
            atomic_write_json(
                run_directory / "runtime.json",
                _runtime_payload(
                    state=state,
                    run_directory=run_directory,
                    data_access=repository.report(),
                    status="running",
                ),
            )
            write_artifact_manifest(
                run_directory, status=f"completed_epoch_{epoch}"
            )
            model.train()
            assert_frozen_encoder(
                model, setup, stage=f"after_epoch_{epoch}_return_to_train"
            )

        _append_frozen_proof(
            proof_path,
            setup=setup,
            model=model,
            stage="after_training",
            epoch=state.completed_epochs,
            optimizer_step=state.optimizer_step,
        )
        repository.assert_no_test_access()
        access = repository.report()
        summary = {
            "schema_version": "fncs-frozen-decoder-final-summary:1.0",
            "created_utc": utc_now(),
            "status": "completed",
            "checkpoint_schema_version": FROZEN_FNCS_CHECKPOINT_SCHEMA,
            "run_directory": str(run_directory),
            "optimizer_steps": state.optimizer_step,
            "completed_epochs": state.completed_epochs,
            "best_validation_route_nll": best_validation_route_nll,
            "selection_metric": config.checkpoint_selection_metric,
            "canonical_best_pt_sha256": (
                config.expected_canonical_checkpoint_sha256
            ),
            "execution_encoder_only_best_pt_sha256": (
                config.expected_execution_encoder_checkpoint_sha256
            ),
            "encoder_state_sha256": config.expected_encoder_state_sha256,
            "encoder_unchanged": True,
            "data_access": access,
            "test_partition_evaluated": False,
            "future_final_decoder_test_required": (
                "new canonical sessions absent from every encoder and decoder split"
            ),
            "scientific_caveat": config.scientific_caveat,
        }
        atomic_write_json(run_directory / "final_summary.json", summary)
        atomic_write_json(
            run_directory / "runtime.json",
            _runtime_payload(
                state=state,
                run_directory=run_directory,
                data_access=access,
                status="completed",
            ),
        )
        write_artifact_manifest(run_directory, status="completed")
        return summary
    except Exception as exc:
        access = repository.report()
        atomic_write_json(
            run_directory / "failure.json",
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
            run_directory / "runtime.json",
            _runtime_payload(
                state=state,
                run_directory=run_directory,
                data_access=access,
                status="failed",
            ),
        )
        write_artifact_manifest(run_directory, status="failed")
        raise


__all__ = [
    "run_training",
    "validate_resolved_resume_contract",
    "write_artifact_manifest",
]
