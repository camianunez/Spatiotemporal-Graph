from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable, Literal, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch import nn

from fortnite_encoder.contracts import EncoderBatch
from fortnite_encoder.planner_contracts import PlannerObservation
from fortnite_encoder.planner_supervision import PlannerQuery, PlannerTargetSource
from fortnite_parallel_trajectory.contracts import (
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
)
from fortnite_parallel_trajectory.losses import route_mixture_nll
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel

from .checkpoint import (
    TrainingState,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .config import (
    ParallelTrajectoryTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
    TrainingNumericalError,
)
from .data import (
    ParallelTrajectoryQueryExample,
    ParallelTrajectorySessionRepository,
    collate_parallel_trajectory_query_examples,
    eligible_parallel_trajectory_queries,
    prepare_parallel_trajectory_query_examples,
    session_batches,
    stable_seed,
    training_session_order,
    uniform_sample,
)
from .metrics import ParallelTrajectoryMetricAccumulator
from .optimization import (
    OptimizerContract,
    PiecewiseLinearScheduler,
    apply_freeze_policy,
    build_adamw_optimizer,
    gradient_group_report,
)
from .provenance import (
    TransferProof,
    VerifiedTrainingSetup,
    canonical_json,
    file_sha256,
    initialize_model,
    verify_training_setup,
)


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    device: torch.device
    precision: Literal["bf16"] = "bf16"
    autocast_dtype: torch.dtype = torch.bfloat16


def resolve_precision() -> PrecisionSpec:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise TrainingConfigurationError("training supports exactly one process")
    if not torch.cuda.is_available():
        raise TrainingConfigurationError("parallel trajectory training requires CUDA")
    device = torch.device("cuda:0")
    if not torch.cuda.is_bf16_supported():
        raise TrainingConfigurationError("CUDA device 0 does not support BF16")
    name = torch.cuda.get_device_name(device)
    if "RTX 4090" not in name:
        raise TrainingConfigurationError(
            f"training is pinned to RTX 4090, received {name!r}"
        )
    if torch.version.cuda != "13.0":
        raise TrainingConfigurationError(
            f"training is pinned to CUDA 13.0, received {torch.version.cuda!r}"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return PrecisionSpec(device=device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def environment_metadata(spec: PrecisionSpec) -> dict[str, Any]:
    try:
        pyarrow_version = importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError:
        pyarrow_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(spec.device),
        "cuda_device_capability": list(torch.cuda.get_device_capability(spec.device)),
        "device": str(spec.device),
        "precision": spec.precision,
        "autocast_dtype": str(spec.autocast_dtype),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def append_jsonl(path: Path, value: Any) -> None:
    line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    prior = path.read_text(encoding="utf-8") if path.is_file() else ""
    _atomic_write_text(path, prior + line + "\n")


def _pin_encoder_batch(batch: EncoderBatch) -> EncoderBatch:
    return EncoderBatch(
        **{
            field.name: getattr(batch, field.name).pin_memory()
            for field in fields(batch)
        }
    )


def _move_inputs(
    batch: EncoderBatch,
    observation: PlannerObservation,
    targets: ParallelTrajectoryTargets,
    spec: PrecisionSpec,
    *,
    pin_memory: bool,
) -> tuple[EncoderBatch, PlannerObservation, ParallelTrajectoryTargets]:
    if pin_memory:
        batch = _pin_encoder_batch(batch)
        observation = PlannerObservation(
            **{
                field.name: getattr(observation, field.name).pin_memory()
                for field in fields(observation)
            }
        )
        targets = targets.to("cpu")
        targets = ParallelTrajectoryTargets(
            **{
                field.name: (
                    getattr(targets, field.name).pin_memory()
                    if isinstance(getattr(targets, field.name), torch.Tensor)
                    else getattr(targets, field.name)
                )
                for field in fields(targets)
            }
        )
    return (
        batch.to(spec.device, non_blocking=pin_memory),
        observation.to(spec.device, non_blocking=pin_memory),
        targets.to(spec.device, non_blocking=pin_memory),
    )


def _chunks(
    values: Sequence[ParallelTrajectoryQueryExample], width: int
) -> Iterable[Sequence[ParallelTrajectoryQueryExample]]:
    for start in range(0, len(values), width):
        yield values[start : start + width]


def _validate_output_target_alignment(
    output: ParallelTrajectoryOutput, targets: ParallelTrajectoryTargets
) -> None:
    if not torch.equal(output.query_mask, targets.query_mask):
        raise TrainingCompatibilityError(
            "model and full-match target query eligibility differ"
        )
    if output.query_mask.numel() and not bool(output.query_mask.all()):
        raise TrainingCompatibilityError(
            "prepared examples unexpectedly contain an ineligible query"
        )
    for name in ("mode_logits", "means", "scales", "correlations"):
        value = getattr(output, name)
        if not bool(torch.isfinite(value).all()):
            raise TrainingNumericalError(f"nonfinite model output: {name}")
    if bool((output.scales <= 0).any()) or bool(
        (output.correlations.abs() >= 1).any()
    ):
        raise TrainingNumericalError("invalid covariance parameter")


@dataclass(frozen=True, slots=True)
class TrainingBatchResult:
    route_nll: float
    nll_per_valid_horizon: float
    valid_route_count: int
    valid_horizon_count: int
    pre_clip_gradient_norm: float
    post_clip_gradient_norm_bound: float
    gradient_clipped: bool
    gradient_groups: Mapping[str, Mapping[str, Any]]
    learning_rates_used: tuple[float, ...]
    next_learning_rates: tuple[float, ...]
    scheduler_record: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_training_batch(
    *,
    model: ParallelTrajectoryModel,
    examples: Sequence[ParallelTrajectoryQueryExample],
    optimizer: torch.optim.Optimizer,
    scheduler: PiecewiseLinearScheduler,
    config: ParallelTrajectoryTrainingConfig,
    spec: PrecisionSpec,
    metrics: ParallelTrajectoryMetricAccumulator | None = None,
) -> TrainingBatchResult:
    if not examples:
        raise TrainingConfigurationError("training batch has no examples")
    total_valid_routes = sum(
        int((example.targets.query_mask & example.targets.target_mask.any(dim=-1)).sum())
        for example in examples
    )
    total_valid_horizons = sum(
        int(
            (
                example.targets.target_mask
                & example.targets.query_mask.unsqueeze(-1)
            ).sum()
        )
        for example in examples
    )
    if total_valid_routes <= 0 or total_valid_horizons <= 0:
        raise TrainingNumericalError("training batch has no valid route supervision")
    optimizer.zero_grad(set_to_none=True)
    route_nll_sum = 0.0
    observed_routes = 0
    observed_horizons = 0
    for chunk in _chunks(examples, config.max_queries_per_forward):
        batch, observation, targets = collate_parallel_trajectory_query_examples(chunk)
        batch, observation, targets = _move_inputs(
            batch,
            observation,
            targets,
            spec,
            pin_memory=config.pin_memory,
        )
        with torch.autocast(
            device_type="cuda", dtype=spec.autocast_dtype, enabled=True
        ):
            output = model(batch, observation, targets)
        _validate_output_target_alignment(output, targets)
        loss = route_mixture_nll(output, targets)
        if (
            not bool(torch.isfinite(loss.loss))
            or loss.valid_route_count <= 0
            or loss.valid_horizon_count <= 0
        ):
            raise TrainingNumericalError("route-mixture NLL is nonfinite or unsupervised")
        # The sole objective is the exact global valid-route mean across chunks.
        (loss.per_route_nll.sum() / total_valid_routes).backward()
        route_nll_sum += float(loss.per_route_nll.detach().float().sum().item())
        observed_routes += loss.valid_route_count
        observed_horizons += loss.valid_horizon_count
        if metrics is not None:
            metrics.update(output, targets, loss)
    if observed_routes != total_valid_routes or observed_horizons != total_valid_horizons:
        raise TrainingCompatibilityError(
            "chunked loss counts differ from full target supervision counts"
        )
    group_report = gradient_group_report(model)
    intended = {
        name: row
        for name, row in group_report.items()
        if int(row["trainable_parameter_count"]) > 0
    }
    invalid_groups = [
        name
        for name, row in intended.items()
        if not row["all_gradients_finite"]
        or int(row["parameters_with_gradient"]) <= 0
        or not math.isfinite(float(row["gradient_norm"]))
        or float(row["gradient_norm"]) <= 0.0
    ]
    if invalid_groups:
        raise TrainingNumericalError(
            f"intended trainable groups have invalid gradients: {invalid_groups}"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    pre_clip = float(
        torch.nn.utils.clip_grad_norm_(
            trainable,
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        ).item()
    )
    if not math.isfinite(pre_clip) or pre_clip <= 0.0:
        raise TrainingNumericalError("pre-clipping gradient norm is not finite and positive")
    schedule_record = tuple(scheduler.step_record())
    learning_rates_used = tuple(float(group["lr"]) for group in optimizer.param_groups)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return TrainingBatchResult(
        route_nll=route_nll_sum / total_valid_routes,
        nll_per_valid_horizon=route_nll_sum / total_valid_horizons,
        valid_route_count=total_valid_routes,
        valid_horizon_count=total_valid_horizons,
        pre_clip_gradient_norm=pre_clip,
        post_clip_gradient_norm_bound=min(pre_clip, config.gradient_clip_norm),
        gradient_clipped=pre_clip > config.gradient_clip_norm,
        gradient_groups=group_report,
        learning_rates_used=learning_rates_used,
        next_learning_rates=tuple(scheduler.get_last_lr()),
        scheduler_record=schedule_record,
    )


def sample_training_batch_queries(
    sources: Mapping[str, PlannerTargetSource],
    session_ids: Sequence[str],
    *,
    seed: int,
    epoch: int,
    session_batch_index: int,
    queries_per_session: int,
) -> tuple[PlannerQuery, ...]:
    universe = tuple(
        query
        for session_id in session_ids
        for query in eligible_parallel_trajectory_queries(sources[session_id])
    )
    return uniform_sample(
        universe,
        queries_per_session * len(session_ids),
        seed=stable_seed(
            seed,
            "parallel-trajectory-training-query-batch-v1",
            epoch,
            session_batch_index,
            tuple(session_ids),
        ),
    )


def fixed_validation_queries(
    repository: ParallelTrajectorySessionRepository,
    session_ids: Sequence[str],
    *,
    seed: int,
    queries_per_session: int,
) -> tuple[PlannerQuery, ...]:
    selected: list[PlannerQuery] = []
    for session_id in sorted(session_ids):
        source = repository.get(session_id)
        eligible = eligible_parallel_trajectory_queries(source)
        sample = uniform_sample(
            eligible,
            queries_per_session,
            seed=stable_seed(
                seed,
                "parallel-trajectory-validation-query-sample-v1",
                session_id,
            ),
        )
        if len(sample) != queries_per_session:
            raise TrainingConfigurationError(
                f"validation session {session_id} has only {len(sample)} eligible queries"
            )
        selected.extend(sample)
    return tuple(selected)


@torch.no_grad()
def evaluate_validation(
    *,
    model: ParallelTrajectoryModel,
    repository: ParallelTrajectorySessionRepository,
    validation_session_ids: Sequence[str],
    config: ParallelTrajectoryTrainingConfig,
    spec: PrecisionSpec,
    validation_queries: Sequence[PlannerQuery] | None = None,
) -> tuple[dict[str, Any], tuple[PlannerQuery, ...]]:
    queries = tuple(validation_queries) if validation_queries is not None else fixed_validation_queries(
        repository,
        validation_session_ids,
        seed=config.seed,
        queries_per_session=config.validation_queries_per_session,
    )
    expected = len(validation_session_ids) * config.validation_queries_per_session
    if len(queries) != expected:
        raise TrainingCompatibilityError("fixed validation query count changed")
    model.eval()
    accumulator = ParallelTrajectoryMetricAccumulator(repository._profile)
    sources = {
        session_id: repository.get(session_id)
        for session_id in validation_session_ids
    }
    for start in range(0, len(queries), config.max_queries_per_forward):
        chunk_queries = queries[start : start + config.max_queries_per_forward]
        examples = prepare_parallel_trajectory_query_examples(
            sources,
            chunk_queries,
            context_length_ticks=config.context_length_ticks,
        )
        if len(examples) != len(chunk_queries):
            raise TrainingCompatibilityError(
                "fixed validation query lost supervision during preparation"
            )
        batch, observation, targets = collate_parallel_trajectory_query_examples(examples)
        batch, observation, targets = _move_inputs(
            batch,
            observation,
            targets,
            spec,
            pin_memory=config.pin_memory,
        )
        with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
            output = model(batch, observation, targets)
        _validate_output_target_alignment(output, targets)
        loss = route_mixture_nll(output, targets)
        accumulator.update(output, targets, loss)
    metrics = accumulator.metrics()
    if metrics["route_nll"] is None or not math.isfinite(float(metrics["route_nll"])):
        raise TrainingNumericalError("validation route NLL is not finite")
    return metrics, queries


def _partition_map(setup: VerifiedTrainingSetup) -> dict[str, str]:
    result: dict[str, str] = {}
    for partition, session_ids in setup.split_session_ids.items():
        for session_id in session_ids:
            if session_id in result:
                raise TrainingCompatibilityError("session appears in multiple partitions")
            result[session_id] = partition
    if set(result) != set(setup.session_paths):
        raise TrainingCompatibilityError("verified session paths differ from exact split")
    return result


def _resolved_run_contract(
    setup: VerifiedTrainingSetup,
    transfer_proof: TransferProof,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "parallel-trajectory-resolved-training:1.0",
        "training_configuration": setup.config.to_dict(),
        "checkpoint_bindings": setup.checkpoint_bindings(),
        "transfer_proof": transfer_proof.to_dict(),
        "optimizer_configuration": optimizer_contract.to_dict(),
        "scheduler_configuration": scheduler.contract(),
        "environment": dict(environment),
        "validation_sampling": {
            "version": "parallel-trajectory-validation-query-sample-v1",
            "queries_per_session": setup.config.validation_queries_per_session,
            "access_timing": "epoch_boundaries_only",
            "checkpoint_selection_metric": "validation_route_mixture_nll",
        },
        "data_seal": {
            "configuration_construction": ["metadata_only"],
            "preflight_active_partitions": ["train"],
            "full_training_active_partitions": ["train", "validation"],
            "test_partition": "sealed_for_entire_task",
        },
    }


def _runtime_payload(
    *,
    state: TrainingState,
    run_directory: Path,
    status: str,
    data_access: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "parallel-trajectory-runtime:1.0",
        "status": status,
        "process_id": os.getpid(),
        "completed_epochs": state.completed_epochs,
        "current_epoch": state.current_epoch,
        "next_batch_index": state.next_batch_index,
        "optimizer_step": state.optimizer_step,
        "first_optimizer_step_completed": state.first_optimizer_step_completed,
        "last_checkpoint_exists": (run_directory / "last.pt").is_file(),
        "best_checkpoint_exists": (run_directory / "best.pt").is_file(),
        "validation_session_parquet_open_count": data_access[
            "validation_session_parquet_open_count"
        ],
        "test_session_parquet_open_attempt_count": data_access[
            "test_session_parquet_open_attempt_count"
        ],
        "test_session_parquet_open_count": data_access[
            "test_session_parquet_open_count"
        ],
    }


def run_training(
    config: ParallelTrajectoryTrainingConfig | str | Path,
    *,
    resume: Literal["last"] | str | Path | None = None,
) -> dict[str, Any]:
    setup = verify_training_setup(config)
    config = setup.config
    spec = resolve_precision()
    seed_everything(config.seed)
    environment = environment_metadata(spec)
    run_directory = Path(config.run_directory)
    resolved_path = run_directory / "resolved_config.json"
    metrics_path = run_directory / "metrics.jsonl"
    if resume is None:
        if run_directory.exists():
            raise TrainingConfigurationError(
                f"refusing to overwrite existing run directory: {run_directory}"
            )
        run_directory.mkdir(parents=True, exist_ok=False)
    elif not run_directory.is_dir():
        raise TrainingCompatibilityError("resume run directory does not exist")

    model, transfer_proof = initialize_model(setup, device=spec.device)
    optimizer, optimizer_contract = build_adamw_optimizer(model, config)
    scheduler = PiecewiseLinearScheduler(optimizer, config)
    resolved = _resolved_run_contract(
        setup, transfer_proof, optimizer_contract, scheduler, environment
    )
    if resume is None:
        atomic_write_json(run_directory / "environment.json", environment)
        atomic_write_json(resolved_path, resolved)
        atomic_write_json(run_directory / "split_manifest.json", setup.split_manifest)
        atomic_write_json(
            run_directory / "transferred_tensor_manifest.json",
            setup.transfer.transferred_tensor_manifest,
        )
        atomic_write_json(
            run_directory / "transfer_proof.json", transfer_proof.to_dict()
        )
        _atomic_write_text(metrics_path, "")
    else:
        if not resolved_path.is_file():
            raise TrainingCompatibilityError("resume run is missing resolved_config.json")
        try:
            persisted = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingCompatibilityError(
                f"cannot read persisted run configuration: {exc}"
            ) from exc
        if canonical_json(persisted) != canonical_json(resolved):
            raise TrainingCompatibilityError("persisted run configuration changed")

    partitions = _partition_map(setup)
    repository = ParallelTrajectorySessionRepository(
        setup.session_paths,
        partitions,
        active_partitions=("train", "validation"),
        test_session_ids=setup.split_session_ids["test"],
        profile=setup.profile,
    )
    repository.attach_data_access_audit(
        run_directory / "data_access.json",
        split_manifest_sha256=setup.split_manifest_sha256,
    )
    state = TrainingState()
    best_validation_route_nll: float | None = None
    epoch_metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
    if resume is not None:
        resume_path = (
            run_directory / "last.pt" if str(resume) == "last" else Path(resume)
        )
        state, best_validation_route_nll, amp_state = load_training_checkpoint(
            resume_path,
            setup=setup,
            expected_transfer_proof=transfer_proof,
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            metrics=epoch_metrics,
            restore_rng=True,
        )
        if amp_state is not None:
            raise TrainingCompatibilityError("BF16 run unexpectedly contains AMP scaler state")

    def save_last() -> None:
        save_training_checkpoint(
            run_directory / "last.pt",
            setup=setup,
            transfer_proof=transfer_proof,
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            state=state,
            best_validation_route_nll=best_validation_route_nll,
            metrics=epoch_metrics,
            amp_state=None,
        )
        atomic_write_json(
            run_directory / "latest.json",
            {
                "checkpoint": "last.pt",
                "checkpoint_sha256": file_sha256(run_directory / "last.pt"),
                **state.to_dict(),
            },
        )

    if resume is None:
        apply_freeze_policy(model, 1)
        save_last()
    access = repository.report()
    atomic_write_json(
        run_directory / "runtime.json",
        _runtime_payload(
            state=state,
            run_directory=run_directory,
            status="running",
            data_access=access,
        ),
    )

    train_ids = setup.split_session_ids["train"]
    validation_ids = setup.split_session_ids["validation"]
    validation_queries: tuple[PlannerQuery, ...] | None = None
    try:
        while state.optimizer_step < config.max_optimizer_steps:
            epoch = state.current_epoch
            if epoch > config.max_epochs:
                raise TrainingCompatibilityError("epoch counter exceeded fixed contract")
            phase = apply_freeze_policy(model, epoch)
            ordered = training_session_order(train_ids, seed=config.seed, epoch=epoch)
            batches = session_batches(ordered, config.batch_size)
            if len(batches) != config.updates_per_epoch:
                raise TrainingCompatibilityError("deterministic epoch no longer has 43 batches")
            if state.next_batch_index > len(batches):
                raise TrainingCompatibilityError("resume sampler position exceeds epoch")
            model.train()
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
                result = run_training_batch(
                    model=model,
                    examples=examples,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    spec=spec,
                    metrics=epoch_metrics,
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
                        "freeze_phase": phase,
                        **result.to_dict(),
                    },
                )
                if state.optimizer_step == 1:
                    save_last()
                access = repository.report()
                if not access["zero_test_session_parquet_attempts"] or not access[
                    "zero_test_session_parquet_opens"
                ]:
                    raise TrainingCompatibilityError("sealed test access was detected")
                atomic_write_json(
                    run_directory / "runtime.json",
                    _runtime_payload(
                        state=state,
                        run_directory=run_directory,
                        status="running",
                        data_access=access,
                    ),
                )
                print(
                    json.dumps(
                        {
                            "event": "optimizer_step",
                            "optimizer_step": state.optimizer_step,
                            "epoch": epoch,
                            "route_nll": result.route_nll,
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
            started = time.perf_counter()
            validation_metrics, validation_queries = evaluate_validation(
                model=model,
                repository=repository,
                validation_session_ids=validation_ids,
                config=config,
                spec=spec,
                validation_queries=validation_queries,
            )
            validation_seconds = time.perf_counter() - started
            selection_value = float(validation_metrics["route_nll"])
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
            append_jsonl(
                metrics_path,
                {
                    "type": "validation_epoch",
                    "completed_epoch": epoch,
                    "optimizer_step": state.optimizer_step,
                    "metrics": validation_metrics,
                    "selection_metric": "validation_route_mixture_nll",
                    "selection_value": selection_value,
                    "improved": improved,
                    "validation_seconds": validation_seconds,
                    "query_count": len(validation_queries),
                },
            )
            epoch_metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
            if improved:
                save_training_checkpoint(
                    run_directory / "best.pt",
                    setup=setup,
                    transfer_proof=transfer_proof,
                    model=model,
                    optimizer=optimizer,
                    optimizer_contract=optimizer_contract,
                    scheduler=scheduler,
                    state=state,
                    best_validation_route_nll=best_validation_route_nll,
                    metrics=epoch_metrics,
                    amp_state=None,
                )
                atomic_write_json(
                    run_directory / "best.json",
                    {
                        "checkpoint": "best.pt",
                        "checkpoint_sha256": file_sha256(run_directory / "best.pt"),
                        "completed_epoch": epoch,
                        "optimizer_step": state.optimizer_step,
                        "selection_metric": "validation_route_mixture_nll",
                        "value": selection_value,
                        "oracle_metrics_used_for_selection": False,
                    },
                )
            save_last()
            access = repository.report()
            if not access["zero_test_session_parquet_attempts"] or not access[
                "zero_test_session_parquet_opens"
            ]:
                raise TrainingCompatibilityError("sealed test access was detected")
            atomic_write_json(
                run_directory / "runtime.json",
                _runtime_payload(
                    state=state,
                    run_directory=run_directory,
                    status="running",
                    data_access=access,
                ),
            )

        access = repository.report()
        summary = {
            "status": "completed",
            "checkpoint_schema_version": "parallel_trajectory_training_checkpoint_v1",
            "optimizer_steps": state.optimizer_step,
            "completed_epochs": state.completed_epochs,
            "best_validation_route_nll": best_validation_route_nll,
            "selection_metric": "validation_route_mixture_nll",
            "test_partition_evaluated": False,
            "data_access": access,
            "device": str(spec.device),
            "precision": spec.precision,
            "scientific_caveat": (
                "Training completion does not establish empirical validity; a later "
                "locked comparison against static and constant-velocity baselines is required."
            ),
        }
        atomic_write_json(run_directory / "final_summary.json", summary)
        atomic_write_json(
            run_directory / "runtime.json",
            _runtime_payload(
                state=state,
                run_directory=run_directory,
                status="completed",
                data_access=access,
            ),
        )
        return summary
    except Exception as exc:
        access = repository.report()
        atomic_write_json(
            run_directory / "failure.json",
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:2048],
                "training_state": state.to_dict(),
                "data_access": access,
            },
        )
        atomic_write_json(
            run_directory / "runtime.json",
            _runtime_payload(
                state=state,
                run_directory=run_directory,
                status="failed",
                data_access=access,
            ),
        )
        raise


__all__ = [
    "PrecisionSpec",
    "TrainingBatchResult",
    "append_jsonl",
    "atomic_write_json",
    "environment_metadata",
    "evaluate_validation",
    "fixed_validation_queries",
    "resolve_precision",
    "run_training",
    "run_training_batch",
    "sample_training_batch_queries",
    "seed_everything",
]
