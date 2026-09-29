from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from fncs_encoder_training.artifacts import environment_report
from fncs_encoder_training.common import (
    atomic_write_json,
    file_sha256,
    seed_everything,
    tensor_state_sha256,
    utc_now,
)
from fortnite_encoder.contracts import EncoderBatch, EncoderOutput, TensorizedMatch
from fortnite_encoder.planner_contracts import PlannerObservation
from fortnite_encoder.planner_supervision import PlannerQuery, PlannerTargetSource
from fortnite_parallel_trajectory.contracts import (
    ParallelTrajectoryBatch,
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
)
from fortnite_parallel_trajectory.losses import route_mixture_nll
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
    _move_inputs,
    resolve_precision,
    sample_training_batch_queries,
)

from .binding import (
    ArchitectureSourceChanged,
    VerifiedSetup,
    assert_frozen_encoder,
    build_frozen_model,
    downstream_parameter_state,
    verify_setup,
)
from .checkpoint import TrainingState, load_checkpoint, save_checkpoint
from .config import FrozenFNCSTrainingConfig, TrainingCompatibilityError
from .data import FNCSParallelTrajectorySessionRepository
from .optimization import (
    PiecewiseLinearScheduler,
    build_optimizer,
    gradient_report,
)


def _run_test_gate(
    *,
    name: str,
    paths: Sequence[str],
    workspace: Path,
    output_directory: Path,
) -> dict[str, Any]:
    temporary = output_directory / "pytest-temporary" / name
    log_path = output_directory / f"pytest-{name}.log"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "-q",
        "--basetemp",
        str(temporary),
        "-o",
        f"cache_dir={temporary / 'cache'}",
    ]
    environment = os.environ.copy()
    source = str(workspace / "ml" / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log = result.stdout + result.stderr
    log_path.write_text(log, encoding="utf-8")
    report = {
        "name": name,
        "command": command,
        "return_code": result.returncode,
        "passed": result.returncode == 0,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
        "log_sha256": file_sha256(log_path),
    }
    if result.returncode != 0:
        raise RuntimeError(f"test gate {name} failed; see {log_path}")
    return report


def _tensor_fields_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if hasattr(left, "__dataclass_fields__"):
        return all(
            _tensor_fields_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _tensor_fields_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _prediction_signature(output: ParallelTrajectoryOutput) -> dict[str, torch.Tensor]:
    return {
        name: getattr(output, name).detach().cpu().clone()
        for name in (
            "mode_logits",
            "mode_probabilities",
            "means",
            "scales",
            "correlations",
            "primary_mode_index",
            "primary_route_normalized",
            "marginal_mean_route_normalized",
            "query_mask",
        )
    }


def _signatures_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _move_examples(
    examples: Sequence[ParallelTrajectoryQueryExample],
    *,
    config: FrozenFNCSTrainingConfig,
    spec: Any,
) -> tuple[EncoderBatch, PlannerObservation, ParallelTrajectoryTargets]:
    batch, observation, targets = collate_parallel_trajectory_query_examples(examples)
    return _move_inputs(
        batch,
        observation,
        targets,
        spec,
        pin_memory=config.pin_memory,
    )


@torch.no_grad()
def _evaluate_signature(
    model: torch.nn.Module,
    examples: Sequence[ParallelTrajectoryQueryExample],
    *,
    config: FrozenFNCSTrainingConfig,
    spec: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    batch, observation, targets = _move_examples(
        examples, config=config, spec=spec
    )
    model.eval()
    with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
        output = model(batch, observation, targets)
    loss = route_mixture_nll(output, targets)
    return _prediction_signature(output), {
        "route_nll": float(loss.loss.detach().float().item()),
        "valid_route_count": loss.valid_route_count,
        "valid_horizon_count": loss.valid_horizon_count,
    }


def _output_contract(output: ParallelTrajectoryOutput) -> dict[str, Any]:
    q = output.query_mask.shape[0]
    expected = {
        "mode_logits": [q, 5],
        "mode_probabilities": [q, 5],
        "means": [q, 5, 12, 2],
        "scales": [q, 5, 12, 2],
        "correlations": [q, 5, 12],
        "query_mask": [q],
    }
    actual = {name: list(getattr(output, name).shape) for name in expected}
    if actual != expected:
        raise RuntimeError(f"parallel decoder output shapes changed: {actual}")
    if not bool(torch.isfinite(output.mode_logits).all()):
        raise RuntimeError("mode logits are nonfinite")
    if not bool(torch.isfinite(output.means).all()):
        raise RuntimeError("trajectory means are nonfinite")
    if not bool(torch.isfinite(output.scales).all()) or bool(
        (output.scales <= 0).any()
    ):
        raise RuntimeError("trajectory scales are not finite and positive")
    if not bool(torch.isfinite(output.correlations).all()) or bool(
        (output.correlations.abs() >= 1).any()
    ):
        raise RuntimeError("trajectory correlations violate abs(rho)<1")
    determinants = (
        output.scales.float()[..., 0].square()
        * output.scales.float()[..., 1].square()
        * (1.0 - output.correlations.float().square())
    )
    if not bool(torch.isfinite(determinants).all()) or bool(
        (determinants <= 0).any()
    ):
        raise RuntimeError("trajectory covariance is not positive-definite")
    return {
        "expected_shapes": expected,
        "actual_shapes": actual,
        "mode_probabilities_sum_to_one": bool(
            torch.allclose(
                output.mode_probabilities.float().sum(dim=-1),
                torch.ones(q, device=output.mode_probabilities.device),
                rtol=1e-5,
                atol=1e-5,
            )
        ),
        "finite_outputs": True,
        "strictly_positive_scales": True,
        "correlation_magnitude_below_one": True,
        "positive_covariance_determinants": True,
    }


def _encoder_determinism(
    model: Any,
    examples: Sequence[ParallelTrajectoryQueryExample],
    *,
    config: FrozenFNCSTrainingConfig,
    spec: Any,
) -> dict[str, Any]:
    batch, observation, _ = _move_examples(examples, config=config, spec=spec)
    model.eval()
    sanitized = model.sanitize(batch, observation)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=spec.autocast_dtype
    ):
        first = model.encoder(sanitized.encoder_batch)
        second = model.encoder(sanitized.encoder_batch)
    if not isinstance(first, EncoderOutput) or not isinstance(second, EncoderOutput):
        raise RuntimeError("encoder output type changed")
    fields_equal = {
        "z": torch.equal(first.z, second.z),
        "team_alive_mask": torch.equal(
            first.team_alive_mask, second.team_alive_mask
        ),
        "time_mask": torch.equal(first.time_mask, second.time_mask),
    }
    if not all(fields_equal.values()):
        raise RuntimeError("frozen encoder output is not bit-deterministic")
    return {
        "identical_repeated_input": True,
        "bit_identical_fields": fields_equal,
        "encoder_eval_mode": not model.encoder.training,
        "encoder_dropout_disabled": True,
        "precision": "bf16",
    }


def _future_mutation_causality(
    *,
    source: PlannerTargetSource,
    query: PlannerQuery,
    model: Any,
    config: FrozenFNCSTrainingConfig,
    spec: Any,
) -> dict[str, Any]:
    position_tensor = (
        source.match.absolute_tick_index == query.absolute_tick_index
    ).nonzero(as_tuple=False)
    if position_tensor.numel() != 1:
        raise RuntimeError("causality query tick is not unique")
    position = int(position_tensor.item())
    if position + 1 >= source.match.num_timesteps:
        raise RuntimeError("causality query has no future state to mutate")
    match = source.match
    future = slice(position + 1, None)
    player_xyz = match.player_xyz_uu.clone()
    player_xyz[future] = player_xyz[future] + 1234.5
    current_circle = match.current_circle_uu.clone()
    current_circle[future] = current_circle[future] - 432.25
    target_circle = match.target_circle_uu.clone()
    target_circle[future] = target_circle[future] + 777.75
    mutated_match = replace(
        match,
        player_xyz_uu=player_xyz,
        current_circle_uu=current_circle,
        target_circle_uu=target_circle,
    )
    mutated_source = replace(source, match=mutated_match)
    original_examples = prepare_parallel_trajectory_query_examples(
        {query.session_id: source},
        (query,),
        context_length_ticks=config.context_length_ticks,
    )
    mutated_examples = prepare_parallel_trajectory_query_examples(
        {query.session_id: mutated_source},
        (query,),
        context_length_ticks=config.context_length_ticks,
    )
    if len(original_examples) != 1 or len(mutated_examples) != 1:
        raise RuntimeError("causality mutation changed query eligibility")
    original_batch, original_observation, original_targets = _move_examples(
        original_examples, config=config, spec=spec
    )
    mutated_batch, mutated_observation, _ = _move_examples(
        mutated_examples, config=config, spec=spec
    )
    if not _tensor_fields_equal(original_batch, mutated_batch):
        raise RuntimeError("future mutation changed causal encoder inputs")
    if not _tensor_fields_equal(original_observation, mutated_observation):
        raise RuntimeError("future mutation changed causal planner observation")
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=spec.autocast_dtype
    ):
        original_output = model(
            original_batch, original_observation, original_targets
        )
        mutated_output = model(
            mutated_batch, mutated_observation, original_targets
        )
    if not _signatures_equal(
        _prediction_signature(original_output),
        _prediction_signature(mutated_output),
    ):
        raise RuntimeError("future replay mutation changed decoder inference")
    return {
        "session_id": query.session_id,
        "team_id": query.team_id,
        "query_absolute_tick": query.absolute_tick_index,
        "mutated_future_fields": [
            "player_xyz_uu",
            "current_circle_uu",
            "target_circle_uu",
        ],
        "causal_encoder_inputs_bit_identical": True,
        "causal_planner_observation_bit_identical": True,
        "decoder_inference_bit_identical": True,
    }


def _real_bf16_optimizer_step(
    *,
    setup: VerifiedSetup,
    model: Any,
    examples: Sequence[ParallelTrajectoryQueryExample],
    optimizer: torch.optim.Optimizer,
    scheduler: PiecewiseLinearScheduler,
    spec: Any,
) -> tuple[dict[str, Any], ParallelTrajectoryMetricAccumulator]:
    config = setup.config
    batch, observation, targets = _move_examples(
        examples, config=config, spec=spec
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    assert_frozen_encoder(model, setup, stage="preflight_before_backward")
    scheduler_record = scheduler.step_record()
    with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
        output = model(batch, observation, targets)
    output_report = _output_contract(output)
    loss = route_mixture_nll(output, targets)
    if (
        loss.valid_route_count <= 0
        or loss.valid_horizon_count <= 0
        or not bool(torch.isfinite(loss.loss))
    ):
        raise RuntimeError("real BF16 route loss is nonfinite or unsupervised")
    loss.loss.backward()
    gradients = gradient_report(model)
    encoder_gradient = gradients["encoder"]
    if (
        encoder_gradient["trainable_parameter_count"] != 0
        or encoder_gradient["parameters_with_gradient"] != 0
        or encoder_gradient["parameters_with_nonzero_gradient"] != 0
    ):
        raise RuntimeError("frozen encoder received a gradient")
    for owner in ("decoder", "congestion_predictor", "congestion_tokenizer"):
        row = gradients[owner]
        if (
            row["trainable_parameter_count"] <= 0
            or row["parameters_with_gradient"] <= 0
            or row["parameters_with_nonzero_gradient"] <= 0
            or row["all_gradients_finite"] is not True
            or row["gradient_norm"] <= 0.0
        ):
            raise RuntimeError(f"downstream gradient gate failed for {owner}: {row}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    pre_clip = float(
        torch.nn.utils.clip_grad_norm_(
            trainable,
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        ).item()
    )
    post_clip = math.sqrt(
        sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in trainable
            if parameter.grad is not None
        )
    )
    if (
        not math.isfinite(pre_clip)
        or pre_clip <= 0.0
        or post_clip > config.gradient_clip_norm + 1e-5
    ):
        raise RuntimeError("global gradient clipping gate failed")
    before = tensor_state_sha256(downstream_parameter_state(model))
    learning_rates_used = [float(group["lr"]) for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    after = tensor_state_sha256(downstream_parameter_state(model))
    if before == after:
        raise RuntimeError("BF16 optimizer step did not update downstream parameters")
    frozen = assert_frozen_encoder(model, setup, stage="preflight_after_optimizer_step")
    metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
    metrics.update(output, targets, loss)
    optimizer.zero_grad(set_to_none=True)
    return {
        "cuda_bf16_forward": True,
        "finite_route_nll": float(loss.loss.detach().float().item()),
        "valid_route_count": loss.valid_route_count,
        "valid_horizon_count": loss.valid_horizon_count,
        "output_contract": output_report,
        "gradient_report": gradients,
        "pre_clip_global_gradient_norm": pre_clip,
        "post_clip_global_gradient_norm": post_clip,
        "gradient_clip_norm": config.gradient_clip_norm,
        "gradient_clipping_verified": True,
        "learning_rates_used": learning_rates_used,
        "scheduler_record": scheduler_record,
        "scheduler_step_index": scheduler.step_index,
        "downstream_parameter_sha256_before": before,
        "downstream_parameter_sha256_after": after,
        "downstream_parameters_updated": True,
        "frozen_encoder": frozen,
    }, metrics


def _tiny_overfit(
    *,
    setup: VerifiedSetup,
    examples: Sequence[ParallelTrajectoryQueryExample],
    spec: Any,
) -> dict[str, Any]:
    config = setup.config
    seed_everything(config.seed + 17)
    model, _ = build_frozen_model(
        setup, device=spec.device, initialization_seed=config.seed + 17
    )
    optimizer, _ = build_optimizer(model, config)
    scheduler = PiecewiseLinearScheduler(optimizer, config)
    batch, observation, targets = _move_examples(
        examples, config=config, spec=spec
    )

    @torch.no_grad()
    def evaluate() -> tuple[float, ParallelTrajectoryOutput]:
        model.eval()
        with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
            output = model(batch, observation, targets)
        loss = route_mixture_nll(output, targets)
        return float(loss.loss.detach().float().item()), output

    initial_nll, _ = evaluate()
    losses: list[float] = []
    for _ in range(64):
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
            output = model(batch, observation, targets)
        loss = route_mixture_nll(output, targets)
        if not bool(torch.isfinite(loss.loss)):
            raise RuntimeError("tiny-overfit loss became nonfinite")
        loss.loss.backward()
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(
            trainable, config.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        assert_frozen_encoder(model, setup, stage="tiny_overfit_step")
        losses.append(float(loss.loss.detach().float().item()))
    final_nll, final_output = evaluate()
    required_decrease = max(0.05, abs(initial_nll) * 0.02)
    if not final_nll <= initial_nll - required_decrease:
        raise RuntimeError(
            "tiny-overfit loss did not decrease materially: "
            f"initial={initial_nll}, final={final_nll}"
        )
    _output_contract(final_output)
    maximum = float(final_output.means.detach().float().abs().max().item())
    if maximum <= 1e-4:
        raise RuntimeError("tiny-overfit trajectories collapsed to stationary zero")
    if maximum >= 100.0:
        raise RuntimeError("tiny-overfit trajectories exploded")
    mean_probabilities = final_output.mode_probabilities.detach().float().mean(dim=0)
    entropy = float(
        -(
            final_output.mode_probabilities.detach().float()
            * torch.log(final_output.mode_probabilities.detach().float().clamp_min(1e-12))
        )
        .sum(dim=-1)
        .mean()
        .item()
    )
    effective_modes = math.exp(entropy)
    if float(mean_probabilities.max().item()) >= 0.999 or effective_modes <= 1.01:
        raise RuntimeError("tiny-overfit immediately collapsed to one route mode")
    return {
        "training_partition_only": True,
        "steps": 64,
        "initial_route_nll": initial_nll,
        "final_route_nll": final_nll,
        "absolute_decrease": initial_nll - final_nll,
        "required_decrease": required_decrease,
        "loss_trace_first": losses[:5],
        "loss_trace_last": losses[-5:],
        "finite_outputs_and_covariances": True,
        "maximum_absolute_normalized_coordinate": maximum,
        "stationary_collapse": False,
        "explosive_collapse": False,
        "single_mode_collapse": False,
        "mean_mode_probabilities": mean_probabilities.cpu().tolist(),
        "mean_mode_entropy_nats": entropy,
        "effective_mode_count": effective_modes,
        "encoder_unchanged": True,
    }


def _artifact_manifest(destination: Path, *, status: str) -> dict[str, Any]:
    artifacts = [
        {
            "path": path.relative_to(destination).as_posix(),
            "size_bytes": path.stat().st_size,
            "raw_sha256": file_sha256(path),
        }
        for path in sorted(destination.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and "pytest-temporary" not in path.parts
    ]
    return {
        "schema_version": "fncs-frozen-decoder-preflight-artifacts:1.0",
        "created_utc": utc_now(),
        "status": status,
        "artifact_count_excluding_manifest": len(artifacts),
        "artifacts": artifacts,
    }


def run_preflight(
    config_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    run_tests: bool = True,
) -> dict[str, Any]:
    config = FrozenFNCSTrainingConfig.from_json(config_path)
    destination = (
        config.preflight_directory
        if output_directory is None
        else Path(output_directory).resolve()
    )
    if destination.exists():
        raise FileExistsError(f"preflight output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    test_gates: list[dict[str, Any]] = []
    repository: FNCSParallelTrajectorySessionRepository | None = None
    smoke_directory = destination / "disposable-smoke"
    try:
        setup = verify_setup(config)
        workspace = Path(__file__).resolve().parents[3]
        if run_tests:
            gates = (
                (
                    "fncs-frozen-training-focused",
                    ("ml/tests/test_parallel_trajectory_fncs_training.py",),
                ),
                (
                    "parallel-trajectory-architecture",
                    (
                        "ml/tests/test_parallel_trajectory.py",
                        "ml/tests/test_parallel_trajectory_overfit.py",
                    ),
                ),
                ("complete-ml-suite", ("ml/tests",)),
            )
            for name, paths in gates:
                test_gates.append(
                    _run_test_gate(
                        name=name,
                        paths=paths,
                        workspace=workspace,
                        output_directory=destination,
                    )
                )
        spec = resolve_precision()
        seed_everything(config.seed)
        repository = FNCSParallelTrajectorySessionRepository(
            paths=setup.session_paths,
            partitions=setup.partition_by_session,
            active_partitions=("train",),
            test_session_ids=setup.split_session_ids["test"],
            profile=setup.profile,
            ledger_path=destination / "data_access.json",
            split_manifest_sha256=config.expected_split_manifest_sha256,
            phase="training_only_disposable_preflight",
        )
        ordered = training_session_order(
            setup.split_session_ids["train"], seed=config.seed, epoch=1
        )
        first_session_batch = session_batches(ordered, config.batch_size)[0]
        sources = {
            session_id: repository.get(session_id)
            for session_id in first_session_batch
        }
        queries = sample_training_batch_queries(
            sources,
            first_session_batch,
            seed=config.seed,
            epoch=1,
            session_batch_index=0,
            queries_per_session=config.queries_per_session,
        )
        expected_query_count = config.batch_size * config.queries_per_session
        if len(queries) != expected_query_count:
            raise RuntimeError(
                f"preflight sampled {len(queries)} queries, expected {expected_query_count}"
            )
        examples = prepare_parallel_trajectory_query_examples(
            sources,
            queries,
            context_length_ticks=config.context_length_ticks,
        )
        if len(examples) != expected_query_count:
            raise RuntimeError("preflight query lost target supervision")
        small_examples = examples[: config.max_queries_per_forward]

        model, initial_downstream = build_frozen_model(setup, device=spec.device)
        initial_frozen = assert_frozen_encoder(
            model, setup, stage="preflight_before_training"
        )
        optimizer, optimizer_contract = build_optimizer(model, config)
        scheduler = PiecewiseLinearScheduler(optimizer, config)
        encoder_determinism = _encoder_determinism(
            model,
            small_examples,
            config=config,
            spec=spec,
        )
        causality = _future_mutation_causality(
            source=sources[queries[0].session_id],
            query=queries[0],
            model=model,
            config=config,
            spec=spec,
        )
        real_step, metrics = _real_bf16_optimizer_step(
            setup=setup,
            model=model,
            examples=small_examples,
            optimizer=optimizer,
            scheduler=scheduler,
            spec=spec,
        )
        if scheduler.step_index != 1:
            raise RuntimeError("scheduler did not advance exactly once")

        smoke_directory.mkdir(parents=True, exist_ok=False)
        checkpoint_path = smoke_directory / "smoke.pt"
        state = TrainingState(
            completed_epochs=0,
            optimizer_step=1,
            current_epoch=1,
            next_batch_index=1,
            first_optimizer_step_completed=True,
        )
        save_checkpoint(
            checkpoint_path,
            setup=setup,
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            state=state,
            initial_downstream_parameters=initial_downstream,
            best_validation_route_nll=None,
            metrics=metrics,
            data_access=repository.report(),
        )
        before, before_loss = _evaluate_signature(
            model, small_examples, config=config, spec=spec
        )
        reloaded_model, reloaded_initial = build_frozen_model(
            setup, device=spec.device
        )
        if (
            reloaded_initial["initial_downstream_parameter_sha256"]
            != initial_downstream["initial_downstream_parameter_sha256"]
        ):
            raise RuntimeError("downstream initialization is not deterministic")
        reloaded_optimizer, reloaded_contract = build_optimizer(
            reloaded_model, config
        )
        reloaded_scheduler = PiecewiseLinearScheduler(reloaded_optimizer, config)
        reloaded_metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
        loaded_state, loaded_best = load_checkpoint(
            checkpoint_path,
            setup=setup,
            model=reloaded_model,
            optimizer=reloaded_optimizer,
            optimizer_contract=reloaded_contract,
            scheduler=reloaded_scheduler,
            initial_downstream_parameters=reloaded_initial,
            metrics=reloaded_metrics,
            restore_rng=False,
        )
        after, after_loss = _evaluate_signature(
            reloaded_model, small_examples, config=config, spec=spec
        )
        if (
            not _signatures_equal(before, after)
            or loaded_state != state
            or loaded_best is not None
        ):
            raise RuntimeError("strict checkpoint reload changed inference or state")
        checkpoint_reload = {
            "checkpoint_schema_version": (
                "fncs-frozen-parallel-trajectory-checkpoint:1.0"
            ),
            "strict_downstream_reload": True,
            "strict_optimizer_reload": True,
            "strict_scheduler_reload": True,
            "frozen_encoder_rebound_from_encoder_only_best": True,
            "evaluation_outputs_bit_identical": True,
            "before": before_loss,
            "after": after_loss,
            "checkpoint_raw_sha256_before_removal": file_sha256(checkpoint_path),
        }
        moving = tuple(
            example
            for example in examples
            if float(
                example.targets.target_displacements[
                    example.targets.target_mask.unsqueeze(-1).expand_as(
                        example.targets.target_displacements
                    )
                ]
                .abs()
                .max()
                .item()
            )
            > 1e-3
        )
        overfit_examples = (moving or tuple(examples))[
            : config.max_queries_per_forward
        ]
        tiny_overfit = _tiny_overfit(
            setup=setup,
            examples=overfit_examples,
            spec=spec,
        )
        access = repository.report()
        if (
            access["zero_validation_attempts"] is not True
            or access["zero_validation_opens"] is not True
            or access["zero_test_attempts"] is not True
            or access["zero_test_opens"] is not True
        ):
            raise RuntimeError("disposable preflight accessed validation or test data")
        shutil.rmtree(smoke_directory)
        test_temporary = destination / "pytest-temporary"
        if test_temporary.exists():
            shutil.rmtree(test_temporary)
        if smoke_directory.exists() or test_temporary.exists():
            raise RuntimeError("disposable preflight artifacts were not removed")
        environment = {
            **environment_report(workspace),
            "precision": "bf16",
            "autocast_dtype": "torch.bfloat16",
            "complete_command_line": [sys.executable, *sys.argv],
        }
        report = {
            "schema_version": "fncs-frozen-decoder-preflight:1.0",
            "status": "passed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "configuration_path": str(config.configuration_path),
            "configuration_raw_sha256": config.configuration_raw_sha256,
            "architecture_source_digest": setup.architecture_source_manifest[
                "source_tree_sha256"
            ],
            "training_source_digest": setup.training_source_manifest[
                "source_tree_sha256"
            ],
            "encoder_source_digest": config.expected_encoder_source_digest,
            "test_gates": test_gates,
            "environment": environment,
            "architecture_contract_unchanged": True,
            "architecture_source_diff": {
                "missing_files": [],
                "unexpected_files": [],
                "changed_files": [],
            },
            "encoder_binding": dict(setup.encoder_binding.report),
            "initial_downstream_parameters": initial_downstream,
            "initial_frozen_encoder": initial_frozen,
            "optimizer_contract": optimizer_contract.to_dict(),
            "scheduler_contract": scheduler.contract(),
            "encoder_determinism": encoder_determinism,
            "causal_future_mutation_invariance": causality,
            "real_bf16_optimizer_step": real_step,
            "checkpoint_reload": checkpoint_reload,
            "tiny_train_only_overfit": tiny_overfit,
            "data_access": access,
            "validation_access_during_disposable_tests": {
                "attempts": 0,
                "opens": 0,
            },
            "test_access": {"attempts": 0, "opens": 0},
            "disposable_artifacts_removed": True,
            "full_training_authorized": True,
            "test_partition_evaluated": False,
            "scientific_caveat": config.scientific_caveat,
        }
        atomic_write_json(destination / "preflight_report.json", report)
        atomic_write_json(
            destination / "artifact_manifest.json",
            _artifact_manifest(destination, status="passed"),
        )
        return report
    except ArchitectureSourceChanged as exc:
        atomic_write_json(
            destination / "architecture_source_diff.json",
            {
                "schema_version": "parallel-trajectory-source-diff:1.0",
                "status": "changed_stop_required",
                "expected_source_digest": exc.expected,
                "actual_source_digest": exc.actual,
                **exc.diff,
            },
        )
        blocked = {
            "schema_version": "fncs-frozen-decoder-blocked-preflight:1.0",
            "status": "blocked",
            "completed_at": datetime.now().astimezone().isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "full_training_authorized": False,
        }
        atomic_write_json(destination / "blocked_preflight.json", blocked)
        atomic_write_json(
            destination / "artifact_manifest.json",
            _artifact_manifest(destination, status="blocked"),
        )
        raise
    except Exception as exc:
        if smoke_directory.exists():
            shutil.rmtree(smoke_directory)
        test_temporary = destination / "pytest-temporary"
        if test_temporary.exists():
            shutil.rmtree(test_temporary)
        blocked = {
            "schema_version": "fncs-frozen-decoder-blocked-preflight:1.0",
            "status": "blocked",
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "configuration_path": str(config.configuration_path),
            "configuration_raw_sha256": config.configuration_raw_sha256,
            "test_gates": test_gates,
            "exception_type": type(exc).__name__,
            "message": str(exc)[:8000],
            "data_access": None if repository is None else repository.report(),
            "full_training_authorized": False,
        }
        atomic_write_json(destination / "blocked_preflight.json", blocked)
        atomic_write_json(
            destination / "artifact_manifest.json",
            _artifact_manifest(destination, status="blocked"),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the training-only frozen-FNCS decoder preflight."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_preflight(
        arguments.config,
        output_directory=arguments.output_directory,
        run_tests=not arguments.skip_tests,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_preflight"]
