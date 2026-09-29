from __future__ import annotations

import argparse
from dataclasses import asdict, fields
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

from fortnite_parallel_trajectory.contracts import ParallelTrajectoryOutput
from fortnite_parallel_trajectory.losses import route_mixture_nll

from .checkpoint import TrainingState, load_training_checkpoint, save_training_checkpoint
from .config import ParallelTrajectoryTrainingConfig, TrainingCompatibilityError
from .data import (
    ParallelTrajectoryQueryExample,
    ParallelTrajectorySessionRepository,
    collate_parallel_trajectory_query_examples,
    prepare_parallel_trajectory_query_examples,
    session_batches,
    training_session_order,
)
from .metrics import ParallelTrajectoryMetricAccumulator
from .optimization import (
    PiecewiseLinearScheduler,
    apply_freeze_policy,
    build_adamw_optimizer,
)
from .provenance import (
    file_sha256,
    initialize_model,
    package_source_digest,
    verify_training_setup,
)
from .training import (
    PrecisionSpec,
    _move_inputs,
    atomic_write_json,
    environment_metadata,
    resolve_precision,
    run_training_batch,
    sample_training_batch_queries,
    seed_everything,
)


def _run_test_gate(
    *,
    name: str,
    paths: Sequence[str],
    repository_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    base = output_directory / "pytest-temporary" / name
    log_path = output_directory / f"pytest-{name}.log"
    base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "-q",
        "--basetemp",
        str(base),
        "-o",
        f"cache_dir={base / 'cache'}",
    ]
    environment = dict(os.environ)
    ml_source = str(repository_root / "ml" / "src")
    environment["PYTHONPATH"] = (
        ml_source
        if not environment.get("PYTHONPATH")
        else ml_source + os.pathsep + environment["PYTHONPATH"]
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    result = {
        "name": name,
        "command": command,
        "return_code": completed.returncode,
        "passed": completed.returncode == 0,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path.resolve()),
        "log_sha256": file_sha256(log_path),
    }
    if completed.returncode != 0:
        tail = completed.stdout[-4000:]
        raise RuntimeError(f"pytest gate {name!r} failed:\n{tail}")
    return result


def _partition_map(setup: Any) -> dict[str, str]:
    return {
        session_id: partition
        for partition, session_ids in setup.split_session_ids.items()
        for session_id in session_ids
    }


@torch.no_grad()
def _evaluation_signature(
    model: torch.nn.Module,
    examples: Sequence[ParallelTrajectoryQueryExample],
    config: ParallelTrajectoryTrainingConfig,
    spec: PrecisionSpec,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model.eval()
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
    loss = route_mixture_nll(output, targets)
    signature = {
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
    return signature, {
        "route_nll": float(loss.loss.detach().float().item()),
        "valid_route_count": loss.valid_route_count,
        "valid_horizon_count": loss.valid_horizon_count,
    }


def _strict_signature_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _optimizer_group_gradient_report(
    optimizer: torch.optim.Optimizer,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{index}"))
        trainable = [parameter for parameter in group["params"] if parameter.requires_grad]
        gradients = [
            parameter.grad for parameter in trainable if parameter.grad is not None
        ]
        squared = sum(
            float(value.detach().float().square().sum().item()) for value in gradients
        )
        report[name] = {
            "trainable_parameter_count": len(trainable),
            "parameters_with_gradient": len(gradients),
            "all_gradients_finite": all(
                bool(torch.isfinite(value).all()) for value in gradients
            ),
            "gradient_norm": math.sqrt(squared),
        }
    return report


def _tiny_overfit(
    *,
    setup: Any,
    examples: Sequence[ParallelTrajectoryQueryExample],
    spec: PrecisionSpec,
) -> dict[str, Any]:
    config = setup.config
    seed_everything(config.seed + 17)
    model, _ = initialize_model(setup, device=spec.device, seed=config.seed + 17)
    optimizer, _ = build_adamw_optimizer(model, config)
    scheduler = PiecewiseLinearScheduler(optimizer, config)
    apply_freeze_policy(model, 1)
    batch, observation, targets = collate_parallel_trajectory_query_examples(examples)
    batch, observation, targets = _move_inputs(
        batch,
        observation,
        targets,
        spec,
        pin_memory=config.pin_memory,
    )
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=spec.autocast_dtype
    ):
        prepared = model.prepare(batch, observation)

    @torch.no_grad()
    def evaluate() -> tuple[float, ParallelTrajectoryOutput]:
        model.eval()
        with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
            output = model.forward_prepared(prepared, targets)
        loss = route_mixture_nll(output, targets)
        return float(loss.loss.detach().float().item()), output

    initial_nll, _ = evaluate()
    losses: list[float] = []
    # Evaluation mode intentionally disables dropout while retaining gradients,
    # making this fixed real-data decoder overfit bit-reproducible.
    for _ in range(64):
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
            output = model.forward_prepared(prepared, targets)
        loss = route_mixture_nll(output, targets)
        if not bool(torch.isfinite(loss.loss)):
            raise RuntimeError("tiny-overfit route NLL became nonfinite")
        loss.loss.backward()
        decoder_parameters = [
            parameter for parameter in model.decoder.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(
            decoder_parameters,
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.loss.detach().float().item()))
    final_nll, final_output = evaluate()
    required_decrease = max(0.05, abs(initial_nll) * 0.02)
    if not final_nll <= initial_nll - required_decrease:
        raise RuntimeError(
            "tiny-overfit route NLL did not decrease materially: "
            f"initial={initial_nll}, final={final_nll}, required={required_decrease}"
        )
    tensors = (
        final_output.primary_route_normalized,
        final_output.marginal_mean_route_normalized,
        final_output.means,
        final_output.scales,
        final_output.correlations,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise RuntimeError("tiny-overfit output contains nonfinite trajectories")
    determinants = (
        final_output.scales.float()[..., 0].square()
        * final_output.scales.float()[..., 1].square()
        * (1.0 - final_output.correlations.float().square())
    )
    if not bool(torch.isfinite(determinants).all()) or bool((determinants <= 0).any()):
        raise RuntimeError("tiny-overfit covariance is not finite positive-definite")
    maximum = float(final_output.means.detach().float().abs().max().item())
    movement = torch.linalg.vector_norm(
        final_output.means.detach().float(), dim=-1
    )
    if maximum <= 1e-4:
        raise RuntimeError("all five tiny-overfit routes collapsed to stationary zero")
    if maximum >= 100.0:
        raise RuntimeError("tiny-overfit routes exhibit explosive normalized coordinates")
    accumulator = ParallelTrajectoryMetricAccumulator(setup.profile)
    accumulator.update(final_output, targets, route_mixture_nll(final_output, targets))
    diagnostics = accumulator.metrics()
    return {
        "steps": 64,
        "initial_route_nll": initial_nll,
        "final_route_nll": final_nll,
        "absolute_decrease": initial_nll - final_nll,
        "relative_decrease": (
            (initial_nll - final_nll) / abs(initial_nll)
            if initial_nll != 0.0
            else None
        ),
        "finite_primary_marginal_and_all_modes": True,
        "finite_positive_covariances": True,
        "maximum_absolute_normalized_coordinate": maximum,
        "maximum_mode_route_radius_normalized": float(movement.max().item()),
        "universal_stationary_collapse": False,
        "explosive_coordinate_collapse": False,
        "mode_utilization": diagnostics["mode_utilization"],
        "loss_trace_first": losses[:5],
        "loss_trace_last": losses[-5:],
    }


def run_preflight(
    config_path: str | Path,
    *,
    output_directory: str | Path,
    run_tests: bool = True,
) -> dict[str, Any]:
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"preflight output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    gate_results: list[dict[str, Any]] = []
    repository: ParallelTrajectorySessionRepository | None = None
    smoke_directory = destination / "disposable-smoke"
    try:
        setup = verify_training_setup(config_path)
        config = setup.config
        repository_root = Path(__file__).resolve().parents[3]
        if run_tests:
            test_gates = (
                (
                    "focused-training-integration",
                    ("ml/tests/test_parallel_trajectory_training.py",),
                ),
                (
                    "architecture-package",
                    (
                        "ml/tests/test_parallel_trajectory.py",
                        "ml/tests/test_parallel_trajectory_overfit.py",
                    ),
                ),
                (
                    "encoder-planner-regression",
                    (
                        "ml/tests/test_model.py",
                        "ml/tests/test_training.py",
                        "ml/tests/test_planner.py",
                        "ml/tests/test_planner_policy.py",
                        "ml/tests/test_planner_supervision.py",
                        "ml/tests/test_planner_training.py",
                    ),
                ),
                ("complete-ml-suite", ("ml/tests",)),
            )
            for name, paths in test_gates:
                gate_results.append(
                    _run_test_gate(
                        name=name,
                        paths=paths,
                        repository_root=repository_root,
                        output_directory=destination,
                    )
                )

        architecture_root = repository_root / "ml" / "src" / "fortnite_parallel_trajectory"
        encoder_root = repository_root / "ml" / "src" / "fortnite_encoder"
        early_zone_root = repository_root / "ml" / "src" / "fortnite_early_zone"
        training_root = Path(__file__).resolve().parent
        protected = {
            "architecture_package_digest": package_source_digest(architecture_root),
            "encoder_package_digest": package_source_digest(encoder_root),
            "early_zone_package_digest": package_source_digest(early_zone_root),
            "training_package_digest": package_source_digest(training_root),
        }
        expected = {
            "architecture_package_digest": config.architecture_package_digest,
            "encoder_package_digest": config.current_encoder_digest,
            "early_zone_package_digest": "08ee0e99c13be0ebcc1e855728bd710417c9709ffc94007baa41c50e8efd74dc",
            "training_package_digest": config.training_package_digest,
        }
        if protected != expected:
            raise TrainingCompatibilityError(
                f"protected package digest mismatch: actual={protected}, expected={expected}"
            )

        spec = resolve_precision()
        seed_everything(config.seed)
        partitions = _partition_map(setup)
        repository = ParallelTrajectorySessionRepository(
            setup.session_paths,
            partitions,
            active_partitions=("train",),
            test_session_ids=setup.split_session_ids["test"],
            profile=setup.profile,
        )
        repository.attach_data_access_audit(
            destination / "data_access.json",
            split_manifest_sha256=setup.split_manifest_sha256,
        )
        ordered = training_session_order(
            setup.split_session_ids["train"], seed=config.seed, epoch=1
        )
        first_session_batch = session_batches(ordered, config.batch_size)[0]
        sources = {session_id: repository.get(session_id) for session_id in first_session_batch}
        queries = sample_training_batch_queries(
            sources,
            first_session_batch,
            seed=config.seed,
            epoch=1,
            session_batch_index=0,
            queries_per_session=config.queries_per_session,
        )
        expected_queries = config.batch_size * config.queries_per_session
        if len(queries) != expected_queries:
            raise RuntimeError("real preflight batch did not satisfy the exact query contract")
        examples = prepare_parallel_trajectory_query_examples(
            sources, queries, context_length_ticks=config.context_length_ticks
        )
        if len(examples) != expected_queries:
            raise RuntimeError("real preflight examples lost supervision")

        model, transfer_proof = initialize_model(setup, device=spec.device)
        optimizer, optimizer_contract = build_adamw_optimizer(model, config)
        scheduler = PiecewiseLinearScheduler(optimizer, config)
        freeze_phase = apply_freeze_policy(model, 1)
        metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
        # Capture the six optimizer-group gradients before the production helper
        # clears them by performing a duplicate, no-step backward on four queries.
        small_examples = examples[: config.max_queries_per_forward]
        batch, observation, targets = collate_parallel_trajectory_query_examples(small_examples)
        batch, observation, targets = _move_inputs(
            batch, observation, targets, spec, pin_memory=config.pin_memory
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast(device_type="cuda", dtype=spec.autocast_dtype):
            probe_output = model(batch, observation, targets)
        probe_loss = route_mixture_nll(probe_output, targets)
        probe_loss.loss.backward()
        optimizer_group_gradients = _optimizer_group_gradient_report(optimizer)
        for group_name in ("decoder_decay", "decoder_no_decay"):
            row = optimizer_group_gradients[group_name]
            if (
                row["parameters_with_gradient"] <= 0
                or not row["all_gradients_finite"]
                or row["gradient_norm"] <= 0.0
            ):
                raise RuntimeError(f"preflight gradient gate failed for {group_name}")
        optimizer.zero_grad(set_to_none=True)
        batch_result = run_training_batch(
            model=model,
            examples=examples,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            spec=spec,
            metrics=metrics,
        )
        if scheduler.step_index != 1:
            raise RuntimeError("preflight scheduler did not advance exactly once")
        if batch_result.valid_route_count <= 0 or batch_result.valid_horizon_count <= 0:
            raise RuntimeError("preflight batch had no positive supervision")
        if not math.isfinite(batch_result.route_nll):
            raise RuntimeError("preflight route NLL is nonfinite")

        smoke_directory.mkdir(parents=True, exist_ok=False)
        checkpoint_path = smoke_directory / "smoke.pt"
        state = TrainingState(
            completed_epochs=0,
            optimizer_step=1,
            current_epoch=1,
            next_batch_index=1,
            first_optimizer_step_completed=True,
        )
        save_training_checkpoint(
            checkpoint_path,
            setup=setup,
            transfer_proof=transfer_proof,
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            state=state,
            best_validation_route_nll=None,
            metrics=metrics,
            amp_state=None,
        )
        before, before_loss = _evaluation_signature(
            model, small_examples, config, spec
        )
        reloaded_model, reloaded_proof = initialize_model(setup, device=spec.device)
        reloaded_optimizer, reloaded_contract = build_adamw_optimizer(
            reloaded_model, config
        )
        reloaded_scheduler = PiecewiseLinearScheduler(reloaded_optimizer, config)
        apply_freeze_policy(reloaded_model, 1)
        reloaded_metrics = ParallelTrajectoryMetricAccumulator(setup.profile)
        reloaded_state, reloaded_best, reloaded_amp = load_training_checkpoint(
            checkpoint_path,
            setup=setup,
            expected_transfer_proof=reloaded_proof,
            model=reloaded_model,
            optimizer=reloaded_optimizer,
            optimizer_contract=reloaded_contract,
            scheduler=reloaded_scheduler,
            metrics=reloaded_metrics,
            restore_rng=False,
        )
        after, after_loss = _evaluation_signature(
            reloaded_model, small_examples, config, spec
        )
        if not _strict_signature_equal(before, after):
            raise RuntimeError("checkpoint reload changed evaluation outputs")
        if reloaded_state != state or reloaded_best is not None or reloaded_amp is not None:
            raise RuntimeError("checkpoint reload changed non-model training state")

        moving_examples = tuple(
            example
            for example in examples
            if float(
                example.targets.target_displacements[
                    example.targets.target_mask.unsqueeze(-1).expand_as(
                        example.targets.target_displacements
                    )
                ].abs().max().item()
            )
            > 1e-3
        )
        overfit_examples = (moving_examples or tuple(examples))[
            : config.max_queries_per_forward
        ]
        overfit = _tiny_overfit(
            setup=setup,
            examples=overfit_examples,
            spec=spec,
        )
        access = repository.report()
        if not access["zero_validation_and_test_attempts"] or not access[
            "zero_validation_and_test_opens"
        ]:
            raise RuntimeError("preflight accessed validation or test data")

        checkpoint_reload = {
            "checkpoint_schema_version": "parallel_trajectory_training_checkpoint_v1",
            "strict_reload": True,
            "evaluation_outputs_bit_identical": True,
            "before": before_loss,
            "after": after_loss,
            "checkpoint_raw_sha256_before_removal": file_sha256(checkpoint_path),
        }
        shutil.rmtree(smoke_directory)
        test_temporary = destination / "pytest-temporary"
        if test_temporary.exists():
            shutil.rmtree(test_temporary)
        if smoke_directory.exists() or test_temporary.exists():
            raise RuntimeError("disposable preflight artifacts were not removed")

        report = {
            "schema_version": "parallel-trajectory-training-preflight:1.0",
            "status": "passed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "configuration_path": str(Path(config_path).resolve()),
            "configuration_raw_sha256": file_sha256(config_path),
            "test_gates": gate_results,
            "protected_package_digests": protected,
            "environment": environment_metadata(spec),
            "data_access": access,
            "real_training_step": {
                **batch_result.to_dict(),
                "freeze_phase": freeze_phase,
                "optimizer_group_gradients": optimizer_group_gradients,
                "cuda_bf16": True,
                "gradient_clipping_confirmed": True,
                "scheduler_step_confirmed": True,
            },
            "checkpoint_reload": checkpoint_reload,
            "tiny_overfit": overfit,
            "disposable_smoke_artifacts_removed": True,
            "full_training_authorized": True,
            "test_partition_evaluated": False,
        }
        atomic_write_json(destination / "preflight_report.json", report)
        atomic_write_json(
            destination / "artifact_manifest.json",
            {
                "schema_version": "parallel-trajectory-preflight-artifacts:1.0",
                "artifacts": [
                    {
                        "path": path.name,
                        "sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(destination.iterdir())
                    if path.is_file()
                    and path.name != "artifact_manifest.json"
                ],
            },
        )
        return report
    except Exception as exc:
        if smoke_directory.exists():
            shutil.rmtree(smoke_directory)
        test_temporary = destination / "pytest-temporary"
        if test_temporary.exists():
            shutil.rmtree(test_temporary)
        blocked = {
            "schema_version": "parallel-trajectory-blocked-preflight:1.0",
            "status": "blocked",
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "configuration_path": str(Path(config_path).resolve()),
            "test_gates": gate_results,
            "exception_type": type(exc).__name__,
            "message": str(exc)[:8000],
            "data_access": None if repository is None else repository.report(),
            "full_training_authorized": False,
        }
        atomic_write_json(destination / "blocked_preflight.json", blocked)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed training-only preflight for parallel trajectory v1."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Development-only: omit pytest gates (never use to authorize a full run).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_preflight(
        arguments.config,
        output_directory=arguments.output_directory,
        run_tests=not arguments.skip_tests,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_preflight"]
