from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import torch

from fncs_encoder_training.artifacts import environment_report
from fncs_encoder_training.common import (
    append_jsonl,
    atomic_torch_save,
    restore_rng_state,
    rng_state,
    tensor_state_sha256,
)
from fortnite_parallel_trajectory_fncs_recovery.recovery import _pid_alive
from fortnite_parallel_trajectory_fncs_training.binding import (
    assert_frozen_encoder,
    build_frozen_model,
    downstream_state,
    load_downstream_state,
)
from fortnite_parallel_trajectory_fncs_training.data import (
    FNCSParallelTrajectorySessionRepository,
)
from fortnite_parallel_trajectory_fncs_training.optimization import (
    PiecewiseLinearScheduler,
    build_optimizer,
)
from fortnite_parallel_trajectory_training.data import (
    prepare_parallel_trajectory_query_examples,
    session_batches,
    training_session_order,
)
from fortnite_parallel_trajectory_training.training import (
    resolve_precision,
    run_training_batch,
    sample_training_batch_queries,
    seed_everything,
)

from .bindings import load_lineage_setups
from .common import (
    ANALYSIS_DIRECTORY,
    EXPECTED_ENCODER_STATE_SHA256,
    EXPECTED_FIXED_COMPUTE_STEP,
    EXPECTED_INITIAL_DOWNSTREAM_SHA256,
    REPOSITORY_ROOT,
    SCALE_RUN_ROOT,
    SEED,
    PostTrainingError,
    atomic_replace,
    current_utc,
    file_record,
    file_sha256,
    payload_sha256,
    read_json,
    read_jsonl,
    replace_json,
    require,
    verify_self_seal,
    write_new_json,
)


SCALE_CHECKPOINT_SCHEMA = "fncs-decoder-fixed-compute-scale-checkpoint:1.0"
SCALE_STATUS_SCHEMA = "fncs-decoder-fixed-compute-scale-status:1.0"
TRAINABLE_SIZES = (41, 115, 230, 460)
CHECKPOINT_INTERVAL = 20


def scale_run_directory(size: int) -> Path:
    require(size in TRAINABLE_SIZES, f"unsupported trainable subset size: {size}")
    return SCALE_RUN_ROOT / f"n{size:04d}"


def _contract_paths() -> tuple[Path, Path, Path]:
    return (
        ANALYSIS_DIRECTORY / "validation_comparison_contract.json",
        ANALYSIS_DIRECTORY / "scale_study_contract.json",
        ANALYSIS_DIRECTORY / "nested_subset_manifest.json",
    )


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validation_path, scale_path, subset_path = _contract_paths()
    validation = read_json(validation_path, "validation comparison contract")
    scale = read_json(scale_path, "scale-study contract")
    subsets = read_json(subset_path, "nested-subset manifest")
    verify_self_seal(validation, "contract_payload_sha256")
    verify_self_seal(scale, "contract_payload_sha256")
    verify_self_seal(subsets, "manifest_payload_sha256")
    require(
        scale.get("fixed_compute", {}).get("optimizer_updates")
        == EXPECTED_FIXED_COMPUTE_STEP,
        "scale contract optimizer endpoint changed",
    )
    require(
        scale.get("subset_construction", {}).get("seed") == SEED,
        "scale contract subset seed changed",
    )
    declared_source = scale.get("frozen_sources", {}).get("scale_trainer_sha256")
    require(declared_source == file_sha256(Path(__file__)), "scale trainer changed after lock")
    frozen_files = scale.get("frozen_sources", {}).get("files")
    require(isinstance(frozen_files, list) and frozen_files, "scale frozen-source list is missing")
    for record in frozen_files:
        require(isinstance(record, Mapping), "scale frozen-source record is invalid")
        relative = record.get("path")
        expected = record.get("sha256")
        require(isinstance(relative, str) and isinstance(expected, str), "scale frozen-source binding is invalid")
        source = (REPOSITORY_ROOT / relative).resolve()
        require(source.is_relative_to(REPOSITORY_ROOT.resolve()), "scale frozen-source path escaped the repository")
        require(file_sha256(source) == expected, f"scale frozen source changed: {relative}")
    return validation, scale, subsets


def _subset_ids(manifest: Mapping[str, Any], size: int) -> tuple[str, ...]:
    subsets = manifest.get("subsets")
    require(isinstance(subsets, Mapping), "nested subset manifest has no subsets")
    row = subsets.get(str(size))
    require(isinstance(row, Mapping), f"nested subset manifest has no S_{size}")
    ids = row.get("session_ids")
    require(
        isinstance(ids, list)
        and len(ids) == size
        and all(isinstance(value, str) for value in ids)
        and len(set(ids)) == size,
        f"S_{size} session IDs are invalid",
    )
    require(payload_sha256(ids) == row.get("session_ids_sha256"), f"S_{size} hash changed")
    return tuple(ids)


def _run_binding(
    *, size: int, session_ids: Sequence[str], scale_contract: Path, subset_manifest: Path
) -> dict[str, Any]:
    return {
        "subset_size": size,
        "session_ids_sha256": payload_sha256(list(session_ids)),
        "scale_contract_sha256": file_sha256(scale_contract),
        "nested_subset_manifest_sha256": file_sha256(subset_manifest),
        "scale_trainer_sha256": file_sha256(Path(__file__)),
        "seed": SEED,
        "initialization_seed": SEED,
        "initial_downstream_parameter_sha256": EXPECTED_INITIAL_DOWNSTREAM_SHA256,
        "encoder_state_sha256": EXPECTED_ENCODER_STATE_SHA256,
        "optimizer_updates": EXPECTED_FIXED_COMPUTE_STEP,
        "original_schedule_total_updates": 9200,
    }


def _status(run: Path) -> dict[str, Any] | None:
    path = run / "status.json"
    return read_json(path, f"{run.name} status") if path.is_file() else None


def _prior_sizes_complete(size: int) -> None:
    index = TRAINABLE_SIZES.index(size)
    for prior in TRAINABLE_SIZES[:index]:
        status = _status(scale_run_directory(prior))
        require(
            status is not None
            and status.get("status")
            in {"trained_pending_locked_validation", "completed"}
            and status.get("optimizer_step") == EXPECTED_FIXED_COMPUTE_STEP,
            f"predeclared order requires N={prior} to finish before N={size}",
        )


def _save_checkpoint(
    path: Path,
    *,
    binding: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: PiecewiseLinearScheduler,
    optimizer_step: int,
    setup: Any,
    data_access: Mapping[str, Any],
) -> None:
    data = dict(data_access)
    require(
        data.get("zero_test_attempts") is True and data.get("zero_test_opens") is True,
        "refusing to checkpoint after forbidden test access",
    )
    frozen = assert_frozen_encoder(
        model, setup, stage=f"fixed_compute_checkpoint_step_{optimizer_step}"
    )
    state = downstream_state(model)
    atomic_torch_save(
        path,
        {
            "checkpoint_schema_version": SCALE_CHECKPOINT_SCHEMA,
            "created_utc": current_utc(),
            "type": "FrozenFNCSParallelTrajectoryFixedComputeDiagnostic",
            "binding": dict(binding),
            "optimizer_step": optimizer_step,
            "downstream_state": state,
            "downstream_state_sha256": tensor_state_sha256(state),
            "frozen_encoder_snapshot": frozen,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": rng_state(),
            "data_access": data,
            "validation_evaluation_count": 0,
            "checkpoint_selection_performed": False,
            "test_partition_evaluated": False,
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    binding: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: PiecewiseLinearScheduler,
    setup: Any,
) -> int:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {
        "checkpoint_schema_version",
        "created_utc",
        "type",
        "binding",
        "optimizer_step",
        "downstream_state",
        "downstream_state_sha256",
        "frozen_encoder_snapshot",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "data_access",
        "validation_evaluation_count",
        "checkpoint_selection_performed",
        "test_partition_evaluated",
    }
    require(isinstance(raw, Mapping) and set(raw) == expected_keys, "scale checkpoint schema changed")
    require(raw["checkpoint_schema_version"] == SCALE_CHECKPOINT_SCHEMA, "scale checkpoint version changed")
    require(raw["type"] == "FrozenFNCSParallelTrajectoryFixedComputeDiagnostic", "scale checkpoint type changed")
    require(raw["binding"] == dict(binding), "scale checkpoint binding changed")
    state = raw["downstream_state"]
    require(isinstance(state, Mapping), "scale checkpoint downstream state is missing")
    require(
        tensor_state_sha256(state) == raw["downstream_state_sha256"],
        "scale checkpoint embedded downstream hash is invalid",
    )
    load_downstream_state(model, state)
    optimizer.load_state_dict(raw["optimizer_state"])
    scheduler.load_state_dict(raw["scheduler_state"])
    step = raw["optimizer_step"]
    require(type(step) is int and 0 <= step <= EXPECTED_FIXED_COMPUTE_STEP, "scale checkpoint step is invalid")
    require(scheduler.step_index == step, "scale checkpoint scheduler/step mismatch")
    frozen = raw["frozen_encoder_snapshot"]
    require(
        isinstance(frozen, Mapping)
        and frozen.get("encoder_state_sha256") == EXPECTED_ENCODER_STATE_SHA256
        and frozen.get("matches_canonical") is True,
        "scale checkpoint frozen encoder changed",
    )
    require(
        raw["validation_evaluation_count"] == 0
        and raw["checkpoint_selection_performed"] is False
        and raw["test_partition_evaluated"] is False
        and raw["data_access"].get("zero_test_attempts") is True
        and raw["data_access"].get("zero_test_opens") is True,
        "scale checkpoint violates validation/test policy",
    )
    assert_frozen_encoder(model, setup, stage="fixed_compute_checkpoint_reload")
    restore_rng_state(raw["rng_state"])
    return step


def load_trained_scale_model(
    *, size: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Strictly load a terminal scale model for the locked evaluator."""

    _, scale_contract, subset_manifest = _load_contracts()
    session_ids = _subset_ids(subset_manifest, size)
    run = scale_run_directory(size)
    status = _status(run)
    require(
        status is not None
        and status.get("status") in {"trained_pending_locked_validation", "completed"}
        and status.get("optimizer_step") == EXPECTED_FIXED_COMPUTE_STEP,
        f"N={size} is not ready for locked validation",
    )
    current_setup, _, _, _ = load_lineage_setups()
    seed_everything(SEED)
    model, initial = build_frozen_model(current_setup, device=device)
    require(
        initial["initial_downstream_parameter_sha256"]
        == EXPECTED_INITIAL_DOWNSTREAM_SHA256,
        "scale model initialization hash changed",
    )
    optimizer, _ = build_optimizer(model, current_setup.config)
    scheduler = PiecewiseLinearScheduler(optimizer, current_setup.config)
    scale_path = ANALYSIS_DIRECTORY / "scale_study_contract.json"
    subset_path = ANALYSIS_DIRECTORY / "nested_subset_manifest.json"
    binding = _run_binding(
        size=size,
        session_ids=session_ids,
        scale_contract=scale_path,
        subset_manifest=subset_path,
    )
    step = _load_checkpoint(
        run / "last.pt",
        binding=binding,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        setup=current_setup,
    )
    require(step == EXPECTED_FIXED_COMPUTE_STEP, "scale model is not at step 1840")
    model.eval()
    return model, {
        "checkpoint": file_record(run / "last.pt"),
        "binding": binding,
        "status": status,
    }


def _truncate_metrics_to_step(path: Path, step: int) -> None:
    if not path.is_file():
        return
    rows = read_jsonl(path, "scale metrics")
    durable = [
        row
        for row in rows
        if row.get("type") != "optimizer_step"
        or int(row.get("optimizer_step", -1)) <= step
    ]
    data = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for row in durable
    )
    atomic_replace(path, data)


def _step_location(step_zero_based: int, updates_per_cycle: int) -> tuple[int, int]:
    require(step_zero_based >= 0, "optimizer step cannot be negative")
    return divmod(step_zero_based, updates_per_cycle)[0] + 1, divmod(
        step_zero_based, updates_per_cycle
    )[1]


def exposure_summary(session_ids: Sequence[str], optimizer_updates: int) -> dict[str, Any]:
    updates_per_cycle = math.ceil(len(session_ids) / 2)
    counts: Counter[str] = Counter()
    update_query_counts: Counter[int] = Counter()
    for step_zero in range(optimizer_updates):
        cycle, batch_index = _step_location(step_zero, updates_per_cycle)
        ordered = training_session_order(session_ids, seed=SEED, epoch=cycle)
        batches = session_batches(ordered, 2)
        batch = batches[batch_index]
        counts.update(batch)
        update_query_counts[len(batch) * 16] += 1
    total_occurrences = sum(counts.values())
    completed_cycles, partial_updates = divmod(optimizer_updates, updates_per_cycle)
    return {
        "updates_per_subset_cycle": updates_per_cycle,
        "completed_full_cycles": completed_cycles,
        "partial_cycle_updates": partial_updates,
        "effective_passes": total_occurrences / len(session_ids),
        "total_session_exposures": total_occurrences,
        "total_sampled_query_exposures": total_occurrences * 16,
        "per_session_exposure_minimum": min(counts.values()),
        "per_session_exposure_maximum": max(counts.values()),
        "per_session_exposure_mean": total_occurrences / len(session_ids),
        "optimizer_updates_by_query_count": {
            str(key): value for key, value in sorted(update_query_counts.items())
        },
        "per_session_exposures": dict(sorted(counts.items())),
    }


def train_scale(size: int) -> dict[str, Any]:
    require(size in TRAINABLE_SIZES, f"N={size} is not a trainable scale point")
    _prior_sizes_complete(size)
    _, _, subset_manifest = _load_contracts()
    session_ids = _subset_ids(subset_manifest, size)
    current_setup, _, _, _ = load_lineage_setups()
    train_partition = set(current_setup.split_session_ids["train"])
    validation_partition = set(current_setup.split_session_ids["validation"])
    test_partition = set(current_setup.split_session_ids["test"])
    require(set(session_ids) <= train_partition, "scale subset contains a non-training session")
    require(not set(session_ids) & validation_partition, "scale subset overlaps validation")
    require(not set(session_ids) & test_partition, "scale subset overlaps forbidden test")
    require(
        current_setup.config.batch_size == 2
        and current_setup.config.queries_per_session == 16,
        "fixed-compute exposure accounting requires the locked 2x16 batch contract",
    )

    run = scale_run_directory(size)
    scale_contract_path = ANALYSIS_DIRECTORY / "scale_study_contract.json"
    subset_manifest_path = ANALYSIS_DIRECTORY / "nested_subset_manifest.json"
    binding = _run_binding(
        size=size,
        session_ids=session_ids,
        scale_contract=scale_contract_path,
        subset_manifest=subset_manifest_path,
    )
    existing = _status(run)
    if existing is not None and existing.get("status") in {
        "trained_pending_locked_validation",
        "completed",
    }:
        require(existing.get("optimizer_step") == EXPECTED_FIXED_COMPUTE_STEP, "terminal scale status has wrong step")
        return existing
    if existing is not None:
        prior_pid = existing.get("process_id")
        if type(prior_pid) is int and prior_pid != os.getpid() and _pid_alive(prior_pid):
            raise PostTrainingError(f"N={size} trainer PID {prior_pid} is already active")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_new_json(
            run / "contract_binding.json",
            {
                "schema_version": "fncs-decoder-fixed-compute-scale-binding:1.0",
                "created_utc": current_utc(),
                "binding": binding,
                "session_ids": list(session_ids),
                "checkpoint_selection_performed": False,
                "validation_access_before_step1840": False,
                "test_access_allowed": False,
            },
        )
        (run / "metrics.jsonl").touch(exist_ok=False)

    seed_everything(SEED)
    spec = resolve_precision()
    model, initial = build_frozen_model(current_setup, device=spec.device)
    require(
        initial["initial_downstream_parameter_sha256"]
        == EXPECTED_INITIAL_DOWNSTREAM_SHA256,
        "fresh diagnostic initialization does not match the original hash",
    )
    optimizer, optimizer_contract = build_optimizer(model, current_setup.config)
    scheduler = PiecewiseLinearScheduler(optimizer, current_setup.config)
    subset_partitions = dict(current_setup.partition_by_session)
    for session_id in train_partition - set(session_ids):
        subset_partitions[session_id] = "excluded_train"
    repository = FNCSParallelTrajectorySessionRepository(
        paths=current_setup.session_paths,
        partitions=subset_partitions,
        active_partitions=("train",),
        test_session_ids=current_setup.split_session_ids["test"],
        profile=current_setup.profile,
        ledger_path=run / "data_access.json",
        split_manifest_sha256=current_setup.config.expected_split_manifest_sha256,
        phase=f"fixed_compute_n{size}",
        resume_ledger=(run / "data_access.json").is_file(),
    )
    checkpoint_path = run / "last.pt"
    if checkpoint_path.is_file():
        step = _load_checkpoint(
            checkpoint_path,
            binding=binding,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            setup=current_setup,
        )
        _truncate_metrics_to_step(run / "metrics.jsonl", step)
    else:
        step = 0
        write_new_json(run / "initial_downstream_parameters.json", initial)
        write_new_json(
            run / "environment.json",
            {
                **environment_report(Path(__file__).resolve().parents[3]),
                "created_utc": current_utc(),
                "process_id": os.getpid(),
                "precision": "bf16",
                "autocast_dtype": "torch.bfloat16",
                "scale_trainer_sha256": file_sha256(Path(__file__)),
            },
        )
        _save_checkpoint(
            checkpoint_path,
            binding=binding,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_step=0,
            setup=current_setup,
            data_access=repository.report(),
        )
    status = {
        "schema_version": SCALE_STATUS_SCHEMA,
        "updated_utc": current_utc(),
        "status": "running",
        "subset_size": size,
        "optimizer_step": step,
        "endpoint_optimizer_step": EXPECTED_FIXED_COMPUTE_STEP,
        "process_id": os.getpid(),
        "checkpoint_selection_performed": False,
        "validation_evaluation_count": 0,
        "test_partition_evaluated": False,
        "zero_test_attempts": True,
        "zero_test_opens": True,
    }
    replace_json(run / "status.json", status)
    updates_per_cycle = math.ceil(size / current_setup.config.batch_size)
    while step < EXPECTED_FIXED_COMPUTE_STEP:
        cycle, batch_index = _step_location(step, updates_per_cycle)
        ordered = training_session_order(session_ids, seed=SEED, epoch=cycle)
        batches = session_batches(ordered, current_setup.config.batch_size)
        require(len(batches) == updates_per_cycle, "scale loader length changed")
        batch_session_ids = batches[batch_index]
        sources = {
            session_id: repository.get(session_id) for session_id in batch_session_ids
        }
        queries = sample_training_batch_queries(
            sources,
            batch_session_ids,
            seed=SEED,
            epoch=cycle,
            session_batch_index=batch_index,
            queries_per_session=current_setup.config.queries_per_session,
        )
        expected_queries = len(batch_session_ids) * current_setup.config.queries_per_session
        require(len(queries) == expected_queries, "scale training query sample is short")
        examples = prepare_parallel_trajectory_query_examples(
            sources,
            queries,
            context_length_ticks=current_setup.config.context_length_ticks,
        )
        require(len(examples) == expected_queries, "scale training query lost supervision")
        model.train()
        assert_frozen_encoder(model, current_setup, stage=f"fixed_compute_n{size}_step_{step}_pre")
        result = run_training_batch(
            model=model,
            examples=examples,
            optimizer=optimizer,
            scheduler=scheduler,
            config=current_setup.config,  # type: ignore[arg-type]
            spec=spec,
            metrics=None,
        )
        step += 1
        assert_frozen_encoder(model, current_setup, stage=f"fixed_compute_n{size}_step_{step}_post")
        repository.assert_no_test_access()
        append_jsonl(
            run / "metrics.jsonl",
            {
                "type": "optimizer_step",
                "subset_size": size,
                "subset_cycle": cycle,
                "cycle_batch_index": batch_index,
                "optimizer_step": step,
                "session_ids": list(batch_session_ids),
                "sampled_query_count": expected_queries,
                "frozen_encoder": True,
                **result.to_dict(),
            },
        )
        if step % CHECKPOINT_INTERVAL == 0 or step == EXPECTED_FIXED_COMPUTE_STEP:
            _save_checkpoint(
                checkpoint_path,
                binding=binding,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_step=step,
                setup=current_setup,
                data_access=repository.report(),
            )
        status.update(
            {
                "updated_utc": current_utc(),
                "optimizer_step": step,
                "last_route_nll": result.route_nll,
                "last_subset_cycle": cycle,
                "last_cycle_batch_index": batch_index,
                "zero_test_attempts": repository.report()["zero_test_attempts"],
                "zero_test_opens": repository.report()["zero_test_opens"],
            }
        )
        replace_json(run / "status.json", status)
        print(
            json.dumps(
                {
                    "event": "fixed_compute_optimizer_step",
                    "subset_size": size,
                    "optimizer_step": step,
                    "endpoint": EXPECTED_FIXED_COMPUTE_STEP,
                    "route_nll": result.route_nll,
                    "test_attempts": 0,
                    "test_opens": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model.eval()
    frozen = assert_frozen_encoder(model, current_setup, stage=f"fixed_compute_n{size}_terminal")
    rows = [row for row in read_jsonl(run / "metrics.jsonl") if row.get("type") == "optimizer_step"]
    require(
        [row["optimizer_step"] for row in rows] == list(range(1, EXPECTED_FIXED_COMPUTE_STEP + 1)),
        "scale metrics are not exactly contiguous through step 1840",
    )
    exposures = exposure_summary(session_ids, EXPECTED_FIXED_COMPUTE_STEP)
    final_status = {
        **status,
        "updated_utc": current_utc(),
        "status": "trained_pending_locked_validation",
        "optimizer_step": EXPECTED_FIXED_COMPUTE_STEP,
        "checkpoint": file_record(checkpoint_path),
        "scheduler_step_index": scheduler.step_index,
        "scheduler_contract": scheduler.contract(),
        "optimizer_contract": optimizer_contract.to_dict(),
        "initial_downstream_parameter_sha256": initial[
            "initial_downstream_parameter_sha256"
        ],
        "terminal_downstream_state_sha256": tensor_state_sha256(
            downstream_state(model)
        ),
        "encoder_state_sha256": frozen["encoder_state_sha256"],
        "exposure_summary": exposures,
        "validation_evaluation_count": 0,
        "checkpoint_selection_performed": False,
        "test_partition_evaluated": False,
        "data_access": {
            key: repository.report()[key]
            for key in (
                "session_request_counts",
                "dataset_attempt_counts",
                "dataset_open_counts",
                "test_session_request_attempt_count",
                "test_session_parquet_open_attempt_count",
                "test_session_parquet_open_count",
                "zero_test_attempts",
                "zero_test_opens",
            )
        },
    }
    replace_json(run / "status.json", final_status)
    return final_status


def train_all() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for size in TRAINABLE_SIZES:
        try:
            results[str(size)] = train_scale(size)
        except Exception as exc:
            _retain_failure(size, exc)
            raise
    return {"status": "trained_pending_locked_validation", "runs": results}


def _retain_failure(size: int, exc: Exception) -> None:
    run = scale_run_directory(size)
    if not run.is_dir():
        return
    previous = _status(run) or {}
    replace_json(
        run / "status.json",
        {
            **previous,
            "schema_version": SCALE_STATUS_SCHEMA,
            "updated_utc": current_utc(),
            "status": "failed_retained",
            "subset_size": size,
            "process_id": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def status_report() -> dict[str, Any]:
    return {
        str(size): (_status(scale_run_directory(size)) or {"status": "not_started"})
        for size in TRAINABLE_SIZES
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-compute FNCS scale diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    one = subparsers.add_parser("train", help="train one predeclared scale point")
    one.add_argument("size", type=int, choices=TRAINABLE_SIZES)
    subparsers.add_parser("train-all", help="train all scale points in predeclared order")
    subparsers.add_parser("status", help="report scale-run status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            result = train_scale(args.size)
        elif args.command == "train-all":
            result = train_all()
        else:
            result = status_report()
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        if args.command == "train":
            _retain_failure(args.size, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_INTERVAL",
    "SCALE_CHECKPOINT_SCHEMA",
    "SCALE_STATUS_SCHEMA",
    "TRAINABLE_SIZES",
    "exposure_summary",
    "load_trained_scale_model",
    "scale_run_directory",
    "status_report",
    "train_all",
    "train_scale",
]
