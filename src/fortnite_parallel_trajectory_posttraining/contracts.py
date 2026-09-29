from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bindings import load_lineage_setups
from .common import (
    ANALYSIS_DIRECTORY,
    CHILD_RUN,
    CONFIG_PATH,
    EXPECTED_ENCODER_STATE_SHA256,
    EXPECTED_FINAL_ROUTE_NLL,
    EXPECTED_FINAL_STEP,
    EXPECTED_FIXED_COMPUTE_STEP,
    EXPECTED_INITIAL_DOWNSTREAM_SHA256,
    FINALIZATION_DIRECTORY,
    HORIZONS_SECONDS,
    LINEAGE_NAME,
    PARENT_RUN,
    RECOVERY_AUDIT,
    REPOSITORY_ROOT,
    SCALE_RUN_ROOT,
    SCALE_SESSION_COUNTS,
    SEED,
    STEP1840_CHECKPOINT,
    current_utc,
    file_record,
    file_sha256,
    make_file_read_only,
    payload_sha256,
    read_json,
    require,
    self_seal,
    verify_self_seal,
    write_new_json,
)


VALIDATION_CONTRACT_PATH = ANALYSIS_DIRECTORY / "validation_comparison_contract.json"
SCALE_CONTRACT_PATH = ANALYSIS_DIRECTORY / "scale_study_contract.json"
FUTURE_TEST_RESERVATION_PATH = ANALYSIS_DIRECTORY / "future_test_reservation.json"


VALIDATION_SOURCE_PATHS = (
    "ml/src/fortnite_parallel_trajectory_posttraining/common.py",
    "ml/src/fortnite_parallel_trajectory_posttraining/bindings.py",
    "ml/src/fortnite_parallel_trajectory_posttraining/metrics.py",
    "ml/src/fortnite_parallel_trajectory_posttraining/evaluate.py",
    "ml/src/fortnite_parallel_trajectory_fncs_recovery/recovery.py",
    "ml/src/fortnite_parallel_trajectory_fncs_training/binding.py",
    "ml/src/fortnite_parallel_trajectory_fncs_training/data.py",
    "ml/src/fortnite_parallel_trajectory_training/data.py",
    "ml/src/fortnite_parallel_trajectory_training/training.py",
    "ml/src/fortnite_encoder/planner_policy.py",
    "ml/src/fortnite_encoder/planner_supervision.py",
    "ml/src/fortnite_encoder/tensorize.py",
    "ml/src/fortnite_parallel_trajectory/inference.py",
    "ml/src/fortnite_parallel_trajectory/losses.py",
    "ml/src/fortnite_parallel_trajectory/model.py",
    "ml/src/fortnite_parallel_trajectory/targets.py",
)

SCALE_SOURCE_PATHS = (
    "ml/src/fortnite_parallel_trajectory_posttraining/common.py",
    "ml/src/fortnite_parallel_trajectory_posttraining/bindings.py",
    "ml/src/fortnite_parallel_trajectory_posttraining/scale.py",
    "ml/src/fortnite_parallel_trajectory_fncs_recovery/recovery.py",
    "ml/src/fortnite_parallel_trajectory_fncs_training/binding.py",
    "ml/src/fortnite_parallel_trajectory_fncs_training/data.py",
    "ml/src/fortnite_parallel_trajectory_fncs_training/optimization.py",
    "ml/src/fortnite_parallel_trajectory_training/data.py",
    "ml/src/fortnite_parallel_trajectory_training/training.py",
    "ml/src/fortnite_encoder/planner_supervision.py",
    "ml/src/fortnite_encoder/tensorize.py",
    "ml/src/fortnite_parallel_trajectory/losses.py",
    "ml/src/fortnite_parallel_trajectory/model.py",
    "ml/src/fortnite_parallel_trajectory/targets.py",
)


def _source_records(paths: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in paths:
        path = (REPOSITORY_ROOT / relative).resolve()
        require(path.is_relative_to(REPOSITORY_ROOT.resolve()), "source path escaped repository")
        require(path.is_file(), f"locked source is missing: {relative}")
        records.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return records


def _verify_finalization() -> dict[str, Any]:
    manifest_path = FINALIZATION_DIRECTORY / "finalization_manifest.json"
    manifest = read_json(manifest_path, "finalization manifest")
    verify_self_seal(manifest)
    require(
        manifest.get("status") == "complete_and_file_level_read_only",
        "lineage finalization is not complete",
    )
    for row in manifest.get("artifacts", []):
        require(isinstance(row, Mapping), "finalization artifact record is invalid")
        path = FINALIZATION_DIRECTORY / str(row.get("path"))
        require(
            path.is_file() and file_sha256(path) == row.get("sha256"),
            f"finalization artifact changed: {path.name}",
        )
    completed = read_json(FINALIZATION_DIRECTORY / "completed_lineage.json")
    verify_self_seal(completed)
    state = completed.get("completed_state", {})
    require(
        state.get("optimizer_steps") == EXPECTED_FINAL_STEP
        and state.get("epochs") == 20
        and state.get("runtime_status") == "completed"
        and state.get("best_checkpoint_step") == EXPECTED_FINAL_STEP
        and state.get("best_validation_route_nll") == EXPECTED_FINAL_ROUTE_NLL,
        "finalized endpoint differs from the selected checkpoint contract",
    )
    access = read_json(FINALIZATION_DIRECTORY / "data_access_finalization.json")
    verify_self_seal(access)
    require(all(access.get("gates", {}).values()), "finalized lineage has a failed access gate")
    return manifest


def _existing_locked_contracts() -> dict[str, Any] | None:
    paths = (VALIDATION_CONTRACT_PATH, SCALE_CONTRACT_PATH, FUTURE_TEST_RESERVATION_PATH)
    if not any(path.exists() for path in paths):
        return None
    require(all(path.is_file() for path in paths), "protocol lock is only partially present")
    values = {
        "validation": read_json(VALIDATION_CONTRACT_PATH),
        "scale": read_json(SCALE_CONTRACT_PATH),
        "future_test": read_json(FUTURE_TEST_RESERVATION_PATH),
    }
    verify_self_seal(values["validation"], "contract_payload_sha256")
    verify_self_seal(values["scale"], "contract_payload_sha256")
    verify_self_seal(values["future_test"], "reservation_payload_sha256")
    return values


def lock_contracts() -> dict[str, Any]:
    existing = _existing_locked_contracts()
    if existing is not None:
        return {
            "status": "already_locked",
            "contracts": {
                "validation": file_record(VALIDATION_CONTRACT_PATH),
                "scale": file_record(SCALE_CONTRACT_PATH),
                "future_test": file_record(FUTURE_TEST_RESERVATION_PATH),
            },
        }

    require(
        not (ANALYSIS_DIRECTORY / "evaluation_status.json").exists()
        and not (ANALYSIS_DIRECTORY / "raw_validation").exists(),
        "validation activity preceded protocol publication",
    )
    require(not SCALE_RUN_ROOT.exists(), "scale training preceded protocol publication")
    finalization = _verify_finalization()
    current_setup, checkpoint_setup, _, _ = load_lineage_setups()
    config = current_setup.config
    require(
        config.seed == SEED
        and config.initialization_seed == SEED
        and config.batch_size == 2
        and config.queries_per_session == 16
        and config.validation_queries_per_session == 32
        and config.max_queries_per_forward == 4
        and config.max_optimizer_steps == 9200,
        "resolved training configuration changed",
    )
    split_counts = {key: len(value) for key, value in current_setup.split_session_ids.items()}
    require(split_counts == {"train": 920, "validation": 115, "test": 115}, "split changed")
    validation_ids = sorted(current_setup.split_session_ids["validation"])
    test_ids = sorted(current_setup.split_session_ids["test"])
    lock_time = current_utc()
    finalization_binding = {
        "manifest": file_record(FINALIZATION_DIRECTORY / "finalization_manifest.json"),
        "manifest_payload_sha256": finalization["payload_sha256"],
        "parent_directory": str(PARENT_RUN),
        "recovery_child_directory": str(CHILD_RUN),
        "recovery_audit_directory": str(RECOVERY_AUDIT),
        "source_lineage_is_read_only": True,
        "further_training_of_final_lineage_authorized": False,
    }
    validation_sources = _source_records(VALIDATION_SOURCE_PATHS)
    validation = self_seal(
        {
            "schema_version": "fncs-decoder-validation-comparison-contract:1.0",
            "created_utc": lock_time,
            "publication_precedes_all_post_training_validation_inference": True,
            "development_set_only": True,
            "finalization_binding": finalization_binding,
            "selected_checkpoint": {
                **file_record(CHILD_RUN / "best.pt"),
                "logical_optimizer_step": EXPECTED_FINAL_STEP,
                "completed_epochs": 20,
                "selection_metric": "validation_route_mixture_nll",
                "selection_value": EXPECTED_FINAL_ROUTE_NLL,
                "selection_is_already_final": True,
                "post_training_results_cannot_change_selection": True,
            },
            "validation_population": {
                "partition": "encoder_validation",
                "session_count": 115,
                "session_ids": validation_ids,
                "session_ids_sha256": payload_sha256(validation_ids),
                "queries_per_session": 32,
                "total_queries": 3680,
                "sampler": "parallel-trajectory-validation-query-sample-v1",
                "seed": SEED,
                "algorithm": (
                    "For each lexicographically sorted validation session, enumerate canonical "
                    "eligible phase-1..8 routes, then take up to 32 without replacement using "
                    "torch.randperm seeded by stable_seed(seed, sampler, session_id)."
                ),
                "same_queries_for_every_method_and_checkpoint": True,
                "checkpoint_selection_reuse_disclosure": (
                    "This validation partition selected best.pt during training and is reused only "
                    "for development diagnostics; estimates are not held-out test performance."
                ),
            },
            "test_exclusion": {
                "original_encoder_test_session_count": 115,
                "original_encoder_test_session_ids_sha256": payload_sha256(test_ids),
                "repository_active_partitions": ["validation"],
                "denylist_applied_before_reader": True,
                "access_requests_allowed": False,
                "dataset_open_attempts_allowed": False,
                "evaluation_allowed": False,
            },
            "targets_and_masks": {
                "target_contract": "parallel-trajectory-targets:1.0",
                "horizons_seconds": list(HORIZONS_SECONDS),
                "current_and_future_positions": "living-team centroids",
                "future_displacements": "current-relative and normalized by 8192 world units per axis",
                "masking": (
                    "Exact public target builder: missing timestamps/coordinates/invalid phase or "
                    "out-of-envelope coordinates mask the affected horizon; lifecycle ambiguity, "
                    "team elimination, or phase >=9 mask that horizon and the remaining suffix."
                ),
                "out_of_envelope_policy": "mask_without_clamping",
                "model_and_baselines_share_identical_target_mask": True,
            },
            "methods": {
                "decoder": (
                    "Choose one coherent route: the five-mode route with greatest mixture "
                    "probability (first mode on an exact tie), then use that mode's mean XY at all horizons."
                ),
                "static_position": (
                    "Repeat the causal query-time living-team centroid at every horizon."
                ),
                "constant_velocity": (
                    "Use the most recent earlier valid living-team centroid in the causal 64-tick "
                    "window with an unchanged roster alive-mask and life indices, stopping at the first "
                    "lifecycle discontinuity; extrapolate uncapped XY velocity to each horizon. Fall back "
                    "to static if no such positive-delta observation exists."
                ),
                "causal_policy": "Baselines operate only on sanitized causal model inputs.",
                "deterministic_baseline_nll": (
                    "Unavailable: no predeclared probabilistic scale/covariance contract exists, so no NLL is fabricated."
                ),
            },
            "metrics": {
                "primary_probabilistic": (
                    "Exact route-mixture NLL: FP32 bivariate Gaussian log densities summed over valid "
                    "horizons inside each route mode, one logsumexp over five coherent modes, mean over valid routes."
                ),
                "ade_meters": "Mean within each valid route, then equal-weight mean across valid routes.",
                "fde_60_meters": "Euclidean error at 60 s over routes whose 60 s target is valid.",
                "last_valid_horizon_fde_meters": "Euclidean error at each route's last valid horizon.",
                "per_horizon_error_meters": "Pooled Euclidean error over valid targets at each horizon.",
                "trajectory_hit_rate": {
                    "distance": "Euclidean XY distance after dividing each axis by its 8192-world-unit cell extent.",
                    "inclusive_radii_cells": [1, 2, 4],
                    "per_horizon": True,
                    "aggregate_deadlines_seconds": [15, 30, 45, 60],
                    "aggregate_weighting": "Pooled valid query-horizon pairs through the deadline.",
                    "terminology": "Trajectory hit rate, never classification accuracy.",
                },
            },
            "uncertainty": {
                "unit": "validation session",
                "paired_for_method_and_checkpoint_differences": True,
                "resamples": 10000,
                "rng": "NumPy PCG64",
                "seed": 20260902,
                "confidence": 0.95,
                "interval": "percentile with NumPy linear quantiles",
                "aggregation": "Resample sessions with replacement, then recompute pooled numerator/denominator.",
                "empty_denominator_replicates": "Omit and report; fail if all replicates are empty.",
            },
            "decoder_diagnostics": {
                "oracle_only": ["minADE@5", "minFDE@5"],
                "mode_use": ["probability entropy", "effective modes", "mean probabilities", "top-mode counts"],
                "collapse_flags": {
                    "probability": "maximum route probability >= 0.98",
                    "geometric": "mean pairwise valid-horizon separation < 0.25 cell",
                    "unused": "mode is never highest-probability on any validation query",
                },
                "nonfinite_predictions_allowed": False,
            },
            "frozen_sources": {
                "evaluator_sha256": file_sha256(
                    REPOSITORY_ROOT / "ml/src/fortnite_parallel_trajectory_posttraining/evaluate.py"
                ),
                "files": validation_sources,
            },
            "immutable_bindings": {
                "config": file_record(CONFIG_PATH),
                "split_manifest_sha256": config.expected_split_manifest_sha256,
                "dataset_inventory_sha256": config.expected_dataset_inventory_sha256,
                "world_grid_profile_hash": current_setup.profile.profile_hash,
                "encoder_state_sha256": EXPECTED_ENCODER_STATE_SHA256,
                "initial_downstream_parameter_sha256": EXPECTED_INITIAL_DOWNSTREAM_SHA256,
            },
        },
        "contract_payload_sha256",
    )
    scale_sources = _source_records(SCALE_SOURCE_PATHS)
    scale = self_seal(
        {
            "schema_version": "fncs-decoder-fixed-compute-scale-contract:1.0",
            "created_utc": lock_time,
            "publication_precedes_scale_training_and_validation": True,
            "purpose": (
                "Controlled diagnostic of training-session count at fixed optimizer updates; "
                "not checkpoint selection, not a universal scaling law, and not a final test."
            ),
            "finalization_binding": finalization_binding,
            "ordered_training_session_counts": list(SCALE_SESSION_COUNTS),
            "execution_order": "41, 115, 230, 460, then the authentic existing 920-session endpoint",
            "subset_construction": {
                "universe": "exact 920-session encoder training partition",
                "seed": SEED,
                "score": "SHA256(UTF8(decimal seed) || NUL || UTF8(canonical session ID))",
                "ordering": "ascending score bytes, then ascending canonical session ID",
                "subsets": "S_N is the first N ordered sessions; therefore all subsets are nested prefixes",
                "no_outcome_or_validation_information": True,
            },
            "fixed_compute": {
                "optimizer_updates": EXPECTED_FIXED_COMPUTE_STEP,
                "batch_size_sessions": 2,
                "queries_per_session_per_exposure": 16,
                "gradient_accumulation_steps": 1,
                "precision": "CUDA BF16 autocast with FP32 route-mixture NLL",
                "optimizer": "same four-group AdamW contract as the finalized lineage",
                "learning_rate_schedule": "same original absolute 9200-update piecewise-linear schedule",
                "original_schedule_total_updates": 9200,
                "schedule_boundaries": {
                    "warmup_updates_inclusive": [1, 460],
                    "constant_updates_inclusive": [461, 920],
                    "linear_decay_updates_inclusive": [921, 9200],
                    "final_learning_rate_ratio": 0.1,
                },
                "endpoint_schedule_position": 1840,
                "same_initialization_seed": SEED,
                "same_initial_downstream_parameter_sha256": EXPECTED_INITIAL_DOWNSTREAM_SHA256,
                "same_frozen_encoder_state_sha256": EXPECTED_ENCODER_STATE_SHA256,
                "same_architecture_targets_masks_objective_and_sampler": True,
                "validation_access_before_endpoint_allowed": False,
                "checkpoint_selection_performed": False,
            },
            "endpoint_sources": {
                "41_115_230_460": "fresh runs initialized identically and stopped exactly at update 1840",
                "920": {
                    "source": "authentic durable checkpoint copied during recovery before any continuation",
                    **file_record(STEP1840_CHECKPOINT),
                    "logical_optimizer_step": EXPECTED_FIXED_COMPUTE_STEP,
                    "continued_or_reconstructed_for_study": False,
                },
            },
            "validation": {
                "contract": str(VALIDATION_CONTRACT_PATH),
                "contract_payload_sha256": validation["contract_payload_sha256"],
                "one_locked_evaluation_after_every_endpoint_is_durable": True,
                "same_115_sessions_and_3680_queries_for_all_scale_points": True,
                "checkpoint_selection_performed": False,
            },
            "reporting": {
                "per_subset": [
                    "unique session count",
                    "candidate window count",
                    "eligible route count",
                    "valid target count per horizon",
                    "replay/table hashes and build/data/provenance status",
                    "effective passes and per-session exposure counts",
                ],
                "metrics": ["route NLL", "ADE", "FDE", "per-horizon errors", "trajectory hit rates"],
                "successive_change": "larger-minus-smaller, absolute and percent, with paired session bootstrap",
                "limitations": [
                    "one initialization seed",
                    "one deterministic nested ordering",
                    "validation-session intervals omit initialization and subset-selection uncertainty",
                    "smaller sets receive more effective passes under equal update count",
                ],
            },
            "failure_and_resume": {
                "durable_checkpoint_interval_updates": 20,
                "retain_failure_state": True,
                "resume_from_exact_optimizer_scheduler_rng_and_sampler_position": True,
                "predeclared_order_enforced": True,
            },
            "access_policy": {
                "scale_training_repository_partition": "selected training subset only",
                "other_training_sessions": "inactive and rejected before reader",
                "validation": "inactive throughout training; enabled only by locked evaluator after step 1840",
                "original_test": "denylisted before reader; no request, open attempt, or evaluation permitted",
                "final_9200_lineage_extension_authorized": False,
            },
            "frozen_sources": {
                "scale_trainer_sha256": file_sha256(
                    REPOSITORY_ROOT / "ml/src/fortnite_parallel_trajectory_posttraining/scale.py"
                ),
                "files": scale_sources,
            },
            "immutable_bindings": {
                "config": file_record(CONFIG_PATH),
                "split_manifest_sha256": config.expected_split_manifest_sha256,
                "dataset_inventory_sha256": config.expected_dataset_inventory_sha256,
                "world_grid_profile_hash": current_setup.profile.profile_hash,
                "current_setup_source_digest": current_setup.training_source_manifest["source_tree_sha256"],
                "checkpoint_setup_source_digest": checkpoint_setup.training_source_manifest["source_tree_sha256"],
            },
        },
        "contract_payload_sha256",
    )
    reservation = self_seal(
        {
            "schema_version": "fncs-decoder-future-final-test-reservation:1.0",
            "created_utc": lock_time,
            "status": "reserved_pending_future_collection",
            "reserved_session_capacity": 230,
            "minimum_release_session_count": 115,
            "preferred_release_session_count": 230,
            "current_reserved_session_ids": [],
            "current_reserved_session_count": 0,
            "why_empty_now": (
                "No independent post-contract canonical FNCS collection exists in the workspace. "
                "Slots are reserved prospectively without inspecting future outcomes."
            ),
            "eligibility": {
                "first_ingested_after_utc": lock_time,
                "not_in_current_1150_session_corpus": True,
                "not_in_any_training_validation_or_original_test_partition": True,
                "same target and causal-input compatibility checks required": True,
            },
            "assignment": (
                "At least the first 115 and preferably the first 230 eligible canonical sessions "
                "under ascending SHA256(UTF8('future-final-test-v1') || NUL || UTF8(session ID)) "
                "are sealed before labels, model outputs, or outcomes are examined."
            ),
            "release_gate": (
                "Use exactly once only after all model, metric, calibration, threshold, and report decisions "
                "are frozen; never tune or select a checkpoint on these sessions."
            ),
            "current_study_may_access_future_reservation": False,
            "original_encoder_test_remains_untouched": True,
        },
        "reservation_payload_sha256",
    )
    ANALYSIS_DIRECTORY.mkdir(parents=True, exist_ok=False)
    write_new_json(VALIDATION_CONTRACT_PATH, validation)
    write_new_json(SCALE_CONTRACT_PATH, scale)
    write_new_json(FUTURE_TEST_RESERVATION_PATH, reservation)
    for path in (VALIDATION_CONTRACT_PATH, SCALE_CONTRACT_PATH, FUTURE_TEST_RESERVATION_PATH):
        make_file_read_only(path)
    return {
        "status": "locked_before_analysis",
        "contracts": {
            "validation": file_record(VALIDATION_CONTRACT_PATH),
            "scale": file_record(SCALE_CONTRACT_PATH),
            "future_test": file_record(FUTURE_TEST_RESERVATION_PATH),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lock post-training analysis contracts.")
    parser.add_argument("command", choices=("lock", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lock":
        result = lock_contracts()
    else:
        existing = _existing_locked_contracts()
        result = {"status": "locked" if existing is not None else "not_locked"}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FUTURE_TEST_RESERVATION_PATH",
    "SCALE_CONTRACT_PATH",
    "VALIDATION_CONTRACT_PATH",
    "lock_contracts",
]
