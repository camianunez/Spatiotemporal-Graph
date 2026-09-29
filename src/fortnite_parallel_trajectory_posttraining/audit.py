from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    ANALYSIS_DIRECTORY,
    CHILD_RUN,
    EXPECTED_ENCODER_STATE_SHA256,
    EXPECTED_FINAL_ROUTE_NLL,
    EXPECTED_FINAL_STEP,
    EXPECTED_FIXED_COMPUTE_STEP,
    FINALIZATION_DIRECTORY,
    PARENT_RUN,
    RECOVERY_AUDIT,
    REPOSITORY_ROOT,
    SCALE_RUN_ROOT,
    STEP1840_CHECKPOINT,
    current_utc,
    directory_inventory,
    file_record,
    file_sha256,
    inventory_sha256,
    is_read_only,
    make_file_read_only,
    make_tree_files_read_only,
    read_json,
    read_jsonl,
    require,
    self_seal,
    verify_self_seal,
    write_new_json,
)
from .evaluate import MODEL_LABELS
from .scale import TRAINABLE_SIZES, scale_run_directory


AUDIT_PATH = ANALYSIS_DIRECTORY / "final_integrity_audit.json"
MANIFEST_PATH = ANALYSIS_DIRECTORY / "artifact_manifest.json"


def _verify_access_file(path: Path) -> dict[str, Any]:
    value = read_json(path, f"access artifact {path}")
    zero_fields = (
        "test_session_request_attempt_count",
        "test_session_parquet_open_attempt_count",
        "test_session_parquet_open_count",
        "test_access_attempts",
        "test_parquet_opens",
        "test_evaluator_invocations",
    )
    for field in zero_fields:
        if field in value:
            require(value[field] == 0, f"{path.name} records nonzero {field}")
    for field in ("session_request_counts", "session_open_counts", "dataset_attempt_counts", "dataset_open_counts"):
        counts = value.get(field)
        if isinstance(counts, Mapping):
            require(int(counts.get("test", 0)) == 0, f"{path.name} records test activity in {field}")
    if "zero_test_attempts" in value:
        require(value["zero_test_attempts"] is True, f"{path.name} fails zero-test-attempt gate")
    if "zero_test_opens" in value:
        require(value["zero_test_opens"] is True, f"{path.name} fails zero-test-open gate")
    if "test_partition_evaluated" in value:
        require(value["test_partition_evaluated"] is False, f"{path.name} records test evaluation")
    events = value.get("events")
    if isinstance(events, list):
        require(
            all(event.get("partition") != "test" for event in events if isinstance(event, Mapping)),
            f"{path.name} contains a test event",
        )
    return {
        "artifact": file_record(path),
        "zero_test_requests_attempts_opens_and_evaluations": True,
    }


def _verify_lineage_unchanged() -> dict[str, Any]:
    finalization = read_json(FINALIZATION_DIRECTORY / "finalization_manifest.json")
    verify_self_seal(finalization)
    results: dict[str, Any] = {}
    for label, path in (
        ("parent", PARENT_RUN),
        ("child", CHILD_RUN),
        ("recovery", RECOVERY_AUDIT),
    ):
        inventory = directory_inventory(path)
        observed = inventory_sha256(inventory)
        expected = finalization["source_inventory_hashes"][label]
        require(observed == expected, f"finalized {label} tree changed")
        require(all(row["read_only"] is True for row in inventory), f"{label} tree is not fully read-only")
        results[label] = {
            "file_count": len(inventory),
            "inventory_sha256": observed,
            "expected_inventory_sha256": expected,
            "all_files_read_only": True,
        }
    return results


def _verify_scale_runs() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for size in TRAINABLE_SIZES:
        run = scale_run_directory(size)
        status = read_json(run / "status.json", f"N={size} status")
        require(
            status.get("status") == "completed"
            and status.get("optimizer_step") == EXPECTED_FIXED_COMPUTE_STEP
            and status.get("scheduler_step_index") == EXPECTED_FIXED_COMPUTE_STEP
            and status.get("validation_evaluation_count") == 1
            and status.get("checkpoint_selection_performed") is False
            and status.get("test_partition_evaluated") is False
            and status.get("encoder_state_sha256") == EXPECTED_ENCODER_STATE_SHA256,
            f"N={size} terminal status is invalid",
        )
        checkpoint = run / "last.pt"
        require(
            file_sha256(checkpoint) == status.get("checkpoint", {}).get("sha256"),
            f"N={size} checkpoint changed after training",
        )
        rows = [
            row
            for row in read_jsonl(run / "metrics.jsonl", f"N={size} metrics")
            if row.get("type") == "optimizer_step"
        ]
        require(
            [row.get("optimizer_step") for row in rows]
            == list(range(1, EXPECTED_FIXED_COMPUTE_STEP + 1)),
            f"N={size} optimizer history is not exactly 1..1840",
        )
        require(all(row.get("frozen_encoder") is True for row in rows), f"N={size} has an unfrozen step")
        require(
            all(
                isinstance(row.get("route_nll"), (int, float))
                and math.isfinite(float(row["route_nll"]))
                for row in rows
            ),
            f"N={size} metrics are nonfinite",
        )
        access = _verify_access_file(run / "data_access.json")
        results[str(size)] = {
            "status": file_record(run / "status.json"),
            "checkpoint": file_record(checkpoint),
            "metrics": file_record(run / "metrics.jsonl"),
            "optimizer_step_count": len(rows),
            "terminal_downstream_state_sha256": status["terminal_downstream_state_sha256"],
            "access": access,
        }
    return results


def _verify_analysis_outputs() -> dict[str, Any]:
    for path, field in (
        (ANALYSIS_DIRECTORY / "validation_comparison_contract.json", "contract_payload_sha256"),
        (ANALYSIS_DIRECTORY / "scale_study_contract.json", "contract_payload_sha256"),
        (ANALYSIS_DIRECTORY / "nested_subset_manifest.json", "manifest_payload_sha256"),
        (ANALYSIS_DIRECTORY / "future_test_reservation.json", "reservation_payload_sha256"),
        (ANALYSIS_DIRECTORY / "historical_comparability.json", "payload_sha256"),
        (ANALYSIS_DIRECTORY / "decision_summary.json", "payload_sha256"),
    ):
        value = read_json(path)
        verify_self_seal(value, field)
    require(
        all(
            is_read_only(ANALYSIS_DIRECTORY / name)
            for name in (
                "validation_comparison_contract.json",
                "scale_study_contract.json",
                "nested_subset_manifest.json",
                "future_test_reservation.json",
                "historical_comparability.json",
            )
        ),
        "one or more predeclared/sealed contracts are writable",
    )
    evaluation = read_json(ANALYSIS_DIRECTORY / "evaluation_status.json")
    require(
        evaluation.get("test_partition_evaluated") is False
        and evaluation.get("test_access_attempts") == 0
        and evaluation.get("test_parquet_opens") == 0
        and evaluation.get("baseline_derivation", {}).get("completed") is True
        and set(evaluation.get("models", {})) == set(MODEL_LABELS)
        and all(evaluation["models"][label].get("completed") is True for label in MODEL_LABELS),
        "locked evaluation status is incomplete or crossed the test boundary",
    )
    for label in MODEL_LABELS:
        row = evaluation["models"][label]
        require(
            isinstance(row.get("route_nll"), (int, float))
            and math.isfinite(float(row["route_nll"])),
            f"{label} route NLL is nonfinite",
        )
        path = ANALYSIS_DIRECTORY / "raw_validation" / f"{label}.npz"
        require(file_sha256(path) == row.get("arrays_sha256"), f"{label} arrays changed")
    baseline = read_json(ANALYSIS_DIRECTORY / "baseline_results.json")
    observed = baseline.get("primary_probabilistic_metric", {}).get("decoder_route_mixture_nll")
    require(
        isinstance(observed, (int, float))
        and math.isclose(float(observed), EXPECTED_FINAL_ROUTE_NLL, rel_tol=0.0, abs_tol=2e-6),
        "final locked NLL did not reproduce",
    )
    subsets = read_json(ANALYSIS_DIRECTORY / "nested_subset_manifest.json")
    require(
        subsets.get("training_universe_count") == 920
        and subsets.get("validation_overlap_count") == 0
        and subsets.get("test_overlap_count") == 0
        and subsets.get("nested_prefixes_verified") is True,
        "nested subset manifest gates failed",
    )
    access_paths = sorted(ANALYSIS_DIRECTORY.glob("data_access*.json"))
    access_paths.append(ANALYSIS_DIRECTORY / "training_subset_census_access_final.json")
    access_results = {
        path.name: _verify_access_file(path)
        for path in dict.fromkeys(access_paths)
        if path.is_file()
    }
    required = [
        "baseline_results.json",
        "fixed_compute_scale_results.json",
        "optimization_progress_results.json",
        "final_report.md",
        "decision_summary.json",
    ]
    require(all((ANALYSIS_DIRECTORY / name).is_file() for name in required), "report outputs are incomplete")
    return {
        "evaluation_status": file_record(ANALYSIS_DIRECTORY / "evaluation_status.json"),
        "model_evaluation_count": len(MODEL_LABELS),
        "final_route_nll_reproduced": float(observed),
        "nested_subset_gates": {
            "training_sessions": 920,
            "validation_overlap": 0,
            "test_overlap": 0,
            "nested": True,
        },
        "access_artifacts": access_results,
        "required_outputs": {name: file_record(ANALYSIS_DIRECTORY / name) for name in required},
    }


def finalize_analysis() -> dict[str, Any]:
    if MANIFEST_PATH.is_file():
        manifest = read_json(MANIFEST_PATH)
        verify_self_seal(manifest)
        return {"status": "already_finalized", "manifest": file_record(MANIFEST_PATH)}

    lineage = _verify_lineage_unchanged()
    scale = _verify_scale_runs()
    analysis = _verify_analysis_outputs()
    step1840 = file_record(STEP1840_CHECKPOINT)
    require(step1840["read_only"] is True, "authentic step-1840 checkpoint is not read-only")
    audit_payload = self_seal(
        {
            "schema_version": "fncs-decoder-post-training-final-integrity-audit:1.0",
            "created_utc": current_utc(),
            "status": "all_gates_passed",
            "lineage_unchanged": lineage,
            "selected_final_checkpoint": {
                **file_record(CHILD_RUN / "best.pt"),
                "logical_optimizer_step": EXPECTED_FINAL_STEP,
                "validation_route_nll": EXPECTED_FINAL_ROUTE_NLL,
                "further_training_authorized": False,
            },
            "authentic_full_data_fixed_compute_checkpoint": step1840,
            "scale_runs": scale,
            "analysis": analysis,
            "future_test": {
                "reservation": file_record(ANALYSIS_DIRECTORY / "future_test_reservation.json"),
                "reserved_capacity": 230,
                "currently_collected_sessions": 0,
                "accessed_during_current_study": False,
            },
            "global_access_conclusion": {
                "original_115_session_encoder_test_requested": False,
                "original_115_session_encoder_test_open_attempted": False,
                "original_115_session_encoder_test_opened": False,
                "original_115_session_encoder_test_evaluated": False,
            },
            "gates": {
                "completed_lineage_remains_byte_identical_and_read_only": True,
                "validation_protocol_predeclared_and_read_only": True,
                "scale_protocol_and_nested_subsets_predeclared": True,
                "all_scale_runs_exactly_1840_updates": True,
                "all_scale_encoders_frozen_and_identical": True,
                "all_six_models_evaluated_on_same_locked_validation_population": True,
                "final_route_nll_reproduced": True,
                "baseline_and_scale_uncertainty_reported": True,
                "historical_comparability_boundary_reported": True,
                "original_test_never_accessed": True,
                "future_final_test_capacity_reserved": True,
                "final_report_and_plot_data_present": True,
            },
            "source_files": [
                file_record(path)
                for path in sorted(
                    (REPOSITORY_ROOT / "ml/src/fortnite_parallel_trajectory_posttraining").glob("*.py")
                )
            ],
        }
    )
    if AUDIT_PATH.is_file():
        existing = read_json(AUDIT_PATH)
        verify_self_seal(existing)
        require(existing == audit_payload, "existing final audit differs")
    else:
        write_new_json(AUDIT_PATH, audit_payload)

    scale_sealed_count = make_tree_files_read_only(SCALE_RUN_ROOT)
    analysis_sealed_count = make_tree_files_read_only(ANALYSIS_DIRECTORY)
    analysis_inventory = directory_inventory(ANALYSIS_DIRECTORY)
    scale_inventory = directory_inventory(SCALE_RUN_ROOT)
    manifest = self_seal(
        {
            "schema_version": "fncs-decoder-post-training-artifact-manifest:1.0",
            "created_utc": current_utc(),
            "status": "complete_and_file_level_read_only",
            "manifest_excludes_itself_to_avoid_recursion": True,
            "final_audit": file_record(AUDIT_PATH),
            "analysis_directory": str(ANALYSIS_DIRECTORY),
            "analysis_file_count_excluding_manifest": len(analysis_inventory),
            "analysis_inventory": analysis_inventory,
            "analysis_inventory_sha256": inventory_sha256(analysis_inventory),
            "scale_run_root": str(SCALE_RUN_ROOT),
            "scale_file_count": len(scale_inventory),
            "scale_inventory": scale_inventory,
            "scale_inventory_sha256": inventory_sha256(scale_inventory),
            "seal_counts": {
                "analysis_files_processed": analysis_sealed_count,
                "scale_files_processed": scale_sealed_count,
            },
            "lineage_finalization_manifest": file_record(
                FINALIZATION_DIRECTORY / "finalization_manifest.json"
            ),
            "original_test_partition_evaluated": False,
        }
    )
    write_new_json(MANIFEST_PATH, manifest)
    make_file_read_only(MANIFEST_PATH)
    reloaded = read_json(MANIFEST_PATH)
    verify_self_seal(reloaded)
    require(is_read_only(MANIFEST_PATH), "post-training manifest is not read-only")
    return {"status": "completed", "manifest": file_record(MANIFEST_PATH)}


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(json.dumps(finalize_analysis(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AUDIT_PATH", "MANIFEST_PATH", "finalize_analysis"]
