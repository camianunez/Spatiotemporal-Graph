from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from fncs_encoder_training.common import source_tree_manifest, tensor_state_sha256
from fortnite_parallel_trajectory_fncs_recovery.recovery import (
    NEW_TRAINING_SOURCE_SHA256,
    OLD_TRAINING_SOURCE_SHA256,
    _all_tensor_values_finite,
    _build_setups,
    _pid_alive,
    _strict_checkpoint_report,
    structure_sha256,
)
from fortnite_parallel_trajectory_fncs_training.binding import (
    assert_frozen_encoder,
    build_frozen_model,
)

from .common import (
    CHILD_RUN,
    CONFIG_PATH,
    EXPECTED_ENCODER_STATE_SHA256,
    EXPECTED_FINAL_ROUTE_NLL,
    EXPECTED_FINAL_STEP,
    EXPECTED_FIXED_COMPUTE_STEP,
    EXPECTED_INITIAL_DOWNSTREAM_SHA256,
    FINALIZATION_DIRECTORY,
    PARENT_RUN,
    RECOVERY_AUDIT,
    REPOSITORY_ROOT,
    STEP1840_CHECKPOINT,
    current_utc,
    directory_inventory,
    file_record,
    file_sha256,
    inventory_sha256,
    make_file_read_only,
    make_tree_files_read_only,
    payload_sha256,
    read_json,
    read_jsonl,
    require,
    self_seal,
    write_new_json,
)


FINALIZATION_SCHEMA = "fncs-decoder-completed-lineage-finalization:1.0"


def _finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    return False


def _tensor_raw_sha256(value: torch.Tensor) -> str:
    item = value.detach().cpu().contiguous()
    return hashlib.sha256(item.view(torch.uint8).numpy().tobytes()).hexdigest()


def _checkpoint_tensor_comparison(
    best_path: Path, last_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best_state = best.get("downstream_state")
    last_state = last.get("downstream_state")
    require(isinstance(best_state, Mapping), "best.pt downstream state is missing")
    require(isinstance(last_state, Mapping), "last.pt downstream state is missing")
    require(set(best_state) == set(last_state), "best/last downstream tensor names differ")
    rows: list[dict[str, Any]] = []
    all_equal = True
    for name in sorted(best_state):
        left = best_state[name]
        right = last_state[name]
        require(isinstance(left, torch.Tensor), f"best downstream {name} is not a tensor")
        require(isinstance(right, torch.Tensor), f"last downstream {name} is not a tensor")
        identical = (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
            and _tensor_raw_sha256(left) == _tensor_raw_sha256(right)
        )
        all_equal = all_equal and identical
        rows.append(
            {
                "name": name,
                "dtype": str(left.dtype),
                "shape": list(left.shape),
                "numel": left.numel(),
                "best_raw_tensor_sha256": _tensor_raw_sha256(left),
                "last_raw_tensor_sha256": _tensor_raw_sha256(right),
                "byte_identical": identical,
            }
        )
    best_summary = {
        "embedded_downstream_state_sha256": best.get("downstream_state_sha256"),
        "recomputed_downstream_state_sha256": tensor_state_sha256(best_state),
        "downstream_tensor_count": len(best_state),
        "all_tensor_values_finite": _all_tensor_values_finite(best_state),
    }
    last_summary = {
        "embedded_downstream_state_sha256": last.get("downstream_state_sha256"),
        "recomputed_downstream_state_sha256": tensor_state_sha256(last_state),
        "downstream_tensor_count": len(last_state),
        "all_tensor_values_finite": _all_tensor_values_finite(last_state),
    }
    require(
        best_summary["embedded_downstream_state_sha256"]
        == best_summary["recomputed_downstream_state_sha256"],
        "best.pt embedded downstream hash is invalid",
    )
    require(
        last_summary["embedded_downstream_state_sha256"]
        == last_summary["recomputed_downstream_state_sha256"],
        "last.pt embedded downstream hash is invalid",
    )
    require(all_equal, "final best.pt and last.pt downstream tensors are not byte-identical")
    del best, last
    return best_summary, last_summary, rows


def _manifest_audit(root: Path, *, allowed_mismatches: set[str]) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = read_json(manifest_path, f"{root.name} artifact manifest")
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for item in manifest.get("artifacts", []):
        require(isinstance(item, Mapping), "artifact manifest row is malformed")
        relative = item.get("path")
        require(isinstance(relative, str) and relative, "artifact manifest path is invalid")
        source = root / relative
        actual = file_record(source, base=root)
        matches = (
            actual["size_bytes"] == item.get("size_bytes")
            and actual["sha256"] == item.get("raw_sha256")
        )
        if not matches:
            mismatches.append(relative)
        rows.append(
            {
                "path": relative,
                "recorded_size_bytes": item.get("size_bytes"),
                "recorded_sha256": item.get("raw_sha256"),
                "current_size_bytes": actual["size_bytes"],
                "current_sha256": actual["sha256"],
                "matches_current_file": matches,
            }
        )
    require(
        set(mismatches) == allowed_mismatches,
        f"{root.name} artifact-manifest mismatches changed: {mismatches}",
    )
    return {
        "manifest": file_record(manifest_path),
        "recorded_status": manifest.get("status"),
        "entry_count": len(rows),
        "entries": rows,
        "current_mismatches": mismatches,
        "mismatch_interpretation": (
            "Known append-after-manifest artifacts are externally rehashed here; "
            "the historical manifest itself is preserved byte-for-byte."
            if mismatches
            else "All historical manifest entries match current bytes."
        ),
        "all_unexpected_mismatches_absent": True,
    }


def _access_report(path: Path, *, label: str) -> dict[str, Any]:
    raw = read_json(path, label)
    events = raw.get("events", [])
    require(isinstance(events, list), f"{label} events are missing")
    sealed = set(raw.get("sealed_test_session_ids", []))
    test_events = [
        event
        for event in events
        if isinstance(event, Mapping)
        and (event.get("partition") == "test" or event.get("session_id") in sealed)
    ]
    report = {
        "artifact": file_record(path),
        "schema_version": raw.get("schema_version"),
        "event_count": len(events),
        "sealed_test_session_count": raw.get("sealed_test_session_count"),
        "sealed_test_session_id_count": len(sealed),
        "sealed_test_session_ids_sha256": payload_sha256(sorted(sealed)),
        "test_event_count": len(test_events),
        "test_session_request_attempt_count": raw.get(
            "test_session_request_attempt_count"
        ),
        "test_session_parquet_open_attempt_count": raw.get(
            "test_session_parquet_open_attempt_count"
        ),
        "test_session_parquet_open_count": raw.get("test_session_parquet_open_count"),
        "test_partition_evaluated": raw.get("test_partition_evaluated"),
        "zero_test_attempts": raw.get("zero_test_attempts"),
        "zero_test_opens": raw.get("zero_test_opens"),
        "partition_request_counts": raw.get("session_request_counts"),
        "partition_dataset_attempt_counts": raw.get("dataset_attempt_counts"),
        "partition_dataset_open_counts": raw.get("dataset_open_counts"),
    }
    require(len(sealed) == 115, f"{label} does not seal exactly 115 test sessions")
    require(not test_events, f"{label} contains forbidden test-session events")
    require(
        report["test_session_request_attempt_count"] == 0
        and report["test_session_parquet_open_attempt_count"] == 0
        and report["test_session_parquet_open_count"] == 0
        and report["test_partition_evaluated"] is False
        and report["zero_test_attempts"] is True
        and report["zero_test_opens"] is True,
        f"{label} records forbidden test access",
    )
    return report


def _metrics_summary(parent: Path, child: Path) -> dict[str, Any]:
    parent_rows = read_jsonl(parent / "metrics.jsonl", "parent metrics")
    child_rows = read_jsonl(child / "metrics.jsonl", "child metrics")
    require(_finite_json(parent_rows), "parent metrics contain nonfinite JSON values")
    require(_finite_json(child_rows), "child metrics contain nonfinite JSON values")
    parent_steps = [row for row in parent_rows if row.get("type") == "optimizer_step"]
    child_steps = [row for row in child_rows if row.get("type") == "optimizer_step"]
    durable_parent = [
        row for row in parent_steps if int(row.get("optimizer_step", -1)) <= 1840
    ]
    excluded_parent = [
        row for row in parent_steps if int(row.get("optimizer_step", -1)) > 1840
    ]
    require(
        [row["optimizer_step"] for row in durable_parent] == list(range(1, 1841)),
        "parent durable metrics are not exactly steps 1..1840",
    )
    require(
        [row["optimizer_step"] for row in child_steps] == list(range(1841, 9201)),
        "child metrics are not exactly steps 1841..9200",
    )
    combined_steps = durable_parent + child_steps
    require(
        [row["optimizer_step"] for row in combined_steps] == list(range(1, 9201)),
        "logical optimizer history is not contiguous 1..9200",
    )
    parent_validation = [
        row
        for row in parent_rows
        if row.get("type") == "validation_epoch"
        and int(row.get("optimizer_step", -1)) <= 1840
    ]
    child_validation = [
        row for row in child_rows if row.get("type") == "validation_epoch"
    ]
    validation = parent_validation + child_validation
    require(
        [row["completed_epoch"] for row in validation] == list(range(1, 21)),
        "validation history does not cover epochs 1..20 exactly once",
    )
    require(
        [row["optimizer_step"] for row in validation]
        == [460 * epoch for epoch in range(1, 21)],
        "validation optimizer-step endpoints changed",
    )
    for row in validation:
        require(
            row.get("selection_metric") == "validation_route_mixture_nll",
            "validation selection metric changed",
        )
        require(
            math.isclose(
                float(row["selection_value"]),
                float(row["metrics"]["route_nll"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "validation selection value and route NLL differ",
        )
    values = [float(row["selection_value"]) for row in validation]
    best_index = min(range(len(values)), key=values.__getitem__)
    require(best_index == 19, "validation-selected checkpoint is not epoch 20")
    require(
        math.isclose(values[-1], EXPECTED_FINAL_ROUTE_NLL, rel_tol=0.0, abs_tol=1e-12),
        "final validation route NLL changed",
    )
    validation_trajectory = [
        {
            "completed_epoch": int(row["completed_epoch"]),
            "optimizer_step": int(row["optimizer_step"]),
            "route_nll": float(row["selection_value"]),
            "top_mode_ade_meters": row["metrics"]["primary_mode"]["ade_meters"],
            "last_valid_fde_meters": row["metrics"]["primary_mode"]["fde_meters"],
            "valid_route_count": row["metrics"]["valid_route_count"],
            "valid_horizon_count": row["metrics"]["valid_horizon_count"],
            "improved_incumbent": row["improved"],
            "validation_seconds": row["validation_seconds"],
        }
        for row in validation
    ]
    return {
        "schema_version": "fncs-decoder-training-metrics-summary:1.0",
        "created_utc": current_utc(),
        "parent_metrics": file_record(parent / "metrics.jsonl"),
        "child_metrics": file_record(child / "metrics.jsonl"),
        "logical_optimizer_step_count": len(combined_steps),
        "logical_optimizer_steps": {"first": 1, "last": 9200},
        "parent_durable_optimizer_step_count": len(durable_parent),
        "child_optimizer_step_count": len(child_steps),
        "excluded_nondurable_parent_tail": {
            "count": len(excluded_parent),
            "first_optimizer_step": (
                excluded_parent[0]["optimizer_step"] if excluded_parent else None
            ),
            "last_optimizer_step": (
                excluded_parent[-1]["optimizer_step"] if excluded_parent else None
            ),
            "reason": (
                "The recovery contract binds parent state at durable step 1840; "
                "later parent log rows were not checkpoint-durable and are replaced "
                "by child rows beginning at step 1841."
            ),
        },
        "continuity_gates": {
            "parent_durable_steps_exactly_1_through_1840": True,
            "child_steps_exactly_1841_through_9200": True,
            "no_duplicated_durable_steps": True,
            "no_missing_durable_steps": True,
            "validation_epochs_exactly_1_through_20": True,
        },
        "validation_trajectory": validation_trajectory,
        "best_validation": validation_trajectory[best_index],
        "endpoint_change": {
            "epoch_19_to_20_route_nll_absolute": values[-1] - values[-2],
            "epoch_19_to_20_route_nll_percent": (values[-1] / values[-2] - 1.0) * 100.0,
            "epoch_18_to_20_route_nll_absolute": values[-1] - values[-3],
            "minimum_occurs_at_endpoint": True,
            "interpretation_policy": (
                "Endpoint trend is descriptive only and cannot authorize extending "
                "the completed lineage."
            ),
        },
    }


def _frozen_encoder_proof(parent: Path, child: Path, current_setup: Any) -> dict[str, Any]:
    model, initial = build_frozen_model(current_setup, device="cpu")
    fresh = assert_frozen_encoder(model, current_setup, stage="post_training_finalization")
    require(
        initial["initial_downstream_parameter_sha256"]
        == EXPECTED_INITIAL_DOWNSTREAM_SHA256,
        "fresh downstream initialization hash changed",
    )
    canonical_tensors = {
        row["name"]: row for row in fresh.get("tensors", [])
    }
    require(len(canonical_tensors) == 124, "canonical encoder tensor count changed")
    proof_summaries: list[dict[str, Any]] = []
    for root in (parent, child):
        proof_path = root / "frozen_encoder_proof.json"
        proof = read_json(proof_path, f"{root.name} frozen proof")
        snapshots = proof.get("snapshots")
        require(isinstance(snapshots, list) and snapshots, "frozen proof has no snapshots")
        stages: list[dict[str, Any]] = []
        for snapshot in snapshots:
            require(isinstance(snapshot, Mapping), "frozen snapshot is malformed")
            tensor_rows = snapshot.get("tensors")
            require(isinstance(tensor_rows, list), "frozen snapshot tensor rows are missing")
            observed = {row.get("name"): row for row in tensor_rows if isinstance(row, Mapping)}
            require(set(observed) == set(canonical_tensors), "frozen snapshot tensor names changed")
            exact = all(
                observed[name].get("raw_tensor_sha256")
                == canonical_tensors[name].get("raw_tensor_sha256")
                and observed[name].get("dtype") == canonical_tensors[name].get("dtype")
                and observed[name].get("shape") == canonical_tensors[name].get("shape")
                for name in canonical_tensors
            )
            require(exact, f"frozen encoder snapshot changed at {snapshot.get('stage')}")
            require(
                snapshot.get("encoder_state_sha256") == EXPECTED_ENCODER_STATE_SHA256
                and snapshot.get("matches_canonical") is True
                and snapshot.get("requires_grad_parameter_count") == 0
                and snapshot.get("parameters_with_gradient") == 0,
                "frozen encoder snapshot invariant failed",
            )
            stages.append(
                {
                    "stage": snapshot.get("stage"),
                    "epoch": snapshot.get("epoch"),
                    "optimizer_step": snapshot.get("optimizer_step"),
                    "encoder_state_sha256": snapshot.get("encoder_state_sha256"),
                    "tensor_count": len(observed),
                    "all_tensor_hashes_match_fresh_canonical_encoder": True,
                }
            )
        require(
            proof.get("all_recorded_snapshots_match") is True
            and proof.get("canonical_encoder_state_sha256")
            == EXPECTED_ENCODER_STATE_SHA256,
            "frozen encoder proof summary changed",
        )
        proof_summaries.append(
            {
                "run": str(root),
                "artifact": file_record(proof_path),
                "snapshot_count": len(snapshots),
                "stages": stages,
            }
        )
    del model
    return {
        "schema_version": "fncs-decoder-frozen-encoder-final-proof:1.0",
        "created_utc": current_utc(),
        "canonical_encoder_state_sha256": EXPECTED_ENCODER_STATE_SHA256,
        "canonical_encoder_tensor_count": 124,
        "fresh_strict_reload": {
            "encoder_state_sha256": fresh["encoder_state_sha256"],
            "tensor_count": fresh["tensor_count"],
            "requires_grad_parameter_count": fresh["requires_grad_parameter_count"],
            "parameters_with_gradient": fresh["parameters_with_gradient"],
            "matches_canonical": fresh["matches_canonical"],
        },
        "initial_downstream_parameter_sha256": initial[
            "initial_downstream_parameter_sha256"
        ],
        "lineage_proofs": proof_summaries,
        "gates": {
            "every_recorded_tensor_hash_matches_fresh_canonical_encoder": True,
            "encoder_state_unchanged_for_entire_lineage": True,
            "encoder_never_trainable": True,
            "encoder_never_received_gradients": True,
        },
    }


def _required_artifacts(parent: Path, child: Path, recovery: Path) -> dict[str, Any]:
    groups = {
        "parent": [
            "best.pt",
            "last.pt",
            "best.json",
            "metrics.jsonl",
            "resolved_config.json",
            "split_manifest.json",
            "encoder_binding.json",
            "frozen_encoder_proof.json",
            "runtime.json",
            "environment.json",
            "data_access.json",
            "initial_downstream_parameters.json",
            "artifact_manifest.json",
        ],
        "child": [
            "best.pt",
            "last.pt",
            "best.json",
            "final_summary.json",
            "metrics.jsonl",
            "resolved_config.json",
            "split_manifest.json",
            "encoder_binding.json",
            "frozen_encoder_proof.json",
            "runtime.json",
            "environment.json",
            "data_access.json",
            "initial_downstream_parameters.json",
            "recovery_parent.json",
            "launch.json",
            "launch_verification.json",
            "first_durable_step.json",
            "artifact_manifest.json",
            "stdout.log",
            "stderr.log",
        ],
        "recovery": [
            "superseding_authorization.json",
            "step1840_checkpoint_binding.json",
            "source_difference_report.json",
            "resume_compatibility.json",
            "recovery_decision.json",
            "parent_inventory.json",
            "environment.json",
            "data_access_ledger.json",
            "deterministic_restore_comparison.json",
            "artifact_manifest.json",
            "checkpoint-copy/last-step-00001840.pt",
        ],
    }
    roots = {"parent": parent, "child": child, "recovery": recovery}
    return {
        name: [file_record(roots[name] / relative, base=roots[name]) for relative in files]
        for name, files in groups.items()
    }


def build_finalization_payloads(
    *, parent: Path, child: Path, recovery: Path, config_path: Path
) -> dict[str, dict[str, Any]]:
    require(parent.resolve() == PARENT_RUN.resolve(), "physical parent path changed")
    require(child.resolve() == CHILD_RUN.resolve(), "physical recovery-child path changed")
    require(recovery.resolve() == RECOVERY_AUDIT.resolve(), "recovery audit path changed")
    parent_resolved = read_json(parent / "resolved_config.json", "parent resolved config")
    current_setup, checkpoint_setup, old_manifest, current_manifest = _build_setups(
        config_path, parent_resolved
    )
    require(
        old_manifest.get("source_tree_sha256") == OLD_TRAINING_SOURCE_SHA256,
        "parent training source manifest changed",
    )
    require(
        current_manifest.get("source_tree_sha256") == NEW_TRAINING_SOURCE_SHA256,
        "recovery training source manifest changed",
    )

    child_resolved = read_json(child / "resolved_config.json", "child resolved config")
    best_json = read_json(child / "best.json", "final best metadata")
    latest_json = read_json(child / "latest.json", "final last metadata")
    runtime = read_json(child / "runtime.json", "final runtime")
    final_summary = read_json(child / "final_summary.json", "final summary")
    recovery_parent = read_json(child / "recovery_parent.json", "recovery parent record")
    recovery_decision = read_json(recovery / "recovery_decision.json", "recovery decision")
    launch_verification = read_json(
        child / "launch_verification.json", "child launch verification"
    )

    require(
        best_json.get("completed_epoch") == 20
        and best_json.get("optimizer_step") == EXPECTED_FINAL_STEP
        and best_json.get("selection_metric") == "validation_route_mixture_nll"
        and math.isclose(
            float(best_json.get("value")),
            EXPECTED_FINAL_ROUTE_NLL,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "best.json does not identify the final selected step and route NLL",
    )
    require(
        best_json.get("checkpoint") == "best.pt"
        and best_json.get("checkpoint_sha256") == file_sha256(child / "best.pt"),
        "best.json checkpoint binding changed",
    )
    require(
        latest_json.get("checkpoint") == "last.pt"
        and latest_json.get("checkpoint_sha256") == file_sha256(child / "last.pt")
        and latest_json.get("optimizer_step") == EXPECTED_FINAL_STEP
        and latest_json.get("completed_epochs") == 20,
        "latest.json does not identify completed step 9200",
    )
    require(
        runtime.get("status") == "completed"
        and runtime.get("optimizer_step") == EXPECTED_FINAL_STEP
        and runtime.get("completed_epochs") == 20,
        "runtime does not record completed step 9200",
    )
    require(
        final_summary.get("status") == "completed"
        and final_summary.get("optimizer_steps") == EXPECTED_FINAL_STEP
        and final_summary.get("completed_epochs") == 20
        and math.isclose(
            float(final_summary.get("best_validation_route_nll")),
            EXPECTED_FINAL_ROUTE_NLL,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "final summary changed",
    )
    require(
        recovery_parent.get("parent_run") == str(parent.resolve())
        and recovery_parent.get("verified_read_only_checkpoint_copy")
        == str(STEP1840_CHECKPOINT.resolve())
        and recovery_parent.get("next_optimizer_step") == 1841
        and recovery_parent.get("endpoint_optimizer_step") == 9200,
        "recovery parent-child binding changed",
    )
    require(
        recovery_decision.get("status") == "authorized"
        and recovery_decision.get("decision") == "launch_exact_child_continuation"
        and recovery_decision.get("launch_milestone", {}).get("status") == "passed"
        and recovery_decision.get("launch_milestone", {}).get("worker_pid")
        == runtime.get("process_id")
        == launch_verification.get("worker_pid")
        and launch_verification.get("status") == "passed",
        "recovery decision/child runtime binding changed",
    )
    worker_pid = int(runtime["process_id"])
    require(not _pid_alive(worker_pid), "completed lineage worker is unexpectedly active")
    require((child / "stderr.log").stat().st_size == 0, "child stderr is not empty")

    parent_last = _strict_checkpoint_report(
        STEP1840_CHECKPOINT, checkpoint_setup, label="authentic_parent_step1840"
    )
    parent_best = _strict_checkpoint_report(
        parent / "best.pt", checkpoint_setup, label="parent_best"
    )
    final_best = _strict_checkpoint_report(
        child / "best.pt", current_setup, label="final_best"
    )
    final_last = _strict_checkpoint_report(
        child / "last.pt", current_setup, label="final_last"
    )
    require(
        parent_last["training_state"]["optimizer_step"] == EXPECTED_FIXED_COMPUTE_STEP
        and parent_last["scheduler_step_index"] == EXPECTED_FIXED_COMPUTE_STEP
        and parent_last["optimizer"]["recorded_step_values"]
        == [EXPECTED_FIXED_COMPUTE_STEP],
        "authentic fixed-compute checkpoint state changed",
    )
    expected_final_state = {
        "completed_epochs": 20,
        "optimizer_step": 9200,
        "current_epoch": 21,
        "next_batch_index": 0,
        "first_optimizer_step_completed": True,
    }
    for name, report in (("best.pt", final_best), ("last.pt", final_last)):
        require(report["training_state"] == expected_final_state, f"{name} state is not final")
        require(report["scheduler_step_index"] == 9200, f"{name} scheduler is not complete")
        require(
            report["scheduler_state"].get("total_steps") == 9200
            and report["optimizer"]["recorded_step_values"] == [9200]
            and report["optimizer"]["state_entry_count"] == 135
            and report["optimizer"]["parameters_with_first_and_second_moments"] == 135,
            f"{name} optimizer/scheduler state looks restarted or incomplete",
        )
        require(
            report["best_validation_route_nll"] is not None
            and math.isclose(
                float(report["best_validation_route_nll"]),
                EXPECTED_FINAL_ROUTE_NLL,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{name} best metric changed",
        )
        require(
            report["data_access_zero_test_attempts"] is True
            and report["data_access_zero_test_opens"] is True
            and report["test_partition_evaluated"] is False,
            f"{name} records forbidden test access",
        )

    best_tensor_summary, last_tensor_summary, tensor_rows = (
        _checkpoint_tensor_comparison(child / "best.pt", child / "last.pt")
    )
    checkpoint_comparison = {
        "schema_version": "fncs-decoder-checkpoint-comparison:1.0",
        "created_utc": current_utc(),
        "authentic_parent_step1840": parent_last,
        "parent_best_before_recovery": parent_best,
        "final_best": final_best,
        "final_last": final_last,
        "final_best_tensor_summary": best_tensor_summary,
        "final_last_tensor_summary": last_tensor_summary,
        "downstream_tensor_comparison": tensor_rows,
        "gates": {
            "best_identifies_step_9200": True,
            "best_identifies_validation_route_nll_7_0043": True,
            "last_identifies_completed_step_9200": True,
            "all_embedded_downstream_hashes_validate": True,
            "best_and_last_downstream_tensors_byte_identical": True,
            "optimizer_moments_at_step_9200": True,
            "scheduler_at_original_terminal_step_9200": True,
            "authentic_step1840_checkpoint_available_and_compatible": True,
        },
    }

    metrics = _metrics_summary(parent, child)
    frozen = _frozen_encoder_proof(parent, child, current_setup)
    parent_access = _access_report(parent / "data_access.json", label="parent access ledger")
    child_access = _access_report(child / "data_access.json", label="child access ledger")
    recovery_access = _access_report(
        recovery / "data_access_ledger.json", label="recovery access ledger"
    )
    sealed_hashes = {
        parent_access["sealed_test_session_ids_sha256"],
        child_access["sealed_test_session_ids_sha256"],
        recovery_access["sealed_test_session_ids_sha256"],
    }
    require(len(sealed_hashes) == 1, "test denylist changed across lineage records")
    data_access = {
        "schema_version": "fncs-decoder-data-access-finalization:1.0",
        "created_utc": current_utc(),
        "parent": parent_access,
        "child": child_access,
        "recovery": recovery_access,
        "test_denylist_sha256": next(iter(sealed_hashes)),
        "test_session_count": 115,
        "gates": {
            "zero_test_requests": True,
            "zero_test_parquet_open_attempts": True,
            "zero_test_parquet_opens": True,
            "zero_test_events": True,
            "test_partition_never_evaluated": True,
            "same_115_session_denylist_throughout": True,
        },
    }

    parent_manifest = _manifest_audit(
        parent,
        allowed_mismatches={"data_access.json", "metrics.jsonl", "runtime.json"},
    )
    child_manifest = _manifest_audit(child, allowed_mismatches={"stdout.log"})
    source_difference = read_json(
        recovery / "source_difference_report.json", "source difference report"
    )
    require(
        source_difference.get("model_execution_changed") is False
        and source_difference.get("losses_changed") is False
        and source_difference.get("optimizer_updates_changed") is False
        and source_difference.get("scheduler_changed") is False
        and source_difference.get("sampling_changed") is False
        and source_difference.get("checkpoint_payload_changed") is False,
        "recovery source transition was not resume-validation-only",
    )
    architecture_manifest = source_tree_manifest(
        REPOSITORY_ROOT / "ml" / "src" / "fortnite_parallel_trajectory"
    )
    require(
        architecture_manifest.get("source_tree_sha256")
        == current_setup.architecture_source_manifest.get("source_tree_sha256"),
        "architecture source manifest changed",
    )

    required = _required_artifacts(parent, child, recovery)
    completed_lineage = {
        "schema_version": "fncs-decoder-completed-lineage:1.0",
        "created_utc": current_utc(),
        "logical_lineage_name": PARENT_RUN.name,
        "physical_lineage": {
            "parent": str(parent.resolve()),
            "recovery_child": str(child.resolve()),
            "recovery_audit": str(recovery.resolve()),
            "authentic_step1840_checkpoint": str(STEP1840_CHECKPOINT.resolve()),
            "logical_step_ranges": [
                {"directory": "parent", "first": 1, "last": 1840},
                {"directory": "recovery_child", "first": 1841, "last": 9200},
            ],
        },
        "completed_state": {
            "optimizer_steps": 9200,
            "epochs": 20,
            "best_validation_route_nll": EXPECTED_FINAL_ROUTE_NLL,
            "best_checkpoint_step": 9200,
            "selection_metric": "validation_route_mixture_nll",
            "runtime_status": "completed",
            "worker_pid_active_after_completion": False,
            "stderr_size_bytes": 0,
            "normal_completion_evidence": [
                "terminal final_summary.json status=completed",
                "terminal runtime.json status=completed at step 9200/epoch 20",
                "worker PID is no longer active",
                "stderr.log is empty",
            ],
        },
        "resolved_contracts": {
            "configuration": file_record(config_path),
            "parent_resolved_config": file_record(parent / "resolved_config.json"),
            "child_resolved_config": file_record(child / "resolved_config.json"),
            "split_manifest": file_record(child / "split_manifest.json"),
            "encoder_binding": file_record(child / "encoder_binding.json"),
            "runtime": file_record(child / "runtime.json"),
            "environment": file_record(child / "environment.json"),
            "recovery_parent": file_record(child / "recovery_parent.json"),
            "source_difference_report": file_record(
                recovery / "source_difference_report.json"
            ),
            "parent_training_source_manifest": old_manifest,
            "recovery_training_source_manifest": current_manifest,
            "architecture_source_manifest": architecture_manifest,
        },
        "existing_artifact_manifest_audits": {
            "parent": parent_manifest,
            "child": child_manifest,
            "recovery": file_record(recovery / "artifact_manifest.json"),
        },
        "required_artifact_hashes": required,
        "gates": {
            "physical_parent_and_child_identified": True,
            "all_required_artifacts_present_and_externally_hashed": True,
            "configuration_split_encoder_runtime_environment_and_sources_bound": True,
            "logical_steps_contiguous_1_through_9200": True,
            "best_and_last_strictly_reload": True,
            "frozen_encoder_unchanged": True,
            "zero_forbidden_test_access": True,
            "completed_lineage_not_modified_during_verification": True,
        },
    }
    # Bind the large supporting reports without recursively embedding them.
    completed_lineage["supporting_payload_sha256"] = {
        "checkpoint_comparison": payload_sha256(checkpoint_comparison),
        "training_metrics_summary": payload_sha256(metrics),
        "frozen_encoder_final_proof": payload_sha256(frozen),
        "data_access_finalization": payload_sha256(data_access),
    }
    return {
        "completed_lineage.json": self_seal(completed_lineage),
        "checkpoint_comparison.json": self_seal(checkpoint_comparison),
        "training_metrics_summary.json": self_seal(metrics),
        "frozen_encoder_final_proof.json": self_seal(frozen),
        "data_access_finalization.json": self_seal(data_access),
    }


def finalize_lineage(
    destination: str | Path = FINALIZATION_DIRECTORY,
) -> dict[str, Any]:
    output = Path(destination).resolve()
    require(not output.exists(), f"refusing to overwrite finalization directory: {output}")
    payloads = build_finalization_payloads(
        parent=PARENT_RUN.resolve(),
        child=CHILD_RUN.resolve(),
        recovery=RECOVERY_AUDIT.resolve(),
        config_path=CONFIG_PATH.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        write_new_json(output / name, payload)

    # The completed parent/child lineage and recovery evidence become file-level
    # read-only only after every strict gate above has passed.
    sealed_counts = {
        "parent_files": make_tree_files_read_only(PARENT_RUN),
        "child_files": make_tree_files_read_only(CHILD_RUN),
        "recovery_files": make_tree_files_read_only(RECOVERY_AUDIT),
    }
    for external in (
        PARENT_RUN.with_name(PARENT_RUN.name + ".stdout.log"),
        PARENT_RUN.with_name(PARENT_RUN.name + ".stderr.log"),
    ):
        if external.is_file():
            make_file_read_only(external)
    for name in payloads:
        make_file_read_only(output / name)

    parent_inventory = directory_inventory(PARENT_RUN)
    child_inventory = directory_inventory(CHILD_RUN)
    recovery_inventory = directory_inventory(RECOVERY_AUDIT)
    output_rows = [file_record(output / name, base=output) for name in sorted(payloads)]
    require(all(row["read_only"] for row in output_rows), "finalization payloads are writable")
    require(all(row["read_only"] for row in parent_inventory), "parent lineage is writable")
    require(all(row["read_only"] for row in child_inventory), "child lineage is writable")
    require(all(row["read_only"] for row in recovery_inventory), "recovery evidence is writable")
    manifest = self_seal(
        {
            "schema_version": "fncs-decoder-finalization-artifact-manifest:1.0",
            "created_utc": current_utc(),
            "status": "complete_and_file_level_read_only",
            "finalization_directory": str(output),
            "immutability": {
                "mechanism": "Windows read-only attribute on every file",
                "source_lineage_sealed_after_successful_verification": True,
                "manifest_self_hash_excluded_to_avoid_recursion": True,
                "sealed_file_counts": sealed_counts,
            },
            "artifacts": output_rows,
            "source_inventory_hashes": {
                "parent": inventory_sha256(parent_inventory),
                "child": inventory_sha256(child_inventory),
                "recovery": inventory_sha256(recovery_inventory),
            },
            "source_file_counts": {
                "parent": len(parent_inventory),
                "child": len(child_inventory),
                "recovery": len(recovery_inventory),
            },
        }
    )
    write_new_json(output / "finalization_manifest.json", manifest)
    make_file_read_only(output / "finalization_manifest.json")
    final_rows = directory_inventory(output)
    require(len(final_rows) == 6, "finalization directory does not contain exactly six files")
    require(all(row["read_only"] for row in final_rows), "finalization directory has writable files")
    return {
        "status": "completed",
        "schema_version": FINALIZATION_SCHEMA,
        "finalization_directory": str(output),
        "artifact_count": len(final_rows),
        "artifact_manifest_sha256": file_sha256(output / "finalization_manifest.json"),
        "lineage_files_read_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize the completed FNCS decoder lineage.")
    parser.add_argument("--destination", type=Path, default=FINALIZATION_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(finalize_lineage(args.destination), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
