from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fortnite_encoder.config import RotationLossConfig
from fortnite_encoder.contracts import RotationLogits, RotationTargets
from fortnite_encoder.losses import compute_rotation_loss
from fncs_encoder_training.common import atomic_write_json, rendered_json
from fncs_encoder_training.evaluate import (
    EvaluationAccessLedger,
    EvaluationContractError,
    EvaluationMetrics,
    LOCKED_CHECKPOINT_SHA256,
    REQUIRED_ARTIFACTS,
    _metric_payload,
    _predeclared_metrics,
    _seal,
    _static_safety_audit,
    _verify_metric_payload,
    _verify_seal,
)


def test_artifact_self_seal_detects_mutation() -> None:
    artifact = _seal({"schema_version": "fixture:1.0", "value": 7})
    _verify_seal(artifact, "fixture")
    changed = dict(artifact)
    changed["value"] = 8
    with pytest.raises(EvaluationContractError, match="self-hash"):
        _verify_seal(changed, "fixture")


def test_deterministic_metric_payload_is_byte_stable_and_self_bound() -> None:
    first = _metric_payload({"partition": "validation", "metric": 1.25})
    second = _metric_payload({"metric": 1.25, "partition": "validation"})
    _verify_metric_payload(first, "first")
    _verify_metric_payload(second, "second")
    assert rendered_json(first) == rendered_json(second)
    changed = dict(first)
    changed["metric"] = 2.0
    with pytest.raises(EvaluationContractError, match="self-hash"):
        _verify_metric_payload(changed, "changed")


def test_access_ledger_requires_contract_and_allows_only_one_test_pass(
    tmp_path: Path,
) -> None:
    test_id = "test-session"
    validation_id = "validation-session"
    ledger = EvaluationAccessLedger(
        path=tmp_path / "data_access_ledger.json",
        partition_by_session={test_id: "test", validation_id: "validation"},
        test_session_ids=(test_id,),
        split_manifest_sha256="1" * 64,
    )
    with pytest.raises(EvaluationContractError, match="not active"):
        ledger.attempt(test_id, "match_samples.parquet", "fixture")
    assert ledger.test_attempts == 0
    assert ledger.test_opens == 0

    for number in (1, 2):
        ledger.start_validation_pass(number)
        ledger.attempt(validation_id, "match_samples.parquet", "fixture")
        ledger.opened(validation_id, "match_samples.parquet", "fixture")
        ledger.complete_validation_pass(number)

    contract = {
        "authorization": {
            "authorized": True,
            "ordered_test_session_ids": [test_id],
        },
        "payload_sha256": "2" * 64,
    }
    contract_path = tmp_path / "contract.json"
    atomic_write_json(contract_path, contract)
    ledger.authorize_test(contract_path, contract)
    ledger.start_test_pass()
    ledger.attempt(test_id, "match_samples.parquet", "fixture")
    ledger.opened(test_id, "match_samples.parquet", "fixture")
    ledger.complete_test_pass()
    report = json.loads((tmp_path / "data_access_ledger.json").read_text())
    _verify_seal(report, "ledger")
    assert report["test_evaluation_passes_started"] == 1
    assert report["test_evaluation_passes_completed"] == 1
    assert report["test_partition_unsealed"] is True
    with pytest.raises(EvaluationContractError, match="cannot be retried"):
        ledger.start_test_pass()


def test_access_ledger_rejects_session_outside_authorized_list(tmp_path: Path) -> None:
    ledger = EvaluationAccessLedger(
        path=tmp_path / "ledger.json",
        partition_by_session={"authorized": "test", "other": "test"},
        test_session_ids=("authorized",),
        split_manifest_sha256="3" * 64,
    )
    for number in (1, 2):
        ledger.start_validation_pass(number)
        ledger.complete_validation_pass(number)
    contract = {
        "authorization": {
            "authorized": True,
            "ordered_test_session_ids": ["authorized"],
        },
        "payload_sha256": "4" * 64,
    }
    contract_path = tmp_path / "contract.json"
    atomic_write_json(contract_path, contract)
    ledger.authorize_test(contract_path, contract)
    ledger.start_test_pass()
    with pytest.raises(EvaluationContractError, match="unauthorized"):
        ledger.attempt("other", "match_samples.parquet", "fixture")
    assert ledger.test_attempts == 0


def test_secondary_metric_reductions_and_counts_are_fixed() -> None:
    position_logits = torch.full((1, 1, 2, 3, 5), -4.0)
    position_targets = torch.tensor([[[[0, 1, 2], [4, 4, 4]]]])
    for team in range(2):
        for horizon in range(3):
            label = int(position_targets[0, 0, team, horizon])
            position_logits[0, 0, team, horizon, label] = 4.0
    entry_logits = torch.tensor([[[[3.0, 0.0, -1.0], [0.0, 3.0, -1.0]]]])
    survival_logits = torch.tensor([[[2.0, -2.0]]])
    placement_logits = torch.full((1, 1, 2, 5), -3.0)
    placement_logits[0, 0, 0, 0] = 3.0
    placement_logits[0, 0, 1, 4] = 3.0
    query_mask = torch.ones((1, 1, 2), dtype=torch.bool)
    logits = RotationLogits(
        future_position=position_logits,
        zone_entry=entry_logits,
        survival=survival_logits,
        placement=placement_logits,
        query_mask=query_mask,
    )
    targets = RotationTargets(
        future_position=position_targets,
        future_position_mask=torch.ones((1, 1, 2, 3), dtype=torch.bool),
        zone_entry=torch.tensor([[[0, 1]]]),
        zone_entry_mask=query_mask.clone(),
        survival=torch.tensor([[[1.0, 0.0]]]),
        survival_mask=query_mask.clone(),
        placement=torch.tensor([[[0, 4]]]),
        placement_mask=query_mask.clone(),
    )
    weights = RotationLossConfig()
    loss = compute_rotation_loss(logits, targets, weights)
    accumulator = EvaluationMetrics(weights, position_classes=5)
    accumulator.update(SimpleNamespace(logits=logits), targets, loss)
    result = accumulator.result()
    assert result["valid_target_counts"] == {
        "future_position_15s": 2,
        "future_position_30s": 2,
        "future_position_60s": 2,
        "zone_entry": 2,
        "survival": 2,
        "placement": 2,
    }
    for horizon in ("15s", "30s", "60s"):
        assert result["accuracies"]["future_position"][horizon][
            "regular_cell_top1"
        ]["value"] == 1.0
        assert result["accuracies"]["future_position"][horizon][
            "eliminated_class"
        ]["value"] == 1.0
    assert result["accuracies"]["zone_entry_top1"]["value"] == 1.0
    assert result["accuracies"]["survival_threshold_0_5"]["value"] == 1.0
    assert result["accuracies"]["placement_top1"]["value"] == 1.0
    assert result["placement_macro_f1"]["value"] == 0.4
    assert result["nonfinite"]["total_count"] == 0


def test_predeclared_metric_contract_has_no_window_bootstrap() -> None:
    metrics = _predeclared_metrics()
    assert metrics["primary"]["name"] == "future_position"
    assert metrics["confidence_intervals"]["reported"] is False
    assert metrics["confidence_intervals"]["individual_window_bootstrap_used"] is False
    assert "placement_macro_f1" in metrics["secondary"]
    assert "validation_to_test_absolute_and_relative_generalization_gaps" in metrics[
        "secondary"
    ]


def test_evaluator_source_has_no_training_backward_optimizer_or_scheduler_calls() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "fncs_encoder_training" / "evaluate.py"
    report = _static_safety_audit(source)
    assert report == {
        "ast_parsed": True,
        "backward_call_count": 0,
        "training_entry_point_call_count": 0,
        "optimizer_construction_call_count": 0,
        "scheduler_construction_call_count": 0,
        "passed": True,
    }
    text = source.read_text(encoding="utf-8")
    assert "with torch.inference_mode():" in text
    assert "model.eval()" in text


def test_locked_checkpoint_uses_complete_sha256_and_required_artifact_set() -> None:
    assert LOCKED_CHECKPOINT_SHA256 == (
        "dc8d2dc2e9bfce5ed9cacd557190dd33ca2d8af5d6225ae54464234d8cc3992a"
    )
    assert len(LOCKED_CHECKPOINT_SHA256) == 64
    assert REQUIRED_ARTIFACTS == {
        "checkpoint_selection.json",
        "evaluation_contract.json",
        "validation_reproduction.json",
        "test_evaluation.json",
        "generalization_gap.json",
        "data_access_ledger.json",
        "environment.json",
        "evaluator_source_manifest.json",
        "final_evaluation_summary.json",
        "artifact_manifest.json",
    }

