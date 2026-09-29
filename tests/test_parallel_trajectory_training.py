from __future__ import annotations

import ast
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from fortnite_encoder.world_grid import WorldGridProfile
from fortnite_parallel_trajectory.contracts import (
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
)
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel
from fortnite_parallel_trajectory_training.config import (
    ParallelTrajectoryTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
)
from fortnite_parallel_trajectory_training.data import (
    ParallelTrajectorySessionRepository,
    collate_parallel_trajectory_targets,
    stable_seed,
    training_session_order,
    uniform_sample,
)
from fortnite_parallel_trajectory_training.metrics import (
    ParallelTrajectoryMetricAccumulator,
)
from fortnite_parallel_trajectory_training.optimization import (
    PiecewiseLinearScheduler,
    apply_freeze_policy,
    build_adamw_optimizer,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "ml" / "src" / "fortnite_parallel_trajectory_training"


def _profile() -> WorldGridProfile:
    return WorldGridProfile.load(
        ROOT / "configs" / "world-grid" / "model-world-grid-v1.json"
    )


def _config(**changes) -> ParallelTrajectoryTrainingConfig:
    digest = "0" * 64
    values = {
        "schema_version": "parallel_trajectory_training_config_v1",
        "architecture_id": "parallel_trajectory_decoder_v1",
        "target_contract_version": "parallel-trajectory-targets:1.0",
        "run_directory": "runs/new",
        "source_checkpoint": "runs/source/best.pt",
        "expected_source_checkpoint_sha256": digest,
        "forbidden_phase_weighted_checkpoint": "runs/forbidden/last.pt",
        "forbidden_phase_weighted_checkpoint_sha256": digest,
        "canonical_source_config": "configs/source.json",
        "expected_canonical_source_config_raw_sha256": digest,
        "lineage_report": "audit/lineage.json",
        "expected_lineage_report_raw_sha256": digest,
        "lineage_classification": "scope_changed_only",
        "legacy_encoder_digest": digest,
        "current_encoder_digest": digest,
        "architecture_package_digest": digest,
        "training_package_digest": digest,
        "expected_transferred_tensor_count": 146,
        "expected_transferred_tensor_manifest_sha256": digest,
        "dataset_root": "FNDATA/decoded-v2-new",
        "split_manifest_path": "configs/split.json",
        "expected_split_manifest_sha256": digest,
        "expected_split_manifest_raw_sha256": digest,
        "expected_split_counts": {"train": 86, "validation": 11, "test": 10},
        "expected_ancestral_source_split_counts": {
            "train": 33,
            "validation": 4,
            "test": 4,
        },
        "expected_ingestion_report_sha256": digest,
        "dataset_validation_path": "audit/validation.json",
        "expected_dataset_validation_report_hash": digest,
        "expected_dataset_validation_raw_sha256": digest,
        "world_grid_profile_path": "configs/world-grid.json",
        "expected_world_grid_profile_hash": digest,
        "expected_world_grid_profile_raw_sha256": digest,
        "world_grid_audit_path": "audit/world-grid.json",
        "expected_world_grid_audit_payload_sha256": digest,
        "expected_world_grid_audit_raw_sha256": digest,
        "expected_world_grid_coordinate_audit_sha256": digest,
        "seed": 20260804,
        "batch_size": 2,
        "queries_per_session": 16,
        "validation_queries_per_session": 32,
        "accumulation_steps": 1,
        "context_length_ticks": 64,
        "max_queries_per_forward": 4,
        "max_epochs": 20,
        "updates_per_epoch": 43,
        "max_optimizer_steps": 860,
        "decoder_learning_rate": 3e-4,
        "congestion_learning_rate": 3e-4,
        "encoder_learning_rate": 3e-5,
        "adamw_betas": (0.9, 0.999),
        "adamw_epsilon": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "scheduler_type": "parallel_trajectory_piecewise_linear_v1",
        "scheduler_warmup_updates": 43,
        "scheduler_plateau_updates": 43,
        "scheduler_decay_updates": 774,
        "freeze_inherited_epochs": 1,
        "device": "cuda:0",
        "precision": "bf16",
        "pin_memory": True,
    }
    values.update(changes)
    return ParallelTrajectoryTrainingConfig(**values)


class _DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def _model() -> ParallelTrajectoryModel:
    torch.manual_seed(401)
    return ParallelTrajectoryModel(_profile(), _DummyEncoder())


def test_strict_versioned_configuration_and_exact_numeric_contract(tmp_path: Path) -> None:
    config = _config()
    raw = config.to_dict()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = ParallelTrajectoryTrainingConfig.from_json(path)
    assert loaded.schema_version == "parallel_trajectory_training_config_v1"
    assert Path(loaded.run_directory) == (tmp_path / "runs/new").resolve()
    assert loaded.expected_split_counts == {"train": 86, "validation": 11, "test": 10}
    assert loaded.expected_ancestral_source_split_counts == {
        "train": 33,
        "validation": 4,
        "test": 4,
    }

    raw["unknown"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TrainingConfigurationError, match="keys mismatch"):
        ParallelTrajectoryTrainingConfig.from_json(path)
    with pytest.raises(TrainingConfigurationError, match="fixed at 43"):
        _config(scheduler_warmup_updates=42, scheduler_decay_updates=775)


def test_optimizer_groups_decay_exclusions_freeze_and_fixed_scheduler() -> None:
    model = _model()
    optimizer, contract = build_adamw_optimizer(model, _config())
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "decoder_decay",
        "decoder_no_decay",
        "congestion_decay",
        "congestion_no_decay",
        "encoder_decay",
        "encoder_no_decay",
    ]
    names = [name for _, group_names in contract.group_parameter_names for name in group_names]
    assert len(names) == len(set(names)) == len(tuple(model.parameters()))
    for group in optimizer.param_groups:
        if group["group_name"].endswith("no_decay"):
            assert group["weight_decay"] == 0.0
        else:
            assert group["weight_decay"] == 0.01

    assert apply_freeze_policy(model, 1) == "inherited_frozen"
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert apply_freeze_policy(model, 2) == "all_trainable"
    assert all(parameter.requires_grad for parameter in model.parameters())

    scheduler = PiecewiseLinearScheduler(optimizer, _config())
    assert scheduler.ratio_at(1) == pytest.approx(1 / 43)
    assert scheduler.ratio_at(43) == 1.0
    assert scheduler.ratio_at(44) == 1.0
    assert scheduler.ratio_at(86) == 1.0
    assert scheduler.ratio_at(87) == pytest.approx(773 / 774)
    assert scheduler.ratio_at(860) == 0.0
    for _ in range(860):
        scheduler.step()
    assert scheduler.step_index == 860
    with pytest.raises(RuntimeError, match="advanced beyond"):
        scheduler.step()


def test_repository_rejects_validation_and_test_before_any_reader(tmp_path: Path) -> None:
    calls: list[str] = []

    def reader(path):
        calls.append(str(path))
        raise AssertionError("reader must not be invoked")

    paths = {
        "train-id": tmp_path / "train",
        "validation-id": tmp_path / "validation",
        "test-id": tmp_path / "test",
    }
    repository = ParallelTrajectorySessionRepository(
        paths,
        {
            "train-id": "train",
            "validation-id": "validation",
            "test-id": "test",
        },
        active_partitions=("train",),
        test_session_ids=("test-id",),
        profile=_profile(),
        match_loader=reader,
    )
    repository.attach_data_access_audit(
        tmp_path / "access.json", split_manifest_sha256="1" * 64
    )
    with pytest.raises(TrainingCompatibilityError, match="inactive"):
        repository.get("validation-id")
    with pytest.raises(TrainingCompatibilityError, match="sealed test"):
        repository.get("test-id")
    assert calls == []
    report = repository.report()
    assert report["zero_validation_and_test_attempts"]
    assert report["zero_validation_and_test_opens"]
    assert report["test_session_parquet_open_attempt_count"] == 0
    assert report["test_session_parquet_open_count"] == 0


def _targets(value: float) -> ParallelTrajectoryTargets:
    displacement = torch.zeros(1, 12, 2)
    displacement[:, :2] = value
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[:, :2] = True
    return ParallelTrajectoryTargets(
        target_displacements=displacement,
        target_mask=mask,
        query_mask=torch.ones(1, dtype=torch.bool),
        current_xy=torch.zeros(1, 2),
        future_xy=torch.zeros(1, 12, 2),
        world_grid_profile_id="model-world-grid-v1",
        world_grid_profile_hash=_profile().profile_hash,
    )


def test_target_collation_and_deterministic_sampling_contracts() -> None:
    combined = collate_parallel_trajectory_targets((_targets(1.0), _targets(2.0)))
    assert combined.target_displacements.shape == (2, 12, 2)
    assert combined.target_mask.sum().item() == 4
    assert combined.query_mask.tolist() == [True, True]
    values = tuple(range(20))
    first = uniform_sample(values, 8, seed=stable_seed(7, "sample"))
    second = uniform_sample(values, 8, seed=stable_seed(7, "sample"))
    assert first == second and len(set(first)) == 8
    order = training_session_order(
        tuple(f"session-{index}" for index in range(10)), seed=7, epoch=1
    )
    assert order == training_session_order(
        tuple(reversed(tuple(f"session-{index}" for index in range(10)))),
        seed=7,
        epoch=1,
    )


def test_reporting_metrics_include_all_required_nonobjective_diagnostics() -> None:
    targets = _targets(0.5)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    probabilities = torch.softmax(logits, dim=-1)
    means = torch.zeros(1, 5, 12, 2)
    for mode in range(5):
        means[:, mode] = 0.1 * mode
    primary_index = probabilities.argmax(dim=-1)
    primary = means[torch.arange(1), primary_index]
    marginal = (probabilities[:, :, None, None] * means).sum(dim=1)
    output = ParallelTrajectoryOutput(
        mode_logits=logits,
        mode_probabilities=probabilities,
        means=means,
        scales=torch.ones(1, 5, 12, 2),
        correlations=torch.zeros(1, 5, 12),
        primary_mode_index=primary_index,
        primary_route_normalized=primary,
        marginal_mean_route_normalized=marginal,
        query_mask=torch.ones(1, dtype=torch.bool),
        horizon_mask=targets.target_mask,
        current_xy=torch.zeros(1, 2),
    )
    accumulator = ParallelTrajectoryMetricAccumulator(_profile())
    accumulator.update(output, targets)
    report = accumulator.metrics()
    assert report["optimization_objective"] == "route_mixture_nll_only"
    assert report["valid_route_count"] == 1
    assert report["valid_horizon_count"] == 2
    assert report["primary_mode"]["ade_meters"] is not None
    assert report["marginal_mean"]["fde_meters"] is not None
    assert report["oracle_diagnostic_only"]["minade_meters"] is not None
    utilization = report["mode_utilization"]
    assert len(utilization["mean_probabilities"]) == 5
    assert len(utilization["mean_posterior_route_responsibilities"]) == 5
    assert sum(utilization["argmax_mode_counts"]) == 1
    assert utilization["mean_pairwise_route_separation_meters"] > 0


def test_training_integration_does_not_import_rejected_model_families() -> None:
    imports: set[str] = set()
    for path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert not any(name.startswith("fortnite_early_zone") for name in imports)
    training_source = (PACKAGE_ROOT / "training.py").read_text(encoding="utf-8")
    for forbidden in (
        "teacher_forcing",
        "beam_search",
        "movement_bce",
        "diversity_loss",
        "entropy_regularization",
    ):
        assert forbidden not in training_source
    assert "route_mixture_nll" in training_source
