from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from fortnite_encoder.model import SpatiotemporalEncoder
from fortnite_encoder.world_grid import WorldGridProfile
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel
from fortnite_parallel_trajectory_fncs_training.binding import FrozenEncoder
from fortnite_parallel_trajectory_fncs_training.config import (
    FrozenFNCSTrainingConfig,
    TrainingConfigurationError,
)
from fortnite_parallel_trajectory_fncs_training.data import DatasetAccessLedger
from fortnite_parallel_trajectory_fncs_training.optimization import (
    PiecewiseLinearScheduler,
    build_optimizer,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "configs" / "parallel-trajectory-training-fncs-frozen-encoder-v1.json"
)
PACKAGE_ROOT = (
    ROOT / "ml" / "src" / "fortnite_parallel_trajectory_fncs_training"
)


def _config() -> FrozenFNCSTrainingConfig:
    return FrozenFNCSTrainingConfig.from_json(CONFIG_PATH)


def _profile() -> WorldGridProfile:
    return WorldGridProfile.load(
        ROOT / "configs" / "world-grid" / "model-world-grid-v1.json"
    )


def test_pinned_fncs_decoder_configuration_and_recomputed_schedule() -> None:
    config = _config()
    assert config.expected_split_counts == {
        "train": 920,
        "validation": 115,
        "test": 115,
    }
    assert config.batch_size == 2
    assert config.updates_per_epoch == 460
    assert config.max_epochs == 20
    assert config.max_optimizer_steps == 9200
    assert config.scheduler_warmup_updates == 460
    assert config.scheduler_constant_updates == 460
    assert config.scheduler_decay_updates == 8280
    assert config.scheduler_final_lr_ratio == 0.1
    assert config.objective == "route_mixture_nll_only"
    assert config.congestion_auxiliary_objective == "none"
    assert config.seed == config.initialization_seed == 20260727


def test_frozen_encoder_cannot_leave_eval_and_optimizer_has_no_encoder_group() -> None:
    config = _config()
    encoder = FrozenEncoder(SpatiotemporalEncoder())
    model = ParallelTrajectoryModel(_profile(), encoder)
    model.train()
    assert not model.encoder.training
    assert not model.encoder.module.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.congestion_predictor.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.congestion_tokenizer.parameters()
    )

    optimizer, contract = build_optimizer(model, config)
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "decoder_decay",
        "decoder_no_decay",
        "congestion_decay",
        "congestion_no_decay",
    ]
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert not (encoder_ids & optimizer_ids)
    assert contract.encoder_optimizer_parameter_count == 0
    assert contract.to_dict()["encoder_optimizer_group_present"] is False

    scheduler = PiecewiseLinearScheduler(optimizer, config)
    assert scheduler.ratio_at(1) == pytest.approx(1 / 460)
    assert scheduler.ratio_at(460) == 1.0
    assert scheduler.ratio_at(461) == 1.0
    assert scheduler.ratio_at(920) == 1.0
    assert scheduler.ratio_at(921) == pytest.approx(1.0 - 0.9 / 8280)
    assert scheduler.ratio_at(9200) == pytest.approx(0.1)


def test_access_ledger_records_and_blocks_before_any_test_reader(tmp_path: Path) -> None:
    ledger = DatasetAccessLedger(
        tmp_path / "access.json",
        partition_by_session={
            "train-id": "train",
            "validation-id": "validation",
            "test-id": "test",
        },
        test_session_ids=("test-id",),
        split_manifest_sha256="1" * 64,
        allowed_partitions=("train",),
        phase="unit-test",
    )
    ledger.request("train-id", operation="load")
    ledger.dataset_attempt(
        "train-id", operation="load", table_names=("match_samples.parquet",)
    )
    ledger.dataset_opened(
        "train-id",
        operation="load",
        table_names=("match_samples.parquet",),
        compatibility_recovery=None,
    )
    with pytest.raises(TrainingConfigurationError, match="denylisted"):
        ledger.request("test-id", operation="load")
    report = ledger.report()
    assert report["test_session_request_attempt_count"] == 1
    assert report["test_session_parquet_open_attempt_count"] == 0
    assert report["test_session_parquet_open_count"] == 0
    assert report["zero_test_attempts"] is False
    assert report["zero_test_opens"] is True
    assert report["no_decoder_test_split_constructed"] is True


def test_fncs_training_source_has_no_historical_downstream_transfer_path() -> None:
    imports: set[str] = set()
    called_names: set[str] = set()
    for path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)
    assert not any(name.startswith("fortnite_early_zone") for name in imports)
    assert not any(name.startswith("fortnite_parallel_trajectory_recovery") for name in imports)
    assert "load_transfer_bundle" not in called_names
    assert "apply_transfer" not in called_names
    assert "load_transfer_checkpoint" not in called_names


def test_checkpoint_source_only_serializes_downstream_state() -> None:
    source = (PACKAGE_ROOT / "checkpoint.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "downstream_state" in string_literals
    assert "model_state" not in string_literals
    assert "encoder_state" not in string_literals
