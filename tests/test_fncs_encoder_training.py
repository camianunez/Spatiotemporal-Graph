from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from fortnite_encoder.config import EncoderConfig, RotationHeadConfig, RotationLossConfig
from fortnite_encoder.rotation import RotationModel
from fortnite_encoder.training import LossMetricAccumulator

from fncs_encoder_training.artifacts import parameter_report
from fncs_encoder_training.common import atomic_torch_save, file_sha256
from fncs_encoder_training.config import load_config
from fncs_encoder_training.data_access import DataAccessLedger
from fncs_encoder_training.dataset import build_split_manifest, load_bound_split
from fncs_encoder_training.optimization import (
    PiecewiseLinearScheduler,
    ScheduleContract,
    build_adamw_optimizer,
)
from fncs_encoder_training.runtime import (
    TrainingState,
    _checkpoint_payload,
    _load_checkpoint,
)


def _workspace() -> Path:
    return Path(__file__).resolve().parents[2]


def test_canonical_fncs_config_resolves_fixed_contract() -> None:
    config = load_config(
        _workspace() / "configs" / "encoder-fncs-pilot-20260817-fresh-run-2.json"
    )
    assert (config.train_sessions, config.validation_sessions, config.test_sessions) == (
        920,
        115,
        115,
    )
    assert config.seed == 20260727
    assert config.batch_size == 4
    assert config.epochs == 20
    assert config.precision == "bf16"
    assert config.device == "cuda:0"
    assert config.regularization_loss_weight == 0.0
    assert math.isclose(
        config.scheduler_warmup_fraction
        + config.scheduler_constant_fraction
        + config.scheduler_decay_fraction,
        1.0,
    )


def test_new_loader_recomputes_piecewise_scheduler_boundaries() -> None:
    contract = ScheduleContract.derive(
        4600,
        warmup_fraction=0.05,
        constant_fraction=0.05,
        linear_decay_fraction=0.9,
        final_lr_ratio=0.1,
    )
    assert (contract.warmup_steps, contract.constant_steps, contract.linear_decay_steps) == (
        230,
        230,
        4140,
    )
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = PiecewiseLinearScheduler(optimizer, contract)
    assert math.isclose(scheduler.ratio_at(1), 1 / 230)
    assert scheduler.ratio_at(230) == 1.0
    assert scheduler.ratio_at(460) == 1.0
    assert scheduler.stage_at(461) == "linear_decay"
    assert scheduler.ratio_at(4600) == 0.1
    for _ in range(4600):
        scheduler.step()
    assert scheduler.step_index == 4600


class _GroupingFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.bias = nn.Parameter(torch.ones(2))
        self.normalization = nn.LayerNorm(2)
        self.relative_lag_bias = nn.Parameter(torch.ones(2, 2))


def test_adamw_groups_exclude_bias_normalization_and_relative_lag() -> None:
    model = _GroupingFixture()
    optimizer, contract = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    )
    groups = dict(contract.group_parameter_names)
    assert groups["decay"] == ("weight",)
    assert set(groups["no_decay"]) == {
        "bias",
        "normalization.bias",
        "normalization.weight",
        "relative_lag_bias",
    }
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_scheduler_reload_preserves_peak_lr_before_optimizer_state_restore() -> None:
    contract = ScheduleContract.derive(
        4600,
        warmup_fraction=0.05,
        constant_fraction=0.05,
        linear_decay_fraction=0.9,
        final_lr_ratio=0.1,
    )
    model = _GroupingFixture()
    optimizer, _ = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    )
    scheduler = PiecewiseLinearScheduler(optimizer, contract)
    optimizer.step()
    scheduler.step()

    restored_model = _GroupingFixture()
    restored_optimizer, _ = build_adamw_optimizer(
        restored_model,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    )
    restored_scheduler = PiecewiseLinearScheduler(restored_optimizer, contract)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())
    assert restored_scheduler.contract() == scheduler.contract()
    assert restored_scheduler.step_index == 1
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()


def test_split_is_exact_deterministic_and_self_bound(tmp_path: Path) -> None:
    sessions = [
        {
            "game_session_id": f"session-{index:04d}",
            "source_replay_sha256": f"{index:064x}",
            "data_status": "accepted_with_warnings",
            "provenance_status": "incomplete",
            "trust_partition": "unattested",
        }
        for index in range(1150)
    ]
    inventory = {
        "accepted_canonical_sessions": 1150,
        "ingestion_report_sha256": "1" * 64,
        "dataset_validation_sha256": "2" * 64,
        "sessions": sessions,
    }
    first = build_split_manifest(
        inventory,
        seed=20260727,
        train_count=920,
        validation_count=115,
        test_count=115,
    )
    second = build_split_manifest(
        inventory,
        seed=20260727,
        train_count=920,
        validation_count=115,
        test_count=115,
    )
    assert first["counts"] == {"train": 920, "validation": 115, "test": 115}
    assert first["split_manifest_sha256"] == second["split_manifest_sha256"]
    assert first["ordered_session_ids"] == second["ordered_session_ids"]
    path = tmp_path / "split.json"
    from fncs_encoder_training.dataset import write_split_manifest

    write_split_manifest(path, first)
    assert load_bound_split(path, first["split_manifest_sha256"])[
        "ordered_source_replay_sha256"
    ] == first["ordered_source_replay_sha256"]


def test_atomic_torch_checkpoint_is_readable_and_hashed(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(path, {"value": torch.arange(4)})
    assert len(file_sha256(path)) == 64
    assert torch.equal(
        torch.load(path, map_location="cpu", weights_only=True)["value"],
        torch.arange(4),
    )


def test_production_checkpoint_restores_partial_epoch_metrics(tmp_path: Path) -> None:
    encoder_config = EncoderConfig(
        d_model=16,
        n_heads=4,
        player_embedding_dim=4,
        edge_hidden_dim=8,
        ffn_dim=32,
        spatial_layers=1,
        temporal_layers=1,
        max_timesteps=16,
    )
    head_config = RotationHeadConfig(
        grid_height=2,
        grid_width=2,
        horizons_seconds=(5, 10, 15),
        horizon_embedding_dim=4,
        hidden_dim=8,
        entry_angle_bins=4,
    )
    loss_config = RotationLossConfig()
    torch.manual_seed(7)
    model = RotationModel(encoder_config=encoder_config, head_config=head_config)
    initial = parameter_report(model, seed=7)
    optimizer, optimizer_contract = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    )
    schedule_contract = ScheduleContract.derive(
        10,
        warmup_fraction=0.2,
        constant_fraction=0.2,
        linear_decay_fraction=0.6,
        final_lr_ratio=0.1,
    )
    scheduler = PiecewiseLinearScheduler(optimizer, schedule_contract)
    accumulator = LossMetricAccumulator(loss_config)
    metric_state = {
        "horizon_numerators": [10.0, 20.0, 30.0],
        "horizon_counts": [2, 4, 6],
        "entry_numerator": 12.0,
        "entry_count": 3,
        "survival_numerator": 15.0,
        "survival_count": 5,
        "placement_numerator": 14.0,
        "placement_count": 7,
    }
    accumulator.load_state_dict(metric_state)
    resolved = {
        "resolved_training_config_sha256": "a" * 64,
        "encoder_config": asdict(encoder_config),
        "head_config": asdict(head_config),
        "loss_config": asdict(loss_config),
        "bindings": {"fixture": "bound"},
        "sampler": {"seed": 7},
        "checkpoint_selection_metric": "future_position",
        "seed": 7,
    }
    ledger = DataAccessLedger(
        path=tmp_path / "ledger.json",
        partition_by_session={},
        test_session_ids=(),
        split_manifest_sha256="b" * 64,
        phase="test",
        allowed_partitions=("train",),
    )
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_bytes(b'{"type":"optimizer_step"}\n')
    state = TrainingState(current_epoch=0, next_batch_index=1)
    checkpoint_path = tmp_path / "last.pt"
    atomic_torch_save(
        checkpoint_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            optimizer_contract=optimizer_contract,
            scheduler=scheduler,
            state=state,
            epoch_accumulator=accumulator,
            resolved=resolved,
            initial_parameters=initial,
            metrics_path=metrics_path,
            ledger=ledger,
        ),
    )

    restored_model = RotationModel(
        encoder_config=encoder_config,
        head_config=head_config,
    )
    restored_optimizer, restored_contract = build_adamw_optimizer(
        restored_model,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    )
    restored_scheduler = PiecewiseLinearScheduler(
        restored_optimizer, schedule_contract
    )
    restored_accumulator = LossMetricAccumulator(loss_config)
    restored_state = _load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        optimizer_contract=restored_contract,
        scheduler=restored_scheduler,
        resolved=resolved,
        initial_parameters=initial,
        metrics_path=tmp_path / "restored-metrics.jsonl",
        epoch_accumulator=restored_accumulator,
    )
    assert restored_state.next_batch_index == 1
    assert restored_accumulator.state_dict() == metric_state
