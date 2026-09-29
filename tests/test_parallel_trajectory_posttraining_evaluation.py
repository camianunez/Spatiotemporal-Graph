from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fortnite_parallel_trajectory.losses import route_mixture_nll
from fortnite_parallel_trajectory_fncs_training.data import DatasetAccessLedger
from fortnite_parallel_trajectory_posttraining import evaluate as locked
from fortnite_parallel_trajectory_posttraining.common import PostTrainingError
from fortnite_parallel_trajectory_posttraining.metrics import (
    MetricContribution,
    bootstrap_estimate,
    bootstrap_paired_difference,
    point_metrics,
)
from fortnite_parallel_trajectory_posttraining.report import (
    _correct_oracle_best_of_five,
)


def _synthetic_planner_batch() -> SimpleNamespace:
    batch_size, time_steps, teams, players = 2, 3, 1, 2
    query_mask = torch.zeros((batch_size, time_steps, teams), dtype=torch.bool)
    query_mask[:, 2, 0] = True
    xyz = torch.zeros((batch_size, time_steps, teams, players, 3))
    # First route has centroids 1, 3, 5 world units at 0, 5, 10 seconds.
    xyz[0, 0, 0, :, 0] = torch.tensor([0.0, 2.0])
    xyz[0, 1, 0, :, 0] = torch.tensor([2.0, 4.0])
    xyz[0, 2, 0, :, 0] = torch.tensor([4.0, 6.0])
    # Second route has a query-time centroid of 10, but a lifecycle break at t=1.
    xyz[1, :, 0, :, 0] = torch.tensor(
        [[6.0, 8.0], [7.0, 9.0], [9.0, 11.0]]
    )
    life_index = torch.zeros(
        (batch_size, time_steps, teams, players), dtype=torch.int64
    )
    life_index[1, 2, 0] = 1
    encoder = SimpleNamespace(
        player_xyz_uu=xyz,
        player_slot_mask=torch.ones((batch_size, teams, players), dtype=torch.bool),
        player_alive=torch.ones(
            (batch_size, time_steps, teams, players), dtype=torch.bool
        ),
        player_coord_mask=torch.ones(
            (batch_size, time_steps, teams, players), dtype=torch.bool
        ),
        life_index=life_index,
        time_mask=torch.ones((batch_size, time_steps), dtype=torch.bool),
        match_elapsed_s=torch.tensor(
            [[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]]
        ),
    )
    return SimpleNamespace(query_mask=query_mask, encoder_batch=encoder)


def test_static_and_constant_velocity_forecasts_are_causal() -> None:
    current, static, velocity, fallbacks = locked.derive_causal_baselines(
        _synthetic_planner_batch()
    )
    np.testing.assert_allclose(current.numpy(), [[5.0, 0.0], [10.0, 0.0]])
    np.testing.assert_allclose(
        static.numpy(),
        np.repeat(np.asarray([[[5.0, 0.0]], [[10.0, 0.0]]]), 12, axis=1),
    )
    horizons = np.arange(5.0, 61.0, 5.0)
    np.testing.assert_allclose(velocity[0, :, 0].numpy(), 5.0 + 0.4 * horizons)
    np.testing.assert_allclose(velocity[0, :, 1].numpy(), 0.0)
    np.testing.assert_allclose(velocity[1].numpy(), static[1].numpy())
    assert fallbacks == (None, "second_comparable_causal_observation_unavailable")


def test_query_and_horizon_masks_control_route_nll_reduction() -> None:
    q, modes, horizons = 3, 5, 12
    logits = torch.zeros((q, modes), dtype=torch.float32)
    means = torch.zeros((q, modes, horizons, 2), dtype=torch.float32)
    scales = torch.ones_like(means)
    correlations = torch.zeros((q, modes, horizons), dtype=torch.float32)
    targets = torch.zeros((q, horizons, 2), dtype=torch.float32)
    masks = torch.zeros((q, horizons), dtype=torch.bool)
    masks[0, 0] = True
    masks[1, :2] = True
    masks[2, 0] = True
    targets[0, 1] = 1000.0  # Masked horizon must not affect the likelihood.
    targets[2, 0] = 1000.0  # Masked query must not affect the reduction.
    query_mask = torch.tensor([True, True, False])

    result = route_mixture_nll(
        logits,
        means,
        scales,
        correlations,
        targets,
        masks,
        query_mask,
    )

    unit_nll = math.log(2.0 * math.pi)
    assert result.valid_route_count == 2
    assert result.valid_horizon_count == 3
    assert result.loss.item() == pytest.approx(1.5 * unit_nll, abs=1e-6)
    assert result.per_route_nll[0].item() == pytest.approx(unit_nll, abs=1e-6)
    assert result.per_route_nll[1].item() == pytest.approx(2.0 * unit_nll, abs=1e-6)
    assert result.per_route_nll[2].item() == 0.0


def test_ade_fde_and_trajectory_hit_rates_respect_horizon_masks() -> None:
    prediction = np.zeros((2, 12, 2), dtype=np.float64)
    truth = np.zeros_like(prediction)
    truth[0, 0, 0] = 1.0
    truth[0, 1] = np.nan
    truth[0, 2, 0] = 3.0
    truth[0, 11, 0] = 12.0
    truth[1, 0, 0] = 10.0
    mask = np.zeros((2, 12), dtype=np.bool_)
    mask[0, [0, 2, 11]] = True
    mask[1, 0] = True

    metrics, _ = point_metrics(
        prediction_xy_uu=prediction,
        truth_xy_uu=truth,
        target_mask=mask,
        session_ids=np.asarray(["session-a", "session-b"]),
        meters_per_world_unit=1.0,
        cell_width_world_units=1.0,
        cell_height_world_units=1.0,
    )

    assert metrics["valid_query_count"] == 2
    assert metrics["valid_target_count"] == 4
    assert metrics["nonfinite_scored_error_count"] == 0
    assert metrics["ade_meters"] == pytest.approx(23.0 / 3.0)
    assert metrics["last_valid_horizon_fde_meters"] == pytest.approx(11.0)
    assert metrics["fde_60_meters"] == pytest.approx(12.0)
    at_five = metrics["per_horizon"]["5"]
    assert at_five["valid_target_count"] == 2
    assert at_five["mean_euclidean_error_meters"] == pytest.approx(5.5)
    assert at_five["trajectory_hit_rate"]["within_1_cells"] == pytest.approx(0.5)
    assert metrics["per_horizon"]["10"]["valid_target_count"] == 0
    assert metrics["per_horizon"]["15"]["trajectory_hit_rate"]["within_4_cells"] == 1.0


def test_minade_at_five_is_reduced_over_modes_not_horizons() -> None:
    modes = np.zeros((1, 5, 12, 2), dtype=np.float64)
    truth = np.zeros((1, 12, 2), dtype=np.float64)
    mask = np.zeros((1, 12), dtype=np.bool_)
    mask[0, :2] = True
    modes[0, 0, 0, 0] = 1.0
    modes[0, 0, 1, 0] = 101.0
    modes[0, 1:, :2, 0] = 10.0

    result = _correct_oracle_best_of_five(
        mode_xy_uu=modes,
        truth_xy_uu=truth,
        target_mask=mask,
        meters_per_world_unit=1.0,
    )

    assert result["minADE@5_meters"] == 10.0
    assert result["minFDE@5_meters"] == 10.0


def test_session_bootstrap_is_deterministic_and_paired() -> None:
    left = MetricContribution(
        numerator=np.asarray([2.0, 8.0, 3.0]),
        denominator=np.asarray([1.0, 2.0, 1.0]),
    )
    right = MetricContribution(
        numerator=np.asarray([4.0, 12.0, 5.0]),
        denominator=np.asarray([1.0, 2.0, 1.0]),
    )
    first = bootstrap_estimate(left, seed=17, resamples=500)
    second = bootstrap_estimate(left, seed=17, resamples=500)
    assert first == second
    paired_first = bootstrap_paired_difference(left, right, seed=17, resamples=500)
    paired_second = bootstrap_paired_difference(left, right, seed=17, resamples=500)
    assert paired_first == paired_second
    assert paired_first["bootstrap"]["unit"] == "session"
    assert paired_first["bootstrap"]["paired"] is True
    assert paired_first["left_minus_right"] == pytest.approx(-2.0)


def test_strict_completed_checkpoint_loading_and_step_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    model = DummyModel()
    state = {"decoder.weight": torch.tensor([1.0])}
    checkpoint = {
        "downstream_state": state,
        "downstream_state_sha256": "synthetic-state-hash",
        "training_state": {"optimizer_step": 9200},
        "frozen_encoder_snapshot": {
            "encoder_state_sha256": locked.EXPECTED_ENCODER_STATE_SHA256,
            "matches_canonical": True,
        },
        "test_partition_evaluated": False,
        "data_access": {"zero_test_attempts": True, "zero_test_opens": True},
    }
    loaded: list[dict[str, torch.Tensor]] = []
    monkeypatch.setattr(locked, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(
        locked,
        "build_frozen_model",
        lambda _setup, *, device: (
            model.to(device),
            {
                "initial_downstream_parameter_sha256": locked.EXPECTED_INITIAL_DOWNSTREAM_SHA256
            },
        ),
    )
    monkeypatch.setattr(locked, "tensor_state_sha256", lambda _state: "synthetic-state-hash")
    monkeypatch.setattr(locked, "load_downstream_state", lambda _model, value: loaded.append(value))
    monkeypatch.setattr(locked, "assert_frozen_encoder", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    result = locked._load_completed_model(
        checkpoint_path=tmp_path / "best.pt",
        setup=object(),
        device=torch.device("cpu"),
        expected_step=9200,
    )
    assert result is model
    assert result.training is False
    assert loaded == [state]

    checkpoint["training_state"]["optimizer_step"] = 9199
    with pytest.raises(PostTrainingError, match="optimizer step changed"):
        locked._load_completed_model(
            checkpoint_path=tmp_path / "best.pt",
            setup=object(),
            device=torch.device("cpu"),
            expected_step=9200,
        )


def test_test_partition_request_is_denied_before_reader(tmp_path) -> None:
    ledger = DatasetAccessLedger(
        tmp_path / "access.json",
        partition_by_session={"validation-session": "validation", "test-session": "test"},
        test_session_ids=("test-session",),
        split_manifest_sha256="0" * 64,
        allowed_partitions=("validation",),
        phase="synthetic_locked_evaluation",
    )
    with pytest.raises(Exception, match="denylisted encoder-test session request blocked"):
        ledger.request("test-session", operation="synthetic_probe")
    report = ledger.report()
    assert report["test_session_request_attempt_count"] == 1
    assert report["test_session_parquet_open_attempt_count"] == 0
    assert report["test_session_parquet_open_count"] == 0
    assert report["events"][-1]["outcome"] == "blocked_before_reader"
