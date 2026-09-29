from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_parallel_trajectory_locked.py"
)
SPEC = importlib.util.spec_from_file_location("locked_parallel_evaluator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
locked = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = locked
SPEC.loader.exec_module(locked)


def test_self_seal_round_trip_and_tamper_detection() -> None:
    payload = locked.self_seal({"value": 7}, "payload_sha256")
    locked.verify_self_seal(payload, "payload_sha256")
    payload["value"] = 8
    try:
        locked.verify_self_seal(payload, "payload_sha256")
    except locked.LockedEvaluationError:
        pass
    else:
        raise AssertionError("tampered self-sealed payload was accepted")


def test_route_error_values_are_route_equal_and_allow_mask_holes() -> None:
    prediction = np.zeros((2, 12, 2), dtype=np.float64)
    truth = np.zeros_like(prediction)
    truth[0, 0, 0] = 1.0
    truth[0, 2, 0] = 3.0
    truth[1, 0, 0] = 10.0
    mask = np.zeros((2, 12), dtype=np.bool_)
    mask[0, [0, 2]] = True
    mask[1, 0] = True
    _errors, ade, fde = locked._route_error_values(
        prediction, truth, mask, meters_per_world_unit=1.0
    )
    np.testing.assert_allclose(ade, [2.0, 10.0])
    np.testing.assert_allclose(fde, [3.0, 10.0])
    assert float(np.mean(ade)) == 6.0


def test_clustered_bootstrap_is_deterministic_and_paired() -> None:
    differences = np.asarray([-2.0, -4.0, 1.0, 3.0], dtype=np.float64)
    sessions = np.asarray(["a", "a", "b", "b"], dtype=str)
    first = locked.clustered_paired_bootstrap(
        differences, sessions, label="unit", resamples=200
    )
    second = locked.clustered_paired_bootstrap(
        differences, sessions, label="unit", resamples=200
    )
    assert first == second
    assert first["point_difference_meters"] == -0.5
    assert first["session_cluster_count"] == 2
    assert first["paired_route_count"] == 4


def test_standard_bivariate_log_density_at_origin() -> None:
    points = np.zeros((1, 1, 2), dtype=np.float64)
    means = np.zeros((1, 1, 2), dtype=np.float64)
    scales = np.ones((1, 1, 2), dtype=np.float64)
    correlations = np.zeros((1, 1), dtype=np.float64)
    actual = locked._component_log_density(points, means, scales, correlations)
    np.testing.assert_allclose(actual, [[-math.log(2.0 * math.pi)]])


def test_coherent_sampling_reuses_one_mode_for_complete_route() -> None:
    probabilities = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    means = np.zeros((5, 12, 2), dtype=np.float64)
    means[1] = 5.0
    scales = np.full_like(means, 1e-9)
    correlations = np.zeros((5, 12), dtype=np.float64)
    first = locked._sample_coherent_routes(
        np.random.default_rng(10), probabilities, means, scales, correlations, 8
    )
    second = locked._sample_coherent_routes(
        np.random.default_rng(10), probabilities, means, scales, correlations, 8
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, 5.0, atol=1e-8)


def test_static_route_coherence_detects_zero_motion_without_nonfinite_values() -> None:
    current = np.asarray([[50.0, 50.0], [60.0, 60.0]], dtype=np.float64)
    routes = np.repeat(current[:, None, :], 12, axis=1)
    profile = SimpleNamespace(
        world_x_min=0.0,
        world_x_max=100.0,
        world_y_min=0.0,
        world_y_max=100.0,
        world_unit_scale=SimpleNamespace(meters_per_world_unit=1.0),
    )
    setup = SimpleNamespace(profile=profile)
    report = locked.route_coherence_for_method(routes, current, setup)
    assert report["zero_motion_route_rate"] == 1.0
    assert report["nonfinite_output_rate"] == 0.0
    assert report["out_of_bounds_route_rate"] == 0.0
    assert report["near_identical_route_rate"] == 1.0


def test_mixture_and_calibration_reports_are_deterministic() -> None:
    q = 2
    probabilities = np.full((q, 5), 0.2, dtype=np.float64)
    means = np.zeros((q, 5, 12, 2), dtype=np.float64)
    for mode in range(5):
        means[:, mode, :, 0] = float(mode) * 2.0
    current = np.full((q, 2), 50.0, dtype=np.float64)
    mode_world = current[:, None, None, :] + means
    arrays = locked.PartitionArrays(
        partition="validation",
        query_manifest=tuple(
            {
                "session_id": f"s{row}",
                "team_id": row + 1,
                "absolute_tick_index": 10,
            }
            for row in range(q)
        ),
        session_ids=np.asarray(["s0", "s1"], dtype=str),
        phases=np.asarray([1, 2], dtype=np.int64),
        current_xy_uu=current,
        truth_xy_uu=np.repeat(current[:, None, :], 12, axis=1),
        target_displacements=np.zeros((q, 12, 2), dtype=np.float64),
        target_mask=np.ones((q, 12), dtype=np.bool_),
        primary_xy_uu=mode_world[:, 0],
        secondary_xy_uu=np.sum(probabilities[:, :, None, None] * mode_world, axis=1),
        static_xy_uu=np.repeat(current[:, None, :], 12, axis=1),
        constant_velocity_xy_uu=np.repeat(current[:, None, :], 12, axis=1),
        mode_xy_uu=mode_world,
        mode_logits=np.zeros((q, 5), dtype=np.float64),
        mode_probabilities=probabilities,
        means_normalized=means,
        scales_normalized=np.full((q, 5, 12, 2), 0.5, dtype=np.float64),
        correlations=np.zeros((q, 5, 12), dtype=np.float64),
        route_nll=np.ones(q, dtype=np.float64),
        constant_velocity_fallbacks=(None, None),
        causal_input_hashes=("0" * 64,),
    )
    profile = SimpleNamespace(
        cell_width_world_units=1.0,
        cell_height_world_units=1.0,
        world_x_min=0.0,
        world_x_max=100.0,
        world_y_min=0.0,
        world_y_max=100.0,
        world_unit_scale=SimpleNamespace(meters_per_world_unit=1.0),
    )
    setup = SimpleNamespace(profile=profile)
    accuracy_values = {
        "primary": (
            np.zeros((q, 12)),
            np.zeros(q),
            np.zeros(q),
        )
    }
    mixture = locked.mixture_report(arrays, setup, accuracy_values)
    assert len(mixture["selected_mode_counts"]) == 5
    assert math.isfinite(mixture["oracle_best_of_five_diagnostic_only"]["minade_meters"])
    first = locked.calibration_report(arrays, setup)
    second = locked.calibration_report(arrays, setup)
    assert first == second
    assert first["predictive_samples_per_set"] == 512
