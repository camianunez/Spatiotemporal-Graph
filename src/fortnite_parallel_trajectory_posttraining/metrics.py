from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .common import HORIZONS_SECONDS, require


BOOTSTRAP_SEED = 20260902
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
HIT_RADII_CELLS = (1.0, 2.0, 4.0)
AGGREGATE_HORIZONS_SECONDS = (15, 30, 45, 60)


@dataclass(frozen=True, slots=True)
class MetricContribution:
    numerator: np.ndarray
    denominator: np.ndarray

    def __post_init__(self) -> None:
        require(self.numerator.ndim == 1, "metric numerators must be one-dimensional")
        require(self.denominator.shape == self.numerator.shape, "metric contribution shapes differ")
        require(bool(np.isfinite(self.numerator).all()), "metric numerators are nonfinite")
        require(bool(np.isfinite(self.denominator).all()), "metric denominators are nonfinite")
        require(bool((self.denominator >= 0).all()), "metric denominators are negative")

    @property
    def estimate(self) -> float | None:
        denominator = float(self.denominator.sum())
        return float(self.numerator.sum() / denominator) if denominator > 0 else None


def _errors(
    prediction_xy_uu: np.ndarray,
    truth_xy_uu: np.ndarray,
    target_mask: np.ndarray,
    *,
    meters_per_world_unit: float,
) -> np.ndarray:
    require(prediction_xy_uu.shape == truth_xy_uu.shape, "prediction/truth shapes differ")
    require(target_mask.shape == prediction_xy_uu.shape[:2], "target-mask shape differs")
    result = np.linalg.norm(prediction_xy_uu - truth_xy_uu, axis=-1)
    result = result.astype(np.float64, copy=False) * meters_per_world_unit
    result[~target_mask] = np.nan
    return result


def query_ade(errors_meters: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    q = target_mask.shape[0]
    result = np.full(q, np.nan, dtype=np.float64)
    for row in range(q):
        valid = target_mask[row]
        if valid.any():
            result[row] = float(errors_meters[row, valid].mean())
    return result


def query_last_valid_fde(errors_meters: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    q = target_mask.shape[0]
    result = np.full(q, np.nan, dtype=np.float64)
    for row in range(q):
        valid = np.flatnonzero(target_mask[row])
        if valid.size:
            result[row] = float(errors_meters[row, valid[-1]])
    return result


def _session_contribution(
    session_ids: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MetricContribution:
    sessions = tuple(sorted(set(session_ids.tolist())))
    numerator: list[float] = []
    denominator: list[float] = []
    for session in sessions:
        selected = (session_ids == session) & valid
        observed = values[selected]
        if transform is not None:
            observed = transform(observed)
        numerator.append(float(observed.sum()) if observed.size else 0.0)
        denominator.append(float(observed.size))
    return MetricContribution(
        numerator=np.asarray(numerator, dtype=np.float64),
        denominator=np.asarray(denominator, dtype=np.float64),
    )


def point_metric_contributions(
    *,
    prediction_xy_uu: np.ndarray,
    truth_xy_uu: np.ndarray,
    target_mask: np.ndarray,
    session_ids: np.ndarray,
    meters_per_world_unit: float,
    cell_width_world_units: float,
    cell_height_world_units: float,
) -> tuple[dict[str, MetricContribution], np.ndarray, np.ndarray]:
    errors_meters = _errors(
        prediction_xy_uu,
        truth_xy_uu,
        target_mask,
        meters_per_world_unit=meters_per_world_unit,
    )
    ade = query_ade(errors_meters, target_mask)
    last_fde = query_last_valid_fde(errors_meters, target_mask)
    contributions: dict[str, MetricContribution] = {
        "ade_meters": _session_contribution(
            session_ids, ade, np.isfinite(ade)
        ),
        "last_valid_fde_meters": _session_contribution(
            session_ids, last_fde, np.isfinite(last_fde)
        ),
        "fde_60_meters": _session_contribution(
            session_ids,
            errors_meters[:, -1],
            target_mask[:, -1] & np.isfinite(errors_meters[:, -1]),
        ),
    }
    delta = prediction_xy_uu - truth_xy_uu
    cell_distance = np.sqrt(
        (delta[..., 0] / cell_width_world_units) ** 2
        + (delta[..., 1] / cell_height_world_units) ** 2
    )
    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        valid = target_mask[:, horizon_index] & np.isfinite(errors_meters[:, horizon_index])
        contributions[f"error_at_{horizon}s_meters"] = _session_contribution(
            session_ids, errors_meters[:, horizon_index], valid
        )
        for radius in HIT_RADII_CELLS:
            contributions[f"hit_at_{horizon}s_within_{radius:g}_cells"] = (
                _session_contribution(
                    session_ids,
                    cell_distance[:, horizon_index],
                    valid,
                    transform=lambda value, threshold=radius: (
                        value <= threshold
                    ).astype(np.float64),
                )
            )
    for endpoint in AGGREGATE_HORIZONS_SECONDS:
        eligible_horizons = np.asarray(HORIZONS_SECONDS) <= endpoint
        repeated_sessions = np.repeat(session_ids, int(eligible_horizons.sum()))
        valid = target_mask[:, eligible_horizons].reshape(-1)
        values = cell_distance[:, eligible_horizons].reshape(-1)
        for radius in HIT_RADII_CELLS:
            contributions[f"aggregate_through_{endpoint}s_hit_within_{radius:g}_cells"] = (
                _session_contribution(
                    repeated_sessions,
                    values,
                    valid & np.isfinite(values),
                    transform=lambda value, threshold=radius: (
                        value <= threshold
                    ).astype(np.float64),
                )
            )
    return contributions, errors_meters, cell_distance


def point_metrics(
    *,
    prediction_xy_uu: np.ndarray,
    truth_xy_uu: np.ndarray,
    target_mask: np.ndarray,
    session_ids: np.ndarray,
    meters_per_world_unit: float,
    cell_width_world_units: float,
    cell_height_world_units: float,
) -> tuple[dict[str, Any], dict[str, MetricContribution]]:
    contributions, errors_meters, _ = point_metric_contributions(
        prediction_xy_uu=prediction_xy_uu,
        truth_xy_uu=truth_xy_uu,
        target_mask=target_mask,
        session_ids=session_ids,
        meters_per_world_unit=meters_per_world_unit,
        cell_width_world_units=cell_width_world_units,
        cell_height_world_units=cell_height_world_units,
    )
    per_horizon: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        per_horizon[str(horizon)] = {
            "valid_target_count": int(target_mask[:, horizon_index].sum()),
            "mean_euclidean_error_meters": contributions[
                f"error_at_{horizon}s_meters"
            ].estimate,
            "trajectory_hit_rate": {
                f"within_{radius:g}_cells": contributions[
                    f"hit_at_{horizon}s_within_{radius:g}_cells"
                ].estimate
                for radius in HIT_RADII_CELLS
            },
        }
    aggregate_hits: dict[str, Any] = {}
    for endpoint in AGGREGATE_HORIZONS_SECONDS:
        aggregate_hits[str(endpoint)] = {
            f"within_{radius:g}_cells": contributions[
                f"aggregate_through_{endpoint}s_hit_within_{radius:g}_cells"
            ].estimate
            for radius in HIT_RADII_CELLS
        }
    result = {
        "ade_meters": contributions["ade_meters"].estimate,
        "fde_60_meters": contributions["fde_60_meters"].estimate,
        "last_valid_horizon_fde_meters": contributions[
            "last_valid_fde_meters"
        ].estimate,
        "per_horizon": per_horizon,
        "aggregate_trajectory_hit_rate_through_horizon": aggregate_hits,
        "valid_query_count": int(target_mask.any(axis=1).sum()),
        "valid_target_count": int(target_mask.sum()),
        "nonfinite_scored_error_count": int(
            np.count_nonzero(target_mask & ~np.isfinite(errors_meters))
        ),
    }
    return result, contributions


def scalar_contribution(
    *, session_ids: np.ndarray, values: np.ndarray, valid: np.ndarray | None = None
) -> MetricContribution:
    resolved = np.isfinite(values) if valid is None else valid & np.isfinite(values)
    return _session_contribution(session_ids, values, resolved)


def bootstrap_estimate(
    contribution: MetricContribution,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    n = contribution.numerator.size
    require(n > 1, "session bootstrap requires at least two sessions")
    estimate = contribution.estimate
    require(estimate is not None, "metric is unavailable")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, n, size=(resamples, n), endpoint=False)
    denominators = contribution.denominator[indices].sum(axis=1)
    valid = denominators > 0
    require(bool(valid.any()), "all session-bootstrap replicates lack observations")
    values = contribution.numerator[indices].sum(axis=1)[valid] / denominators[valid]
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    lower, upper = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
    )
    return {
        "estimate": estimate,
        "95_percent_ci": [float(lower), float(upper)],
        "bootstrap": {
            "unit": "session",
            "seed": seed,
            "resamples_requested": resamples,
            "resamples_valid": int(valid.sum()),
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile",
            "session_count": n,
        },
    }


def bootstrap_paired_difference(
    left: MetricContribution,
    right: MetricContribution,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    require(left.numerator.shape == right.numerator.shape, "paired bootstrap shapes differ")
    n = left.numerator.size
    require(n > 1, "paired bootstrap requires at least two sessions")
    left_estimate = left.estimate
    right_estimate = right.estimate
    require(left_estimate is not None and right_estimate is not None, "metric is unavailable")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, n, size=(resamples, n), endpoint=False)
    left_den = left.denominator[indices].sum(axis=1)
    right_den = right.denominator[indices].sum(axis=1)
    valid = (left_den > 0) & (right_den > 0)
    require(bool(valid.any()), "all paired-bootstrap replicates lack observations")
    left_values = left.numerator[indices].sum(axis=1)[valid] / left_den[valid]
    right_values = right.numerator[indices].sum(axis=1)[valid] / right_den[valid]
    differences = left_values - right_values
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    lower, upper = np.quantile(
        differences, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
    )
    percentage = (left_values / right_values - 1.0) * 100.0
    percentage = percentage[np.isfinite(percentage)]
    percent_ci = (
        np.quantile(percentage, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
        if percentage.size
        else (np.nan, np.nan)
    )
    return {
        "left": left_estimate,
        "right": right_estimate,
        "left_minus_right": left_estimate - right_estimate,
        "left_minus_right_percent_of_right": (
            (left_estimate / right_estimate - 1.0) * 100.0
            if right_estimate != 0.0
            else None
        ),
        "difference_95_percent_ci": [float(lower), float(upper)],
        "percentage_change_95_percent_ci": [
            None if not math.isfinite(float(percent_ci[0])) else float(percent_ci[0]),
            None if not math.isfinite(float(percent_ci[1])) else float(percent_ci[1]),
        ],
        "bootstrap": {
            "unit": "session",
            "paired": True,
            "seed": seed,
            "resamples_requested": resamples,
            "resamples_valid": int(valid.sum()),
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile",
            "session_count": n,
        },
    }


def decoder_mode_diagnostics(
    *,
    mode_xy_uu: np.ndarray,
    mode_probabilities: np.ndarray,
    truth_xy_uu: np.ndarray,
    target_mask: np.ndarray,
    meters_per_world_unit: float,
    cell_width_world_units: float,
    cell_height_world_units: float,
) -> dict[str, Any]:
    q, k, h, coordinates = mode_xy_uu.shape
    require((k, h, coordinates) == (5, 12, 2), "decoder mode output shape changed")
    require(mode_probabilities.shape == (q, 5), "mode probability shape changed")
    delta = mode_xy_uu - truth_xy_uu[:, None, :, :]
    distance_m = np.linalg.norm(delta, axis=-1) * meters_per_world_unit
    minade = np.full(q, np.nan, dtype=np.float64)
    minfde = np.full(q, np.nan, dtype=np.float64)
    for row in range(q):
        valid = np.flatnonzero(target_mask[row])
        if not valid.size:
            continue
        minade[row] = float(distance_m[row, :, valid].mean(axis=1).min())
        minfde[row] = float(distance_m[row, :, valid[-1]].min())
    safe_probabilities = np.clip(mode_probabilities, 1e-12, 1.0)
    entropy = -(safe_probabilities * np.log(safe_probabilities)).sum(axis=1)
    effective = np.exp(entropy)
    argmax = np.argmax(mode_probabilities, axis=1)
    argmax_counts = np.bincount(argmax, minlength=5)
    pairwise_cell_distances: list[np.ndarray] = []
    for left in range(5):
        for right in range(left + 1, 5):
            pair_delta = mode_xy_uu[:, left] - mode_xy_uu[:, right]
            pairwise_cell_distances.append(
                np.sqrt(
                    (pair_delta[..., 0] / cell_width_world_units) ** 2
                    + (pair_delta[..., 1] / cell_height_world_units) ** 2
                )
            )
    separation = np.stack(pairwise_cell_distances, axis=1)
    valid_expanded = target_mask[:, None, :]
    per_query_separation = np.nansum(
        np.where(valid_expanded, separation, np.nan), axis=(1, 2)
    ) / np.maximum(valid_expanded.sum(axis=(1, 2)) * separation.shape[1], 1)
    mean_probability = mode_probabilities.mean(axis=0)
    global_effective = float(
        np.exp(-(mean_probability * np.log(np.clip(mean_probability, 1e-12, 1.0))).sum())
    )
    nonfinite_by_field = {
        "mode_coordinates": int(np.count_nonzero(~np.isfinite(mode_xy_uu))),
        "mode_probabilities": int(np.count_nonzero(~np.isfinite(mode_probabilities))),
    }
    return {
        "oracle_diagnostic_only": {
            "minADE_at_5_meters": float(np.nanmean(minade)),
            "minFDE_at_5_meters": float(np.nanmean(minfde)),
            "query_count": int(np.isfinite(minade).sum()),
        },
        "mode_probability_entropy_nats": {
            "mean": float(entropy.mean()),
            "minimum": float(entropy.min()),
            "maximum": float(entropy.max()),
        },
        "effective_number_of_used_modes": {
            "mean_per_query_exp_entropy": float(effective.mean()),
            "minimum_per_query_exp_entropy": float(effective.min()),
            "global_exp_entropy_of_mean_probabilities": global_effective,
        },
        "mean_mode_probabilities": mean_probability.tolist(),
        "argmax_mode_counts": argmax_counts.tolist(),
        "mode_collapse_counts": {
            "queries_with_max_probability_at_least_0_98": int(
                (mode_probabilities.max(axis=1) >= 0.98).sum()
            ),
            "queries_with_mean_pairwise_separation_below_0_25_cell": int(
                (per_query_separation < 0.25).sum()
            ),
            "modes_never_selected_as_top_mode": int((argmax_counts == 0).sum()),
        },
        "mean_pairwise_mode_separation_cells": float(per_query_separation.mean()),
        "nonfinite_prediction_counts": nonfinite_by_field,
    }


__all__ = [
    "AGGREGATE_HORIZONS_SECONDS",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "HIT_RADII_CELLS",
    "MetricContribution",
    "bootstrap_estimate",
    "bootstrap_paired_difference",
    "decoder_mode_diagnostics",
    "point_metric_contributions",
    "point_metrics",
    "query_ade",
    "query_last_valid_fde",
    "scalar_contribution",
]
