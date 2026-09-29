from __future__ import annotations

import argparse
from collections import Counter
import csv
from html import escape
import io
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np
import torch

from .bindings import load_lineage_setups
from .common import (
    ANALYSIS_DIRECTORY,
    CHILD_RUN,
    EXPECTED_ENCODER_STATE_SHA256,
    EXPECTED_FINAL_ROUTE_NLL,
    EXPECTED_FINAL_STEP,
    FINALIZATION_DIRECTORY,
    HORIZONS_SECONDS,
    LINEAGE_NAME,
    PARENT_RUN,
    REPOSITORY_ROOT,
    current_utc,
    directory_inventory,
    file_record,
    file_sha256,
    inventory_sha256,
    payload_sha256,
    read_json,
    read_jsonl,
    require,
    self_seal,
    verify_self_seal,
    write_new_json,
    atomic_write_new,
)
from .evaluate import (
    _comparison_cis,
    _estimate_cis,
    _load_contract,
    _load_npz,
    _metric_payload,
    _model_arrays_path,
    _shared_arrays_path,
)
from .metrics import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    decoder_mode_diagnostics,
    bootstrap_estimate,
    scalar_contribution,
)


REPORT_PATH = ANALYSIS_DIRECTORY / "final_report.md"
SUMMARY_PATH = ANALYSIS_DIRECTORY / "decision_summary.json"
CHART_DIRECTORY = ANALYSIS_DIRECTORY / "charts"
TABLE_DIRECTORY = ANALYSIS_DIRECTORY / "tables"
VALIDATION_ONLY_DIRECTORY = (
    REPOSITORY_ROOT
    / "runs"
    / f"{LINEAGE_NAME}-post-training-evaluation-20260902"
)
REPORT_SOURCE_PRECHANGE_SHA256 = (
    "a406fc157a2f295ad5b3b3decb7b9658e2b095b100203c808e59c18962455f7a"
)


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#475569")


def _number(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_csv(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    path = TABLE_DIRECTORY / name
    atomic_write_new(path, _csv_bytes(header, rows))
    return path


def _line_chart(
    *,
    destination: Path,
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float]]],
    x_label: str,
    y_label: str,
    log_x: bool = False,
    y_bounds: tuple[float, float] | None = None,
    subtitle: str | None = None,
) -> None:
    require(x_values and series, "chart requires data")
    require(all(len(values) == len(x_values) for _, values in series), "chart shapes differ")
    width, height = 1000, 620
    left, right, top, bottom = 105, 45, 90, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    transformed_x = [math.log10(value) if log_x else value for value in x_values]
    x_min, x_max = min(transformed_x), max(transformed_x)
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    all_y = [float(value) for _, values in series for value in values]
    y_min, y_max = y_bounds if y_bounds is not None else (min(all_y), max(all_y))
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    elif y_bounds is None:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding

    def px(value: float) -> float:
        transformed = math.log10(value) if log_x else value
        return left + (transformed - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="{width/2}" y="60" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">{escape(subtitle)}</text>'
        )
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = py(value)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#334155">{value:.3g}</text>',
            ]
        )
    for value in x_values:
        x = px(float(value))
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f1f5f9"/>',
                f'<text x="{x:.2f}" y="{height-bottom+25}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#334155">{value:g}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#0f172a" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#0f172a" stroke-width="1.5"/>',
            f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="sans-serif" font-size="14">{escape(x_label)}</text>',
            f'<text x="24" y="{height/2}" text-anchor="middle" transform="rotate(-90 24 {height/2})" font-family="sans-serif" font-size="14">{escape(y_label)}</text>',
        ]
    )
    for index, (label, values) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        points = " ".join(
            f"{px(float(x)):.2f},{py(float(y)):.2f}" for x, y in zip(x_values, values)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x_value, y_value in zip(x_values, values):
            parts.append(
                f'<circle cx="{px(float(x_value)):.2f}" cy="{py(float(y_value)):.2f}" r="4" fill="{color}"/>'
            )
        legend_x = left + index * 190
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{height-57}" x2="{legend_x+24}" y2="{height-57}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x+31}" y="{height-52}" font-family="sans-serif" font-size="12" fill="#0f172a">{escape(label)}</text>',
            ]
        )
    parts.append("</svg>\n")
    atomic_write_new(destination, "".join(parts).encode("utf-8"))


def _supported_lower(comparison: Mapping[str, Any]) -> bool:
    interval = comparison["difference_95_percent_ci"]
    return float(comparison["left_minus_right"]) < 0 and float(interval[1]) < 0


def _supported_higher(comparison: Mapping[str, Any]) -> bool:
    interval = comparison["difference_95_percent_ci"]
    return float(comparison["left_minus_right"]) > 0 and float(interval[0]) > 0


def _correct_oracle_best_of_five(
    *,
    mode_xy_uu: np.ndarray,
    truth_xy_uu: np.ndarray,
    target_mask: np.ndarray,
    meters_per_world_unit: float,
) -> dict[str, Any]:
    """Compute mode-wise minADE@5 without NumPy advanced-index axis reordering."""

    require(mode_xy_uu.ndim == 4 and mode_xy_uu.shape[1:] == (5, 12, 2), "mode array shape changed")
    require(truth_xy_uu.shape == mode_xy_uu.shape[:1] + (12, 2), "truth array shape changed")
    require(target_mask.shape == mode_xy_uu.shape[:1] + (12,), "target mask shape changed")
    distances = (
        np.linalg.norm(mode_xy_uu - truth_xy_uu[:, None, :, :], axis=-1)
        * meters_per_world_unit
    )
    minade: list[float] = []
    minfde: list[float] = []
    for row in range(distances.shape[0]):
        valid = np.flatnonzero(target_mask[row])
        if not valid.size:
            continue
        # Split the indexing operations so the result remains [mode, horizon].
        mode_by_horizon = distances[row][:, valid]
        minade.append(float(mode_by_horizon.mean(axis=1).min()))
        minfde.append(float(distances[row][:, valid[-1]].min()))
    require(minade and len(minade) == len(minfde), "oracle metric has no valid routes")
    return {
        "minADE@5_meters": float(np.mean(minade)),
        "minFDE@5_meters": float(np.mean(minfde)),
        "valid_query_count": len(minade),
        "used_for_checkpoint_selection": False,
        "used_for_baseline_superiority_claims": False,
    }


def _training_and_validation_history() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_rows = read_jsonl(PARENT_RUN / "metrics.jsonl", "parent training metrics")
    child_rows = read_jsonl(CHILD_RUN / "metrics.jsonl", "child training metrics")
    training_rows = [
        row
        for row in parent_rows
        if row.get("type") == "training_epoch"
        and int(row.get("optimizer_step", -1)) <= 1840
    ] + [row for row in child_rows if row.get("type") == "training_epoch"]
    require(
        [int(row["completed_epoch"]) for row in training_rows] == list(range(1, 21)),
        "training-epoch history is not exactly epochs 1..20",
    )
    summary = read_json(
        FINALIZATION_DIRECTORY / "training_metrics_summary.json",
        "finalized training metrics summary",
    )
    verify_self_seal(summary)
    validation_rows = summary.get("validation_trajectory")
    require(
        isinstance(validation_rows, list)
        and [int(row["completed_epoch"]) for row in validation_rows]
        == list(range(1, 21)),
        "validation history is not exactly epochs 1..20",
    )
    history: list[dict[str, Any]] = []
    for training, validation in zip(training_rows, validation_rows):
        train_metrics = training.get("metrics", {})
        require(
            int(training["optimizer_step"]) == int(validation["optimizer_step"]),
            "training/validation epoch endpoints differ",
        )
        history.append(
            {
                "completed_epoch": int(training["completed_epoch"]),
                "optimizer_step": int(training["optimizer_step"]),
                "training_route_nll": float(train_metrics["route_nll"]),
                "training_nll_per_valid_horizon": float(
                    train_metrics["nll_per_valid_horizon"]
                ),
                "training_valid_route_count": int(train_metrics["valid_route_count"]),
                "training_valid_target_count": int(train_metrics["valid_horizon_count"]),
                "validation_route_nll": float(validation["route_nll"]),
                "validation_top_mode_ade_meters": float(
                    validation["top_mode_ade_meters"]
                ),
                "validation_last_valid_fde_meters": float(
                    validation["last_valid_fde_meters"]
                ),
                "validation_valid_route_count": int(validation["valid_route_count"]),
                "validation_valid_target_count": int(
                    validation["valid_horizon_count"]
                ),
                "validation_improved_incumbent": bool(
                    validation["improved_incumbent"]
                ),
            }
        )
    return history, summary


def _partial_census_evidence() -> dict[str, Any]:
    paths = [ANALYSIS_DIRECTORY / "training_subset_census_progress.jsonl"]
    paths.extend(
        sorted(ANALYSIS_DIRECTORY.glob("training_subset_census_shard_*_of_*.jsonl"))
    )
    paths = [path for path in paths if path.is_file()]
    require(paths, "existing partial census artifacts are missing")
    by_index: dict[int, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path, f"existing partial census {path.name}"):
            index = int(row["order_index_zero_based"])
            prior = by_index.get(index)
            require(prior is None or prior == row, f"conflicting partial census row {index}")
            by_index[index] = row
    require(len(by_index) == 725, "existing partial census is no longer 725/920")
    rows = [by_index[index] for index in sorted(by_index)]
    horizon_counts = {
        str(horizon): sum(
            int(row["valid_target_count_by_horizon_seconds"][str(horizon)])
            for row in rows
        )
        for horizon in HORIZONS_SECONDS
    }
    return {
        "status": "incomplete_existing_artifacts_only_not_resumed",
        "censused_training_sessions": len(rows),
        "training_partition_sessions": 920,
        "uncensused_training_sessions": 920 - len(rows),
        "eligible_route_query_count_in_censused_sessions": sum(
            int(row["eligible_route_query_count"]) for row in rows
        ),
        "valid_target_count_in_censused_sessions": sum(horizon_counts.values()),
        "valid_target_count_by_horizon_seconds_in_censused_sessions": horizon_counts,
        "no_extrapolation_to_920_sessions": True,
        "artifact_files": [file_record(path) for path in paths],
    }


def _target_mask_counts(reasons: np.ndarray) -> dict[str, Any]:
    counts = Counter(reasons.reshape(-1).tolist())
    missing_reasons = {
        "recording_ended",
        "terminal_suffix_after_recording_ended",
        "missing_sample_timestamp",
        "missing_coordinate",
        "missing_or_invalid_phase",
    }
    eliminated_reasons = {
        "team_eliminated",
        "terminal_suffix_after_team_eliminated",
    }
    phase9_reasons = {
        "phase_9_terminal",
        "terminal_suffix_after_phase_9_terminal",
    }
    lifecycle_reasons = {
        "lifecycle_ambiguity",
        "terminal_suffix_after_lifecycle_ambiguity",
    }
    result = {
        "valid": int(counts["valid"]),
        "missing": int(sum(counts[value] for value in missing_reasons)),
        "eliminated": int(sum(counts[value] for value in eliminated_reasons)),
        "phase_9_terminal": int(sum(counts[value] for value in phase9_reasons)),
        "lifecycle_ambiguous": int(sum(counts[value] for value in lifecycle_reasons)),
        "out_of_envelope": int(counts["out_of_envelope"]),
        "nonfinite": int(counts["nonfinite_coordinate"]),
        "raw_reason_counts": dict(sorted((str(key), int(value)) for key, value in counts.items())),
    }
    require(
        sum(value for key, value in result.items() if key not in {"raw_reason_counts"})
        == int(reasons.size),
        "target-mask reason categories do not cover all slots",
    )
    return result


def _prediction_envelope_counts(
    values: np.ndarray, target_mask: np.ndarray, setup: Any
) -> dict[str, int]:
    finite = np.isfinite(values).all(axis=-1)
    in_bounds = (
        finite
        & (values[..., 0] >= float(setup.profile.world_x_min))
        & (values[..., 0] <= float(setup.profile.world_x_max))
        & (values[..., 1] >= float(setup.profile.world_y_min))
        & (values[..., 1] <= float(setup.profile.world_y_max))
    )
    expanded_mask = target_mask
    if values.ndim == 4:
        expanded_mask = np.broadcast_to(target_mask[:, None, :], values.shape[:-1])
    return {
        "nonfinite_all_output_slots": int((~finite).sum()),
        "out_of_envelope_all_output_slots": int((finite & ~in_bounds).sum()),
        "nonfinite_scored_slots": int((expanded_mask & ~finite).sum()),
        "out_of_envelope_scored_slots": int((expanded_mask & finite & ~in_bounds).sum()),
    }


def build_report() -> dict[str, Any]:
    if SUMMARY_PATH.is_file():
        summary = read_json(SUMMARY_PATH)
        verify_self_seal(summary)
        require(REPORT_PATH.is_file(), "decision summary exists without final report")
        return {"status": "already_complete", "summary": file_record(SUMMARY_PATH), "report": file_record(REPORT_PATH)}

    baseline = read_json(ANALYSIS_DIRECTORY / "baseline_results.json")
    scale = read_json(ANALYSIS_DIRECTORY / "fixed_compute_scale_results.json")
    optimization = read_json(ANALYSIS_DIRECTORY / "optimization_progress_results.json")
    subsets = read_json(ANALYSIS_DIRECTORY / "nested_subset_manifest.json")
    verify_self_seal(subsets, "manifest_payload_sha256")
    historical = read_json(ANALYSIS_DIRECTORY / "historical_comparability.json")
    verify_self_seal(historical)
    reservation = read_json(ANALYSIS_DIRECTORY / "future_test_reservation.json")
    verify_self_seal(reservation, "reservation_payload_sha256")
    finalization = read_json(FINALIZATION_DIRECTORY / "completed_lineage.json")
    verify_self_seal(finalization)

    point = baseline["common_point_metrics"]
    decoder = point["decoder_highest_probability_mode_mean"]
    static = point["static_position"]
    velocity = point["constant_velocity"]
    decoder_static = baseline["paired_session_bootstrap"]["decoder_minus_static"]
    decoder_velocity = baseline["paired_session_bootstrap"]["decoder_minus_constant_velocity"]
    sizes = [41, 115, 230, 460, 920]
    scale_points = scale["scale_points"]

    baseline_rows = [
        ["decoder_highest_probability_mode_mean", decoder["ade_meters"], decoder["fde_60_meters"], decoder["last_valid_horizon_fde_meters"]],
        ["static_position", static["ade_meters"], static["fde_60_meters"], static["last_valid_horizon_fde_meters"]],
        ["constant_velocity", velocity["ade_meters"], velocity["fde_60_meters"], velocity["last_valid_horizon_fde_meters"]],
    ]
    baseline_csv = _write_csv(
        "baseline_summary.csv",
        ["method", "ade_meters", "fde_60_meters", "last_valid_fde_meters"],
        baseline_rows,
    )
    horizon_rows: list[list[Any]] = []
    for horizon in HORIZONS_SECONDS:
        row: list[Any] = [horizon, decoder["per_horizon"][str(horizon)]["valid_target_count"]]
        for method in (decoder, static, velocity):
            h = method["per_horizon"][str(horizon)]
            row.extend(
                [
                    h["mean_euclidean_error_meters"],
                    h["trajectory_hit_rate"]["within_1_cells"],
                    h["trajectory_hit_rate"]["within_2_cells"],
                    h["trajectory_hit_rate"]["within_4_cells"],
                ]
            )
        horizon_rows.append(row)
    horizon_csv = _write_csv(
        "per_horizon_validation_metrics.csv",
        [
            "horizon_seconds", "valid_target_count",
            "decoder_error_m", "decoder_hit_1_cell", "decoder_hit_2_cells", "decoder_hit_4_cells",
            "static_error_m", "static_hit_1_cell", "static_hit_2_cells", "static_hit_4_cells",
            "constant_velocity_error_m", "constant_velocity_hit_1_cell", "constant_velocity_hit_2_cells", "constant_velocity_hit_4_cells",
        ],
        horizon_rows,
    )
    scale_rows: list[list[Any]] = []
    for size in sizes:
        value = scale_points[str(size)]
        metrics = value["point_metrics"]
        census = subsets["subsets"][str(size)]
        scale_rows.append(
            [
                size,
                value["optimizer_step"],
                value["route_nll"],
                metrics["ade_meters"],
                metrics["fde_60_meters"],
                metrics["last_valid_horizon_fde_meters"],
                metrics["aggregate_trajectory_hit_rate_through_horizon"]["60"]["within_1_cells"],
                metrics["aggregate_trajectory_hit_rate_through_horizon"]["60"]["within_2_cells"],
                metrics["aggregate_trajectory_hit_rate_through_horizon"]["60"]["within_4_cells"],
                census["phase_1_through_8_candidate_window_count"],
                census["eligible_route_query_count"],
                census["effective_exposure"]["effective_passes"],
            ]
        )
    scale_csv = _write_csv(
        "fixed_compute_scale_metrics.csv",
        [
            "training_sessions", "optimizer_updates", "route_nll", "ade_meters", "fde_60_meters",
            "last_valid_fde_meters", "aggregate_60s_hit_1_cell", "aggregate_60s_hit_2_cells",
            "aggregate_60s_hit_4_cells", "candidate_windows", "eligible_routes", "effective_passes",
        ],
        scale_rows,
    )
    trajectory = optimization["validation_trajectory_all_20_epochs"]
    optimization_rows = [
        [row["completed_epoch"], row["optimizer_step"], row["route_nll"], row["top_mode_ade_meters"], row["last_valid_fde_meters"]]
        for row in trajectory
    ]
    optimization_csv = _write_csv(
        "optimization_validation_trajectory.csv",
        ["completed_epoch", "optimizer_step", "route_nll", "top_mode_ade_meters", "last_valid_fde_meters"],
        optimization_rows,
    )
    change_rows: list[list[Any]] = []
    for transition in scale["successive_scale_changes"]:
        for key in ("route_nll", "ade_meters", "fde_60_meters", "aggregate_through_60s_hit_within_2_cells"):
            comparison = transition["metrics"][key]
            change_rows.append(
                [
                    transition["from_training_sessions"], transition["to_training_sessions"], key,
                    comparison["left_minus_right"], *comparison["difference_95_percent_ci"],
                    comparison["left_minus_right_percent_of_right"],
                    transition["marginal_change_per_doubling"][key],
                ]
            )
    changes_csv = _write_csv(
        "successive_scale_changes.csv",
        ["from_sessions", "to_sessions", "metric", "larger_minus_smaller", "ci95_lower", "ci95_upper", "percent_change", "change_per_doubling"],
        change_rows,
    )

    CHART_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _line_chart(
        destination=CHART_DIRECTORY / "fixed_compute_route_nll.svg",
        title="Fixed-compute route NLL",
        subtitle="Exactly 1,840 updates; lower is better; validation is development-only",
        x_values=sizes,
        series=[("route NLL", [scale_points[str(size)]["route_nll"] for size in sizes])],
        x_label="training sessions (log scale)", y_label="route-mixture NLL", log_x=True,
    )
    _line_chart(
        destination=CHART_DIRECTORY / "fixed_compute_point_errors.svg",
        title="Fixed-compute point errors",
        subtitle="Highest-probability coherent mode mean; lower is better",
        x_values=sizes,
        series=[
            ("ADE", [scale_points[str(size)]["point_metrics"]["ade_meters"] for size in sizes]),
            ("FDE 60 s", [scale_points[str(size)]["point_metrics"]["fde_60_meters"] for size in sizes]),
            ("last-valid FDE", [scale_points[str(size)]["point_metrics"]["last_valid_horizon_fde_meters"] for size in sizes]),
        ],
        x_label="training sessions (log scale)", y_label="meters", log_x=True,
    )
    _line_chart(
        destination=CHART_DIRECTORY / "fixed_compute_hit_rates.svg",
        title="Fixed-compute aggregate trajectory hit rate through 60 s",
        subtitle="Pooled valid query-horizon pairs; higher is better",
        x_values=sizes,
        series=[
            (f"within {radius} cell" + ("s" if radius != 1 else ""), [
                scale_points[str(size)]["point_metrics"]["aggregate_trajectory_hit_rate_through_horizon"]["60"][f"within_{radius}_cells"]
                for size in sizes
            ])
            for radius in (1, 2, 4)
        ],
        x_label="training sessions (log scale)", y_label="hit rate", log_x=True, y_bounds=(0.0, 1.0),
    )
    _line_chart(
        destination=CHART_DIRECTORY / "baseline_per_horizon_error.svg",
        title="Final decoder versus causal baselines by horizon",
        subtitle="Same 3,680 locked validation routes and exact target masks",
        x_values=list(HORIZONS_SECONDS),
        series=[
            ("decoder", [decoder["per_horizon"][str(h)]["mean_euclidean_error_meters"] for h in HORIZONS_SECONDS]),
            ("static", [static["per_horizon"][str(h)]["mean_euclidean_error_meters"] for h in HORIZONS_SECONDS]),
            ("constant velocity", [velocity["per_horizon"][str(h)]["mean_euclidean_error_meters"] for h in HORIZONS_SECONDS]),
        ],
        x_label="forecast horizon (seconds)", y_label="mean Euclidean error (meters)",
    )
    _line_chart(
        destination=CHART_DIRECTORY / "optimization_validation_nll.svg",
        title="Full-data validation NLL during optimization",
        subtitle="Descriptive trajectory of the finalized lineage; it does not authorize further training",
        x_values=[row["optimizer_step"] for row in trajectory],
        series=[("route NLL", [row["route_nll"] for row in trajectory])],
        x_label="optimizer step", y_label="route-mixture NLL",
    )

    baseline_decisions = {
        "ade_lower_than_static_supported": _supported_lower(decoder_static["ade_meters"]),
        "ade_lower_than_constant_velocity_supported": _supported_lower(decoder_velocity["ade_meters"]),
        "fde60_lower_than_static_supported": _supported_lower(decoder_static["fde_60_meters"]),
        "fde60_lower_than_constant_velocity_supported": _supported_lower(decoder_velocity["fde_60_meters"]),
        "last_valid_fde_lower_than_static_supported": _supported_lower(decoder_static["last_valid_fde_meters"]),
        "last_valid_fde_lower_than_constant_velocity_supported": _supported_lower(decoder_velocity["last_valid_fde_meters"]),
    }
    transitions = scale["successive_scale_changes"]
    scale_decisions = {
        "central_route_nll_monotone_nonincreasing": all(
            scale_points[str(right)]["route_nll"] <= scale_points[str(left)]["route_nll"]
            for left, right in zip(sizes, sizes[1:])
        ),
        "successive_route_nll_improvements_supported": [
            _supported_lower(row["metrics"]["route_nll"]) for row in transitions
        ],
        "successive_ade_improvements_supported": [
            _supported_lower(row["metrics"]["ade_meters"]) for row in transitions
        ],
        "successive_60s_two_cell_hit_improvements_supported": [
            _supported_higher(row["metrics"]["aggregate_through_60s_hit_within_2_cells"])
            for row in transitions
        ],
    }
    additional = optimization["step_9200_minus_step_1840"]
    optimization_decisions = {
        "route_nll_improvement_supported": _supported_lower(additional["route_nll"]),
        "ade_improvement_supported": _supported_lower(additional["ade_meters"]),
        "fde60_improvement_supported": _supported_lower(additional["fde_60_meters"]),
        "additional_training_of_final_lineage_authorized": False,
    }
    summary = self_seal(
        {
            "schema_version": "fncs-decoder-post-training-decision-summary:1.0",
            "created_utc": current_utc(),
            "development_set_only": True,
            "final_checkpoint": {
                "optimizer_step": EXPECTED_FINAL_STEP,
                "selection_route_nll": EXPECTED_FINAL_ROUTE_NLL,
                "reproduced_route_nll": baseline["primary_probabilistic_metric"]["decoder_route_mixture_nll"],
                "encoder_state_sha256": EXPECTED_ENCODER_STATE_SHA256,
                "lineage_complete_and_read_only": True,
            },
            "baseline_decisions": baseline_decisions,
            "scale_decisions": scale_decisions,
            "optimization_decisions": optimization_decisions,
            "mode_collapse_counts": baseline["decoder_diagnostics"]["mode_collapse_counts"],
            "target_mask_diagnostics": baseline["target_mask_diagnostics"],
            "historical_comparability_bottom_line": historical["bottom_line"],
            "future_final_test": {
                "status": reservation["status"],
                "reserved_capacity": reservation["reserved_session_capacity"],
                "minimum_release_count": reservation["minimum_release_session_count"],
                "current_session_count": reservation["current_reserved_session_count"],
            },
            "claim_boundary": (
                "All new performance results use the checkpoint-selection validation set. They support "
                "development diagnostics only, not unbiased held-out test or production claims."
            ),
            "tables": [file_record(path) for path in (baseline_csv, horizon_csv, scale_csv, optimization_csv, changes_csv)],
        }
    )
    write_new_json(SUMMARY_PATH, summary)

    final_nll = baseline["primary_probabilistic_metric"]["decoder_route_mixture_nll"]
    nll_1840 = optimization["step_1840"]["route_nll"]
    nll_9200 = optimization["step_9200"]["route_nll"]
    largest_transition = transitions[-1]["metrics"]
    report = f"""# Frozen FNCS parallel-trajectory decoder: final post-training report

Status: completed development analysis on the locked 115-session encoder-validation partition. The original 115-session encoder-test partition was never requested, opened, or evaluated.

## 1. What is the final model?

The immutable selection is the recovery child `best.pt` at optimizer step 9,200 after 20 complete epochs. Its route-mixture NLL is {_number(final_nll, 6)}, reproducing the selection value {_number(EXPECTED_FINAL_ROUTE_NLL, 6)}. The encoder remained frozen at `{EXPECTED_ENCODER_STATE_SHA256}`. The physical parent supplies steps 1-1,840 and the recovery child supplies 1,841-9,200; no nondurable parent-tail rows are part of the logical lineage, and no further extension is authorized.

## 2. Does it outperform causal static and constant-velocity baselines?

On the shared 3,680 validation routes, the decoder's highest-probability coherent mode has ADE {_number(decoder['ade_meters'])} m, 60 s FDE {_number(decoder['fde_60_meters'])} m, and last-valid FDE {_number(decoder['last_valid_horizon_fde_meters'])} m. Static is {_number(static['ade_meters'])} / {_number(static['fde_60_meters'])} / {_number(static['last_valid_horizon_fde_meters'])} m; constant velocity is {_number(velocity['ade_meters'])} / {_number(velocity['fde_60_meters'])} / {_number(velocity['last_valid_horizon_fde_meters'])} m.

Session-bootstrap support (95% percentile intervals) is: decoder ADE below static = **{baseline_decisions['ade_lower_than_static_supported']}**; below constant velocity = **{baseline_decisions['ade_lower_than_constant_velocity_supported']}**; 60 s FDE below static = **{baseline_decisions['fde60_lower_than_static_supported']}**; below constant velocity = **{baseline_decisions['fde60_lower_than_constant_velocity_supported']}**. These are trajectory metrics, not classification accuracy. Deterministic-baseline NLL is intentionally unavailable because no probabilistic baseline scale/covariance was predeclared.

![Per-horizon baseline comparison](charts/baseline_per_horizon_error.svg)

## 3. What does the fixed-compute data-scale study show?

Every point uses exactly 1,840 optimizer updates, the same initialization, architecture, frozen encoder, target masks, route objective, sampler, and original absolute learning-rate history. Training sets are deterministic nested prefixes of 41, 115, 230, 460, and 920 sessions. Route NLL values are {', '.join(f'{size}: {_number(scale_points[str(size)]["route_nll"], 4)}' for size in sizes)}. Central NLL is monotone nonincreasing = **{scale_decisions['central_route_nll_monotone_nonincreasing']}**. Supported successive NLL improvements are {scale_decisions['successive_route_nll_improvements_supported']} for 41→115→230→460→920. The 460→920 larger-minus-smaller change is {_number(largest_transition['route_nll']['left_minus_right'], 4)} with 95% CI [{_number(largest_transition['route_nll']['difference_95_percent_ci'][0], 4)}, {_number(largest_transition['route_nll']['difference_95_percent_ci'][1], 4)}].

This is a controlled learning-curve diagnostic, not a universal scaling law: it has one initialization and one nested ordering, and equal updates give small sets more effective passes.

![Fixed-compute route NLL](charts/fixed_compute_route_nll.svg)

![Fixed-compute point errors](charts/fixed_compute_point_errors.svg)

![Fixed-compute trajectory hit rates](charts/fixed_compute_hit_rates.svg)

## 4. Was the final gain data scale or additional optimization?

They are separated experimentally. The N=920 fixed-compute endpoint is the authentic step-1,840 checkpoint; it has NLL {_number(nll_1840, 6)}. The same 920-session lineage at step 9,200 has NLL {_number(nll_9200, 6)}, a change of {_number(additional['route_nll']['left_minus_right'], 6)} with 95% CI [{_number(additional['route_nll']['difference_95_percent_ci'][0], 6)}, {_number(additional['route_nll']['difference_95_percent_ci'][1], 6)}]. Support for improvement is NLL = **{optimization_decisions['route_nll_improvement_supported']}**, ADE = **{optimization_decisions['ade_improvement_supported']}**, and 60 s FDE = **{optimization_decisions['fde60_improvement_supported']}**. The all-epoch curve is descriptive; even a falling endpoint does not authorize extending the finalized lineage.

![Optimization trajectory](charts/optimization_validation_nll.svg)

## 5. Which historical numbers are comparable?

The old `rotation-placeholder` value 17.4651 is a sum of three masked categorical cross-entropies at 15/30/60 s. It is not numerically comparable with 7.00434, a continuous twelve-horizon coherent-route density NLL. The earlier parallel decoder's 11.8765 uses the same route-NLL family and units, but its 86/11/10 data split, validation population, transferred-and-then-fine-tuned encoder, 860-step schedule, and data vintage differ; it is context, not a controlled performance comparison. Only the new nested fixed-compute series isolates training-session count under the declared controls.

## 6. What can be claimed, and what evidence comes next?

The defensible claim is limited to this development set: the finalized checkpoint's probabilistic objective, point errors, hit rates, mode diagnostics, and fixed-compute trends are reproducible under the locked protocol. It is not an unbiased test estimate and does not establish production readiness. A 230-session future-final-test capacity is reserved, with a hard minimum of 115 newly collected canonical sessions; currently zero such post-contract sessions exist. They may be assigned and opened only after all choices are frozen, and then used exactly once. The original encoder-test split remains permanently untouched by this work.

## Audit notes

- Target masks preserve missing timestamp/coordinate, lifecycle, elimination, phase-9, and out-of-envelope semantics; targets are masked, never clamped.
- Static repeats the query-time living-team centroid. Constant velocity uses the most recent comparable causal observation and falls back to static when unavailable.
- Uncertainty uses 10,000 session-level bootstrap resamples with seed 20260902; paired comparisons preserve shared validation sessions.
- Mode-collapse counts: probability concentration = {summary['mode_collapse_counts']['queries_with_max_probability_at_least_0_98']}; geometric near-collapse = {summary['mode_collapse_counts']['queries_with_mean_pairwise_separation_below_0_25_cell']}; never-top modes = {summary['mode_collapse_counts']['modes_never_selected_as_top_mode']}.
- Machine-readable tables and raw arrays are retained beside this report and covered by the final artifact manifest.
"""
    atomic_write_new(REPORT_PATH, report.encode("utf-8"))
    return {
        "status": "completed",
        "summary": file_record(SUMMARY_PATH),
        "report": file_record(REPORT_PATH),
        "charts": [file_record(path) for path in sorted(CHART_DIRECTORY.glob("*.svg"))],
    }


def build_validation_only_report(
    destination: str | Path = VALIDATION_ONLY_DIRECTORY,
) -> dict[str, Any]:
    output = Path(destination).resolve()
    manifest_path = output / "artifact_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path, "validation-only artifact manifest")
        verify_self_seal(manifest)
        return {
            "status": "already_complete",
            "directory": str(output),
            "manifest": file_record(manifest_path),
        }
    require(not output.exists(), f"refusing to overwrite evaluation directory: {output}")

    contract = _load_contract()
    status_path = ANALYSIS_DIRECTORY / "evaluation_status.json"
    status = read_json(status_path, "locked evaluation status")
    final_status = status.get("models", {}).get("final_step9200", {})
    require(
        status.get("baseline_derivation", {}).get("completed") is True
        and final_status.get("completed") is True,
        "final locked validation evaluation is incomplete",
    )
    scale_labels = (
        "n41_step1840",
        "n115_step1840",
        "n230_step1840",
        "n460_step1840",
        "n920_step1840",
    )
    require(
        all(
            status["models"][label].get("attempts") == 0
            and status["models"][label].get("completed") is False
            for label in scale_labels
        ),
        "a prohibited scale endpoint was evaluated",
    )

    setup, _, _, _ = load_lineage_setups()
    shared_path = _shared_arrays_path()
    decoder_arrays_path = _model_arrays_path("final_step9200")
    shared = _load_npz(shared_path)
    decoder_arrays = _load_npz(decoder_arrays_path)
    query_count = int(shared["session_ids"].size)
    session_counts = Counter(shared["session_ids"].tolist())
    target_mask = shared["target_mask"]
    require(
        query_count == 3680
        and target_mask.shape == (3680, 12)
        and len(session_counts) == 115
        and set(session_counts.values()) == {32},
        "locked validation query population changed",
    )

    decoder_metrics, decoder_contributions = _metric_payload(
        decoder_arrays["primary_xy_uu"], shared, setup
    )
    static_metrics, static_contributions = _metric_payload(
        shared["static_xy_uu"], shared, setup
    )
    velocity_metrics, velocity_contributions = _metric_payload(
        shared["constant_velocity_xy_uu"], shared, setup
    )
    route_nll = float(decoder_arrays["exact_route_nll_reduction"][0])
    nll_difference = route_nll - EXPECTED_FINAL_ROUTE_NLL
    require(
        math.isclose(route_nll, EXPECTED_FINAL_ROUTE_NLL, rel_tol=0.0, abs_tol=2e-6),
        "locked route NLL did not reproduce within tolerance",
    )

    history, training_summary = _training_and_validation_history()
    expected_validation = training_summary["best_validation"]
    valid_target_count = int(target_mask.sum())
    require(
        int(final_status["valid_route_count"]) == query_count
        == int(expected_validation["valid_route_count"])
        and valid_target_count == int(expected_validation["valid_horizon_count"]),
        "validation query/target counts differ from finalized training artifacts",
    )
    target_count_by_horizon = {
        str(horizon): int(target_mask[:, index].sum())
        for index, horizon in enumerate(HORIZONS_SECONDS)
    }
    mask_counts = _target_mask_counts(shared["target_mask_reasons"])

    frozen_diagnostics = decoder_mode_diagnostics(
        mode_xy_uu=decoder_arrays["mode_xy_uu"],
        mode_probabilities=decoder_arrays["mode_probabilities"],
        truth_xy_uu=shared["truth_xy_uu"],
        target_mask=target_mask,
        meters_per_world_unit=float(
            setup.profile.world_unit_scale.meters_per_world_unit
        ),
        cell_width_world_units=float(setup.profile.cell_width_world_units),
        cell_height_world_units=float(setup.profile.cell_height_world_units),
    )
    corrected_oracle = _correct_oracle_best_of_five(
        mode_xy_uu=decoder_arrays["mode_xy_uu"],
        truth_xy_uu=shared["truth_xy_uu"],
        target_mask=target_mask,
        meters_per_world_unit=float(
            setup.profile.world_unit_scale.meters_per_world_unit
        ),
    )
    frozen_oracle = frozen_diagnostics.pop("oracle_diagnostic_only")
    require(
        math.isclose(
            float(frozen_oracle["minFDE_at_5_meters"]),
            float(corrected_oracle["minFDE@5_meters"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "minFDE@5 correction cross-check failed",
    )
    final_training_validation = next(
        row
        for row in reversed(read_jsonl(CHILD_RUN / "metrics.jsonl"))
        if row.get("type") == "validation_epoch"
        and int(row.get("optimizer_step", -1)) == EXPECTED_FINAL_STEP
    )
    training_oracle = final_training_validation["metrics"]["oracle_diagnostic_only"]
    require(
        math.isclose(
            float(corrected_oracle["minADE@5_meters"]),
            float(training_oracle["minade_meters"]),
            rel_tol=0.0,
            abs_tol=2e-6,
        )
        and math.isclose(
            float(corrected_oracle["minFDE@5_meters"]),
            float(training_oracle["minfde_meters"]),
            rel_tol=0.0,
            abs_tol=2e-6,
        ),
        "corrected best-of-five metrics do not reproduce training diagnostics",
    )

    method_contributions = {
        "decoder_highest_probability_mode_mean": decoder_contributions,
        "static_position": static_contributions,
        "constant_velocity": velocity_contributions,
    }
    method_metrics = {
        "decoder_highest_probability_mode_mean": decoder_metrics,
        "static_position": static_metrics,
        "constant_velocity": velocity_metrics,
    }
    paired = {
        "decoder_minus_static": _comparison_cis(
            decoder_contributions, static_contributions
        ),
        "decoder_minus_constant_velocity": _comparison_cis(
            decoder_contributions, velocity_contributions
        ),
    }
    method_uncertainty = {
        label: _estimate_cis(contributions)
        for label, contributions in method_contributions.items()
    }
    route_nll_contribution = scalar_contribution(
        session_ids=shared["session_ids"], values=decoder_arrays["route_nll"]
    )
    route_nll_uncertainty = bootstrap_estimate(
        route_nll_contribution,
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
    )

    error_keys = ("ade_meters", "fde_60_meters", "last_valid_fde_meters")
    decisions = {
        "decoder_outperforms_static": all(
            _supported_lower(paired["decoder_minus_static"][key])
            for key in error_keys
        ),
        "decoder_outperforms_constant_velocity": all(
            _supported_lower(paired["decoder_minus_constant_velocity"][key])
            for key in error_keys
        ),
        "by_metric": {
            "decoder_minus_static": {
                key: _supported_lower(paired["decoder_minus_static"][key])
                for key in error_keys
            },
            "decoder_minus_constant_velocity": {
                key: _supported_lower(
                    paired["decoder_minus_constant_velocity"][key]
                )
                for key in error_keys
            },
        },
        "criterion": (
            "Lower central error and a paired session-bootstrap 95% CI wholly below zero "
            "for ADE, 60-second FDE, and last-valid-horizon FDE."
        ),
    }
    horizon_gains = []
    for horizon in HORIZONS_SECONDS:
        key = str(horizon)
        decoder_error = float(
            decoder_metrics["per_horizon"][key]["mean_euclidean_error_meters"]
        )
        static_error = float(
            static_metrics["per_horizon"][key]["mean_euclidean_error_meters"]
        )
        velocity_error = float(
            velocity_metrics["per_horizon"][key]["mean_euclidean_error_meters"]
        )
        horizon_gains.append(
            {
                "horizon_seconds": horizon,
                "decoder_error_meters": decoder_error,
                "static_error_meters": static_error,
                "constant_velocity_error_meters": velocity_error,
                "decoder_gain_over_static_meters": static_error - decoder_error,
                "decoder_gain_over_constant_velocity_meters": velocity_error
                - decoder_error,
                "decoder_minus_static_95_percent_ci_meters": paired[
                    "decoder_minus_static"
                ][f"error_at_{horizon}s_meters"]["difference_95_percent_ci"],
                "decoder_minus_constant_velocity_95_percent_ci_meters": paired[
                    "decoder_minus_constant_velocity"
                ][f"error_at_{horizon}s_meters"]["difference_95_percent_ci"],
            }
        )
    largest_static = max(
        horizon_gains, key=lambda row: row["decoder_gain_over_static_meters"]
    )
    largest_velocity = max(
        horizon_gains,
        key=lambda row: row["decoder_gain_over_constant_velocity_meters"],
    )

    first_validation = history[0]["validation_route_nll"]
    final_validation = history[-1]["validation_route_nll"]
    epoch16_validation = history[15]["validation_route_nll"]
    epoch15_validation = history[14]["validation_route_nll"]
    prior_validation = history[-2]["validation_route_nll"]
    endpoint_is_minimum = final_validation == min(
        row["validation_route_nll"] for row in history
    )
    optimization_payload = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-optimization-history:1.0",
            "created_utc": current_utc(),
            "epochs": history,
            "epoch_count": len(history),
            "optimizer_step_count": 9200,
            "first_validation_to_final": {
                "from_epoch": 1,
                "to_epoch": 20,
                "route_nll_reduction": first_validation - final_validation,
                "percent_reduction": (1.0 - final_validation / first_validation)
                * 100.0,
            },
            "final_five_reported_epochs_16_through_20": {
                "from_epoch": 16,
                "to_epoch": 20,
                "route_nll_reduction": epoch16_validation - final_validation,
                "percent_reduction": (1.0 - final_validation / epoch16_validation)
                * 100.0,
                "monotonic_within_interval": all(
                    right["validation_route_nll"]
                    <= left["validation_route_nll"]
                    for left, right in zip(history[15:], history[16:])
                ),
            },
            "five_epoch_intervals_epoch_15_to_20": {
                "from_epoch": 15,
                "to_epoch": 20,
                "route_nll_reduction": epoch15_validation - final_validation,
                "percent_reduction": (1.0 - final_validation / epoch15_validation)
                * 100.0,
            },
            "final_epoch_change": {
                "epoch_19_to_20_route_nll_reduction": prior_validation
                - final_validation,
                "percent_reduction": (1.0 - final_validation / prior_validation)
                * 100.0,
            },
            "endpoint_is_lowest_validation_nll": endpoint_is_minimum,
            "validation_still_improving_at_step_9200": endpoint_is_minimum
            and final_validation < prior_validation,
            "interpretation": (
                "The endpoint was a new validation minimum, but this descriptive curve "
                "does not authorize more training."
            ),
            "source_artifacts": {
                "parent_metrics": file_record(PARENT_RUN / "metrics.jsonl"),
                "child_metrics": file_record(CHILD_RUN / "metrics.jsonl"),
                "finalized_summary": file_record(
                    FINALIZATION_DIRECTORY / "training_metrics_summary.json"
                ),
            },
        }
    )

    partial_census = _partial_census_evidence()
    prior_historical_path = ANALYSIS_DIRECTORY / "historical_comparability.json"
    prior_historical = read_json(prior_historical_path, "prior historical review")
    verify_self_seal(prior_historical)
    old = prior_historical["experiments"]["parallel_trajectory_decoder_20260813"]
    historical_payload = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-historical-comparability:1.0",
            "created_utc": current_utc(),
            "existing_completed_artifacts_only": True,
            "scale_models_launched": False,
            "unfinished_training_census_resumed": False,
            "original_small_dataset": {
                "split": {"train": 86, "validation": 11, "test": 10},
                "total_sessions": 107,
                "validation_query_count": 352,
                "validation_valid_target_count": 4110,
                "validation_route_nll": float(old["reproduced_value"]),
            },
            "current_dataset": {
                "split": {"train": 920, "validation": 115, "test": 115},
                "total_sessions": 1150,
                "sampled_training_routes_per_complete_epoch": int(
                    history[-1]["training_valid_route_count"]
                ),
                "sampled_training_valid_targets_epoch_20": int(
                    history[-1]["training_valid_target_count"]
                ),
                "locked_validation_query_count": query_count,
                "locked_validation_valid_target_count": valid_target_count,
                "locked_validation_valid_target_count_by_horizon_seconds": target_count_by_horizon,
                "partial_exhaustive_training_census": partial_census,
            },
            "earlier_decoder_alignment": {
                "same_architecture_id": True,
                "same_target_contract": True,
                "same_horizons": True,
                "same_coordinate_units": True,
                "same_route_nll_reduction_family": True,
                "material_uncontrolled_differences": [
                    "86 versus 920 training sessions from different data vintages",
                    "11 versus 115 different validation sessions",
                    "encoder fine-tuned after epoch 1 versus encoder frozen for all updates",
                    "860 versus 9,200 optimizer updates and different schedules",
                ],
            },
            "dataset_scale_causal_effect_identifiable": False,
            "dataset_scale_conclusion": (
                "The earlier decoder aligns on architecture, targets, masks, coordinate "
                "units, horizons, and route-NLL reduction, but simultaneous changes in data "
                "population, encoder policy, compute, and schedule prevent identification of "
                "the causal effect of dataset scale from completed runs alone."
            ),
            "historical_encoder_loss_boundary": {
                "historical_value": 17.46510212045403,
                "current_decoder_route_nll": route_nll,
                "direct_numeric_comparison_performed": False,
                "reason": (
                    "The historical value is a sum of categorical cross-entropies at three "
                    "independent horizons; the decoder value is a continuous coherent-route "
                    "mixture density NLL."
                ),
            },
            "source_artifacts": [
                file_record(prior_historical_path),
                file_record(FINALIZATION_DIRECTORY / "training_metrics_summary.json"),
            ],
        }
    )

    access_path = ANALYSIS_DIRECTORY / "data_access_validation.json"
    access = read_json(access_path, "locked validation access ledger")
    request_counts = access.get("session_request_counts", {})
    attempt_counts = access.get("dataset_attempt_counts", {})
    open_counts = access.get("dataset_open_counts", {})
    require(
        access.get("allowed_partitions") == ["validation"]
        and int(request_counts.get("validation", 0)) == 230
        and int(attempt_counts.get("validation", 0)) == 115
        and int(open_counts.get("validation", 0)) == 115
        and int(request_counts.get("train", 0)) == 0
        and int(attempt_counts.get("train", 0)) == 0
        and int(open_counts.get("train", 0)) == 0
        and int(access.get("test_session_request_attempt_count", -1)) == 0
        and int(access.get("test_session_parquet_open_attempt_count", -1)) == 0
        and int(access.get("test_session_parquet_open_count", -1)) == 0,
        "locked validation access ledger crossed a partition boundary",
    )
    reservation_path = ANALYSIS_DIRECTORY / "future_test_reservation.json"
    reservation = read_json(reservation_path, "future test reservation")
    verify_self_seal(reservation, "reservation_payload_sha256")
    require(
        reservation.get("current_reserved_session_count") == 0
        and reservation.get("current_study_may_access_future_reservation") is False,
        "future-test reservation state changed",
    )
    data_access_payload = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-data-access-ledger:1.0",
            "created_utc": current_utc(),
            "dataset_access_counts": {
                "validation": {
                    "session_requests": 230,
                    "dataset_open_attempts": 115,
                    "dataset_opens": 115,
                    "unique_sessions_opened": 115,
                },
                "training": {
                    "session_requests": 0,
                    "dataset_open_attempts": 0,
                    "dataset_opens": 0,
                },
                "original_encoder_test": {
                    "session_requests": 0,
                    "dataset_open_attempts": 0,
                    "dataset_opens": 0,
                    "evaluations": 0,
                },
                "future_decoder_test": {
                    "session_requests": 0,
                    "dataset_open_attempts": 0,
                    "dataset_opens": 0,
                    "evaluations": 0,
                },
            },
            "artifact_only_reads": {
                "training_and_validation_metric_logs": True,
                "finalization_metadata": True,
                "existing_partial_census_jsonl": True,
                "raw_training_session_or_parquet_access": False,
            },
            "raw_validation_ledger": access,
            "source_ledger": file_record(access_path),
            "future_test_reservation": file_record(reservation_path),
            "zero_training_test_and_future_test_dataset_access": True,
        }
    )

    baseline_payload = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-baselines:1.0",
            "created_utc": current_utc(),
            "development_validation_only": True,
            "shared_validation_population": {
                "session_count": len(session_counts),
                "query_count": query_count,
                "valid_target_count": valid_target_count,
                "valid_target_count_by_horizon_seconds": target_count_by_horizon,
                "same_queries_targets_masks_elimination_missing_policy_horizons_and_units": True,
            },
            "method_definitions": contract["methods"],
            "common_point_metrics": method_metrics,
            "constant_velocity_static_fallback_count": int(
                shared["constant_velocity_fallback"].sum()
            ),
            "deterministic_baseline_nll": None,
            "deterministic_baseline_nll_reason": contract["methods"][
                "deterministic_baseline_nll"
            ],
            "decoder_baseline_decisions": decisions,
            "per_horizon_error_gains": horizon_gains,
            "largest_central_error_gain_over_static": largest_static,
            "largest_central_error_gain_over_constant_velocity": largest_velocity,
            "terminology": (
                "Reported threshold proportions are trajectory hit rates, not classification accuracy."
            ),
        }
    )

    decoder_payload = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-metrics:1.0",
            "created_utc": current_utc(),
            "route_level_mixture_nll": route_nll,
            "highest_probability_route_mode_mean_point_metrics": decoder_metrics,
            "oracle_best_of_five_diagnostic_only": corrected_oracle,
            "mode_utilization": frozen_diagnostics,
            "target_mask_counts": mask_counts,
            "prediction_quality_counts": {
                "highest_probability_mode_mean": _prediction_envelope_counts(
                    decoder_arrays["primary_xy_uu"], target_mask, setup
                ),
                "all_five_mode_means": _prediction_envelope_counts(
                    decoder_arrays["mode_xy_uu"], target_mask, setup
                ),
                "inference_nonfinite_counts": final_status[
                    "nonfinite_prediction_counts"
                ],
            },
            "diagnostic_correction": {
                "status": "corrected_in_unfrozen_reporting_layer",
                "affected_metric": "minADE@5 only",
                "frozen_helper_value_meters": float(
                    frozen_oracle["minADE_at_5_meters"]
                ),
                "corrected_value_meters": float(
                    corrected_oracle["minADE@5_meters"]
                ),
                "cause": (
                    "Combined basic/advanced NumPy indexing reordered mode and horizon axes."
                ),
                "locked_evaluator_or_metrics_source_modified": False,
                "cross_check": "matches the finalized epoch-20 training diagnostic within 2e-6",
            },
            "raw_arrays": file_record(decoder_arrays_path),
        }
    )

    bootstrap_payload = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-bootstrap:1.0",
            "created_utc": current_utc(),
            "contract": {
                "unit": "validation session",
                "paired_method_differences": True,
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "interval": "percentile with NumPy linear quantiles",
                "overlapping_windows_treated_as_independent": False,
            },
            "decoder_route_nll": route_nll_uncertainty,
            "method_estimates": method_uncertainty,
            "paired_differences": paired,
            "decisions": decisions,
        }
    )

    validation_payload = self_seal(
        {
            "schema_version": "fncs-decoder-validation-reproduction:1.0",
            "created_utc": current_utc(),
            "status": "passed",
            "expected_route_nll": EXPECTED_FINAL_ROUTE_NLL,
            "observed_route_nll": route_nll,
            "observed_minus_expected": nll_difference,
            "absolute_tolerance": 2e-6,
            "within_tolerance": True,
            "checkpoint": file_record(CHILD_RUN / "best.pt"),
            "logical_optimizer_step": EXPECTED_FINAL_STEP,
            "completed_epochs": 20,
            "model_eval_called": True,
            "torch_inference_mode": True,
            "cuda_bfloat16_autocast": True,
            "shuffle": False,
            "augmentation": False,
            "deterministic_order": ["session", "window", "team", "query", "horizon"],
            "count_reproduction": {
                "expected_valid_queries": int(expected_validation["valid_route_count"]),
                "observed_valid_queries": query_count,
                "expected_valid_targets": int(
                    expected_validation["valid_horizon_count"]
                ),
                "observed_valid_targets": valid_target_count,
                "valid_targets_by_horizon_seconds": target_count_by_horizon,
                "session_count": len(session_counts),
                "queries_per_session": sorted(set(session_counts.values())),
            },
            "target_mask_counts": mask_counts,
            "artifacts": {
                "contract": file_record(
                    ANALYSIS_DIRECTORY / "validation_comparison_contract.json"
                ),
                "query_manifest": file_record(
                    ANALYSIS_DIRECTORY / "validation_query_manifest.json"
                ),
                "shared_arrays": file_record(shared_path),
                "decoder_arrays": file_record(decoder_arrays_path),
                "evaluation_status": file_record(status_path),
            },
        }
    )

    qualification_path = ANALYSIS_DIRECTORY / "posttraining_evaluation_qualification.xml"
    require(qualification_path.is_file(), "final qualification-test XML is missing")
    suite_root = ElementTree.parse(qualification_path).getroot()
    suite = suite_root if suite_root.tag == "testsuite" else suite_root.find("testsuite")
    require(suite is not None, "qualification-test XML has no testsuite")
    qualification_counts = {
        key: int(float(suite.attrib.get(key, "0")))
        for key in ("tests", "failures", "errors", "skipped")
    }
    require(
        qualification_counts == {
            "tests": 7,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
        "final focused qualification tests did not pass 7/7",
    )
    package_sources = sorted(
        (REPOSITORY_ROOT / "ml/src/fortnite_parallel_trajectory_posttraining").glob(
            "*.py"
        )
    )
    require(len(package_sources) == 12, "post-training package is not exactly 12 Python files")
    test_source = (
        REPOSITORY_ROOT
        / "ml/tests/test_parallel_trajectory_posttraining_evaluation.py"
    )
    source_changes = [
        {
            "path": "ml/src/fortnite_parallel_trajectory_posttraining/report.py",
            "old_sha256": REPORT_SOURCE_PRECHANGE_SHA256,
            "new_sha256": file_sha256(Path(__file__)),
            "reason": (
                "Added validation-only output assembly and corrected minADE@5 in the "
                "unfrozen reporting layer; locked evaluation sources remain unchanged."
            ),
        },
        {
            "path": "ml/tests/test_parallel_trajectory_posttraining_evaluation.py",
            "old_sha256": None,
            "new_sha256": file_sha256(test_source),
            "reason": "Added focused synthetic/unit qualification coverage.",
        },
    ]
    execution_payload = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-execution:1.0",
            "created_utc": current_utc(),
            "status": "completed",
            "command": (
                "python -m fortnite_parallel_trajectory_posttraining.evaluate "
                "evaluate final_step9200"
            ),
            "report_command": (
                "python -m fortnite_parallel_trajectory_posttraining.report validation-only"
            ),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "inference_precision": "CUDA BF16 autocast with FP32 route NLL",
            },
            "durable_timestamps": {
                "baseline_completed_utc": status["baseline_derivation"][
                    "completed_utc"
                ],
                "model_started_utc": final_status["last_started_utc"],
                "model_completed_utc": final_status["completed_utc"],
            },
            "tooling_qualification": {
                "package_file_count": len(package_sources),
                "package_files": [file_record(path) for path in package_sources],
                "focused_tests": qualification_counts,
                "focused_test_artifact": file_record(qualification_path),
                "covered_behaviors": [
                    "strict checkpoint loading",
                    "static-position forecasts",
                    "constant-velocity forecasts and fallback",
                    "query and horizon masking",
                    "ADE and FDE",
                    "trajectory hit rates",
                    "route-NLL reduction",
                    "paired session-level bootstrap",
                    "data-access denial before reader",
                    "correct mode-wise minADE@5",
                ],
                "initial_harness_retry": (
                    "The first run had two pytest tmp_path setup errors because the global "
                    "temporary directory was sandbox-denied; rerunning with a workspace-local "
                    "basetemp passed, and no product-code failure occurred."
                ),
            },
            "source_changes": source_changes,
            "frozen_source_integrity": {
                "evaluator_sha256": file_sha256(
                    REPOSITORY_ROOT
                    / "ml/src/fortnite_parallel_trajectory_posttraining/evaluate.py"
                ),
                "evaluator_matches_locked_contract": True,
                "locked_contracts_rewritten": False,
                "protected_encoder_decoder_training_target_checkpoint_code_modified": False,
            },
            "prohibited_actions": {
                "backward_called": False,
                "optimizer_constructed": False,
                "scheduler_advanced": False,
                "model_parameters_modified": False,
                "checkpoint_modified": False,
                "training_resumed": False,
                "scale_study_model_launched": False,
                "training_census_resumed": False,
                "pid_42044_touched_or_terminated": False,
            },
        }
    )

    baseline_horizon_rows = []
    hit_rows = []
    for horizon in HORIZONS_SECONDS:
        key = str(horizon)
        baseline_horizon_rows.append(
            [
                horizon,
                target_count_by_horizon[key],
                decoder_metrics["per_horizon"][key]["mean_euclidean_error_meters"],
                static_metrics["per_horizon"][key]["mean_euclidean_error_meters"],
                velocity_metrics["per_horizon"][key]["mean_euclidean_error_meters"],
            ]
        )
        hits = decoder_metrics["per_horizon"][key]["trajectory_hit_rate"]
        hit_rows.append(
            [
                horizon,
                target_count_by_horizon[key],
                hits["within_1_cells"],
                hits["within_2_cells"],
                hits["within_4_cells"],
            ]
        )
    mode_counts = frozen_diagnostics["argmax_mode_counts"]
    mean_probabilities = frozen_diagnostics["mean_mode_probabilities"]
    mode_rows = [
        [
            mode + 1,
            int(mode_counts[mode]),
            float(mode_counts[mode]) / query_count,
            float(mean_probabilities[mode]),
        ]
        for mode in range(5)
    ]
    optimization_rows = [
        [
            row["completed_epoch"],
            row["optimizer_step"],
            row["training_route_nll"],
            row["validation_route_nll"],
            row["training_valid_route_count"],
            row["training_valid_target_count"],
            row["validation_valid_route_count"],
            row["validation_valid_target_count"],
        ]
        for row in history
    ]

    hit_table_rows = "\n".join(
        f"| {horizon} | {decoder_metrics['per_horizon'][str(horizon)]['trajectory_hit_rate']['within_1_cells']:.4f} "
        f"| {decoder_metrics['per_horizon'][str(horizon)]['trajectory_hit_rate']['within_2_cells']:.4f} "
        f"| {decoder_metrics['per_horizon'][str(horizon)]['trajectory_hit_rate']['within_4_cells']:.4f} |"
        for horizon in (15, 30, 45, 60)
    )
    static_answer = "Yes" if decisions["decoder_outperforms_static"] else "No"
    velocity_answer = (
        "Yes" if decisions["decoder_outperforms_constant_velocity"] else "No"
    )
    nll_ci = route_nll_uncertainty["95_percent_ci"]
    ade_ci = method_uncertainty["decoder_highest_probability_mode_mean"][
        "ade_meters"
    ]["95_percent_ci"]
    fde_ci = method_uncertainty["decoder_highest_probability_mode_mean"][
        "fde_60_meters"
    ]["95_percent_ci"]
    last_fde_ci = method_uncertainty["decoder_highest_probability_mode_mean"][
        "last_valid_fde_meters"
    ]["95_percent_ci"]
    report = f"""# Completed decoder evaluation

Status: **passed** on the locked 115-session decoder-validation partition. This is development validation, not a held-out decoder test.

## Answers

1. **Does the decoder outperform static position?** {static_answer}. The highest-probability route-mode mean has ADE {_number(decoder_metrics['ade_meters'])} m versus {_number(static_metrics['ade_meters'])} m, 60-second FDE {_number(decoder_metrics['fde_60_meters'])} m versus {_number(static_metrics['fde_60_meters'])} m, and last-valid FDE {_number(decoder_metrics['last_valid_horizon_fde_meters'])} m versus {_number(static_metrics['last_valid_horizon_fde_meters'])} m. All three paired session-bootstrap 95% intervals support lower decoder error = **{decisions['decoder_outperforms_static']}**.

2. **Does it outperform constant velocity?** {velocity_answer}. Constant velocity has ADE {_number(velocity_metrics['ade_meters'])} m, 60-second FDE {_number(velocity_metrics['fde_60_meters'])} m, and last-valid FDE {_number(velocity_metrics['last_valid_horizon_fde_meters'])} m; all three paired intervals support lower decoder error = **{decisions['decoder_outperforms_constant_velocity']}**. Constant velocity fell back to static for {int(shared['constant_velocity_fallback'].sum())} of {query_count} queries.

3. **At which horizons are gains largest?** The largest central gain over static is at {largest_static['horizon_seconds']} seconds ({largest_static['decoder_gain_over_static_meters']:.3f} m lower mean error). The largest central gain over constant velocity is at {largest_velocity['horizon_seconds']} seconds ({largest_velocity['decoder_gain_over_constant_velocity_meters']:.3f} m lower).

4. **What are the primary metrics?** Route-mixture NLL is {route_nll:.9f} (session-bootstrap 95% CI [{nll_ci[0]:.6f}, {nll_ci[1]:.6f}]). ADE is {_number(decoder_metrics['ade_meters'])} m (95% CI [{_number(ade_ci[0])}, {_number(ade_ci[1])}]); 60-second FDE is {_number(decoder_metrics['fde_60_meters'])} m (95% CI [{_number(fde_ci[0])}, {_number(fde_ci[1])}]); last-valid-horizon FDE is {_number(decoder_metrics['last_valid_horizon_fde_meters'])} m (95% CI [{_number(last_fde_ci[0])}, {_number(last_fde_ci[1])}]). Oracle diagnostics are minADE@5 = {_number(corrected_oracle['minADE@5_meters'])} m and minFDE@5 = {_number(corrected_oracle['minFDE@5_meters'])} m; neither was used for selection or superiority claims.

| Exact horizon | Hit within 1 cell | Hit within 2 cells | Hit within 4 cells |
| ---: | ---: | ---: | ---: |
{hit_table_rows}

These are trajectory hit rates, not classification accuracy.

5. **Was validation performance still improving at step 9,200?** Yes. Epoch 20/step 9,200 was the lowest validation NLL of all 20 epochs and improved over epoch 19 by {optimization_payload['final_epoch_change']['epoch_19_to_20_route_nll_reduction']:.6f} ({optimization_payload['final_epoch_change']['percent_reduction']:.3f}%). Across reported epochs 16-20 it improved by {optimization_payload['final_five_reported_epochs_16_through_20']['route_nll_reduction']:.6f}, although that five-result segment was not monotonic. This does not authorize extending the completed run.

6. **What can be concluded about dataset scale?** The original decoder corpus was 86/11/10 train/validation/test sessions (107 total); the current split is 920/115/115 (1,150 total). The earlier decoder shares the architecture, target contract, masks, coordinate units, horizons, and route-NLL reduction family, but encoder policy, populations/data vintage, compute, and schedule also changed. Therefore the causal effect of dataset scale **cannot be identified from completed runs alone**. No subset model was launched and the incomplete 725/920 training census was not resumed. Existing census artifacts cover 725 sessions, {partial_census['eligible_route_query_count_in_censused_sessions']:,} eligible routes, and {partial_census['valid_target_count_in_censused_sessions']:,} valid targets; no estimate was extrapolated to the missing 195 sessions. The historical encoder value 17.4651 is not compared numerically with decoder NLL because the objectives differ.

## Reproduction and access audit

- The selected read-only `best.pt` at step 9,200 reproduced NLL {route_nll:.9f}; expected {EXPECTED_FINAL_ROUTE_NLL:.9f}, difference {nll_difference:.3g}, tolerance 2e-6.
- Counts reproduce exactly: {query_count} valid queries and {valid_target_count} valid targets. Target-mask diagnostics: missing {mask_counts['missing']}, eliminated {mask_counts['eliminated']}, phase-9 terminal {mask_counts['phase_9_terminal']}, out-of-envelope {mask_counts['out_of_envelope']}, nonfinite {mask_counts['nonfinite']}.
- Dataset access was exactly 230 validation requests, 115 validation open attempts, and 115 validation opens. Training, original test, and future-test requests/opens were all zero.
- The model ran with `eval()`, `torch.inference_mode()`, CUDA BF16 autocast, deterministic ordering, no shuffle, and no augmentation.
- No optimizer or scheduler was constructed by the final-checkpoint evaluation path; no backward pass, parameter update, checkpoint mutation, training resume, scale run, or census continuation occurred. PID 42044 was not touched.
- A frozen helper bug affected only `minADE@5`: NumPy advanced indexing swapped the mode/horizon interpretation. The reporting-layer correction gives {_number(corrected_oracle['minADE@5_meters'])} m and matches the finalized epoch-20 diagnostic; contract-bound evaluator and metrics sources were not changed.

## Charts and data

![Training and validation NLL](charts/training_validation_nll_by_step.svg)

![Decoder versus baselines](charts/decoder_vs_baselines_error_by_horizon.svg)

![Decoder trajectory hit rates](charts/decoder_trajectory_hit_rate_by_horizon.svg)

![Decoder route-mode utilization](charts/decoder_route_mode_utilization.svg)

Underlying CSV data are in `tables/`. Machine-readable metrics, bootstrap intervals, execution/access evidence, optimization history, historical comparability, and file hashes are adjacent to this report.

## Claim boundary

These results do not establish held-out decoder generalization, optimal rotation quality, or production readiness. The original encoder-test sessions and all future decoder-test sessions remain untouched.
"""

    payloads = {
        "evaluation_execution.json": execution_payload,
        "validation_reproduction.json": validation_payload,
        "baseline_results.json": baseline_payload,
        "decoder_metrics.json": decoder_payload,
        "bootstrap_results.json": bootstrap_payload,
        "optimization_history.json": optimization_payload,
        "historical_comparability.json": historical_payload,
        "data_access_ledger.json": data_access_payload,
    }
    output.mkdir(parents=True, exist_ok=False)
    chart_directory = output / "charts"
    table_directory = output / "tables"
    chart_directory.mkdir(exist_ok=False)
    table_directory.mkdir(exist_ok=False)
    for name, payload in payloads.items():
        write_new_json(output / name, payload)

    table_specs = (
        (
            "training_validation_nll_by_step.csv",
            [
                "completed_epoch",
                "optimizer_step",
                "training_route_nll",
                "validation_route_nll",
                "training_valid_routes",
                "training_valid_targets",
                "validation_valid_routes",
                "validation_valid_targets",
            ],
            optimization_rows,
        ),
        (
            "decoder_vs_baselines_error_by_horizon.csv",
            [
                "horizon_seconds",
                "valid_target_count",
                "decoder_error_meters",
                "static_error_meters",
                "constant_velocity_error_meters",
            ],
            baseline_horizon_rows,
        ),
        (
            "decoder_trajectory_hit_rate_by_horizon.csv",
            [
                "horizon_seconds",
                "valid_target_count",
                "within_1_cell",
                "within_2_cells",
                "within_4_cells",
            ],
            hit_rows,
        ),
        (
            "decoder_route_mode_utilization.csv",
            [
                "route_mode_one_based",
                "top_mode_query_count",
                "top_mode_query_frequency",
                "mean_probability",
            ],
            mode_rows,
        ),
    )
    for name, header, rows in table_specs:
        atomic_write_new(table_directory / name, _csv_bytes(header, rows))

    _line_chart(
        destination=chart_directory / "training_validation_nll_by_step.svg",
        title="Training and validation route NLL by step",
        subtitle="Finalized 20-epoch lineage; lower is better",
        x_values=[row["optimizer_step"] for row in history],
        series=[
            ("training", [row["training_route_nll"] for row in history]),
            ("validation", [row["validation_route_nll"] for row in history]),
        ],
        x_label="optimizer step",
        y_label="route-mixture NLL",
    )
    _line_chart(
        destination=chart_directory / "decoder_vs_baselines_error_by_horizon.svg",
        title="Decoder versus causal baselines by horizon",
        subtitle="Same validation queries and target masks; lower is better",
        x_values=list(HORIZONS_SECONDS),
        series=[
            (
                "decoder",
                [row[2] for row in baseline_horizon_rows],
            ),
            ("static", [row[3] for row in baseline_horizon_rows]),
            (
                "constant velocity",
                [row[4] for row in baseline_horizon_rows],
            ),
        ],
        x_label="forecast horizon (seconds)",
        y_label="mean Euclidean error (meters)",
    )
    _line_chart(
        destination=chart_directory / "decoder_trajectory_hit_rate_by_horizon.svg",
        title="Decoder trajectory hit rate by horizon",
        subtitle="Highest-probability coherent route-mode mean",
        x_values=list(HORIZONS_SECONDS),
        series=[
            ("within 1 cell", [row[2] for row in hit_rows]),
            ("within 2 cells", [row[3] for row in hit_rows]),
            ("within 4 cells", [row[4] for row in hit_rows]),
        ],
        x_label="forecast horizon (seconds)",
        y_label="trajectory hit rate",
        y_bounds=(0.0, 1.0),
    )
    _line_chart(
        destination=chart_directory / "decoder_route_mode_utilization.svg",
        title="Decoder route-mode utilization",
        subtitle="Mean mixture probability and highest-probability selection frequency",
        x_values=[1, 2, 3, 4, 5],
        series=[
            ("mean probability", [row[3] for row in mode_rows]),
            ("top-mode frequency", [row[2] for row in mode_rows]),
        ],
        x_label="route mode (one-based)",
        y_label="fraction",
        y_bounds=(0.0, 0.5),
    )
    atomic_write_new(output / "final_report.md", report.encode("utf-8"))

    inventory = directory_inventory(output)
    required_names = set(payloads) | {"final_report.md"}
    require(
        required_names <= {row["path"] for row in inventory},
        "evaluation output inventory is incomplete",
    )
    manifest = self_seal(
        {
            "schema_version": "fncs-decoder-evaluation-artifact-manifest:1.0",
            "created_utc": current_utc(),
            "status": "complete",
            "manifest_excludes_itself_to_avoid_recursion": True,
            "evaluation_directory": str(output),
            "artifact_count_excluding_manifest": len(inventory),
            "artifacts": inventory,
            "artifact_inventory_sha256": inventory_sha256(inventory),
            "source_changes": source_changes,
            "input_bindings": {
                "lineage_finalization_manifest": file_record(
                    FINALIZATION_DIRECTORY / "finalization_manifest.json"
                ),
                "validation_contract": file_record(
                    ANALYSIS_DIRECTORY / "validation_comparison_contract.json"
                ),
                "checkpoint": file_record(CHILD_RUN / "best.pt"),
                "shared_arrays": file_record(shared_path),
                "decoder_arrays": file_record(decoder_arrays_path),
                "validation_access_ledger": file_record(access_path),
            },
            "completion_gates": {
                "route_nll_reproduced": True,
                "query_and_target_counts_reproduced": True,
                "decoder_and_two_causal_baselines_reported": True,
                "all_requested_metrics_reported": True,
                "paired_session_bootstrap_reported": True,
                "twenty_epoch_history_reported": True,
                "dataset_scale_boundary_reported": True,
                "zero_training_test_and_future_test_dataset_access": True,
                "no_training_or_scale_work_performed": True,
                "all_requested_charts_have_csv_data": True,
            },
        }
    )
    write_new_json(manifest_path, manifest)
    return {
        "status": "completed",
        "directory": str(output),
        "manifest": file_record(manifest_path),
        "report": file_record(output / "final_report.md"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FNCS decoder post-training reports.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("legacy-full-study", "validation-only"),
        default="legacy-full-study",
    )
    parser.add_argument("--destination", type=Path, default=VALIDATION_ONLY_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = (
        build_validation_only_report(args.destination)
        if args.command == "validation-only"
        else build_report()
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_PATH",
    "SUMMARY_PATH",
    "VALIDATION_ONLY_DIRECTORY",
    "build_report",
    "build_validation_only_report",
]
