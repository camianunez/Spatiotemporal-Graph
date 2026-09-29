from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch import Tensor

from fncs_encoder_training.common import tensor_state_sha256
from fortnite_encoder.contracts import EncoderBatch
from fortnite_encoder.planner_contracts import PlannerBatch, PlannerObservation
from fortnite_encoder.planner_policy import apply_planner_input_policy
from fortnite_encoder.planner_supervision import PlannerQuery, _count_at
from fortnite_encoder.tensorize import SAMPLE_INTERVAL_SECONDS, _FLOAT_TOLERANCE
from fortnite_parallel_trajectory.losses import route_mixture_nll
from fortnite_parallel_trajectory.targets import build_parallel_trajectory_targets
from fortnite_parallel_trajectory_fncs_training.binding import (
    assert_frozen_encoder,
    build_frozen_model,
    load_downstream_state,
)
from fortnite_parallel_trajectory_fncs_training.data import (
    FNCSParallelTrajectorySessionRepository,
)
from fortnite_parallel_trajectory_training.data import (
    collate_parallel_trajectory_query_examples,
    prepare_parallel_trajectory_query_examples,
)
from fortnite_parallel_trajectory_training.training import (
    fixed_validation_queries,
    resolve_precision,
    seed_everything,
)

from .bindings import load_lineage_setups
from .common import (
    ANALYSIS_DIRECTORY,
    CHILD_RUN,
    EXPECTED_ENCODER_STATE_SHA256,
    EXPECTED_FINAL_ROUTE_NLL,
    EXPECTED_FINAL_STEP,
    EXPECTED_FIXED_COMPUTE_STEP,
    EXPECTED_INITIAL_DOWNSTREAM_SHA256,
    HORIZONS_SECONDS,
    PARENT_RUN,
    REPOSITORY_ROOT,
    SEED,
    STEP1840_CHECKPOINT,
    atomic_replace,
    current_utc,
    file_record,
    file_sha256,
    payload_sha256,
    read_json,
    replace_json,
    require,
    verify_self_seal,
    write_new_json,
)
from .metrics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    HIT_RADII_CELLS,
    MetricContribution,
    bootstrap_estimate,
    bootstrap_paired_difference,
    decoder_mode_diagnostics,
    point_metrics,
    scalar_contribution,
)
from .scale import TRAINABLE_SIZES, load_trained_scale_model, scale_run_directory


VALIDATION_ARRAY_SCHEMA = "fncs-decoder-validation-arrays:1.0"
MODEL_LABELS = (
    "final_step9200",
    "n41_step1840",
    "n115_step1840",
    "n230_step1840",
    "n460_step1840",
    "n920_step1840",
)
MAX_QUERIES_PER_FORWARD = 4


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _query_payload(query: PlannerQuery) -> dict[str, Any]:
    return asdict(query)


def _query_indices(batch: PlannerBatch) -> list[tuple[int, int, int]]:
    rows = batch.query_mask.nonzero(as_tuple=False).cpu().tolist()
    batch_size = int(batch.encoder_batch.player_xyz_uu.shape[0])
    require(len(rows) == batch_size, "each validation item must contain one query")
    require([row[0] for row in rows] == list(range(batch_size)), "validation query order changed")
    return [(int(a), int(b), int(c)) for a, b, c in rows]


def _valid_living_centroid(
    batch: PlannerBatch, batch_index: int, time_index: int, team_index: int
) -> Tensor | None:
    encoder = batch.encoder_batch
    roster = encoder.player_slot_mask[batch_index, team_index]
    valid = (
        encoder.player_alive[batch_index, time_index, team_index]
        & encoder.player_coord_mask[batch_index, time_index, team_index]
        & roster
    )
    if not bool(valid.any()):
        return None
    coordinates = encoder.player_xyz_uu[
        batch_index, time_index, team_index, valid, :2
    ].detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(coordinates).all()):
        return None
    return coordinates.mean(dim=0)


def _same_lifecycle(
    batch: PlannerBatch,
    batch_index: int,
    earlier_time: int,
    query_time: int,
    team_index: int,
) -> bool:
    encoder = batch.encoder_batch
    roster = encoder.player_slot_mask[batch_index, team_index]
    return bool(
        torch.equal(
            encoder.player_alive[batch_index, earlier_time, team_index] & roster,
            encoder.player_alive[batch_index, query_time, team_index] & roster,
        )
        and torch.equal(
            encoder.life_index[batch_index, earlier_time, team_index, roster],
            encoder.life_index[batch_index, query_time, team_index, roster],
        )
    )


def derive_causal_baselines(
    planner_batch: PlannerBatch,
) -> tuple[Tensor, Tensor, Tensor, tuple[str | None, ...]]:
    """Derive static and uncapped two-observation constant-velocity forecasts."""

    encoder = planner_batch.encoder_batch
    require(encoder.player_xyz_uu.device.type == "cpu", "baseline inputs must remain on CPU")
    horizons = torch.tensor(HORIZONS_SECONDS, dtype=torch.float64)
    currents: list[Tensor] = []
    static_rows: list[Tensor] = []
    velocity_rows: list[Tensor] = []
    fallback_reasons: list[str | None] = []
    for batch_index, query_time, team_index in _query_indices(planner_batch):
        current = _valid_living_centroid(
            planner_batch, batch_index, query_time, team_index
        )
        require(current is not None, "eligible validation query lacks current position")
        static = current.repeat(len(HORIZONS_SECONDS), 1)
        previous: Tensor | None = None
        previous_seconds: float | None = None
        for earlier_time in range(query_time - 1, -1, -1):
            if not bool(encoder.time_mask[batch_index, earlier_time]):
                break
            if not _same_lifecycle(
                planner_batch,
                batch_index,
                earlier_time,
                query_time,
                team_index,
            ):
                break
            candidate = _valid_living_centroid(
                planner_batch, batch_index, earlier_time, team_index
            )
            if candidate is not None:
                previous = candidate
                previous_seconds = float(
                    encoder.match_elapsed_s[batch_index, earlier_time].item()
                )
                break
        query_seconds = float(encoder.match_elapsed_s[batch_index, query_time].item())
        velocity: Tensor | None = None
        if previous is not None and previous_seconds is not None:
            elapsed = query_seconds - previous_seconds
            if math.isfinite(elapsed) and elapsed > 0.0:
                candidate_velocity = (current - previous) / elapsed
                if bool(torch.isfinite(candidate_velocity).all()):
                    velocity = candidate_velocity
        currents.append(current)
        static_rows.append(static)
        if velocity is None:
            velocity_rows.append(static.clone())
            fallback_reasons.append("second_comparable_causal_observation_unavailable")
        else:
            velocity_rows.append(current + horizons[:, None] * velocity)
            fallback_reasons.append(None)
    return (
        torch.stack(currents),
        torch.stack(static_rows),
        torch.stack(velocity_rows),
        tuple(fallback_reasons),
    )


def _target_mask_reasons(source: Any, query: PlannerQuery) -> tuple[str, ...]:
    match = source.match
    team = match.team_ids.index(query.team_id)
    tick_matches = (
        match.absolute_tick_index == query.absolute_tick_index
    ).nonzero(as_tuple=False)
    require(tick_matches.numel() == 1, "validation query tick is not unique")
    tick = int(tick_matches.item())
    times = match.match_elapsed_s.cpu().double()
    query_time = float(times[tick].item())
    recording_end = float(times[-1].item())
    reasons: list[str] = []
    terminal_reason: str | None = None
    for horizon in HORIZONS_SECONDS:
        if terminal_reason is not None:
            reasons.append(f"terminal_suffix_after_{terminal_reason}")
            continue
        target_time = query_time + horizon
        if target_time > recording_end + _FLOAT_TOLERANCE:
            reasons.append("recording_ended")
            terminal_reason = "recording_ended"
            continue
        matches = torch.isclose(
            times,
            torch.tensor(target_time, dtype=torch.float64),
            rtol=0.0,
            atol=_FLOAT_TOLERANCE,
        ).nonzero(as_tuple=False)
        if matches.numel() == 0:
            reasons.append("missing_sample_timestamp")
            continue
        require(matches.numel() == 1, "validation target timestamp is duplicated")
        target_tick = int(matches.item())
        roster = match.player_slot_mask[team]
        sampled_living = int((match.player_alive[target_tick, team] & roster).sum().item())
        lifecycle_living = _count_at(source, team, target_time)
        if lifecycle_living is None or lifecycle_living != sampled_living:
            reasons.append("lifecycle_ambiguity")
            terminal_reason = "lifecycle_ambiguity"
            continue
        if sampled_living <= 0:
            reasons.append("team_eliminated")
            terminal_reason = "team_eliminated"
            continue
        phase = int(match.zone_phase[target_tick].item())
        if phase >= 9:
            reasons.append("phase_9_terminal")
            terminal_reason = "phase_9_terminal"
            continue
        if not 1 <= phase <= 8:
            reasons.append("missing_or_invalid_phase")
            continue
        if source.centroid_status[target_tick][team] != "valid":
            reasons.append("missing_coordinate")
            continue
        xy = source.centroids_xy_uu[target_tick, team].double()
        if not bool(torch.isfinite(xy).all()):
            reasons.append("nonfinite_coordinate")
            continue
        if source.profile.cell(xy.tolist()) is None:
            reasons.append("out_of_envelope")
            continue
        reasons.append("valid")
    return tuple(reasons)


def _load_contract() -> dict[str, Any]:
    path = ANALYSIS_DIRECTORY / "validation_comparison_contract.json"
    contract = read_json(path, "validation comparison contract")
    verify_self_seal(contract, "contract_payload_sha256")
    require(
        contract.get("frozen_sources", {}).get("evaluator_sha256")
        == file_sha256(Path(__file__)),
        "validation evaluator changed after contract publication",
    )
    frozen_files = contract.get("frozen_sources", {}).get("files")
    require(isinstance(frozen_files, list) and frozen_files, "validation frozen-source list is missing")
    for record in frozen_files:
        require(isinstance(record, Mapping), "validation frozen-source record is invalid")
        relative = record.get("path")
        expected = record.get("sha256")
        require(isinstance(relative, str) and isinstance(expected, str), "validation frozen-source binding is invalid")
        source = (REPOSITORY_ROOT / relative).resolve()
        require(source.is_relative_to(REPOSITORY_ROOT.resolve()), "validation frozen-source path escaped the repository")
        require(file_sha256(source) == expected, f"validation frozen source changed: {relative}")
    require(
        contract.get("selected_checkpoint", {}).get("sha256")
        == file_sha256(CHILD_RUN / "best.pt"),
        "selected best.pt changed after validation contract publication",
    )
    require(
        contract.get("validation_population", {}).get("session_count") == 115
        and contract.get("validation_population", {}).get("queries_per_session") == 32,
        "validation population contract changed",
    )
    return contract


def _evaluation_status() -> dict[str, Any]:
    path = ANALYSIS_DIRECTORY / "evaluation_status.json"
    if path.is_file():
        return read_json(path, "evaluation status")
    return {
        "schema_version": "fncs-decoder-validation-evaluation-status:1.0",
        "created_utc": current_utc(),
        "updated_utc": current_utc(),
        "baseline_derivation": {"attempts": 0, "completed": False},
        "models": {
            label: {"attempts": 0, "completed": False} for label in MODEL_LABELS
        },
        "test_partition_evaluated": False,
        "test_access_attempts": 0,
        "test_parquet_opens": 0,
    }


def _save_evaluation_status(status: Mapping[str, Any]) -> None:
    value = dict(status)
    value["updated_utc"] = current_utc()
    replace_json(ANALYSIS_DIRECTORY / "evaluation_status.json", value)


def _build_validation_context() -> tuple[Any, Any, dict[str, Any], tuple[PlannerQuery, ...]]:
    current_setup, checkpoint_setup, _, _ = load_lineage_setups()
    ledger_path = ANALYSIS_DIRECTORY / "data_access_validation.json"
    repository = FNCSParallelTrajectorySessionRepository(
        paths=current_setup.session_paths,
        partitions=current_setup.partition_by_session,
        active_partitions=("validation",),
        test_session_ids=current_setup.split_session_ids["test"],
        profile=current_setup.profile,
        ledger_path=ledger_path,
        split_manifest_sha256=current_setup.config.expected_split_manifest_sha256,
        phase="locked_validation_comparison",
        resume_ledger=ledger_path.is_file(),
    )
    validation_ids = current_setup.split_session_ids["validation"]
    queries = fixed_validation_queries(
        repository,
        validation_ids,
        seed=SEED,
        queries_per_session=32,
    )
    require(len(queries) == 115 * 32, "validation query count changed")
    sources = {session_id: repository.get(session_id) for session_id in validation_ids}
    repository.assert_no_test_access()
    return current_setup, checkpoint_setup, {"repository": repository, "sources": sources}, queries


def _shared_arrays_path() -> Path:
    return ANALYSIS_DIRECTORY / "raw_validation" / "shared_targets_and_baselines.npz"


def _model_arrays_path(label: str) -> Path:
    return ANALYSIS_DIRECTORY / "raw_validation" / f"{label}.npz"


def _derive_shared_arrays(
    *, setup: Any, sources: Mapping[str, Any], queries: Sequence[PlannerQuery]
) -> dict[str, Any]:
    status = _evaluation_status()
    path = _shared_arrays_path()
    if status["baseline_derivation"]["completed"] is True:
        require(path.is_file(), "completed baseline status lacks arrays")
        return {"status": "already_completed", "artifact": file_record(path)}
    status["baseline_derivation"]["attempts"] += 1
    _save_evaluation_status(status)
    session_rows: list[str] = []
    current_chunks: list[np.ndarray] = []
    truth_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    static_chunks: list[np.ndarray] = []
    velocity_chunks: list[np.ndarray] = []
    fallback_rows: list[bool] = []
    reason_rows: list[list[str]] = []
    manifest = [_query_payload(query) for query in queries]
    for start in range(0, len(queries), MAX_QUERIES_PER_FORWARD):
        chunk_queries = queries[start : start + MAX_QUERIES_PER_FORWARD]
        examples = prepare_parallel_trajectory_query_examples(
            sources,
            chunk_queries,
            context_length_ticks=setup.config.context_length_ticks,
        )
        require(len(examples) == len(chunk_queries), "validation query lost supervision")
        batch, observation, targets = collate_parallel_trajectory_query_examples(examples)
        sanitized = apply_planner_input_policy(batch, observation)
        current, static, velocity, fallbacks = derive_causal_baselines(sanitized)
        require(targets.current_xy is not None, "validation current positions are missing")
        require(targets.future_xy is not None, "validation future positions are missing")
        require(
            torch.equal(current.to(torch.float32), targets.current_xy),
            "baseline current positions differ from decoder targets",
        )
        for query, target_mask in zip(chunk_queries, targets.target_mask):
            reasons = _target_mask_reasons(sources[query.session_id], query)
            require(
                [reason == "valid" for reason in reasons] == target_mask.tolist(),
                "target-mask diagnostic classification differs from target builder",
            )
            reason_rows.append(list(reasons))
        session_rows.extend(query.session_id for query in chunk_queries)
        current_chunks.append(current.numpy().astype(np.float64))
        truth_chunks.append(targets.future_xy.numpy().astype(np.float64))
        mask_chunks.append(targets.target_mask.numpy().astype(np.bool_))
        static_chunks.append(static.numpy().astype(np.float64))
        velocity_chunks.append(velocity.numpy().astype(np.float64))
        fallback_rows.extend(value is not None for value in fallbacks)
    arrays = {
        "session_ids": np.asarray(session_rows, dtype="U32"),
        "current_xy_uu": np.concatenate(current_chunks),
        "truth_xy_uu": np.concatenate(truth_chunks),
        "target_mask": np.concatenate(mask_chunks),
        "static_xy_uu": np.concatenate(static_chunks),
        "constant_velocity_xy_uu": np.concatenate(velocity_chunks),
        "constant_velocity_fallback": np.asarray(fallback_rows, dtype=np.bool_),
        "target_mask_reasons": np.asarray(reason_rows, dtype="U64"),
    }
    require(arrays["target_mask"].any(axis=1).all(), "validation route has no target")
    _atomic_save_npz(path, **arrays)
    manifest_payload = {
        "schema_version": "fncs-decoder-validation-query-manifest:1.0",
        "sampler": "parallel-trajectory-validation-query-sample-v1",
        "seed": SEED,
        "queries_per_session": 32,
        "query_count": len(manifest),
        "queries": manifest,
        "query_manifest_sha256": payload_sha256(manifest),
        "shared_arrays": file_record(path),
    }
    manifest_path = ANALYSIS_DIRECTORY / "validation_query_manifest.json"
    if not manifest_path.exists():
        write_new_json(
            manifest_path, {"created_utc": current_utc(), **manifest_payload}
        )
    else:
        existing_manifest = read_json(manifest_path)
        existing_without_time = dict(existing_manifest)
        existing_without_time.pop("created_utc", None)
        require(
            existing_without_time == manifest_payload,
            "validation query manifest changed",
        )
    status = _evaluation_status()
    status["baseline_derivation"].update(
        {
            "completed": True,
            "completed_utc": current_utc(),
            "arrays_sha256": file_sha256(path),
            "query_manifest_sha256": payload_sha256(manifest),
            "constant_velocity_fallback_count": int(arrays["constant_velocity_fallback"].sum()),
        }
    )
    _save_evaluation_status(status)
    return {"status": "completed", "artifact": file_record(path)}


def _load_completed_model(
    *, checkpoint_path: Path, setup: Any, device: torch.device, expected_step: int
) -> torch.nn.Module:
    seed_everything(SEED)
    model, initial = build_frozen_model(setup, device=device)
    require(
        initial["initial_downstream_parameter_sha256"]
        == EXPECTED_INITIAL_DOWNSTREAM_SHA256,
        "decoder initialization binding changed",
    )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(raw, Mapping), "decoder checkpoint is not a mapping")
    state = raw.get("downstream_state")
    require(isinstance(state, Mapping), "decoder checkpoint downstream state is missing")
    require(
        tensor_state_sha256(state) == raw.get("downstream_state_sha256"),
        "decoder checkpoint embedded downstream hash is invalid",
    )
    require(
        raw.get("training_state", {}).get("optimizer_step") == expected_step,
        "decoder checkpoint optimizer step changed",
    )
    frozen = raw.get("frozen_encoder_snapshot")
    require(
        isinstance(frozen, Mapping)
        and frozen.get("encoder_state_sha256") == EXPECTED_ENCODER_STATE_SHA256
        and frozen.get("matches_canonical") is True,
        "decoder checkpoint frozen encoder changed",
    )
    require(
        raw.get("test_partition_evaluated") is False
        and raw.get("data_access", {}).get("zero_test_attempts") is True
        and raw.get("data_access", {}).get("zero_test_opens") is True,
        "decoder checkpoint records forbidden test access",
    )
    load_downstream_state(model, state)
    assert_frozen_encoder(model, setup, stage=f"locked_validation_step_{expected_step}")
    model.eval()
    return model


def _move_inputs(
    batch: EncoderBatch, observation: PlannerObservation, targets: Any, device: torch.device
) -> tuple[EncoderBatch, PlannerObservation, Any]:
    return batch.to(device), observation.to(device), targets.to(device)


def _evaluate_model(
    *,
    label: str,
    model: torch.nn.Module,
    setup: Any,
    sources: Mapping[str, Any],
    queries: Sequence[PlannerQuery],
) -> dict[str, Any]:
    status = _evaluation_status()
    destination = _model_arrays_path(label)
    if status["models"][label]["completed"] is True:
        require(destination.is_file(), f"completed {label} status lacks arrays")
        return {"status": "already_completed", "artifact": file_record(destination)}
    status["models"][label]["attempts"] += 1
    status["models"][label]["last_started_utc"] = current_utc()
    _save_evaluation_status(status)
    width = float(setup.profile.cell_width_world_units)
    height = float(setup.profile.cell_height_world_units)
    displacement_scale = np.asarray([width, height], dtype=np.float64)
    primary_chunks: list[np.ndarray] = []
    mode_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    route_nll_chunks: list[np.ndarray] = []
    route_nll_sum = 0.0
    valid_route_count = 0
    nonfinite_counts: Counter[str] = Counter()
    device = next(model.parameters()).device
    for start in range(0, len(queries), MAX_QUERIES_PER_FORWARD):
        chunk_queries = queries[start : start + MAX_QUERIES_PER_FORWARD]
        examples = prepare_parallel_trajectory_query_examples(
            sources,
            chunk_queries,
            context_length_ticks=setup.config.context_length_ticks,
        )
        require(len(examples) == len(chunk_queries), "validation model query lost supervision")
        batch, observation, targets = collate_parallel_trajectory_query_examples(examples)
        require(targets.current_xy is not None, "validation current coordinates are missing")
        batch_gpu, observation_gpu, targets_gpu = _move_inputs(
            batch, observation, targets, device
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            output = model(batch_gpu, observation_gpu, targets_gpu)
        require(torch.equal(output.query_mask, targets_gpu.query_mask), "output/target query masks differ")
        require(bool(output.query_mask.all()), "validation model produced an ineligible query")
        loss = route_mixture_nll(output, targets_gpu)
        require(loss.valid_route_count == len(chunk_queries), "route NLL lost a query")
        values = {
            "means": output.means.detach().float().cpu().numpy().astype(np.float64),
            "probabilities": output.mode_probabilities.detach().float().cpu().numpy().astype(np.float64),
            "scales": output.scales.detach().float().cpu().numpy().astype(np.float64),
            "correlations": output.correlations.detach().float().cpu().numpy().astype(np.float64),
            "route_nll": loss.per_route_nll.detach().float().cpu().numpy().astype(np.float64),
        }
        for name, value in values.items():
            nonfinite_counts[name] += int(np.count_nonzero(~np.isfinite(value)))
        require(not any(nonfinite_counts.values()), f"{label} produced nonfinite predictions")
        current = targets.current_xy.numpy().astype(np.float64)
        modes_world = current[:, None, None, :] + values["means"] * displacement_scale
        selected = np.argmax(values["probabilities"], axis=1)
        primary = modes_world[np.arange(len(chunk_queries)), selected]
        primary_chunks.append(primary)
        mode_chunks.append(modes_world)
        probability_chunks.append(values["probabilities"])
        route_nll_chunks.append(values["route_nll"])
        route_nll_sum += float(loss.per_route_nll.detach().float().sum().item())
        valid_route_count += loss.valid_route_count
    arrays = {
        "primary_xy_uu": np.concatenate(primary_chunks),
        "mode_xy_uu": np.concatenate(mode_chunks),
        "mode_probabilities": np.concatenate(probability_chunks),
        "route_nll": np.concatenate(route_nll_chunks),
        "exact_route_nll_reduction": np.asarray(
            [route_nll_sum / valid_route_count], dtype=np.float64
        ),
    }
    _atomic_save_npz(destination, **arrays)
    status = _evaluation_status()
    status["models"][label].update(
        {
            "completed": True,
            "completed_utc": current_utc(),
            "arrays_sha256": file_sha256(destination),
            "route_nll": float(arrays["exact_route_nll_reduction"][0]),
            "valid_route_count": valid_route_count,
            "nonfinite_prediction_counts": dict(nonfinite_counts),
        }
    )
    _save_evaluation_status(status)
    return {"status": "completed", "artifact": file_record(destination)}


def evaluate(
    label: str,
    *,
    _prepared: tuple[Any, Any, dict[str, Any], tuple[PlannerQuery, ...]] | None = None,
) -> dict[str, Any]:
    require(label in MODEL_LABELS, f"unknown validation model label: {label}")
    _load_contract()
    setup, checkpoint_setup, context, queries = (
        _build_validation_context() if _prepared is None else _prepared
    )
    repository = context["repository"]
    sources = context["sources"]
    shared = _derive_shared_arrays(setup=setup, sources=sources, queries=queries)
    spec = resolve_precision()
    if label.startswith("n"):
        for required_size in TRAINABLE_SIZES:
            required_status = read_json(
                scale_run_directory(required_size) / "status.json",
                f"N={required_size} scale status",
            )
            require(
                required_status.get("status")
                in {"trained_pending_locked_validation", "completed"}
                and required_status.get("optimizer_step")
                == EXPECTED_FIXED_COMPUTE_STEP,
                "all fresh scale endpoints must be durable before any scale validation",
            )
    if label == "final_step9200":
        model = _load_completed_model(
            checkpoint_path=CHILD_RUN / "best.pt",
            setup=setup,
            device=spec.device,
            expected_step=EXPECTED_FINAL_STEP,
        )
    elif label == "n920_step1840":
        model = _load_completed_model(
            checkpoint_path=STEP1840_CHECKPOINT,
            setup=checkpoint_setup,
            device=spec.device,
            expected_step=EXPECTED_FIXED_COMPUTE_STEP,
        )
    else:
        size = int(label.removeprefix("n").removesuffix("_step1840"))
        model, _ = load_trained_scale_model(size=size, device=spec.device)
    result = _evaluate_model(
        label=label,
        model=model,
        setup=setup,
        sources=sources,
        queries=queries,
    )
    repository.assert_no_test_access()
    if label.startswith("n") and label != "n920_step1840":
        size = int(label.removeprefix("n").removesuffix("_step1840"))
        run = scale_run_directory(size)
        scale_status = read_json(run / "status.json")
        require(
            scale_status.get("validation_evaluation_count") in {0, 1},
            "scale validation count is invalid",
        )
        if scale_status.get("validation_evaluation_count") == 0:
            scale_status.update(
                {
                    "status": "completed",
                    "validation_evaluation_count": 1,
                    "validation_completed_utc": current_utc(),
                    "validation_arrays": file_record(_model_arrays_path(label)),
                    "validation_route_nll": _evaluation_status()["models"][label]["route_nll"],
                }
            )
            replace_json(run / "status.json", scale_status)
    del model
    torch.cuda.empty_cache()
    return {"label": label, "shared": shared, "model": result}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"validation arrays are missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _metric_payload(
    prediction: np.ndarray, shared: Mapping[str, np.ndarray], setup: Any
) -> tuple[dict[str, Any], dict[str, MetricContribution]]:
    return point_metrics(
        prediction_xy_uu=prediction,
        truth_xy_uu=shared["truth_xy_uu"],
        target_mask=shared["target_mask"],
        session_ids=shared["session_ids"],
        meters_per_world_unit=float(setup.profile.world_unit_scale.meters_per_world_unit),
        cell_width_world_units=float(setup.profile.cell_width_world_units),
        cell_height_world_units=float(setup.profile.cell_height_world_units),
    )


def _comparison_cis(
    left: Mapping[str, MetricContribution], right: Mapping[str, MetricContribution]
) -> dict[str, Any]:
    return {
        key: bootstrap_paired_difference(
            left[key],
            right[key],
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        for key in _comparison_metric_keys()
    }


def _comparison_metric_keys(*, include_nll: bool = False) -> tuple[str, ...]:
    keys = [
        "ade_meters",
        "fde_60_meters",
        "last_valid_fde_meters",
        *[
            f"error_at_{horizon}s_meters" for horizon in HORIZONS_SECONDS
        ],
        *[
            f"hit_at_{horizon}s_within_{radius:g}_cells"
            for horizon in HORIZONS_SECONDS
            for radius in HIT_RADII_CELLS
        ],
        *[
            f"aggregate_through_{endpoint}s_hit_within_{radius:g}_cells"
            for endpoint in (15, 30, 45, 60)
            for radius in HIT_RADII_CELLS
        ],
    ]
    if include_nll:
        keys.insert(0, "route_nll")
    return tuple(keys)


def _estimate_cis(
    contributions: Mapping[str, MetricContribution], *, include_nll: bool = False
) -> dict[str, Any]:
    return {
        key: bootstrap_estimate(
            contributions[key],
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        for key in _comparison_metric_keys(include_nll=include_nll)
    }


def aggregate_results() -> dict[str, Any]:
    _load_contract()
    status = _evaluation_status()
    require(
        all(status["models"][label]["completed"] is True for label in MODEL_LABELS),
        "all six locked model evaluations must finish before aggregation",
    )
    setup, _, _, _ = load_lineage_setups()
    shared = _load_npz(_shared_arrays_path())
    model_arrays = {label: _load_npz(_model_arrays_path(label)) for label in MODEL_LABELS}
    static_metrics, static_contrib = _metric_payload(shared["static_xy_uu"], shared, setup)
    velocity_metrics, velocity_contrib = _metric_payload(
        shared["constant_velocity_xy_uu"], shared, setup
    )
    final_metrics, final_contrib = _metric_payload(
        model_arrays["final_step9200"]["primary_xy_uu"], shared, setup
    )
    reason_counts = Counter(shared["target_mask_reasons"].reshape(-1).tolist())
    elimination_reasons = {
        "team_eliminated",
        "phase_9_terminal",
        "terminal_suffix_after_team_eliminated",
        "terminal_suffix_after_phase_9_terminal",
    }
    baseline_payload = {
        "schema_version": "fncs-decoder-validation-baseline-results:1.0",
        "created_utc": current_utc(),
        "development_set_only": True,
        "validation_session_count": 115,
        "validation_query_count": int(shared["session_ids"].size),
        "valid_target_count": int(shared["target_mask"].sum()),
        "primary_probabilistic_metric": {
            "decoder_route_mixture_nll": float(
                model_arrays["final_step9200"]["exact_route_nll_reduction"][0]
            ),
            "deterministic_baseline_nll": "unavailable_no_preexisting_probabilistic_baseline_contract",
            "direct_nll_comparison_to_deterministic_baselines_performed": False,
        },
        "common_point_metrics": {
            "decoder_highest_probability_mode_mean": final_metrics,
            "static_position": static_metrics,
            "constant_velocity": velocity_metrics,
        },
        "paired_session_bootstrap": {
            "decoder_minus_static": _comparison_cis(final_contrib, static_contrib),
            "decoder_minus_constant_velocity": _comparison_cis(
                final_contrib, velocity_contrib
            ),
        },
        "session_bootstrap_uncertainty": {
            "decoder_highest_probability_mode_mean": _estimate_cis(final_contrib),
            "static_position": _estimate_cis(static_contrib),
            "constant_velocity": _estimate_cis(velocity_contrib),
        },
        "constant_velocity_fallback": {
            "count": int(shared["constant_velocity_fallback"].sum()),
            "query_count": int(shared["constant_velocity_fallback"].size),
            "fallback": "static_position",
        },
        "target_mask_diagnostics": {
            "reason_counts": dict(sorted(reason_counts.items())),
            "valid_target_count": reason_counts["valid"],
            "missing_target_mask_count": int(
                sum(
                    count
                    for reason, count in reason_counts.items()
                    if reason != "valid" and reason not in elimination_reasons
                )
            ),
            "elimination_or_phase9_mask_count": int(
                sum(reason_counts[reason] for reason in elimination_reasons)
            ),
        },
        "decoder_diagnostics": decoder_mode_diagnostics(
            mode_xy_uu=model_arrays["final_step9200"]["mode_xy_uu"],
            mode_probabilities=model_arrays["final_step9200"]["mode_probabilities"],
            truth_xy_uu=shared["truth_xy_uu"],
            target_mask=shared["target_mask"],
            meters_per_world_unit=float(
                setup.profile.world_unit_scale.meters_per_world_unit
            ),
            cell_width_world_units=float(setup.profile.cell_width_world_units),
            cell_height_world_units=float(setup.profile.cell_height_world_units),
        ),
        "artifacts": {
            "shared_arrays": file_record(_shared_arrays_path()),
            "decoder_arrays": file_record(_model_arrays_path("final_step9200")),
        },
        "terminology": "ADE, FDE, and trajectory hit rate are trajectory metrics, not classification accuracy.",
    }
    observed_final = baseline_payload["primary_probabilistic_metric"][
        "decoder_route_mixture_nll"
    ]
    require(
        math.isclose(observed_final, EXPECTED_FINAL_ROUTE_NLL, rel_tol=0.0, abs_tol=2e-6),
        "locked final route NLL does not reproduce checkpoint-selection validation",
    )

    scale_metrics: dict[str, Any] = {}
    scale_contributions: dict[str, dict[str, MetricContribution]] = {}
    for size in (41, 115, 230, 460, 920):
        label = f"n{size}_step1840"
        point, contribution = _metric_payload(
            model_arrays[label]["primary_xy_uu"], shared, setup
        )
        nll_values = model_arrays[label]["route_nll"]
        nll_contribution = scalar_contribution(
            session_ids=shared["session_ids"], values=nll_values
        )
        contribution = dict(contribution)
        contribution["route_nll"] = nll_contribution
        scale_contributions[str(size)] = contribution
        scale_metrics[str(size)] = {
            "training_session_count": size,
            "optimizer_step": 1840,
            "route_nll": float(
                model_arrays[label]["exact_route_nll_reduction"][0]
            ),
            "point_metrics": point,
            "session_bootstrap_uncertainty": _estimate_cis(
                contribution, include_nll=True
            ),
            "checkpoint": file_record(
                STEP1840_CHECKPOINT
                if size == 920
                else scale_run_directory(size) / "last.pt"
            ),
        }
    transitions: list[dict[str, Any]] = []
    sizes = (41, 115, 230, 460, 920)
    comparison_keys = _comparison_metric_keys(include_nll=True)
    for smaller, larger in zip(sizes, sizes[1:]):
        doubling = math.log2(larger / smaller)
        comparisons = {
            key: bootstrap_paired_difference(
                scale_contributions[str(larger)][key],
                scale_contributions[str(smaller)][key],
                seed=BOOTSTRAP_SEED,
                resamples=BOOTSTRAP_RESAMPLES,
            )
            for key in comparison_keys
        }
        transitions.append(
            {
                "from_training_sessions": smaller,
                "to_training_sessions": larger,
                "log2_session_ratio": doubling,
                "metrics": comparisons,
                "marginal_change_per_doubling": {
                    key: comparisons[key]["left_minus_right"] / doubling
                    for key in comparison_keys
                },
            }
        )
    subset_manifest = read_json(
        ANALYSIS_DIRECTORY / "nested_subset_manifest.json", "nested subset manifest"
    )
    scale_payload = {
        "schema_version": "fncs-decoder-fixed-compute-scale-results:1.0",
        "created_utc": current_utc(),
        "interpretation": (
            "Controlled learning-curve estimate from one initialization and one nested "
            "subset ordering; not a universal scaling law. Validation bootstrap intervals "
            "do not capture initialization or subset-selection uncertainty."
        ),
        "fixed_compute": {
            "optimizer_updates": 1840,
            "same_initialization": True,
            "same_original_absolute_learning_rate_history": True,
            "same_architecture_loss_masks_targets_and_encoder": True,
            "same_validation_population": True,
            "checkpoint_selection_performed": False,
        },
        "scale_points": scale_metrics,
        "successive_scale_changes": transitions,
        "nested_subset_counts": {
            size: {
                "unique_sessions": subset_manifest["subsets"][size]["unique_session_count"],
                "eligible_route_queries": subset_manifest["subsets"][size][
                    "eligible_route_query_count"
                ],
            }
            for size in map(str, sizes)
        },
        "bootstrap_scope_caveat": (
            "Paired session-level intervals quantify validation-session sampling only."
        ),
    }

    full_1840 = model_arrays["n920_step1840"]
    full_9200 = model_arrays["final_step9200"]
    full_1840_point, full_1840_contrib = _metric_payload(
        full_1840["primary_xy_uu"], shared, setup
    )
    full_9200_point, full_9200_contrib = _metric_payload(
        full_9200["primary_xy_uu"], shared, setup
    )
    full_1840_contrib = dict(full_1840_contrib)
    full_9200_contrib = dict(full_9200_contrib)
    full_1840_contrib["route_nll"] = scalar_contribution(
        session_ids=shared["session_ids"], values=full_1840["route_nll"]
    )
    full_9200_contrib["route_nll"] = scalar_contribution(
        session_ids=shared["session_ids"], values=full_9200["route_nll"]
    )
    metrics_summary = read_json(
        ANALYSIS_DIRECTORY.parents[1]
        / "audit"
        / f"{PARENT_RUN.name}-finalization-20260902"
        / "training_metrics_summary.json",
        "training metrics finalization",
    )
    optimization_payload = {
        "schema_version": "fncs-decoder-additional-optimization-results:1.0",
        "created_utc": current_utc(),
        "interpretation": (
            "Effect of additional optimization from step 1840 to step 9200 on the same "
            "920-session training set; this is not an effect of dataset size."
        ),
        "step_1840": {
            "route_nll": float(full_1840["exact_route_nll_reduction"][0]),
            "point_metrics": full_1840_point,
            "session_bootstrap_uncertainty": _estimate_cis(
                full_1840_contrib, include_nll=True
            ),
        },
        "step_9200": {
            "route_nll": float(full_9200["exact_route_nll_reduction"][0]),
            "point_metrics": full_9200_point,
            "session_bootstrap_uncertainty": _estimate_cis(
                full_9200_contrib, include_nll=True
            ),
        },
        "step_9200_minus_step_1840": {
            key: bootstrap_paired_difference(
                full_9200_contrib[key],
                full_1840_contrib[key],
                seed=BOOTSTRAP_SEED,
                resamples=BOOTSTRAP_RESAMPLES,
            )
            for key in comparison_keys
        },
        "validation_trajectory_all_20_epochs": metrics_summary[
            "validation_trajectory"
        ],
        "endpoint_trend": metrics_summary["endpoint_change"],
        "training_extension_authorized": False,
    }

    destinations = {
        "baseline_results.json": baseline_payload,
        "fixed_compute_scale_results.json": scale_payload,
        "optimization_progress_results.json": optimization_payload,
    }
    for name, payload in destinations.items():
        path = ANALYSIS_DIRECTORY / name
        require(not path.exists(), f"refusing to overwrite aggregate result: {path}")
        write_new_json(path, payload)
    return {
        "status": "completed",
        "artifacts": {
            name: file_record(ANALYSIS_DIRECTORY / name) for name in destinations
        },
    }


def evaluate_all() -> dict[str, Any]:
    _load_contract()
    prepared = _build_validation_context()
    results: dict[str, Any] = {}
    for label in MODEL_LABELS:
        results[label] = evaluate(label, _prepared=prepared)
    results["aggregation"] = aggregate_results()
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked validation-only decoder analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    one = subparsers.add_parser("evaluate", help="evaluate one locked checkpoint")
    one.add_argument("label", choices=MODEL_LABELS)
    subparsers.add_parser("evaluate-all", help="evaluate all locked checkpoints and aggregate")
    subparsers.add_parser("aggregate", help="aggregate already-completed evaluations")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate(args.label)
    elif args.command == "evaluate-all":
        result = evaluate_all()
    else:
        result = aggregate_results()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODEL_LABELS",
    "aggregate_results",
    "derive_causal_baselines",
    "evaluate",
    "evaluate_all",
]
