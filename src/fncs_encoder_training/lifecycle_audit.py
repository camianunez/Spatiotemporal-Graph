from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq
import torch

from fortnite_encoder.config import EncoderConfig, RotationHeadConfig
from fortnite_encoder.contracts import RotationSupervision, TensorizedMatch
from fortnite_encoder.supervision import (
    _PLAYER_EVENT_SCHEMA,
)
from fortnite_encoder.tensorize import _INPUT_SCHEMAS, load_match_session
from fortnite_encoder.training import (
    WindowRequest,
    collate_training_windows,
    training_window_requests,
)
from fortnite_encoder.world_grid import WorldGridProfile

from .common import (
    atomic_write_json,
    canonical_json,
    file_sha256,
    source_tree_manifest,
    utc_now,
)
from .config import FreshEncoderConfig, load_config
from .data_access import AuditedSessionRepository, DataAccessLedger
from .dataset import load_bound_inventory, load_bound_split
from .lifecycle import (
    LEGACY_FALSE_POSITIVE,
    validate_lifecycle_context,
)
from .tensorize_compatibility import validate_zone_timing_context
from .runtime import _records


ENCODER_DIGEST = "a0eb81ae908cfe5dd7c48da0834e129312687f59f80c8c074a983d612944f56e"
REQUIRED_ARTIFACTS = (
    "lifecycle_failure_reproduction.json",
    "lifecycle_contract_analysis.json",
    "offending_rows.json",
    "tensorizer_source_difference.json",
    "tensorization_audit.json",
    "deterministic_pass_comparison.json",
    "relaunch_decision.json",
    "data_access_ledger.json",
    "artifact_manifest.json",
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _table_row_count(item: Mapping[str, Any], name: str) -> int:
    for table in item["tables"]:
        if str(table["path"]).endswith("/" + name):
            return int(table["row_count"])
    raise ValueError(f"inventory session {item['game_session_id']} has no {name}")


def _load_inputs(
    config_path: Path,
) -> tuple[
    FreshEncoderConfig,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    WorldGridProfile,
    tuple[Any, ...],
    dict[str, str],
    dict[str, int],
]:
    config = load_config(config_path)
    preflight = Path(config.preflight_directory)
    resolved = json.loads(
        (preflight / "resolved_training_config.json").read_text(encoding="utf-8")
    )
    bindings = resolved["bindings"]
    inventory = load_bound_inventory(
        preflight / "dataset_inventory.json",
        bindings["dataset_validation_sha256"],
    )
    split = load_bound_split(
        preflight / "split_manifest.json",
        bindings["split_manifest_sha256"],
    )
    profile = WorldGridProfile.load(
        config.world_grid_profile,
        expected_hash=config.expected_world_grid_profile_hash,
    )
    records, partition_by_session = _records(
        inventory,
        split,
        dataset_root=Path(config.dataset_root),
        profile=profile,
    )
    inventory_by_id = {
        item["game_session_id"]: item for item in inventory["sessions"]
    }
    lengths = {
        session_id: _table_row_count(inventory_by_id[session_id], "match_samples.parquet")
        for partition in ("train", "validation")
        for session_id in split["ordered_session_ids"][partition]
    }
    return (
        config,
        resolved,
        inventory,
        split,
        profile,
        records,
        partition_by_session,
        lengths,
    )


def _head_config(resolved: Mapping[str, Any]) -> RotationHeadConfig:
    values = dict(resolved["head_config"])
    values["horizons_seconds"] = tuple(values["horizons_seconds"])
    return RotationHeadConfig(**values)


def _eligible_requests(
    *,
    split: Mapping[str, Any],
    lengths: Mapping[str, int],
    window_length: int,
    stride: int,
) -> tuple[tuple[WindowRequest, str], ...]:
    output: list[tuple[WindowRequest, str]] = []
    for partition in ("train", "validation"):
        for session_id in split["ordered_session_ids"][partition]:
            length = lengths[session_id]
            if length <= 0:
                raise ValueError(f"session {session_id} has no sampled ticks")
            if length <= window_length:
                starts = (0,)
            else:
                starts = range(0, length - window_length + 1, stride)
            for start in starts:
                output.append(
                    (
                        WindowRequest(
                            session_id,
                            start,
                            min(window_length, length - start),
                        ),
                        partition,
                    )
                )
    return tuple(output)


def _batch(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _update_text(hasher: Any, value: Any) -> None:
    raw = canonical_json(value).encode("utf-8")
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)


def _update_tensor(hasher: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    _update_text(
        hasher,
        {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
    )
    raw = value.numpy().tobytes(order="C")
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)


def _hash_collated(
    *,
    encoder: Any,
    supervision: Any,
    tensor_hasher: Any,
    mask_hasher: Any,
    target_hasher: Any,
) -> None:
    for field in fields(encoder.batch):
        value = getattr(encoder.batch, field.name)
        destination = mask_hasher if value.dtype == torch.bool else tensor_hasher
        _update_tensor(destination, "encoder." + field.name, value)
    for field in fields(supervision.targets):
        value = getattr(supervision.targets, field.name)
        if value is None:
            _update_text(target_hasher, {"name": "target." + field.name, "value": None})
            continue
        destination = mask_hasher if value.dtype == torch.bool else target_hasher
        _update_tensor(destination, "target." + field.name, value)


def _equal_field(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    return left == right


def _validate_window(
    *,
    request: WindowRequest,
    match: TensorizedMatch,
    supervision: RotationSupervision,
    full_match: TensorizedMatch,
    full_supervision: RotationSupervision,
    head_config: RotationHeadConfig,
) -> dict[str, int]:
    if match.num_timesteps != request.length:
        raise ValueError("window tensor length does not match its request")
    if int(match.absolute_tick_index[0]) != request.start_tick:
        raise ValueError("window absolute start tick changed")
    start = request.start_tick
    stop = start + request.length
    for name in (
        "player_xyz_uu",
        "player_alive",
        "player_coord_mask",
        "life_index",
        "current_circle_uu",
        "target_circle_uu",
        "zone_mask",
        "zone_phase",
        "phase_times_s",
        "match_elapsed_s",
        "players_remaining",
        "teams_remaining",
        "absolute_tick_index",
        "time_mask",
    ):
        if not torch.equal(getattr(match, name).cpu(), getattr(full_match, name)[start:stop].cpu()):
            raise ValueError(f"window slicing changed encoder field {name}")
    for name in ("player_slot_mask", "team_slot_mask"):
        if not torch.equal(getattr(match, name).cpu(), getattr(full_match, name).cpu()):
            raise ValueError(f"window slicing changed encoder field {name}")
    if match.session_id != full_match.session_id or match.team_ids != full_match.team_ids:
        raise ValueError("window slicing changed encoder metadata")
    for field in fields(supervision.targets):
        actual = getattr(supervision.targets, field.name)
        full_value = getattr(full_supervision.targets, field.name)
        expected = full_value[start:stop] if full_value is not None else None
        if not _equal_field(actual, expected):
            raise ValueError(f"window slicing changed target field {field.name}")
    if not torch.equal(
        supervision.absolute_tick_index.cpu(),
        full_supervision.absolute_tick_index[start:stop].cpu(),
    ) or supervision.metadata != full_supervision.metadata:
        raise ValueError("window slicing changed target metadata")

    floating_fields = 0
    for field in fields(match):
        value = getattr(match, field.name)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            floating_fields += 1
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"window contains non-finite encoder field {field.name}")
    if bool((match.player_coord_mask & ~match.player_alive).any()):
        raise ValueError("a dead player has a coordinate mask")
    if bool(
        (
            match.player_alive
            & ~match.player_slot_mask.unsqueeze(0)
        ).any()
    ):
        raise ValueError("an unoccupied roster slot is marked alive")
    if not bool(match.time_mask.all()) or not bool(match.team_slot_mask.all()):
        raise ValueError("an unpadded window contains a false time/team mask")

    if request.start_tick == 0:
        if bool(match.prior_state_available):
            raise ValueError("first window invented prior lifecycle state")
    else:
        prior_tick = request.start_tick - 1
        if not bool(match.prior_state_available):
            raise ValueError("noninitial window lost prior lifecycle state")
        for name in (
            "player_xyz_uu",
            "player_alive",
            "player_coord_mask",
            "life_index",
        ):
            if not torch.equal(
                getattr(match, "prior_" + name).cpu(),
                getattr(full_match, name)[prior_tick].cpu(),
            ):
                raise ValueError(f"window prior sidecar changed {name}")

    query = (
        match.player_alive
        & match.player_slot_mask.unsqueeze(0)
    ).any(dim=-1)
    query_count = int(query.sum())
    if query_count <= 0:
        raise ValueError("window has no eligible model query")
    targets = supervision.targets
    masks = [targets.future_position_mask.any(dim=-1), targets.zone_entry_mask]
    if targets.survival_mask is not None:
        masks.append(targets.survival_mask)
    if targets.placement_mask is not None:
        masks.append(targets.placement_mask)
    for mask in masks:
        if bool((mask & ~query).any()):
            raise ValueError("a loss target is marked valid for an ineligible query")
    available = torch.zeros_like(query)
    for mask in masks:
        available |= mask
    loss_target_count = int((available & query).sum())
    if loss_target_count <= 0:
        raise ValueError("window has no available loss target")

    position_mask = targets.future_position_mask
    if position_mask.any():
        labels = targets.future_position[position_mask]
        if int(labels.min()) < 0 or int(labels.max()) >= head_config.num_position_classes:
            raise ValueError("world-grid target mapping is outside the class contract")

    first_alive = int(
        (
            match.player_alive[0]
            & match.player_slot_mask
        ).sum()
    )
    first_dead = int(match.player_slot_mask.sum()) - first_alive
    boundary_life_transitions = 0
    immediately_before_transitions = 0
    if request.start_tick > 0:
        boundary_life_transitions = int(
            (
                full_match.life_index[request.start_tick]
                != full_match.life_index[request.start_tick - 1]
            ).sum()
        )
    if request.start_tick > 1:
        immediately_before_transitions = int(
            (
                full_match.life_index[request.start_tick - 1]
                != full_match.life_index[request.start_tick - 2]
            ).sum()
        )
    return {
        "finite_encoder_field_checks": floating_fields,
        "query_eligible_count": query_count,
        "loss_target_available_count": loss_target_count,
        "first_tick_alive_players": first_alive,
        "first_tick_dead_players": first_dead,
        "life_transitions_at_first_visible_tick": boundary_life_transitions,
        "life_transitions_immediately_before_window": immediately_before_transitions,
    }


def _validate_full_lifecycle(
    *,
    session_id: str,
    directory: Path,
    ledger: DataAccessLedger,
) -> dict[str, Any]:
    for name in ("player_samples.parquet", "player_events.parquet"):
        ledger.attempt(session_id, name, "event_aware_lifecycle_validation")
    player_table = pq.read_table(directory / "player_samples.parquet")
    event_table = pq.read_table(directory / "player_events.parquet")
    for name in ("player_samples.parquet", "player_events.parquet"):
        ledger.opened(session_id, name, "event_aware_lifecycle_validation")
    if not player_table.schema.equals(
        _INPUT_SCHEMAS["player_samples.parquet"], check_metadata=False
    ):
        raise ValueError("player sample schema changed during lifecycle audit")
    if not event_table.schema.equals(_PLAYER_EVENT_SCHEMA, check_metadata=False):
        raise ValueError("player event schema changed during lifecycle audit")
    return validate_lifecycle_context(
        player_table.to_pylist(),
        event_table.to_pylist(),
    )


def _validate_full_zone_timing(
    *,
    session_id: str,
    directory: Path,
    ledger: DataAccessLedger,
) -> dict[str, Any]:
    for name in ("zone_phases.parquet", "zone_samples.parquet"):
        ledger.attempt(session_id, name, "producer_zone_timing_validation")
    phase_table = pq.read_table(directory / "zone_phases.parquet")
    sample_table = pq.read_table(directory / "zone_samples.parquet")
    for name in ("zone_phases.parquet", "zone_samples.parquet"):
        ledger.opened(session_id, name, "producer_zone_timing_validation")
    for name, table in (
        ("zone_phases.parquet", phase_table),
        ("zone_samples.parquet", sample_table),
    ):
        if not table.schema.equals(_INPUT_SCHEMAS[name], check_metadata=False):
            raise ValueError(f"{name} schema changed during zone timing audit")
    return validate_zone_timing_context(
        phase_table.to_pylist(),
        sample_table.to_pylist(),
    )


def _run_exhaustive_pass(
    *,
    index: int,
    output_directory: Path,
    config: FreshEncoderConfig,
    resolved: Mapping[str, Any],
    split: Mapping[str, Any],
    profile: WorldGridProfile,
    records: Sequence[Any],
    partition_by_session: Mapping[str, str],
    lengths: Mapping[str, int],
) -> dict[str, Any]:
    started = time.perf_counter()
    requests_with_partition = _eligible_requests(
        split=split,
        lengths=lengths,
        window_length=config.window_length_ticks,
        stride=config.window_stride_ticks,
    )
    requests = tuple(value[0] for value in requests_with_partition)
    partition_by_request = tuple(value[1] for value in requests_with_partition)
    train_ids = tuple(split["ordered_session_ids"]["train"])
    validation_ids = tuple(split["ordered_session_ids"]["validation"])
    test_ids = tuple(split["ordered_session_ids"]["test"])
    active = set(train_ids) | set(validation_ids)
    ledger_path = output_directory / f"data_access_ledger_pass_{index}.json"
    ledger = DataAccessLedger(
        path=ledger_path,
        partition_by_session=partition_by_session,
        test_session_ids=test_ids,
        split_manifest_sha256=split["split_manifest_sha256"],
        phase=f"exhaustive_loader_pass_{index}",
        allowed_partitions=("train", "validation"),
    )
    head_config = _head_config(resolved)
    repository = AuditedSessionRepository(
        tuple(record for record in records if record.session_id in active),
        profile,
        head_config,
        0,
        False,
        ledger,
    )
    by_record = {record.session_id: record for record in records}
    remaining = Counter(request.session_id for request in requests)
    lifecycle_validated: set[str] = set()
    context_hasher = hashlib.sha256()
    zone_context_hasher = hashlib.sha256()
    tensor_hasher = hashlib.sha256()
    mask_hasher = hashlib.sha256()
    target_hasher = hashlib.sha256()
    order_hasher = hashlib.sha256()
    batch_hasher = hashlib.sha256()
    counters: Counter[str] = Counter()
    affected_contexts: dict[str, dict[str, Any]] = {}
    zone_affected_contexts: dict[str, dict[str, Any]] = {}
    completed_sessions = 0
    try:
        for batch_index, request_batch in enumerate(
            _batch(requests, config.batch_size)
        ):
            for request in request_batch:
                if request.session_id in lifecycle_validated:
                    continue
                context = _validate_full_lifecycle(
                    session_id=request.session_id,
                    directory=by_record[request.session_id].path,
                    ledger=ledger,
                )
                _update_text(context_hasher, context)
                if context["collapsed_transition_count"]:
                    affected_contexts[request.session_id] = context
                zone_context = _validate_full_zone_timing(
                    session_id=request.session_id,
                    directory=by_record[request.session_id].path,
                    ledger=ledger,
                )
                _update_text(zone_context_hasher, zone_context)
                if zone_context["strict_mismatch_count"]:
                    zone_affected_contexts[request.session_id] = zone_context
                lifecycle_validated.add(request.session_id)

            items = repository.windows(request_batch)
            encoder, supervision = collate_training_windows(
                items,
                expected_profile_id=profile.profile_id,
                expected_profile_hash=profile.profile_hash,
                head_config=head_config,
            )
            request_payload = [
                {
                    "session_id": request.session_id,
                    "start_tick": request.start_tick,
                    "length": request.length,
                }
                for request in request_batch
            ]
            _update_text(batch_hasher, {"batch_index": batch_index, "requests": request_payload})
            for request in request_batch:
                _update_text(
                    order_hasher,
                    {
                        "session_id": request.session_id,
                        "start_tick": request.start_tick,
                        "length": request.length,
                    },
                )
            _hash_collated(
                encoder=encoder,
                supervision=supervision,
                tensor_hasher=tensor_hasher,
                mask_hasher=mask_hasher,
                target_hasher=target_hasher,
            )
            for request, item in zip(request_batch, items):
                full = repository.cache[request.session_id]
                values = _validate_window(
                    request=request,
                    match=item[0],
                    supervision=item[1],
                    full_match=full.match,
                    full_supervision=full.supervision,
                    head_config=head_config,
                )
                counters.update(values)
                counters["validated_windows"] += 1

            for request in request_batch:
                remaining[request.session_id] -= 1
                if remaining[request.session_id] == 0:
                    repository.cache.pop(request.session_id, None)
                    completed_sessions += 1
                    if completed_sessions % 10 == 0 or completed_sessions == len(active):
                        print(
                            json.dumps(
                                {
                                    "event": "exhaustive_pass_progress",
                                    "pass_index": index,
                                    "completed_sessions": completed_sessions,
                                    "total_sessions": len(active),
                                    "validated_windows": counters["validated_windows"],
                                    "affected_sessions": len(affected_contexts),
                                    "zone_timing_affected_sessions": len(
                                        zone_affected_contexts
                                    ),
                                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        if lifecycle_validated != active or completed_sessions != len(active):
            raise ValueError("exhaustive pass did not validate every active session")
        if set(repository.lifecycle_recoveries) != set(affected_contexts):
            raise ValueError("legacy failures and event-proven recoveries differ")
        if set(repository.zone_timing_recoveries) != set(zone_affected_contexts):
            raise ValueError(
                "legacy zone failures and producer-proven recoveries differ"
            )
        ledger.persist()
        access = ledger.report()
        if not access["zero_test_file_attempts"] or not access["zero_test_file_opens"]:
            raise ValueError("sealed test access occurred during exhaustive audit")
        partition_counts = Counter(partition_by_request)
        result = {
            "schema_version": "fncs-exhaustive-tensorization-pass:1.0",
            "created_utc": utc_now(),
            "pass_index": index,
            "status": "passed",
            "session_counts": {
                "train": len(train_ids),
                "validation": len(validation_ids),
                "test_sealed": len(test_ids),
            },
            "eligible_window_definition": {
                "window_length_ticks": config.window_length_ticks,
                "window_stride_ticks": config.window_stride_ticks,
                "full_stride_aligned_windows_only": True,
                "session_order": "split train order then split validation order",
                "window_order_within_session": "ascending start tick",
            },
            "eligible_window_counts": dict(sorted(partition_counts.items())),
            "eligible_window_count": len(requests),
            "batch_size": config.batch_size,
            "batch_count": (len(requests) + config.batch_size - 1) // config.batch_size,
            "window_order_sha256": order_hasher.hexdigest(),
            "batch_order_sha256": batch_hasher.hexdigest(),
            "tensor_sha256": tensor_hasher.hexdigest(),
            "mask_sha256": mask_hasher.hexdigest(),
            "target_sha256": target_hasher.hexdigest(),
            "lifecycle_context_sha256": context_hasher.hexdigest(),
            "zone_timing_context_sha256": zone_context_hasher.hexdigest(),
            "failure_count": 0,
            "validation_counters": dict(sorted(counters.items())),
            "affected_session_count": len(affected_contexts),
            "affected_sessions": [
                affected_contexts[session_id]
                for session_id in sorted(affected_contexts)
            ],
            "recovery_reports": [
                repository.lifecycle_recoveries[session_id]
                for session_id in sorted(repository.lifecycle_recoveries)
            ],
            "zone_timing_affected_session_count": len(zone_affected_contexts),
            "zone_timing_affected_sessions": [
                zone_affected_contexts[session_id]
                for session_id in sorted(zone_affected_contexts)
            ],
            "zone_timing_recovery_reports": [
                repository.zone_timing_recoveries[session_id]
                for session_id in sorted(repository.zone_timing_recoveries)
            ],
            "data_access": access,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(output_directory / f"tensorization_pass_{index}.json", result)
        return result
    finally:
        repository.close()


def _replay_step_56(
    *,
    output_directory: Path,
    config: FreshEncoderConfig,
    resolved: Mapping[str, Any],
    split: Mapping[str, Any],
    profile: WorldGridProfile,
    records: Sequence[Any],
    partition_by_session: Mapping[str, str],
    lengths: Mapping[str, int],
) -> dict[str, Any]:
    requests = training_window_requests(
        lengths,
        split["ordered_session_ids"]["train"],
        seed=config.seed,
        epoch=0,
        window_length_ticks=config.window_length_ticks,
        window_stride_ticks=config.window_stride_ticks,
    )
    batches = tuple(_batch(requests, config.batch_size))
    test_ids = tuple(split["ordered_session_ids"]["test"])
    ledger = DataAccessLedger(
        path=output_directory / "data_access_ledger_step_56_replay.json",
        partition_by_session=partition_by_session,
        test_session_ids=test_ids,
        split_manifest_sha256=split["split_manifest_sha256"],
        phase="deterministic_step_56_replay",
        allowed_partitions=("train",),
    )
    head_config = _head_config(resolved)
    replay_ids = {
        request.session_id
        for batch_index in range(57)
        for request in batches[batch_index]
    }
    repository = AuditedSessionRepository(
        tuple(record for record in records if record.session_id in replay_ids),
        profile,
        head_config,
        0,
        False,
        ledger,
    )
    try:
        batch_56_hash = hashlib.sha256()
        for batch_index in range(57):
            request_batch = batches[batch_index]
            items = repository.windows(request_batch)
            encoder, supervision = collate_training_windows(
                items,
                expected_profile_id=profile.profile_id,
                expected_profile_hash=profile.profile_hash,
                head_config=head_config,
            )
            if batch_index == 56:
                tensor_hash = hashlib.sha256()
                mask_hash = hashlib.sha256()
                target_hash = hashlib.sha256()
                _hash_collated(
                    encoder=encoder,
                    supervision=supervision,
                    tensor_hasher=tensor_hash,
                    mask_hasher=mask_hash,
                    target_hasher=target_hash,
                )
                _update_text(
                    batch_56_hash,
                    {
                        "tensor": tensor_hash.hexdigest(),
                        "mask": mask_hash.hexdigest(),
                        "target": target_hash.hexdigest(),
                    },
                )
            for request in request_batch:
                repository.cache.pop(request.session_id, None)
        ledger.persist()
        access = ledger.report()
        if not access["zero_test_file_attempts"] or not access["zero_test_file_opens"]:
            raise ValueError("sealed test access occurred during step-56 replay")
        failing_batch = batches[56]
        failing_session = "f72eac611fd24b94a7ad2ec9c667739f"
        if failing_session not in repository.lifecycle_recoveries:
            raise ValueError("the original failing session was not recovered at batch 56")
        return {
            "status": "passed",
            "seed": config.seed,
            "epoch": 0,
            "replayed_batch_index_range_inclusive": [0, 56],
            "replayed_batch_count": 57,
            "batch_56_requests": [
                {
                    "session_id": request.session_id,
                    "start_tick": request.start_tick,
                    "end_tick_inclusive": request.start_tick + request.length - 1,
                    "length": request.length,
                }
                for request in failing_batch
            ],
            "previously_failing_session": failing_session,
            "previously_failing_batch_tensorized": True,
            "batch_56_combined_hash": batch_56_hash.hexdigest(),
            "optimizer_updates_performed": 0,
            "test_access": {
                "attempts": access["test_session_parquet_attempt_count"],
                "opens": access["test_session_parquet_open_count"],
            },
            "data_access": access,
        }
    finally:
        repository.close()


def _prepare_failure_artifacts(
    *,
    output_directory: Path,
    workspace: Path,
    config: FreshEncoderConfig,
    resolved: Mapping[str, Any],
    inventory: Mapping[str, Any],
    split: Mapping[str, Any],
    lengths: Mapping[str, int],
) -> None:
    requests = training_window_requests(
        lengths,
        split["ordered_session_ids"]["train"],
        seed=config.seed,
        epoch=0,
        window_length_ticks=config.window_length_ticks,
        window_stride_ticks=config.window_stride_ticks,
    )
    batch_56 = tuple(_batch(requests, config.batch_size))[56]
    inventory_by_id = {
        item["game_session_id"]: item for item in inventory["sessions"]
    }
    reproductions: list[dict[str, Any]] = []
    selected_session = None
    for attempt in range(1, 3):
        selected = None
        for offset, request in enumerate(batch_56):
            directory = Path(config.dataset_root) / inventory_by_id[request.session_id]["directory"]
            try:
                load_match_session(directory)
            except Exception as exc:
                selected = {
                    "batch_offset": offset,
                    "session_id": request.session_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                break
        if selected is None:
            raise ValueError("legacy batch-56 failure did not reproduce")
        reproductions.append({"attempt": attempt, **selected})
        selected_session = selected["session_id"]
    if selected_session != "f72eac611fd24b94a7ad2ec9c667739f":
        raise ValueError("deterministic reproduction selected an unexpected session")
    if any(item["message"] != LEGACY_FALSE_POSITIVE for item in reproductions):
        raise ValueError("deterministic reproduction produced a different exception")

    request = next(item for item in batch_56 if item.session_id == selected_session)
    session_item = inventory_by_id[selected_session]
    directory = Path(config.dataset_root) / session_item["directory"]
    player_rows = pq.read_table(directory / "player_samples.parquet").to_pylist()
    event_rows = pq.read_table(directory / "player_events.parquet").to_pylist()
    context = validate_lifecycle_context(player_rows, event_rows)
    transition = context["collapsed_transitions"][0]
    player_id = transition["player_id"]
    previous_time = float(transition["previous_sample_time_seconds"])
    current_time = float(transition["current_sample_time_seconds"])
    relevant_samples = [
        row
        for row in player_rows
        if row["player_id"] == player_id
        and previous_time - 5.0 <= float(row["match_time_seconds"]) <= current_time + 10.0
    ]
    relevant_events = [
        row
        for row in event_rows
        if row["player_id"] == player_id
        and previous_time - 5.0 <= float(row["match_time_seconds"]) <= current_time + 5.0
        and row["event_type"] in {"dbno", "death", "reboot"}
    ]
    match_rows = pq.read_table(directory / "match_samples.parquet").to_pylist()
    relevant_match = [
        row
        for row in match_rows
        if previous_time - 5.0 <= float(row["match_time_seconds"]) <= current_time + 10.0
    ]
    assignment = next(
        item for item in split["assignments"] if item["game_session_id"] == selected_session
    )
    table_hashes = {
        Path(table["path"]).name: table["sha256"] for table in session_item["tables"]
    }
    reproduction = {
        "schema_version": "fncs-lifecycle-failure-reproduction:1.0",
        "created_utc": utc_now(),
        "status": "reproduced",
        "optimizer_updates_performed": 0,
        "seed": config.seed,
        "epoch": 0,
        "zero_based_batch_index": 56,
        "failure_report_next_batch_index": 56,
        "batch_size": config.batch_size,
        "window_length_ticks": config.window_length_ticks,
        "window_stride_ticks": config.window_stride_ticks,
        "batch_requests": [
            {
                "session_id": item.session_id,
                "start_tick": item.start_tick,
                "end_tick_inclusive": item.start_tick + item.length - 1,
                "length": item.length,
            }
            for item in batch_56
        ],
        "reproductions": reproductions,
        "same_record_selected": len({item["session_id"] for item in reproductions}) == 1,
        "same_exception_selected": len({item["message"] for item in reproductions}) == 1,
        "canonical_session_id": selected_session,
        "source_replay_sha256": assignment["source_replay_sha256"],
        "player_id": player_id,
        "team_id": transition["team_id"],
        "window": {
            "start_tick": request.start_tick,
            "end_tick_inclusive": request.start_tick + request.length - 1,
            "start_time_seconds": request.start_tick * 5.0,
            "end_time_seconds": (request.start_tick + request.length - 1) * 5.0,
        },
        "offending_sample": {
            "absolute_tick": int(round(current_time / 5.0)),
            "timestamp_seconds": current_time,
            "life_index": transition["current_life_index"],
            "alive": transition["current_alive"],
        },
        "exact_events": transition["events"],
        "sample_is_outside_selected_window": int(round(current_time / 5.0)) < request.start_tick,
        "classification": "A_valid_dataset_v2_state_tensorizer_bug",
        "test_partition_evaluated": False,
        "test_parquet_attempts": 0,
        "test_parquet_opens": 0,
    }
    atomic_write_json(output_directory / "lifecycle_failure_reproduction.json", reproduction)
    atomic_write_json(
        output_directory / "offending_rows.json",
        {
            "schema_version": "fncs-lifecycle-offending-rows:1.0",
            "canonical_session_id": selected_session,
            "source_replay_sha256": assignment["source_replay_sha256"],
            "table_sha256": table_hashes,
            "player_samples": relevant_samples,
            "player_events": relevant_events,
            "match_samples": relevant_match,
        },
    )
    atomic_write_json(
        output_directory / "lifecycle_contract_analysis.json",
        {
            "schema_version": "fncs-lifecycle-contract-analysis:1.0",
            "created_utc": utc_now(),
            "classification": "A_valid_dataset_v2_state_tensorizer_bug",
            "authoritative_contract": {
                "document": "docs/DATASET_V2.md",
                "parser_timeline": "src/FortniteReplayParser.Core/Sampling/LifecycleTimeline.cs",
                "parser_sampler": "src/FortniteReplayParser.Core/Sampling/DatasetBuilder.cs",
                "rules": [
                    "player samples use a fixed five-second grid",
                    "exact death events set alive false immediately",
                    "completed reboot events increment life_index while dead and set alive true",
                    "multiple exact events can occur between adjacent samples",
                ],
            },
            "observed_event_sequence": transition["events"],
            "derived_state_at_585_seconds": {"alive": False, "life_index": 2},
            "defective_assumption": (
                "fortnite_encoder.tensorize.load_match_session inferred that every "
                "sample-visible life-index increase must itself be an alive sample"
            ),
            "why_assumption_is_false": (
                "the reboot began life 2 at 580.8084365315735 seconds and a death "
                "ended that life at 584.970492623231 seconds before the 585-second sample"
            ),
            "correction": (
                "FNCS orchestration catches only the pinned legacy exception, validates "
                "the complete per-player event stream between full-session samples, uses "
                "a disposable validator-only shadow representation, and restores the raw "
                "life-index tensor before slicing or collation"
            ),
            "dataset_republication_required": False,
            "validation_weakened": False,
        },
    )

    baseline_encoder = json.loads(
        (Path(config.preflight_directory) / "encoder_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    current_encoder = source_tree_manifest(workspace / "ml" / "src" / "fortnite_encoder")
    if current_encoder["source_tree_sha256"] != ENCODER_DIGEST:
        raise ValueError("protected encoder digest changed during lifecycle correction")
    baseline_training = json.loads(
        (
            Path(config.preflight_directory)
            / "training_orchestration_source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    current_training = source_tree_manifest(
        workspace / "ml" / "src" / "fncs_encoder_training"
    )
    old_entries = {item["path"]: item["sha256"] for item in baseline_training["entries"]}
    new_entries = {item["path"]: item["sha256"] for item in current_training["entries"]}
    changed = [
        {
            "path": path,
            "before_sha256": old_entries.get(path),
            "after_sha256": new_entries.get(path),
        }
        for path in sorted(set(old_entries) | set(new_entries))
        if old_entries.get(path) != new_entries.get(path)
    ]
    fixture = workspace / "ml" / "tests" / "fixtures" / "fncs_lifecycle_collapsed_reboot_death.json"
    atomic_write_json(
        output_directory / "tensorizer_source_difference.json",
        {
            "schema_version": "fncs-tensorizer-source-difference:1.0",
            "protected_encoder": {
                "before_sha256": baseline_encoder["source_tree_sha256"],
                "after_sha256": current_encoder["source_tree_sha256"],
                "unchanged": baseline_encoder["source_tree_sha256"]
                == current_encoder["source_tree_sha256"]
                == ENCODER_DIGEST,
                "changed_files": [],
            },
            "training_orchestration": {
                "before_sha256": baseline_training["source_tree_sha256"],
                "after_sha256": current_training["source_tree_sha256"],
                "changed_files": changed,
            },
            "exact_correction_scope": [
                "ml/src/fncs_encoder_training/lifecycle.py",
                "ml/src/fncs_encoder_training/data_access.py",
                "ml/src/fncs_encoder_training/lifecycle_audit.py",
            ],
            "regression_fixture": {
                "path": str(fixture),
                "sha256": file_sha256(fixture),
                "redacted": True,
            },
            "architecture_changed": False,
        },
    )


def _combine_access(
    output_directory: Path,
    pass_reports: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    reports = [*(item["data_access"] for item in pass_reports), replay["data_access"]]
    return {
        "schema_version": "fncs-lifecycle-recovery-data-access:1.0",
        "created_utc": utc_now(),
        "split_manifest_sha256": split["split_manifest_sha256"],
        "allowed_partitions": ["train", "validation"],
        "sealed_test_session_count": len(split["ordered_session_ids"]["test"]),
        "phases": reports,
        "totals": {
            "test_session_parquet_attempt_count": sum(
                item["test_session_parquet_attempt_count"] for item in reports
            ),
            "test_session_parquet_open_count": sum(
                item["test_session_parquet_open_count"] for item in reports
            ),
            "validation_session_parquet_attempt_count": sum(
                item["validation_session_parquet_attempt_count"] for item in reports
            ),
            "validation_session_parquet_open_count": sum(
                item["validation_session_parquet_open_count"] for item in reports
            ),
        },
        "zero_test_file_attempts": all(item["zero_test_file_attempts"] for item in reports),
        "zero_test_file_opens": all(item["zero_test_file_opens"] for item in reports),
        "test_partition_evaluated": False,
    }


def _run_pass_worker(
    index: int,
    config_path: str,
    output_directory: str,
) -> dict[str, Any]:
    (
        config,
        resolved,
        _inventory,
        split,
        profile,
        records,
        partition_by_session,
        lengths,
    ) = _load_inputs(Path(config_path))
    return _run_exhaustive_pass(
        index=index,
        output_directory=Path(output_directory),
        config=config,
        resolved=resolved,
        split=split,
        profile=profile,
        records=records,
        partition_by_session=partition_by_session,
        lengths=lengths,
    )


def run_audit(
    *,
    config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    workspace = config_path.parent.parent
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    failed_run_inventory_path = (
        workspace
        / "audit"
        / "encoder-fncs-pilot-20260817-fresh-run-2-inventory.json"
    )
    failed_run_inventory = json.loads(
        failed_run_inventory_path.read_text(encoding="utf-8")
    )
    if (
        failed_run_inventory.get("file_count") != 37
        or failed_run_inventory.get("sha256_coverage", {}).get("complete") is not True
        or failed_run_inventory.get("active_trainer_process_check", {}).get(
            "no_python_trainer_process_active"
        )
        is not True
    ):
        raise ValueError("failed run inventory is incomplete or a trainer remains active")
    atomic_write_json(output / "failed_run_inventory.json", failed_run_inventory)
    (
        config,
        resolved,
        inventory,
        split,
        profile,
        records,
        partition_by_session,
        lengths,
    ) = _load_inputs(config_path)
    _prepare_failure_artifacts(
        output_directory=output,
        workspace=workspace,
        config=config,
        resolved=resolved,
        inventory=inventory,
        split=split,
        lengths=lengths,
    )
    # The passes are independent processes over the same immutable inputs.  Each
    # preserves the exact deterministic request/batch order; running them
    # concurrently changes only wall-clock scheduling, not loader semantics.
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_1 = executor.submit(
            _run_pass_worker, 1, str(config_path), str(output)
        )
        future_2 = executor.submit(
            _run_pass_worker, 2, str(config_path), str(output)
        )
        pass_1 = future_1.result()
        pass_2 = future_2.result()
    compared = (
        "eligible_window_count",
        "batch_count",
        "window_order_sha256",
        "batch_order_sha256",
        "tensor_sha256",
        "mask_sha256",
        "target_sha256",
        "lifecycle_context_sha256",
        "zone_timing_context_sha256",
        "failure_count",
        "validation_counters",
        "affected_session_count",
        "zone_timing_affected_session_count",
    )
    comparison = {
        "schema_version": "fncs-deterministic-pass-comparison:1.0",
        "created_utc": utc_now(),
        "status": "passed",
        "compared_fields": {
            name: {
                "pass_1": pass_1[name],
                "pass_2": pass_2[name],
                "identical": pass_1[name] == pass_2[name],
            }
            for name in compared
        },
    }
    comparison["all_compared_fields_identical"] = all(
        item["identical"] for item in comparison["compared_fields"].values()
    )
    if not comparison["all_compared_fields_identical"]:
        raise ValueError("the two exhaustive loader passes are not deterministic")
    atomic_write_json(output / "deterministic_pass_comparison.json", comparison)

    replay = _replay_step_56(
        output_directory=output,
        config=config,
        resolved=resolved,
        split=split,
        profile=profile,
        records=records,
        partition_by_session=partition_by_session,
        lengths=lengths,
    )
    tensorization_audit = {
        "schema_version": "fncs-tensorization-audit:1.0",
        "created_utc": utc_now(),
        "status": "passed",
        "root_cause_classification": "A_valid_dataset_v2_state_tensorizer_bug",
        "pass_1": {
            key: value
            for key, value in pass_1.items()
            if key not in {"affected_sessions", "recovery_reports", "data_access"}
        },
        "pass_2": {
            key: value
            for key, value in pass_2.items()
            if key not in {"affected_sessions", "recovery_reports", "data_access"}
        },
        "affected_session_count": pass_1["affected_session_count"],
        "affected_sessions": pass_1["affected_sessions"],
        "affected_window_count": sum(
            1
            for request, _ in _eligible_requests(
                split=split,
                lengths=lengths,
                window_length=config.window_length_ticks,
                stride=config.window_stride_ticks,
            )
            if request.session_id
            in {item["session_id"] for item in pass_1["affected_sessions"]}
        ),
        "additional_gate_finding": {
            "classification": "A_valid_dataset_v2_state_tensorizer_bug",
            "legacy_exception": (
                "zone sample does not identify the currently active phase"
            ),
            "cause": (
                "producer uses a 0.001-second activation epsilon while the "
                "pinned tensorizer uses 0.000001 seconds"
            ),
            "affected_session_count": pass_1[
                "zone_timing_affected_session_count"
            ],
            "affected_sessions": pass_1["zone_timing_affected_sessions"],
            "producer_contract_mismatch_count": 0,
        },
        "step_56_replay": {
            key: value for key, value in replay.items() if key != "data_access"
        },
        "all_required_validations": {
            "lifecycle_transitions": True,
            "producer_zone_phase_timing": True,
            "finite_encoder_inputs": True,
            "masks": True,
            "target_ranges": True,
            "world_grid_mappings": True,
            "tensor_shapes": True,
            "query_eligibility": True,
            "loss_target_availability": True,
            "full_match_equals_window_slice": True,
            "window_boundaries_use_prior_full_session_state": True,
        },
        "test_partition_evaluated": False,
        "test_parquet_attempts": 0,
        "test_parquet_opens": 0,
    }
    atomic_write_json(output / "tensorization_audit.json", tensorization_audit)
    access = _combine_access(output, (pass_1, pass_2), replay, split)
    atomic_write_json(output / "data_access_ledger.json", access)
    atomic_write_json(
        output / "relaunch_decision.json",
        {
            "schema_version": "fncs-relaunch-decision:1.0",
            "created_utc": utc_now(),
            "decision": "pending_cuda_bf16_preflight",
            "run_2_resume_authorized": False,
            "run_3_authorized": False,
            "class_a_root_cause_proven": True,
            "regression_tests_required_before_authorization": True,
            "two_exhaustive_passes_passed": True,
            "deterministic_pass_comparison_passed": True,
            "step_56_replay_passed": True,
            "zero_test_access": True,
        },
    )
    result = {
        "status": "passed_pending_cuda_bf16_preflight",
        "output_directory": str(output),
        "affected_session_count": pass_1["affected_session_count"],
        "eligible_window_count_per_pass": pass_1["eligible_window_count"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps({"event": "lifecycle_audit_complete", **result}, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce and exhaustively audit the FNCS lifecycle failure."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args(argv)
    run_audit(
        config_path=arguments.config,
        output_directory=arguments.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
