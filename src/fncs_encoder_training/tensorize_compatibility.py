from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, replace
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fortnite_encoder.config import RotationHeadConfig
from fortnite_encoder.contracts import TensorizedMatch
from fortnite_encoder.supervision import (
    _PLAYER_EVENT_SCHEMA,
    load_rotation_supervision,
)
from fortnite_encoder.tensorize import (
    _INPUT_SCHEMAS,
    _validate_circle_sample,
    load_match_session,
)
from fortnite_encoder.training import (
    TrainingConfigurationError,
    _FullSession,
    _SessionRecord,
)
from fortnite_encoder.world_grid import WorldGridProfile

from .lifecycle import (
    LEGACY_FALSE_POSITIVE,
    _normalized_sample_rows,
    _raw_life_tensor,
    validate_lifecycle_context,
)


LEGACY_ZONE_FALSE_POSITIVE = (
    "zone sample does not identify the currently active phase"
)
PRODUCER_TIMESTAMP_EPSILON_SECONDS = 0.001
PINNED_TENSORIZER_TIMESTAMP_EPSILON_SECONDS = 0.000001


class TensorizationCompatibilityError(ValueError):
    """A legacy-loader failure is not justified by the Dataset v2 contract."""


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TensorizationCompatibilityError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TensorizationCompatibilityError(f"{field} must be finite")
    return result


def _positive_phase(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TensorizationCompatibilityError(
            f"{field} must be a positive integer"
        )
    return value


def _active_phase(
    time_seconds: float,
    phases: Mapping[int, Mapping[str, Any]],
    epsilon_seconds: float,
) -> int:
    return max(
        (
            phase
            for phase, row in phases.items()
            if float(row["activation_time_seconds"])
            <= time_seconds + epsilon_seconds
        ),
        default=0,
    )


def validate_zone_timing_context(
    phase_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconcile sampled phases with the producer's declared 1 ms epsilon."""

    if not phase_rows:
        raise TensorizationCompatibilityError(
            "zone_phases.parquet must not be empty"
        )
    phases: dict[int, Mapping[str, Any]] = {}
    session_ids: set[str] = set()
    for row in phase_rows:
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise TensorizationCompatibilityError(
                "zone phase session_id must be nonempty"
            )
        phase = _positive_phase(row.get("zone_phase"), "zone_phase")
        if phase in phases:
            raise TensorizationCompatibilityError("zone phases must be unique")
        _finite(row.get("activation_time_seconds"), "activation_time_seconds")
        phases[phase] = row
        session_ids.add(session_id)
    if len(session_ids) != 1:
        raise TensorizationCompatibilityError("zone phases contain mixed session IDs")
    if sorted(phases) != list(range(1, len(phases) + 1)):
        raise TensorizationCompatibilityError(
            "zone phases must be consecutively numbered from one"
        )
    session_id = next(iter(session_ids))

    if not sample_rows:
        raise TensorizationCompatibilityError(
            "zone_samples.parquet must not be empty"
        )
    sample_times: set[float] = set()
    strict_mismatches: list[dict[str, Any]] = []
    for row in sample_rows:
        if row.get("session_id") != session_id:
            raise TensorizationCompatibilityError(
                "zone sample session_id is not aligned"
            )
        time_seconds = _finite(
            row.get("match_time_seconds"), "sample match_time_seconds"
        )
        if time_seconds in sample_times:
            raise TensorizationCompatibilityError(
                "zone samples contain a duplicate tick"
            )
        sample_times.add(time_seconds)
        actual_raw = row.get("zone_phase")
        actual = (
            0
            if actual_raw is None
            else _positive_phase(actual_raw, "sample zone_phase")
        )
        producer_phase = _active_phase(
            time_seconds,
            phases,
            PRODUCER_TIMESTAMP_EPSILON_SECONDS,
        )
        if actual != producer_phase:
            raise TensorizationCompatibilityError(
                "zone sample disagrees with the producer timestamp contract"
            )
        if actual > 0:
            try:
                _validate_circle_sample(dict(row), dict(phases[actual]), time_seconds)
            except ValueError as exc:
                raise TensorizationCompatibilityError(
                    f"zone sample geometry/countdown is invalid: {exc}"
                ) from exc
        pinned_phase = _active_phase(
            time_seconds,
            phases,
            PINNED_TENSORIZER_TIMESTAMP_EPSILON_SECONDS,
        )
        if actual != pinned_phase:
            activation = float(phases[actual]["activation_time_seconds"])
            delta = activation - time_seconds
            if not (
                PINNED_TENSORIZER_TIMESTAMP_EPSILON_SECONDS
                < delta
                <= PRODUCER_TIMESTAMP_EPSILON_SECONDS
            ):
                raise TensorizationCompatibilityError(
                    "phase mismatch is outside the producer-only epsilon interval"
                )
            strict_mismatches.append(
                {
                    "session_id": session_id,
                    "match_time_seconds": time_seconds,
                    "sample_zone_phase": actual,
                    "pinned_tensorizer_phase": pinned_phase,
                    "producer_phase": producer_phase,
                    "activation_time_seconds": activation,
                    "activation_minus_sample_seconds": delta,
                    "target": {
                        "x": row.get("target_x"),
                        "y": row.get("target_y"),
                        "z": row.get("target_z"),
                        "radius": row.get("target_radius"),
                    },
                    "time_until_closure_seconds": row.get(
                        "time_until_closure_seconds"
                    ),
                }
            )

    return {
        "schema_version": "fncs-zone-timing-context-validation:1.0",
        "session_id": session_id,
        "phase_count": len(phases),
        "sample_row_count": len(sample_rows),
        "producer_timestamp_epsilon_seconds": (
            PRODUCER_TIMESTAMP_EPSILON_SECONDS
        ),
        "pinned_tensorizer_timestamp_epsilon_seconds": (
            PINNED_TENSORIZER_TIMESTAMP_EPSILON_SECONDS
        ),
        "strict_mismatch_count": len(strict_mismatches),
        "strict_mismatches": strict_mismatches,
        "status": "valid",
    }


def _has_legacy_lifecycle_pattern(
    player_rows: Sequence[Mapping[str, Any]],
) -> bool:
    by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in player_rows:
        by_player[str(row["player_id"])].append(row)
    for rows in by_player.values():
        rows.sort(key=lambda row: float(row["match_time_seconds"]))
        for previous, current in zip(rows, rows[1:]):
            if (
                int(current["life_index"]) > int(previous["life_index"])
                and not bool(current["alive"])
            ):
                return True
    return False


def _normalized_phase_rows(
    phase_rows: Sequence[Mapping[str, Any]],
    zone_context: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    replacement_by_phase: dict[int, float] = {}
    for mismatch in zone_context["strict_mismatches"]:
        phase = int(mismatch["sample_zone_phase"])
        tick = float(mismatch["match_time_seconds"])
        replacement_by_phase[phase] = min(
            tick,
            replacement_by_phase.get(phase, tick),
        )
    normalized = [dict(row) for row in phase_rows]
    by_phase = {int(row["zone_phase"]): row for row in normalized}
    changed = 0
    for phase, tick in replacement_by_phase.items():
        current = by_phase[phase]
        if float(current["activation_time_seconds"]) != tick:
            current["activation_time_seconds"] = tick
            changed += 1
        if phase > 1:
            previous = by_phase[phase - 1]
            if float(previous["closure_time_seconds"]) != tick:
                previous["closure_time_seconds"] = tick
                changed += 1
    return normalized, changed


def _raw_phase_times_tensor(
    phase_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    match: TensorizedMatch,
) -> torch.Tensor:
    phases = {int(row["zone_phase"]): row for row in phase_rows}
    time_index = {
        float(value): index
        for index, value in enumerate(match.match_elapsed_s.cpu().tolist())
    }
    result = torch.zeros_like(match.phase_times_s)
    for row in sample_rows:
        phase_raw = row["zone_phase"]
        if phase_raw is None:
            continue
        phase = phases[int(phase_raw)]
        tick = time_index[float(row["match_time_seconds"])]
        result[tick] = torch.tensor(
            (
                float(phase["activation_time_seconds"]),
                float(phase["shrink_start_time_seconds"]),
                float(phase["closure_time_seconds"]),
            ),
            dtype=result.dtype,
        )
    return result


def load_match_session_with_compatibility_context(
    path: str | Path,
    *,
    legacy_exception: str,
) -> tuple[TensorizedMatch, dict[str, Any]]:
    """Recover only exact, context-proven legacy tensorizer false positives."""

    if legacy_exception not in {
        LEGACY_FALSE_POSITIVE,
        LEGACY_ZONE_FALSE_POSITIVE,
    }:
        raise TensorizationCompatibilityError(
            f"unsupported legacy exception: {legacy_exception}"
        )
    directory = Path(path)
    player_table = pq.read_table(directory / "player_samples.parquet")
    phase_table = pq.read_table(directory / "zone_phases.parquet")
    zone_table = pq.read_table(directory / "zone_samples.parquet")
    for name, table in (
        ("player_samples.parquet", player_table),
        ("zone_phases.parquet", phase_table),
        ("zone_samples.parquet", zone_table),
    ):
        if not table.schema.equals(_INPUT_SCHEMAS[name], check_metadata=False):
            raise TensorizationCompatibilityError(
                f"{name} has an unexpected schema"
            )
    player_rows = player_table.to_pylist()
    phase_rows = phase_table.to_pylist()
    zone_rows = zone_table.to_pylist()

    lifecycle_context: dict[str, Any] | None = None
    normalized_player_rows = [dict(row) for row in player_rows]
    changed_player_rows = 0
    has_lifecycle_pattern = _has_legacy_lifecycle_pattern(player_rows)
    if legacy_exception == LEGACY_FALSE_POSITIVE or has_lifecycle_pattern:
        event_table = pq.read_table(directory / "player_events.parquet")
        if not event_table.schema.equals(_PLAYER_EVENT_SCHEMA, check_metadata=False):
            raise TensorizationCompatibilityError(
                "player_events.parquet has an unexpected schema"
            )
        lifecycle_context = validate_lifecycle_context(
            player_rows,
            event_table.to_pylist(),
        )
        if lifecycle_context["collapsed_transition_count"] <= 0:
            raise TensorizationCompatibilityError(
                "legacy lifecycle failure is not an event-proven collapsed transition"
            )
        normalized_player_rows, changed_player_rows = _normalized_sample_rows(
            player_rows
        )

    zone_context = validate_zone_timing_context(phase_rows, zone_rows)
    if (
        legacy_exception == LEGACY_ZONE_FALSE_POSITIVE
        and zone_context["strict_mismatch_count"] <= 0
    ):
        raise TensorizationCompatibilityError(
            "legacy zone failure is not a producer-epsilon phase transition"
        )
    normalized_phase_rows, changed_phase_values = _normalized_phase_rows(
        phase_rows,
        zone_context,
    )

    with tempfile.TemporaryDirectory(prefix="fncs-tensorize-shadow-") as temporary:
        shadow = Path(temporary)
        for name in ("match_samples.parquet", "zone_samples.parquet"):
            shutil.copyfile(directory / name, shadow / name)
        pq.write_table(
            pa.Table.from_pylist(
                normalized_player_rows,
                schema=player_table.schema,
            ),
            shadow / "player_samples.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                normalized_phase_rows,
                schema=phase_table.schema,
            ),
            shadow / "zone_phases.parquet",
        )
        shadow_match = load_match_session(shadow)

    raw_life = _raw_life_tensor(player_rows, shadow_match)
    raw_phase_times = _raw_phase_times_tensor(
        phase_rows,
        zone_rows,
        shadow_match,
    )
    match = replace(
        shadow_match,
        life_index=raw_life,
        phase_times_s=raw_phase_times,
    )
    restored_fields = {"life_index", "phase_times_s"}
    for field in fields(match):
        if field.name in restored_fields:
            continue
        left = getattr(shadow_match, field.name)
        right = getattr(match, field.name)
        if isinstance(left, torch.Tensor):
            if not torch.equal(left, right):
                raise AssertionError(f"recovery changed tensor field {field.name}")
        elif left != right:
            raise AssertionError(f"recovery changed metadata field {field.name}")

    session_id = str(zone_context["session_id"])
    report = {
        "schema_version": "fncs-tensorization-compatibility-recovery:1.0",
        "session_id": session_id,
        "legacy_exception": legacy_exception,
        "lifecycle": lifecycle_context
        or {
            "session_id": session_id,
            "collapsed_transition_count": 0,
            "status": "not_required_no_legacy_pattern",
        },
        "zone_timing": zone_context,
        "shadow_rows_with_delayed_life_index": changed_player_rows,
        "shadow_phase_time_values_changed": changed_phase_values,
        "returned_tensor_uses_raw_life_index": True,
        "returned_tensor_uses_raw_phase_times": True,
        "non_restored_tensor_fields_changed": [],
        "status": "recovered",
    }
    return match, report


def recover_compatible_full_session(
    record: _SessionRecord,
    profile: WorldGridProfile,
    head_config: RotationHeadConfig,
    *,
    legacy_exception: str,
) -> tuple[_FullSession, dict[str, Any]]:
    match, report = load_match_session_with_compatibility_context(
        record.path,
        legacy_exception=legacy_exception,
    )
    if match.session_id != record.session_id:
        raise TrainingConfigurationError(
            f"session directory {record.session_id} contains {match.session_id}"
        )
    supervision = load_rotation_supervision(
        record.path,
        match,
        profile,
        head_config,
    )
    if (
        supervision.metadata.world_grid_profile_id != profile.profile_id
        or supervision.metadata.world_grid_profile_hash != profile.profile_hash
        or record.world_grid_profile_id != profile.profile_id
        or record.world_grid_profile_hash != profile.profile_hash
        or not profile.supports(record.build, record.world_identity)
    ):
        raise TrainingConfigurationError(
            f"session {record.session_id} is bound to an unexpected world-grid profile"
        )
    return _FullSession(match=match, supervision=supervision), report
