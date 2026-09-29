from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import torch

from fortnite_encoder.contracts import EncoderBatch, TensorizedMatch
from fortnite_encoder.planner_contracts import PlannerObservation
from fortnite_encoder.planner_policy import PlannerPolicyConfig, build_planner_observation
from fortnite_encoder.planner_supervision import (
    PlannerQuery,
    PlannerTargetSource,
    eligible_planner_queries,
    load_planner_target_source,
)
from fortnite_encoder.tensorize import (
    collate_encoder_inputs,
    load_match_session,
    slice_window,
)
from fortnite_encoder.world_grid import WorldGridProfile
from fortnite_parallel_trajectory.contracts import ParallelTrajectoryTargets
from fortnite_parallel_trajectory.targets import build_parallel_trajectory_targets

from .config import TrainingCompatibilityError, TrainingConfigurationError


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stable_seed(*values: Any) -> int:
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def uniform_sample(
    queries: Sequence[PlannerQuery], count: int, *, seed: int
) -> tuple[PlannerQuery, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not queries:
        return ()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(queries), generator=generator).tolist()
    return tuple(queries[index] for index in order[: min(count, len(order))])


def training_session_order(
    session_ids: Sequence[str], *, seed: int, epoch: int
) -> tuple[str, ...]:
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("epoch must be positive")
    ordered = tuple(sorted(session_ids))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, "training-session-order", epoch))
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    return tuple(ordered[index] for index in permutation)


def session_batches(
    values: Sequence[str], batch_size: int
) -> tuple[tuple[str, ...], ...]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(
        tuple(values[start : start + batch_size])
        for start in range(0, len(values), batch_size)
    )


@dataclass(frozen=True, slots=True)
class ParallelTrajectoryQueryExample:
    """One full-match-derived target paired with one trailing causal slice."""

    match: TensorizedMatch
    focal_team_index: int
    targets: ParallelTrajectoryTargets
    query: PlannerQuery
    query_phase: int

    def __post_init__(self) -> None:
        if not isinstance(self.match, TensorizedMatch):
            raise TypeError("match must be a TensorizedMatch")
        if type(self.focal_team_index) is not int or not (
            0 <= self.focal_team_index < self.match.num_teams
        ):
            raise ValueError("focal_team_index is outside the causal match")
        if not isinstance(self.targets, ParallelTrajectoryTargets):
            raise TypeError("targets must be ParallelTrajectoryTargets")
        if self.targets.target_mask.shape[0] != 1:
            raise ValueError("each example must contain exactly one target route")
        if not bool(self.targets.query_mask[0]):
            raise ValueError("each example must contain an eligible query")
        if not bool(self.targets.target_mask[0].any()):
            raise ValueError("each example requires at least one valid horizon")
        if not isinstance(self.query, PlannerQuery):
            raise TypeError("query must be PlannerQuery")
        if not 1 <= self.query_phase <= 8:
            raise ValueError("query_phase must be in the inclusive range 1..8")


def _query_position(source: PlannerTargetSource, query: PlannerQuery) -> int:
    if query.session_id != source.match.session_id:
        raise ValueError("query session does not match target source")
    matches = (
        source.match.absolute_tick_index == query.absolute_tick_index
    ).nonzero(as_tuple=False)
    if matches.numel() != 1:
        raise ValueError("query tick is not present exactly once")
    return int(matches.item())


def _slice_target(targets: ParallelTrajectoryTargets, index: int) -> ParallelTrajectoryTargets:
    if type(index) is not int or not 0 <= index < targets.target_mask.shape[0]:
        raise IndexError("target row is outside the target batch")
    item = slice(index, index + 1)
    return ParallelTrajectoryTargets(
        target_displacements=targets.target_displacements[item].clone(),
        target_mask=targets.target_mask[item].clone(),
        query_mask=targets.query_mask[item].clone(),
        current_xy=(
            None if targets.current_xy is None else targets.current_xy[item].clone()
        ),
        future_xy=(
            None if targets.future_xy is None else targets.future_xy[item].clone()
        ),
        queries=(targets.queries[index],) if targets.queries else (),
        world_grid_profile_id=targets.world_grid_profile_id,
        world_grid_profile_hash=targets.world_grid_profile_hash,
        target_schema_id=targets.target_schema_id,
    )


def collate_parallel_trajectory_targets(
    values: Iterable[ParallelTrajectoryTargets],
) -> ParallelTrajectoryTargets:
    rows = tuple(values)
    if not rows:
        raise ValueError("at least one target row is required")
    if any(row.target_mask.shape[0] != 1 for row in rows):
        raise ValueError("target collation expects single-query rows")
    profile_ids = {row.world_grid_profile_id for row in rows}
    profile_hashes = {row.world_grid_profile_hash for row in rows}
    schema_ids = {row.target_schema_id for row in rows}
    if len(profile_ids) != 1 or len(profile_hashes) != 1 or len(schema_ids) != 1:
        raise ValueError("target provenance differs within a batch")
    if any((row.current_xy is None) != (rows[0].current_xy is None) for row in rows):
        raise ValueError("current-coordinate metadata differs within a batch")
    if any((row.future_xy is None) != (rows[0].future_xy is None) for row in rows):
        raise ValueError("future-coordinate metadata differs within a batch")
    return ParallelTrajectoryTargets(
        target_displacements=torch.cat(
            [row.target_displacements for row in rows], dim=0
        ),
        target_mask=torch.cat([row.target_mask for row in rows], dim=0),
        query_mask=torch.cat([row.query_mask for row in rows], dim=0),
        current_xy=(
            None
            if rows[0].current_xy is None
            else torch.cat([row.current_xy for row in rows if row.current_xy is not None])
        ),
        future_xy=(
            None
            if rows[0].future_xy is None
            else torch.cat([row.future_xy for row in rows if row.future_xy is not None])
        ),
        queries=tuple(query for row in rows for query in row.queries),
        world_grid_profile_id=rows[0].world_grid_profile_id,
        world_grid_profile_hash=rows[0].world_grid_profile_hash,
        target_schema_id=rows[0].target_schema_id,
    )


def eligible_parallel_trajectory_queries(
    source: PlannerTargetSource,
) -> tuple[PlannerQuery, ...]:
    """Return phase-1..8 queries having at least one public-contract target."""

    if not isinstance(source, PlannerTargetSource):
        raise TypeError("source must be PlannerTargetSource")
    candidates = tuple(
        query
        for query in eligible_planner_queries(source)
        if 1
        <= int(source.match.zone_phase[_query_position(source, query)].item())
        <= 8
    )
    if not candidates:
        return ()
    targets = build_parallel_trajectory_targets(source, candidates)
    return tuple(
        query
        for index, query in enumerate(candidates)
        if bool(targets.query_mask[index]) and bool(targets.target_mask[index].any())
    )


def prepare_parallel_trajectory_query_examples(
    sources: Mapping[str, PlannerTargetSource],
    queries: Sequence[PlannerQuery],
    *,
    context_length_ticks: int = 64,
) -> tuple[ParallelTrajectoryQueryExample, ...]:
    """Build labels on complete matches before taking any causal input slice."""

    if type(context_length_ticks) is not int or context_length_ticks <= 0:
        raise ValueError("context_length_ticks must be positive")
    if len(set(queries)) != len(queries):
        raise ValueError("parallel trajectory queries must be unique")
    target_rows: dict[PlannerQuery, ParallelTrajectoryTargets] = {}
    phases: dict[PlannerQuery, int] = {}
    for session_id in sorted({query.session_id for query in queries}):
        source = sources.get(session_id)
        if source is None:
            raise ValueError(f"query source {session_id} was not loaded")
        session_queries = tuple(
            query for query in queries if query.session_id == session_id
        )
        # The full match deliberately remains available through target construction.
        full = build_parallel_trajectory_targets(source, session_queries)
        for index, query in enumerate(session_queries):
            row = _slice_target(full, index)
            position = _query_position(source, query)
            phase = int(source.match.zone_phase[position].item())
            if (
                1 <= phase <= 8
                and bool(row.query_mask[0])
                and bool(row.target_mask[0].any())
            ):
                target_rows[query] = row
                phases[query] = phase

    examples: list[ParallelTrajectoryQueryExample] = []
    for query in queries:
        target = target_rows.get(query)
        if target is None:
            continue
        source = sources[query.session_id]
        match = source.match
        position = _query_position(source, query)
        start_position = max(0, position - context_length_ticks + 1)
        start_tick = int(match.absolute_tick_index[start_position].item())
        causal = slice_window(
            match,
            start_tick=start_tick,
            length=position - start_position + 1,
        )
        examples.append(
            ParallelTrajectoryQueryExample(
                match=causal,
                focal_team_index=match.team_ids.index(query.team_id),
                targets=target,
                query=query,
                query_phase=phases[query],
            )
        )
    return tuple(examples)


def collate_parallel_trajectory_query_examples(
    examples: Sequence[ParallelTrajectoryQueryExample],
    policy_config: PlannerPolicyConfig | None = None,
) -> tuple[EncoderBatch, PlannerObservation, ParallelTrajectoryTargets]:
    if not examples:
        raise ValueError("at least one parallel trajectory example is required")
    collated = collate_encoder_inputs(example.match for example in examples)
    observation = build_planner_observation(
        collated.batch,
        [example.focal_team_index for example in examples],
        policy_config,
    )
    targets = collate_parallel_trajectory_targets(
        example.targets for example in examples
    )
    return collated.batch, observation, targets


class ParallelTrajectorySessionRepository:
    """Explicit-partition source cache with pre-reader test and inactive guards."""

    def __init__(
        self,
        paths: Mapping[str, Path],
        partitions: Mapping[str, str],
        *,
        active_partitions: Iterable[str],
        test_session_ids: Iterable[str],
        profile: WorldGridProfile,
        match_loader: Callable[[str | Path], TensorizedMatch] = load_match_session,
        target_loader: Callable[
            [str | Path, TensorizedMatch, WorldGridProfile], PlannerTargetSource
        ] = load_planner_target_source,
    ) -> None:
        active_names = frozenset(active_partitions)
        if not active_names or not active_names <= {"train", "validation"}:
            raise TrainingConfigurationError(
                "active partitions must be a nonempty subset of train/validation"
            )
        all_paths = {name: Path(path).resolve() for name, path in paths.items()}
        all_partitions = dict(partitions)
        if set(all_paths) != set(all_partitions):
            raise TrainingConfigurationError(
                "repository paths and partition labels must match exactly"
            )
        tests = set(test_session_ids)
        active_ids = {
            session_id
            for session_id, partition in all_partitions.items()
            if partition in active_names
        }
        if active_ids & tests:
            raise TrainingConfigurationError("active and test session IDs overlap")
        self._paths = {name: all_paths[name] for name in active_ids}
        self._partitions = {name: all_partitions[name] for name in active_ids}
        self._active_partitions = active_names
        self._tests = tests
        self._profile = profile
        self._match_loader = match_loader
        self._target_loader = target_loader
        self._cache: dict[str, PlannerTargetSource] = {}
        self.open_attempt_session_ids: list[str] = []
        self.opened_session_ids: list[str] = []
        self._audit_path: Path | None = None
        self._split_hash: str | None = None

    def attach_data_access_audit(
        self, path: str | Path, *, split_manifest_sha256: str
    ) -> None:
        self._audit_path = Path(path)
        self._split_hash = split_manifest_sha256
        if self._audit_path.is_file():
            try:
                previous = json.loads(self._audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TrainingCompatibilityError(
                    f"cannot resume data-access ledger: {exc}"
                ) from exc
            attempts = previous.get("parquet_open_attempt_session_ids")
            opened = previous.get("parquet_opened_session_ids")
            if (
                not isinstance(previous, dict)
                or previous.get("split_manifest_sha256") != split_manifest_sha256
                or previous.get("active_partitions") != sorted(self._active_partitions)
                or not isinstance(attempts, list)
                or not isinstance(opened, list)
                or any(not isinstance(value, str) for value in attempts + opened)
            ):
                raise TrainingCompatibilityError(
                    "persisted data-access ledger is incompatible"
                )
            if set(attempts) & self._tests or set(opened) & self._tests:
                raise TrainingCompatibilityError(
                    "persisted data-access ledger contains sealed test access"
                )
            self.open_attempt_session_ids.extend(attempts)
            self.opened_session_ids.extend(opened)
        self._write_audit()

    def get(self, session_id: str) -> PlannerTargetSource:
        # This guard runs before an attempt is recorded and before either reader.
        if session_id in self._tests or session_id not in self._paths:
            raise TrainingCompatibilityError(
                f"attempted to open inactive or sealed test session {session_id}"
            )
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached
        path = self._paths[session_id]
        self.open_attempt_session_ids.append(session_id)
        self._write_audit()
        match = self._match_loader(path)
        if match.session_id != session_id:
            raise TrainingCompatibilityError(
                "session identity changed after split verification"
            )
        source = self._target_loader(path, match, self._profile)
        self._cache[session_id] = source
        self.opened_session_ids.append(session_id)
        self._write_audit()
        return source

    def report(self) -> dict[str, Any]:
        attempted_tests = sorted(set(self.open_attempt_session_ids) & self._tests)
        opened_tests = sorted(set(self.opened_session_ids) & self._tests)
        attempts_by_partition = {
            partition: [
                value
                for value in self.open_attempt_session_ids
                if self._partitions.get(value) == partition
            ]
            for partition in ("train", "validation")
        }
        opens_by_partition = {
            partition: [
                value
                for value in self.opened_session_ids
                if self._partitions.get(value) == partition
            ]
            for partition in ("train", "validation")
        }
        return {
            "schema_version": "parallel-trajectory-data-access:1.0",
            "split_manifest_sha256": self._split_hash,
            "active_partitions": sorted(self._active_partitions),
            "repository_session_ids": sorted(self._paths),
            "repository_partition_counts": {
                partition: sum(
                    value == partition for value in self._partitions.values()
                )
                for partition in ("train", "validation")
            },
            "excluded_test_session_ids": sorted(self._tests),
            "parquet_open_attempt_session_ids": list(
                self.open_attempt_session_ids
            ),
            "parquet_opened_session_ids": list(self.opened_session_ids),
            "training_session_parquet_open_attempt_count": len(
                attempts_by_partition["train"]
            ),
            "training_session_parquet_open_count": len(opens_by_partition["train"]),
            "validation_session_parquet_open_attempt_count": len(
                attempts_by_partition["validation"]
            ),
            "validation_session_parquet_open_count": len(
                opens_by_partition["validation"]
            ),
            "test_session_parquet_open_attempt_count": len(attempted_tests),
            "test_session_parquet_open_count": len(opened_tests),
            "zero_test_session_parquet_attempts": not attempted_tests,
            "zero_test_session_parquet_opens": not opened_tests,
            "zero_validation_and_test_attempts": (
                not attempts_by_partition["validation"] and not attempted_tests
            ),
            "zero_validation_and_test_opens": (
                not opens_by_partition["validation"] and not opened_tests
            ),
        }

    def _write_audit(self) -> None:
        if self._audit_path is not None:
            _atomic_write_json(self._audit_path, self.report())


__all__ = [
    "ParallelTrajectoryQueryExample",
    "ParallelTrajectorySessionRepository",
    "collate_parallel_trajectory_query_examples",
    "collate_parallel_trajectory_targets",
    "eligible_parallel_trajectory_queries",
    "prepare_parallel_trajectory_query_examples",
    "session_batches",
    "stable_seed",
    "training_session_order",
    "uniform_sample",
]
