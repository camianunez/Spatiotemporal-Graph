from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fncs_encoder_training.common import atomic_write_json, utc_now
from fncs_encoder_training.lifecycle import LEGACY_FALSE_POSITIVE
from fncs_encoder_training.tensorize_compatibility import (
    LEGACY_ZONE_FALSE_POSITIVE,
    load_match_session_with_compatibility_context,
)
from fortnite_encoder.contracts import TensorizedMatch
from fortnite_encoder.planner_supervision import (
    PlannerTargetSource,
    load_planner_target_source,
)
from fortnite_encoder.tensorize import load_match_session
from fortnite_encoder.world_grid import WorldGridProfile

from .config import TrainingCompatibilityError, TrainingConfigurationError


INPUT_TABLES = (
    "match_samples.parquet",
    "player_samples.parquet",
    "zone_samples.parquet",
    "zone_phases.parquet",
    "team_samples.parquet",
    "player_events.parquet",
)


class DatasetAccessLedger:
    """Durable per-session access ledger with a pre-reader test denylist."""

    schema_version = "fncs-frozen-decoder-data-access:1.0"

    def __init__(
        self,
        path: str | Path,
        *,
        partition_by_session: Mapping[str, str],
        test_session_ids: Sequence[str],
        split_manifest_sha256: str,
        allowed_partitions: Sequence[str],
        phase: str,
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.partition_by_session = dict(partition_by_session)
        self.test_session_ids = frozenset(test_session_ids)
        self.split_manifest_sha256 = split_manifest_sha256
        self.allowed_partitions = tuple(allowed_partitions)
        self.phase = phase
        if not self.test_session_ids:
            raise TrainingConfigurationError("the test denylist must not be empty")
        if not set(self.allowed_partitions) <= {"train", "validation"}:
            raise TrainingConfigurationError(
                "allowed partitions must be a subset of train/validation"
            )
        self.events: list[dict[str, Any]] = []
        self.operation_counts: Counter[str] = Counter()
        self.dataset_attempt_counts: Counter[str] = Counter()
        self.dataset_open_counts: Counter[str] = Counter()
        self.session_request_counts: Counter[str] = Counter()
        self.session_open_counts: Counter[str] = Counter()
        if self.path.exists():
            if not resume:
                raise TrainingConfigurationError(
                    f"refusing to overwrite data-access ledger: {self.path}"
                )
            self._restore()
        self.persist()

    def _restore(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingCompatibilityError(
                f"cannot restore data-access ledger: {exc}"
            ) from exc
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != self.schema_version
            or raw.get("split_manifest_sha256") != self.split_manifest_sha256
            or raw.get("allowed_partitions") != list(self.allowed_partitions)
            or raw.get("sealed_test_session_ids")
            != sorted(self.test_session_ids)
        ):
            raise TrainingCompatibilityError("persisted data-access ledger changed")
        events = raw.get("events")
        if not isinstance(events, list) or any(
            not isinstance(event, dict) for event in events
        ):
            raise TrainingCompatibilityError("persisted access events are invalid")
        self.events = list(events)
        self.operation_counts.update(raw.get("operation_counts", {}))
        self.dataset_attempt_counts.update(raw.get("dataset_attempt_counts", {}))
        self.dataset_open_counts.update(raw.get("dataset_open_counts", {}))
        self.session_request_counts.update(raw.get("session_request_counts", {}))
        self.session_open_counts.update(raw.get("session_open_counts", {}))

    def _partition(self, session_id: str) -> str:
        partition = self.partition_by_session.get(session_id)
        if partition is None:
            self.operation_counts["unbound:request"] += 1
            self.events.append(
                {
                    "sequence": len(self.events) + 1,
                    "session_id": session_id,
                    "partition": "unbound",
                    "phase": self.phase,
                    "operation": "session_request",
                    "outcome": "blocked_before_reader",
                }
            )
            self.persist()
            raise TrainingConfigurationError(
                f"dataset access attempted for unbound session {session_id}"
            )
        return partition

    def request(self, session_id: str, *, operation: str) -> str:
        partition = self._partition(session_id)
        self.session_request_counts[partition] += 1
        blocked = (
            session_id in self.test_session_ids
            or partition == "test"
            or partition not in self.allowed_partitions
        )
        self.operation_counts[
            f"{self.phase}:{operation}:{'blocked' if blocked else 'allowed'}"
        ] += 1
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "session_id": session_id,
                "partition": partition,
                "phase": self.phase,
                "operation": operation,
                "outcome": "blocked_before_reader" if blocked else "allowed",
            }
        )
        self.persist()
        if session_id in self.test_session_ids or partition == "test":
            raise TrainingConfigurationError(
                f"denylisted encoder-test session request blocked: {session_id}"
            )
        if partition not in self.allowed_partitions:
            raise TrainingConfigurationError(
                f"partition {partition} is inactive during {self.phase}"
            )
        return partition

    def dataset_attempt(
        self,
        session_id: str,
        *,
        operation: str,
        table_names: Sequence[str],
    ) -> None:
        partition = self.partition_by_session[session_id]
        if partition == "test" or session_id in self.test_session_ids:
            raise AssertionError("test access reached the dataset reader boundary")
        self.dataset_attempt_counts[partition] += 1
        self.operation_counts[f"{self.phase}:{operation}:dataset_attempt"] += 1
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "session_id": session_id,
                "partition": partition,
                "phase": self.phase,
                "operation": operation,
                "outcome": "dataset_open_attempt",
                "tables": list(table_names),
            }
        )
        self.persist()

    def dataset_opened(
        self,
        session_id: str,
        *,
        operation: str,
        table_names: Sequence[str],
        compatibility_recovery: Mapping[str, Any] | None,
    ) -> None:
        partition = self.partition_by_session[session_id]
        if partition == "test" or session_id in self.test_session_ids:
            raise AssertionError("a denylisted encoder-test session was opened")
        self.dataset_open_counts[partition] += 1
        self.session_open_counts[partition] += 1
        self.operation_counts[f"{self.phase}:{operation}:dataset_open"] += 1
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "session_id": session_id,
                "partition": partition,
                "phase": self.phase,
                "operation": operation,
                "outcome": "dataset_open_success",
                "tables": list(table_names),
                "compatibility_recovery": (
                    None
                    if compatibility_recovery is None
                    else {
                        "schema_version": compatibility_recovery.get(
                            "schema_version"
                        ),
                        "legacy_exception": compatibility_recovery.get(
                            "legacy_exception"
                        ),
                        "status": compatibility_recovery.get("status"),
                        "collapsed_transition_count": compatibility_recovery.get(
                            "lifecycle", {}
                        ).get("collapsed_transition_count"),
                        "zone_timing_strict_mismatch_count": (
                            compatibility_recovery.get("zone_timing", {}).get(
                                "strict_mismatch_count"
                            )
                        ),
                    }
                ),
            }
        )
        self.persist()

    def report(self) -> dict[str, Any]:
        test_requests = int(self.session_request_counts["test"])
        test_attempts = int(self.dataset_attempt_counts["test"])
        test_opens = int(self.dataset_open_counts["test"])
        validation_requests = int(self.session_request_counts["validation"])
        validation_attempts = int(self.dataset_attempt_counts["validation"])
        validation_opens = int(self.dataset_open_counts["validation"])
        return {
            "schema_version": self.schema_version,
            "updated_utc": utc_now(),
            "phase": self.phase,
            "allowed_partitions": list(self.allowed_partitions),
            "split_manifest_sha256": self.split_manifest_sha256,
            "sealed_test_session_ids": sorted(self.test_session_ids),
            "sealed_test_session_count": len(self.test_session_ids),
            "no_decoder_test_split_constructed": True,
            "session_request_counts": dict(sorted(self.session_request_counts.items())),
            "session_open_counts": dict(sorted(self.session_open_counts.items())),
            "dataset_attempt_counts": dict(sorted(self.dataset_attempt_counts.items())),
            "dataset_open_counts": dict(sorted(self.dataset_open_counts.items())),
            "operation_counts": dict(sorted(self.operation_counts.items())),
            "validation_session_request_count": validation_requests,
            "validation_dataset_attempt_count": validation_attempts,
            "validation_dataset_open_count": validation_opens,
            "test_session_request_attempt_count": test_requests,
            "test_session_parquet_open_attempt_count": test_attempts,
            "test_session_parquet_open_count": test_opens,
            "zero_test_attempts": test_requests == 0 and test_attempts == 0,
            "zero_test_opens": test_opens == 0,
            "zero_validation_attempts": (
                validation_requests == 0 and validation_attempts == 0
            ),
            "zero_validation_opens": validation_opens == 0,
            "test_partition_evaluated": False,
            "events": list(self.events),
        }

    def persist(self) -> None:
        atomic_write_json(self.path, self.report())


class FNCSParallelTrajectorySessionRepository:
    """Load exact FNCS split sessions under an audited fail-closed policy."""

    def __init__(
        self,
        *,
        paths: Mapping[str, Path],
        partitions: Mapping[str, str],
        active_partitions: Sequence[str],
        test_session_ids: Sequence[str],
        profile: WorldGridProfile,
        ledger_path: str | Path,
        split_manifest_sha256: str,
        phase: str,
        resume_ledger: bool = False,
    ) -> None:
        if set(paths) != set(partitions):
            raise TrainingConfigurationError(
                "repository paths and split partitions differ"
            )
        self._paths = {name: Path(value).resolve() for name, value in paths.items()}
        self._partitions = dict(partitions)
        self._profile = profile
        self._cache: dict[str, PlannerTargetSource] = {}
        self.tensorization_recoveries: dict[str, Mapping[str, Any]] = {}
        self.ledger = DatasetAccessLedger(
            ledger_path,
            partition_by_session=self._partitions,
            test_session_ids=test_session_ids,
            split_manifest_sha256=split_manifest_sha256,
            allowed_partitions=active_partitions,
            phase=phase,
            resume=resume_ledger,
        )

    def _load_match(
        self, session_id: str, path: Path
    ) -> tuple[TensorizedMatch, Mapping[str, Any] | None]:
        recovery: Mapping[str, Any] | None = None
        try:
            match = load_match_session(path)
        except ValueError as exc:
            message = str(exc)
            if message not in {LEGACY_FALSE_POSITIVE, LEGACY_ZONE_FALSE_POSITIVE}:
                raise
            match, recovery = load_match_session_with_compatibility_context(
                path, legacy_exception=message
            )
        if match.session_id != session_id:
            raise TrainingCompatibilityError(
                f"session directory {session_id} contains {match.session_id}"
            )
        return match, recovery

    def get(self, session_id: str) -> PlannerTargetSource:
        self.ledger.request(session_id, operation="target_source_request")
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached
        path = self._paths[session_id]
        self.ledger.dataset_attempt(
            session_id,
            operation="causal_inputs_and_future_targets",
            table_names=INPUT_TABLES,
        )
        try:
            match, recovery = self._load_match(session_id, path)
            source = load_planner_target_source(path, match, self._profile)
        except Exception:
            self.ledger.persist()
            raise
        self._cache[session_id] = source
        if recovery is not None:
            self.tensorization_recoveries[session_id] = recovery
        self.ledger.dataset_opened(
            session_id,
            operation="causal_inputs_and_future_targets",
            table_names=INPUT_TABLES,
            compatibility_recovery=recovery,
        )
        return source

    def report(self) -> dict[str, Any]:
        report = self.ledger.report()
        report["repository_partition_counts"] = {
            partition: sum(value == partition for value in self._partitions.values())
            for partition in ("train", "validation", "test")
        }
        report["cached_session_count"] = len(self._cache)
        report["tensorization_recovery_session_count"] = len(
            self.tensorization_recoveries
        )
        report["tensorization_recovery_session_ids"] = sorted(
            self.tensorization_recoveries
        )
        return report

    def assert_no_test_access(self) -> None:
        report = self.report()
        if not report["zero_test_attempts"] or not report["zero_test_opens"]:
            raise TrainingCompatibilityError(
                "encoder-test access was attempted or completed"
            )


__all__ = [
    "DatasetAccessLedger",
    "FNCSParallelTrajectorySessionRepository",
    "INPUT_TABLES",
]
