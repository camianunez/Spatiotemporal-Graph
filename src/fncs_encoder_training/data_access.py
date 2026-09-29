from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from fortnite_encoder.config import EncoderConfig, RotationHeadConfig
from fortnite_encoder.training import (
    TrainingConfigurationError,
    WindowRequest,
    _SessionRecord,
    _SessionRepository,
)
from fortnite_encoder.world_grid import WorldGridProfile

from .common import atomic_write_json, utc_now
from .lifecycle import LEGACY_FALSE_POSITIVE
from .tensorize_compatibility import (
    LEGACY_ZONE_FALSE_POSITIVE,
    recover_compatible_full_session,
)


TRAINING_READ_SEQUENCE = (
    "match_samples.parquet",
    "player_samples.parquet",
    "zone_samples.parquet",
    "zone_phases.parquet",
    "team_samples.parquet",
    "player_events.parquet",
    "zone_phases.parquet",
)


class DataAccessLedger:
    def __init__(
        self,
        *,
        path: str | Path,
        partition_by_session: Mapping[str, str],
        test_session_ids: Sequence[str],
        split_manifest_sha256: str,
        phase: str,
        allowed_partitions: Sequence[str],
    ) -> None:
        self.path = Path(path)
        self.partition_by_session = dict(partition_by_session)
        self.test_session_ids = frozenset(test_session_ids)
        self.split_manifest_sha256 = split_manifest_sha256
        self.phase = phase
        self.allowed_partitions = tuple(allowed_partitions)
        self.session_attempts: list[str] = []
        self.session_opens: list[str] = []
        self.base_session_attempt_count = 0
        self.base_session_open_count = 0
        self.file_attempt_counts: Counter[str] = Counter()
        self.file_open_counts: Counter[str] = Counter()
        self.operation_counts: Counter[str] = Counter()
        if self.path.is_file():
            prior = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                isinstance(prior, dict)
                and prior.get("schema_version") == "fncs-encoder-data-access:1.0"
                and prior.get("split_manifest_sha256") == split_manifest_sha256
            ):
                self.session_attempts.extend(prior.get("unique_attempted_session_ids", []))
                self.session_opens.extend(prior.get("unique_opened_session_ids", []))
                self.base_session_attempt_count = max(
                    0,
                    int(prior.get("session_attempt_count", 0)) - len(self.session_attempts),
                )
                self.base_session_open_count = max(
                    0,
                    int(prior.get("session_open_count", 0)) - len(self.session_opens),
                )
                self.file_attempt_counts.update(prior.get("file_attempt_counts", {}))
                self.file_open_counts.update(prior.get("file_open_counts", {}))
                self.operation_counts.update(prior.get("operation_counts", {}))

    def set_phase(self, phase: str, allowed_partitions: Sequence[str]) -> None:
        self.phase = phase
        self.allowed_partitions = tuple(allowed_partitions)
        self.persist()

    def _partition(self, session_id: str) -> str:
        partition = self.partition_by_session.get(session_id)
        if partition is None:
            raise TrainingConfigurationError(
                f"data access attempted for unbound session {session_id}"
            )
        return partition

    def attempt(self, session_id: str, table_name: str, operation: str) -> None:
        partition = self._partition(session_id)
        if session_id in self.test_session_ids or partition == "test":
            self.file_attempt_counts[f"test:{table_name}"] += 1
            self.persist()
            raise TrainingConfigurationError(
                f"sealed test-file attempt blocked before open: {session_id}/{table_name}"
            )
        if partition not in self.allowed_partitions:
            raise TrainingConfigurationError(
                f"partition {partition} is not active during {self.phase}"
            )
        self.session_attempts.append(session_id)
        self.file_attempt_counts[f"{partition}:{table_name}"] += 1
        self.operation_counts[f"{self.phase}:{operation}:attempt"] += 1

    def opened(self, session_id: str, table_name: str, operation: str) -> None:
        partition = self._partition(session_id)
        if partition == "test":
            raise AssertionError("a sealed test file was opened")
        self.session_opens.append(session_id)
        self.file_open_counts[f"{partition}:{table_name}"] += 1
        self.operation_counts[f"{self.phase}:{operation}:open"] += 1

    def report(self) -> dict[str, Any]:
        test_attempts = sum(
            count
            for name, count in self.file_attempt_counts.items()
            if name.startswith("test:")
        )
        test_opens = sum(
            count
            for name, count in self.file_open_counts.items()
            if name.startswith("test:")
        )
        validation_attempts = sum(
            count
            for name, count in self.file_attempt_counts.items()
            if name.startswith("validation:")
        )
        validation_opens = sum(
            count
            for name, count in self.file_open_counts.items()
            if name.startswith("validation:")
        )
        return {
            "schema_version": "fncs-encoder-data-access:1.0",
            "updated_utc": utc_now(),
            "phase": self.phase,
            "allowed_partitions": list(self.allowed_partitions),
            "split_manifest_sha256": self.split_manifest_sha256,
            "sealed_test_session_ids": sorted(self.test_session_ids),
            "session_attempt_count": self.base_session_attempt_count + len(self.session_attempts),
            "session_open_count": self.base_session_open_count + len(self.session_opens),
            "unique_attempted_session_ids": sorted(set(self.session_attempts)),
            "unique_opened_session_ids": sorted(set(self.session_opens)),
            "file_attempt_counts": dict(sorted(self.file_attempt_counts.items())),
            "file_open_counts": dict(sorted(self.file_open_counts.items())),
            "operation_counts": dict(sorted(self.operation_counts.items())),
            "validation_session_parquet_attempt_count": validation_attempts,
            "validation_session_parquet_open_count": validation_opens,
            "test_session_parquet_attempt_count": test_attempts,
            "test_session_parquet_open_count": test_opens,
            "zero_test_file_attempts": test_attempts == 0,
            "zero_test_file_opens": test_opens == 0,
            "test_partition_evaluated": False,
        }

    def persist(self) -> None:
        atomic_write_json(self.path, self.report())


class AuditedSessionRepository(_SessionRepository):
    def __init__(
        self,
        records: Sequence[_SessionRecord],
        profile: WorldGridProfile,
        head_config: RotationHeadConfig,
        num_workers: int,
        persistent_workers: bool,
        ledger: DataAccessLedger,
    ) -> None:
        super().__init__(records, profile, head_config, num_workers, persistent_workers)
        self.ledger = ledger
        self.lifecycle_recoveries: dict[str, dict[str, Any]] = {}
        self.zone_timing_recoveries: dict[str, dict[str, Any]] = {}

    def _full(self, session_id: str):  # type: ignore[no-untyped-def]
        if session_id not in self.cache:
            for table_name in TRAINING_READ_SEQUENCE:
                self.ledger.attempt(session_id, table_name, "full_session_load")
            try:
                value = super()._full(session_id)
            except ValueError as exc:
                legacy_exception = str(exc)
                if legacy_exception not in {
                    LEGACY_FALSE_POSITIVE,
                    LEGACY_ZONE_FALSE_POSITIVE,
                }:
                    raise
                value, recovery = recover_compatible_full_session(
                    self.records[session_id],
                    self.profile,
                    self.head_config,
                    legacy_exception=legacy_exception,
                )
                self.cache[session_id] = value
                if recovery["lifecycle"]["collapsed_transition_count"]:
                    self.lifecycle_recoveries[session_id] = recovery
                if recovery["zone_timing"]["strict_mismatch_count"]:
                    self.zone_timing_recoveries[session_id] = recovery
            for table_name in TRAINING_READ_SEQUENCE:
                self.ledger.opened(session_id, table_name, "full_session_load")
            self.ledger.persist()
            return value
        return super()._full(session_id)

    def windows(self, requests: Sequence[WindowRequest]):  # type: ignore[no-untyped-def]
        for request in requests:
            partition = self.ledger._partition(request.session_id)
            if partition == "test" or request.session_id in self.ledger.test_session_ids:
                raise TrainingConfigurationError(
                    f"sealed test window request blocked: {request.session_id}"
                )
            if partition not in self.ledger.allowed_partitions:
                raise TrainingConfigurationError(
                    f"partition {partition} is inactive during {self.ledger.phase}"
                )
        return super().windows(requests)


def audited_session_lengths(
    records: Sequence[_SessionRecord],
    encoder_config: EncoderConfig,
    *,
    active_session_ids: Sequence[str],
    ledger: DataAccessLedger,
) -> dict[str, int]:
    by_id = {record.session_id: record for record in records}
    output: dict[str, int] = {}
    for session_id in active_session_ids:
        record = by_id[session_id]
        table_name = "match_samples.parquet"
        ledger.attempt(session_id, table_name, "session_length_footer")
        path = record.path / table_name
        try:
            count = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:
            raise TrainingConfigurationError(
                f"cannot inspect match table {path}: {exc}"
            ) from exc
        ledger.opened(session_id, table_name, "session_length_footer")
        if count <= 0:
            raise TrainingConfigurationError(f"session {session_id} has no match samples")
        if count > encoder_config.max_timesteps:
            raise TrainingConfigurationError(
                f"session {session_id} has {count} ticks, above max_timesteps"
            )
        output[session_id] = count
    ledger.persist()
    return output
