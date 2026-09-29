from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Union, get_args, get_origin, get_type_hints

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor, nn

from .config import EncoderConfig, RotationHeadConfig, RotationLossConfig
from .contracts import (
    CollatedEncoderInput,
    CollatedRotationSupervision,
    EncoderBatch,
    RotationLossOutput,
    RotationSupervision,
    TensorizedMatch,
)
from .losses import compute_rotation_loss
from .rotation import RotationModel
from .supervision import (
    collate_rotation_supervision,
    load_rotation_supervision,
    slice_rotation_supervision,
)
from .tensorize import collate_encoder_inputs, load_match_session, slice_window
from .world_grid import (
    WorldGridProfile,
    WorldGridProfileError,
)


DATASET_SCHEMA_VERSION = "2.0.0"
SPLIT_MANIFEST_SCHEMA_VERSION = "2.0"
CHECKPOINT_SCHEMA_VERSION = "2.0"
TRAINING_HORIZONS_SECONDS = (15, 30, 60)
_PARTITIONS = ("train", "validation", "test")
_TRUST_PARTITIONS = ("attested", "unattested")


class TrainingConfigurationError(ValueError):
    """Raised when a training configuration or persisted contract is invalid."""


class TrainingCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot be resumed exactly."""


class TrainingNumericalError(RuntimeError):
    """Raised on a bounded, fail-closed numerical diagnostic."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _strict_dataclass(
    cls: type[Any],
    raw: Any,
    location: str,
) -> Any:
    if not isinstance(raw, dict):
        raise TrainingConfigurationError(f"{location} must be a JSON object")
    declared = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - declared)
    if unknown:
        raise TrainingConfigurationError(
            f"{location} contains unknown fields: {', '.join(unknown)}"
        )
    hints = get_type_hints(cls)
    converted = {
        key: _strict_value(hints[key], value, f"{location}.{key}")
        for key, value in raw.items()
    }
    try:
        return cls(**converted)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigurationError(f"{location}: {exc}") from exc


def _strict_value(annotation: Any, value: Any, location: str) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        errors: list[str] = []
        for option in arguments:
            if option is type(None):
                continue
            try:
                return _strict_value(option, value, location)
            except TrainingConfigurationError as exc:
                errors.append(str(exc))
        raise TrainingConfigurationError(
            f"{location} does not match its declared type"
        )
    if origin is Literal:
        if value not in arguments:
            raise TrainingConfigurationError(
                f"{location} must be one of {arguments}"
            )
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise TrainingConfigurationError(f"{location} must be a JSON array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _strict_value(arguments[0], item, f"{location}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise TrainingConfigurationError(
                f"{location} must have exactly {len(arguments)} items"
            )
        return tuple(
            _strict_value(option, item, f"{location}[{index}]")
            for index, (option, item) in enumerate(zip(arguments, value))
        )
    if dataclasses.is_dataclass(annotation):
        return _strict_dataclass(annotation, value, location)
    if annotation is bool:
        if type(value) is not bool:
            raise TrainingConfigurationError(f"{location} must be boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise TrainingConfigurationError(f"{location} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingConfigurationError(f"{location} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise TrainingConfigurationError(f"{location} must be finite")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise TrainingConfigurationError(f"{location} must be a string")
        return value
    if annotation is Any:
        return value
    raise TrainingConfigurationError(
        f"{location} uses unsupported declared type {annotation!r}"
    )


@dataclass(frozen=True, slots=True)
class WorldGridProfileConfig:
    path: str
    expected_profile_hash: str
    audit_path: str

    def __post_init__(self) -> None:
        for name in ("path", "audit_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")
        if (
            not isinstance(self.expected_profile_hash, str)
            or len(self.expected_profile_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_profile_hash
            )
        ):
            raise ValueError("expected_profile_hash must be lowercase SHA-256")

    def load(self, dataset_root: str | Path) -> WorldGridProfile:
        try:
            return WorldGridProfile.load(
                self.path,
                expected_hash=self.expected_profile_hash,
                audit_path=self.audit_path,
                ingestion_report_path=(
                    Path(dataset_root) / "ingestion-report.json"
                ),
            )
        except WorldGridProfileError as exc:
            raise TrainingConfigurationError(
                f"invalid world-grid profile binding: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Strict, JSON-serializable configuration for one deterministic run."""

    dataset_root: str
    run_directory: str
    seed: int | None = None
    window_length_ticks: int | None = None
    window_stride_ticks: int | None = None
    batch_size: int | None = None
    early_stopping_patience: int | None = None
    world_grid_profile: WorldGridProfileConfig | None = None
    source_partitions: tuple[Literal["attested", "unattested"], ...] = (
        "attested",
    )
    include_session_ids: tuple[str, ...] = ()
    exclude_session_ids: tuple[str, ...] = ()
    provenance_manifest_path: str | None = None
    split_manifest_path: str | None = None
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    accumulation_steps: int = 1
    max_epochs: int | None = None
    max_optimizer_steps: int | None = None
    learning_rate: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_epsilon: float = 1e-8
    weight_decay: float = 1e-2
    scheduler_warmup_steps: int | None = None
    scheduler_warmup_fraction: float = 0.05
    scheduler_min_lr_ratio: float = 0.1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    device: str = "auto"
    gradient_clip_norm: float | None = 1.0
    early_stopping_metric: str = "future_position"
    early_stopping_min_delta: float = 0.0
    validation_every_epochs: int = 1
    checkpoint_every_optimizer_steps: int | None = None
    history_every_optimizer_steps: int = 1
    last_checkpoint_every_microbatches: int = 1
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    head_config: RotationHeadConfig = field(default_factory=RotationHeadConfig)
    loss_config: RotationLossConfig = field(default_factory=RotationLossConfig)

    def __post_init__(self) -> None:
        for name in ("dataset_root", "run_directory"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")
        required = {
            "seed": self.seed,
            "window_length_ticks": self.window_length_ticks,
            "window_stride_ticks": self.window_stride_ticks,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience,
            "world_grid_profile": self.world_grid_profile,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "required training fields are missing: " + ", ".join(missing)
            )
        if self.world_grid_profile is not None and not isinstance(
            self.world_grid_profile, WorldGridProfileConfig
        ):
            raise ValueError(
                "world_grid_profile must be a WorldGridProfileConfig"
            )
        for name, expected_type in {
            "encoder_config": EncoderConfig,
            "head_config": RotationHeadConfig,
            "loss_config": RotationLossConfig,
        }.items():
            if not isinstance(getattr(self, name), expected_type):
                raise ValueError(f"{name} must be {expected_type.__name__}")
        for name in ("provenance_manifest_path", "split_manifest_path"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a nonempty path string when set")
        if self.max_epochs is None and self.max_optimizer_steps is None:
            raise ValueError(
                "at least one of max_epochs or max_optimizer_steps is required"
            )
        for name in (
            "seed",
            "window_length_ticks",
            "window_stride_ticks",
            "batch_size",
            "early_stopping_patience",
            "accumulation_steps",
            "num_workers",
            "validation_every_epochs",
            "history_every_optimizer_steps",
            "last_checkpoint_every_microbatches",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"seed", "num_workers"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "max_epochs",
            "max_optimizer_steps",
            "scheduler_warmup_steps",
            "checkpoint_every_optimizer_steps",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer when set")
        if not self.source_partitions or any(
            value not in _TRUST_PARTITIONS for value in self.source_partitions
        ):
            raise ValueError("source_partitions must contain attested or unattested")
        if len(set(self.source_partitions)) != len(self.source_partitions):
            raise ValueError("source_partitions must not contain duplicates")
        for name, values in {
            "include_session_ids": self.include_session_ids,
            "exclude_session_ids": self.exclude_session_ids,
        }.items():
            if len(set(values)) != len(values) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain unique nonempty strings")
        overlap = set(self.include_session_ids) & set(self.exclude_session_ids)
        if overlap:
            raise ValueError("include_session_ids and exclude_session_ids overlap")
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0.0
            for value in fractions
        ) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError("split fractions must be nonnegative and sum to 1")
        if self.train_fraction <= 0.0 or self.validation_fraction <= 0.0:
            raise ValueError("train and validation fractions must be positive")
        for name, value, lower, inclusive in (
            ("learning_rate", self.learning_rate, 0.0, False),
            ("adamw_epsilon", self.adamw_epsilon, 0.0, False),
            ("weight_decay", self.weight_decay, 0.0, True),
            ("scheduler_warmup_fraction", self.scheduler_warmup_fraction, 0.0, True),
            ("scheduler_min_lr_ratio", self.scheduler_min_lr_ratio, 0.0, False),
            ("early_stopping_min_delta", self.early_stopping_min_delta, 0.0, True),
        ):
            valid_lower = value >= lower if inclusive else value > lower
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not valid_lower
            ):
                raise ValueError(f"{name} has an invalid value")
        if self.scheduler_warmup_fraction > 1.0:
            raise ValueError("scheduler_warmup_fraction must be <= 1")
        if not 0.0 < self.scheduler_min_lr_ratio <= 1.0:
            raise ValueError("scheduler_min_lr_ratio must be in (0, 1]")
        if (
            not isinstance(self.adamw_betas, tuple)
            or len(self.adamw_betas) != 2
            or any(
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(float(beta))
                or not 0.0 <= beta < 1.0
                for beta in self.adamw_betas
            )
        ):
            raise ValueError("adamw_betas must be in [0, 1)")
        if self.gradient_clip_norm is not None and (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, (int, float))
            or not math.isfinite(float(self.gradient_clip_norm))
            or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be positive when set")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if type(self.pin_memory) is not bool or type(self.persistent_workers) is not bool:
            raise ValueError("worker flags must be boolean")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16, or fp16")
        if self.head_config.horizons_seconds != TRAINING_HORIZONS_SECONDS:
            raise ValueError(
                "training requires horizons_seconds exactly (15, 30, 60)"
            )
        if (
            self.head_config.grid_height != 32
            or self.head_config.grid_width != 32
        ):
            raise ValueError(
                "training requires the verified 32x32 world-grid class contract"
            )
        if not self.device.strip():
            raise ValueError("device must be nonempty")
        if not self.early_stopping_metric:
            raise ValueError("early_stopping_metric must be nonempty")

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingConfigurationError(
                f"cannot read training config {source}: {exc}"
            ) from exc
        config = _strict_dataclass(cls, raw, "config")
        base = source.parent

        def resolve_path(value: str | None) -> str | None:
            if value is None:
                return None
            candidate = Path(value)
            return str(
                (candidate if candidate.is_absolute() else base / candidate).resolve()
            )

        return dataclasses.replace(
            config,
            dataset_root=resolve_path(config.dataset_root),
            run_directory=resolve_path(config.run_directory),
            provenance_manifest_path=resolve_path(config.provenance_manifest_path),
            split_manifest_path=resolve_path(config.split_manifest_path),
            world_grid_profile=(
                dataclasses.replace(
                    config.world_grid_profile,
                    path=resolve_path(config.world_grid_profile.path),
                    audit_path=resolve_path(
                        config.world_grid_profile.audit_path
                    ),
                )
                if config.world_grid_profile is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class SplitSessionAssignment:
    session_id: str
    group_id: str
    partition: Literal["train", "validation", "test"]
    trust_partition: Literal["attested", "unattested"]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema_version: str
    dataset_schema_version: str
    ingestion_report_sha256: str
    seed: int
    world_grid_profile_id: str
    world_grid_profile_hash: str
    world_grid_audit_sha256: str
    assignments: tuple[SplitSessionAssignment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported split manifest schema_version")
        if self.dataset_schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError("split manifest dataset schema must be 2.0.0")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("split manifest seed must be nonnegative")
        if (
            not isinstance(self.world_grid_profile_id, str)
            or not self.world_grid_profile_id
        ):
            raise ValueError(
                "split manifest world_grid_profile_id must be nonempty"
            )
        for name in (
            "ingestion_report_sha256",
            "world_grid_profile_hash",
            "world_grid_audit_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"split manifest {name} must be lowercase SHA-256")
        if any(
            not isinstance(item, SplitSessionAssignment)
            for item in self.assignments
        ):
            raise ValueError(
                "split manifest assignments must be SplitSessionAssignment"
            )
        session_ids = [item.session_id for item in self.assignments]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("split manifest assigns a session more than once")
        group_partitions: dict[str, str] = {}
        for item in self.assignments:
            if not item.session_id or not item.group_id:
                raise ValueError("split assignment identifiers must be nonempty")
            if item.partition not in _PARTITIONS:
                raise ValueError("split assignment partition is invalid")
            if item.trust_partition not in _TRUST_PARTITIONS:
                raise ValueError("split assignment trust partition is invalid")
            existing = group_partitions.setdefault(item.group_id, item.partition)
            if existing != item.partition:
                raise ValueError("a split group spans multiple partitions")

    @classmethod
    def from_json(cls, path: str | Path) -> SplitManifest:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingConfigurationError(
                f"cannot read split manifest {source}: {exc}"
            ) from exc
        return _strict_dataclass(cls, raw, "split_manifest")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def session_ids(self, partition: str) -> tuple[str, ...]:
        return tuple(
            item.session_id
            for item in self.assignments
            if item.partition == partition
        )


@dataclass(frozen=True, slots=True)
class ResolvedTrainingConfig:
    """Effective user configuration plus runtime- and data-derived values."""

    training: TrainingConfig
    dataset_root: str
    run_directory: str
    dataset_schema_version: str
    ingestion_report_sha256: str
    world_grid_profile_hash: str
    world_grid_audit_sha256: str
    split_manifest_sha256: str
    source_revision: str | None
    device: str
    precision: Literal["fp32", "bf16", "fp16"]
    scaler_enabled: bool
    discovered_session_count: int
    train_session_count: int
    validation_session_count: int
    test_session_count: int
    microbatches_per_epoch: int
    total_optimizer_steps: int
    warmup_steps: int
    effective_encoder_dropouts: dict[str, float]
    class_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    session_id: str
    trust_partition: Literal["attested", "unattested"]
    path: Path
    replay_sha256: str | None
    provenance_status: str | None
    build: str
    world_identity: str
    world_grid_profile_id: str
    world_grid_profile_hash: str


def _read_ingestion_report(
    dataset_root: Path,
) -> tuple[dict[str, Any], str]:
    path = dataset_root / "ingestion-report.json"
    if not path.is_file():
        raise TrainingConfigurationError(
            f"dataset-v2 ingestion report is missing: {path}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(f"invalid ingestion report: {exc}") from exc
    if not isinstance(report, dict):
        raise TrainingConfigurationError("ingestion report must be an object")
    if report.get("schema_version") != "2.0":
        raise TrainingConfigurationError(
            "ingestion report schema_version must be 2.0"
        )
    if report.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise TrainingConfigurationError(
            f"dataset schema must be {DATASET_SCHEMA_VERSION}"
        )
    if not isinstance(report.get("items"), list):
        raise TrainingConfigurationError("ingestion report items must be an array")
    return report, _file_sha256(path)


def _load_world_grid_binding(
    config: TrainingConfig,
) -> tuple[WorldGridProfile, dict[str, dict[str, Any]], str]:
    assert config.world_grid_profile is not None
    profile = config.world_grid_profile.load(config.dataset_root)
    audit_path = Path(config.world_grid_profile.audit_path)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(
            f"cannot read world-grid audit {audit_path}: {exc}"
        ) from exc
    if not isinstance(audit, dict) or not isinstance(audit.get("replays"), list):
        raise TrainingConfigurationError(
            "world-grid audit must contain a replays array"
        )
    audit_hash = audit.get("audit_hash")
    if audit_hash != profile.publication_audit.audit_sha256:
        raise TrainingConfigurationError(
            "world-grid audit/profile hash binding mismatch"
        )
    by_hash: dict[str, dict[str, Any]] = {}
    for raw in audit["replays"]:
        if not isinstance(raw, dict):
            raise TrainingConfigurationError(
                "world-grid audit replay entries must be objects"
            )
        replay_hash = raw.get("replay_sha256")
        if (
            not isinstance(replay_hash, str)
            or len(replay_hash) != 64
        ):
            raise TrainingConfigurationError(
                "world-grid audit replay_sha256 is invalid"
            )
        normalized = replay_hash.lower()
        if normalized in by_hash:
            raise TrainingConfigurationError(
                f"world-grid audit duplicates replay hash {normalized}"
            )
        by_hash[normalized] = raw
    return profile, by_hash, audit_hash


def discover_dataset_sessions(
    config: TrainingConfig,
) -> tuple[tuple[_SessionRecord, ...], str, str]:
    """Discover report-backed normalized sessions without opening Parquet data."""

    dataset_root = Path(config.dataset_root)
    report, report_hash = _read_ingestion_report(dataset_root)
    profile, audit_by_hash, _ = _load_world_grid_binding(config)
    report_items: dict[str, dict[str, Any]] = {}
    for raw in report["items"]:
        if not isinstance(raw, dict):
            continue
        session_id = raw.get("session_id")
        if isinstance(session_id, str) and session_id:
            if session_id in report_items:
                raise TrainingConfigurationError(
                    f"ingestion report duplicates session_id {session_id}"
                )
            report_items[session_id] = raw

    include = set(config.include_session_ids)
    exclude = set(config.exclude_session_ids)
    records: list[_SessionRecord] = []
    seen: set[str] = set()
    for trust in config.source_partitions:
        partition_root = dataset_root / trust
        if not partition_root.is_dir():
            raise TrainingConfigurationError(
                f"configured dataset partition is missing: {partition_root}"
            )
        for directory in sorted(partition_root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir():
                continue
            session_id = directory.name
            if include and session_id not in include:
                continue
            if session_id in exclude:
                continue
            if session_id in seen:
                raise TrainingConfigurationError(
                    f"session {session_id} exists in multiple source partitions"
                )
            raw = report_items.get(session_id)
            if raw is None:
                raise TrainingConfigurationError(
                    f"session {session_id} is not backed by the ingestion report"
                )
            expected_disposition = f"included_{trust}"
            if raw.get("disposition") != expected_disposition:
                raise TrainingConfigurationError(
                    f"session {session_id} report disposition is not {expected_disposition}"
                )
            replay_hash = (
                raw.get("replay_sha256")
                if isinstance(raw.get("replay_sha256"), str)
                else None
            )
            if replay_hash is None:
                raise TrainingConfigurationError(
                    f"session {session_id} has no replay hash"
                )
            audited = audit_by_hash.get(replay_hash.lower())
            if audited is None:
                raise TrainingConfigurationError(
                    f"session {session_id} has no compatible world-grid audit entry"
                )
            if audited.get("session_id") != session_id:
                raise TrainingConfigurationError(
                    f"world-grid audit session mismatch for {session_id}"
                )
            build_profile = raw.get("build_profile")
            build = (
                build_profile.get("build")
                if isinstance(build_profile, dict)
                and isinstance(build_profile.get("build"), str)
                else None
            )
            if build is None or audited.get("build") != build:
                raise TrainingConfigurationError(
                    f"world-grid audit build mismatch for {session_id}"
                )
            world_identity = audited.get("world_identity")
            if (
                not isinstance(world_identity, str)
                or not profile.supports(build, world_identity)
            ):
                raise TrainingConfigurationError(
                    f"session {session_id} has no compatible world-grid profile"
                )
            records.append(
                _SessionRecord(
                    session_id=session_id,
                    trust_partition=trust,
                    path=directory,
                    replay_sha256=replay_hash,
                    provenance_status=(
                        raw.get("provenance_status")
                        if isinstance(raw.get("provenance_status"), str)
                        else None
                    ),
                    build=build,
                    world_identity=world_identity,
                    world_grid_profile_id=profile.profile_id,
                    world_grid_profile_hash=profile.profile_hash,
                )
            )
            seen.add(session_id)
    missing = include - seen
    if missing:
        raise TrainingConfigurationError(
            "included session IDs were not discovered: " + ", ".join(sorted(missing))
        )
    if not records:
        raise TrainingConfigurationError("no dataset sessions matched the configuration")
    return tuple(records), DATASET_SCHEMA_VERSION, report_hash


def _provenance_groups(
    records: Sequence[_SessionRecord],
    manifest_path: str | None,
) -> dict[str, str]:
    entries_by_hash: dict[str, dict[str, Any]] = {}
    if manifest_path is not None:
        path = Path(manifest_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingConfigurationError(
                f"cannot read provenance manifest {path}: {exc}"
            ) from exc
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") not in {"1.0", "2.0"}
            or not isinstance(raw.get("entries"), list)
        ):
            raise TrainingConfigurationError("invalid provenance manifest")
        for entry in raw["entries"]:
            if not isinstance(entry, dict):
                raise TrainingConfigurationError(
                    "provenance manifest entries must be objects"
                )
            replay_hash = entry.get("replay_sha256")
            if not isinstance(replay_hash, str) or len(replay_hash) != 64:
                raise TrainingConfigurationError(
                    "provenance replay_sha256 must be a 64-character string"
                )
            normalized = replay_hash.lower()
            if normalized in entries_by_hash:
                raise TrainingConfigurationError(
                    f"provenance manifest duplicates replay hash {normalized}"
                )
            entries_by_hash[normalized] = entry

    output: dict[str, str] = {}
    for record in records:
        group_id = f"session:{record.session_id}"
        complete = (
            record.trust_partition == "attested"
            and record.provenance_status in {"complete", "valid", "attested"}
            and record.replay_sha256 is not None
        )
        entry = (
            entries_by_hash.get(record.replay_sha256.lower())
            if complete and record.replay_sha256 is not None
            else None
        )
        if entry is not None:
            if entry.get("game_session_id") != record.session_id:
                raise TrainingConfigurationError(
                    f"provenance game_session_id mismatch for {record.session_id}"
                )
            event_session_id = entry.get("event_session_id")
            event_id = entry.get("event_id")
            if isinstance(event_session_id, str) and event_session_id:
                group_id = (
                    f"event_session:{event_id}:{event_session_id}"
                    if isinstance(event_id, str) and event_id
                    else f"event_session:{event_session_id}"
                )
            elif isinstance(event_id, str) and event_id:
                group_id = f"event:{event_id}"
        output[record.session_id] = group_id
    return output


def _partition_group_counts(group_count: int, config: TrainingConfig) -> tuple[int, int, int]:
    if group_count < 2:
        raise TrainingConfigurationError(
            "at least two independent groups are required for train/validation"
        )
    fractions = (
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    )
    raw = [fraction * group_count for fraction in fractions]
    counts = [int(math.floor(value)) for value in raw]
    remaining = group_count - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    for required in (0, 1):
        if counts[required] == 0:
            donors = sorted(
                range(3),
                key=lambda index: (counts[index], -index),
                reverse=True,
            )
            donor = next(
                (
                    index
                    for index in donors
                    if index != required
                    and counts[index] > (1 if index in (0, 1) else 0)
                ),
                None,
            )
            if donor is None:
                raise TrainingConfigurationError(
                    "split fractions cannot produce nonempty train/validation groups"
                )
            counts[donor] -= 1
            counts[required] += 1
    return counts[0], counts[1], counts[2]


def build_split_manifest(
    config: TrainingConfig,
    records: Sequence[_SessionRecord],
    ingestion_report_sha256: str,
) -> SplitManifest:
    """Build or validate the canonical group-isolated split manifest."""

    assert config.seed is not None
    assert config.world_grid_profile is not None
    profile = config.world_grid_profile.load(config.dataset_root)
    records_by_id = {record.session_id: record for record in records}
    if config.split_manifest_path is not None:
        manifest = SplitManifest.from_json(config.split_manifest_path)
        expected_ids = set(records_by_id)
        actual_ids = {item.session_id for item in manifest.assignments}
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise TrainingConfigurationError(
                f"explicit split session mismatch; missing={missing[:8]}, extra={extra[:8]}"
            )
        if manifest.ingestion_report_sha256 != ingestion_report_sha256:
            raise TrainingConfigurationError(
                "explicit split ingestion-report fingerprint mismatch"
            )
        if manifest.seed != config.seed:
            raise TrainingConfigurationError("explicit split seed mismatch")
        if (
            manifest.world_grid_profile_id != profile.profile_id
            or manifest.world_grid_profile_hash != profile.profile_hash
            or manifest.world_grid_audit_sha256
            != profile.publication_audit.audit_sha256
        ):
            raise TrainingConfigurationError(
                "explicit split world-grid-profile mismatch"
            )
        for item in manifest.assignments:
            if records_by_id[item.session_id].trust_partition != item.trust_partition:
                raise TrainingConfigurationError(
                    f"explicit split trust partition mismatch for {item.session_id}"
                )
        if not manifest.session_ids("train") or not manifest.session_ids("validation"):
            raise TrainingConfigurationError(
                "explicit split requires nonempty train and validation partitions"
            )
        return dataclasses.replace(
            manifest,
            assignments=tuple(
                sorted(manifest.assignments, key=lambda item: item.session_id)
            ),
        )

    group_for_session = _provenance_groups(
        records,
        config.provenance_manifest_path,
    )
    group_ids = sorted(set(group_for_session.values()))
    random.Random(config.seed).shuffle(group_ids)
    train_count, validation_count, _ = _partition_group_counts(
        len(group_ids), config
    )
    group_partition = {
        group_id: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, group_id in enumerate(group_ids)
    }
    assignments = tuple(
        SplitSessionAssignment(
            session_id=record.session_id,
            group_id=group_for_session[record.session_id],
            partition=group_partition[group_for_session[record.session_id]],  # type: ignore[arg-type]
            trust_partition=record.trust_partition,
        )
        for record in sorted(records, key=lambda item: item.session_id)
    )
    return SplitManifest(
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        dataset_schema_version=DATASET_SCHEMA_VERSION,
        ingestion_report_sha256=ingestion_report_sha256,
        seed=config.seed,
        world_grid_profile_id=profile.profile_id,
        world_grid_profile_hash=profile.profile_hash,
        world_grid_audit_sha256=profile.publication_audit.audit_sha256,
        assignments=assignments,
    )


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    device: torch.device
    precision: Literal["fp32", "bf16", "fp16"]
    autocast_dtype: torch.dtype | None
    scaler_enabled: bool


def resolve_precision(
    device_request: str,
    precision_request: Literal["auto", "fp32", "bf16", "fp16"],
) -> PrecisionSpec:
    """Resolve the single-process device and protected autocast mode."""

    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as exc:
        raise TrainingConfigurationError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise TrainingConfigurationError("rotation training supports WORLD_SIZE=1 only")

    normalized_device = device_request.strip().lower()
    if normalized_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(device_request)
        except (TypeError, RuntimeError) as exc:
            raise TrainingConfigurationError(
                f"invalid device request {device_request!r}"
            ) from exc
    if device.type not in {"cpu", "cuda"}:
        raise TrainingConfigurationError("device must resolve to CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingConfigurationError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if (
        device.type == "cuda"
        and device.index is not None
        and not 0 <= device.index < torch.cuda.device_count()
    ):
        raise TrainingConfigurationError(
            f"CUDA device index {device.index} is unavailable"
        )

    if precision_request == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                precision: Literal["fp32", "bf16", "fp16"] = "bf16"
            else:
                precision = "fp16"
        else:
            precision = "fp32"
    else:
        precision = precision_request
    if precision in {"bf16", "fp16"} and device.type != "cuda":
        raise TrainingConfigurationError(
            f"{precision} training requires a CUDA device"
        )
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise TrainingConfigurationError(
            "bf16 was requested but the CUDA device does not support it"
        )
    dtype = (
        torch.bfloat16
        if precision == "bf16"
        else torch.float16
        if precision == "fp16"
        else None
    )
    return PrecisionSpec(
        device=device,
        precision=precision,
        autocast_dtype=dtype,
        scaler_enabled=precision == "fp16",
    )


def _autocast_context(spec: PrecisionSpec):
    if spec.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=spec.device.type,
        dtype=spec.autocast_dtype,
        enabled=True,
    )


@dataclass(frozen=True, slots=True)
class OptimizerContract:
    decay_parameter_names: tuple[str, ...]
    no_decay_parameter_names: tuple[str, ...]
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def build_adamw_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> tuple[torch.optim.AdamW, OptimizerContract]:
    """Create stable AdamW groups from sorted, named parameters."""

    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        excluded = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "relative_lag_bias" in name
        )
        (no_decay if excluded else decay).append((name, parameter))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in decay],
                "weight_decay": float(config.weight_decay),
            },
            {
                "params": [parameter for _, parameter in no_decay],
                "weight_decay": 0.0,
            },
        ],
        lr=float(config.learning_rate),
        betas=tuple(float(value) for value in config.adamw_betas),
        eps=float(config.adamw_epsilon),
    )
    contract = OptimizerContract(
        decay_parameter_names=tuple(name for name, _ in decay),
        no_decay_parameter_names=tuple(name for name, _ in no_decay),
        learning_rate=float(config.learning_rate),
        betas=tuple(float(value) for value in config.adamw_betas),
        epsilon=float(config.adamw_epsilon),
        weight_decay=float(config.weight_decay),
    )
    return optimizer, contract


class WarmupCosineScheduler:
    """Linear warm-up followed by cosine decay to a fixed LR ratio.

    ``step_index`` is the LR assigned to the *next* optimizer update. The
    constructor installs ``lr(0)``; every successful update calls ``step()``
    exactly once, leaving ``lr(S)`` installed after the final update.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        max_lr: float,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if (
            type(warmup_steps) is not int
            or warmup_steps < 0
            or warmup_steps > total_steps
        ):
            raise ValueError("warmup_steps must be in [0, total_steps]")
        if not math.isfinite(max_lr) or max_lr <= 0.0:
            raise ValueError("max_lr must be positive")
        if not math.isfinite(min_lr_ratio) or not 0.0 < min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = float(max_lr)
        self.min_lr_ratio = float(min_lr_ratio)
        self.step_index = 0
        self._set_lr(self.lr_at(0))

    def lr_at(self, step: int) -> float:
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise ValueError("scheduler step must be in [0, total_steps]")
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * (step / self.warmup_steps)
        cosine_span = self.total_steps - self.warmup_steps
        if cosine_span == 0:
            return self.max_lr * self.min_lr_ratio
        progress = (step - self.warmup_steps) / cosine_span
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return self.max_lr * ratio

    def _set_lr(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self) -> float:
        if self.step_index >= self.total_steps:
            raise RuntimeError("scheduler advanced beyond total_steps")
        self.step_index += 1
        value = self.lr_at(self.step_index)
        self._set_lr(value)
        return value

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "min_lr_ratio": self.min_lr_ratio,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        contract = {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "min_lr_ratio": self.min_lr_ratio,
        }
        for key, expected in contract.items():
            if state.get(key) != expected:
                raise TrainingCompatibilityError(
                    f"scheduler {key} mismatch: {state.get(key)!r} != {expected!r}"
                )
        step_index = state.get("step_index")
        if type(step_index) is not int or not 0 <= step_index <= self.total_steps:
            raise TrainingCompatibilityError("invalid scheduler step_index")
        self.step_index = step_index
        self._set_lr(self.lr_at(step_index))


@dataclass(frozen=True, slots=True)
class WindowRequest:
    session_id: str
    start_tick: int
    length: int


def training_window_requests(
    session_lengths: Mapping[str, int],
    session_ids: Sequence[str],
    *,
    seed: int,
    epoch: int,
    window_length_ticks: int,
    window_stride_ticks: int,
) -> tuple[WindowRequest, ...]:
    """Choose one uniformly sampled, stride-aligned window per session."""

    ordered = sorted(session_ids)
    order_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:epoch:{epoch}".encode()).digest()[:8],
        "big",
    )
    random.Random(order_seed).shuffle(ordered)
    requests: list[WindowRequest] = []
    for session_id in ordered:
        length = session_lengths[session_id]
        if length <= 0:
            raise TrainingConfigurationError(
                f"session {session_id} has no sampled ticks"
            )
        if length <= window_length_ticks:
            requests.append(WindowRequest(session_id, 0, length))
            continue
        starts = tuple(
            range(0, length - window_length_ticks + 1, window_stride_ticks)
        )
        window_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}:epoch:{epoch}:session:{session_id}".encode()
            ).digest()[:8],
            "big",
        )
        start = starts[random.Random(window_seed).randrange(len(starts))]
        requests.append(
            WindowRequest(session_id, start, window_length_ticks)
        )
    return tuple(requests)


def validation_window_requests(
    session_lengths: Mapping[str, int],
    session_ids: Sequence[str],
    *,
    window_length_ticks: int,
) -> tuple[WindowRequest, ...]:
    requests: list[WindowRequest] = []
    for session_id in sorted(session_ids):
        length = session_lengths[session_id]
        if length <= 0:
            raise TrainingConfigurationError(
                f"session {session_id} has no sampled ticks"
            )
        for start in range(0, length, window_length_ticks):
            requests.append(
                WindowRequest(
                    session_id,
                    start,
                    min(window_length_ticks, length - start),
                )
            )
    return tuple(requests)


@dataclass(frozen=True, slots=True)
class _FullSession:
    match: TensorizedMatch
    supervision: RotationSupervision


def _load_full_session(
    record: _SessionRecord,
    profile: WorldGridProfile,
    head_config: RotationHeadConfig,
) -> _FullSession:
    match = load_match_session(record.path)
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
    return _FullSession(match=match, supervision=supervision)


def _slice_full_session(
    full: _FullSession,
    request: WindowRequest,
) -> tuple[TensorizedMatch, RotationSupervision]:
    match = slice_window(full.match, request.start_tick, request.length)
    supervision = slice_rotation_supervision(
        full.supervision,
        request.start_tick,
        request.length,
    )
    if not torch.equal(
        match.absolute_tick_index.cpu(),
        supervision.absolute_tick_index.cpu(),
    ):
        raise TrainingConfigurationError(
            f"encoder/target tick misalignment for {request.session_id}"
        )
    if (
        match.session_id != supervision.metadata.session_id
        or match.team_ids != supervision.metadata.team_ids
    ):
        raise TrainingConfigurationError(
            f"encoder/target metadata misalignment for {request.session_id}"
        )
    return match, supervision


def collate_training_windows(
    items: Sequence[tuple[TensorizedMatch, RotationSupervision]],
    *,
    expected_profile_id: str,
    expected_profile_hash: str,
    head_config: RotationHeadConfig,
) -> tuple[CollatedEncoderInput, CollatedRotationSupervision]:
    if not items:
        raise ValueError("at least one training window is required")
    for match, supervision in items:
        if (
            match.session_id != supervision.metadata.session_id
            or match.team_ids != supervision.metadata.team_ids
            or not torch.equal(
                match.absolute_tick_index.cpu(),
                supervision.absolute_tick_index.cpu(),
            )
        ):
            raise TrainingConfigurationError("encoder and target windows are misaligned")
        if (
            supervision.metadata.world_grid_profile_id != expected_profile_id
            or supervision.metadata.world_grid_profile_hash
            != expected_profile_hash
        ):
            raise TrainingConfigurationError(
                f"session {match.session_id} uses a mixed or unbound world-grid profile"
            )
    encoder = collate_encoder_inputs(match for match, _ in items)
    supervision = collate_rotation_supervision(item for _, item in items)
    validate_training_batch(
        encoder,
        supervision,
        expected_profile_id=expected_profile_id,
        expected_profile_hash=expected_profile_hash,
        head_config=head_config,
    )
    return encoder, supervision


def _assert_finite_masked(
    name: str,
    tensor: Tensor,
    mask: Tensor,
    session_ids: Sequence[str],
) -> None:
    expanded_mask = mask
    while expanded_mask.ndim < tensor.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(tensor)
    invalid_count = int((~torch.isfinite(tensor) & expanded_mask).sum().item())
    if invalid_count:
        diagnostic = {
            "kind": "nonfinite_input",
            "tensor": name,
            "invalid_count": invalid_count,
            "valid_location_count": int(expanded_mask.sum().item()),
            "sessions": list(session_ids[:8]),
        }
        raise TrainingNumericalError(_canonical_json(diagnostic))


def validate_training_batch(
    encoder: CollatedEncoderInput,
    supervision: CollatedRotationSupervision,
    *,
    expected_profile_id: str,
    expected_profile_hash: str,
    head_config: RotationHeadConfig,
) -> None:
    """Validate metadata, class contracts, masks, and semantic finite inputs."""

    if head_config.horizons_seconds != TRAINING_HORIZONS_SECONDS:
        raise TrainingConfigurationError(
            "training batch horizons must be exactly (15, 30, 60)"
        )
    if (
        encoder.metadata.session_ids != supervision.metadata.session_ids
        or encoder.metadata.team_ids != supervision.metadata.team_ids
    ):
        raise TrainingConfigurationError("collated encoder/target metadata misalignment")
    if any(
        profile != expected_profile_id
        for profile in supervision.metadata.world_grid_profile_ids
    ):
        raise TrainingConfigurationError(
            "mixed or unbound world-grid profiles in batch"
        )
    if any(
        profile_hash != expected_profile_hash
        for profile_hash in supervision.metadata.world_grid_profile_hashes
    ):
        raise TrainingConfigurationError(
            "world-grid profile hash mismatch in batch"
        )
    batch = encoder.batch
    targets = supervision.targets
    query_axes = batch.player_xyz_uu.shape[:3]
    if targets.future_position.shape != (*query_axes, 3):
        raise TrainingConfigurationError("future-position query axes are invalid")
    if targets.zone_entry.shape != query_axes:
        raise TrainingConfigurationError("zone-entry query axes are invalid")
    if head_config.enable_survival != (targets.survival is not None):
        raise TrainingConfigurationError("survival target enablement mismatch")
    if head_config.enable_survival != (targets.survival_mask is not None):
        raise TrainingConfigurationError("survival mask enablement mismatch")
    if head_config.enable_placement != (targets.placement is not None):
        raise TrainingConfigurationError("placement target enablement mismatch")
    if head_config.enable_placement != (targets.placement_mask is not None):
        raise TrainingConfigurationError("placement mask enablement mismatch")

    position_mask = targets.future_position_mask
    if position_mask.any():
        labels = targets.future_position[position_mask]
        if int(labels.min()) < 0 or int(labels.max()) >= head_config.num_position_classes:
            raise TrainingConfigurationError("future-position label is out of range")
    if targets.zone_entry_mask.any():
        labels = targets.zone_entry[targets.zone_entry_mask]
        if int(labels.min()) < 0 or int(labels.max()) >= head_config.num_entry_classes:
            raise TrainingConfigurationError("zone-entry label is out of range")
    if targets.survival is not None and targets.survival_mask is not None:
        values = targets.survival[targets.survival_mask]
        if values.numel() and (
            not bool(torch.isfinite(values).all())
            or not bool(((values == 0.0) | (values == 1.0)).all())
        ):
            raise TrainingConfigurationError("survival labels must be finite binary values")
    if targets.placement is not None and targets.placement_mask is not None:
        labels = targets.placement[targets.placement_mask]
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= 5):
            raise TrainingConfigurationError("placement label is out of range")

    time_team = batch.time_mask[:, :, None] & batch.team_slot_mask[:, None, :]
    roster = batch.player_slot_mask[:, None, :, :]
    valid_player_coordinates = (
        time_team[:, :, :, None]
        & roster
        & batch.player_alive
        & batch.player_coord_mask
    )
    _assert_finite_masked(
        "player_xyz_uu",
        batch.player_xyz_uu,
        valid_player_coordinates,
        encoder.metadata.session_ids,
    )
    _assert_finite_masked(
        "match_elapsed_s",
        batch.match_elapsed_s,
        batch.time_mask,
        encoder.metadata.session_ids,
    )
    zone_valid = batch.time_mask & batch.zone_mask
    for name in ("current_circle_uu", "target_circle_uu", "phase_times_s"):
        _assert_finite_masked(
            name,
            getattr(batch, name),
            zone_valid,
            encoder.metadata.session_ids,
        )
    prior_valid = (
        batch.prior_state_available[:, None, None]
        & batch.team_slot_mask[:, :, None]
        & batch.player_slot_mask
        & batch.prior_player_alive
        & batch.prior_player_coord_mask
    )
    _assert_finite_masked(
        "prior_player_xyz_uu",
        batch.prior_player_xyz_uu,
        prior_valid,
        encoder.metadata.session_ids,
    )


class LossMetricAccumulator:
    """Count-weighted reconstruction of the existing independent mean losses."""

    def __init__(self, loss_config: RotationLossConfig) -> None:
        self.loss_config = loss_config
        self.horizon_numerators = [0.0, 0.0, 0.0]
        self.horizon_counts = [0, 0, 0]
        self.entry_numerator = 0.0
        self.entry_count = 0
        self.survival_numerator = 0.0
        self.survival_count = 0
        self.placement_numerator = 0.0
        self.placement_count = 0

    def update(self, loss: RotationLossOutput) -> None:
        if len(loss.future_position_by_horizon) != 3:
            raise TrainingConfigurationError("loss output must contain three horizons")
        for index, (value, count) in enumerate(
            zip(
                loss.future_position_by_horizon,
                loss.future_position_valid_counts,
            )
        ):
            self.horizon_numerators[index] += float(value.detach().cpu()) * count
            self.horizon_counts[index] += count
        self.entry_numerator += (
            float(loss.zone_entry.detach().cpu()) * loss.zone_entry_valid_count
        )
        self.entry_count += loss.zone_entry_valid_count
        self.survival_numerator += (
            float(loss.survival.detach().cpu()) * loss.survival_valid_count
        )
        self.survival_count += loss.survival_valid_count
        self.placement_numerator += (
            float(loss.placement.detach().cpu()) * loss.placement_valid_count
        )
        self.placement_count += loss.placement_valid_count

    @staticmethod
    def _mean(numerator: float, count: int) -> float | None:
        return numerator / count if count else None

    def metrics(self) -> dict[str, Any]:
        horizon_means = [
            self._mean(numerator, count)
            for numerator, count in zip(
                self.horizon_numerators,
                self.horizon_counts,
            )
        ]
        future = (
            sum(value for value in horizon_means if value is not None)
            if any(value is not None for value in horizon_means)
            else None
        )
        entry = self._mean(self.entry_numerator, self.entry_count)
        survival = self._mean(self.survival_numerator, self.survival_count)
        placement = self._mean(self.placement_numerator, self.placement_count)
        total: float | None = None
        if any(value is not None for value in (future, entry, survival, placement)):
            total = future if future is not None else 0.0
            if entry is not None:
                total += float(self.loss_config.entry_weight) * entry
            if survival is not None:
                total += float(self.loss_config.survival_weight) * survival
            if placement is not None:
                total += float(self.loss_config.placement_weight) * placement
        return {
            "future_position_15s": horizon_means[0],
            "future_position_30s": horizon_means[1],
            "future_position_60s": horizon_means[2],
            "future_position": future,
            "zone_entry": entry,
            "survival": survival,
            "placement": placement,
            "total": total,
            "counts": {
                "future_position_15s": self.horizon_counts[0],
                "future_position_30s": self.horizon_counts[1],
                "future_position_60s": self.horizon_counts[2],
                "zone_entry": self.entry_count,
                "survival": self.survival_count,
                "placement": self.placement_count,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "horizon_numerators": list(self.horizon_numerators),
            "horizon_counts": list(self.horizon_counts),
            "entry_numerator": self.entry_numerator,
            "entry_count": self.entry_count,
            "survival_numerator": self.survival_numerator,
            "survival_count": self.survival_count,
            "placement_numerator": self.placement_numerator,
            "placement_count": self.placement_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "horizon_numerators",
            "horizon_counts",
            "entry_numerator",
            "entry_count",
            "survival_numerator",
            "survival_count",
            "placement_numerator",
            "placement_count",
        }
        if set(state) != expected:
            raise TrainingCompatibilityError("metric accumulator state is invalid")
        numerators = state["horizon_numerators"]
        counts = state["horizon_counts"]
        if (
            not isinstance(numerators, list)
            or not isinstance(counts, list)
            or len(numerators) != 3
            or len(counts) != 3
        ):
            raise TrainingCompatibilityError(
                "metric accumulator horizon state is invalid"
            )
        self.horizon_numerators = [float(value) for value in numerators]
        self.horizon_counts = [int(value) for value in counts]
        self.entry_numerator = float(state["entry_numerator"])
        self.entry_count = int(state["entry_count"])
        self.survival_numerator = float(state["survival_numerator"])
        self.survival_count = int(state["survival_count"])
        self.placement_numerator = float(state["placement_numerator"])
        self.placement_count = int(state["placement_count"])


def _loss_has_supervision(loss: RotationLossOutput) -> bool:
    return (
        sum(loss.future_position_valid_counts)
        + loss.zone_entry_valid_count
        + loss.survival_valid_count
        + loss.placement_valid_count
        > 0
    )


def _validate_finite_logits_and_loss(
    model_output: Any,
    loss: RotationLossOutput,
    session_ids: Sequence[str],
) -> None:
    logits = model_output.logits
    named_logits = {
        "future_position": logits.future_position,
        "zone_entry": logits.zone_entry,
        "survival": logits.survival,
        "placement": logits.placement,
    }
    for name, tensor in named_logits.items():
        if tensor is not None and not bool(torch.isfinite(tensor).all()):
            diagnostic = {
                "kind": "nonfinite_logits",
                "tensor": name,
                "invalid_count": int((~torch.isfinite(tensor)).sum().item()),
                "element_count": tensor.numel(),
                "sessions": list(session_ids[:8]),
            }
            raise TrainingNumericalError(_canonical_json(diagnostic))
    named_losses = {
        "total": loss.total,
        "future_position": loss.future_position,
        "zone_entry": loss.zone_entry,
        "survival": loss.survival,
        "placement": loss.placement,
    }
    for index, value in enumerate(loss.future_position_by_horizon):
        named_losses[f"future_position_{TRAINING_HORIZONS_SECONDS[index]}s"] = value
    for name, tensor in named_losses.items():
        if not bool(torch.isfinite(tensor).all()):
            diagnostic = {
                "kind": "nonfinite_loss",
                "component": name,
                "sessions": list(session_ids[:8]),
                "counts": loss.valid_label_counts,
            }
            raise TrainingNumericalError(_canonical_json(diagnostic))


def _gradient_norm_and_finite(
    model: nn.Module,
    session_ids: Sequence[str],
) -> float:
    squared_norm = torch.zeros((), dtype=torch.float64)
    invalid_names: list[str] = []
    gradient_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_count += 1
        if not bool(torch.isfinite(gradient).all()):
            invalid_names.append(name)
            continue
        norm = torch.linalg.vector_norm(gradient.detach().to(dtype=torch.float64))
        squared_norm += norm.cpu().square()
    if invalid_names:
        diagnostic = {
            "kind": "nonfinite_gradient",
            "parameter_count": len(invalid_names),
            "parameters": invalid_names[:16],
            "sessions": list(session_ids[:8]),
        }
        raise TrainingNumericalError(_canonical_json(diagnostic))
    if gradient_count == 0:
        raise TrainingNumericalError(
            _canonical_json(
                {
                    "kind": "missing_gradients",
                    "sessions": list(session_ids[:8]),
                }
            )
        )
    norm_value = math.sqrt(float(squared_norm))
    if not math.isfinite(norm_value):
        raise TrainingNumericalError(
            _canonical_json(
                {
                    "kind": "nonfinite_gradient_norm",
                    "sessions": list(session_ids[:8]),
                }
            )
        )
    return norm_value


def _validate_scaled_gradients(
    model: nn.Module,
    session_ids: Sequence[str],
) -> None:
    """Reject corrupt pending gradients before they can be checkpointed."""

    invalid_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
    ]
    if invalid_names:
        raise TrainingNumericalError(
            _canonical_json(
                {
                    "kind": "nonfinite_scaled_gradient",
                    "parameter_count": len(invalid_names),
                    "parameters": invalid_names[:16],
                    "sessions": list(session_ids[:8]),
                }
            )
        )


@dataclass(slots=True)
class TrainingState:
    epoch: int = 0
    microbatch_index: int = 0
    optimizer_step: int = 0
    data_microbatches: int = 0
    supervised_microbatches: int = 0
    accumulation_count: int = 0
    best_metric: float | None = None
    best_optimizer_step: int | None = None
    non_improvements: int = 0
    stopped_early: bool = False
    last_validation_optimizer_step: int | None = None
    last_validation_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrainingState:
        expected = {item.name for item in fields(cls)}
        if set(raw) != expected:
            raise TrainingCompatibilityError(
                "checkpoint training state fields do not match"
            )
        try:
            return cls(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise TrainingCompatibilityError(
                f"invalid checkpoint training state: {exc}"
            ) from exc


def _read_source_revision(start: Path) -> str | None:
    current = start.resolve()
    for directory in (current, *current.parents):
        marker = directory / ".git"
        git_directory = marker
        if marker.is_file():
            try:
                line = marker.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if not line.startswith("gitdir:"):
                return None
            candidate = Path(line.split(":", 1)[1].strip())
            git_directory = (
                candidate
                if candidate.is_absolute()
                else (directory / candidate).resolve()
            )
        if not git_directory.is_dir():
            continue
        try:
            head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                reference = head[5:]
                ref_path = git_directory / reference
                if ref_path.is_file():
                    revision = ref_path.read_text(encoding="utf-8").strip()
                else:
                    revision = ""
                    packed = git_directory / "packed-refs"
                    if packed.is_file():
                        for line in packed.read_text(encoding="utf-8").splitlines():
                            if line.endswith(f" {reference}"):
                                revision = line.split(" ", 1)[0]
                                break
                return revision if len(revision) == 40 else None
            return head if len(head) == 40 else None
        except OSError:
            return None
    return None


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _append_jsonl_atomic(path: Path, value: Any) -> None:
    existing = path.read_bytes() if path.is_file() else b""
    line = (_canonical_json(value) + "\n").encode("utf-8")
    _atomic_write_bytes(path, existing + line)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise TrainingCompatibilityError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise TrainingCompatibilityError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _pending_gradients(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _restore_pending_gradients(
    model: nn.Module,
    gradients: Mapping[str, Tensor],
    accumulation_count: int,
) -> None:
    parameters = dict(model.named_parameters())
    unknown = sorted(set(gradients) - set(parameters))
    if unknown:
        raise TrainingCompatibilityError(
            f"checkpoint has unknown pending gradients: {unknown[:8]}"
        )
    for parameter in parameters.values():
        parameter.grad = None
    for name, gradient in gradients.items():
        parameter = parameters[name]
        if gradient.shape != parameter.shape:
            raise TrainingCompatibilityError(
                f"pending gradient shape mismatch for {name}"
            )
        parameter.grad = gradient.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).clone()
    if accumulation_count == 0 and gradients:
        raise TrainingCompatibilityError(
            "checkpoint has gradients with zero accumulation_count"
        )
    if accumulation_count > 0 and not gradients:
        raise TrainingCompatibilityError(
            "checkpoint has pending accumulation without gradients"
        )


def _class_contract(head: RotationHeadConfig) -> dict[str, Any]:
    return {
        "horizons_seconds": list(head.horizons_seconds),
        "position_classes": head.num_position_classes,
        "entry_classes": head.num_entry_classes,
        "survival_enabled": head.enable_survival,
        "placement_enabled": head.enable_placement,
        "placement_classes": 5 if head.enable_placement else None,
    }


def _compatibility_contract(
    resolved: ResolvedTrainingConfig,
    optimizer_contract: OptimizerContract,
) -> dict[str, Any]:
    training = resolved.training
    source_revision = resolved.source_revision
    return {
        "encoder_config": _jsonable(training.encoder_config),
        "head_config": _jsonable(training.head_config),
        "loss_config": _jsonable(training.loss_config),
        "class_contract": _class_contract(training.head_config),
        "dataset_schema_version": resolved.dataset_schema_version,
        "ingestion_report_sha256": resolved.ingestion_report_sha256,
        "split_manifest_sha256": resolved.split_manifest_sha256,
        "world_grid_profile_hash": resolved.world_grid_profile_hash,
        "world_grid_audit_sha256": resolved.world_grid_audit_sha256,
        "precision": resolved.precision,
        "device": resolved.device,
        "sampler": {
            "seed": training.seed,
            "window_length_ticks": training.window_length_ticks,
            "window_stride_ticks": training.window_stride_ticks,
            "batch_size": training.batch_size,
            "accumulation_steps": training.accumulation_steps,
        },
        "training_control": {
            "max_epochs": training.max_epochs,
            "max_optimizer_steps": training.max_optimizer_steps,
            "gradient_clip_norm": training.gradient_clip_norm,
            "early_stopping_metric": training.early_stopping_metric,
            "early_stopping_patience": training.early_stopping_patience,
            "early_stopping_min_delta": training.early_stopping_min_delta,
            "validation_every_epochs": training.validation_every_epochs,
        },
        "optimizer": optimizer_contract.to_dict(),
        "scheduler": {
            "total_steps": resolved.total_optimizer_steps,
            "warmup_steps": resolved.warmup_steps,
            "max_lr": training.learning_rate,
            "min_lr_ratio": training.scheduler_min_lr_ratio,
        },
        "source_revision": source_revision,
    }


def _checkpoint_payload(
    *,
    resolved: ResolvedTrainingConfig,
    manifest: SplitManifest,
    model: RotationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
    train_metrics: LossMetricAccumulator,
    optimizer_contract: OptimizerContract,
) -> dict[str, Any]:
    history_path = Path(resolved.run_directory) / "history.jsonl"
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "wrapper": {
            "type": "RotationModel",
            "training": model.training,
            "state_dict": model.state_dict(),
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "pending_gradients": _pending_gradients(model),
        "training_state": state.to_dict(),
        "epoch_metric_state": train_metrics.state_dict(),
        "rng_state": _rng_state(),
        "history_jsonl": (
            history_path.read_bytes() if history_path.is_file() else b""
        ),
        "resolved_config": resolved.to_dict(),
        "split_manifest": manifest.to_dict(),
        "loss_weights": _jsonable(resolved.training.loss_config),
        "class_contract": _class_contract(resolved.training.head_config),
        "compatibility": _compatibility_contract(
            resolved,
            optimizer_contract,
        ),
    }


def _save_checkpoint(
    path: Path,
    **kwargs: Any,
) -> None:
    _atomic_torch_save(path, _checkpoint_payload(**kwargs))


def _load_checkpoint(
    path: Path,
    *,
    resolved: ResolvedTrainingConfig,
    model: RotationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    train_metrics: LossMetricAccumulator,
    optimizer_contract: OptimizerContract,
) -> TrainingState:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingCompatibilityError(
            f"cannot load checkpoint {path}: {exc}"
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_schema_version")
        != CHECKPOINT_SCHEMA_VERSION
    ):
        raise TrainingCompatibilityError("unsupported checkpoint schema")
    expected = _compatibility_contract(resolved, optimizer_contract)
    actual = checkpoint.get("compatibility")
    if not isinstance(actual, dict):
        raise TrainingCompatibilityError("checkpoint compatibility state is missing")
    if _canonical_json(actual) != _canonical_json(expected):
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise TrainingCompatibilityError(
            "checkpoint compatibility mismatch: " + ", ".join(differing)
        )
    wrapper = checkpoint.get("wrapper")
    if (
        not isinstance(wrapper, dict)
        or wrapper.get("type") != "RotationModel"
        or type(wrapper.get("training")) is not bool
    ):
        raise TrainingCompatibilityError("checkpoint model wrapper is invalid")
    try:
        model.load_state_dict(wrapper["state_dict"], strict=True)
        model.train(wrapper["training"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise TrainingCompatibilityError(
            f"checkpoint state cannot be restored exactly: {exc}"
        ) from exc
    state = TrainingState.from_dict(checkpoint["training_state"])
    if state.optimizer_step != scheduler.step_index:
        raise TrainingCompatibilityError(
            "optimizer and scheduler counters do not match"
        )
    if not 0 <= state.accumulation_count < resolved.training.accumulation_steps:
        raise TrainingCompatibilityError("invalid accumulation_count")
    train_metrics.load_state_dict(checkpoint["epoch_metric_state"])
    gradients = checkpoint.get("pending_gradients")
    if not isinstance(gradients, dict):
        raise TrainingCompatibilityError("pending gradient state is invalid")
    _restore_pending_gradients(model, gradients, state.accumulation_count)
    history_jsonl = checkpoint.get("history_jsonl")
    if not isinstance(history_jsonl, bytes):
        raise TrainingCompatibilityError("checkpoint history state is invalid")
    _atomic_write_bytes(
        Path(resolved.run_directory) / "history.jsonl",
        history_jsonl,
    )
    # RNG restoration is deliberately last: loading must not perturb continuation.
    _restore_rng_state(checkpoint["rng_state"])
    return state


class _SessionRepository:
    def __init__(
        self,
        records: Sequence[_SessionRecord],
        profile: WorldGridProfile,
        head_config: RotationHeadConfig,
        num_workers: int,
        persistent_workers: bool,
    ) -> None:
        self.records = {record.session_id: record for record in records}
        self.profile = profile
        self.head_config = head_config
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.cache: dict[str, _FullSession] = {}
        self._executor: Any = None

    def _full(self, session_id: str) -> _FullSession:
        cached = self.cache.get(session_id)
        if cached is None:
            cached = _load_full_session(
                self.records[session_id],
                self.profile,
                self.head_config,
            )
            self.cache[session_id] = cached
        return cached

    def _window(
        self,
        request: WindowRequest,
    ) -> tuple[TensorizedMatch, RotationSupervision]:
        return _slice_full_session(self._full(request.session_id), request)

    def windows(
        self,
        requests: Sequence[WindowRequest],
    ) -> list[tuple[TensorizedMatch, RotationSupervision]]:
        if self.num_workers == 0:
            return [self._window(request) for request in requests]
        from concurrent.futures import ThreadPoolExecutor

        if self.persistent_workers:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
            executor = self._executor
            return list(executor.map(self._window, requests))
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            return list(executor.map(self._window, requests))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def _session_lengths(
    records: Sequence[_SessionRecord],
    encoder_config: EncoderConfig,
) -> dict[str, int]:
    output: dict[str, int] = {}
    for record in records:
        path = record.path / "match_samples.parquet"
        if not path.is_file():
            raise TrainingConfigurationError(
                f"required match table is missing: {path}"
            )
        try:
            count = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:
            raise TrainingConfigurationError(
                f"cannot inspect match table {path}: {exc}"
            ) from exc
        if count <= 0:
            raise TrainingConfigurationError(
                f"session {record.session_id} contains no match samples"
            )
        if count > encoder_config.max_timesteps:
            raise TrainingConfigurationError(
                f"session {record.session_id} has {count} ticks but "
                f"encoder max_timesteps is {encoder_config.max_timesteps}"
            )
        output[record.session_id] = count
    return output


def _batch_requests(
    requests: Sequence[WindowRequest],
    batch_size: int,
) -> tuple[tuple[WindowRequest, ...], ...]:
    return tuple(
        tuple(requests[start : start + batch_size])
        for start in range(0, len(requests), batch_size)
    )


def _pin_encoder_batch(batch: EncoderBatch) -> EncoderBatch:
    return EncoderBatch(
        **{
            item.name: getattr(batch, item.name).pin_memory()
            for item in fields(batch)
        }
    )


def _move_batch(
    encoder: CollatedEncoderInput,
    supervision: CollatedRotationSupervision,
    spec: PrecisionSpec,
    pin_memory: bool,
) -> tuple[CollatedEncoderInput, CollatedRotationSupervision]:
    if pin_memory and spec.device.type == "cuda":
        encoder = dataclasses.replace(
            encoder,
            batch=_pin_encoder_batch(encoder.batch),
        )
        target_values = {
            item.name: (
                value.pin_memory() if value is not None else None
            )
            for item in fields(supervision.targets)
            for value in (getattr(supervision.targets, item.name),)
        }
        supervision = dataclasses.replace(
            supervision,
            targets=type(supervision.targets)(**target_values),
        )
    non_blocking = pin_memory and spec.device.type == "cuda"
    return (
        encoder.to(spec.device, non_blocking=non_blocking),
        supervision.to(spec.device, non_blocking=non_blocking),
    )


def _derive_schedule(
    config: TrainingConfig,
    train_session_count: int,
) -> tuple[int, int, int]:
    assert config.batch_size is not None
    microbatches = math.ceil(train_session_count / config.batch_size)
    by_epochs: int | None = None
    if config.max_epochs is not None:
        by_epochs = (
            microbatches * config.max_epochs
        ) // config.accumulation_steps
    if config.max_optimizer_steps is not None and by_epochs is not None:
        total = min(config.max_optimizer_steps, by_epochs)
    elif config.max_optimizer_steps is not None:
        total = config.max_optimizer_steps
    else:
        assert by_epochs is not None
        total = by_epochs
    if total <= 0:
        raise TrainingConfigurationError(
            "training limits cannot produce one complete optimizer update"
        )
    warmup = (
        config.scheduler_warmup_steps
        if config.scheduler_warmup_steps is not None
        else math.floor(config.scheduler_warmup_fraction * total)
    )
    if warmup > total:
        raise TrainingConfigurationError(
            "scheduler_warmup_steps exceeds total optimizer steps"
        )
    return microbatches, total, warmup


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def _resolved_config(
    config: TrainingConfig,
    *,
    dataset_schema_version: str,
    ingestion_report_sha256: str,
    manifest: SplitManifest,
    precision: PrecisionSpec,
    source_revision: str | None,
    microbatches_per_epoch: int,
    total_optimizer_steps: int,
    warmup_steps: int,
) -> ResolvedTrainingConfig:
    counts = {
        partition: len(manifest.session_ids(partition))
        for partition in _PARTITIONS
    }
    return ResolvedTrainingConfig(
        training=config,
        dataset_root=str(Path(config.dataset_root).resolve()),
        run_directory=str(Path(config.run_directory).resolve()),
        dataset_schema_version=dataset_schema_version,
        ingestion_report_sha256=ingestion_report_sha256,
        world_grid_profile_hash=manifest.world_grid_profile_hash,
        world_grid_audit_sha256=manifest.world_grid_audit_sha256,
        split_manifest_sha256=manifest.fingerprint,
        source_revision=source_revision,
        device=str(precision.device),
        precision=precision.precision,
        scaler_enabled=precision.scaler_enabled,
        discovered_session_count=len(manifest.assignments),
        train_session_count=counts["train"],
        validation_session_count=counts["validation"],
        test_session_count=counts["test"],
        microbatches_per_epoch=microbatches_per_epoch,
        total_optimizer_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        effective_encoder_dropouts={
            "feature_projection": (
                config.encoder_config.effective_feature_projection_dropout
            ),
            "spatial_attention": (
                config.encoder_config.effective_spatial_attention_dropout
            ),
            "spatial_residual_and_ffn": (
                config.encoder_config.effective_spatial_residual_dropout
            ),
            "temporal_attention": (
                config.encoder_config.effective_temporal_attention_dropout
            ),
            "temporal_residual_and_ffn": (
                config.encoder_config.effective_temporal_residual_dropout
            ),
            "head_hidden": float(config.head_config.dropout),
        },
        class_contract=_class_contract(config.head_config),
    )


def _environment_metadata(
    resolved: ResolvedTrainingConfig,
) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": resolved.device,
        "precision": resolved.precision,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "source_revision": resolved.source_revision,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _checkpoint_kwargs(
    *,
    resolved: ResolvedTrainingConfig,
    manifest: SplitManifest,
    model: RotationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    state: TrainingState,
    train_metrics: LossMetricAccumulator,
    optimizer_contract: OptimizerContract,
) -> dict[str, Any]:
    return {
        "resolved": resolved,
        "manifest": manifest,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "state": state,
        "train_metrics": train_metrics,
        "optimizer_contract": optimizer_contract,
    }


def _write_latest_metadata(
    run_directory: Path,
    state: TrainingState,
    checkpoint_name: str,
) -> None:
    _atomic_write_json(
        run_directory / "latest.json",
        {
            "checkpoint": checkpoint_name,
            "epoch": state.epoch,
            "microbatch_index": state.microbatch_index,
            "optimizer_step": state.optimizer_step,
            "data_microbatches": state.data_microbatches,
            "accumulation_count": state.accumulation_count,
        },
    )


def _evaluate(
    *,
    model: RotationModel,
    repository: _SessionRepository,
    requests: Sequence[WindowRequest],
    config: TrainingConfig,
    spec: PrecisionSpec,
) -> tuple[dict[str, Any], float]:
    assert config.batch_size is not None
    assert config.world_grid_profile is not None
    profile = config.world_grid_profile.load(config.dataset_root)
    accumulator = LossMetricAccumulator(config.loss_config)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for request_batch in _batch_requests(requests, config.batch_size):
            items = repository.windows(request_batch)
            encoder, supervision = collate_training_windows(
                items,
                expected_profile_id=profile.profile_id,
                expected_profile_hash=profile.profile_hash,
                head_config=config.head_config,
            )
            encoder, supervision = _move_batch(
                encoder,
                supervision,
                spec,
                config.pin_memory,
            )
            with _autocast_context(spec):
                output = model(encoder.batch)
                loss = compute_rotation_loss(
                    output.logits,
                    supervision.targets,
                    config.loss_config,
                )
            _validate_finite_logits_and_loss(
                output,
                loss,
                encoder.metadata.session_ids,
            )
            accumulator.update(loss)
    return accumulator.metrics(), time.perf_counter() - started


def _monitor_value(metrics: Mapping[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        raise TrainingConfigurationError(
            f"unknown early-stopping metric {metric_name!r}"
        )
    value = metrics[metric_name]
    if value is None:
        raise TrainingConfigurationError(
            f"early-stopping metric {metric_name!r} has no valid labels"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(
            f"early-stopping metric {metric_name!r} is not scalar"
        )
    return float(value)


def _prepare_config(config: TrainingConfig) -> TrainingConfig:
    def absolute(value: str | None) -> str | None:
        return str(Path(value).resolve()) if value is not None else None

    return dataclasses.replace(
        config,
        dataset_root=absolute(config.dataset_root),
        run_directory=absolute(config.run_directory),
        provenance_manifest_path=absolute(config.provenance_manifest_path),
        split_manifest_path=absolute(config.split_manifest_path),
        world_grid_profile=(
            dataclasses.replace(
                config.world_grid_profile,
                path=absolute(config.world_grid_profile.path),
                audit_path=absolute(config.world_grid_profile.audit_path),
            )
            if config.world_grid_profile is not None
            else None
        ),
    )


def run_rotation_training(
    config: TrainingConfig | str | Path,
    *,
    resume: Literal["last"] | str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic, single-process rotation-model training.

    The returned dictionary is the same canonical object written to
    ``final_summary.json``. Test-partition sessions are never loaded here.
    """

    if isinstance(config, (str, Path)):
        parsed = TrainingConfig.from_json(config)
    elif isinstance(config, TrainingConfig):
        parsed = _prepare_config(config)
    else:
        raise TypeError("config must be TrainingConfig or a JSON config path")
    config = parsed
    assert config.seed is not None
    assert config.batch_size is not None
    assert config.window_length_ticks is not None
    assert config.window_stride_ticks is not None
    assert config.early_stopping_patience is not None
    assert config.world_grid_profile is not None
    profile = config.world_grid_profile.load(config.dataset_root)

    run_directory = Path(config.run_directory)
    persisted_manifest_path = run_directory / "split_manifest.json"
    manifest_config = config
    if resume is not None and persisted_manifest_path.is_file():
        manifest_config = dataclasses.replace(
            config,
            split_manifest_path=str(persisted_manifest_path),
        )
    records, dataset_schema, report_hash = discover_dataset_sessions(config)
    manifest = build_split_manifest(manifest_config, records, report_hash)
    train_ids = manifest.session_ids("train")
    validation_ids = manifest.session_ids("validation")
    if not train_ids or not validation_ids:
        raise TrainingConfigurationError(
            "training requires nonempty train and validation partitions"
        )
    lengths = _session_lengths(records, config.encoder_config)
    microbatches, total_steps, warmup_steps = _derive_schedule(
        config,
        len(train_ids),
    )
    spec = resolve_precision(config.device, config.precision)
    source_revision = _read_source_revision(Path.cwd())
    resolved = _resolved_config(
        config,
        dataset_schema_version=dataset_schema,
        ingestion_report_sha256=report_hash,
        manifest=manifest,
        precision=spec,
        source_revision=source_revision,
        microbatches_per_epoch=microbatches,
        total_optimizer_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    resolved_path = run_directory / "resolved_config.json"
    if resume is None and resolved_path.exists():
        raise TrainingConfigurationError(
            f"run directory already contains a training run: {run_directory}"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    if resume is None:
        _atomic_write_json(resolved_path, resolved.to_dict())
        _atomic_write_json(persisted_manifest_path, manifest.to_dict())
        _atomic_write_json(
            run_directory / "environment.json",
            _environment_metadata(resolved),
        )

    model = RotationModel(
        encoder_config=config.encoder_config,
        head_config=config.head_config,
    ).to(spec.device)
    optimizer, optimizer_contract = build_adamw_optimizer(model, config)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_lr=config.learning_rate,
        min_lr_ratio=config.scheduler_min_lr_ratio,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=spec.scaler_enabled,
    )
    state = TrainingState()
    train_metrics = LossMetricAccumulator(config.loss_config)
    if resume is not None:
        resume_path = (
            run_directory / "last.pt"
            if str(resume) == "last"
            else Path(resume).resolve()
        )
        state = _load_checkpoint(
            resume_path,
            resolved=resolved,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_metrics=train_metrics,
            optimizer_contract=optimizer_contract,
        )
    else:
        optimizer.zero_grad(set_to_none=True)

    records_by_id = {record.session_id: record for record in records}
    active_records = tuple(
        records_by_id[session_id]
        for session_id in (*train_ids, *validation_ids)
    )
    repository = _SessionRepository(
        active_records,
        profile,
        config.head_config,
        config.num_workers,
        config.persistent_workers,
    )
    validation_requests = validation_window_requests(
        lengths,
        validation_ids,
        window_length_ticks=config.window_length_ticks,
    )
    checkpoint_arguments = lambda: _checkpoint_kwargs(
        resolved=resolved,
        manifest=manifest,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        train_metrics=train_metrics,
        optimizer_contract=optimizer_contract,
    )
    _save_checkpoint(
        run_directory / "last.pt",
        **checkpoint_arguments(),
    )
    _write_latest_metadata(run_directory, state, "last.pt")

    def save_last() -> None:
        _save_checkpoint(
            run_directory / "last.pt",
            **checkpoint_arguments(),
        )
        _write_latest_metadata(run_directory, state, "last.pt")

    def validate_and_consider_best(completed_epoch: int | None) -> None:
        metrics, validation_seconds = _evaluate(
            model=model,
            repository=repository,
            requests=validation_requests,
            config=config,
            spec=spec,
        )
        current = _monitor_value(metrics, config.early_stopping_metric)
        improved = (
            state.best_metric is None
            or current < state.best_metric - config.early_stopping_min_delta
        )
        if improved:
            state.best_metric = current
            state.best_optimizer_step = state.optimizer_step
            state.non_improvements = 0
        else:
            state.non_improvements += 1
            if state.non_improvements >= config.early_stopping_patience:
                state.stopped_early = True
        record = {
            "type": "validation",
            "completed_epoch": completed_epoch,
            "optimizer_step": state.optimizer_step,
            "metrics": metrics,
            "monitored_metric": config.early_stopping_metric,
            "monitored_value": current,
            "improved": improved,
            "consecutive_non_improvements": state.non_improvements,
            "validation_seconds": validation_seconds,
        }
        _append_jsonl_atomic(run_directory / "history.jsonl", record)
        state.last_validation_optimizer_step = state.optimizer_step
        state.last_validation_metrics = metrics
        if improved:
            _save_checkpoint(
                run_directory / "best.pt",
                **checkpoint_arguments(),
            )
            _atomic_write_json(
                run_directory / "best.json",
                {
                    "checkpoint": "best.pt",
                    "optimizer_step": state.optimizer_step,
                    "epoch": state.epoch,
                    "metric": config.early_stopping_metric,
                    "value": current,
                },
            )

    try:
        while True:
            reached_step_limit = (
                config.max_optimizer_steps is not None
                and state.optimizer_step >= config.max_optimizer_steps
            )
            reached_epoch_limit = (
                config.max_epochs is not None
                and state.epoch >= config.max_epochs
            )
            if reached_step_limit or reached_epoch_limit or state.stopped_early:
                break

            requests = training_window_requests(
                lengths,
                train_ids,
                seed=config.seed,
                epoch=state.epoch,
                window_length_ticks=config.window_length_ticks,
                window_stride_ticks=config.window_stride_ticks,
            )
            batches = _batch_requests(requests, config.batch_size)
            if state.microbatch_index > len(batches):
                raise TrainingCompatibilityError(
                    "checkpoint microbatch position exceeds deterministic epoch"
                )
            while state.microbatch_index < len(batches):
                if (
                    config.max_optimizer_steps is not None
                    and state.optimizer_step >= config.max_optimizer_steps
                ):
                    break
                request_batch = batches[state.microbatch_index]
                items = repository.windows(request_batch)
                encoder, supervision = collate_training_windows(
                    items,
                    expected_profile_id=profile.profile_id,
                    expected_profile_hash=profile.profile_hash,
                    head_config=config.head_config,
                )
                encoder, supervision = _move_batch(
                    encoder,
                    supervision,
                    spec,
                    config.pin_memory,
                )
                model.train()
                with _autocast_context(spec):
                    output = model(encoder.batch)
                    loss = compute_rotation_loss(
                        output.logits,
                        supervision.targets,
                        config.loss_config,
                    )
                _validate_finite_logits_and_loss(
                    output,
                    loss,
                    encoder.metadata.session_ids,
                )
                train_metrics.update(loss)
                supervised = _loss_has_supervision(loss)
                step_record: dict[str, Any] | None = None
                if supervised:
                    scaled_loss = loss.total / config.accumulation_steps
                    scaler.scale(scaled_loss).backward()
                    _validate_scaled_gradients(
                        model,
                        encoder.metadata.session_ids,
                    )
                    state.accumulation_count += 1
                    state.supervised_microbatches += 1
                    if state.accumulation_count == config.accumulation_steps:
                        scaler.unscale_(optimizer)
                        pre_clip_norm = _gradient_norm_and_finite(
                            model,
                            encoder.metadata.session_ids,
                        )
                        clipped = (
                            config.gradient_clip_norm is not None
                            and pre_clip_norm > config.gradient_clip_norm
                        )
                        if config.gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config.gradient_clip_norm,
                                error_if_nonfinite=True,
                            )
                        learning_rate_used = scheduler.get_last_lr()[0]
                        optimizer_started = time.perf_counter()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_seconds = time.perf_counter() - optimizer_started
                        state.optimizer_step += 1
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        state.accumulation_count = 0
                        one_batch_metrics = LossMetricAccumulator(config.loss_config)
                        one_batch_metrics.update(loss)
                        step_record = {
                            "type": "optimizer_step",
                            "epoch": state.epoch,
                            "microbatch_index": state.microbatch_index,
                            "optimizer_step": state.optimizer_step,
                            "metrics": one_batch_metrics.metrics(),
                            "learning_rate_used": learning_rate_used,
                            "next_learning_rate": scheduler.get_last_lr()[0],
                            "pre_clip_gradient_norm": pre_clip_norm,
                            "gradient_clipped": clipped,
                            "optimizer_step_seconds": optimizer_seconds,
                        }
                state.data_microbatches += 1
                state.microbatch_index += 1
                if (
                    step_record is not None
                    and state.optimizer_step % config.history_every_optimizer_steps == 0
                ):
                    _append_jsonl_atomic(
                        run_directory / "history.jsonl",
                        step_record,
                    )
                if (
                    step_record is not None
                    and config.checkpoint_every_optimizer_steps is not None
                    and state.optimizer_step
                    % config.checkpoint_every_optimizer_steps
                    == 0
                ):
                    periodic_name = (
                        f"checkpoint-step-{state.optimizer_step:08d}.pt"
                    )
                    _save_checkpoint(
                        run_directory / periodic_name,
                        **checkpoint_arguments(),
                    )
                if (
                    state.data_microbatches
                    % config.last_checkpoint_every_microbatches
                    == 0
                ):
                    save_last()

            if state.microbatch_index < len(batches):
                continue

            completed_epoch = state.epoch
            train_record = {
                "type": "training_epoch",
                "completed_epoch": completed_epoch,
                "optimizer_step": state.optimizer_step,
                "metrics": train_metrics.metrics(),
                "data_microbatches": state.data_microbatches,
                "supervised_microbatches": state.supervised_microbatches,
                "pending_accumulation": state.accumulation_count,
            }
            _append_jsonl_atomic(
                run_directory / "history.jsonl",
                train_record,
            )
            state.epoch += 1
            state.microbatch_index = 0
            train_metrics = LossMetricAccumulator(config.loss_config)
            if state.epoch % config.validation_every_epochs == 0:
                validate_and_consider_best(completed_epoch)
            save_last()

        if (
            not state.stopped_early
            and state.last_validation_optimizer_step != state.optimizer_step
        ) or state.best_metric is None:
            validate_and_consider_best(
                state.epoch - 1 if state.microbatch_index == 0 else None
            )
        save_last()
    except TrainingNumericalError as exc:
        optimizer.zero_grad(set_to_none=True)
        state.accumulation_count = 0
        try:
            diagnostic = json.loads(str(exc))
        except json.JSONDecodeError:
            diagnostic = {"kind": "numerical_failure", "message": str(exc)[:512]}
        diagnostic["epoch"] = state.epoch
        diagnostic["microbatch_index"] = state.microbatch_index
        diagnostic["optimizer_step"] = state.optimizer_step
        _atomic_write_json(run_directory / "failure.json", diagnostic)
        raise
    finally:
        repository.close()

    summary = {
        "status": "early_stopped" if state.stopped_early else "completed",
        "epoch": state.epoch,
        "microbatch_index": state.microbatch_index,
        "optimizer_steps": state.optimizer_step,
        "data_microbatches": state.data_microbatches,
        "supervised_microbatches": state.supervised_microbatches,
        "pending_accumulation": state.accumulation_count,
        "best_metric": state.best_metric,
        "best_metric_name": config.early_stopping_metric,
        "best_optimizer_step": state.best_optimizer_step,
        "last_validation_metrics": state.last_validation_metrics,
        "last_learning_rate": scheduler.get_last_lr()[0],
        "split_manifest_sha256": manifest.fingerprint,
        "world_grid_profile_hash": profile.profile_hash,
        "world_grid_audit_sha256": profile.publication_audit.audit_sha256,
        "test_partition_evaluated": False,
    }
    _atomic_write_json(run_directory / "final_summary.json", summary)
    return summary
