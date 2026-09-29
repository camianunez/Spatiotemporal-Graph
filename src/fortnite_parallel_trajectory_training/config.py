from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Mapping


TRAINING_CONFIG_SCHEMA = "parallel_trajectory_training_config_v1"
TRAINING_CHECKPOINT_SCHEMA = "parallel_trajectory_training_checkpoint_v1"
SCHEDULER_TYPE = "parallel_trajectory_piecewise_linear_v1"
ARCHITECTURE_ID = "parallel_trajectory_decoder_v1"
TARGET_CONTRACT_VERSION = "parallel-trajectory-targets:1.0"


class TrainingConfigurationError(ValueError):
    """Raised when a training or provenance contract is not exact."""


class TrainingCompatibilityError(RuntimeError):
    """Raised when persisted state cannot be used exactly."""


class TrainingNumericalError(RuntimeError):
    """Raised when a fail-closed numerical gate is violated."""


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingConfigurationError(f"{location} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingConfigurationError(f"{location} must be a positive integer")
    return value


def _positive_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TrainingConfigurationError(f"{location} must be finite and positive")
    return result


def _strict_keys(raw: Any, declared: set[str], location: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingConfigurationError(f"{location} must be a JSON object")
    missing = sorted(declared - set(raw))
    unknown = sorted(set(raw) - declared)
    if missing or unknown:
        raise TrainingConfigurationError(
            f"{location} keys mismatch: missing={missing}, unknown={unknown}"
        )
    return dict(raw)


def _split_counts(value: Any, location: str) -> dict[str, int]:
    raw = _strict_keys(value, {"train", "validation", "test"}, location)
    return {
        name: _positive_int(raw[name], f"{location}.{name}")
        for name in ("train", "validation", "test")
    }


@dataclass(frozen=True, slots=True)
class ParallelTrajectoryTrainingConfig:
    schema_version: str
    architecture_id: str
    target_contract_version: str
    run_directory: str
    source_checkpoint: str
    expected_source_checkpoint_sha256: str
    forbidden_phase_weighted_checkpoint: str
    forbidden_phase_weighted_checkpoint_sha256: str
    canonical_source_config: str
    expected_canonical_source_config_raw_sha256: str
    lineage_report: str
    expected_lineage_report_raw_sha256: str
    lineage_classification: str
    legacy_encoder_digest: str
    current_encoder_digest: str
    architecture_package_digest: str
    training_package_digest: str
    expected_transferred_tensor_count: int
    expected_transferred_tensor_manifest_sha256: str
    dataset_root: str
    split_manifest_path: str
    expected_split_manifest_sha256: str
    expected_split_manifest_raw_sha256: str
    expected_split_counts: dict[str, int]
    expected_ancestral_source_split_counts: dict[str, int]
    expected_ingestion_report_sha256: str
    dataset_validation_path: str
    expected_dataset_validation_report_hash: str
    expected_dataset_validation_raw_sha256: str
    world_grid_profile_path: str
    expected_world_grid_profile_hash: str
    expected_world_grid_profile_raw_sha256: str
    world_grid_audit_path: str
    expected_world_grid_audit_payload_sha256: str
    expected_world_grid_audit_raw_sha256: str
    expected_world_grid_coordinate_audit_sha256: str
    seed: int
    batch_size: int
    queries_per_session: int
    validation_queries_per_session: int
    accumulation_steps: int
    context_length_ticks: int
    max_queries_per_forward: int
    max_epochs: int
    updates_per_epoch: int
    max_optimizer_steps: int
    decoder_learning_rate: float
    congestion_learning_rate: float
    encoder_learning_rate: float
    adamw_betas: tuple[float, float]
    adamw_epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    scheduler_type: str
    scheduler_warmup_updates: int
    scheduler_plateau_updates: int
    scheduler_decay_updates: int
    freeze_inherited_epochs: int
    device: Literal["cuda:0"]
    precision: Literal["bf16"]
    pin_memory: bool

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_CONFIG_SCHEMA:
            raise TrainingConfigurationError("unsupported training configuration schema")
        if self.architecture_id != ARCHITECTURE_ID:
            raise TrainingConfigurationError("architecture identifier is not frozen v1")
        if self.target_contract_version != TARGET_CONTRACT_VERSION:
            raise TrainingConfigurationError("target contract is not frozen v1")
        if self.lineage_classification not in {
            "scope_changed_only",
            "source_changed_noncritical",
            "source_changed_runtime_compatible",
            "both_scope_and_source_changed",
        }:
            raise TrainingConfigurationError("lineage classification does not authorize training")
        for name in (
            "expected_source_checkpoint_sha256",
            "forbidden_phase_weighted_checkpoint_sha256",
            "expected_canonical_source_config_raw_sha256",
            "expected_lineage_report_raw_sha256",
            "legacy_encoder_digest",
            "current_encoder_digest",
            "architecture_package_digest",
            "training_package_digest",
            "expected_transferred_tensor_manifest_sha256",
            "expected_split_manifest_sha256",
            "expected_split_manifest_raw_sha256",
            "expected_ingestion_report_sha256",
            "expected_dataset_validation_report_hash",
            "expected_dataset_validation_raw_sha256",
            "expected_world_grid_profile_hash",
            "expected_world_grid_profile_raw_sha256",
            "expected_world_grid_audit_payload_sha256",
            "expected_world_grid_audit_raw_sha256",
            "expected_world_grid_coordinate_audit_sha256",
        ):
            _sha256(getattr(self, name), f"config.{name}")
        for name in (
            "run_directory",
            "source_checkpoint",
            "forbidden_phase_weighted_checkpoint",
            "canonical_source_config",
            "lineage_report",
            "dataset_root",
            "split_manifest_path",
            "dataset_validation_path",
            "world_grid_profile_path",
            "world_grid_audit_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TrainingConfigurationError(f"config.{name} must be a nonempty path")
        exact_ints = {
            "seed": 20260804,
            "batch_size": 2,
            "queries_per_session": 16,
            "validation_queries_per_session": 32,
            "accumulation_steps": 1,
            "context_length_ticks": 64,
            "max_queries_per_forward": 4,
            "max_epochs": 20,
            "updates_per_epoch": 43,
            "max_optimizer_steps": 860,
            "scheduler_warmup_updates": 43,
            "scheduler_plateau_updates": 43,
            "scheduler_decay_updates": 774,
            "freeze_inherited_epochs": 1,
            "expected_transferred_tensor_count": 146,
        }
        for name, expected in exact_ints.items():
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise TrainingConfigurationError(
                    f"config.{name} is fixed at {expected!r}, received {value!r}"
                )
        expected_floats = {
            "decoder_learning_rate": 3e-4,
            "congestion_learning_rate": 3e-4,
            "encoder_learning_rate": 3e-5,
            "adamw_epsilon": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        }
        for name, expected in expected_floats.items():
            value = _positive_float(getattr(self, name), f"config.{name}")
            if value != expected:
                raise TrainingConfigurationError(
                    f"config.{name} is fixed at {expected!r}, received {value!r}"
                )
        if tuple(float(value) for value in self.adamw_betas) != (0.9, 0.999):
            raise TrainingConfigurationError("AdamW betas are fixed at (0.9, 0.999)")
        if self.scheduler_type != SCHEDULER_TYPE:
            raise TrainingConfigurationError("scheduler type is not the frozen piecewise schedule")
        if (
            self.scheduler_warmup_updates
            + self.scheduler_plateau_updates
            + self.scheduler_decay_updates
            != self.max_optimizer_steps
        ):
            raise TrainingConfigurationError("scheduler boundaries do not cover 860 updates")
        if self.max_epochs * self.updates_per_epoch != self.max_optimizer_steps:
            raise TrainingConfigurationError("epoch/update contract is inconsistent")
        if self.device != "cuda:0" or self.precision != "bf16":
            raise TrainingConfigurationError("training requires cuda:0 with BF16")
        if type(self.pin_memory) is not bool:
            raise TrainingConfigurationError("pin_memory must be boolean")
        if self.expected_split_counts != {"train": 86, "validation": 11, "test": 10}:
            raise TrainingConfigurationError(
                "decoded-v2-new planner split must be 86/11/10"
            )
        if self.expected_ancestral_source_split_counts != {
            "train": 33,
            "validation": 4,
            "test": 4,
        }:
            raise TrainingConfigurationError(
                "ancestral decoded-v2 encoder split must be separately bound as 33/4/4"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "ParallelTrajectoryTrainingConfig":
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingConfigurationError(
                f"cannot read training configuration {source}: {exc}"
            ) from exc
        raw = _strict_keys(raw, {item.name for item in fields(cls)}, "config")
        raw["adamw_betas"] = tuple(raw["adamw_betas"])
        raw["expected_split_counts"] = _split_counts(
            raw["expected_split_counts"], "config.expected_split_counts"
        )
        raw["expected_ancestral_source_split_counts"] = _split_counts(
            raw["expected_ancestral_source_split_counts"],
            "config.expected_ancestral_source_split_counts",
        )
        base = source.parent
        for name in (
            "run_directory",
            "source_checkpoint",
            "forbidden_phase_weighted_checkpoint",
            "canonical_source_config",
            "lineage_report",
            "dataset_root",
            "split_manifest_path",
            "dataset_validation_path",
            "world_grid_profile_path",
            "world_grid_audit_path",
        ):
            candidate = Path(raw[name])
            raw[name] = str(
                (candidate if candidate.is_absolute() else base / candidate).resolve()
            )
        try:
            return cls(**raw)
        except TypeError as exc:
            raise TrainingConfigurationError(f"invalid training configuration: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["adamw_betas"] = list(self.adamw_betas)
        return result


__all__ = [
    "ARCHITECTURE_ID",
    "ParallelTrajectoryTrainingConfig",
    "SCHEDULER_TYPE",
    "TARGET_CONTRACT_VERSION",
    "TRAINING_CHECKPOINT_SCHEMA",
    "TRAINING_CONFIG_SCHEMA",
    "TrainingCompatibilityError",
    "TrainingConfigurationError",
    "TrainingNumericalError",
]
