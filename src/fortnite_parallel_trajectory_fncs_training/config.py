from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping

from fncs_encoder_training.common import file_sha256


FROZEN_FNCS_CONFIG_SCHEMA = "fncs-frozen-parallel-trajectory-training:1.0"
FROZEN_FNCS_CHECKPOINT_SCHEMA = (
    "fncs-frozen-parallel-trajectory-checkpoint:1.0"
)
ARCHITECTURE_ID = "parallel_trajectory_decoder_v1"
TARGET_CONTRACT_ID = "parallel-trajectory-targets:1.0"
SCHEDULER_TYPE = "fncs_frozen_decoder_piecewise_linear_v1"


class TrainingConfigurationError(ValueError):
    """The pinned training configuration is invalid."""


class TrainingCompatibilityError(RuntimeError):
    """A bound artifact or runtime contract no longer matches."""


class TrainingNumericalError(RuntimeError):
    """A numerical fail-closed gate was violated."""


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingConfigurationError(
            f"{location} must be a lowercase full SHA-256 digest"
        )
    return value


def _strict_keys(raw: Any, expected: set[str], location: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TrainingConfigurationError(f"{location} must be an object")
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise TrainingConfigurationError(
            f"{location} keys mismatch: missing={missing}, unknown={unknown}"
        )
    return raw


def _positive_int(value: Any, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingConfigurationError(f"{location} must be a positive integer")
    return value


def _finite_float(value: Any, location: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise TrainingConfigurationError(f"{location} must be {qualifier}")
    return float(value)


@dataclass(frozen=True, slots=True)
class FrozenFNCSTrainingConfig:
    schema_version: str
    architecture_id: str
    target_contract_version: str
    run_directory: Path
    preflight_directory: Path
    canonical_encoder_run_directory: Path
    canonical_checkpoint: Path
    expected_canonical_checkpoint_sha256: str
    canonical_checkpoint_metadata: Path
    expected_canonical_checkpoint_metadata_sha256: str
    execution_encoder_checkpoint: Path
    expected_execution_encoder_checkpoint_sha256: str
    execution_encoder_metadata: Path
    expected_execution_encoder_metadata_sha256: str
    expected_encoder_state_sha256: str
    expected_encoder_tensor_count: int
    selected_epoch: int
    selected_optimizer_step: int
    selected_validation_future_position_loss: float
    expected_encoder_source_digest: str
    expected_architecture_package_digest: str
    expected_architecture_files: Mapping[str, str]
    verified_decoder_training_contract: Path
    expected_verified_decoder_training_contract_sha256: str
    expected_verified_decoder_training_source_digest: str
    expected_training_source_digest: str
    dataset_root: Path
    dataset_inventory: Path
    expected_dataset_inventory_sha256: str
    ingestion_report: Path
    expected_ingestion_report_sha256: str
    dataset_validation: Path
    expected_dataset_validation_raw_sha256: str
    expected_dataset_validation_sha256: str
    split_manifest: Path
    expected_split_manifest_raw_sha256: str
    expected_split_manifest_sha256: str
    expected_split_counts: Mapping[str, int]
    world_grid_profile: Path
    expected_world_grid_profile_raw_sha256: str
    expected_world_grid_profile_hash: str
    world_grid_publication_audit: Path
    expected_world_grid_publication_audit_raw_sha256: str
    expected_world_grid_publication_audit_sha256: str
    world_grid_compatibility_audit: Path
    expected_world_grid_compatibility_audit_raw_sha256: str
    expected_world_grid_coordinate_audit_sha256: str
    expected_world_grid_validation_audit_sha256: str
    seed: int
    initialization_seed: int
    batch_size: int
    queries_per_session: int
    validation_queries_per_session: int
    accumulation_steps: int
    context_length_ticks: int
    max_queries_per_forward: int
    max_epochs: int
    updates_per_epoch: int
    max_optimizer_steps: int
    downstream_learning_rate: float
    adamw_betas: tuple[float, float]
    adamw_epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    decoder_dropout: float
    scheduler_type: str
    scheduler_warmup_fraction: float
    scheduler_constant_fraction: float
    scheduler_decay_fraction: float
    scheduler_final_lr_ratio: float
    validation_every_epochs: int
    checkpoint_every_optimizer_steps: int
    device: str
    precision: str
    pin_memory: bool
    objective: str
    congestion_auxiliary_objective: str
    checkpoint_selection_metric: str
    scientific_caveat: str
    configuration_path: Path = field(init=False, repr=False, compare=False)
    configuration_raw_sha256: str = field(init=False, repr=False, compare=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "FrozenFNCSTrainingConfig":
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingConfigurationError(
                f"cannot read training configuration: {exc}"
            ) from exc
        declared = {
            item.name
            for item in fields(cls)
            if item.init
        }
        values = _strict_keys(raw, declared, "training configuration")
        base = source.parent
        path_fields = {
            "run_directory",
            "preflight_directory",
            "canonical_encoder_run_directory",
            "canonical_checkpoint",
            "canonical_checkpoint_metadata",
            "execution_encoder_checkpoint",
            "execution_encoder_metadata",
            "verified_decoder_training_contract",
            "dataset_root",
            "dataset_inventory",
            "ingestion_report",
            "dataset_validation",
            "split_manifest",
            "world_grid_profile",
            "world_grid_publication_audit",
            "world_grid_compatibility_audit",
        }
        normalized = dict(values)
        for name in path_fields:
            value = normalized[name]
            if not isinstance(value, str) or not value:
                raise TrainingConfigurationError(f"{name} must be a path string")
            candidate = Path(value)
            normalized[name] = (
                candidate if candidate.is_absolute() else base / candidate
            ).resolve()
        betas = normalized["adamw_betas"]
        if not isinstance(betas, list) or len(betas) != 2:
            raise TrainingConfigurationError("adamw_betas must contain two values")
        normalized["adamw_betas"] = tuple(
            _finite_float(value, f"adamw_betas[{index}]", positive=True)
            for index, value in enumerate(betas)
        )
        try:
            parsed = cls(**normalized)
        except TypeError as exc:
            raise TrainingConfigurationError(str(exc)) from exc
        object.__setattr__(parsed, "configuration_path", source)
        object.__setattr__(parsed, "configuration_raw_sha256", file_sha256(source))
        return parsed

    def __post_init__(self) -> None:
        fixed_strings = {
            "schema_version": FROZEN_FNCS_CONFIG_SCHEMA,
            "architecture_id": ARCHITECTURE_ID,
            "target_contract_version": TARGET_CONTRACT_ID,
            "scheduler_type": SCHEDULER_TYPE,
            "device": "cuda:0",
            "precision": "bf16",
            "objective": "route_mixture_nll_only",
            "congestion_auxiliary_objective": "none",
            "checkpoint_selection_metric": "validation_route_mixture_nll",
        }
        for name, expected in fixed_strings.items():
            if getattr(self, name) != expected:
                raise TrainingConfigurationError(f"{name} is fixed at {expected!r}")
        digest_fields = (
            "expected_canonical_checkpoint_sha256",
            "expected_canonical_checkpoint_metadata_sha256",
            "expected_execution_encoder_checkpoint_sha256",
            "expected_execution_encoder_metadata_sha256",
            "expected_encoder_state_sha256",
            "expected_encoder_source_digest",
            "expected_architecture_package_digest",
            "expected_verified_decoder_training_contract_sha256",
            "expected_verified_decoder_training_source_digest",
            "expected_training_source_digest",
            "expected_dataset_inventory_sha256",
            "expected_ingestion_report_sha256",
            "expected_dataset_validation_raw_sha256",
            "expected_dataset_validation_sha256",
            "expected_split_manifest_raw_sha256",
            "expected_split_manifest_sha256",
            "expected_world_grid_profile_raw_sha256",
            "expected_world_grid_profile_hash",
            "expected_world_grid_publication_audit_raw_sha256",
            "expected_world_grid_publication_audit_sha256",
            "expected_world_grid_compatibility_audit_raw_sha256",
            "expected_world_grid_coordinate_audit_sha256",
            "expected_world_grid_validation_audit_sha256",
        )
        for name in digest_fields:
            _sha256(getattr(self, name), name)
        if not isinstance(self.expected_architecture_files, Mapping):
            raise TrainingConfigurationError(
                "expected_architecture_files must be an object"
            )
        expected_architecture_names = {
            "__init__.py",
            "config.py",
            "contracts.py",
            "decoder.py",
            "inference.py",
            "losses.py",
            "model.py",
            "targets.py",
        }
        if set(self.expected_architecture_files) != expected_architecture_names:
            raise TrainingConfigurationError(
                "expected_architecture_files does not describe the verified package"
            )
        for name, digest in self.expected_architecture_files.items():
            _sha256(digest, f"expected_architecture_files[{name!r}]")
        integer_fields = (
            "expected_encoder_tensor_count",
            "selected_epoch",
            "selected_optimizer_step",
            "seed",
            "initialization_seed",
            "batch_size",
            "queries_per_session",
            "validation_queries_per_session",
            "accumulation_steps",
            "context_length_ticks",
            "max_queries_per_forward",
            "max_epochs",
            "updates_per_epoch",
            "max_optimizer_steps",
            "validation_every_epochs",
            "checkpoint_every_optimizer_steps",
        )
        for name in integer_fields:
            _positive_int(getattr(self, name), name)
        numeric_fields = (
            "selected_validation_future_position_loss",
            "downstream_learning_rate",
            "adamw_epsilon",
            "weight_decay",
            "gradient_clip_norm",
            "decoder_dropout",
            "scheduler_warmup_fraction",
            "scheduler_constant_fraction",
            "scheduler_decay_fraction",
            "scheduler_final_lr_ratio",
        )
        for name in numeric_fields:
            _finite_float(
                getattr(self, name),
                name,
                positive=name not in {"weight_decay", "decoder_dropout"},
            )
        if dict(self.expected_split_counts) != {
            "train": 920,
            "validation": 115,
            "test": 115,
        }:
            raise TrainingConfigurationError("expected_split_counts is fixed at 920/115/115")
        fixed_numbers: dict[str, Any] = {
            "expected_encoder_tensor_count": 124,
            "selected_epoch": 19,
            "selected_optimizer_step": 4370,
            "selected_validation_future_position_loss": 7.842962951838928,
            "seed": 20260727,
            "initialization_seed": 20260727,
            "batch_size": 2,
            "queries_per_session": 16,
            "validation_queries_per_session": 32,
            "accumulation_steps": 1,
            "context_length_ticks": 64,
            "max_queries_per_forward": 4,
            "max_epochs": 20,
            "updates_per_epoch": 460,
            "max_optimizer_steps": 9200,
            "downstream_learning_rate": 3e-4,
            "adamw_betas": (0.9, 0.999),
            "adamw_epsilon": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "decoder_dropout": 0.1,
            "scheduler_warmup_fraction": 0.05,
            "scheduler_constant_fraction": 0.05,
            "scheduler_decay_fraction": 0.9,
            "scheduler_final_lr_ratio": 0.1,
            "validation_every_epochs": 1,
            "checkpoint_every_optimizer_steps": 460,
            "pin_memory": False,
        }
        for name, expected in fixed_numbers.items():
            if getattr(self, name) != expected:
                raise TrainingConfigurationError(f"{name} is fixed at {expected!r}")
        if self.updates_per_epoch != math.ceil(
            int(self.expected_split_counts["train"]) / self.batch_size
        ):
            raise TrainingConfigurationError(
                "updates_per_epoch must be recomputed from the training-loader length"
            )
        if self.max_optimizer_steps != self.updates_per_epoch * self.max_epochs:
            raise TrainingConfigurationError(
                "max_optimizer_steps must cover exactly 20 complete epochs"
            )
        fractions = (
            self.scheduler_warmup_fraction
            + self.scheduler_constant_fraction
            + self.scheduler_decay_fraction
        )
        if not math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise TrainingConfigurationError("scheduler fractions must sum to one")
        if not isinstance(self.pin_memory, bool):
            raise TrainingConfigurationError("pin_memory must be boolean")
        if not isinstance(self.scientific_caveat, str) or not self.scientific_caveat:
            raise TrainingConfigurationError("scientific_caveat must be nonempty")

    @property
    def scheduler_warmup_updates(self) -> int:
        return round(self.max_optimizer_steps * self.scheduler_warmup_fraction)

    @property
    def scheduler_constant_updates(self) -> int:
        return round(self.max_optimizer_steps * self.scheduler_constant_fraction)

    @property
    def scheduler_decay_updates(self) -> int:
        return (
            self.max_optimizer_steps
            - self.scheduler_warmup_updates
            - self.scheduler_constant_updates
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("configuration_path", None)
        value.pop("configuration_raw_sha256", None)
        for item in fields(self):
            if item.init and isinstance(value.get(item.name), Path):
                value[item.name] = str(value[item.name])
        value["adamw_betas"] = list(self.adamw_betas)
        value["expected_architecture_files"] = dict(
            sorted(self.expected_architecture_files.items())
        )
        value["expected_split_counts"] = dict(self.expected_split_counts)
        return value


__all__ = [
    "ARCHITECTURE_ID",
    "FROZEN_FNCS_CHECKPOINT_SCHEMA",
    "FROZEN_FNCS_CONFIG_SCHEMA",
    "FrozenFNCSTrainingConfig",
    "SCHEDULER_TYPE",
    "TARGET_CONTRACT_ID",
    "TrainingCompatibilityError",
    "TrainingConfigurationError",
    "TrainingNumericalError",
]
