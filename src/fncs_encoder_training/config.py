from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "fncs-fresh-encoder-config:1.0"


@dataclass(frozen=True, slots=True)
class FreshEncoderConfig:
    schema_version: str
    dataset_root: str
    dataset_validation_path: str
    run_directory: str
    preflight_directory: str
    authoritative_run_directory: str
    authoritative_checkpoint: str
    expected_authoritative_checkpoint_sha256: str
    authoritative_resolved_config: str
    expected_authoritative_resolved_config_sha256: str
    authoritative_final_summary: str
    expected_authoritative_final_summary_sha256: str
    scheduler_authority_config: str
    expected_scheduler_authority_config_sha256: str
    encoder_lineage_report: str
    expected_encoder_lineage_report_sha256: str
    encoder_scope_manifest: str
    expected_encoder_scope_manifest_sha256: str
    verified_source_difference_report: str
    expected_verified_source_difference_report_sha256: str
    expected_encoder_source_digest: str
    world_grid_profile: str
    expected_world_grid_profile_raw_sha256: str
    expected_world_grid_profile_hash: str
    world_grid_publication_audit: str
    expected_world_grid_audit_raw_sha256: str
    expected_world_grid_audit_payload_sha256: str
    expected_world_grid_coordinate_audit_sha256: str
    seed: int
    train_sessions: int
    validation_sessions: int
    test_sessions: int
    window_length_ticks: int
    window_stride_ticks: int
    batch_size: int
    accumulation_steps: int
    epochs: int
    learning_rate: float
    adamw_betas: tuple[float, float]
    adamw_epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    entry_loss_weight: float
    survival_loss_weight: float
    placement_loss_weight: float
    regularization_loss_weight: float
    scheduler_warmup_fraction: float
    scheduler_constant_fraction: float
    scheduler_decay_fraction: float
    scheduler_final_lr_ratio: float
    precision: str
    device: str
    validation_every_epochs: int
    checkpoint_selection_metric: str
    checkpoint_every_optimizer_steps: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    num_workers: int
    pin_memory: bool

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        for name in (
            "dataset_root",
            "dataset_validation_path",
            "run_directory",
            "preflight_directory",
            "authoritative_run_directory",
            "authoritative_checkpoint",
            "authoritative_resolved_config",
            "authoritative_final_summary",
            "scheduler_authority_config",
            "encoder_lineage_report",
            "encoder_scope_manifest",
            "verified_source_difference_report",
            "world_grid_profile",
            "world_grid_publication_audit",
            "precision",
            "device",
            "checkpoint_selection_metric",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "expected_authoritative_checkpoint_sha256",
            "expected_authoritative_resolved_config_sha256",
            "expected_authoritative_final_summary_sha256",
            "expected_scheduler_authority_config_sha256",
            "expected_encoder_lineage_report_sha256",
            "expected_encoder_scope_manifest_sha256",
            "expected_verified_source_difference_report_sha256",
            "expected_encoder_source_digest",
            "expected_world_grid_profile_raw_sha256",
            "expected_world_grid_profile_hash",
            "expected_world_grid_audit_raw_sha256",
            "expected_world_grid_audit_payload_sha256",
            "expected_world_grid_coordinate_audit_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in (
            "seed",
            "train_sessions",
            "validation_sessions",
            "test_sessions",
            "window_length_ticks",
            "window_stride_ticks",
            "batch_size",
            "accumulation_steps",
            "epochs",
            "validation_every_epochs",
            "checkpoint_every_optimizer_steps",
            "early_stopping_patience",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.num_workers < 0 or type(self.num_workers) is not int:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.pin_memory) is not bool:
            raise ValueError("pin_memory must be boolean")
        for name in (
            "learning_rate",
            "adamw_epsilon",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "weight_decay",
            "entry_loss_weight",
            "survival_loss_weight",
            "placement_loss_weight",
            "regularization_loss_weight",
            "early_stopping_min_delta",
            "scheduler_final_lr_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.scheduler_final_lr_ratio > 1.0:
            raise ValueError("scheduler_final_lr_ratio must be <= 1")
        if (
            len(self.adamw_betas) != 2
            or any(not 0.0 <= float(value) < 1.0 for value in self.adamw_betas)
        ):
            raise ValueError("adamw_betas must contain two values in [0,1)")
        fractions = (
            self.scheduler_warmup_fraction,
            self.scheduler_constant_fraction,
            self.scheduler_decay_fraction,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in fractions):
            raise ValueError("scheduler fractions must be positive and finite")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
            raise ValueError("scheduler fractions must sum to one")
        if self.precision != "bf16" or self.device != "cuda:0":
            raise ValueError("the FNCS production contract requires cuda:0 BF16")
        if self.regularization_loss_weight != 0.0:
            raise ValueError("the authoritative objective has no explicit regularization loss")
        if self.train_sessions + self.validation_sessions + self.test_sessions != 1150:
            raise ValueError("split counts must total exactly 1,150 sessions")

    @property
    def total_sessions(self) -> int:
        return self.train_sessions + self.validation_sessions + self.test_sessions

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["adamw_betas"] = list(self.adamw_betas)
        return value


def _resolve(base: Path, value: str) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def load_config(path: str | Path) -> FreshEncoderConfig:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    declared = {item.name for item in fields(FreshEncoderConfig)}
    unknown = sorted(set(raw) - declared)
    missing = sorted(declared - set(raw))
    if unknown or missing:
        raise ValueError(f"configuration fields differ; missing={missing}, unknown={unknown}")
    if not isinstance(raw["adamw_betas"], list) or len(raw["adamw_betas"]) != 2:
        raise ValueError("adamw_betas must be a two-item JSON array")
    raw["adamw_betas"] = tuple(float(value) for value in raw["adamw_betas"])
    base = source.parent
    for name in (
        "dataset_root",
        "dataset_validation_path",
        "run_directory",
        "preflight_directory",
        "authoritative_run_directory",
        "authoritative_checkpoint",
        "authoritative_resolved_config",
        "authoritative_final_summary",
        "scheduler_authority_config",
        "encoder_lineage_report",
        "encoder_scope_manifest",
        "verified_source_difference_report",
        "world_grid_profile",
        "world_grid_publication_audit",
    ):
        raw[name] = _resolve(base, raw[name])
    return FreshEncoderConfig(**raw)
