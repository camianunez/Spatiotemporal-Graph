from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pyarrow
import torch
from torch import Tensor, nn

from .config import EncoderConfig
from .contracts import EncoderBatch, TensorizedMatch
from .model import SpatiotemporalEncoder
from .planner import CongestionPredictor, ExpertRotationPlannerModel, PlannerLogits
from .planner_contracts import PlannerObservation, PlannerTargets, PreviousWaypoints
from .planner_losses import (
    PlannerLossComponents,
    PlannerLossConfig,
    compute_planner_loss,
    compute_planner_route_loss,
)
from .planner_policy import (
    OBSERVATION_PROVIDER_ID,
    PlannerPolicyConfig,
    build_planner_observation,
)
from .planner_supervision import (
    PLANNER_TARGET_SCHEMA_ID,
    PlannerQuery,
    PlannerSupervision,
    PlannerTargetDiagnostics,
    PlannerTargetSource,
    build_planner_targets,
    collate_planner_targets,
    eligible_planner_queries,
    load_planner_target_source,
    slice_planner_targets,
)
from .tensorize import collate_encoder_inputs, load_match_session, slice_window
from .training import (
    CHECKPOINT_SCHEMA_VERSION as SOURCE_CHECKPOINT_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    TrainingCompatibilityError,
    TrainingConfigurationError,
    TrainingNumericalError,
    WorldGridProfileConfig,
    _append_jsonl_atomic,
    _atomic_torch_save,
    _atomic_write_json,
    _file_sha256,
    _fingerprint,
    _jsonable,
    _read_source_revision,
    _restore_rng_state,
    _rng_state,
    _seed_everything,
    _strict_dataclass,
)
from .world_grid import (
    WorldGridProfile,
    WorldGridProfileError,
    _payload_hash,
    ingestion_binding_from_report,
)


PLANNER_CHECKPOINT_SCHEMA_VERSION = "3.0"
LEGACY_PLANNER_TRANSFER_SCHEMA_VERSION = "2.0"
UNIFORM_QUERY_SAMPLER_VERSION = "uniform_valid_y0_without_replacement:2.0"
_PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class PlannerDatasetBindingConfig:
    """Pinned provenance for a planner corpus independent of the encoder."""

    split_manifest_path: str
    expected_split_manifest_sha256: str
    dataset_validation_path: str
    expected_dataset_validation_report_hash: str
    expected_ingestion_report_sha256: str

    def __post_init__(self) -> None:
        for name in ("split_manifest_path", "dataset_validation_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")
        for name in (
            "expected_split_manifest_sha256",
            "expected_dataset_validation_report_hash",
            "expected_ingestion_report_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class PlannerTrainingConfig:
    """Strict JSON configuration for deterministic planner fine-tuning."""

    dataset_root: str
    run_directory: str
    source_checkpoint: str
    planner_transfer_checkpoint: str
    expected_planner_transfer_checkpoint_sha256: str
    seed: int | None = None
    batch_size: int | None = None
    queries_per_session: int = 1
    validation_queries_per_session: int = 16
    accumulation_steps: int = 1
    max_queries_per_forward: int = 8
    max_epochs: int | None = None
    max_optimizer_steps: int | None = None
    planner_learning_rate: float = 3e-4
    encoder_learning_rate: float = 3e-5
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_epsilon: float = 1e-8
    weight_decay: float = 0.01
    scheduler_warmup_steps: int | None = None
    scheduler_warmup_fraction: float = 0.05
    scheduler_min_lr_ratio: float = 0.1
    precision: Literal["auto", "fp32", "bf16"] = "auto"
    device: str = "auto"
    pin_memory: bool = False
    gradient_clip_norm: float = 1.0
    validation_every_epochs: int = 1
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    checkpoint_every_optimizer_steps: int | None = None
    metrics_every_optimizer_steps: int = 1
    last_checkpoint_every_microbatches: int = 1
    expected_source_checkpoint_sha256: str | None = None
    world_grid_profile: WorldGridProfileConfig | None = None
    planner_dataset_binding: PlannerDatasetBindingConfig | None = None
    policy_config: PlannerPolicyConfig = field(default_factory=PlannerPolicyConfig)
    loss_config: PlannerLossConfig = field(default_factory=PlannerLossConfig)

    def __post_init__(self) -> None:
        for name in (
            "dataset_root",
            "run_directory",
            "source_checkpoint",
            "planner_transfer_checkpoint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")
        if self.seed is None or self.batch_size is None:
            raise ValueError("seed and batch_size are required")
        if self.max_epochs is None and self.max_optimizer_steps is None:
            raise ValueError(
                "at least one of max_epochs or max_optimizer_steps is required"
            )
        for name in (
            "seed",
            "batch_size",
            "queries_per_session",
            "validation_queries_per_session",
            "accumulation_steps",
            "max_queries_per_forward",
            "validation_every_epochs",
            "metrics_every_optimizer_steps",
            "last_checkpoint_every_microbatches",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "max_epochs",
            "max_optimizer_steps",
            "scheduler_warmup_steps",
            "early_stopping_patience",
            "checkpoint_every_optimizer_steps",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer when set")
        expected_fixed = {
            "planner_learning_rate": 3e-4,
            "encoder_learning_rate": 3e-5,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "scheduler_warmup_fraction": 0.05,
            "scheduler_min_lr_ratio": 0.1,
        }
        for name, expected in expected_fixed.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
            ):
                raise ValueError(f"{name} is fixed at {expected}")
        if (
            not isinstance(self.adamw_betas, tuple)
            or len(self.adamw_betas) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value < 1.0
                for value in self.adamw_betas
            )
        ):
            raise ValueError("adamw_betas must contain two values in [0,1)")
        if (
            isinstance(self.adamw_epsilon, bool)
            or not isinstance(self.adamw_epsilon, (int, float))
            or not math.isfinite(float(self.adamw_epsilon))
            or self.adamw_epsilon <= 0.0
        ):
            raise ValueError("adamw_epsilon must be positive")
        if (
            isinstance(self.early_stopping_min_delta, bool)
            or not isinstance(self.early_stopping_min_delta, (int, float))
            or not math.isfinite(float(self.early_stopping_min_delta))
            or self.early_stopping_min_delta < 0.0
        ):
            raise ValueError("early_stopping_min_delta must be nonnegative")
        if self.precision not in {"auto", "fp32", "bf16"}:
            raise ValueError("precision must be auto, fp32, or bf16")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty")
        if type(self.pin_memory) is not bool:
            raise ValueError("pin_memory must be boolean")
        if not isinstance(self.policy_config, PlannerPolicyConfig):
            raise ValueError("policy_config must be PlannerPolicyConfig")
        if self.policy_config.canonical_id != OBSERVATION_PROVIDER_ID:
            raise ValueError("planner training supports only own_team_only:1.0")
        if not isinstance(self.loss_config, PlannerLossConfig):
            raise ValueError("loss_config must be PlannerLossConfig")
        if self.expected_source_checkpoint_sha256 is not None:
            value = self.expected_source_checkpoint_sha256
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    "expected_source_checkpoint_sha256 must be lowercase SHA-256"
                )
        value = self.expected_planner_transfer_checkpoint_sha256
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                "expected_planner_transfer_checkpoint_sha256 must be lowercase SHA-256"
            )
        if self.world_grid_profile is not None and not isinstance(
            self.world_grid_profile, WorldGridProfileConfig
        ):
            raise ValueError("world_grid_profile must be WorldGridProfileConfig")
        if self.planner_dataset_binding is not None and not isinstance(
            self.planner_dataset_binding, PlannerDatasetBindingConfig
        ):
            raise ValueError(
                "planner_dataset_binding must be PlannerDatasetBindingConfig"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> PlannerTrainingConfig:
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingConfigurationError(
                f"cannot read planner training config {source}: {exc}"
            ) from exc
        config = _strict_dataclass(cls, raw, "config")
        base = source.parent

        def resolved(value: str) -> str:
            candidate = Path(value)
            return str(
                (candidate if candidate.is_absolute() else base / candidate).resolve()
            )

        profile = config.world_grid_profile
        binding = config.planner_dataset_binding
        return dataclasses.replace(
            config,
            dataset_root=resolved(config.dataset_root),
            run_directory=resolved(config.run_directory),
            source_checkpoint=resolved(config.source_checkpoint),
            planner_transfer_checkpoint=resolved(
                config.planner_transfer_checkpoint
            ),
            world_grid_profile=(
                dataclasses.replace(
                    profile,
                    path=resolved(profile.path),
                    audit_path=resolved(profile.audit_path),
                )
                if profile is not None
                else None
            ),
            planner_dataset_binding=(
                dataclasses.replace(
                    binding,
                    split_manifest_path=resolved(binding.split_manifest_path),
                    dataset_validation_path=resolved(
                        binding.dataset_validation_path
                    ),
                )
                if binding is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class SourceEncoderCheckpoint:
    encoder: SpatiotemporalEncoder
    encoder_config: EncoderConfig
    checkpoint_path: str
    checkpoint_sha256: str
    split_manifest: dict[str, Any]
    split_manifest_sha256: str
    context_length_ticks: int
    dataset_schema_version: str
    ingestion_report_sha256: str
    world_grid_profile_id: str
    world_grid_profile_hash: str
    world_grid_audit_sha256: str
    source_revision: str | None
    dataset_root: str | None

    @property
    def session_ids_by_partition(self) -> dict[str, tuple[str, ...]]:
        assignments = self.split_manifest["assignments"]
        return {
            partition: tuple(
                sorted(
                    item["session_id"]
                    for item in assignments
                    if item["partition"] == partition
                )
            )
            for partition in _PARTITIONS
        }


def _validate_sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingCompatibilityError(f"{location} must be lowercase SHA-256")
    return value


def _validate_split_manifest_shape(
    manifest: Any,
    *,
    location: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(manifest, dict):
        raise TrainingCompatibilityError(f"{location} split manifest is missing")
    required = {
        "schema_version",
        "dataset_schema_version",
        "ingestion_report_sha256",
        "seed",
        "world_grid_profile_id",
        "world_grid_profile_hash",
        "world_grid_audit_sha256",
        "assignments",
    }
    if set(manifest) != required:
        raise TrainingCompatibilityError(
            f"{location} split manifest fields do not match schema 2.0"
        )
    if manifest["schema_version"] != "2.0":
        raise TrainingCompatibilityError(f"{location} split must use schema 2.0")
    if manifest["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
        raise TrainingCompatibilityError(
            f"{location} split dataset schema must be {DATASET_SCHEMA_VERSION}"
        )
    seed = manifest.get("seed")
    if type(seed) is not int or seed < 0:
        raise TrainingCompatibilityError(
            f"{location} split seed must be a nonnegative integer"
        )
    profile_id = manifest.get("world_grid_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise TrainingCompatibilityError(
            f"{location} split world-grid profile ID is invalid"
        )
    for name in (
        "ingestion_report_sha256",
        "world_grid_profile_hash",
        "world_grid_audit_sha256",
    ):
        _validate_sha256(manifest.get(name), f"{location} split {name}")
    assignments = manifest["assignments"]
    if not isinstance(assignments, list) or not assignments:
        raise TrainingCompatibilityError(
            f"{location} split assignments must be nonempty"
        )
    seen: set[str] = set()
    group_partitions: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {
            "session_id",
            "group_id",
            "partition",
            "trust_partition",
        }:
            raise TrainingCompatibilityError(
                f"{location} split assignment is invalid"
            )
        session_id = item["session_id"]
        if not isinstance(session_id, str) or not session_id or session_id in seen:
            raise TrainingCompatibilityError(
                f"{location} split session IDs must be unique and nonempty"
            )
        seen.add(session_id)
        group_id = item["group_id"]
        if not isinstance(group_id, str) or not group_id:
            raise TrainingCompatibilityError(
                f"{location} split group IDs must be nonempty"
            )
        if item["partition"] not in _PARTITIONS:
            raise TrainingCompatibilityError(
                f"{location} split partition is invalid"
            )
        if item["trust_partition"] not in {"attested", "unattested"}:
            raise TrainingCompatibilityError(
                f"{location} trust partition is invalid"
            )
        existing = group_partitions.setdefault(group_id, item["partition"])
        if existing != item["partition"]:
            raise TrainingCompatibilityError(
                f"{location} split group spans multiple partitions"
            )
    return manifest, _fingerprint(manifest)


def _validate_source_split(
    manifest: Any,
    compatibility: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    manifest, digest = _validate_split_manifest_shape(
        manifest,
        location="source checkpoint",
    )
    if digest != _validate_sha256(
        compatibility.get("split_manifest_sha256"),
        "compatibility.split_manifest_sha256",
    ):
        raise TrainingCompatibilityError("source split manifest digest mismatch")
    return manifest, digest


def load_source_encoder_checkpoint(
    path: str | Path,
    *,
    encoder: SpatiotemporalEncoder | None = None,
    expected_sha256: str | None = None,
) -> SourceEncoderCheckpoint:
    """Strictly load only ``encoder.*`` tensors from a RotationModel checkpoint."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    checkpoint_sha = _file_sha256(source_path)
    if expected_sha256 is not None and checkpoint_sha != _validate_sha256(
        expected_sha256, "expected source checkpoint hash"
    ):
        raise TrainingCompatibilityError("source checkpoint SHA-256 mismatch")
    try:
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingCompatibilityError(
            f"cannot load source checkpoint {source_path}: {exc}"
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_schema_version")
        != SOURCE_CHECKPOINT_SCHEMA_VERSION
    ):
        raise TrainingCompatibilityError(
            "source checkpoint must be a schema-2.0 RotationModel checkpoint"
        )
    wrapper = checkpoint.get("wrapper")
    if (
        not isinstance(wrapper, dict)
        or wrapper.get("type") != "RotationModel"
        or not isinstance(wrapper.get("state_dict"), Mapping)
    ):
        raise TrainingCompatibilityError("source checkpoint wrapper is not RotationModel")
    compatibility = checkpoint.get("compatibility")
    resolved = checkpoint.get("resolved_config")
    if not isinstance(compatibility, dict) or not isinstance(resolved, dict):
        raise TrainingCompatibilityError("source checkpoint metadata is incomplete")
    raw_encoder_config = compatibility.get("encoder_config")
    if not isinstance(raw_encoder_config, dict):
        raise TrainingCompatibilityError("source encoder configuration is missing")
    try:
        encoder_config = EncoderConfig(**raw_encoder_config)
    except (TypeError, ValueError) as exc:
        raise TrainingCompatibilityError(
            f"source encoder configuration is invalid: {exc}"
        ) from exc
    if encoder is None:
        encoder = SpatiotemporalEncoder(encoder_config)
    elif encoder.config != encoder_config:
        raise TrainingCompatibilityError(
            "injected encoder configuration differs from source checkpoint"
        )
    expected_state = encoder.state_dict()
    wrapper_state = wrapper["state_dict"]
    extracted: dict[str, Tensor] = {}
    for name, value in wrapper_state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TrainingCompatibilityError(
                "source model state must contain named tensors only"
            )
        if name.startswith("encoder."):
            stripped = name[len("encoder.") :]
            if stripped in extracted:
                raise TrainingCompatibilityError(
                    f"source checkpoint duplicates encoder tensor {stripped}"
                )
            extracted[stripped] = value
    missing = sorted(set(expected_state) - set(extracted))
    unexpected = sorted(set(extracted) - set(expected_state))
    if missing or unexpected:
        raise TrainingCompatibilityError(
            "source encoder tensor names mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for name, expected in expected_state.items():
        actual = extracted[name]
        if tuple(actual.shape) != tuple(expected.shape):
            raise TrainingCompatibilityError(
                f"source encoder tensor shape mismatch for {name}: "
                f"{tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        if actual.dtype != expected.dtype:
            raise TrainingCompatibilityError(
                f"source encoder tensor dtype mismatch for {name}: "
                f"{actual.dtype} != {expected.dtype}"
            )
    try:
        encoder.load_state_dict(extracted, strict=True)
    except RuntimeError as exc:
        raise TrainingCompatibilityError(
            f"source encoder state cannot be loaded exactly: {exc}"
        ) from exc

    dataset_schema = compatibility.get("dataset_schema_version")
    if dataset_schema != DATASET_SCHEMA_VERSION:
        raise TrainingCompatibilityError(
            f"source dataset schema must be {DATASET_SCHEMA_VERSION}"
        )
    manifest, split_digest = _validate_source_split(
        checkpoint.get("split_manifest"), compatibility
    )
    ingestion_hash = _validate_sha256(
        compatibility.get("ingestion_report_sha256"),
        "compatibility.ingestion_report_sha256",
    )
    if manifest["ingestion_report_sha256"] != ingestion_hash:
        raise TrainingCompatibilityError("source ingestion binding mismatch")
    profile_hash = _validate_sha256(
        compatibility.get("world_grid_profile_hash"),
        "compatibility.world_grid_profile_hash",
    )
    audit_hash = _validate_sha256(
        compatibility.get("world_grid_audit_sha256"),
        "compatibility.world_grid_audit_sha256",
    )
    if (
        manifest["world_grid_profile_hash"] != profile_hash
        or manifest["world_grid_audit_sha256"] != audit_hash
    ):
        raise TrainingCompatibilityError("source world-grid metadata mismatch")
    resolved_bindings = {
        "dataset_schema_version": dataset_schema,
        "ingestion_report_sha256": ingestion_hash,
        "split_manifest_sha256": split_digest,
        "world_grid_profile_hash": profile_hash,
        "world_grid_audit_sha256": audit_hash,
    }
    for name, expected_value in resolved_bindings.items():
        if resolved.get(name) != expected_value:
            raise TrainingCompatibilityError(
                f"source resolved {name} metadata mismatch"
            )
    sampler = compatibility.get("sampler")
    training = resolved.get("training")
    if not isinstance(sampler, dict) or not isinstance(training, dict):
        raise TrainingCompatibilityError("source sampler metadata is missing")
    if training.get("encoder_config") != raw_encoder_config:
        raise TrainingCompatibilityError(
            "source resolved encoder configuration mismatch"
        )
    resolved_profile = training.get("world_grid_profile")
    if (
        not isinstance(resolved_profile, dict)
        or resolved_profile.get("expected_profile_hash") != profile_hash
    ):
        raise TrainingCompatibilityError(
            "source resolved world-grid profile metadata mismatch"
        )
    context_length = sampler.get("window_length_ticks")
    if (
        type(context_length) is not int
        or context_length <= 0
        or training.get("window_length_ticks") != context_length
    ):
        raise TrainingCompatibilityError("source context length is inconsistent")
    if context_length > encoder_config.max_timesteps:
        raise TrainingCompatibilityError(
            "source context length exceeds encoder positional capacity"
        )
    source_revision = compatibility.get("source_revision")
    if source_revision is not None and not isinstance(source_revision, str):
        raise TrainingCompatibilityError("source revision metadata is invalid")
    if resolved.get("source_revision") != source_revision:
        raise TrainingCompatibilityError("source revision metadata mismatch")
    source_dataset_root = resolved.get("dataset_root")
    if source_dataset_root is not None and (
        not isinstance(source_dataset_root, str)
        or not source_dataset_root.strip()
    ):
        raise TrainingCompatibilityError("source dataset root metadata is invalid")
    return SourceEncoderCheckpoint(
        encoder=encoder,
        encoder_config=encoder_config,
        checkpoint_path=str(source_path),
        checkpoint_sha256=checkpoint_sha,
        split_manifest=manifest,
        split_manifest_sha256=split_digest,
        context_length_ticks=context_length,
        dataset_schema_version=dataset_schema,
        ingestion_report_sha256=ingestion_hash,
        world_grid_profile_id=manifest["world_grid_profile_id"],
        world_grid_profile_hash=profile_hash,
        world_grid_audit_sha256=audit_hash,
        source_revision=source_revision,
        dataset_root=source_dataset_root,
    )


load_pretrained_encoder_checkpoint = load_source_encoder_checkpoint


@dataclass(frozen=True, slots=True)
class PlannerTransferCheckpoint:
    checkpoint_path: str
    checkpoint_sha256: str
    source_checkpoint_sha256: str
    encoder_state: Mapping[str, Tensor]
    congestion_predictor_state: Mapping[str, Tensor]


def _validate_transfer_module_state(
    module: nn.Module,
    state: Mapping[str, Any],
    label: str,
) -> dict[str, Tensor]:
    expected = module.state_dict()
    if not isinstance(state, Mapping):
        raise TrainingCompatibilityError(f"transfer {label} state is missing")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise TrainingCompatibilityError(
            f"transfer {label} state must contain named tensors only"
        )
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise TrainingCompatibilityError(
            f"transfer {label} tensor names mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    result: dict[str, Tensor] = {}
    for name, expected_value in expected.items():
        actual = state[name]
        if tuple(actual.shape) != tuple(expected_value.shape):
            raise TrainingCompatibilityError(
                f"transfer {label} tensor shape mismatch for {name}"
            )
        if actual.dtype != expected_value.dtype:
            raise TrainingCompatibilityError(
                f"transfer {label} tensor dtype mismatch for {name}"
            )
        result[name] = actual.detach().cpu().clone()
    return result


def load_planner_transfer_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    source: SourceEncoderCheckpoint,
    profile: WorldGridProfile,
) -> PlannerTransferCheckpoint:
    """Load only the encoder and congestion predictor from legacy planner v2."""

    transfer_path = Path(path).resolve()
    if not transfer_path.is_file():
        raise FileNotFoundError(
            f"planner transfer checkpoint does not exist: {transfer_path}"
        )
    digest = _file_sha256(transfer_path)
    if digest != _validate_sha256(
        expected_sha256, "expected planner transfer checkpoint hash"
    ):
        raise TrainingCompatibilityError(
            "planner transfer checkpoint SHA-256 mismatch"
        )
    try:
        checkpoint = torch.load(
            transfer_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise TrainingCompatibilityError(
            f"cannot load planner transfer checkpoint {transfer_path}: {exc}"
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_schema_version")
        != LEGACY_PLANNER_TRANSFER_SCHEMA_VERSION
    ):
        raise TrainingCompatibilityError(
            "component transfer requires a legacy planner schema-2.0 checkpoint"
        )
    if checkpoint.get("source_checkpoint_sha256") != source.checkpoint_sha256:
        raise TrainingCompatibilityError(
            "planner transfer rotation-checkpoint provenance mismatch"
        )
    world_grid = checkpoint.get("world_grid_profile")
    if (
        not isinstance(world_grid, dict)
        or world_grid.get("id") != profile.profile_id
        or world_grid.get("hash") != profile.profile_hash
        or world_grid.get("audit_sha256")
        != profile.publication_audit.audit_sha256
    ):
        raise TrainingCompatibilityError(
            "planner transfer world-grid contract mismatch"
        )
    if checkpoint.get("target_schema_id") != "planner-targets:1.0":
        raise TrainingCompatibilityError(
            "planner transfer target schema must be planner-targets:1.0"
        )
    encoder_state = _validate_transfer_module_state(
        source.encoder, checkpoint.get("encoder_state"), "encoder"
    )
    planner_state = checkpoint.get("planner_state")
    if not isinstance(planner_state, Mapping):
        raise TrainingCompatibilityError("transfer planner state is missing")
    prefix = "congestion_predictor."
    congestion_state = {
        name[len(prefix) :]: value
        for name, value in planner_state.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    congestion_state = _validate_transfer_module_state(
        CongestionPredictor(), congestion_state, "congestion predictor"
    )
    return PlannerTransferCheckpoint(
        checkpoint_path=str(transfer_path),
        checkpoint_sha256=digest,
        source_checkpoint_sha256=source.checkpoint_sha256,
        encoder_state=encoder_state,
        congestion_predictor_state=congestion_state,
    )


def apply_planner_component_transfer(
    model: ExpertRotationPlannerModel,
    transfer: PlannerTransferCheckpoint,
) -> None:
    """Restore exactly the two approved components into a fresh planner."""

    try:
        model.encoder.load_state_dict(transfer.encoder_state, strict=True)
        model.planner.congestion_predictor.load_state_dict(
            transfer.congestion_predictor_state, strict=True
        )
    except RuntimeError as exc:
        raise TrainingCompatibilityError(
            f"planner component transfer cannot be applied exactly: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class PlannerOptimizerContract:
    group_parameter_names: tuple[tuple[str, tuple[str, ...]], ...]
    planner_learning_rate: float
    encoder_learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def build_planner_adamw_optimizer(
    model: nn.Module,
    config: PlannerTrainingConfig,
) -> tuple[torch.optim.AdamW, PlannerOptimizerContract]:
    """Build stable planner/encoder by decay/no-decay AdamW groups."""

    layer_norm_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    named_groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "planner_decay": [],
        "planner_no_decay": [],
        "encoder_decay": [],
        "encoder_no_decay": [],
    }
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        if name.startswith("planner."):
            owner = "planner"
        elif name.startswith("encoder."):
            owner = "encoder"
        else:
            raise TrainingConfigurationError(
                f"optimizer encountered parameter outside encoder/planner: {name}"
            )
        no_decay = name.endswith("bias") or id(parameter) in layer_norm_parameters
        named_groups[f"{owner}_{'no_decay' if no_decay else 'decay'}"].append(
            (name, parameter)
        )
    if any(not values for values in named_groups.values()):
        empty = [name for name, values in named_groups.items() if not values]
        raise TrainingConfigurationError(
            f"planner optimizer has empty required groups: {empty}"
        )
    groups: list[dict[str, Any]] = []
    for group_name in (
        "planner_decay",
        "planner_no_decay",
        "encoder_decay",
        "encoder_no_decay",
    ):
        owner = group_name.split("_", 1)[0]
        groups.append(
            {
                "params": [parameter for _, parameter in named_groups[group_name]],
                "lr": float(
                    config.planner_learning_rate
                    if owner == "planner"
                    else config.encoder_learning_rate
                ),
                "weight_decay": (
                    float(config.weight_decay)
                    if group_name.endswith("_decay")
                    and not group_name.endswith("_no_decay")
                    else 0.0
                ),
                "group_name": group_name,
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in config.adamw_betas),
        eps=float(config.adamw_epsilon),
    )
    contract = PlannerOptimizerContract(
        group_parameter_names=tuple(
            (
                group_name,
                tuple(name for name, _ in named_groups[group_name]),
            )
            for group_name in (
                "planner_decay",
                "planner_no_decay",
                "encoder_decay",
                "encoder_no_decay",
            )
        ),
        planner_learning_rate=float(config.planner_learning_rate),
        encoder_learning_rate=float(config.encoder_learning_rate),
        betas=tuple(float(value) for value in config.adamw_betas),
        epsilon=float(config.adamw_epsilon),
        weight_decay=float(config.weight_decay),
    )
    return optimizer, contract


build_planner_optimizer = build_planner_adamw_optimizer


class PlannerWarmupCosineScheduler:
    """Apply one warmup/cosine ratio while preserving encoder/planner LR ratio."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if (
            type(warmup_steps) is not int
            or warmup_steps < 0
            or warmup_steps > total_steps
        ):
            raise ValueError("warmup_steps must be in [0,total_steps]")
        if (
            isinstance(min_lr_ratio, bool)
            or not isinstance(min_lr_ratio, (int, float))
            or not math.isfinite(float(min_lr_ratio))
            or not 0.0 < min_lr_ratio <= 1.0
        ):
            raise ValueError("min_lr_ratio must be in (0,1]")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if any(value <= 0.0 or not math.isfinite(value) for value in self.base_lrs):
            raise ValueError("optimizer groups must begin with positive LRs")
        self.step_index = 0
        self._set_ratio(self.ratio_at(0))

    def ratio_at(self, step: int) -> float:
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise ValueError("scheduler step must be in [0,total_steps]")
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return step / self.warmup_steps
        span = self.total_steps - self.warmup_steps
        if span == 0:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / span
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _set_ratio(self, ratio: float) -> None:
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * ratio

    def step(self) -> float:
        if self.step_index >= self.total_steps:
            raise RuntimeError("scheduler advanced beyond total_steps")
        self.step_index += 1
        ratio = self.ratio_at(self.step_index)
        self._set_ratio(ratio)
        return ratio

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "step_index": self.step_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        contract = {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
        }
        differing = [key for key, value in contract.items() if state.get(key) != value]
        if differing:
            raise TrainingCompatibilityError(
                "planner scheduler mismatch: " + ", ".join(differing)
            )
        step = state.get("step_index")
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise TrainingCompatibilityError("planner scheduler step is invalid")
        self.step_index = step
        self._set_ratio(self.ratio_at(step))


WarmupCosineScheduler = PlannerWarmupCosineScheduler


@dataclass(frozen=True, slots=True)
class PlannerPrecisionSpec:
    device: torch.device
    precision: Literal["fp32", "bf16"]
    autocast_dtype: torch.dtype | None


def resolve_planner_precision(
    device: str = "auto",
    precision: Literal["auto", "fp32", "bf16"] = "auto",
) -> PlannerPrecisionSpec:
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise TrainingConfigurationError(
            "planner training supports exactly one process"
        )
    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        try:
            resolved_device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise TrainingConfigurationError(f"invalid device {device!r}") from exc
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainingConfigurationError("CUDA device requested but unavailable")
        if (
            resolved_device.index is not None
            and not 0 <= resolved_device.index < torch.cuda.device_count()
        ):
            raise TrainingConfigurationError(
                f"CUDA device index {resolved_device.index} is unavailable"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        if precision == "bf16" and not bf16_supported:
            raise TrainingConfigurationError(
                "BF16 requested but unsupported by this CUDA device"
            )
        use_bf16 = precision == "bf16" or (
            precision == "auto" and bf16_supported
        )
        return PlannerPrecisionSpec(
            device=resolved_device,
            precision="bf16" if use_bf16 else "fp32",
            autocast_dtype=torch.bfloat16 if use_bf16 else None,
        )
    if resolved_device.type != "cpu":
        raise TrainingConfigurationError(
            "planner training supports only CPU or CUDA"
        )
    if precision == "bf16":
        raise TrainingConfigurationError("BF16 planner training requires CUDA")
    return PlannerPrecisionSpec(
        device=resolved_device,
        precision="fp32",
        autocast_dtype=None,
    )


resolve_precision = resolve_planner_precision


def _autocast(spec: PlannerPrecisionSpec):
    if spec.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=spec.device.type,
        dtype=spec.autocast_dtype,
        enabled=True,
    )


def _source_tree_digest() -> str:
    root = Path(__file__).resolve().parent
    payload: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py"), key=lambda value: value.as_posix()):
        payload.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
            }
        )
    return _fingerprint(payload)


def _dependency_environment(spec: PlannerPrecisionSpec) -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version("fortnite-encoder")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    cuda_name = None
    cuda_capability = None
    if spec.device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(spec.device)
        cuda_capability = list(torch.cuda.get_device_capability(spec.device))
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "fortnite_encoder": package_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_name": cuda_name,
        "cuda_capability": cuda_capability,
        "resolved_device": str(spec.device),
        "resolved_precision": spec.precision,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def planner_model_shape_signature(model: ExpertRotationPlannerModel) -> dict[str, Any]:
    def signature(module: nn.Module) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in module.state_dict().items()
        ]

    return {
        "encoder": signature(model.encoder),
        "planner": signature(model.planner),
    }


@dataclass(frozen=True, slots=True)
class _PlannerDatasetProvenance:
    manifest: dict[str, Any]
    manifest_sha256: str
    split_session_ids: dict[str, tuple[str, ...]]
    session_paths: dict[str, Path]
    replay_sha256_by_session: dict[str, str]
    ingestion_report_sha256: str
    ingestion_binding_sha256: str
    dataset_validation_report_hash: str | None
    binding_mode: Literal["source", "independent"]


@dataclass(frozen=True, slots=True)
class _VerifiedPlannerSetup:
    config: PlannerTrainingConfig
    source: SourceEncoderCheckpoint
    transfer: PlannerTransferCheckpoint
    profile: WorldGridProfile
    planner: _PlannerDatasetProvenance
    source_replay_sha256_by_session: dict[str, str]
    overlap_report: dict[str, Any]
    microbatches_per_epoch: int
    total_optimizer_steps: int
    warmup_steps: int
    precision_spec: PlannerPrecisionSpec
    environment: dict[str, Any]


def _read_json_object(path: Path, location: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(
            f"cannot read {location} {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TrainingConfigurationError(f"{location} must be a JSON object")
    return raw


def _ingestion_dataset_context(
    dataset_root: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    dict[str, str],
    dict[str, Path],
]:
    report_path = dataset_root / "ingestion-report.json"
    if not report_path.is_file():
        raise TrainingConfigurationError(
            f"dataset ingestion report is missing: {report_path}"
        )
    report = _read_json_object(report_path, "ingestion report")
    try:
        binding = ingestion_binding_from_report(report)
    except WorldGridProfileError as exc:
        raise TrainingConfigurationError(
            f"invalid ingestion report binding: {exc}"
        ) from exc
    report_hash = _file_sha256(report_path)
    replay_by_session = {
        item["session_id"]: item["replay_sha256"]
        for item in binding["accepted_sessions"]
    }
    paths: dict[str, Path] = {}
    expected_by_trust: dict[str, set[str]] = {
        "attested": set(),
        "unattested": set(),
    }
    for item in binding["accepted_sessions"]:
        trust = item["disposition"].removeprefix("included_")
        session_id = item["session_id"]
        expected_by_trust[trust].add(session_id)
        path = dataset_root / trust / session_id
        if not path.is_dir():
            raise TrainingConfigurationError(
                f"included planner session directory is missing: {path}"
            )
        paths[session_id] = path
    for trust, expected in expected_by_trust.items():
        partition_root = dataset_root / trust
        actual = (
            {
                item.name
                for item in partition_root.iterdir()
                if item.is_dir()
            }
            if partition_root.is_dir()
            else set()
        )
        if actual != expected:
            raise TrainingConfigurationError(
                f"dataset partition {trust} does not exactly match the "
                "ingestion report"
            )
    return report, report_hash, binding, replay_by_session, paths


def _split_session_ids(manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    assignments = manifest["assignments"]
    return {
        partition: tuple(
            sorted(
                item["session_id"]
                for item in assignments
                if item["partition"] == partition
            )
        )
        for partition in _PARTITIONS
    }


def _validate_planner_manifest_binding(
    manifest: dict[str, Any],
    manifest_sha256: str,
    *,
    config: PlannerTrainingConfig,
    source: SourceEncoderCheckpoint,
    binding: dict[str, Any],
    ingestion_report_sha256: str,
) -> dict[str, tuple[str, ...]]:
    if config.seed is None:
        raise TrainingConfigurationError("planner seed is required")
    if (
        config.planner_dataset_binding is not None
        and manifest["seed"] != config.seed
    ):
        raise TrainingCompatibilityError("planner split seed mismatch")
    if manifest["ingestion_report_sha256"] != ingestion_report_sha256:
        raise TrainingCompatibilityError(
            "planner split ingestion-report fingerprint mismatch"
        )
    if (
        manifest["world_grid_profile_id"] != source.world_grid_profile_id
        or manifest["world_grid_profile_hash"] != source.world_grid_profile_hash
        or manifest["world_grid_audit_sha256"]
        != source.world_grid_audit_sha256
    ):
        raise TrainingCompatibilityError(
            "planner split world-grid publication binding mismatch"
        )
    expected = {
        item["session_id"]: item["disposition"].removeprefix("included_")
        for item in binding["accepted_sessions"]
    }
    assignments = {item["session_id"]: item for item in manifest["assignments"]}
    if set(assignments) != set(expected):
        missing = sorted(set(expected) - set(assignments))
        extra = sorted(set(assignments) - set(expected))
        raise TrainingCompatibilityError(
            "planner split does not exactly cover accepted ingestion sessions; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    for session_id, trust in expected.items():
        if assignments[session_id]["trust_partition"] != trust:
            raise TrainingCompatibilityError(
                f"planner split trust partition mismatch for {session_id}"
            )
    split_ids = _split_session_ids(manifest)
    if not split_ids["train"] or not split_ids["validation"]:
        raise TrainingConfigurationError(
            "planner split requires nonempty train and validation partitions"
        )
    if manifest_sha256 != _fingerprint(manifest):
        raise TrainingCompatibilityError("planner split manifest digest mismatch")
    return split_ids


def _validate_dataset_validation_report(
    path: Path,
    *,
    expected_report_hash: str,
    ingestion_report_sha256: str,
    ingestion_binding: dict[str, Any],
    source: SourceEncoderCheckpoint,
) -> str:
    report = _read_json_object(path, "planner dataset-validation report")
    actual_hash = report.get("report_hash")
    if actual_hash != expected_report_hash:
        raise TrainingCompatibilityError(
            "planner dataset-validation report hash does not match config"
        )
    if actual_hash != _payload_hash(report, "report_hash"):
        raise TrainingCompatibilityError(
            "planner dataset-validation report content hash mismatch"
        )
    expected_fields = {
        "schema_version": "2.0",
        "status": "compatible_independent_dataset",
        "binding_mode": "independent",
        "profile_id": source.world_grid_profile_id,
        "profile_hash": source.world_grid_profile_hash,
        "publication_audit_sha256": source.world_grid_audit_sha256,
        "ingestion_report_sha256": ingestion_report_sha256,
        "ingestion_binding_sha256": ingestion_binding[
            "ingestion_binding_sha256"
        ],
        "accepted_session_count": ingestion_binding["accepted_session_count"],
        "accepted_session_counts_by_build": ingestion_binding[
            "accepted_session_counts_by_build"
        ],
        "dataset_compatibility_verified": True,
    }
    differing = [
        name
        for name, expected in expected_fields.items()
        if report.get(name) != expected
    ]
    if differing:
        raise TrainingCompatibilityError(
            "planner dataset-validation binding mismatch: "
            + ", ".join(differing)
        )
    if report.get("canonical_counts_match") is not None or report.get(
        "canonical_digest_match"
    ) is not None:
        raise TrainingCompatibilityError(
            "independent validation must not claim canonical corpus equality"
        )
    sources = report.get("coordinate_sources")
    expected_sources = {
        "player_position",
        "player_event_position",
        "eligible_player_centroid",
        "zone_phase_source",
        "zone_phase_target",
        "zone_sample_current",
        "zone_sample_target",
    }
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise TrainingCompatibilityError(
            "planner dataset-validation report lacks all coordinate streams"
        )
    return actual_hash


def _source_replay_bindings(
    source: SourceEncoderCheckpoint,
    *,
    fallback_dataset_root: Path,
) -> dict[str, str]:
    source_root = (
        Path(source.dataset_root)
        if source.dataset_root is not None
        else fallback_dataset_root
    )
    _, report_hash, binding, replay_by_session, _ = _ingestion_dataset_context(
        source_root
    )
    if report_hash != source.ingestion_report_sha256:
        raise TrainingCompatibilityError(
            "source checkpoint ingestion report is unavailable or changed"
        )
    accepted_trust = {
        item["session_id"]: item["disposition"].removeprefix("included_")
        for item in binding["accepted_sessions"]
    }
    for item in source.split_manifest["assignments"]:
        session_id = item["session_id"]
        if (
            session_id not in replay_by_session
            or accepted_trust[session_id] != item["trust_partition"]
        ):
            raise TrainingCompatibilityError(
                f"source split has no exact ingestion binding for {session_id}"
            )
    return replay_by_session


def _session_replay_overlap_report(
    source: SourceEncoderCheckpoint,
    source_replays: Mapping[str, str],
    planner: _PlannerDatasetProvenance,
) -> dict[str, Any]:
    source_train_ids = set(source.session_ids_by_partition["train"])
    source_train_replays = {
        source_replays[session_id] for session_id in source_train_ids
    }
    session_overlaps: dict[str, list[str]] = {}
    replay_overlaps: dict[str, list[str]] = {}
    for partition in _PARTITIONS:
        ids = set(planner.split_session_ids[partition])
        session_overlaps[partition] = sorted(ids & source_train_ids)
        replay_overlaps[partition] = sorted(
            {
                planner.replay_sha256_by_session[session_id]
                for session_id in ids
            }
            & source_train_replays
        )
    violations = [
        {
            "partition": partition,
            "session_id_overlaps": session_overlaps[partition],
            "replay_sha256_overlaps": replay_overlaps[partition],
        }
        for partition in ("validation", "test")
        if session_overlaps[partition] or replay_overlaps[partition]
    ]
    result = {
        "policy": (
            "planner validation/test must be disjoint from encoder training "
            "by session_id and replay_sha256"
        ),
        "source_training_session_count": len(source_train_ids),
        "source_training_replay_sha256_count": len(source_train_replays),
        "planner_partition_counts": {
            partition: len(planner.split_session_ids[partition])
            for partition in _PARTITIONS
        },
        "session_id_overlaps_with_source_train": session_overlaps,
        "replay_sha256_overlaps_with_source_train": replay_overlaps,
        "violations": violations,
        "passed": not violations,
    }
    if violations:
        raise TrainingCompatibilityError(
            "planner validation/test overlaps encoder training by session ID "
            "or replay SHA-256"
        )
    return result


@dataclass(frozen=True, slots=True)
class ResolvedPlannerTrainingConfig:
    training: PlannerTrainingConfig
    dataset_root: str
    run_directory: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    planner_transfer_checkpoint_path: str
    planner_transfer_checkpoint_sha256: str
    planner_transfer_components: tuple[str, ...]
    source_dataset_schema_version: str
    source_ingestion_report_sha256: str
    source_split_manifest: dict[str, Any]
    source_split_manifest_sha256: str
    source_split_session_ids: dict[str, tuple[str, ...]]
    planner_dataset_schema_version: str
    planner_ingestion_report_sha256: str
    planner_ingestion_binding_sha256: str
    planner_split_manifest: dict[str, Any]
    planner_split_manifest_sha256: str
    planner_split_session_ids: dict[str, tuple[str, ...]]
    planner_dataset_validation_report_hash: str | None
    planner_dataset_binding_mode: str
    session_replay_overlap_report: dict[str, Any]
    dataset_schema_version: str
    ingestion_report_sha256: str
    split_manifest_sha256: str
    split_session_ids: dict[str, tuple[str, ...]]
    train_session_count: int
    validation_session_count: int
    test_session_count: int
    context_length_ticks: int
    microbatches_per_epoch: int
    total_optimizer_steps: int
    warmup_steps: int
    device: str
    precision: str
    world_grid_profile_id: str
    world_grid_profile_hash: str
    world_grid_audit_sha256: str
    world_grid_coordinate_audit_sha256: str
    observation_provider_id: str
    target_schema_id: str
    uniform_query_sampler_version: str
    validation_query_sample: tuple[dict[str, Any], ...]
    validation_query_sample_sha256: str
    source_tree_sha256: str
    source_revision: str | None
    source_encoder_revision: str | None
    dependency_environment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def _prepare_config(config: PlannerTrainingConfig) -> PlannerTrainingConfig:
    profile = config.world_grid_profile
    binding = config.planner_dataset_binding
    return dataclasses.replace(
        config,
        dataset_root=str(Path(config.dataset_root).resolve()),
        run_directory=str(Path(config.run_directory).resolve()),
        source_checkpoint=str(Path(config.source_checkpoint).resolve()),
        planner_transfer_checkpoint=str(
            Path(config.planner_transfer_checkpoint).resolve()
        ),
        world_grid_profile=(
            dataclasses.replace(
                profile,
                path=str(Path(profile.path).resolve()),
                audit_path=str(Path(profile.audit_path).resolve()),
            )
            if profile is not None
            else None
        ),
        planner_dataset_binding=(
            dataclasses.replace(
                binding,
                split_manifest_path=str(
                    Path(binding.split_manifest_path).resolve()
                ),
                dataset_validation_path=str(
                    Path(binding.dataset_validation_path).resolve()
                ),
            )
            if binding is not None
            else None
        ),
    )


def _profile_config_from_source(
    source_path: Path,
) -> WorldGridProfileConfig:
    try:
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        raw = checkpoint["resolved_config"]["training"]["world_grid_profile"]
        return WorldGridProfileConfig(
            path=str(Path(raw["path"]).resolve()),
            expected_profile_hash=raw["expected_profile_hash"],
            audit_path=str(Path(raw["audit_path"]).resolve()),
        )
    except Exception as exc:
        raise TrainingConfigurationError(
            "world_grid_profile is required because it cannot be recovered "
            f"from the source checkpoint: {exc}"
        ) from exc


def _derive_schedule(
    config: PlannerTrainingConfig,
    train_session_count: int,
) -> tuple[int, int, int]:
    assert config.batch_size is not None
    microbatches = math.ceil(train_session_count / config.batch_size)
    by_epochs = (
        (microbatches * config.max_epochs) // config.accumulation_steps
        if config.max_epochs is not None
        else None
    )
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


@dataclass(frozen=True, slots=True)
class SelfConditioningWeights:
    teacher: float
    rollout: float


def self_conditioning_weights(
    upcoming_optimizer_update: int,
) -> SelfConditioningWeights:
    """Return the exact update-indexed teacher/rollout mixture."""

    if (
        type(upcoming_optimizer_update) is not int
        or upcoming_optimizer_update <= 0
    ):
        raise ValueError("upcoming_optimizer_update must be positive")
    if upcoming_optimizer_update <= 43:
        rollout = 0.0
    elif upcoming_optimizer_update <= 86:
        rollout = 0.5 * (upcoming_optimizer_update - 43) / 43.0
    else:
        rollout = 0.5
    return SelfConditioningWeights(teacher=1.0 - rollout, rollout=rollout)


def _parse_planner_training_config(
    config: PlannerTrainingConfig | str | Path,
) -> PlannerTrainingConfig:
    if isinstance(config, (str, Path)):
        parsed = PlannerTrainingConfig.from_json(config)
    elif isinstance(config, PlannerTrainingConfig):
        parsed = _prepare_config(config)
    else:
        raise TypeError(
            "config must be PlannerTrainingConfig or a JSON configuration path"
        )
    if parsed.world_grid_profile is None:
        parsed = dataclasses.replace(
            parsed,
            world_grid_profile=_profile_config_from_source(
                Path(parsed.source_checkpoint)
            ),
        )
    return parsed


def _verify_planner_setup(
    config: PlannerTrainingConfig | str | Path,
) -> _VerifiedPlannerSetup:
    parsed = _parse_planner_training_config(config)
    source = load_source_encoder_checkpoint(
        parsed.source_checkpoint,
        expected_sha256=parsed.expected_source_checkpoint_sha256,
    )
    assert parsed.seed is not None
    assert parsed.batch_size is not None
    assert parsed.world_grid_profile is not None
    dataset_root = Path(parsed.dataset_root)
    (
        _,
        ingestion_hash,
        ingestion_binding,
        replay_by_session,
        session_paths,
    ) = _ingestion_dataset_context(dataset_root)

    binding_config = parsed.planner_dataset_binding
    if binding_config is None:
        if ingestion_hash != source.ingestion_report_sha256:
            raise TrainingCompatibilityError(
                "current dataset does not match source checkpoint ingestion report"
            )
        profile = parsed.world_grid_profile.load(dataset_root)
        manifest = source.split_manifest
        manifest_hash = source.split_manifest_sha256
        validation_hash = None
        binding_mode: Literal["source", "independent"] = "source"
    else:
        if ingestion_hash != binding_config.expected_ingestion_report_sha256:
            raise TrainingCompatibilityError(
                "planner ingestion report SHA-256 mismatch"
            )
        try:
            profile = WorldGridProfile.load(
                parsed.world_grid_profile.path,
                expected_hash=(
                    parsed.world_grid_profile.expected_profile_hash
                ),
                audit_path=parsed.world_grid_profile.audit_path,
                ingestion_report_path=dataset_root / "ingestion-report.json",
                audit_binding_mode="independent",
            )
        except WorldGridProfileError as exc:
            raise TrainingConfigurationError(
                f"invalid independent world-grid binding: {exc}"
            ) from exc
        manifest_path = Path(binding_config.split_manifest_path)
        manifest = _read_json_object(
            manifest_path,
            "planner split manifest",
        )
        manifest, manifest_hash = _validate_split_manifest_shape(
            manifest,
            location="planner",
        )
        if manifest_hash != binding_config.expected_split_manifest_sha256:
            raise TrainingCompatibilityError(
                "planner split manifest SHA-256 mismatch"
            )
        validation_hash = _validate_dataset_validation_report(
            Path(binding_config.dataset_validation_path),
            expected_report_hash=(
                binding_config.expected_dataset_validation_report_hash
            ),
            ingestion_report_sha256=ingestion_hash,
            ingestion_binding=ingestion_binding,
            source=source,
        )
        binding_mode = "independent"

    if (
        profile.profile_id != source.world_grid_profile_id
        or profile.profile_hash != source.world_grid_profile_hash
        or profile.publication_audit.audit_sha256
        != source.world_grid_audit_sha256
    ):
        raise TrainingCompatibilityError(
            "world-grid publication binding differs from source checkpoint"
        )
    transfer = load_planner_transfer_checkpoint(
        parsed.planner_transfer_checkpoint,
        expected_sha256=(
            parsed.expected_planner_transfer_checkpoint_sha256
        ),
        source=source,
        profile=profile,
    )
    split_ids = _validate_planner_manifest_binding(
        manifest,
        manifest_hash,
        config=parsed,
        source=source,
        binding=ingestion_binding,
        ingestion_report_sha256=ingestion_hash,
    )
    planner = _PlannerDatasetProvenance(
        manifest=manifest,
        manifest_sha256=manifest_hash,
        split_session_ids=split_ids,
        session_paths=session_paths,
        replay_sha256_by_session=replay_by_session,
        ingestion_report_sha256=ingestion_hash,
        ingestion_binding_sha256=ingestion_binding[
            "ingestion_binding_sha256"
        ],
        dataset_validation_report_hash=validation_hash,
        binding_mode=binding_mode,
    )
    source_replays = _source_replay_bindings(
        source,
        fallback_dataset_root=dataset_root,
    )
    overlap = _session_replay_overlap_report(
        source,
        source_replays,
        planner,
    )
    microbatches, total_steps, warmup_steps = _derive_schedule(
        parsed,
        len(split_ids["train"]),
    )
    spec = resolve_planner_precision(parsed.device, parsed.precision)
    environment = _dependency_environment(spec)
    return _VerifiedPlannerSetup(
        config=parsed,
        source=source,
        transfer=transfer,
        profile=profile,
        planner=planner,
        source_replay_sha256_by_session=source_replays,
        overlap_report=overlap,
        microbatches_per_epoch=microbatches,
        total_optimizer_steps=total_steps,
        warmup_steps=warmup_steps,
        precision_spec=spec,
        environment=environment,
    )


def verify_planner_training(
    config: PlannerTrainingConfig | str | Path,
) -> dict[str, Any]:
    """Verify every launch contract without opening Parquet or writing a run."""

    setup = _verify_planner_setup(config)
    split_ids = setup.planner.split_session_ids
    return {
        "status": "verified",
        "source_checkpoint_sha256": setup.source.checkpoint_sha256,
        "planner_transfer_checkpoint_sha256": (
            setup.transfer.checkpoint_sha256
        ),
        "planner_transfer_components": [
            "encoder",
            "congestion_predictor",
        ],
        "source_ingestion_report_sha256": (
            setup.source.ingestion_report_sha256
        ),
        "source_split_manifest_sha256": setup.source.split_manifest_sha256,
        "planner_ingestion_report_sha256": (
            setup.planner.ingestion_report_sha256
        ),
        "planner_ingestion_binding_sha256": (
            setup.planner.ingestion_binding_sha256
        ),
        "planner_split_manifest_sha256": setup.planner.manifest_sha256,
        "planner_dataset_validation_report_hash": (
            setup.planner.dataset_validation_report_hash
        ),
        "planner_dataset_binding_mode": setup.planner.binding_mode,
        "split_counts": {
            partition: len(split_ids[partition]) for partition in _PARTITIONS
        },
        "microbatches_per_epoch": setup.microbatches_per_epoch,
        "total_optimizer_steps": setup.total_optimizer_steps,
        "warmup_steps": setup.warmup_steps,
        "device": str(setup.precision_spec.device),
        "precision": setup.precision_spec.precision,
        "cuda_available": setup.environment["cuda_available"],
        "cuda_device_name": setup.environment["cuda_device_name"],
        "session_replay_overlap_report": setup.overlap_report,
        "parquet_opened": False,
        "run_directory_created": False,
    }


def _stable_seed(*values: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _uniform_sample(
    queries: Sequence[PlannerQuery],
    count: int,
    seed: int,
) -> tuple[PlannerQuery, ...]:
    if not queries:
        return ()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(queries), generator=generator).tolist()
    return tuple(queries[index] for index in order[: min(count, len(order))])


def _training_session_order(
    session_ids: Sequence[str],
    *,
    seed: int,
    epoch: int,
) -> tuple[str, ...]:
    ordered = tuple(sorted(session_ids))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, "training-session-order", epoch))
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    return tuple(ordered[index] for index in permutation)


def _batches(values: Sequence[str], batch_size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(values[start : start + batch_size])
        for start in range(0, len(values), batch_size)
    )


class _PlannerSessionRepository:
    def __init__(
        self,
        paths: Mapping[str, Path],
        active_session_ids: Iterable[str],
        test_session_ids: Iterable[str],
        profile: WorldGridProfile,
    ) -> None:
        active = set(active_session_ids)
        tests = set(test_session_ids)
        if active & tests:
            raise TrainingConfigurationError(
                "planner repository active and test session IDs overlap"
            )
        self._paths = dict(paths)
        if set(self._paths) != active:
            raise TrainingConfigurationError(
                "active repository IDs do not match train/validation split"
            )
        if set(self._paths) & tests:
            raise TrainingConfigurationError(
                "planner test sessions must be excluded from the repository"
            )
        self._test_session_ids = tests
        self._profile = profile
        self._cache: dict[str, PlannerTargetSource] = {}
        self.opened_session_ids: list[str] = []
        self.open_attempt_session_ids: list[str] = []
        self._data_access_path: Path | None = None
        self._split_manifest_sha256: str | None = None

    def attach_data_access_audit(
        self,
        path: Path,
        *,
        split_manifest_sha256: str,
    ) -> None:
        self._data_access_path = path
        self._split_manifest_sha256 = split_manifest_sha256
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TrainingCompatibilityError(
                    f"cannot resume data-access audit: {exc}"
                ) from exc
            if (
                not isinstance(previous, dict)
                or previous.get("planner_split_manifest_sha256")
                != split_manifest_sha256
            ):
                raise TrainingCompatibilityError(
                    "persisted data-access audit split mismatch"
                )
            prior_attempts = previous.get(
                "parquet_open_attempt_session_ids"
            )
            prior_opened = previous.get("parquet_opened_session_ids")
            if not isinstance(prior_attempts, list) or not isinstance(
                prior_opened, list
            ):
                raise TrainingCompatibilityError(
                    "persisted data-access audit is invalid"
                )
            self.open_attempt_session_ids.extend(prior_attempts)
            self.opened_session_ids.extend(prior_opened)
        self._write_data_access_audit()

    def data_access_report(self) -> dict[str, Any]:
        attempted_tests = sorted(
            set(self.open_attempt_session_ids) & self._test_session_ids
        )
        opened_tests = sorted(
            set(self.opened_session_ids) & self._test_session_ids
        )
        return {
            "schema_version": "1.0",
            "planner_split_manifest_sha256": self._split_manifest_sha256,
            "repository_session_ids": sorted(self._paths),
            "excluded_test_session_ids": sorted(self._test_session_ids),
            "parquet_open_attempt_session_ids": list(
                self.open_attempt_session_ids
            ),
            "parquet_opened_session_ids": list(self.opened_session_ids),
            "test_session_parquet_open_attempt_count": len(attempted_tests),
            "test_session_parquet_open_count": len(opened_tests),
            "zero_test_session_parquet_opens": not attempted_tests
            and not opened_tests,
        }

    def _write_data_access_audit(self) -> None:
        if self._data_access_path is not None:
            _atomic_write_json(
                self._data_access_path,
                self.data_access_report(),
            )

    def get(self, session_id: str) -> PlannerTargetSource:
        if session_id not in self._paths:
            raise TrainingCompatibilityError(
                f"attempted to open inactive or test session {session_id}"
            )
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached
        path = self._paths[session_id]
        self.open_attempt_session_ids.append(session_id)
        self._write_data_access_audit()
        match = load_match_session(path)
        if match.session_id != session_id:
            raise TrainingCompatibilityError(
                "session directory identity changed after source split"
            )
        source = load_planner_target_source(path, match, self._profile)
        self._cache[session_id] = source
        self.opened_session_ids.append(session_id)
        self._write_data_access_audit()
        return source


@dataclass(frozen=True, slots=True)
class PlannerQueryExample:
    match: TensorizedMatch
    focal_team_index: int
    targets: PlannerTargets
    query: PlannerQuery


def prepare_planner_query_examples(
    sources: Mapping[str, PlannerTargetSource],
    queries: Sequence[PlannerQuery],
    *,
    context_length_ticks: int,
) -> tuple[PlannerQueryExample, ...]:
    """Build full-match targets, then create one trailing slice per query."""

    if type(context_length_ticks) is not int or context_length_ticks <= 0:
        raise ValueError("context_length_ticks must be positive")
    positions: dict[PlannerQuery, tuple[PlannerTargets, int]] = {}
    for session_id in sorted({query.session_id for query in queries}):
        if session_id not in sources:
            raise ValueError(f"query source {session_id} was not loaded")
        session_queries = tuple(
            query for query in queries if query.session_id == session_id
        )
        supervision = build_planner_targets(sources[session_id], session_queries)
        for index, query in enumerate(session_queries):
            positions[query] = (
                slice_planner_targets(supervision.targets, index),
                index,
            )

    examples: list[PlannerQueryExample] = []
    for query in queries:
        source = sources[query.session_id]
        match = source.match
        team = match.team_ids.index(query.team_id)
        matches = (
            match.absolute_tick_index == query.absolute_tick_index
        ).nonzero(as_tuple=False)
        if matches.numel() != 1:
            raise ValueError("query tick disappeared from full target source")
        position = int(matches.item())
        start_position = max(0, position - context_length_ticks + 1)
        start_tick = int(match.absolute_tick_index[start_position].item())
        causal = slice_window(
            match,
            start_tick=start_tick,
            length=position - start_position + 1,
        )
        examples.append(
            PlannerQueryExample(
                match=causal,
                focal_team_index=team,
                targets=positions[query][0],
                query=query,
            )
        )
    return tuple(examples)


def collate_planner_query_examples(
    examples: Sequence[PlannerQueryExample],
    policy_config: PlannerPolicyConfig | None = None,
) -> tuple[EncoderBatch, PlannerObservation, PlannerTargets]:
    if not examples:
        raise ValueError("at least one planner query example is required")
    collated = collate_encoder_inputs(example.match for example in examples)
    observation = build_planner_observation(
        collated.batch,
        [example.focal_team_index for example in examples],
        policy_config,
    )
    targets = collate_planner_targets(example.targets for example in examples)
    return collated.batch, observation, targets


class PlannerMetricAccumulator:
    def __init__(self, config: PlannerLossConfig | None = None) -> None:
        self.config = config or PlannerLossConfig()
        self.route_cell_numerator = 0.0
        self.route_cell_count = 0
        self.route_offset_numerator = 0.0
        self.route_offset_count = 0
        self.congestion_numerator = 0.0
        self.congestion_horizon_count = 0

    def update(self, loss: PlannerLossComponents) -> None:
        self.route_cell_numerator += (
            float(loss.route_cell.detach().float().item())
            * loss.route_cell_count
        )
        self.route_cell_count += loss.route_cell_count
        self.route_offset_numerator += (
            float(loss.route_offset.detach().float().item())
            * 2
            * loss.route_offset_count
        )
        self.route_offset_count += loss.route_offset_count
        self.congestion_numerator += (
            float(loss.congestion.detach().float().item())
            * 32
            * 32
            * loss.congestion_horizon_count
        )
        self.congestion_horizon_count += loss.congestion_horizon_count

    def merge(self, other: PlannerMetricAccumulator) -> None:
        self.route_cell_numerator += other.route_cell_numerator
        self.route_cell_count += other.route_cell_count
        self.route_offset_numerator += other.route_offset_numerator
        self.route_offset_count += other.route_offset_count
        self.congestion_numerator += other.congestion_numerator
        self.congestion_horizon_count += other.congestion_horizon_count

    def has_supervision(self) -> bool:
        return bool(
            self.route_cell_count
            or self.route_offset_count
            or self.congestion_horizon_count
        )

    def metrics(self) -> dict[str, Any]:
        route_cell = (
            self.route_cell_numerator / self.route_cell_count
            if self.route_cell_count
            else 0.0
        )
        route_offset = (
            self.route_offset_numerator / (2 * self.route_offset_count)
            if self.route_offset_count
            else 0.0
        )
        congestion = (
            self.congestion_numerator
            / (32 * 32 * self.congestion_horizon_count)
            if self.congestion_horizon_count
            else 0.0
        )
        total = (
            self.config.route_cell_weight * route_cell
            + self.config.route_offset_weight * route_offset
            + self.config.congestion_weight * congestion
        )
        return {
            "total": total,
            "route_cell": route_cell,
            "route_offset": route_offset,
            "congestion": congestion,
            "counts": {
                "route_cell": self.route_cell_count,
                "route_offset": self.route_offset_count,
                "congestion_horizon": self.congestion_horizon_count,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "route_cell_numerator": self.route_cell_numerator,
            "route_cell_count": self.route_cell_count,
            "route_offset_numerator": self.route_offset_numerator,
            "route_offset_count": self.route_offset_count,
            "congestion_numerator": self.congestion_numerator,
            "congestion_horizon_count": self.congestion_horizon_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.state_dict()):
            raise TrainingCompatibilityError(
                "planner metric accumulator fields do not match"
            )
        for name in (
            "route_cell_count",
            "route_offset_count",
            "congestion_horizon_count",
        ):
            value = state[name]
            if type(value) is not int or value < 0:
                raise TrainingCompatibilityError(
                    f"planner metric count {name} is invalid"
                )
            setattr(self, name, value)
        for name in (
            "route_cell_numerator",
            "route_offset_numerator",
            "congestion_numerator",
        ):
            value = state[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TrainingCompatibilityError(
                    f"planner metric numerator {name} is invalid"
                )
            setattr(self, name, float(value))


class SelfConditionedMetricAccumulator:
    """Count-normalized teacher, rollout, blended-route, and congestion metrics."""

    _numeric_fields = (
        "teacher_cell_numerator",
        "teacher_offset_numerator",
        "rollout_cell_numerator",
        "rollout_offset_numerator",
        "blended_cell_numerator",
        "blended_offset_numerator",
        "congestion_numerator",
    )
    _count_fields = (
        "teacher_count",
        "rollout_count",
        "blended_count",
        "congestion_horizon_count",
    )

    def __init__(self, config: PlannerLossConfig | None = None) -> None:
        self.config = config or PlannerLossConfig()
        for name in self._numeric_fields:
            setattr(self, name, 0.0)
        for name in self._count_fields:
            setattr(self, name, 0)

    def update(
        self,
        teacher: PlannerLossComponents,
        rollout: Any,
        mixture: SelfConditioningWeights,
    ) -> None:
        count = teacher.route_cell_count
        self.teacher_cell_numerator += float(
            teacher.route_cell.detach().float().item()
        ) * count
        self.teacher_offset_numerator += float(
            teacher.route_offset.detach().float().item()
        ) * 2 * count
        self.teacher_count += count
        rollout_cell = teacher.route_cell
        rollout_offset = teacher.route_offset
        if rollout is not None:
            if (
                rollout.route_cell_count != count
                or rollout.route_offset_count != count
            ):
                raise RuntimeError("teacher and rollout supervision counts differ")
            rollout_cell = rollout.route_cell
            rollout_offset = rollout.route_offset
            self.rollout_cell_numerator += float(
                rollout_cell.detach().float().item()
            ) * count
            self.rollout_offset_numerator += float(
                rollout_offset.detach().float().item()
            ) * 2 * count
            self.rollout_count += count
        blended_cell = (
            mixture.teacher * teacher.route_cell
            + mixture.rollout * rollout_cell
        )
        blended_offset = (
            mixture.teacher * teacher.route_offset
            + mixture.rollout * rollout_offset
        )
        self.blended_cell_numerator += float(
            blended_cell.detach().float().item()
        ) * count
        self.blended_offset_numerator += float(
            blended_offset.detach().float().item()
        ) * 2 * count
        self.blended_count += count
        self.congestion_numerator += float(
            teacher.congestion.detach().float().item()
        ) * 32 * 32 * teacher.congestion_horizon_count
        self.congestion_horizon_count += teacher.congestion_horizon_count

    def has_supervision(self) -> bool:
        return bool(self.blended_count or self.congestion_horizon_count)

    def metrics(self) -> dict[str, Any]:
        def mean(numerator: float, count: int, dimensions: int = 1) -> float:
            return numerator / (dimensions * count) if count else 0.0

        teacher_cell = mean(self.teacher_cell_numerator, self.teacher_count)
        teacher_offset = mean(
            self.teacher_offset_numerator, self.teacher_count, 2
        )
        rollout_cell = mean(self.rollout_cell_numerator, self.rollout_count)
        rollout_offset = mean(
            self.rollout_offset_numerator, self.rollout_count, 2
        )
        blended_cell = mean(self.blended_cell_numerator, self.blended_count)
        blended_offset = mean(
            self.blended_offset_numerator, self.blended_count, 2
        )
        congestion = mean(
            self.congestion_numerator,
            self.congestion_horizon_count,
            32 * 32,
        )
        total = (
            self.config.route_cell_weight * blended_cell
            + self.config.route_offset_weight * blended_offset
            + self.config.congestion_weight * congestion
        )
        return {
            "total": total,
            "teacher_route_cell": teacher_cell,
            "teacher_route_offset": teacher_offset,
            "rollout_route_cell": rollout_cell,
            "rollout_route_offset": rollout_offset,
            "blended_route_cell": blended_cell,
            "blended_route_offset": blended_offset,
            "route_cell": blended_cell,
            "route_offset": blended_offset,
            "congestion": congestion,
            "counts": {
                "route_cell": self.blended_count,
                "route_offset": self.blended_count,
                "teacher_route": self.teacher_count,
                "rollout_route": self.rollout_count,
                "congestion_horizon": self.congestion_horizon_count,
            },
        }

    def merge_metrics(self, metrics: Mapping[str, Any]) -> None:
        counts = metrics["counts"]
        teacher_count = int(counts["teacher_route"])
        rollout_count = int(counts["rollout_route"])
        blended_count = int(counts["route_cell"])
        congestion_count = int(counts["congestion_horizon"])
        self.teacher_cell_numerator += float(
            metrics["teacher_route_cell"]
        ) * teacher_count
        self.teacher_offset_numerator += float(
            metrics["teacher_route_offset"]
        ) * 2 * teacher_count
        self.rollout_cell_numerator += float(
            metrics["rollout_route_cell"]
        ) * rollout_count
        self.rollout_offset_numerator += float(
            metrics["rollout_route_offset"]
        ) * 2 * rollout_count
        self.blended_cell_numerator += float(
            metrics["blended_route_cell"]
        ) * blended_count
        self.blended_offset_numerator += float(
            metrics["blended_route_offset"]
        ) * 2 * blended_count
        self.congestion_numerator += float(metrics["congestion"]) * (
            32 * 32 * congestion_count
        )
        self.teacher_count += teacher_count
        self.rollout_count += rollout_count
        self.blended_count += blended_count
        self.congestion_horizon_count += congestion_count

    def state_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (*self._numeric_fields, *self._count_fields)
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.state_dict()):
            raise TrainingCompatibilityError(
                "self-conditioned metric accumulator fields do not match"
            )
        for name in self._count_fields:
            value = state[name]
            if type(value) is not int or value < 0:
                raise TrainingCompatibilityError(
                    f"self-conditioned metric count {name} is invalid"
                )
            setattr(self, name, value)
        for name in self._numeric_fields:
            value = state[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TrainingCompatibilityError(
                    f"self-conditioned metric numerator {name} is invalid"
                )
            setattr(self, name, float(value))


def validate_planner_numerics(
    logits: PlannerLogits,
    loss: PlannerLossComponents,
    queries: Sequence[PlannerQuery] = (),
) -> None:
    for name in ("density_logits", "density", "cell_logits", "offsets"):
        value = getattr(logits, name)
        if not bool(torch.isfinite(value).all()):
            raise TrainingNumericalError(
                json.dumps(
                    {
                        "kind": "nonfinite_planner_output",
                        "tensor": name,
                        "queries": [dataclasses.asdict(query) for query in queries[:8]],
                    },
                    sort_keys=True,
                )
            )
    for name in ("total", "route_cell", "route_offset", "congestion"):
        value = getattr(loss, name)
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise TrainingNumericalError(
                json.dumps(
                    {
                        "kind": "nonfinite_planner_loss",
                        "component": name,
                        "queries": [dataclasses.asdict(query) for query in queries[:8]],
                    },
                    sort_keys=True,
                )
            )


def planner_gradient_norm_and_finite(
    model: nn.Module,
    queries: Sequence[PlannerQuery] = (),
) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach()
        if not bool(torch.isfinite(detached).all()):
            raise TrainingNumericalError(
                json.dumps(
                    {
                        "kind": "nonfinite_planner_gradient",
                        "parameter": name,
                        "queries": [dataclasses.asdict(query) for query in queries[:8]],
                    },
                    sort_keys=True,
                )
            )
        squared += detached.double().pow(2).sum().cpu()
    norm = math.sqrt(float(squared.item()))
    if not math.isfinite(norm):
        raise TrainingNumericalError(
            json.dumps({"kind": "nonfinite_planner_gradient_norm"})
        )
    return norm


@dataclass(frozen=True, slots=True)
class PlannerMicrobatchResult:
    metrics: dict[str, Any]
    query_count: int
    supervised: bool

    @property
    def total_loss(self) -> float:
        return float(self.metrics["total"])


def run_planner_microbatch(
    *,
    model: ExpertRotationPlannerModel,
    examples: Sequence[PlannerQueryExample],
    spec: PlannerPrecisionSpec,
    loss_config: PlannerLossConfig | None = None,
    policy_config: PlannerPolicyConfig | None = None,
    max_queries_per_forward: int,
    backward: bool,
    loss_scale: float = 1.0,
    upcoming_optimizer_update: int = 1,
) -> PlannerMicrobatchResult:
    """Shared bounded forward/backward path used by training and smoke tests."""

    if not examples:
        return PlannerMicrobatchResult(
            SelfConditionedMetricAccumulator(loss_config).metrics(), 0, False
        )
    if type(max_queries_per_forward) is not int or max_queries_per_forward <= 0:
        raise ValueError("max_queries_per_forward must be positive")
    if not math.isfinite(loss_scale) or loss_scale <= 0.0:
        raise ValueError("loss_scale must be finite and positive")
    weights = loss_config or PlannerLossConfig()
    mixture = self_conditioning_weights(upcoming_optimizer_update)
    all_targets = collate_planner_targets(example.targets for example in examples)
    global_counts = {
        "route_cell": int(all_targets.route_mask.sum().item()),
        "route_offset": int(all_targets.route_mask.sum().item()),
        "congestion": int(all_targets.congestion_mask.sum().item()),
    }
    accumulator = SelfConditionedMetricAccumulator(weights)
    for start in range(0, len(examples), max_queries_per_forward):
        chunk = examples[start : start + max_queries_per_forward]
        encoder_batch, observation, targets = collate_planner_query_examples(
            chunk, policy_config
        )
        non_blocking = spec.device.type == "cuda"
        encoder_batch = encoder_batch.to(spec.device, non_blocking=non_blocking)
        observation = observation.to(spec.device, non_blocking=non_blocking)
        targets = targets.to(spec.device, non_blocking=non_blocking)
        with _autocast(spec):
            teacher_logits, rollout_logits, generated_prefix = model.teacher_and_rollout(
                encoder_batch,
                observation,
                targets.previous_waypoints,
                include_rollout=mixture.rollout > 0.0,
            )
        teacher_loss = compute_planner_loss(teacher_logits, targets, weights)
        rollout_loss = (
            compute_planner_route_loss(rollout_logits, targets, weights)
            if rollout_logits is not None
            else None
        )
        validate_planner_numerics(
            teacher_logits, teacher_loss, [example.query for example in chunk]
        )
        if rollout_logits is not None:
            validate_planner_numerics(
                rollout_logits,
                teacher_loss,
                [example.query for example in chunk],
            )
            assert rollout_loss is not None
            if not bool(
                torch.isfinite(rollout_loss.route_cell)
                & torch.isfinite(rollout_loss.route_offset)
            ):
                raise TrainingNumericalError(
                    json.dumps({"kind": "nonfinite_rollout_route_loss"})
                )
            if generated_prefix is None or (
                generated_prefix.cells.requires_grad
                or generated_prefix.offsets.requires_grad
            ):
                raise RuntimeError("generated rollout prefixes must be detached")
        accumulator.update(teacher_loss, rollout_loss, mixture)
        if backward and any(global_counts.values()):
            cell_fraction = (
                teacher_loss.route_cell_count / global_counts["route_cell"]
                if global_counts["route_cell"]
                else 1.0 / math.ceil(len(examples) / max_queries_per_forward)
            )
            offset_fraction = (
                teacher_loss.route_offset_count / global_counts["route_offset"]
                if global_counts["route_offset"]
                else 1.0 / math.ceil(len(examples) / max_queries_per_forward)
            )
            congestion_fraction = (
                teacher_loss.congestion_horizon_count / global_counts["congestion"]
                if global_counts["congestion"]
                else 1.0 / math.ceil(len(examples) / max_queries_per_forward)
            )
            rollout_cell = (
                rollout_loss.route_cell
                if rollout_loss is not None
                else teacher_loss.route_cell
            )
            rollout_offset = (
                rollout_loss.route_offset
                if rollout_loss is not None
                else teacher_loss.route_offset
            )
            blended_cell = (
                mixture.teacher * teacher_loss.route_cell
                + mixture.rollout * rollout_cell
            )
            blended_offset = (
                mixture.teacher * teacher_loss.route_offset
                + mixture.rollout * rollout_offset
            )
            objective = (
                weights.route_cell_weight
                * cell_fraction
                * blended_cell
                + weights.route_offset_weight
                * offset_fraction
                * blended_offset
                + weights.congestion_weight
                * congestion_fraction
                * teacher_loss.congestion
            )
            (objective * loss_scale).backward()
            # Catch an overflow immediately, before a pending gradient can be saved.
            planner_gradient_norm_and_finite(
                model, [example.query for example in chunk]
            )
    metrics = accumulator.metrics()
    metrics["self_conditioning"] = {
        "upcoming_optimizer_update": upcoming_optimizer_update,
        "teacher_weight": mixture.teacher,
        "rollout_weight": mixture.rollout,
    }
    return PlannerMicrobatchResult(
        metrics=metrics,
        query_count=len(examples),
        supervised=accumulator.has_supervision(),
    )


@dataclass(slots=True)
class PlannerTrainingState:
    epoch: int = 0
    session_batch_index: int = 0
    optimizer_step: int = 0
    data_microbatches: int = 0
    supervised_microbatches: int = 0
    accumulation_count: int = 0
    best_validation_ade_meters: float | None = None
    best_optimizer_step: int | None = None
    last_validation_optimizer_step: int | None = None
    non_improvements: int = 0
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> PlannerTrainingState:
        if not isinstance(raw, dict) or set(raw) != {
            item.name for item in fields(cls)
        }:
            raise TrainingCompatibilityError(
                "planner checkpoint training state fields do not match"
            )
        try:
            state = cls(**raw)
        except (TypeError, ValueError) as exc:
            raise TrainingCompatibilityError(
                f"invalid planner training state: {exc}"
            ) from exc
        for name in (
            "epoch",
            "session_batch_index",
            "optimizer_step",
            "data_microbatches",
            "supervised_microbatches",
            "accumulation_count",
            "non_improvements",
        ):
            value = getattr(state, name)
            if type(value) is not int or value < 0:
                raise TrainingCompatibilityError(
                    f"planner training state {name} is invalid"
                )
        for name in ("best_validation_ade_meters",):
            value = getattr(state, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TrainingCompatibilityError(
                    f"planner training state {name} is invalid"
                )
        for name in ("best_optimizer_step", "last_validation_optimizer_step"):
            value = getattr(state, name)
            if value is not None and (type(value) is not int or value < 0):
                raise TrainingCompatibilityError(
                    f"planner training state {name} is invalid"
                )
        if type(state.stopped_early) is not bool:
            raise TrainingCompatibilityError("stopped_early must be boolean")
        return state


def _pending_gradients(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _restore_pending_gradients(
    model: nn.Module,
    gradients: Any,
    accumulation_count: int,
) -> None:
    if not isinstance(gradients, Mapping):
        raise TrainingCompatibilityError(
            "planner checkpoint pending gradients are invalid"
        )
    parameters = dict(model.named_parameters())
    unknown = sorted(set(gradients) - set(parameters))
    if unknown:
        raise TrainingCompatibilityError(
            f"planner checkpoint has unknown gradients: {unknown[:8]}"
        )
    for parameter in parameters.values():
        parameter.grad = None
    for name, gradient in gradients.items():
        if not isinstance(gradient, Tensor):
            raise TrainingCompatibilityError(
                f"pending gradient {name} is not a tensor"
            )
        parameter = parameters[name]
        if tuple(gradient.shape) != tuple(parameter.shape):
            raise TrainingCompatibilityError(
                f"pending gradient shape mismatch for {name}"
            )
        if gradient.dtype != parameter.dtype:
            raise TrainingCompatibilityError(
                f"pending gradient dtype mismatch for {name}"
            )
        if not bool(torch.isfinite(gradient).all()):
            raise TrainingCompatibilityError(
                f"pending gradient {name} is non-finite"
            )
        parameter.grad = gradient.to(parameter.device).clone()
    if accumulation_count == 0 and gradients:
        raise TrainingCompatibilityError(
            "checkpoint has gradients with zero accumulation_count"
        )
    if accumulation_count > 0 and not gradients:
        raise TrainingCompatibilityError(
            "checkpoint has pending accumulation without gradients"
        )


def _compatibility_contract(
    *,
    resolved: ResolvedPlannerTrainingConfig,
    optimizer_contract: PlannerOptimizerContract,
    shape_signature: dict[str, Any],
) -> dict[str, Any]:
    config = resolved.training
    return {
        "resolved_config": resolved.to_dict(),
        "model_shape_signature": shape_signature,
        "source_ingestion_report_sha256": (
            resolved.source_ingestion_report_sha256
        ),
        "source_split_manifest_sha256": (
            resolved.source_split_manifest_sha256
        ),
        "planner_ingestion_report_sha256": (
            resolved.planner_ingestion_report_sha256
        ),
        "planner_ingestion_binding_sha256": (
            resolved.planner_ingestion_binding_sha256
        ),
        "planner_split_manifest_sha256": (
            resolved.planner_split_manifest_sha256
        ),
        "planner_dataset_validation_report_hash": (
            resolved.planner_dataset_validation_report_hash
        ),
        "session_replay_overlap_report": (
            resolved.session_replay_overlap_report
        ),
        "split_manifest_sha256": resolved.split_manifest_sha256,
        "split_session_ids": resolved.split_session_ids,
        "dataset_schema_version": resolved.dataset_schema_version,
        "ingestion_report_sha256": resolved.ingestion_report_sha256,
        "world_grid_profile_id": resolved.world_grid_profile_id,
        "world_grid_profile_hash": resolved.world_grid_profile_hash,
        "world_grid_audit_sha256": resolved.world_grid_audit_sha256,
        "observation_provider_id": resolved.observation_provider_id,
        "target_schema_id": resolved.target_schema_id,
        "source_checkpoint_sha256": resolved.source_checkpoint_sha256,
        "planner_transfer_checkpoint_sha256": (
            resolved.planner_transfer_checkpoint_sha256
        ),
        "planner_transfer_components": resolved.planner_transfer_components,
        "source_tree_sha256": resolved.source_tree_sha256,
        "dependency_environment": resolved.dependency_environment,
        "loss_weights": config.loss_config.__dict__
        if hasattr(config.loss_config, "__dict__")
        else _jsonable(config.loss_config),
        "optimizer": optimizer_contract.to_dict(),
        "scheduler": {
            "total_steps": resolved.total_optimizer_steps,
            "warmup_steps": resolved.warmup_steps,
            "min_lr_ratio": config.scheduler_min_lr_ratio,
        },
        "self_conditioning": {
            "teacher_only_through_update": 43,
            "linear_ramp_updates": [44, 86],
            "maximum_rollout_weight": 0.5,
            "position_source": "upcoming_optimizer_update",
        },
        "effective_batch": {
            "batch_size_sessions": config.batch_size,
            "queries_per_session": config.queries_per_session,
            "accumulation_steps": config.accumulation_steps,
            "max_queries_per_forward": config.max_queries_per_forward,
            "uniform_query_sampler_version": (
                resolved.uniform_query_sampler_version
            ),
            "context_length_ticks": resolved.context_length_ticks,
        },
        "validation_query_sample_sha256": (
            resolved.validation_query_sample_sha256
        ),
    }


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_payload(
    *,
    resolved: ResolvedPlannerTrainingConfig,
    split_manifest: dict[str, Any],
    model: ExpertRotationPlannerModel,
    optimizer: torch.optim.Optimizer,
    scheduler: PlannerWarmupCosineScheduler,
    state: PlannerTrainingState,
    epoch_metrics: SelfConditionedMetricAccumulator,
    optimizer_contract: PlannerOptimizerContract,
    shape_signature: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any]:
    compatibility = _compatibility_contract(
        resolved=resolved,
        optimizer_contract=optimizer_contract,
        shape_signature=shape_signature,
    )
    return {
        "checkpoint_schema_version": PLANNER_CHECKPOINT_SCHEMA_VERSION,
        "encoder_state": model.encoder.state_dict(),
        "planner_state": model.planner.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "pending_gradients": _pending_gradients(model),
        "rng_state": _rng_state(),
        "sampler_state": {
            "version": UNIFORM_QUERY_SAMPLER_VERSION,
            "epoch": state.epoch,
            "session_batch_index": state.session_batch_index,
        },
        "metric_accumulators": {
            "epoch": epoch_metrics.state_dict(),
        },
        "training_state": state.to_dict(),
        "resolved_config": resolved.to_dict(),
        "model_shape_signature": shape_signature,
        "split_manifest": split_manifest,
        "split_manifest_sha256": resolved.split_manifest_sha256,
        "source_split_manifest": resolved.source_split_manifest,
        "source_split_manifest_sha256": resolved.source_split_manifest_sha256,
        "planner_split_manifest": resolved.planner_split_manifest,
        "planner_split_manifest_sha256": (
            resolved.planner_split_manifest_sha256
        ),
        "source_ingestion_report_sha256": (
            resolved.source_ingestion_report_sha256
        ),
        "planner_ingestion_report_sha256": (
            resolved.planner_ingestion_report_sha256
        ),
        "planner_ingestion_binding_sha256": (
            resolved.planner_ingestion_binding_sha256
        ),
        "planner_dataset_validation_report_hash": (
            resolved.planner_dataset_validation_report_hash
        ),
        "session_replay_overlap_report": (
            resolved.session_replay_overlap_report
        ),
        "dataset_schema_version": resolved.dataset_schema_version,
        "world_grid_profile": {
            "id": resolved.world_grid_profile_id,
            "hash": resolved.world_grid_profile_hash,
            "audit_sha256": resolved.world_grid_audit_sha256,
            "coordinate_audit_sha256": (
                resolved.world_grid_coordinate_audit_sha256
            ),
        },
        "observation_provider_id": resolved.observation_provider_id,
        "target_schema_id": resolved.target_schema_id,
        "source_checkpoint_sha256": resolved.source_checkpoint_sha256,
        "planner_transfer_checkpoint_sha256": (
            resolved.planner_transfer_checkpoint_sha256
        ),
        "planner_transfer_components": resolved.planner_transfer_components,
        "loss_weights": _jsonable(resolved.training.loss_config),
        "dependency_environment": resolved.dependency_environment,
        "source_tree_sha256": resolved.source_tree_sha256,
        "compatibility": compatibility,
        "metrics_jsonl": (
            metrics_path.read_text(encoding="utf-8")
            if metrics_path.is_file()
            else ""
        ),
    }


def _save_planner_checkpoint(path: Path, **kwargs: Any) -> None:
    _atomic_torch_save(path, _checkpoint_payload(**kwargs))


def _load_planner_checkpoint(
    path: Path,
    *,
    resolved: ResolvedPlannerTrainingConfig,
    model: ExpertRotationPlannerModel,
    optimizer: torch.optim.Optimizer,
    scheduler: PlannerWarmupCosineScheduler,
    epoch_metrics: SelfConditionedMetricAccumulator,
    optimizer_contract: PlannerOptimizerContract,
    shape_signature: dict[str, Any],
    metrics_path: Path,
) -> PlannerTrainingState:
    if not path.is_file():
        raise FileNotFoundError(f"planner resume checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingCompatibilityError(
            f"cannot load planner checkpoint {path}: {exc}"
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_schema_version")
        != PLANNER_CHECKPOINT_SCHEMA_VERSION
    ):
        raise TrainingCompatibilityError("unsupported planner checkpoint schema")
    expected = _compatibility_contract(
        resolved=resolved,
        optimizer_contract=optimizer_contract,
        shape_signature=shape_signature,
    )
    actual = checkpoint.get("compatibility")
    if not isinstance(actual, dict):
        raise TrainingCompatibilityError(
            "planner checkpoint compatibility state is missing"
        )
    if actual != expected:
        keys = sorted(set(actual) | set(expected))
        differing = [key for key in keys if actual.get(key) != expected.get(key)]
        raise TrainingCompatibilityError(
            "planner checkpoint compatibility mismatch: "
            + ", ".join(differing[:16])
        )
    duplicate_contract = {
        "resolved_config": resolved.to_dict(),
        "model_shape_signature": shape_signature,
        "split_manifest_sha256": resolved.split_manifest_sha256,
        "source_split_manifest": resolved.source_split_manifest,
        "source_split_manifest_sha256": resolved.source_split_manifest_sha256,
        "planner_split_manifest": resolved.planner_split_manifest,
        "planner_split_manifest_sha256": (
            resolved.planner_split_manifest_sha256
        ),
        "source_ingestion_report_sha256": (
            resolved.source_ingestion_report_sha256
        ),
        "planner_ingestion_report_sha256": (
            resolved.planner_ingestion_report_sha256
        ),
        "planner_ingestion_binding_sha256": (
            resolved.planner_ingestion_binding_sha256
        ),
        "planner_dataset_validation_report_hash": (
            resolved.planner_dataset_validation_report_hash
        ),
        "session_replay_overlap_report": (
            resolved.session_replay_overlap_report
        ),
        "dataset_schema_version": resolved.dataset_schema_version,
        "observation_provider_id": resolved.observation_provider_id,
        "target_schema_id": resolved.target_schema_id,
        "source_checkpoint_sha256": resolved.source_checkpoint_sha256,
        "planner_transfer_checkpoint_sha256": (
            resolved.planner_transfer_checkpoint_sha256
        ),
        "planner_transfer_components": resolved.planner_transfer_components,
        "loss_weights": _jsonable(resolved.training.loss_config),
        "dependency_environment": resolved.dependency_environment,
        "source_tree_sha256": resolved.source_tree_sha256,
    }
    duplicate_mismatches = [
        name
        for name, expected_value in duplicate_contract.items()
        if checkpoint.get(name) != expected_value
    ]
    if duplicate_mismatches:
        raise TrainingCompatibilityError(
            "planner checkpoint duplicated contract mismatch: "
            + ", ".join(duplicate_mismatches)
        )
    checkpoint_manifest = checkpoint.get("split_manifest")
    if (
        not isinstance(checkpoint_manifest, dict)
        or _fingerprint(checkpoint_manifest)
        != resolved.planner_split_manifest_sha256
    ):
        raise TrainingCompatibilityError(
            "planner checkpoint split manifest content mismatch"
        )
    if checkpoint.get("world_grid_profile") != {
        "id": resolved.world_grid_profile_id,
        "hash": resolved.world_grid_profile_hash,
        "audit_sha256": resolved.world_grid_audit_sha256,
        "coordinate_audit_sha256": (
            resolved.world_grid_coordinate_audit_sha256
        ),
    }:
        raise TrainingCompatibilityError(
            "planner checkpoint world-grid contract mismatch"
        )
    try:
        model.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        model.planner.load_state_dict(checkpoint["planner_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise TrainingCompatibilityError(
            f"planner checkpoint cannot be restored exactly: {exc}"
        ) from exc
    state = PlannerTrainingState.from_dict(checkpoint.get("training_state"))
    if state.optimizer_step != scheduler.step_index:
        raise TrainingCompatibilityError(
            "planner optimizer and scheduler positions are inconsistent"
        )
    if state.optimizer_step > resolved.total_optimizer_steps:
        raise TrainingCompatibilityError(
            "planner checkpoint exceeds the resolved optimizer schedule"
        )
    if state.accumulation_count >= resolved.training.accumulation_steps:
        raise TrainingCompatibilityError(
            "planner checkpoint accumulation position is invalid"
        )
    sampler_state = checkpoint.get("sampler_state")
    if sampler_state != {
        "version": UNIFORM_QUERY_SAMPLER_VERSION,
        "epoch": state.epoch,
        "session_batch_index": state.session_batch_index,
    }:
        raise TrainingCompatibilityError("planner sampler position is inconsistent")
    metrics = checkpoint.get("metric_accumulators")
    if not isinstance(metrics, dict) or set(metrics) != {"epoch"}:
        raise TrainingCompatibilityError(
            "planner checkpoint metric state is incomplete"
        )
    epoch_metrics.load_state_dict(metrics["epoch"])
    _restore_pending_gradients(
        model,
        checkpoint.get("pending_gradients"),
        state.accumulation_count,
    )
    history = checkpoint.get("metrics_jsonl")
    if not isinstance(history, str):
        raise TrainingCompatibilityError("planner metrics history is invalid")
    _atomic_write_text(metrics_path, history)
    try:
        _restore_rng_state(checkpoint["rng_state"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise TrainingCompatibilityError(
            f"planner RNG state cannot be restored exactly: {exc}"
        ) from exc
    return state


def _merge_metrics(
    accumulator: SelfConditionedMetricAccumulator,
    metrics: Mapping[str, Any],
) -> None:
    accumulator.merge_metrics(metrics)


def _fixed_validation_sample(
    repository: _PlannerSessionRepository,
    session_ids: Sequence[str],
    *,
    seed: int,
    queries_per_session: int,
) -> tuple[PlannerQuery, ...]:
    selected: list[PlannerQuery] = []
    for session_id in sorted(session_ids):
        source = repository.get(session_id)
        eligible = eligible_planner_queries(source)
        selected.extend(
            _uniform_sample(
                eligible,
                queries_per_session,
                _stable_seed(
                    seed,
                    "validation",
                    UNIFORM_QUERY_SAMPLER_VERSION,
                    session_id,
                ),
            )
        )
    return tuple(selected)


def _sample_training_batch_queries(
    sources: Mapping[str, PlannerTargetSource],
    session_ids: Sequence[str],
    *,
    seed: int,
    epoch: int,
    session_batch_index: int,
    queries_per_session: int,
) -> tuple[PlannerQuery, ...]:
    universe = tuple(
        query
        for session_id in session_ids
        for query in eligible_planner_queries(sources[session_id])
    )
    return _uniform_sample(
        universe,
        queries_per_session * len(session_ids),
        _stable_seed(
            seed,
            "training-query-batch",
            UNIFORM_QUERY_SAMPLER_VERSION,
            epoch,
            session_batch_index,
            tuple(session_ids),
        ),
    )


ROLLOUT_ADE_METRIC_ID = "mean_per_route_autoregressive_ADE_meters"
ROLLOUT_ADE_METRIC_DEFINITION = (
    "For each query with at least one valid future target, take the mean "
    "Euclidean world-coordinate displacement over its valid subset of the "
    "12 five-second horizons, convert world units to meters, then average "
    "those per-route means equally."
)


class PlannerRolloutADEAccumulator:
    def __init__(self) -> None:
        self.route_ade_world_units_sum = 0.0
        self.valid_route_count = 0
        self.excluded_zero_target_routes = 0
        self.target_waypoint_count = 0
        self.horizon_error_sums = [0.0] * 12
        self.horizon_counts = [0] * 12

    def update(
        self,
        predicted_xy_uu: Tensor,
        target_xy_uu: Tensor,
        route_mask: Tensor,
    ) -> None:
        q = route_mask.shape[0]
        if tuple(predicted_xy_uu.shape) != (q, 12, 2):
            raise ValueError("predicted_xy_uu must have shape [Q,12,2]")
        if tuple(target_xy_uu.shape) != (q, 12, 2):
            raise ValueError("target_xy_uu must have shape [Q,12,2]")
        if tuple(route_mask.shape) != (q, 12) or route_mask.dtype != torch.bool:
            raise ValueError("route_mask must be bool [Q,12]")
        errors = torch.linalg.vector_norm(
            predicted_xy_uu.double() - target_xy_uu.double(), dim=-1
        )
        mask = route_mask.to(device=errors.device)
        counts = mask.sum(dim=-1)
        valid_routes = counts > 0
        if bool(valid_routes.any()):
            route_means = (
                (errors * mask).sum(dim=-1)[valid_routes]
                / counts[valid_routes].double()
            )
            self.route_ade_world_units_sum += float(route_means.sum().item())
            self.valid_route_count += int(valid_routes.sum().item())
        self.excluded_zero_target_routes += int((~valid_routes).sum().item())
        self.target_waypoint_count += int(mask.sum().item())
        for horizon in range(12):
            horizon_mask = mask[:, horizon]
            count = int(horizon_mask.sum().item())
            if count:
                self.horizon_error_sums[horizon] += float(
                    errors[:, horizon].masked_select(horizon_mask).sum().item()
                )
                self.horizon_counts[horizon] += count

    def metrics(self, meters_per_world_unit: float) -> dict[str, Any]:
        if self.valid_route_count:
            world_units = (
                self.route_ade_world_units_sum / self.valid_route_count
            )
        else:
            world_units = math.inf
        per_horizon = []
        for horizon, (total, count) in enumerate(
            zip(self.horizon_error_sums, self.horizon_counts), start=1
        ):
            value = total / count if count else None
            per_horizon.append(
                {
                    "horizon": horizon,
                    "seconds": horizon * 5,
                    "world_units": value,
                    "meters": (
                        value * meters_per_world_unit
                        if value is not None
                        else None
                    ),
                    "valid_waypoint_count": count,
                }
            )
        return {
            "metric": ROLLOUT_ADE_METRIC_ID,
            "metric_definition": ROLLOUT_ADE_METRIC_DEFINITION,
            "world_units": world_units,
            "meters": world_units * meters_per_world_unit,
            "per_horizon": per_horizon,
            "valid_route_count": self.valid_route_count,
            "excluded_zero_target_routes": self.excluded_zero_target_routes,
            "valid_target_waypoint_count": self.target_waypoint_count,
        }


def evaluate_planner_autoregressive(
    *,
    model: ExpertRotationPlannerModel,
    repository: _PlannerSessionRepository,
    validation_queries: Sequence[PlannerQuery],
    validation_session_ids: Sequence[str],
    context_length_ticks: int,
    config: PlannerTrainingConfig,
    spec: PlannerPrecisionSpec,
) -> tuple[dict[str, Any], float]:
    """Evaluate teacher diagnostics and fixed-sample autoregressive ADE."""

    assert config.batch_size is not None
    started = time.perf_counter()
    accumulator = SelfConditionedMetricAccumulator(config.loss_config)
    ade = PlannerRolloutADEAccumulator()
    by_session = {
        session_id: tuple(
            query
            for query in validation_queries
            if query.session_id == session_id
        )
        for session_id in validation_session_ids
    }
    model.eval()
    with torch.no_grad():
        for session_batch in _batches(
            tuple(sorted(validation_session_ids)), config.batch_size
        ):
            # Full sessions are loaded as one batch before query expansion.
            sources = {
                session_id: repository.get(session_id)
                for session_id in session_batch
            }
            queries = tuple(
                query
                for session_id in session_batch
                for query in by_session[session_id]
            )
            examples = prepare_planner_query_examples(
                sources,
                queries,
                context_length_ticks=context_length_ticks,
            )
            for start in range(
                0, len(examples), config.max_queries_per_forward
            ):
                chunk = examples[start : start + config.max_queries_per_forward]
                encoder_batch, observation, targets = (
                    collate_planner_query_examples(chunk, config.policy_config)
                )
                non_blocking = spec.device.type == "cuda"
                encoder_batch = encoder_batch.to(
                    spec.device, non_blocking=non_blocking
                )
                observation = observation.to(
                    spec.device, non_blocking=non_blocking
                )
                targets = targets.to(spec.device, non_blocking=non_blocking)
                with _autocast(spec):
                    planner_batch, encoder_output = model.prepare(
                        encoder_batch, observation
                    )
                    teacher_logits = model.planner(
                        planner_batch,
                        encoder_output,
                        targets.previous_waypoints,
                    )
                    generated = model.planner.greedy(
                        planner_batch, encoder_output
                    )
                teacher_loss = compute_planner_loss(
                    teacher_logits, targets, config.loss_config
                )
                validate_planner_numerics(
                    teacher_logits,
                    teacher_loss,
                    [example.query for example in chunk],
                )
                accumulator.update(
                    teacher_loss,
                    None,
                    SelfConditioningWeights(teacher=1.0, rollout=0.0),
                )
                target_xy, _ = model.planner.waypoints_from_cells(
                    targets.route_cells, targets.route_offsets
                )
                ade.update(
                    generated.waypoints_xy_uu,
                    target_xy,
                    targets.route_mask,
                )
    if not accumulator.has_supervision():
        raise TrainingConfigurationError(
            "fixed validation query sample has no planner supervision"
        )
    rollout_metrics = ade.metrics(
        float(model.planner.world_grid_profile.world_unit_scale.meters_per_world_unit)
    )
    if not math.isfinite(float(rollout_metrics["meters"])):
        raise TrainingConfigurationError(
            "fixed evaluation query sample has no valid future route targets"
        )
    raw_teacher = accumulator.metrics()
    teacher_metrics = {
        "total": (
            config.loss_config.route_cell_weight
            * raw_teacher["teacher_route_cell"]
            + config.loss_config.route_offset_weight
            * raw_teacher["teacher_route_offset"]
            + config.loss_config.congestion_weight
            * raw_teacher["congestion"]
        ),
        "route_cell": raw_teacher["teacher_route_cell"],
        "route_offset": raw_teacher["teacher_route_offset"],
        "congestion": raw_teacher["congestion"],
        "counts": raw_teacher["counts"],
    }
    return {
        "teacher_forced": teacher_metrics,
        "rollout_ade": rollout_metrics,
    }, time.perf_counter() - started


evaluate_planner_teacher_forced = evaluate_planner_autoregressive


def _write_latest(
    run_directory: Path,
    state: PlannerTrainingState,
    checkpoint_name: str,
) -> None:
    _atomic_write_json(
        run_directory / "latest.json",
        {
            "checkpoint": checkpoint_name,
            "epoch": state.epoch,
            "session_batch_index": state.session_batch_index,
            "optimizer_step": state.optimizer_step,
            "data_microbatches": state.data_microbatches,
            "accumulation_count": state.accumulation_count,
        },
    )


def run_planner_training(
    config: PlannerTrainingConfig | str | Path,
    *,
    resume: Literal["last"] | str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic, teacher-forced planner training on its pinned split."""

    setup = _verify_planner_setup(config)
    config = setup.config
    source = setup.source
    transfer = setup.transfer
    profile = setup.profile
    planner = setup.planner
    assert config.seed is not None
    assert config.batch_size is not None
    assert config.world_grid_profile is not None
    dataset_root = Path(config.dataset_root)
    split_ids = planner.split_session_ids
    train_ids = split_ids["train"]
    validation_ids = split_ids["validation"]
    test_ids = split_ids["test"]
    microbatches = setup.microbatches_per_epoch
    total_steps = setup.total_optimizer_steps
    warmup_steps = setup.warmup_steps
    spec = setup.precision_spec
    environment = setup.environment

    run_directory = Path(config.run_directory)
    resolved_path = run_directory / "resolved_config.json"
    metrics_path = run_directory / "metrics.jsonl"
    if resume is None and resolved_path.exists():
        raise TrainingConfigurationError(
            f"run directory already contains planner training: {run_directory}"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    if resume is None:
        # Environment and access proofs appear before the first expensive
        # Parquet-derived validation sample is built.
        _atomic_write_json(run_directory / "environment.json", environment)

    active_ids = (*train_ids, *validation_ids)
    active_paths = {
        session_id: planner.session_paths[session_id]
        for session_id in active_ids
    }
    repository = _PlannerSessionRepository(
        active_paths,
        active_ids,
        test_ids,
        profile,
    )
    repository.attach_data_access_audit(
        run_directory / "data_access.json",
        split_manifest_sha256=planner.manifest_sha256,
    )
    _seed_everything(config.seed)
    model = ExpertRotationPlannerModel(profile, source.encoder)
    apply_planner_component_transfer(model, transfer)
    model = model.to(spec.device)
    optimizer, optimizer_contract = build_planner_adamw_optimizer(model, config)
    scheduler = PlannerWarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=config.scheduler_min_lr_ratio,
    )
    shape_signature = planner_model_shape_signature(model)
    validation_queries = _fixed_validation_sample(
        repository,
        validation_ids,
        seed=config.seed,
        queries_per_session=config.validation_queries_per_session,
    )
    validation_manifest = tuple(
        dataclasses.asdict(query) for query in validation_queries
    )
    resolved = ResolvedPlannerTrainingConfig(
        training=config,
        dataset_root=str(dataset_root.resolve()),
        run_directory=str(Path(config.run_directory).resolve()),
        source_checkpoint_path=source.checkpoint_path,
        source_checkpoint_sha256=source.checkpoint_sha256,
        planner_transfer_checkpoint_path=transfer.checkpoint_path,
        planner_transfer_checkpoint_sha256=transfer.checkpoint_sha256,
        planner_transfer_components=("encoder", "congestion_predictor"),
        source_dataset_schema_version=source.dataset_schema_version,
        source_ingestion_report_sha256=source.ingestion_report_sha256,
        source_split_manifest=source.split_manifest,
        source_split_manifest_sha256=source.split_manifest_sha256,
        source_split_session_ids=source.session_ids_by_partition,
        planner_dataset_schema_version=DATASET_SCHEMA_VERSION,
        planner_ingestion_report_sha256=planner.ingestion_report_sha256,
        planner_ingestion_binding_sha256=planner.ingestion_binding_sha256,
        planner_split_manifest=planner.manifest,
        planner_split_manifest_sha256=planner.manifest_sha256,
        planner_split_session_ids=split_ids,
        planner_dataset_validation_report_hash=(
            planner.dataset_validation_report_hash
        ),
        planner_dataset_binding_mode=planner.binding_mode,
        session_replay_overlap_report=setup.overlap_report,
        dataset_schema_version=DATASET_SCHEMA_VERSION,
        ingestion_report_sha256=planner.ingestion_report_sha256,
        split_manifest_sha256=planner.manifest_sha256,
        split_session_ids=split_ids,
        train_session_count=len(train_ids),
        validation_session_count=len(validation_ids),
        test_session_count=len(test_ids),
        context_length_ticks=source.context_length_ticks,
        microbatches_per_epoch=microbatches,
        total_optimizer_steps=total_steps,
        warmup_steps=warmup_steps,
        device=str(spec.device),
        precision=spec.precision,
        world_grid_profile_id=profile.profile_id,
        world_grid_profile_hash=profile.profile_hash,
        world_grid_audit_sha256=source.world_grid_audit_sha256,
        world_grid_coordinate_audit_sha256=(
            profile.publication_audit.coordinate_audit_sha256
        ),
        observation_provider_id=config.policy_config.canonical_id,
        target_schema_id=PLANNER_TARGET_SCHEMA_ID,
        uniform_query_sampler_version=UNIFORM_QUERY_SAMPLER_VERSION,
        validation_query_sample=validation_manifest,
        validation_query_sample_sha256=_fingerprint(validation_manifest),
        source_tree_sha256=_source_tree_digest(),
        source_revision=_read_source_revision(Path.cwd()),
        source_encoder_revision=source.source_revision,
        dependency_environment=environment,
    )

    if resume is None:
        _atomic_write_json(resolved_path, resolved.to_dict())
        _atomic_write_json(
            run_directory / "split_manifest.json", planner.manifest
        )
        _atomic_write_json(
            run_directory / "source_split_manifest.json",
            source.split_manifest,
        )
        _atomic_write_json(
            run_directory / "planner_split_manifest.json",
            planner.manifest,
        )
        _atomic_write_json(
            run_directory / "session_replay_overlap.json",
            setup.overlap_report,
        )
        _atomic_write_text(metrics_path, "")
    else:
        if not resolved_path.is_file():
            raise TrainingCompatibilityError(
                "resume run is missing resolved_config.json"
            )
        try:
            persisted = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingCompatibilityError(
                f"cannot read persisted planner configuration: {exc}"
            ) from exc
        if persisted != resolved.to_dict():
            raise TrainingCompatibilityError(
                "persisted planner configuration is incompatible with resume"
            )

    state = PlannerTrainingState()
    epoch_metrics = SelfConditionedMetricAccumulator(config.loss_config)
    if resume is None:
        optimizer.zero_grad(set_to_none=True)
    else:
        resume_path = (
            run_directory / "last.pt"
            if str(resume) == "last"
            else Path(resume).resolve()
        )
        state = _load_planner_checkpoint(
            resume_path,
            resolved=resolved,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_metrics=epoch_metrics,
            optimizer_contract=optimizer_contract,
            shape_signature=shape_signature,
            metrics_path=metrics_path,
        )

    checkpoint_arguments = lambda: {
        "resolved": resolved,
        "split_manifest": planner.manifest,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "state": state,
        "epoch_metrics": epoch_metrics,
        "optimizer_contract": optimizer_contract,
        "shape_signature": shape_signature,
        "metrics_path": metrics_path,
    }

    def save_last() -> None:
        _save_planner_checkpoint(
            run_directory / "last.pt", **checkpoint_arguments()
        )
        _write_latest(run_directory, state, "last.pt")

    def validate_and_consider_best(completed_epoch: int | None) -> None:
        metrics, elapsed = evaluate_planner_autoregressive(
            model=model,
            repository=repository,
            validation_queries=validation_queries,
            validation_session_ids=validation_ids,
            context_length_ticks=source.context_length_ticks,
            config=config,
            spec=spec,
        )
        current = float(metrics["rollout_ade"]["meters"])
        improved = (
            state.best_validation_ade_meters is None
            or current
            < state.best_validation_ade_meters
            - config.early_stopping_min_delta
        )
        if improved:
            state.best_validation_ade_meters = current
            state.best_optimizer_step = state.optimizer_step
            state.non_improvements = 0
        else:
            state.non_improvements += 1
            if (
                config.early_stopping_patience is not None
                and state.non_improvements >= config.early_stopping_patience
            ):
                state.stopped_early = True
        _append_jsonl_atomic(
            metrics_path,
            {
                "type": "validation",
                "completed_epoch": completed_epoch,
                "optimizer_step": state.optimizer_step,
                "metrics": metrics,
                "selection_metric": ROLLOUT_ADE_METRIC_ID,
                "selection_value_meters": current,
                "improved": improved,
                "consecutive_non_improvements": state.non_improvements,
                "validation_seconds": elapsed,
                "query_count": len(validation_queries),
            },
        )
        state.last_validation_optimizer_step = state.optimizer_step
        if improved:
            best_path = run_directory / "best.pt"
            _save_planner_checkpoint(
                best_path, **checkpoint_arguments()
            )
            checkpoint_sha256 = _file_sha256(best_path)
            _atomic_write_json(
                run_directory / "best.json",
                {
                    "checkpoint": "best.pt",
                    "checkpoint_sha256": checkpoint_sha256,
                    "optimizer_step": state.optimizer_step,
                    "epoch": state.epoch,
                    "metric": ROLLOUT_ADE_METRIC_ID,
                    "metric_definition": ROLLOUT_ADE_METRIC_DEFINITION,
                    "units": "meters",
                    "value": current,
                    "validation_rollout_ade": metrics["rollout_ade"],
                    "teacher_forced_diagnostics": metrics[
                        "teacher_forced"
                    ],
                },
            )

    save_last()
    try:
        while True:
            reached_steps = (
                config.max_optimizer_steps is not None
                and state.optimizer_step >= config.max_optimizer_steps
            )
            reached_epochs = (
                config.max_epochs is not None and state.epoch >= config.max_epochs
            )
            if reached_steps or reached_epochs or state.stopped_early:
                break
            ordered = _training_session_order(
                train_ids, seed=config.seed, epoch=state.epoch
            )
            session_batches = _batches(ordered, config.batch_size)
            if state.session_batch_index > len(session_batches):
                raise TrainingCompatibilityError(
                    "planner sampler position exceeds deterministic epoch"
                )
            while state.session_batch_index < len(session_batches):
                if (
                    config.max_optimizer_steps is not None
                    and state.optimizer_step >= config.max_optimizer_steps
                ):
                    break
                session_batch = session_batches[state.session_batch_index]
                # This load completes for every session before any query expands.
                sources = {
                    session_id: repository.get(session_id)
                    for session_id in session_batch
                }
                queries = _sample_training_batch_queries(
                    sources,
                    session_batch,
                    seed=config.seed,
                    epoch=state.epoch,
                    session_batch_index=state.session_batch_index,
                    queries_per_session=config.queries_per_session,
                )
                examples = prepare_planner_query_examples(
                    sources,
                    queries,
                    context_length_ticks=source.context_length_ticks,
                )
                model.train()
                result = run_planner_microbatch(
                    model=model,
                    examples=examples,
                    spec=spec,
                    loss_config=config.loss_config,
                    policy_config=config.policy_config,
                    max_queries_per_forward=config.max_queries_per_forward,
                    backward=True,
                    loss_scale=1.0 / config.accumulation_steps,
                    upcoming_optimizer_update=state.optimizer_step + 1,
                )
                _merge_metrics(epoch_metrics, result.metrics)
                step_record: dict[str, Any] | None = None
                if result.supervised:
                    state.accumulation_count += 1
                    state.supervised_microbatches += 1
                    if state.accumulation_count == config.accumulation_steps:
                        pre_clip_norm = planner_gradient_norm_and_finite(
                            model, queries
                        )
                        clipped = pre_clip_norm > config.gradient_clip_norm
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.gradient_clip_norm,
                            error_if_nonfinite=True,
                        )
                        learning_rates_used = scheduler.get_last_lr()
                        optimizer_started = time.perf_counter()
                        optimizer.step()
                        optimizer_seconds = time.perf_counter() - optimizer_started
                        state.optimizer_step += 1
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        state.accumulation_count = 0
                        step_record = {
                            "type": "optimizer_step",
                            "epoch": state.epoch,
                            "session_batch_index": state.session_batch_index,
                            "optimizer_step": state.optimizer_step,
                            "metrics": result.metrics,
                            "query_count": result.query_count,
                            "learning_rates_used": learning_rates_used,
                            "next_learning_rates": scheduler.get_last_lr(),
                            "pre_clip_gradient_norm": pre_clip_norm,
                            "gradient_clipped": clipped,
                            "optimizer_step_seconds": optimizer_seconds,
                        }
                state.data_microbatches += 1
                state.session_batch_index += 1
                if (
                    step_record is not None
                    and state.optimizer_step % config.metrics_every_optimizer_steps
                    == 0
                ):
                    _append_jsonl_atomic(metrics_path, step_record)
                if (
                    step_record is not None
                    and config.checkpoint_every_optimizer_steps is not None
                    and state.optimizer_step
                    % config.checkpoint_every_optimizer_steps
                    == 0
                ):
                    periodic = (
                        f"checkpoint-step-{state.optimizer_step:08d}.pt"
                    )
                    _save_planner_checkpoint(
                        run_directory / periodic, **checkpoint_arguments()
                    )
                if (
                    state.data_microbatches
                    % config.last_checkpoint_every_microbatches
                    == 0
                ):
                    save_last()
            if state.session_batch_index < len(session_batches):
                continue
            completed_epoch = state.epoch
            _append_jsonl_atomic(
                metrics_path,
                {
                    "type": "training_epoch",
                    "completed_epoch": completed_epoch,
                    "optimizer_step": state.optimizer_step,
                    "metrics": epoch_metrics.metrics(),
                    "data_microbatches": state.data_microbatches,
                    "supervised_microbatches": state.supervised_microbatches,
                    "pending_accumulation": state.accumulation_count,
                },
            )
            state.epoch += 1
            state.session_batch_index = 0
            epoch_metrics = SelfConditionedMetricAccumulator(config.loss_config)
            if state.epoch % config.validation_every_epochs == 0:
                validate_and_consider_best(completed_epoch)
            save_last()

        if (
            state.best_validation_ade_meters is None
            or state.last_validation_optimizer_step != state.optimizer_step
        ):
            validate_and_consider_best(
                state.epoch - 1 if state.session_batch_index == 0 else None
            )
        save_last()
    except TrainingNumericalError as exc:
        optimizer.zero_grad(set_to_none=True)
        state.accumulation_count = 0
        try:
            diagnostic = json.loads(str(exc))
        except json.JSONDecodeError:
            diagnostic = {
                "kind": "planner_numerical_failure",
                "message": str(exc)[:512],
            }
        diagnostic.update(
            {
                "epoch": state.epoch,
                "session_batch_index": state.session_batch_index,
                "optimizer_step": state.optimizer_step,
            }
        )
        _atomic_write_json(run_directory / "failure.json", diagnostic)
        raise

    data_access = repository.data_access_report()
    repository._write_data_access_audit()
    if not data_access["zero_test_session_parquet_opens"]:
        raise TrainingCompatibilityError(
            "planner test-session Parquet access was detected"
        )
    trust_partitions = {
        item["trust_partition"] for item in planner.manifest["assignments"]
    }
    provenance_blockers: list[str] = []
    if trust_partitions != {"attested"}:
        provenance_blockers.append(
            "planner split includes unattested dataset sessions"
        )
    if source.source_revision is None:
        provenance_blockers.append(
            "source encoder checkpoint records no source revision"
        )
    summary = {
        "status": "completed",
        "epoch": state.epoch,
        "session_batch_index": state.session_batch_index,
        "optimizer_steps": state.optimizer_step,
        "data_microbatches": state.data_microbatches,
        "supervised_microbatches": state.supervised_microbatches,
        "pending_accumulation": state.accumulation_count,
        "best_validation_ade_meters": state.best_validation_ade_meters,
        "selection_metric": ROLLOUT_ADE_METRIC_ID,
        "best_optimizer_step": state.best_optimizer_step,
        "last_learning_rates": scheduler.get_last_lr(),
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "planner_transfer_checkpoint_sha256": transfer.checkpoint_sha256,
        "planner_transfer_components": [
            "encoder",
            "congestion_predictor",
        ],
        "source_ingestion_report_sha256": source.ingestion_report_sha256,
        "source_split_manifest_sha256": source.split_manifest_sha256,
        "planner_ingestion_report_sha256": planner.ingestion_report_sha256,
        "planner_split_manifest_sha256": planner.manifest_sha256,
        "planner_dataset_validation_report_hash": (
            planner.dataset_validation_report_hash
        ),
        "split_manifest_sha256": planner.manifest_sha256,
        "split_counts": {
            partition: len(split_ids[partition]) for partition in _PARTITIONS
        },
        "context_length_ticks": source.context_length_ticks,
        "test_partition_evaluated": False,
        "test_session_parquet_opened": not data_access[
            "zero_test_session_parquet_opens"
        ],
        "data_access": data_access,
        "session_replay_overlap_report": setup.overlap_report,
        "observation_provider_id": OBSERVATION_PROVIDER_ID,
        "target_schema_id": PLANNER_TARGET_SCHEMA_ID,
        "scientific_provenance_blockers": provenance_blockers,
        "scientifically_final": not provenance_blockers,
    }
    _atomic_write_json(run_directory / "final_summary.json", summary)
    return summary


def smoke_check_planner_training(
    config: PlannerTrainingConfig | str | Path,
) -> dict[str, Any]:
    """Run a full configured BF16 forward/backward without creating a run."""

    setup = _verify_planner_setup(config)
    if setup.precision_spec.precision != "bf16":
        raise TrainingConfigurationError(
            "the full-size planner smoke check requires BF16"
        )
    config = setup.config
    train_id = setup.planner.split_session_ids["train"][0]
    repository = _PlannerSessionRepository(
        {train_id: setup.planner.session_paths[train_id]},
        (train_id,),
        setup.planner.split_session_ids["test"],
        setup.profile,
    )
    source = repository.get(train_id)
    queries = _uniform_sample(
        eligible_planner_queries(source),
        config.max_queries_per_forward,
        _stable_seed(
            config.seed,
            "full-size-self-conditioned-smoke",
            UNIFORM_QUERY_SAMPLER_VERSION,
            train_id,
        ),
    )
    if len(queries) != config.max_queries_per_forward:
        raise TrainingConfigurationError(
            "smoke-check session has too few valid-y0 queries"
        )
    examples = prepare_planner_query_examples(
        {train_id: source},
        queries,
        context_length_ticks=setup.source.context_length_ticks,
    )
    _seed_everything(config.seed)
    model = ExpertRotationPlannerModel(setup.profile, setup.source.encoder)
    apply_planner_component_transfer(model, setup.transfer)
    model = model.to(setup.precision_spec.device).train()
    if setup.precision_spec.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(setup.precision_spec.device)
    result = run_planner_microbatch(
        model=model,
        examples=examples,
        spec=setup.precision_spec,
        loss_config=config.loss_config,
        policy_config=config.policy_config,
        max_queries_per_forward=config.max_queries_per_forward,
        backward=True,
        upcoming_optimizer_update=44,
    )
    gradient_norm = planner_gradient_norm_and_finite(model, queries)
    peak_allocated = None
    peak_reserved = None
    if setup.precision_spec.device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(
            setup.precision_spec.device
        )
        peak_reserved = torch.cuda.max_memory_reserved(
            setup.precision_spec.device
        )
    return {
        "status": "passed",
        "device": str(setup.precision_spec.device),
        "precision": setup.precision_spec.precision,
        "query_count": len(queries),
        "max_queries_per_forward": config.max_queries_per_forward,
        "upcoming_optimizer_update": 44,
        "self_conditioning": result.metrics["self_conditioning"],
        "metrics": result.metrics,
        "gradient_norm": gradient_norm,
        "cuda_peak_memory_allocated_bytes": peak_allocated,
        "cuda_peak_memory_reserved_bytes": peak_reserved,
        "test_session_parquet_open_count": 0,
    }


def _load_locked_best_for_evaluation(
    setup: _VerifiedPlannerSetup,
) -> tuple[ExpertRotationPlannerModel, dict[str, Any], str]:
    """Verify the selection lock before any test repository can be opened."""

    run_directory = Path(setup.config.run_directory)
    best_metadata = _read_json_object(
        run_directory / "best.json", "planner best-checkpoint lock"
    )
    if (
        best_metadata.get("checkpoint") != "best.pt"
        or best_metadata.get("metric") != ROLLOUT_ADE_METRIC_ID
        or best_metadata.get("metric_definition")
        != ROLLOUT_ADE_METRIC_DEFINITION
        or best_metadata.get("units") != "meters"
    ):
        raise TrainingCompatibilityError(
            "best.json is not an autoregressive-ADE selection lock"
        )
    recorded_sha = _validate_sha256(
        best_metadata.get("checkpoint_sha256"),
        "best.json checkpoint_sha256",
    )
    checkpoint_path = run_directory / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"locked planner checkpoint does not exist: {checkpoint_path}"
        )
    if _file_sha256(checkpoint_path) != recorded_sha:
        raise TrainingCompatibilityError(
            "locked best.pt SHA-256 does not match best.json"
        )
    resolved_path = run_directory / "resolved_config.json"
    persisted_resolved = _read_json_object(
        resolved_path, "planner resolved configuration"
    )
    if persisted_resolved.get("training") != setup.config.to_dict():
        raise TrainingCompatibilityError(
            "locked run was not produced by the supplied configuration"
        )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise TrainingCompatibilityError(
            f"cannot load locked planner checkpoint: {exc}"
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_schema_version")
        != PLANNER_CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("resolved_config") != persisted_resolved
        or checkpoint.get("planner_split_manifest_sha256")
        != setup.planner.manifest_sha256
        or checkpoint.get("planner_transfer_checkpoint_sha256")
        != setup.transfer.checkpoint_sha256
    ):
        raise TrainingCompatibilityError(
            "locked planner checkpoint contract does not match this run"
        )
    state = PlannerTrainingState.from_dict(checkpoint.get("training_state"))
    if (
        state.optimizer_step != best_metadata.get("optimizer_step")
        or float(best_metadata.get("value"))
        != float(state.best_validation_ade_meters)
    ):
        raise TrainingCompatibilityError(
            "best.json selection state does not match best.pt"
        )
    model = ExpertRotationPlannerModel(setup.profile, setup.source.encoder)
    apply_planner_component_transfer(model, setup.transfer)
    try:
        model.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        model.planner.load_state_dict(checkpoint["planner_state"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise TrainingCompatibilityError(
            f"locked planner model state cannot be restored exactly: {exc}"
        ) from exc
    return model.to(setup.precision_spec.device), best_metadata, recorded_sha


def evaluate_planner_test(
    config: PlannerTrainingConfig | str | Path,
) -> dict[str, Any]:
    """Idempotently evaluate the already-selected checkpoint on test once."""

    setup = _verify_planner_setup(config)
    model, best_metadata, checkpoint_sha = _load_locked_best_for_evaluation(
        setup
    )
    run_directory = Path(setup.config.run_directory)
    output_path = run_directory / "test_rollout_evaluation.json"
    if output_path.is_file():
        existing = _read_json_object(output_path, "test rollout evaluation")
        if (
            existing.get("schema_version")
            != "planner-test-rollout-evaluation:1.0"
            or existing.get("best_checkpoint_sha256") != checkpoint_sha
            or existing.get("selection_metric") != ROLLOUT_ADE_METRIC_ID
        ):
            raise TrainingCompatibilityError(
                "existing test evaluation does not match the locked best checkpoint"
            )
        return existing

    test_ids = setup.planner.split_session_ids["test"]
    repository = _PlannerSessionRepository(
        {
            session_id: setup.planner.session_paths[session_id]
            for session_id in test_ids
        },
        test_ids,
        (),
        setup.profile,
    )
    queries: list[PlannerQuery] = []
    for session_id in sorted(test_ids):
        source = repository.get(session_id)
        queries.extend(
            _uniform_sample(
                eligible_planner_queries(source),
                32,
                _stable_seed(
                    setup.config.seed,
                    "test",
                    UNIFORM_QUERY_SAMPLER_VERSION,
                    session_id,
                ),
            )
        )
    query_manifest = [dataclasses.asdict(query) for query in queries]
    metrics, elapsed = evaluate_planner_autoregressive(
        model=model,
        repository=repository,
        validation_queries=queries,
        validation_session_ids=test_ids,
        context_length_ticks=setup.source.context_length_ticks,
        config=setup.config,
        spec=setup.precision_spec,
    )
    report = {
        "schema_version": "planner-test-rollout-evaluation:1.0",
        "status": "completed",
        "best_checkpoint": "best.pt",
        "best_checkpoint_sha256": checkpoint_sha,
        "best_optimizer_step": best_metadata["optimizer_step"],
        "selection_metric": ROLLOUT_ADE_METRIC_ID,
        "metric_definition": ROLLOUT_ADE_METRIC_DEFINITION,
        "test_rollout_ade": metrics["rollout_ade"],
        "teacher_forced_diagnostics": metrics["teacher_forced"],
        "test_query_sample": query_manifest,
        "test_query_sample_sha256": _fingerprint(query_manifest),
        "queries_per_session": 32,
        "query_count": len(queries),
        "test_session_count": len(test_ids),
        "uniform_query_sampler_version": UNIFORM_QUERY_SAMPLER_VERSION,
        "evaluation_seconds": elapsed,
        "selection_changed": False,
        "weights_changed": False,
        "planner_transfer_checkpoint_sha256": (
            setup.transfer.checkpoint_sha256
        ),
        "source_checkpoint_sha256": setup.source.checkpoint_sha256,
    }
    _atomic_write_json(output_path, report)
    return report


__all__ = [
    "LEGACY_PLANNER_TRANSFER_SCHEMA_VERSION",
    "PLANNER_CHECKPOINT_SCHEMA_VERSION",
    "ROLLOUT_ADE_METRIC_DEFINITION",
    "ROLLOUT_ADE_METRIC_ID",
    "UNIFORM_QUERY_SAMPLER_VERSION",
    "PlannerMetricAccumulator",
    "PlannerDatasetBindingConfig",
    "PlannerMicrobatchResult",
    "PlannerOptimizerContract",
    "PlannerPrecisionSpec",
    "PlannerQueryExample",
    "PlannerTrainingConfig",
    "PlannerTrainingState",
    "PlannerTransferCheckpoint",
    "PlannerRolloutADEAccumulator",
    "SelfConditionedMetricAccumulator",
    "SelfConditioningWeights",
    "PlannerWarmupCosineScheduler",
    "ResolvedPlannerTrainingConfig",
    "SourceEncoderCheckpoint",
    "WarmupCosineScheduler",
    "build_planner_adamw_optimizer",
    "build_planner_optimizer",
    "apply_planner_component_transfer",
    "collate_planner_query_examples",
    "evaluate_planner_autoregressive",
    "evaluate_planner_test",
    "evaluate_planner_teacher_forced",
    "load_pretrained_encoder_checkpoint",
    "load_planner_transfer_checkpoint",
    "load_source_encoder_checkpoint",
    "planner_gradient_norm_and_finite",
    "planner_model_shape_signature",
    "prepare_planner_query_examples",
    "resolve_planner_precision",
    "resolve_precision",
    "run_planner_microbatch",
    "run_planner_training",
    "self_conditioning_weights",
    "smoke_check_planner_training",
    "verify_planner_training",
    "validate_planner_numerics",
]
