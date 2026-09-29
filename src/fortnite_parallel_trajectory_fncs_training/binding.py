from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from fncs_encoder_training.common import (
    canonical_json,
    file_sha256,
    source_tree_manifest,
    tensor_manifest,
    tensor_state_sha256,
)
from fncs_encoder_training.dataset import load_bound_inventory, load_bound_split
from fortnite_encoder.config import EncoderConfig, RotationHeadConfig
from fortnite_encoder.contracts import EncoderBatch, EncoderOutput
from fortnite_encoder.model import SpatiotemporalEncoder
from fortnite_encoder.rotation import RotationModel
from fortnite_encoder.world_grid import WorldGridProfile
from fortnite_parallel_trajectory import ParallelTrajectoryConfig
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel

from .config import (
    ARCHITECTURE_ID,
    FrozenFNCSTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
)


class ArchitectureSourceChanged(TrainingCompatibilityError):
    def __init__(self, *, expected: str, actual: str, diff: Mapping[str, Any]) -> None:
        super().__init__(
            "parallel trajectory architecture/source scope changed: "
            f"expected={expected}, actual={actual}, diff={canonical_json(diff)}"
        )
        self.expected = expected
        self.actual = actual
        self.diff = dict(diff)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingCompatibilityError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingCompatibilityError(f"{label} must be a JSON object")
    return value


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise TrainingCompatibilityError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise TrainingCompatibilityError(
            f"{label} raw SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def _strict_mapping(raw: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TrainingCompatibilityError(f"{label} must be a mapping")
    missing = sorted(keys - set(raw))
    unexpected = sorted(set(raw) - keys)
    if missing or unexpected:
        raise TrainingCompatibilityError(
            f"{label} keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    return raw


def _tensor_mapping(raw: Any, label: str) -> dict[str, Tensor]:
    if not isinstance(raw, Mapping):
        raise TrainingCompatibilityError(f"{label} must be a tensor mapping")
    output: dict[str, Tensor] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TrainingCompatibilityError(
                f"{label} contains a non-string name or non-tensor value"
            )
        output[name] = value.detach().cpu().contiguous()
    return output


def _dataclass_config(
    cls: type[EncoderConfig] | type[RotationHeadConfig],
    raw: Any,
    label: str,
) -> EncoderConfig | RotationHeadConfig:
    if not isinstance(raw, Mapping):
        raise TrainingCompatibilityError(f"{label} is missing")
    values = dict(raw)
    if cls is RotationHeadConfig and isinstance(values.get("horizons_seconds"), list):
        values["horizons_seconds"] = tuple(values["horizons_seconds"])
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise TrainingCompatibilityError(f"{label} is incompatible: {exc}") from exc


def _strict_load(module: nn.Module, state: Mapping[str, Tensor], label: str) -> None:
    try:
        result = module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise TrainingCompatibilityError(f"strict {label} load failed: {exc}") from exc
    if result.missing_keys or result.unexpected_keys:
        raise TrainingCompatibilityError(
            f"strict {label} load returned missing or unexpected tensors"
        )


def _architecture_difference(
    root: Path, expected_files: Mapping[str, str]
) -> dict[str, Any]:
    actual_files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix())
    }
    expected_names = set(expected_files)
    actual_names = set(actual_files)
    changed = [
        {
            "path": name,
            "expected_sha256": expected_files[name],
            "actual_sha256": actual_files[name],
        }
        for name in sorted(expected_names & actual_names)
        if expected_files[name] != actual_files[name]
    ]
    return {
        "missing_files": sorted(expected_names - actual_names),
        "unexpected_files": sorted(actual_names - expected_names),
        "changed_files": changed,
    }


@dataclass(frozen=True, slots=True)
class EncoderBinding:
    encoder_config: EncoderConfig
    encoder_state: Mapping[str, Tensor]
    report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedSetup:
    config: FrozenFNCSTrainingConfig
    profile: WorldGridProfile
    inventory: Mapping[str, Any]
    split_manifest: Mapping[str, Any]
    split_session_ids: Mapping[str, tuple[str, ...]]
    partition_by_session: Mapping[str, str]
    session_paths: Mapping[str, Path]
    encoder_binding: EncoderBinding
    architecture_source_manifest: Mapping[str, Any]
    training_source_manifest: Mapping[str, Any]
    dataset_bindings: Mapping[str, Any]
    world_grid_bindings: Mapping[str, Any]
    prior_decoder_contract: Mapping[str, Any]

    def checkpoint_bindings(self) -> dict[str, Any]:
        config = self.config
        return {
            "architecture_id": ARCHITECTURE_ID,
            "architecture_config": asdict(ParallelTrajectoryConfig()),
            "architecture_source_digest": self.architecture_source_manifest[
                "source_tree_sha256"
            ],
            "training_source_digest": self.training_source_manifest[
                "source_tree_sha256"
            ],
            "canonical_best_pt_sha256": config.expected_canonical_checkpoint_sha256,
            "execution_encoder_only_best_pt_sha256": (
                config.expected_execution_encoder_checkpoint_sha256
            ),
            "encoder_state_sha256": config.expected_encoder_state_sha256,
            "encoder_source_digest": config.expected_encoder_source_digest,
            "canonical_selected_epoch": config.selected_epoch,
            "canonical_selected_optimizer_step": config.selected_optimizer_step,
            "canonical_validation_future_position_loss": (
                config.selected_validation_future_position_loss
            ),
            "dataset": dict(self.dataset_bindings),
            "world_grid": dict(self.world_grid_bindings),
            "split_manifest_sha256": config.expected_split_manifest_sha256,
            "configuration_raw_sha256": config.configuration_raw_sha256,
            "verified_prior_decoder_training_contract_sha256": (
                config.expected_verified_decoder_training_contract_sha256
            ),
            "test_partition_policy": (
                "denylisted_zero_attempts_zero_opens_no_decoder_test_split"
            ),
        }


def _verify_architecture_source(
    config: FrozenFNCSTrainingConfig, workspace: Path
) -> dict[str, Any]:
    root = workspace / "ml" / "src" / "fortnite_parallel_trajectory"
    manifest = source_tree_manifest(root)
    actual = str(manifest["source_tree_sha256"])
    expected = config.expected_architecture_package_digest
    diff = _architecture_difference(root, config.expected_architecture_files)
    if actual != expected or any(diff.values()):
        raise ArchitectureSourceChanged(expected=expected, actual=actual, diff=diff)
    return manifest


def _verify_prior_decoder_contract(
    config: FrozenFNCSTrainingConfig,
) -> dict[str, Any]:
    _require_file_hash(
        config.verified_decoder_training_contract,
        config.expected_verified_decoder_training_contract_sha256,
        "verified decoder training contract",
    )
    raw = _load_json(
        config.verified_decoder_training_contract,
        "verified decoder training contract",
    )
    required = {
        "architecture_id": ARCHITECTURE_ID,
        "target_contract_version": "parallel-trajectory-targets:1.0",
        "architecture_package_digest": config.expected_architecture_package_digest,
        "training_package_digest": (
            config.expected_verified_decoder_training_source_digest
        ),
        "batch_size": 2,
        "queries_per_session": 16,
        "validation_queries_per_session": 32,
        "accumulation_steps": 1,
        "context_length_ticks": 64,
        "max_queries_per_forward": 4,
        "max_epochs": 20,
        "decoder_learning_rate": 3e-4,
        "congestion_learning_rate": 3e-4,
        "adamw_betas": [0.9, 0.999],
        "adamw_epsilon": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "precision": "bf16",
        "device": "cuda:0",
    }
    changed = {
        name: {"expected": expected, "actual": raw.get(name)}
        for name, expected in required.items()
        if raw.get(name) != expected
    }
    if changed:
        raise TrainingCompatibilityError(
            f"verified decoder training contract changed: {canonical_json(changed)}"
        )
    schedule_total = (
        int(raw.get("scheduler_warmup_updates", 0))
        + int(raw.get("scheduler_plateau_updates", 0))
        + int(raw.get("scheduler_decay_updates", 0))
    )
    proportions = {
        "warmup": raw.get("scheduler_warmup_updates", 0) / schedule_total,
        "constant": raw.get("scheduler_plateau_updates", 0) / schedule_total,
        "linear_decay": raw.get("scheduler_decay_updates", 0) / schedule_total,
    }
    expected_proportions = {
        "warmup": config.scheduler_warmup_fraction,
        "constant": config.scheduler_constant_fraction,
        "linear_decay": config.scheduler_decay_fraction,
    }
    if proportions != expected_proportions:
        raise TrainingCompatibilityError(
            "prior decoder schedule proportions changed: "
            f"actual={proportions}, expected={expected_proportions}"
        )
    return {
        "path": str(config.verified_decoder_training_contract),
        "raw_sha256": config.expected_verified_decoder_training_contract_sha256,
        "source_digest": config.expected_verified_decoder_training_source_digest,
        "preserved_batch_and_sampling_contract": True,
        "preserved_schedule_proportions": proportions,
        "absolute_boundaries_recomputed": {
            "warmup_updates": config.scheduler_warmup_updates,
            "constant_updates": config.scheduler_constant_updates,
            "linear_decay_updates": config.scheduler_decay_updates,
            "total_optimizer_updates": config.max_optimizer_steps,
            "final_lr_ratio": config.scheduler_final_lr_ratio,
        },
        "selection_metric": config.checkpoint_selection_metric,
        "loss_contract": "route_mixture_nll_only",
        "congestion_auxiliary_objective": "none",
    }


def _verify_encoder_binding(config: FrozenFNCSTrainingConfig) -> EncoderBinding:
    canonical_run = config.canonical_encoder_run_directory.resolve()
    for path, label in (
        (config.canonical_checkpoint, "canonical checkpoint"),
        (config.canonical_checkpoint_metadata, "canonical checkpoint metadata"),
        (config.execution_encoder_checkpoint, "execution encoder checkpoint"),
        (config.execution_encoder_metadata, "execution encoder metadata"),
    ):
        if path.parent.resolve() != canonical_run:
            raise TrainingCompatibilityError(
                f"{label} is outside the sole canonical encoder run"
            )
    canonical_hash = _require_file_hash(
        config.canonical_checkpoint,
        config.expected_canonical_checkpoint_sha256,
        "canonical best.pt",
    )
    execution_hash = _require_file_hash(
        config.execution_encoder_checkpoint,
        config.expected_execution_encoder_checkpoint_sha256,
        "execution encoder-only-best.pt",
    )
    _require_file_hash(
        config.canonical_checkpoint_metadata,
        config.expected_canonical_checkpoint_metadata_sha256,
        "canonical best metadata",
    )
    _require_file_hash(
        config.execution_encoder_metadata,
        config.expected_execution_encoder_metadata_sha256,
        "execution encoder metadata",
    )
    best_metadata = _load_json(
        config.canonical_checkpoint_metadata, "canonical best metadata"
    )
    encoder_metadata = _load_json(
        config.execution_encoder_metadata, "execution encoder metadata"
    )
    expected_best_metadata = {
        "checkpoint": "best.pt",
        "checkpoint_raw_sha256": canonical_hash,
        "completed_epoch": config.selected_epoch,
        "metric": "future_position",
        "optimizer_step": config.selected_optimizer_step,
        "value": config.selected_validation_future_position_loss,
    }
    if best_metadata != expected_best_metadata:
        raise TrainingCompatibilityError("canonical best metadata changed")
    if (
        encoder_metadata.get("raw_sha256") != execution_hash
        or encoder_metadata.get("tensor_count") != config.expected_encoder_tensor_count
        or encoder_metadata.get("encoder_state_sha256")
        != config.expected_encoder_state_sha256
        or encoder_metadata.get("optimizer_step") != config.selected_optimizer_step
    ):
        raise TrainingCompatibilityError("execution encoder metadata changed")

    try:
        canonical_raw = torch.load(
            config.canonical_checkpoint, map_location="cpu", weights_only=False
        )
        execution_raw = torch.load(
            config.execution_encoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise TrainingCompatibilityError(f"cannot load encoder artifacts: {exc}") from exc
    canonical = _strict_mapping(
        canonical_raw,
        {
            "checkpoint_schema_version",
            "created_utc",
            "type",
            "model_state",
            "optimizer_state",
            "optimizer_contract",
            "scheduler_state",
            "training_state",
            "epoch_metric_state",
            "sampler_state",
            "rng_state",
            "amp_scaler_state",
            "precision",
            "compatibility",
            "resolved_training_config",
            "initial_parameters",
            "metrics_jsonl",
            "data_access",
            "test_partition_evaluated",
        },
        "canonical best.pt",
    )
    execution = _strict_mapping(
        execution_raw,
        {
            "checkpoint_schema_version",
            "type",
            "encoder_state",
            "encoder_config",
            "feature_contract",
            "source_digest",
            "dataset_bindings",
            "world_grid_bindings",
            "training_state",
            "training_run_provenance",
            "encoder_tensor_manifest",
            "encoder_state_sha256",
        },
        "execution encoder-only-best.pt",
    )
    if canonical["checkpoint_schema_version"] != "fncs-fresh-rotation-training-checkpoint:1.0":
        raise TrainingCompatibilityError("canonical checkpoint schema changed")
    if canonical["type"] != "RotationModelTrainingOnlySupervision":
        raise TrainingCompatibilityError("canonical checkpoint type changed")
    if execution["checkpoint_schema_version"] != "fncs-encoder-only-checkpoint:1.0":
        raise TrainingCompatibilityError("execution checkpoint schema changed")
    if execution["type"] != "SpatiotemporalEncoder":
        raise TrainingCompatibilityError("execution checkpoint type changed")
    if canonical.get("test_partition_evaluated") is not False:
        raise TrainingCompatibilityError(
            "canonical training checkpoint unexpectedly records test evaluation"
        )

    resolved = canonical["resolved_training_config"]
    if not isinstance(resolved, Mapping):
        raise TrainingCompatibilityError("canonical resolved configuration is missing")
    encoder_config = _dataclass_config(
        EncoderConfig, resolved.get("encoder_config"), "canonical encoder config"
    )
    head_config = _dataclass_config(
        RotationHeadConfig, resolved.get("head_config"), "canonical head config"
    )
    assert isinstance(encoder_config, EncoderConfig)
    assert isinstance(head_config, RotationHeadConfig)
    if dict(execution["encoder_config"]) != asdict(encoder_config):
        raise TrainingCompatibilityError(
            "encoder-only architecture configuration differs from best.pt"
        )
    canonical_model_state = _tensor_mapping(
        canonical["model_state"], "canonical model_state"
    )
    verifier = RotationModel(encoder_config, head_config)
    _strict_load(verifier, canonical_model_state, "canonical best.pt")

    execution_state = _tensor_mapping(
        execution["encoder_state"], "execution encoder_state"
    )
    execution_encoder = SpatiotemporalEncoder(encoder_config)
    _strict_load(execution_encoder, execution_state, "encoder-only-best.pt")
    canonical_encoder_state = {
        name.removeprefix("encoder."): tensor
        for name, tensor in canonical_model_state.items()
        if name.startswith("encoder.")
    }
    if set(canonical_encoder_state) != set(execution_state):
        raise TrainingCompatibilityError(
            "encoder tensor names differ between best.pt and encoder-only-best.pt"
        )
    comparisons: list[dict[str, Any]] = []
    for name in sorted(execution_state):
        left = canonical_encoder_state[name]
        right = execution_state[name]
        shape_identical = tuple(left.shape) == tuple(right.shape)
        dtype_identical = left.dtype == right.dtype
        bytes_identical = (
            shape_identical
            and dtype_identical
            and hashlib.sha256(left.view(torch.uint8).numpy().tobytes()).digest()
            == hashlib.sha256(right.view(torch.uint8).numpy().tobytes()).digest()
        )
        comparisons.append(
            {
                "name": name,
                "shape": list(right.shape),
                "dtype": str(right.dtype),
                "name_identical": True,
                "shape_identical": shape_identical,
                "dtype_identical": dtype_identical,
                "bytes_identical": bytes_identical,
                "raw_tensor_sha256": hashlib.sha256(
                    right.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        )
    if not all(
        row["shape_identical"]
        and row["dtype_identical"]
        and row["bytes_identical"]
        for row in comparisons
    ):
        raise TrainingCompatibilityError(
            "an encoder tensor differs between canonical and execution artifacts"
        )
    if len(execution_state) != config.expected_encoder_tensor_count:
        raise TrainingCompatibilityError("encoder tensor count changed")
    state_hash = tensor_state_sha256(execution_state)
    if state_hash != config.expected_encoder_state_sha256:
        raise TrainingCompatibilityError(
            f"encoder tensor-state hash mismatch: {state_hash}"
        )
    if execution["encoder_state_sha256"] != state_hash:
        raise TrainingCompatibilityError(
            "encoder-only internal tensor-state hash is invalid"
        )
    if execution["encoder_tensor_manifest"] != tensor_manifest(execution_state):
        raise TrainingCompatibilityError("encoder-only tensor manifest is invalid")
    training_state = canonical["training_state"]
    if not isinstance(training_state, Mapping):
        raise TrainingCompatibilityError("canonical training state is missing")
    if (
        training_state.get("completed_epochs") != config.selected_epoch
        or training_state.get("optimizer_step") != config.selected_optimizer_step
        or not math.isclose(
            float(training_state.get("best_metric", math.nan)),
            config.selected_validation_future_position_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise TrainingCompatibilityError("canonical selected epoch/step/metric changed")
    compatibility = canonical["compatibility"]
    bindings = (
        compatibility.get("bindings")
        if isinstance(compatibility, Mapping)
        else None
    )
    if not isinstance(bindings, Mapping):
        raise TrainingCompatibilityError("canonical checkpoint bindings are missing")
    if (
        bindings.get("encoder_source_digest") != config.expected_encoder_source_digest
        or execution["source_digest"] != config.expected_encoder_source_digest
    ):
        raise TrainingCompatibilityError("encoder source digest changed")
    if execution["dataset_bindings"] != {
        "ingestion_report_sha256": config.expected_ingestion_report_sha256,
        "dataset_validation_sha256": config.expected_dataset_validation_sha256,
        "split_manifest_sha256": config.expected_split_manifest_sha256,
    }:
        raise TrainingCompatibilityError("encoder-only dataset bindings changed")
    if execution["world_grid_bindings"] != {
        "profile_hash": config.expected_world_grid_profile_hash,
        "publication_audit_sha256": (
            config.expected_world_grid_publication_audit_sha256
        ),
        "validation_audit_sha256": (
            config.expected_world_grid_validation_audit_sha256
        ),
    }:
        raise TrainingCompatibilityError("encoder-only world-grid bindings changed")
    auxiliary_names = sorted(
        name for name in canonical_model_state if not name.startswith("encoder.")
    )
    report = {
        "schema_version": "fncs-frozen-encoder-binding:1.0",
        "canonical_lineage_authority": {
            "path": str(config.canonical_checkpoint),
            "raw_sha256": canonical_hash,
            "sealed_by_full_sha256": True,
            "strict_full_rotation_model_load": True,
            "selected_epoch": config.selected_epoch,
            "selected_optimizer_step": config.selected_optimizer_step,
            "validation_future_position_loss": (
                config.selected_validation_future_position_loss
            ),
        },
        "downstream_execution_artifact": {
            "path": str(config.execution_encoder_checkpoint),
            "raw_sha256": execution_hash,
            "strict_spatiotemporal_encoder_load": True,
        },
        "encoder_source_digest": config.expected_encoder_source_digest,
        "encoder_state_sha256": state_hash,
        "encoder_tensor_count": len(execution_state),
        "every_encoder_tensor_name_shape_dtype_and_byte_identical": True,
        "tensor_comparisons": comparisons,
        "canonical_auxiliary_prediction_head_tensor_count": len(auxiliary_names),
        "canonical_auxiliary_prediction_head_tensor_names": auxiliary_names,
        "auxiliary_prediction_heads_loaded_into_decoder_model": False,
        "execution_uses_encoder_only_best_pt": True,
        "missing_tensors": [],
        "unexpected_tensors": [],
        "architecture_mismatches": [],
        "strict": True,
    }
    return EncoderBinding(
        encoder_config=encoder_config,
        encoder_state={name: tensor.clone() for name, tensor in execution_state.items()},
        report=report,
    )


def _verify_dataset_and_world_grid(
    config: FrozenFNCSTrainingConfig,
) -> tuple[
    WorldGridProfile,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, tuple[str, ...]],
    Mapping[str, str],
    Mapping[str, Path],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    _require_file_hash(
        config.dataset_inventory,
        config.expected_dataset_inventory_sha256,
        "dataset inventory",
    )
    _require_file_hash(
        config.ingestion_report,
        config.expected_ingestion_report_sha256,
        "ingestion report",
    )
    _require_file_hash(
        config.dataset_validation,
        config.expected_dataset_validation_raw_sha256,
        "dataset validation",
    )
    _require_file_hash(
        config.split_manifest,
        config.expected_split_manifest_raw_sha256,
        "encoder split manifest",
    )
    inventory = load_bound_inventory(
        config.dataset_inventory, config.expected_dataset_validation_sha256
    )
    split = load_bound_split(
        config.split_manifest, config.expected_split_manifest_sha256
    )
    if Path(str(inventory.get("dataset_root"))).resolve() != config.dataset_root:
        raise TrainingCompatibilityError("dataset inventory root changed")
    if inventory.get("ingestion_report_sha256") != config.expected_ingestion_report_sha256:
        raise TrainingCompatibilityError("inventory ingestion binding changed")
    if split.get("ingestion_report_sha256") != config.expected_ingestion_report_sha256:
        raise TrainingCompatibilityError("split ingestion binding changed")
    if split.get("dataset_validation_sha256") != config.expected_dataset_validation_sha256:
        raise TrainingCompatibilityError("split dataset-validation binding changed")
    if split.get("seed") != config.seed or split.get("immutable") is not True:
        raise TrainingCompatibilityError("encoder split seed/immutability changed")
    if dict(split.get("counts", {})) != dict(config.expected_split_counts):
        raise TrainingCompatibilityError("encoder split counts changed")
    ordered = split.get("ordered_session_ids")
    if not isinstance(ordered, Mapping):
        raise TrainingCompatibilityError("encoder split ordered IDs are missing")
    split_ids = {
        name: tuple(ordered.get(name, ()))
        for name in ("train", "validation", "test")
    }
    if {name: len(values) for name, values in split_ids.items()} != dict(
        config.expected_split_counts
    ):
        raise TrainingCompatibilityError("ordered encoder split counts changed")
    all_ids = [value for values in split_ids.values() for value in values]
    if len(all_ids) != 1150 or len(set(all_ids)) != 1150:
        raise TrainingCompatibilityError(
            "encoder split IDs are not exactly 1,150 distinct sessions"
        )
    assignments = split.get("assignments")
    sessions = inventory.get("sessions")
    if not isinstance(assignments, list) or not isinstance(sessions, list):
        raise TrainingCompatibilityError("dataset session metadata is missing")
    by_id = {
        item.get("game_session_id"): item
        for item in sessions
        if isinstance(item, Mapping)
    }
    partition_by_session: dict[str, str] = {}
    session_paths: dict[str, Path] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise TrainingCompatibilityError("split assignment is malformed")
        session_id = assignment.get("game_session_id")
        partition = assignment.get("partition")
        item = by_id.get(session_id)
        if (
            not isinstance(session_id, str)
            or partition not in {"train", "validation", "test"}
            or not isinstance(item, Mapping)
        ):
            raise TrainingCompatibilityError("split assignment is unbound")
        if assignment.get("source_replay_sha256") != item.get("source_replay_sha256"):
            raise TrainingCompatibilityError(
                f"split replay hash changed for {session_id}"
            )
        relative = item.get("directory")
        if not isinstance(relative, str) or not relative:
            raise TrainingCompatibilityError(f"dataset directory missing for {session_id}")
        path = (config.dataset_root / relative).resolve()
        if config.dataset_root not in path.parents:
            raise TrainingCompatibilityError(
                f"dataset session escapes the bound root: {session_id}"
            )
        partition_by_session[session_id] = str(partition)
        session_paths[session_id] = path
    if set(session_paths) != set(all_ids):
        raise TrainingCompatibilityError("inventory paths differ from split IDs")

    _require_file_hash(
        config.world_grid_profile,
        config.expected_world_grid_profile_raw_sha256,
        "world-grid profile",
    )
    profile = WorldGridProfile.load(config.world_grid_profile)
    if profile.profile_hash != config.expected_world_grid_profile_hash:
        raise TrainingCompatibilityError("world-grid profile hash changed")
    _require_file_hash(
        config.world_grid_publication_audit,
        config.expected_world_grid_publication_audit_raw_sha256,
        "world-grid publication audit",
    )
    publication = _load_json(
        config.world_grid_publication_audit, "world-grid publication audit"
    )
    if publication.get("audit_hash") != config.expected_world_grid_publication_audit_sha256:
        raise TrainingCompatibilityError("world-grid publication payload changed")
    _require_file_hash(
        config.world_grid_compatibility_audit,
        config.expected_world_grid_compatibility_audit_raw_sha256,
        "world-grid compatibility audit",
    )
    compatibility = _load_json(
        config.world_grid_compatibility_audit, "world-grid compatibility audit"
    )
    required_compatibility = {
        "status": "compatible",
        "accepted_session_count": 1150,
        "profile_hash": config.expected_world_grid_profile_hash,
        "profile_raw_sha256": config.expected_world_grid_profile_raw_sha256,
        "publication_audit_sha256": (
            config.expected_world_grid_publication_audit_sha256
        ),
        "publication_audit_raw_sha256": (
            config.expected_world_grid_publication_audit_raw_sha256
        ),
        "coordinate_audit_sha256": (
            config.expected_world_grid_coordinate_audit_sha256
        ),
        "world_grid_validation_audit_sha256": (
            config.expected_world_grid_validation_audit_sha256
        ),
        "target_policy": "out-of-bounds targets are masked without clamping",
    }
    if any(compatibility.get(name) != value for name, value in required_compatibility.items()):
        raise TrainingCompatibilityError("world-grid compatibility contract changed")
    dataset_bindings = {
        "root": str(config.dataset_root),
        "dataset_schema_version": inventory.get("dataset_schema_version"),
        "inventory_raw_sha256": config.expected_dataset_inventory_sha256,
        "ingestion_report_raw_sha256": config.expected_ingestion_report_sha256,
        "dataset_validation_raw_sha256": (
            config.expected_dataset_validation_raw_sha256
        ),
        "dataset_validation_sha256": config.expected_dataset_validation_sha256,
        "split_manifest_raw_sha256": config.expected_split_manifest_raw_sha256,
        "split_manifest_sha256": config.expected_split_manifest_sha256,
        "split_counts": dict(config.expected_split_counts),
        "seed": config.seed,
        "all_sessions_data_status": "accepted_with_warnings",
        "all_sessions_provenance_status": "incomplete_unattested",
    }
    world_bindings = {
        "profile_id": profile.profile_id,
        "profile_hash": profile.profile_hash,
        "profile_raw_sha256": config.expected_world_grid_profile_raw_sha256,
        "publication_audit_sha256": (
            config.expected_world_grid_publication_audit_sha256
        ),
        "publication_audit_raw_sha256": (
            config.expected_world_grid_publication_audit_raw_sha256
        ),
        "compatibility_audit_raw_sha256": (
            config.expected_world_grid_compatibility_audit_raw_sha256
        ),
        "coordinate_audit_sha256": (
            config.expected_world_grid_coordinate_audit_sha256
        ),
        "validation_audit_sha256": (
            config.expected_world_grid_validation_audit_sha256
        ),
        "out_of_envelope_target_policy": "mask_without_clamping",
    }
    return (
        profile,
        inventory,
        split,
        split_ids,
        partition_by_session,
        session_paths,
        dataset_bindings,
        world_bindings,
    )


def verify_setup(
    config_or_path: FrozenFNCSTrainingConfig | str | Path,
) -> VerifiedSetup:
    config = (
        config_or_path
        if isinstance(config_or_path, FrozenFNCSTrainingConfig)
        else FrozenFNCSTrainingConfig.from_json(config_or_path)
    )
    workspace = Path(__file__).resolve().parents[3]
    architecture = _verify_architecture_source(config, workspace)
    training = source_tree_manifest(Path(__file__).resolve().parent)
    if training["source_tree_sha256"] != config.expected_training_source_digest:
        raise TrainingCompatibilityError(
            "FNCS decoder training source digest changed: "
            f"expected={config.expected_training_source_digest}, "
            f"actual={training['source_tree_sha256']}"
        )
    encoder_source = source_tree_manifest(workspace / "ml" / "src" / "fortnite_encoder")
    if encoder_source["source_tree_sha256"] != config.expected_encoder_source_digest:
        raise TrainingCompatibilityError(
            "immutable encoder source digest changed: "
            f"expected={config.expected_encoder_source_digest}, "
            f"actual={encoder_source['source_tree_sha256']}"
        )
    prior = _verify_prior_decoder_contract(config)
    encoder_binding = _verify_encoder_binding(config)
    (
        profile,
        inventory,
        split,
        split_ids,
        partition_by_session,
        session_paths,
        dataset_bindings,
        world_bindings,
    ) = _verify_dataset_and_world_grid(config)
    if encoder_binding.report["encoder_source_digest"] != encoder_source[
        "source_tree_sha256"
    ]:
        raise TrainingCompatibilityError("encoder artifact/source binding diverged")
    return VerifiedSetup(
        config=config,
        profile=profile,
        inventory=inventory,
        split_manifest=split,
        split_session_ids=split_ids,
        partition_by_session=partition_by_session,
        session_paths=session_paths,
        encoder_binding=encoder_binding,
        architecture_source_manifest=architecture,
        training_source_manifest=training,
        dataset_bindings=dataset_bindings,
        world_grid_bindings=world_bindings,
        prior_decoder_contract=prior,
    )


class FrozenEncoder(nn.Module):
    """Execution-only wrapper that cannot be switched out of evaluation mode."""

    def __init__(self, encoder: SpatiotemporalEncoder) -> None:
        super().__init__()
        if not isinstance(encoder, SpatiotemporalEncoder):
            raise TypeError("encoder must be a SpatiotemporalEncoder")
        self.module = encoder
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.train(False)

    @property
    def config(self) -> EncoderConfig:
        return self.module.config

    def train(self, mode: bool = True) -> "FrozenEncoder":
        del mode
        nn.Module.train(self, False)
        self.training = False
        self.module.eval()
        return self

    def forward(self, batch: EncoderBatch) -> EncoderOutput:
        if self.training or self.module.training:
            raise TrainingCompatibilityError("frozen encoder left evaluation mode")
        if any(parameter.requires_grad for parameter in self.module.parameters()):
            raise TrainingCompatibilityError("frozen encoder parameter became trainable")
        with torch.no_grad():
            return self.module(batch)

    def canonical_state(self) -> dict[str, Tensor]:
        return {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.module.state_dict().items()
        }


def downstream_state(model: ParallelTrajectoryModel) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for owner in ("congestion_predictor", "congestion_tokenizer", "decoder"):
        module = getattr(model, owner)
        for name, tensor in module.state_dict().items():
            state[f"{owner}.{name}"] = tensor.detach().cpu().contiguous()
    return state


def downstream_parameter_state(model: ParallelTrajectoryModel) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for owner in ("congestion_predictor", "congestion_tokenizer", "decoder"):
        module = getattr(model, owner)
        for name, parameter in module.named_parameters():
            state[f"{owner}.{name}"] = parameter.detach().cpu().contiguous()
    return state


def load_downstream_state(
    model: ParallelTrajectoryModel, raw: Mapping[str, Tensor]
) -> None:
    if not isinstance(raw, Mapping):
        raise TrainingCompatibilityError("downstream checkpoint state must be a mapping")
    expected = downstream_state(model)
    if set(raw) != set(expected):
        raise TrainingCompatibilityError(
            "downstream checkpoint tensor names changed: "
            f"missing={sorted(set(expected) - set(raw))}, "
            f"unexpected={sorted(set(raw) - set(expected))}"
        )
    for owner in ("congestion_predictor", "congestion_tokenizer", "decoder"):
        prefix = f"{owner}."
        component = {
            name.removeprefix(prefix): value
            for name, value in raw.items()
            if name.startswith(prefix)
        }
        _strict_load(getattr(model, owner), component, f"{owner} downstream checkpoint")


def build_frozen_model(
    setup: VerifiedSetup,
    *,
    device: torch.device | str,
    initialization_seed: int | None = None,
) -> tuple[ParallelTrajectoryModel, dict[str, Any]]:
    encoder = SpatiotemporalEncoder(setup.encoder_binding.encoder_config)
    _strict_load(
        encoder,
        setup.encoder_binding.encoder_state,
        "execution encoder-only-best.pt",
    )
    frozen = FrozenEncoder(encoder)
    seed = (
        setup.config.initialization_seed
        if initialization_seed is None
        else initialization_seed
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ParallelTrajectoryModel(setup.profile, frozen)
    model.to(device)
    model.train()
    if model.encoder.training or frozen.module.training:
        raise TrainingCompatibilityError("frozen encoder is not in evaluation mode")
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise TrainingCompatibilityError("encoder parameter is trainable")
    downstream = downstream_parameter_state(model)
    report = {
        "schema_version": "fncs-frozen-decoder-initial-parameters:1.0",
        "initialization_seed": seed,
        "fresh_random_initialization": True,
        "historical_downstream_weights_loaded": False,
        "encoder_checkpoint_loaded": str(setup.config.execution_encoder_checkpoint),
        "encoder_auxiliary_heads_present": False,
        "downstream_parameter_tensor_count": len(downstream),
        "downstream_scalar_parameter_count": sum(
            tensor.numel() for tensor in downstream.values()
        ),
        "initial_downstream_parameter_sha256": tensor_state_sha256(downstream),
        "parameters": tensor_manifest(downstream),
        "trainable_components": [
            "congestion_predictor",
            "congestion_tokenizer",
            "parallel_trajectory_decoder_v1",
            "decoder_output_projections",
        ],
        "forbidden_transfer_sources": [
            "historical_planner",
            "early_zone_decoder",
            "phase_weighted_run",
            "previous_parallel_trajectory_decoder",
        ],
    }
    return model, report


def encoder_snapshot(model: ParallelTrajectoryModel) -> dict[str, Any]:
    if not isinstance(model.encoder, FrozenEncoder):
        raise TrainingCompatibilityError("model encoder is not the immutable wrapper")
    state = model.encoder.canonical_state()
    gradients = {
        name: parameter.grad
        for name, parameter in model.encoder.module.named_parameters()
        if parameter.grad is not None
    }
    nonzero = [
        name
        for name, gradient in gradients.items()
        if bool(torch.count_nonzero(gradient).item())
    ]
    return {
        "encoder_state_sha256": tensor_state_sha256(state),
        "tensor_count": len(state),
        "tensors": tensor_manifest(state),
        "requires_grad_parameter_count": sum(
            int(parameter.requires_grad)
            for parameter in model.encoder.module.parameters()
        ),
        "parameters_with_gradient": len(gradients),
        "parameters_with_nonzero_gradient": nonzero,
        "wrapper_training": model.encoder.training,
        "encoder_training": model.encoder.module.training,
    }


def assert_frozen_encoder(
    model: ParallelTrajectoryModel,
    setup: VerifiedSetup,
    *,
    stage: str,
) -> dict[str, Any]:
    snapshot = encoder_snapshot(model)
    failures: list[str] = []
    if snapshot["encoder_state_sha256"] != setup.config.expected_encoder_state_sha256:
        failures.append("tensor_state_hash_changed")
    if snapshot["tensor_count"] != setup.config.expected_encoder_tensor_count:
        failures.append("tensor_count_changed")
    if snapshot["requires_grad_parameter_count"] != 0:
        failures.append("requires_grad_enabled")
    if snapshot["parameters_with_nonzero_gradient"]:
        failures.append("nonzero_encoder_gradient")
    if snapshot["wrapper_training"] or snapshot["encoder_training"]:
        failures.append("encoder_not_in_eval_mode")
    if failures:
        raise TrainingCompatibilityError(
            f"frozen encoder invariant failed at {stage}: {failures}"
        )
    return {
        "stage": stage,
        **snapshot,
        "matches_canonical": True,
    }


__all__ = [
    "ArchitectureSourceChanged",
    "EncoderBinding",
    "FrozenEncoder",
    "VerifiedSetup",
    "assert_frozen_encoder",
    "build_frozen_model",
    "downstream_parameter_state",
    "downstream_state",
    "encoder_snapshot",
    "load_downstream_state",
    "verify_setup",
]
