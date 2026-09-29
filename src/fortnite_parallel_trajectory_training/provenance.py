from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from fortnite_encoder.config import EncoderConfig
from fortnite_encoder.model import SpatiotemporalEncoder
from fortnite_encoder.planner import CongestionPredictor, CongestionTokenizer
from fortnite_encoder.planner_training import _verify_planner_setup
from fortnite_encoder.training import _fingerprint
from fortnite_encoder.world_grid import WorldGridProfile
from fortnite_parallel_trajectory.config import ParallelTrajectoryConfig
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel

from .config import (
    ARCHITECTURE_ID,
    ParallelTrajectoryTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_source_digest(root: str | Path) -> str:
    source = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py"), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _tensor_bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def tensor_manifest(state: Mapping[str, Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_tensor_sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
        }
        for name, value in sorted(state.items())
    ]


def tensor_state_digest(state: Mapping[str, Tensor]) -> str:
    return hashlib.sha256(
        canonical_json(tensor_manifest(state)).encode("utf-8")
    ).hexdigest()


def module_state_digest(module: nn.Module) -> str:
    return tensor_state_digest(module.state_dict())


def _state_mapping(raw: Any, label: str) -> dict[str, Tensor]:
    if not isinstance(raw, Mapping):
        raise TrainingCompatibilityError(f"{label} state is missing")
    if any(not isinstance(name, str) for name in raw):
        raise TrainingCompatibilityError(f"{label} state contains a non-string key")
    if any(not isinstance(value, Tensor) for value in raw.values()):
        raise TrainingCompatibilityError(f"{label} state contains a non-tensor value")
    return {name: value.detach().cpu().clone() for name, value in raw.items()}


def _validated_state(
    module: nn.Module, raw: Mapping[str, Tensor], label: str
) -> dict[str, Tensor]:
    expected = module.state_dict()
    missing = sorted(set(expected) - set(raw))
    unexpected = sorted(set(raw) - set(expected))
    if missing or unexpected:
        raise TrainingCompatibilityError(
            f"{label} names mismatch: missing={missing}, unexpected={unexpected}"
        )
    result: dict[str, Tensor] = {}
    for name, expected_value in expected.items():
        actual = raw[name]
        if tuple(actual.shape) != tuple(expected_value.shape):
            raise TrainingCompatibilityError(f"{label} shape mismatch for {name}")
        if actual.dtype != expected_value.dtype:
            raise TrainingCompatibilityError(f"{label} dtype mismatch for {name}")
        if actual.is_floating_point() and not bool(torch.isfinite(actual).all()):
            raise TrainingCompatibilityError(f"{label} tensor is nonfinite: {name}")
        result[name] = actual.detach().cpu().clone()
    return result


@dataclass(frozen=True, slots=True)
class TransferBundle:
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    encoder_config: EncoderConfig
    encoder_state: Mapping[str, Tensor]
    congestion_predictor_state: Mapping[str, Tensor]
    congestion_tokenizer_state: Mapping[str, Tensor]
    transferred_tensor_manifest: Mapping[str, list[dict[str, Any]]]
    transferred_tensor_manifest_sha256: str
    transferred_tensor_count: int
    source_revision: str | None
    ignored_checkpoint_sections: tuple[str, ...]
    ignored_planner_prefixes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransferProof:
    strict: bool
    transferred_components: tuple[str, ...]
    transferred_tensor_count: int
    transferred_tensor_manifest_sha256: str
    component_state_sha256: Mapping[str, str]
    decoder_before_sha256: str
    decoder_after_sha256: str
    decoder_unchanged: bool
    missing_tensors: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    dtype_mismatches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class VerifiedTrainingSetup:
    config: ParallelTrajectoryTrainingConfig
    profile: WorldGridProfile
    split_manifest: Mapping[str, Any]
    split_manifest_sha256: str
    split_session_ids: Mapping[str, tuple[str, ...]]
    ancestral_split_session_ids: Mapping[str, tuple[str, ...]]
    session_paths: Mapping[str, Path]
    transfer: TransferBundle
    source_checkpoint_metadata: Mapping[str, Any]
    source_revision: str | None
    architecture_config: Mapping[str, Any]
    architecture_package_digest: str
    training_package_digest: str
    lineage_report_sha256: str
    dataset_identity: Mapping[str, Any]
    world_grid_binding: Mapping[str, Any]

    def checkpoint_bindings(self) -> dict[str, Any]:
        return {
            "architecture_id": ARCHITECTURE_ID,
            "architecture_config": dict(self.architecture_config),
            "architecture_package_digest": self.architecture_package_digest,
            "training_package_digest": self.training_package_digest,
            "encoder_lineage_classification": self.config.lineage_classification,
            "encoder_lineage_report_sha256": self.lineage_report_sha256,
            "legacy_encoder_digest": self.config.legacy_encoder_digest,
            "current_encoder_digest": self.config.current_encoder_digest,
            "source_checkpoint_sha256": self.transfer.source_checkpoint_sha256,
            "transferred_tensor_manifest_sha256": (
                self.transfer.transferred_tensor_manifest_sha256
            ),
            "transferred_tensor_count": self.transfer.transferred_tensor_count,
            "encoder_config": dataclasses.asdict(self.transfer.encoder_config),
            "congestion_config": congestion_configuration(),
            "dataset_identity": dict(self.dataset_identity),
            "split_manifest_sha256": self.split_manifest_sha256,
            "split_session_ids": {
                name: list(values) for name, values in self.split_session_ids.items()
            },
            "world_grid_binding": dict(self.world_grid_binding),
            "target_contract_version": self.config.target_contract_version,
            "source_revision": self.source_revision,
        }


def congestion_configuration() -> dict[str, Any]:
    from fortnite_encoder import planner

    return {
        "d_model": planner.D_MODEL,
        "n_heads": planner.N_HEADS,
        "ffn_dim": planner.FFN_DIM,
        "route_steps": planner.ROUTE_STEPS,
        "grid_rows": planner.GRID_ROWS,
        "grid_columns": planner.GRID_COLUMNS,
        "num_regular_cells": planner.NUM_REGULAR_CELLS,
        "zone_vector_width": planner.ZONE_VECTOR_WIDTH,
        "dropout": planner.DROPOUT,
        "token_count": planner.ROUTE_STEPS * planner.NUM_REGULAR_CELLS,
    }


def load_transfer_bundle(
    config: ParallelTrajectoryTrainingConfig,
) -> tuple[TransferBundle, dict[str, Any]]:
    source = Path(config.source_checkpoint).resolve()
    forbidden = Path(config.forbidden_phase_weighted_checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source}")
    digest = file_sha256(source)
    if digest == config.forbidden_phase_weighted_checkpoint_sha256:
        raise TrainingCompatibilityError(
            "phase-weighted checkpoint is explicitly forbidden as ancestry"
        )
    if source == forbidden:
        raise TrainingCompatibilityError(
            "phase-weighted checkpoint path is explicitly forbidden as ancestry"
        )
    if digest != config.expected_source_checkpoint_sha256:
        raise TrainingCompatibilityError("source checkpoint SHA-256 mismatch")
    if forbidden.is_file() and file_sha256(forbidden) != config.forbidden_phase_weighted_checkpoint_sha256:
        raise TrainingCompatibilityError("forbidden checkpoint pin changed")
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingCompatibilityError(f"cannot load source checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_schema_version") != "3.0":
        raise TrainingCompatibilityError("source checkpoint must be planner schema 3.0")
    if checkpoint.get("source_tree_sha256") != config.legacy_encoder_digest:
        raise TrainingCompatibilityError("source checkpoint encoder digest mismatch")
    encoder_config = EncoderConfig()
    encoder = SpatiotemporalEncoder(encoder_config)
    predictor = CongestionPredictor()
    tokenizer = CongestionTokenizer()
    encoder_state = _validated_state(
        encoder, _state_mapping(checkpoint.get("encoder_state"), "encoder"), "encoder"
    )
    planner_state = _state_mapping(checkpoint.get("planner_state"), "planner")
    allowed_prefixes = (
        "congestion_predictor.",
        "congestion_tokenizer.",
        "route_decoder.",
    )
    unknown = sorted(
        name for name in planner_state if not name.startswith(allowed_prefixes)
    )
    if unknown:
        raise TrainingCompatibilityError(
            f"source planner state has unknown prefixes: {unknown[:8]}"
        )

    def extract(prefix: str) -> dict[str, Tensor]:
        return {
            name[len(prefix) :]: value
            for name, value in planner_state.items()
            if name.startswith(prefix)
        }

    predictor_state = _validated_state(
        predictor, extract("congestion_predictor."), "congestion predictor"
    )
    tokenizer_state = _validated_state(
        tokenizer, extract("congestion_tokenizer."), "congestion tokenizer"
    )
    manifest = {
        "encoder": tensor_manifest(encoder_state),
        "congestion_predictor": tensor_manifest(predictor_state),
        "congestion_tokenizer": tensor_manifest(tokenizer_state),
    }
    manifest_hash = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    count = sum(len(rows) for rows in manifest.values())
    if count != config.expected_transferred_tensor_count:
        raise TrainingCompatibilityError("transferred tensor count mismatch")
    if manifest_hash != config.expected_transferred_tensor_manifest_sha256:
        raise TrainingCompatibilityError("transferred tensor manifest hash mismatch")
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise TrainingCompatibilityError("source checkpoint resolved config is missing")
    source_revision = resolved.get("source_revision")
    if source_revision is not None and not isinstance(source_revision, str):
        raise TrainingCompatibilityError("source revision is malformed")
    ignored_sections = tuple(
        name
        for name in (
            "optimizer_state",
            "scheduler_state",
            "rng_state",
            "sampler_state",
            "pending_gradients",
            "metric_accumulators",
            "training_state",
        )
        if name in checkpoint
    )
    metadata = {
        name: checkpoint.get(name)
        for name in (
            "checkpoint_schema_version",
            "source_tree_sha256",
            "source_checkpoint_sha256",
            "planner_split_manifest_sha256",
            "planner_ingestion_report_sha256",
            "planner_ingestion_binding_sha256",
            "planner_dataset_validation_report_hash",
            "source_split_manifest_sha256",
            "source_ingestion_report_sha256",
            "world_grid_profile",
            "dataset_schema_version",
            "observation_provider_id",
            "target_schema_id",
        )
    }
    return (
        TransferBundle(
            source_checkpoint_path=str(source),
            source_checkpoint_sha256=digest,
            encoder_config=encoder_config,
            encoder_state=encoder_state,
            congestion_predictor_state=predictor_state,
            congestion_tokenizer_state=tokenizer_state,
            transferred_tensor_manifest=manifest,
            transferred_tensor_manifest_sha256=manifest_hash,
            transferred_tensor_count=count,
            source_revision=source_revision,
            ignored_checkpoint_sections=ignored_sections,
            ignored_planner_prefixes=("route_decoder",),
        ),
        metadata,
    )


def apply_transfer(
    model: ParallelTrajectoryModel, transfer: TransferBundle
) -> TransferProof:
    if not isinstance(model, ParallelTrajectoryModel):
        raise TypeError("model must be ParallelTrajectoryModel")
    before = module_state_digest(model.decoder)
    try:
        encoder_result = model.encoder.load_state_dict(transfer.encoder_state, strict=True)
        predictor_result = model.congestion_predictor.load_state_dict(
            transfer.congestion_predictor_state, strict=True
        )
        tokenizer_result = model.congestion_tokenizer.load_state_dict(
            transfer.congestion_tokenizer_state, strict=True
        )
    except RuntimeError as exc:
        raise TrainingCompatibilityError(f"strict component transfer failed: {exc}") from exc
    results = (encoder_result, predictor_result, tokenizer_result)
    if any(result.missing_keys or result.unexpected_keys for result in results):
        raise TrainingCompatibilityError("strict transfer returned incompatible keys")
    after = module_state_digest(model.decoder)
    if before != after:
        raise TrainingCompatibilityError("scratch parallel decoder changed during transfer")
    actual = {
        "encoder": module_state_digest(model.encoder),
        "congestion_predictor": module_state_digest(model.congestion_predictor),
        "congestion_tokenizer": module_state_digest(model.congestion_tokenizer),
    }
    expected = {
        "encoder": tensor_state_digest(transfer.encoder_state),
        "congestion_predictor": tensor_state_digest(
            transfer.congestion_predictor_state
        ),
        "congestion_tokenizer": tensor_state_digest(
            transfer.congestion_tokenizer_state
        ),
    }
    if actual != expected:
        raise TrainingCompatibilityError("transferred tensor contents changed")
    return TransferProof(
        strict=True,
        transferred_components=(
            "SpatiotemporalEncoder",
            "CongestionPredictor",
            "CongestionTokenizer",
        ),
        transferred_tensor_count=transfer.transferred_tensor_count,
        transferred_tensor_manifest_sha256=transfer.transferred_tensor_manifest_sha256,
        component_state_sha256=actual,
        decoder_before_sha256=before,
        decoder_after_sha256=after,
        decoder_unchanged=True,
        missing_tensors=(),
        unexpected_tensors=(),
        shape_mismatches=(),
        dtype_mismatches=(),
    )


def initialize_model(
    setup: VerifiedTrainingSetup,
    *,
    device: str | torch.device | None = None,
    seed: int | None = None,
) -> tuple[ParallelTrajectoryModel, TransferProof]:
    torch.manual_seed(setup.config.seed if seed is None else seed)
    model = ParallelTrajectoryModel(
        setup.profile,
        SpatiotemporalEncoder(setup.transfer.encoder_config),
        congestion_predictor=CongestionPredictor(),
        congestion_tokenizer=CongestionTokenizer(),
    )
    proof = apply_transfer(model, setup.transfer)
    if device is not None:
        model.to(torch.device(device))
    return model, proof


def _load_json(path: Path, location: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(f"cannot read {location}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TrainingConfigurationError(f"{location} must be a JSON object")
    return raw


def verify_training_setup(
    config: ParallelTrajectoryTrainingConfig | str | Path,
) -> VerifiedTrainingSetup:
    parsed = (
        ParallelTrajectoryTrainingConfig.from_json(config)
        if isinstance(config, (str, Path))
        else config
    )
    if not isinstance(parsed, ParallelTrajectoryTrainingConfig):
        raise TypeError("config must be a training config or JSON path")
    source_config = Path(parsed.canonical_source_config)
    if file_sha256(source_config) != parsed.expected_canonical_source_config_raw_sha256:
        raise TrainingCompatibilityError("canonical source configuration changed")
    lineage_path = Path(parsed.lineage_report)
    lineage_sha = file_sha256(lineage_path)
    if lineage_sha != parsed.expected_lineage_report_raw_sha256:
        raise TrainingCompatibilityError("encoder lineage report changed")
    lineage = _load_json(lineage_path, "encoder lineage report")
    if (
        lineage.get("classification") != parsed.lineage_classification
        or lineage.get("stage_2_authorized") is not True
        or lineage.get("legacy_digest") != parsed.legacy_encoder_digest
        or lineage.get("current_digest") != parsed.current_encoder_digest
    ):
        raise TrainingCompatibilityError("encoder lineage report does not authorize setup")

    architecture_root = Path(__file__).resolve().parents[1] / "fortnite_parallel_trajectory"
    training_root = Path(__file__).resolve().parent
    architecture_digest = package_source_digest(architecture_root)
    training_digest = package_source_digest(training_root)
    if architecture_digest != parsed.architecture_package_digest:
        raise TrainingCompatibilityError("frozen architecture package digest changed")
    if training_digest != parsed.training_package_digest:
        raise TrainingCompatibilityError("training package digest changed")

    canonical = _verify_planner_setup(source_config)
    if Path(canonical.config.dataset_root).resolve() != Path(parsed.dataset_root).resolve():
        raise TrainingCompatibilityError("decoded-v2-new dataset root mismatch")
    if canonical.planner.manifest_sha256 != parsed.expected_split_manifest_sha256:
        raise TrainingCompatibilityError("planner split manifest hash mismatch")
    split_path = Path(parsed.split_manifest_path)
    if file_sha256(split_path) != parsed.expected_split_manifest_raw_sha256:
        raise TrainingCompatibilityError("planner split manifest raw hash mismatch")
    split_manifest = _load_json(split_path, "planner split manifest")
    if _fingerprint(split_manifest) != parsed.expected_split_manifest_sha256:
        raise TrainingCompatibilityError("planner split canonical hash mismatch")
    split_ids = canonical.planner.split_session_ids
    counts = {name: len(values) for name, values in split_ids.items()}
    if counts != parsed.expected_split_counts:
        raise TrainingCompatibilityError(
            f"decoded-v2-new planner split counts changed: {counts}"
        )
    ancestral_ids = canonical.source.session_ids_by_partition
    ancestral_counts = {name: len(values) for name, values in ancestral_ids.items()}
    if ancestral_counts != parsed.expected_ancestral_source_split_counts:
        raise TrainingCompatibilityError(
            f"ancestral encoder split counts changed: {ancestral_counts}"
        )
    if canonical.planner.ingestion_report_sha256 != parsed.expected_ingestion_report_sha256:
        raise TrainingCompatibilityError("decoded-v2-new ingestion binding changed")
    validation_path = Path(parsed.dataset_validation_path)
    if file_sha256(validation_path) != parsed.expected_dataset_validation_raw_sha256:
        raise TrainingCompatibilityError("dataset validation raw hash changed")
    validation = _load_json(validation_path, "dataset validation report")
    if validation.get("report_hash") != parsed.expected_dataset_validation_report_hash:
        raise TrainingCompatibilityError("dataset validation report hash changed")
    if canonical.planner.dataset_validation_report_hash != parsed.expected_dataset_validation_report_hash:
        raise TrainingCompatibilityError("canonical planner validation binding changed")
    if file_sha256(parsed.world_grid_profile_path) != parsed.expected_world_grid_profile_raw_sha256:
        raise TrainingCompatibilityError("world-grid profile raw hash changed")
    if canonical.profile.profile_hash != parsed.expected_world_grid_profile_hash:
        raise TrainingCompatibilityError("world-grid profile hash changed")
    if file_sha256(parsed.world_grid_audit_path) != parsed.expected_world_grid_audit_raw_sha256:
        raise TrainingCompatibilityError("world-grid audit raw hash changed")
    if canonical.profile.publication_audit.audit_sha256 != parsed.expected_world_grid_audit_payload_sha256:
        raise TrainingCompatibilityError("world-grid audit payload hash changed")

    transfer, checkpoint_metadata = load_transfer_bundle(parsed)
    if checkpoint_metadata.get("planner_split_manifest_sha256") != parsed.expected_split_manifest_sha256:
        raise TrainingCompatibilityError("checkpoint planner split binding changed")
    if checkpoint_metadata.get("planner_ingestion_report_sha256") != parsed.expected_ingestion_report_sha256:
        raise TrainingCompatibilityError("checkpoint ingestion binding changed")
    if checkpoint_metadata.get("planner_dataset_validation_report_hash") != parsed.expected_dataset_validation_report_hash:
        raise TrainingCompatibilityError("checkpoint validation binding changed")
    world_grid = checkpoint_metadata.get("world_grid_profile")
    if not isinstance(world_grid, Mapping) or world_grid.get("hash") != parsed.expected_world_grid_profile_hash:
        raise TrainingCompatibilityError("checkpoint world-grid binding changed")
    resolved = torch.load(
        parsed.source_checkpoint, map_location="cpu", weights_only=False
    ).get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise TrainingCompatibilityError("checkpoint resolved config is missing")
    if resolved.get("world_grid_coordinate_audit_sha256") != parsed.expected_world_grid_coordinate_audit_sha256:
        raise TrainingCompatibilityError("world-grid coordinate audit binding changed")

    dataset_identity = {
        "dataset_root": str(Path(parsed.dataset_root).resolve()),
        "dataset_schema_version": checkpoint_metadata.get("dataset_schema_version"),
        "ingestion_report_sha256": parsed.expected_ingestion_report_sha256,
        "dataset_validation_report_hash": parsed.expected_dataset_validation_report_hash,
        "planner_split_manifest_sha256": parsed.expected_split_manifest_sha256,
        "planner_split_counts": counts,
        "ancestral_encoder_dataset_root": canonical.source.dataset_root,
        "ancestral_encoder_ingestion_report_sha256": (
            canonical.source.ingestion_report_sha256
        ),
        "ancestral_encoder_split_manifest_sha256": (
            canonical.source.split_manifest_sha256
        ),
        "ancestral_encoder_split_counts": ancestral_counts,
        "split_namespace_resolution": (
            "86/11/10 is the decoded-v2-new planner population used by the "
            "self-conditioned run; 33/4/4 is the distinct ancestral decoded-v2 encoder split"
        ),
    }
    world_grid_binding = {
        "profile_id": canonical.profile.profile_id,
        "profile_hash": canonical.profile.profile_hash,
        "profile_raw_sha256": parsed.expected_world_grid_profile_raw_sha256,
        "publication_audit_sha256": parsed.expected_world_grid_audit_payload_sha256,
        "publication_audit_raw_sha256": parsed.expected_world_grid_audit_raw_sha256,
        "coordinate_audit_sha256": parsed.expected_world_grid_coordinate_audit_sha256,
    }
    return VerifiedTrainingSetup(
        config=parsed,
        profile=canonical.profile,
        split_manifest=split_manifest,
        split_manifest_sha256=canonical.planner.manifest_sha256,
        split_session_ids=split_ids,
        ancestral_split_session_ids=ancestral_ids,
        session_paths=canonical.planner.session_paths,
        transfer=transfer,
        source_checkpoint_metadata=checkpoint_metadata,
        source_revision=transfer.source_revision,
        architecture_config=dataclasses.asdict(ParallelTrajectoryConfig()),
        architecture_package_digest=architecture_digest,
        training_package_digest=training_digest,
        lineage_report_sha256=lineage_sha,
        dataset_identity=dataset_identity,
        world_grid_binding=world_grid_binding,
    )


__all__ = [
    "TransferBundle",
    "TransferProof",
    "VerifiedTrainingSetup",
    "apply_transfer",
    "canonical_json",
    "congestion_configuration",
    "file_sha256",
    "initialize_model",
    "module_state_digest",
    "package_source_digest",
    "tensor_manifest",
    "tensor_state_digest",
    "verify_training_setup",
]
