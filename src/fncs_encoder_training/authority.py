from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from fortnite_encoder.config import EncoderConfig, RotationHeadConfig, RotationLossConfig
from fortnite_encoder.supervision import _SUPERVISION_SCHEMAS
from fortnite_encoder.tensorize import _INPUT_SCHEMAS

from .common import file_sha256, source_tree_manifest, tensor_state_sha256, utc_now
from .config import FreshEncoderConfig
from .dataset import EXPECTED_COLUMNS, TABLE_NAMES


def _read_bound_json(path: str, expected_hash: str, label: str) -> dict[str, Any]:
    source = Path(path)
    actual = file_sha256(source)
    if actual != expected_hash:
        raise ValueError(f"{label} raw SHA-256 changed: {actual} != {expected_hash}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"authoritative {name} changed: {actual!r} != {expected!r}")


def verify_authorities(config: FreshEncoderConfig, workspace: str | Path) -> dict[str, Any]:
    workspace_root = Path(workspace).resolve()
    resolved = _read_bound_json(
        config.authoritative_resolved_config,
        config.expected_authoritative_resolved_config_sha256,
        "authoritative encoder resolved configuration",
    )
    final_summary = _read_bound_json(
        config.authoritative_final_summary,
        config.expected_authoritative_final_summary_sha256,
        "authoritative encoder final summary",
    )
    lineage = _read_bound_json(
        config.encoder_lineage_report,
        config.expected_encoder_lineage_report_sha256,
        "verified encoder lineage report",
    )
    verified_scope = _read_bound_json(
        config.encoder_scope_manifest,
        config.expected_encoder_scope_manifest_sha256,
        "verified encoder source-scope manifest",
    )
    verified_difference = _read_bound_json(
        config.verified_source_difference_report,
        config.expected_verified_source_difference_report_sha256,
        "verified encoder source-difference report",
    )
    scheduler_authority = _read_bound_json(
        config.scheduler_authority_config,
        config.expected_scheduler_authority_config_sha256,
        "verified piecewise scheduler configuration",
    )
    checkpoint_path = Path(config.authoritative_checkpoint)
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != config.expected_authoritative_checkpoint_sha256:
        raise ValueError("authoritative RotationModel checkpoint hash changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("checkpoint_schema_version") != "2.0":
        raise ValueError("authoritative RotationModel checkpoint schema changed")
    compatibility = checkpoint.get("compatibility")
    wrapper = checkpoint.get("wrapper")
    if not isinstance(compatibility, Mapping) or not isinstance(wrapper, Mapping):
        raise ValueError("authoritative checkpoint contracts are missing")
    if wrapper.get("type") != "RotationModel" or not isinstance(
        wrapper.get("state_dict"), Mapping
    ):
        raise ValueError("authoritative checkpoint is not a complete RotationModel")

    training = resolved.get("training")
    if not isinstance(training, dict):
        raise ValueError("authoritative resolved training configuration is missing")
    encoder_config = EncoderConfig(**training["encoder_config"])
    head_values = dict(training["head_config"])
    head_values["horizons_seconds"] = tuple(head_values["horizons_seconds"])
    head_config = RotationHeadConfig(**head_values)
    loss_config = RotationLossConfig(**training["loss_config"])
    _assert_equal("checkpoint encoder_config", compatibility.get("encoder_config"), training["encoder_config"])
    _assert_equal("checkpoint head_config", compatibility.get("head_config"), training["head_config"])
    _assert_equal("checkpoint loss_config", compatibility.get("loss_config"), training["loss_config"])
    _assert_equal("completed status", final_summary.get("status"), "completed")
    _assert_equal("completed epoch", final_summary.get("epoch"), config.epochs)
    _assert_equal("test evaluation", final_summary.get("test_partition_evaluated"), False)

    # Resolve every inherited hyperparameter directly against the completed
    # pure RotationModel run.  There is no estimation or checkpoint transfer.
    expected_pairs = {
        "seed": config.seed,
        "window_length_ticks": config.window_length_ticks,
        "window_stride_ticks": config.window_stride_ticks,
        "batch_size": config.batch_size,
        "accumulation_steps": config.accumulation_steps,
        "max_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "adamw_betas": list(config.adamw_betas),
        "adamw_epsilon": config.adamw_epsilon,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
        "validation_every_epochs": config.validation_every_epochs,
        "early_stopping_metric": config.checkpoint_selection_metric,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
    }
    for name, expected in expected_pairs.items():
        actual = training.get(name)
        if name == "adamw_betas" and isinstance(actual, tuple):
            actual = list(actual)
        _assert_equal(name, actual, expected)
    _assert_equal("entry loss weight", loss_config.entry_weight, config.entry_loss_weight)
    _assert_equal("survival loss weight", loss_config.survival_weight, config.survival_loss_weight)
    _assert_equal("placement loss weight", loss_config.placement_weight, config.placement_loss_weight)
    _assert_equal("resolved precision", resolved.get("precision"), "bf16")
    _assert_equal("encoder output width", encoder_config.d_model, 256)
    _assert_equal("encoder dropout", encoder_config.dropout, 0.1)
    _assert_equal("survival head enablement", head_config.enable_survival, True)
    _assert_equal("placement head enablement", head_config.enable_placement, True)

    schedule_total = (
        scheduler_authority.get("scheduler_warmup_updates", 0)
        + scheduler_authority.get("scheduler_plateau_updates", 0)
        + scheduler_authority.get("scheduler_decay_updates", 0)
    )
    if schedule_total <= 0:
        raise ValueError("scheduler authority has no valid phase boundaries")
    schedule_fractions = {
        "warmup": scheduler_authority["scheduler_warmup_updates"] / schedule_total,
        "constant": scheduler_authority["scheduler_plateau_updates"] / schedule_total,
        "linear_decay": scheduler_authority["scheduler_decay_updates"] / schedule_total,
    }
    _assert_equal("warmup schedule fraction", schedule_fractions["warmup"], config.scheduler_warmup_fraction)
    _assert_equal("constant schedule fraction", schedule_fractions["constant"], config.scheduler_constant_fraction)
    _assert_equal("decay schedule fraction", schedule_fractions["linear_decay"], config.scheduler_decay_fraction)

    current_source = source_tree_manifest(
        workspace_root / "ml" / "src" / "fortnite_encoder"
    )
    current_digest = current_source["source_tree_sha256"]
    if current_digest != config.expected_encoder_source_digest:
        raise ValueError("protected fortnite_encoder source digest changed")
    if (
        verified_scope.get("recomputed_digest") != current_digest
        or verified_scope.get("reproduced_exactly") is not True
        or lineage.get("current_digest") != current_digest
        or lineage.get("common_files_byte_identical") is not True
        or lineage.get("runtime_critical_source_differences")
        or verified_difference.get("runtime_model_critical_differences")
        or verified_difference.get("unexplained_differences")
    ):
        raise ValueError("current encoder source no longer matches verified lineage")
    verified_entries = {
        item["path"]: item["sha256"] for item in verified_scope.get("entries", [])
    }
    current_entries = {
        item["path"]: item["sha256"] for item in current_source["entries"]
    }
    if current_entries != verified_entries:
        raise ValueError("protected encoder source files differ byte-for-byte")

    model_state = wrapper["state_dict"]
    encoder_tensors = {
        name.removeprefix("encoder."): tensor
        for name, tensor in model_state.items()
        if isinstance(name, str) and name.startswith("encoder.")
    }
    head_tensors = {
        name.removeprefix("heads."): tensor
        for name, tensor in model_state.items()
        if isinstance(name, str) and name.startswith("heads.")
    }
    if len(encoder_tensors) != 124 or not head_tensors:
        raise ValueError("authoritative checkpoint tensor contract changed")

    source_difference = {
        "schema_version": "fncs-encoder-source-difference:1.0",
        "created_utc": utc_now(),
        "status": "identical",
        "current_encoder_source_digest": current_digest,
        "verified_encoder_source_digest": config.expected_encoder_source_digest,
        "source_bytes_equal": True,
        "included_paths_equal": True,
        "runtime_model_critical_differences": [],
        "unexplained_differences": [],
        "verified_lineage_report_path": config.encoder_lineage_report,
        "verified_lineage_report_raw_sha256": config.expected_encoder_lineage_report_sha256,
        "verified_source_difference_report_path": config.verified_source_difference_report,
        "verified_source_difference_report_raw_sha256": config.expected_verified_source_difference_report_sha256,
        "proceed": True,
    }
    authority = {
        "schema_version": "fncs-encoder-authority:1.0",
        "created_utc": utc_now(),
        "authoritative_run": config.authoritative_run_directory,
        "authoritative_resolved_config": config.authoritative_resolved_config,
        "authoritative_resolved_config_raw_sha256": config.expected_authoritative_resolved_config_sha256,
        "authoritative_checkpoint": config.authoritative_checkpoint,
        "authoritative_checkpoint_raw_sha256": checkpoint_hash,
        "checkpoint_inspection_only": True,
        "checkpoint_weights_loaded_into_new_model": False,
        "authoritative_final_summary": config.authoritative_final_summary,
        "authoritative_final_summary_raw_sha256": config.expected_authoritative_final_summary_sha256,
        "encoder_config": dataclasses.asdict(encoder_config),
        "head_config": dataclasses.asdict(head_config),
        "loss_config": dataclasses.asdict(loss_config),
        "class_contract": dict(compatibility["class_contract"]),
        "resolved_hyperparameters": {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "gradient_accumulation": config.accumulation_steps,
            "context_window_length_ticks": config.window_length_ticks,
            "query_sampling": (
                "one uniformly sampled stride-aligned causal window per canonical "
                "training session per epoch; all valid queries in the window retain "
                "the verified RotationModel masks and reductions"
            ),
            "window_stride_ticks": config.window_stride_ticks,
            "epoch_count": config.epochs,
            "adamw_learning_rate": config.learning_rate,
            "adamw_betas": list(config.adamw_betas),
            "adamw_epsilon": config.adamw_epsilon,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "loss_weights": {
                "entry": config.entry_loss_weight,
                "survival": config.survival_loss_weight,
                "placement": config.placement_loss_weight,
                "regularization": config.regularization_loss_weight,
            },
            "dropout": encoder_config.dropout,
            "precision": config.precision,
            "validation_cadence_epochs": config.validation_every_epochs,
            "checkpoint_selection_metric": config.checkpoint_selection_metric,
            "scheduler_phase_proportions": schedule_fractions,
            "scheduler_final_lr_ratio": config.scheduler_final_lr_ratio,
        },
        "schedule_authority": {
            "path": config.scheduler_authority_config,
            "raw_sha256": config.expected_scheduler_authority_config_sha256,
            "type": scheduler_authority.get("scheduler_type"),
            "absolute_prior_boundaries_not_reused": True,
            "proportions": schedule_fractions,
        },
        "enabled_supervision_heads": [
            "future_position",
            "zone_entry",
            "survival",
            "placement",
        ],
        "forbidden_model_families": [
            "fortnite_early_zone",
            "fortnite_parallel_trajectory",
            "autoregressive_route_decoder",
            "congestion_planner",
            "beam_search",
        ],
        "encoder_tensor_count_inspected": len(encoder_tensors),
        "training_head_tensor_count_inspected": len(head_tensors),
        "authoritative_encoder_state_sha256": tensor_state_sha256(encoder_tensors),
        "authoritative_training_head_state_sha256": tensor_state_sha256(head_tensors),
        "authoritative_complete_model_state_sha256": tensor_state_sha256(model_state),
        "source_difference": source_difference,
        "source_manifest": current_source,
    }
    # Release checkpoint storage as soon as metadata/tensor names have been checked.
    del checkpoint, model_state, encoder_tensors, head_tensors
    return authority


def feature_contract(authority: Mapping[str, Any]) -> dict[str, Any]:
    encoder = authority["encoder_config"]
    return {
        "schema_version": "fncs-encoder-feature-contract:1.0",
        "encoder_output_contract": "[B,T,N,256]",
        "dataset_schema_version": "2.0.0",
        "required_session_tables": list(TABLE_NAMES),
        "encoder_input_tables": [
            "match_samples.parquet",
            "player_samples.parquet",
            "zone_samples.parquet",
            "zone_phases.parquet",
        ],
        "training_supervision_tables": [
            "team_samples.parquet",
            "player_events.parquet",
            "zone_phases.parquet",
        ],
        "validation_only_required_table": "centroid_neighbors.parquet",
        "input_column_allowlist": {
            name: list(columns) for name, columns in EXPECTED_COLUMNS.items()
        },
        "pyarrow_schema_allowlist": {
            name: [
                {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                for field in schema
            ]
            for name, schema in {**_INPUT_SCHEMAS, **_SUPERVISION_SCHEMAS}.items()
        },
        "encoder_config": dict(encoder),
        "static_and_temporal_covariates": (
            "byte-pinned fortnite_encoder.features and tensorize implementation"
        ),
        "mask_contract": "byte-pinned player, team-slot, time, and causal masks",
        "spatial_graph_contract": {
            "neighbor_mode": encoder["neighbor_mode"],
            "knn_k": encoder["knn_k"],
            "edge_width": 8,
        },
        "full_match_behavior": "targets are built over a full match before slicing",
        "context_window_behavior": "causal 64-tick windows retain only the prior-state motion sidecar",
        "team_permutation_contract": "equivariant",
        "player_slot_permutation_contract": "invariant",
        "causal_mask_contract": "future keys are ineligible",
        "source_digest": authority["source_manifest"]["source_tree_sha256"],
    }
