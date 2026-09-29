from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow
import torch
from torch import nn

from .common import (
    atomic_torch_save,
    atomic_write_json,
    file_sha256,
    git_environment,
    tensor_manifest,
    tensor_state_sha256,
    utc_now,
)


def environment_report(workspace: str | Path) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    dependencies: dict[str, str] = {}
    for name in ("numpy", "pyarrow", "torch", "setuptools", "pytest"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not-installed"
    return {
        "schema_version": "fncs-encoder-environment:1.0",
        "created_utc": utc_now(),
        "process_id": os.getpid(),
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_model": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "dependencies": dependencies,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "git": git_environment(workspace),
    }


def parameter_report(model: nn.Module, *, seed: int) -> dict[str, Any]:
    state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
    }
    rows = tensor_manifest(state)
    encoder_rows = [item for item in rows if item["name"].startswith("encoder.")]
    head_rows = [item for item in rows if item["name"].startswith("heads.")]
    return {
        "schema_version": "fncs-initial-parameters:1.0",
        "seed": seed,
        "fresh_random_initialization": True,
        "existing_checkpoint_weights_loaded": False,
        "parameter_count": len(rows),
        "scalar_parameter_count": sum(item["numel"] for item in rows),
        "encoder_parameter_tensor_count": len(encoder_rows),
        "head_parameter_tensor_count": len(head_rows),
        "complete_initial_parameter_sha256": tensor_state_sha256(state),
        "encoder_initial_parameter_sha256": tensor_state_sha256(
            {
                name.removeprefix("encoder."): tensor
                for name, tensor in state.items()
                if name.startswith("encoder.")
            }
        ),
        "training_head_initial_parameter_sha256": tensor_state_sha256(
            {
                name.removeprefix("heads."): tensor
                for name, tensor in state.items()
                if name.startswith("heads.")
            }
        ),
        "parameters": rows,
    }


def checkpoint_tensor_report(checkpoint_path: str | Path) -> dict[str, Any]:
    source = Path(checkpoint_path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint has no model_state mapping")
    rows = tensor_manifest(model_state)
    return {
        "schema_version": "fncs-checkpoint-tensor-manifest:1.0",
        "checkpoint": source.name,
        "checkpoint_raw_sha256": file_sha256(source),
        "checkpoint_size_bytes": source.stat().st_size,
        "optimizer_step": checkpoint.get("training_state", {}).get("optimizer_step"),
        "tensor_count": len(rows),
        "model_state_sha256": tensor_state_sha256(model_state),
        "tensors": rows,
    }


def export_encoder_only(
    path: str | Path,
    *,
    model: nn.Module,
    encoder_config: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
    training_state: Mapping[str, Any],
    training_run_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    encoder_state = {
        name.removeprefix("encoder."): tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith("encoder.")
    }
    manifest = tensor_manifest(encoder_state)
    payload = {
        "checkpoint_schema_version": "fncs-encoder-only-checkpoint:1.0",
        "type": "SpatiotemporalEncoder",
        "encoder_state": encoder_state,
        "encoder_config": dict(encoder_config),
        "feature_contract": dict(feature_contract),
        "source_digest": bindings["encoder_source_digest"],
        "dataset_bindings": {
            "ingestion_report_sha256": bindings["ingestion_report_sha256"],
            "dataset_validation_sha256": bindings["dataset_validation_sha256"],
            "split_manifest_sha256": bindings["split_manifest_sha256"],
        },
        "world_grid_bindings": {
            "profile_hash": bindings["world_grid_profile_hash"],
            "publication_audit_sha256": bindings["world_grid_publication_audit_sha256"],
            "validation_audit_sha256": bindings["world_grid_validation_audit_sha256"],
        },
        "training_state": dict(training_state),
        "training_run_provenance": dict(training_run_provenance),
        "encoder_tensor_manifest": manifest,
        "encoder_state_sha256": tensor_state_sha256(encoder_state),
    }
    atomic_torch_save(path, payload)
    return {
        "path": str(Path(path).resolve()),
        "raw_sha256": file_sha256(path),
        "tensor_count": len(manifest),
        "encoder_state_sha256": payload["encoder_state_sha256"],
        "optimizer_step": training_state.get("optimizer_step"),
    }


def artifact_manifest(
    run_directory: str | Path,
    *,
    status: str,
    exclude_names: tuple[str, ...] = ("artifact_manifest.json",),
) -> dict[str, Any]:
    root = Path(run_directory).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in exclude_names or path.name.endswith(".tmp"):
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "raw_sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": "fncs-encoder-artifact-manifest:1.0",
        "created_utc": utc_now(),
        "status": status,
        "run_directory": str(root),
        "artifact_count_excluding_manifest": len(rows),
        "manifest_self_hash_excluded_to_avoid_recursion": True,
        "artifacts": rows,
    }


def write_artifact_manifest(run_directory: str | Path, *, status: str) -> dict[str, Any]:
    value = artifact_manifest(run_directory, status=status)
    atomic_write_json(Path(run_directory) / "artifact_manifest.json", value)
    return value
