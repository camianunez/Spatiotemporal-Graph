from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping
import uuid

import numpy as np
import torch
from torch import nn

from .config import (
    TRAINING_CHECKPOINT_SCHEMA,
    TrainingCompatibilityError,
)
from .metrics import ParallelTrajectoryMetricAccumulator
from .optimization import OptimizerContract, PiecewiseLinearScheduler
from .provenance import TransferProof, VerifiedTrainingSetup, canonical_json


@dataclass(frozen=True, slots=True)
class TrainingState:
    completed_epochs: int = 0
    optimizer_step: int = 0
    current_epoch: int = 1
    next_batch_index: int = 0
    first_optimizer_step_completed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "completed_epochs",
            "optimizer_step",
            "next_batch_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.current_epoch) is not int or self.current_epoch <= 0:
            raise ValueError("current_epoch must be a positive integer")
        if type(self.first_optimizer_step_completed) is not bool:
            raise TypeError("first_optimizer_step_completed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "TrainingState":
        if not isinstance(raw, Mapping) or set(raw) != {
            "completed_epochs",
            "optimizer_step",
            "current_epoch",
            "next_batch_index",
            "first_optimizer_step_completed",
        }:
            raise TrainingCompatibilityError("checkpoint training state is incompatible")
        try:
            return cls(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise TrainingCompatibilityError(
                f"checkpoint training state is invalid: {exc}"
            ) from exc


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(raw: Mapping[str, Any]) -> None:
    if not isinstance(raw, Mapping) or set(raw) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise TrainingCompatibilityError("checkpoint RNG state is incompatible")
    try:
        random.setstate(raw["python"])
        np.random.set_state(raw["numpy"])
        torch.set_rng_state(raw["torch_cpu"])
        cuda_states = raw["torch_cuda"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise TrainingCompatibilityError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            if len(cuda_states) != torch.cuda.device_count():
                raise TrainingCompatibilityError("CUDA RNG device count changed")
            torch.cuda.set_rng_state_all(cuda_states)
    except TrainingCompatibilityError:
        raise
    except Exception as exc:
        raise TrainingCompatibilityError(
            f"cannot restore checkpoint RNG state: {exc}"
        ) from exc


def checkpoint_bindings(
    setup: VerifiedTrainingSetup,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
) -> dict[str, Any]:
    return {
        **setup.checkpoint_bindings(),
        "training_configuration": setup.config.to_dict(),
        "optimizer_configuration": optimizer_contract.to_dict(),
        "scheduler_configuration": scheduler.contract(),
        "precision": setup.config.precision,
        "device_contract": setup.config.device,
    }


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        # Windows requires a writable descriptor for a durable file flush.
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    path: str | Path,
    *,
    setup: VerifiedTrainingSetup,
    transfer_proof: TransferProof,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    state: TrainingState,
    best_validation_route_nll: float | None,
    metrics: ParallelTrajectoryMetricAccumulator | None = None,
    amp_state: Mapping[str, Any] | None = None,
) -> None:
    if best_validation_route_nll is not None and not np.isfinite(
        best_validation_route_nll
    ):
        raise ValueError("best validation route NLL must be finite or null")
    payload = {
        "checkpoint_schema_version": TRAINING_CHECKPOINT_SCHEMA,
        "bindings": checkpoint_bindings(setup, optimizer_contract, scheduler),
        "transfer_proof": transfer_proof.to_dict(),
        "training_state": state.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "amp_state": None if amp_state is None else dict(amp_state),
        "rng_state": rng_state(),
        "best_validation_route_nll": best_validation_route_nll,
        "metric_accumulator_state": None if metrics is None else metrics.state_dict(),
    }
    _atomic_torch_save(Path(path), payload)


def _strict_mapping(raw: Any, expected: set[str], location: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TrainingCompatibilityError(f"{location} must be a mapping")
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise TrainingCompatibilityError(
            f"{location} keys mismatch: missing={missing}, unknown={unknown}"
        )
    return raw


def load_training_checkpoint(
    path: str | Path,
    *,
    setup: VerifiedTrainingSetup,
    expected_transfer_proof: TransferProof,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    metrics: ParallelTrajectoryMetricAccumulator | None = None,
    restore_rng: bool = True,
) -> tuple[TrainingState, float | None, Mapping[str, Any] | None]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"training checkpoint does not exist: {source}")
    try:
        raw = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingCompatibilityError(f"cannot load training checkpoint: {exc}") from exc
    expected_keys = {
        "checkpoint_schema_version",
        "bindings",
        "transfer_proof",
        "training_state",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "amp_state",
        "rng_state",
        "best_validation_route_nll",
        "metric_accumulator_state",
    }
    checkpoint = _strict_mapping(raw, expected_keys, "checkpoint")
    if checkpoint["checkpoint_schema_version"] != TRAINING_CHECKPOINT_SCHEMA:
        raise TrainingCompatibilityError("training checkpoint schema changed")
    expected_bindings = checkpoint_bindings(setup, optimizer_contract, scheduler)
    if canonical_json(checkpoint["bindings"]) != canonical_json(expected_bindings):
        raise TrainingCompatibilityError("training checkpoint binding mismatch")
    if canonical_json(checkpoint["transfer_proof"]) != canonical_json(
        expected_transfer_proof.to_dict()
    ):
        raise TrainingCompatibilityError("training checkpoint transfer proof mismatch")
    state = TrainingState.from_dict(checkpoint["training_state"])
    model_state = checkpoint["model_state"]
    if not isinstance(model_state, Mapping):
        raise TrainingCompatibilityError("checkpoint model state is missing")
    try:
        result = model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise TrainingCompatibilityError(f"strict model checkpoint load failed: {exc}") from exc
    if result.missing_keys or result.unexpected_keys:
        raise TrainingCompatibilityError("strict model load returned incompatible keys")
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingCompatibilityError(f"optimizer checkpoint load failed: {exc}") from exc
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scheduler.step_index != state.optimizer_step:
        raise TrainingCompatibilityError(
            "scheduler and training optimizer-step counters diverge"
        )
    metric_state = checkpoint["metric_accumulator_state"]
    if metrics is not None:
        if metric_state is None:
            raise TrainingCompatibilityError("checkpoint metric state is missing")
        metrics.load_state_dict(metric_state)
    elif metric_state is not None:
        # A caller may intentionally ignore persisted reporting state.
        _strict_mapping(metric_state, set(metric_state), "metric accumulator state")
    best = checkpoint["best_validation_route_nll"]
    if best is not None and (
        isinstance(best, bool)
        or not isinstance(best, (int, float))
        or not np.isfinite(float(best))
    ):
        raise TrainingCompatibilityError("best validation metric is invalid")
    amp_state = checkpoint["amp_state"]
    if amp_state is not None and not isinstance(amp_state, Mapping):
        raise TrainingCompatibilityError("AMP state must be null or a mapping")
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return state, (None if best is None else float(best)), amp_state


__all__ = [
    "TrainingState",
    "checkpoint_bindings",
    "load_training_checkpoint",
    "restore_rng_state",
    "rng_state",
    "save_training_checkpoint",
]
