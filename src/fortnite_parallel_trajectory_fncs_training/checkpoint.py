from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from fncs_encoder_training.common import (
    atomic_torch_save,
    canonical_json,
    restore_rng_state,
    rng_state,
    tensor_state_sha256,
    utc_now,
)
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel
from fortnite_parallel_trajectory_training.metrics import (
    ParallelTrajectoryMetricAccumulator,
)

from .binding import (
    VerifiedSetup,
    assert_frozen_encoder,
    downstream_state,
    load_downstream_state,
)
from .config import FROZEN_FNCS_CHECKPOINT_SCHEMA, TrainingCompatibilityError
from .optimization import OptimizerContract, PiecewiseLinearScheduler


@dataclass(frozen=True, slots=True)
class TrainingState:
    completed_epochs: int = 0
    optimizer_step: int = 0
    current_epoch: int = 1
    next_batch_index: int = 0
    first_optimizer_step_completed: bool = False

    def __post_init__(self) -> None:
        for name in ("completed_epochs", "optimizer_step", "next_batch_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.current_epoch) is not int or self.current_epoch <= 0:
            raise ValueError("current_epoch must be positive")
        if type(self.first_optimizer_step_completed) is not bool:
            raise TypeError("first_optimizer_step_completed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "TrainingState":
        expected = {
            "completed_epochs",
            "optimizer_step",
            "current_epoch",
            "next_batch_index",
            "first_optimizer_step_completed",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise TrainingCompatibilityError("checkpoint training state changed")
        try:
            return cls(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise TrainingCompatibilityError(
                f"checkpoint training state is invalid: {exc}"
            ) from exc


def checkpoint_bindings(
    setup: VerifiedSetup,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    initial_downstream_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **setup.checkpoint_bindings(),
        "training_configuration": setup.config.to_dict(),
        "optimizer": optimizer_contract.to_dict(),
        "scheduler": scheduler.contract(),
        "precision": setup.config.precision,
        "device": setup.config.device,
        "initialization_seed": setup.config.initialization_seed,
        "initial_downstream_parameter_sha256": initial_downstream_parameters[
            "initial_downstream_parameter_sha256"
        ],
        "objective": setup.config.objective,
        "congestion_auxiliary_objective": (
            setup.config.congestion_auxiliary_objective
        ),
        "checkpoint_selection_metric": setup.config.checkpoint_selection_metric,
    }


def save_checkpoint(
    path: str | Path,
    *,
    setup: VerifiedSetup,
    model: ParallelTrajectoryModel,
    optimizer: torch.optim.Optimizer,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    state: TrainingState,
    initial_downstream_parameters: Mapping[str, Any],
    best_validation_route_nll: float | None,
    metrics: ParallelTrajectoryMetricAccumulator,
    data_access: Mapping[str, Any],
) -> None:
    if best_validation_route_nll is not None and not math.isfinite(
        best_validation_route_nll
    ):
        raise ValueError("best validation route NLL must be finite or null")
    if (
        data_access.get("zero_test_attempts") is not True
        or data_access.get("zero_test_opens") is not True
    ):
        raise TrainingCompatibilityError(
            "refusing to checkpoint after encoder-test access"
        )
    frozen = assert_frozen_encoder(model, setup, stage="checkpoint_save")
    downstream = downstream_state(model)
    payload = {
        "checkpoint_schema_version": FROZEN_FNCS_CHECKPOINT_SCHEMA,
        "created_utc": utc_now(),
        "type": "FrozenFNCSParallelTrajectoryDecoderTraining",
        "bindings": checkpoint_bindings(
            setup, optimizer_contract, scheduler, initial_downstream_parameters
        ),
        "training_state": state.to_dict(),
        "downstream_state": downstream,
        "downstream_state_sha256": tensor_state_sha256(downstream),
        "frozen_encoder_snapshot": frozen,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": rng_state(),
        "best_validation_route_nll": best_validation_route_nll,
        "metric_accumulator_state": metrics.state_dict(),
        "data_access": dict(data_access),
        "test_partition_evaluated": False,
    }
    atomic_torch_save(path, payload)


def _strict_mapping(raw: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TrainingCompatibilityError(f"{label} must be a mapping")
    missing = sorted(expected - set(raw))
    unexpected = sorted(set(raw) - expected)
    if missing or unexpected:
        raise TrainingCompatibilityError(
            f"{label} keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    return raw


def load_checkpoint(
    path: str | Path,
    *,
    setup: VerifiedSetup,
    model: ParallelTrajectoryModel,
    optimizer: torch.optim.Optimizer,
    optimizer_contract: OptimizerContract,
    scheduler: PiecewiseLinearScheduler,
    initial_downstream_parameters: Mapping[str, Any],
    metrics: ParallelTrajectoryMetricAccumulator,
    restore_rng: bool,
) -> tuple[TrainingState, float | None]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"training checkpoint is missing: {source}")
    try:
        raw = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingCompatibilityError(f"cannot load checkpoint: {exc}") from exc
    checkpoint = _strict_mapping(
        raw,
        {
            "checkpoint_schema_version",
            "created_utc",
            "type",
            "bindings",
            "training_state",
            "downstream_state",
            "downstream_state_sha256",
            "frozen_encoder_snapshot",
            "optimizer_state",
            "scheduler_state",
            "rng_state",
            "best_validation_route_nll",
            "metric_accumulator_state",
            "data_access",
            "test_partition_evaluated",
        },
        "decoder checkpoint",
    )
    if checkpoint["checkpoint_schema_version"] != FROZEN_FNCS_CHECKPOINT_SCHEMA:
        raise TrainingCompatibilityError("decoder checkpoint schema changed")
    if checkpoint["type"] != "FrozenFNCSParallelTrajectoryDecoderTraining":
        raise TrainingCompatibilityError("decoder checkpoint type changed")
    if checkpoint["test_partition_evaluated"] is not False:
        raise TrainingCompatibilityError("checkpoint records forbidden test evaluation")
    expected_bindings = checkpoint_bindings(
        setup, optimizer_contract, scheduler, initial_downstream_parameters
    )
    if canonical_json(checkpoint["bindings"]) != canonical_json(expected_bindings):
        raise TrainingCompatibilityError("decoder checkpoint bindings changed")
    downstream = checkpoint["downstream_state"]
    if not isinstance(downstream, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in downstream.items()
    ):
        raise TrainingCompatibilityError("downstream checkpoint state is invalid")
    if tensor_state_sha256(downstream) != checkpoint["downstream_state_sha256"]:
        raise TrainingCompatibilityError("downstream checkpoint tensor hash is invalid")
    load_downstream_state(model, downstream)
    frozen = checkpoint["frozen_encoder_snapshot"]
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("encoder_state_sha256")
        != setup.config.expected_encoder_state_sha256
        or frozen.get("matches_canonical") is not True
    ):
        raise TrainingCompatibilityError("checkpoint frozen-encoder proof changed")
    assert_frozen_encoder(model, setup, stage="checkpoint_reload")
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingCompatibilityError(
            f"optimizer checkpoint load failed: {exc}"
        ) from exc
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    state = TrainingState.from_dict(checkpoint["training_state"])
    if scheduler.step_index != state.optimizer_step:
        raise TrainingCompatibilityError(
            "scheduler and optimizer-step counters diverge"
        )
    metrics.load_state_dict(checkpoint["metric_accumulator_state"])
    access = checkpoint["data_access"]
    if (
        not isinstance(access, Mapping)
        or access.get("zero_test_attempts") is not True
        or access.get("zero_test_opens") is not True
    ):
        raise TrainingCompatibilityError("checkpoint contains forbidden test access")
    best = checkpoint["best_validation_route_nll"]
    if best is not None and (
        isinstance(best, bool)
        or not isinstance(best, (int, float))
        or not math.isfinite(float(best))
    ):
        raise TrainingCompatibilityError("checkpoint best metric is invalid")
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return state, None if best is None else float(best)


__all__ = [
    "TrainingState",
    "checkpoint_bindings",
    "load_checkpoint",
    "save_checkpoint",
]
