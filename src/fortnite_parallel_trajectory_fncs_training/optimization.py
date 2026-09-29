from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import torch
from torch import nn

from fortnite_parallel_trajectory.model import ParallelTrajectoryModel

from .config import (
    SCHEDULER_TYPE,
    FrozenFNCSTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
)


GROUP_NAMES = (
    "decoder_decay",
    "decoder_no_decay",
    "congestion_decay",
    "congestion_no_decay",
)


@dataclass(frozen=True, slots=True)
class OptimizerContract:
    optimizer_type: str
    group_parameter_names: tuple[tuple[str, tuple[str, ...]], ...]
    downstream_learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    no_decay_rule: str
    encoder_parameter_count: int
    encoder_optimizer_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_type": self.optimizer_type,
            "group_parameter_names": [
                [name, list(parameter_names)]
                for name, parameter_names in self.group_parameter_names
            ],
            "downstream_learning_rate": self.downstream_learning_rate,
            "betas": list(self.betas),
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "no_decay_rule": self.no_decay_rule,
            "encoder_parameter_count": self.encoder_parameter_count,
            "encoder_optimizer_parameter_count": (
                self.encoder_optimizer_parameter_count
            ),
            "encoder_optimizer_group_present": False,
        }


def build_optimizer(
    model: ParallelTrajectoryModel,
    config: FrozenFNCSTrainingConfig,
) -> tuple[torch.optim.AdamW, OptimizerContract]:
    layer_norm_parameters = {
        id(parameter)
        for module in (
            model.congestion_predictor,
            model.congestion_tokenizer,
            model.decoder,
        )
        for child in module.modules()
        if isinstance(child, nn.LayerNorm)
        for parameter in child.parameters(recurse=False)
    }
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        name: [] for name in GROUP_NAMES
    }
    seen: set[int] = set()
    downstream_modules = {
        "decoder": model.decoder,
        "congestion_predictor": model.congestion_predictor,
        "congestion_tokenizer": model.congestion_tokenizer,
    }
    for owner, module in downstream_modules.items():
        component = "decoder" if owner == "decoder" else "congestion"
        for local_name, parameter in sorted(module.named_parameters()):
            name = f"{owner}.{local_name}"
            if id(parameter) in seen:
                raise TrainingConfigurationError(
                    f"downstream optimizer parameter is aliased: {name}"
                )
            seen.add(id(parameter))
            if not parameter.requires_grad:
                raise TrainingConfigurationError(
                    f"downstream parameter is unexpectedly frozen: {name}"
                )
            no_decay = name.endswith(".bias") or id(parameter) in layer_norm_parameters
            group = f"{component}_{'no_decay' if no_decay else 'decay'}"
            grouped[group].append((name, parameter))
    empty = [name for name, values in grouped.items() if not values]
    if empty:
        raise TrainingConfigurationError(f"required optimizer groups are empty: {empty}")
    expected_ids = {
        id(parameter)
        for module in downstream_modules.values()
        for parameter in module.parameters()
    }
    if seen != expected_ids:
        raise TrainingConfigurationError(
            "downstream optimizer ownership is not exhaustive"
        )
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    if seen & encoder_ids:
        raise TrainingConfigurationError("encoder parameter entered the optimizer")
    parameter_groups: list[dict[str, Any]] = []
    for group_name in GROUP_NAMES:
        parameter_groups.append(
            {
                "params": [parameter for _, parameter in grouped[group_name]],
                "lr": config.downstream_learning_rate,
                "weight_decay": (
                    config.weight_decay if group_name.endswith("_decay") else 0.0
                ),
                "group_name": group_name,
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=config.adamw_betas,
        eps=config.adamw_epsilon,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != expected_ids or optimizer_ids & encoder_ids:
        raise TrainingConfigurationError(
            "optimizer parameters do not equal the complete downstream boundary"
        )
    contract = OptimizerContract(
        optimizer_type="AdamW",
        group_parameter_names=tuple(
            (name, tuple(parameter_name for parameter_name, _ in grouped[name]))
            for name in GROUP_NAMES
        ),
        downstream_learning_rate=config.downstream_learning_rate,
        betas=config.adamw_betas,
        epsilon=config.adamw_epsilon,
        weight_decay=config.weight_decay,
        no_decay_rule="biases_and_torch.nn.LayerNorm_parameters",
        encoder_parameter_count=len(encoder_ids),
        encoder_optimizer_parameter_count=0,
    )
    return optimizer, contract


class PiecewiseLinearScheduler:
    """One-indexed 5% warmup, 5% plateau, 90% decay to 10%."""

    scheduler_type = SCHEDULER_TYPE

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: FrozenFNCSTrainingConfig,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = config.max_optimizer_steps
        self.warmup_steps = config.scheduler_warmup_updates
        self.constant_steps = config.scheduler_constant_updates
        self.decay_steps = config.scheduler_decay_updates
        self.final_lr_ratio = config.scheduler_final_lr_ratio
        if (
            self.warmup_steps + self.constant_steps + self.decay_steps
            != self.total_steps
        ):
            raise TrainingConfigurationError(
                "scheduler sections do not cover the optimizer-step contract"
            )
        self.peak_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if any(not math.isfinite(value) or value <= 0.0 for value in self.peak_lrs):
            raise TrainingConfigurationError("optimizer peak LRs must be positive")
        self.step_index = 0
        self._set_for_update(1)

    def ratio_at(self, update: int) -> float:
        if type(update) is not int or not 1 <= update <= self.total_steps:
            raise ValueError(f"scheduler update must be in [1,{self.total_steps}]")
        if update <= self.warmup_steps:
            return update / self.warmup_steps
        if update <= self.warmup_steps + self.constant_steps:
            return 1.0
        decay_position = update - self.warmup_steps - self.constant_steps
        return 1.0 - (1.0 - self.final_lr_ratio) * (
            decay_position / self.decay_steps
        )

    def stage_at(self, update: int) -> Literal["warmup", "constant", "linear_decay"]:
        self.ratio_at(update)
        if update <= self.warmup_steps:
            return "warmup"
        if update <= self.warmup_steps + self.constant_steps:
            return "constant"
        return "linear_decay"

    @property
    def upcoming_update(self) -> int:
        return min(self.step_index + 1, self.total_steps)

    def _set_for_update(self, update: int) -> None:
        ratio = self.ratio_at(update)
        for group, peak in zip(self.optimizer.param_groups, self.peak_lrs):
            group["lr"] = peak * ratio

    def step_record(self) -> list[dict[str, Any]]:
        update = self.upcoming_update
        return [
            {
                "group_name": str(group.get("group_name", f"group_{index}")),
                "stage": self.stage_at(update),
                "ratio": self.ratio_at(update),
                "peak_learning_rate": peak,
                "learning_rate_used": float(group["lr"]),
            }
            for index, (group, peak) in enumerate(
                zip(self.optimizer.param_groups, self.peak_lrs)
            )
        ]

    def step(self) -> None:
        if self.step_index >= self.total_steps:
            raise RuntimeError("scheduler advanced beyond its fixed update count")
        self.step_index += 1
        if self.step_index < self.total_steps:
            self._set_for_update(self.step_index + 1)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def contract(self) -> dict[str, Any]:
        return {
            "scheduler_type": self.scheduler_type,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "constant_steps": self.constant_steps,
            "linear_decay_steps": self.decay_steps,
            "final_lr_ratio": self.final_lr_ratio,
            "peak_lrs": list(self.peak_lrs),
            "update_indexing": "one_indexed_learning_rate_used_then_step",
            "boundaries": {
                "warmup_updates_inclusive": [1, self.warmup_steps],
                "constant_updates_inclusive": [
                    self.warmup_steps + 1,
                    self.warmup_steps + self.constant_steps,
                ],
                "linear_decay_updates_inclusive": [
                    self.warmup_steps + self.constant_steps + 1,
                    self.total_steps,
                ],
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {**self.contract(), "step_index": self.step_index}

    def load_state_dict(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            raise TrainingCompatibilityError("scheduler state must be a mapping")
        expected = self.contract()
        if set(raw) != set(expected) | {"step_index"} or any(
            raw.get(name) != value for name, value in expected.items()
        ):
            raise TrainingCompatibilityError("scheduler checkpoint contract changed")
        step = raw.get("step_index")
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise TrainingCompatibilityError("scheduler checkpoint step is invalid")
        self.step_index = step
        if self.step_index < self.total_steps:
            self._set_for_update(self.step_index + 1)
        else:
            self._set_for_update(self.total_steps)


def gradient_report(model: ParallelTrajectoryModel) -> dict[str, dict[str, Any]]:
    owners = {
        "decoder": tuple(model.decoder.named_parameters()),
        "congestion_predictor": tuple(model.congestion_predictor.named_parameters()),
        "congestion_tokenizer": tuple(model.congestion_tokenizer.named_parameters()),
        "encoder": tuple(model.encoder.named_parameters()),
    }
    report: dict[str, dict[str, Any]] = {}
    for owner, named_parameters in owners.items():
        trainable = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
        ]
        gradients = [
            (name, parameter.grad)
            for name, parameter in named_parameters
            if parameter.grad is not None
        ]
        nonzero_names = [
            name
            for name, gradient in gradients
            if gradient is not None and bool(torch.count_nonzero(gradient).item())
        ]
        squared = sum(
            float(gradient.detach().float().square().sum().item())
            for _, gradient in gradients
            if gradient is not None
        )
        report[owner] = {
            "parameter_count": len(named_parameters),
            "trainable_parameter_count": len(trainable),
            "parameters_with_gradient": len(gradients),
            "parameters_with_nonzero_gradient": len(nonzero_names),
            "nonzero_gradient_parameter_names": nonzero_names,
            "all_gradients_finite": all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for _, gradient in gradients
            ),
            "gradient_norm": math.sqrt(squared),
        }
    return report


__all__ = [
    "GROUP_NAMES",
    "OptimizerContract",
    "PiecewiseLinearScheduler",
    "build_optimizer",
    "gradient_report",
]
