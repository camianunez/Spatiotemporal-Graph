from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import torch
from torch import nn

from fortnite_parallel_trajectory.model import ParallelTrajectoryModel

from .config import (
    SCHEDULER_TYPE,
    ParallelTrajectoryTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
)


_GROUP_NAMES = (
    "decoder_decay",
    "decoder_no_decay",
    "congestion_decay",
    "congestion_no_decay",
    "encoder_decay",
    "encoder_no_decay",
)


@dataclass(frozen=True, slots=True)
class OptimizerContract:
    optimizer_type: str
    group_parameter_names: tuple[tuple[str, tuple[str, ...]], ...]
    decoder_learning_rate: float
    congestion_learning_rate: float
    encoder_learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    no_decay_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_type": self.optimizer_type,
            "group_parameter_names": [
                [name, list(parameter_names)]
                for name, parameter_names in self.group_parameter_names
            ],
            "decoder_learning_rate": self.decoder_learning_rate,
            "congestion_learning_rate": self.congestion_learning_rate,
            "encoder_learning_rate": self.encoder_learning_rate,
            "betas": list(self.betas),
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "no_decay_rule": self.no_decay_rule,
        }


def _named_parameters_with_aliases(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    try:
        return tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with old torch only
        return tuple(model.named_parameters())


def build_adamw_optimizer(
    model: ParallelTrajectoryModel,
    config: ParallelTrajectoryTrainingConfig,
) -> tuple[torch.optim.AdamW, OptimizerContract]:
    """Create decoder/congestion/encoder crossed with decay/no-decay."""

    if not isinstance(model, ParallelTrajectoryModel):
        raise TypeError("model must be ParallelTrajectoryModel")
    layer_norm_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        name: [] for name in _GROUP_NAMES
    }
    seen: dict[int, str] = {}
    for name, parameter in sorted(_named_parameters_with_aliases(model)):
        prior = seen.get(id(parameter))
        if prior is not None:
            raise TrainingConfigurationError(
                f"optimizer parameter is aliased by {prior!r} and {name!r}"
            )
        seen[id(parameter)] = name
        if name.startswith("decoder."):
            owner = "decoder"
        elif name.startswith("congestion_predictor.") or name.startswith(
            "congestion_tokenizer."
        ):
            owner = "congestion"
        elif name.startswith("encoder."):
            owner = "encoder"
        else:
            raise TrainingConfigurationError(
                f"optimizer encountered a parameter without an owner: {name}"
            )
        no_decay = name.endswith(".bias") or id(parameter) in layer_norm_parameters
        group = f"{owner}_{'no_decay' if no_decay else 'decay'}"
        grouped[group].append((name, parameter))

    empty = [name for name in _GROUP_NAMES if not grouped[name]]
    if empty:
        raise TrainingConfigurationError(f"required optimizer groups are empty: {empty}")
    unique_model_parameters = {id(parameter) for parameter in model.parameters()}
    if set(seen) != unique_model_parameters:
        raise TrainingConfigurationError("optimizer ownership is not exhaustive")

    learning_rates = {
        "decoder": float(config.decoder_learning_rate),
        "congestion": float(config.congestion_learning_rate),
        "encoder": float(config.encoder_learning_rate),
    }
    parameter_groups: list[dict[str, Any]] = []
    for group_name in _GROUP_NAMES:
        owner = group_name.split("_", 1)[0]
        parameter_groups.append(
            {
                "params": [parameter for _, parameter in grouped[group_name]],
                "lr": learning_rates[owner],
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
        parameter_groups,
        betas=tuple(float(value) for value in config.adamw_betas),
        eps=float(config.adamw_epsilon),
    )
    contract = OptimizerContract(
        optimizer_type="AdamW",
        group_parameter_names=tuple(
            (name, tuple(parameter_name for parameter_name, _ in grouped[name]))
            for name in _GROUP_NAMES
        ),
        decoder_learning_rate=float(config.decoder_learning_rate),
        congestion_learning_rate=float(config.congestion_learning_rate),
        encoder_learning_rate=float(config.encoder_learning_rate),
        betas=tuple(float(value) for value in config.adamw_betas),
        epsilon=float(config.adamw_epsilon),
        weight_decay=float(config.weight_decay),
        no_decay_rule="biases_and_torch.nn.LayerNorm_parameters",
    )
    return optimizer, contract


class PiecewiseLinearScheduler:
    """One-indexed 43 warmup, 43 constant, 774 linear-decay schedule."""

    scheduler_type = SCHEDULER_TYPE

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: ParallelTrajectoryTrainingConfig,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = config.max_optimizer_steps
        self.warmup_steps = config.scheduler_warmup_updates
        self.plateau_steps = config.scheduler_plateau_updates
        self.decay_steps = config.scheduler_decay_updates
        if (
            self.warmup_steps + self.plateau_steps + self.decay_steps
            != self.total_steps
        ):
            raise ValueError("scheduler sections do not cover all optimizer steps")
        self.peak_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if any(value <= 0.0 or not math.isfinite(value) for value in self.peak_lrs):
            raise ValueError("optimizer groups must begin at positive peak LRs")
        self.step_index = 0
        self._set_for_update(1)

    def ratio_at(self, update: int) -> float:
        if type(update) is not int or not 1 <= update <= self.total_steps:
            raise ValueError(
                f"scheduler update must be in [1,{self.total_steps}]"
            )
        if update <= self.warmup_steps:
            return update / self.warmup_steps
        if update <= self.warmup_steps + self.plateau_steps:
            return 1.0
        return (self.total_steps - update) / self.decay_steps

    def stage_at(self, update: int) -> Literal["warmup", "constant", "linear_decay"]:
        self.ratio_at(update)
        if update <= self.warmup_steps:
            return "warmup"
        if update <= self.warmup_steps + self.plateau_steps:
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
            "constant_steps": self.plateau_steps,
            "linear_decay_steps": self.decay_steps,
            "peak_lrs": list(self.peak_lrs),
            "update_indexing": "one_indexed_learning_rate_used_then_step",
        }

    def state_dict(self) -> dict[str, Any]:
        return {**self.contract(), "step_index": self.step_index}

    def load_state_dict(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            raise TrainingCompatibilityError("scheduler state must be a mapping")
        if any(raw.get(name) != value for name, value in self.contract().items()):
            raise TrainingCompatibilityError("scheduler checkpoint contract mismatch")
        step = raw.get("step_index")
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise TrainingCompatibilityError("scheduler checkpoint step is invalid")
        self.step_index = step
        self._set_for_update(self.upcoming_update)


def apply_freeze_policy(model: ParallelTrajectoryModel, epoch: int) -> str:
    if not isinstance(model, ParallelTrajectoryModel):
        raise TypeError("model must be ParallelTrajectoryModel")
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("epoch must be positive")
    inherited_frozen = epoch == 1
    for module in (
        model.encoder,
        model.congestion_predictor,
        model.congestion_tokenizer,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(not inherited_frozen)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    phase = "inherited_frozen" if inherited_frozen else "all_trainable"
    verify_freeze_policy(model, phase)
    return phase


def verify_freeze_policy(model: ParallelTrajectoryModel, phase: str) -> None:
    if phase not in {"inherited_frozen", "all_trainable"}:
        raise ValueError("unknown freeze phase")
    inherited_should_train = phase == "all_trainable"
    inherited = (
        tuple(model.encoder.parameters())
        + tuple(model.congestion_predictor.parameters())
        + tuple(model.congestion_tokenizer.parameters())
    )
    if any(
        parameter.requires_grad != inherited_should_train
        for parameter in inherited
    ):
        raise TrainingCompatibilityError("inherited freeze policy is inconsistent")
    if any(not parameter.requires_grad for parameter in model.decoder.parameters()):
        raise TrainingCompatibilityError("scratch decoder must always be trainable")


def gradient_group_report(
    model: ParallelTrajectoryModel,
) -> dict[str, dict[str, Any]]:
    owners = {
        "decoder": tuple(model.decoder.parameters()),
        "congestion": tuple(model.congestion_predictor.parameters())
        + tuple(model.congestion_tokenizer.parameters()),
        "encoder": tuple(model.encoder.parameters()),
    }
    report: dict[str, dict[str, Any]] = {}
    for name, parameters in owners.items():
        trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
        gradients = tuple(
            parameter.grad for parameter in trainable if parameter.grad is not None
        )
        finite = all(bool(torch.isfinite(value).all()) for value in gradients)
        squared = sum(
            float(value.detach().float().square().sum().item()) for value in gradients
        )
        report[name] = {
            "trainable_parameter_count": len(trainable),
            "parameters_with_gradient": len(gradients),
            "all_gradients_finite": finite,
            "gradient_norm": math.sqrt(squared),
        }
    return report


__all__ = [
    "OptimizerContract",
    "PiecewiseLinearScheduler",
    "apply_freeze_policy",
    "build_adamw_optimizer",
    "gradient_group_report",
    "verify_freeze_policy",
]
