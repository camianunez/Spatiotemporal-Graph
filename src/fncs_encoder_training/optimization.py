from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ScheduleContract:
    scheduler_type: str
    total_optimizer_steps: int
    warmup_steps: int
    constant_steps: int
    linear_decay_steps: int
    warmup_fraction: float
    constant_fraction: float
    linear_decay_fraction: float
    final_lr_ratio: float
    update_indexing: str = "one_indexed_learning_rate_used_then_step"

    @classmethod
    def derive(
        cls,
        total_optimizer_steps: int,
        *,
        warmup_fraction: float,
        constant_fraction: float,
        linear_decay_fraction: float,
        final_lr_ratio: float,
    ) -> ScheduleContract:
        if type(total_optimizer_steps) is not int or total_optimizer_steps <= 0:
            raise ValueError("total_optimizer_steps must be positive")
        if not math.isclose(
            warmup_fraction + constant_fraction + linear_decay_fraction,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scheduler fractions must sum to one")
        warmup = round(total_optimizer_steps * warmup_fraction)
        constant = round(total_optimizer_steps * constant_fraction)
        decay = total_optimizer_steps - warmup - constant
        if min(warmup, constant, decay) <= 0:
            raise ValueError("every scheduler phase must contain an optimizer step")
        return cls(
            scheduler_type="fncs_encoder_piecewise_linear_v1",
            total_optimizer_steps=total_optimizer_steps,
            warmup_steps=warmup,
            constant_steps=constant,
            linear_decay_steps=decay,
            warmup_fraction=warmup_fraction,
            constant_fraction=constant_fraction,
            linear_decay_fraction=linear_decay_fraction,
            final_lr_ratio=final_lr_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_type": self.scheduler_type,
            "total_optimizer_steps": self.total_optimizer_steps,
            "warmup_steps": self.warmup_steps,
            "constant_steps": self.constant_steps,
            "linear_decay_steps": self.linear_decay_steps,
            "warmup_fraction": self.warmup_fraction,
            "constant_fraction": self.constant_fraction,
            "linear_decay_fraction": self.linear_decay_fraction,
            "final_lr_ratio": self.final_lr_ratio,
            "update_indexing": self.update_indexing,
            "boundaries": {
                "warmup_updates_inclusive": [1, self.warmup_steps],
                "constant_updates_inclusive": [
                    self.warmup_steps + 1,
                    self.warmup_steps + self.constant_steps,
                ],
                "linear_decay_updates_inclusive": [
                    self.warmup_steps + self.constant_steps + 1,
                    self.total_optimizer_steps,
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class OptimizerContract:
    optimizer_type: str
    group_parameter_names: tuple[tuple[str, tuple[str, ...]], ...]
    learning_rate: float
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
            "learning_rate": self.learning_rate,
            "betas": list(self.betas),
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "no_decay_rule": self.no_decay_rule,
        }


def build_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    betas: tuple[float, float],
    epsilon: float,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, OptimizerContract]:
    """Reproduce the verified RotationModel decay/no-decay grouping exactly."""

    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if id(parameter) in seen:
            raise ValueError(f"optimizer parameter is aliased: {name}")
        seen.add(id(parameter))
        if not parameter.requires_grad:
            continue
        excluded = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "relative_lag_bias" in name
        )
        (no_decay if excluded else decay).append((name, parameter))
    if not decay or not no_decay:
        raise ValueError("both verified AdamW parameter groups must be nonempty")
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if trainable != {id(parameter) for _, parameter in (*decay, *no_decay)}:
        raise ValueError("optimizer parameter grouping is not exhaustive")
    groups = [
        {
            "params": [parameter for _, parameter in decay],
            "lr": float(learning_rate),
            "weight_decay": float(weight_decay),
            "group_name": "decay",
        },
        {
            "params": [parameter for _, parameter in no_decay],
            "lr": float(learning_rate),
            "weight_decay": 0.0,
            "group_name": "no_decay",
        },
    ]
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in betas),
        eps=float(epsilon),
    )
    contract = OptimizerContract(
        optimizer_type="AdamW",
        group_parameter_names=(
            ("decay", tuple(name for name, _ in decay)),
            ("no_decay", tuple(name for name, _ in no_decay)),
        ),
        learning_rate=float(learning_rate),
        betas=tuple(float(value) for value in betas),
        epsilon=float(epsilon),
        weight_decay=float(weight_decay),
        no_decay_rule=(
            "parameter.ndim < 2 OR name ends '.bias' OR name contains "
            "'relative_lag_bias'; configuration-equivalent to the verified "
            "RotationModel optimizer"
        ),
    )
    return optimizer, contract


class PiecewiseLinearScheduler:
    """One-indexed warmup, constant, then linear decay to a fixed final LR."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        contract: ScheduleContract,
    ) -> None:
        self.optimizer = optimizer
        self.schedule_contract = contract
        self.peak_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if any(not math.isfinite(value) or value <= 0.0 for value in self.peak_lrs):
            raise ValueError("optimizer groups must begin at positive peak LRs")
        self.step_index = 0
        self._set_for_update(1)

    @property
    def total_steps(self) -> int:
        return self.schedule_contract.total_optimizer_steps

    @property
    def upcoming_update(self) -> int:
        return min(self.step_index + 1, self.total_steps)

    def ratio_at(self, update: int) -> float:
        contract = self.schedule_contract
        if type(update) is not int or not 1 <= update <= contract.total_optimizer_steps:
            raise ValueError(f"scheduler update must be in [1,{contract.total_optimizer_steps}]")
        if update <= contract.warmup_steps:
            return update / contract.warmup_steps
        if update <= contract.warmup_steps + contract.constant_steps:
            return 1.0
        decay_position = update - contract.warmup_steps - contract.constant_steps
        remaining = (contract.linear_decay_steps - decay_position) / contract.linear_decay_steps
        return contract.final_lr_ratio + (1.0 - contract.final_lr_ratio) * remaining

    def stage_at(self, update: int) -> Literal["warmup", "constant", "linear_decay"]:
        self.ratio_at(update)
        contract = self.schedule_contract
        if update <= contract.warmup_steps:
            return "warmup"
        if update <= contract.warmup_steps + contract.constant_steps:
            return "constant"
        return "linear_decay"

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
            for index, (group, peak) in enumerate(zip(self.optimizer.param_groups, self.peak_lrs))
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
            **self.schedule_contract.to_dict(),
            "peak_lrs": list(self.peak_lrs),
        }

    def state_dict(self) -> dict[str, Any]:
        return {**self.contract(), "step_index": self.step_index}

    def load_state_dict(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            raise ValueError("scheduler state must be a mapping")
        expected = self.contract()
        if any(raw.get(name) != value for name, value in expected.items()):
            raise ValueError("scheduler checkpoint contract mismatch")
        step = raw.get("step_index")
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise ValueError("scheduler checkpoint step is invalid")
        self.step_index = step
        self._set_for_update(self.upcoming_update)
