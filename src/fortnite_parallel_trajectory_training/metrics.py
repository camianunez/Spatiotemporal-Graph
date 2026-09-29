from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from fortnite_encoder.world_grid import WorldGridProfile
from fortnite_parallel_trajectory.config import NUM_ROUTE_MODES
from fortnite_parallel_trajectory.contracts import (
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
    RouteMixtureLossOutput,
)
from fortnite_parallel_trajectory.losses import route_mixture_nll

from .config import TrainingCompatibilityError


def _component_log_likelihood(
    output: ParallelTrajectoryOutput,
    targets: ParallelTrajectoryTargets,
) -> Tensor:
    means = output.means.float()
    scales = output.scales.float()
    correlations = output.correlations.float()
    mask = targets.target_mask.to(means.device)
    target = torch.where(
        mask.unsqueeze(-1),
        targets.target_displacements.to(means.device),
        torch.zeros_like(targets.target_displacements, device=means.device),
    ).float()
    residual = target[:, None] - means
    x = residual[..., 0] / scales[..., 0]
    y = residual[..., 1] / scales[..., 1]
    one_minus = 1.0 - correlations.square()
    quadratic = (
        x.square() - 2.0 * correlations * x * y + y.square()
    ) / one_minus
    log_density = (
        -math.log(2.0 * math.pi)
        - torch.log(scales[..., 0])
        - torch.log(scales[..., 1])
        - 0.5 * torch.log(one_minus)
        - 0.5 * quadratic
    )
    return torch.where(
        mask[:, None], log_density, torch.zeros_like(log_density)
    ).sum(dim=-1)


class ParallelTrajectoryMetricAccumulator:
    """Count-weighted reporting diagnostics; none of these augment the loss."""

    def __init__(self, profile: WorldGridProfile) -> None:
        if not isinstance(profile, WorldGridProfile):
            raise TypeError("profile must be WorldGridProfile")
        self.profile = profile
        self.route_nll_sum = 0.0
        self.valid_route_count = 0
        self.valid_horizon_count = 0
        self.primary_ade_sum = 0.0
        self.primary_fde_sum = 0.0
        self.marginal_ade_sum = 0.0
        self.marginal_fde_sum = 0.0
        self.oracle_minade_sum = 0.0
        self.oracle_minfde_sum = 0.0
        self.trajectory_query_count = 0
        self.mode_probability_sums = [0.0] * NUM_ROUTE_MODES
        self.posterior_sums = [0.0] * NUM_ROUTE_MODES
        self.mode_entropy_sum = 0.0
        self.effective_mode_count_sum = 0.0
        self.argmax_mode_counts = [0] * NUM_ROUTE_MODES
        self.pairwise_separation_sum = 0.0
        self.mode_diagnostic_query_count = 0
        self.maximum_absolute_normalized_coordinate = 0.0
        self.maximum_scale = 0.0
        self.minimum_scale = math.inf
        self.maximum_absolute_correlation = 0.0
        self.all_outputs_finite = True

    def update(
        self,
        output: ParallelTrajectoryOutput,
        targets: ParallelTrajectoryTargets,
        loss: RouteMixtureLossOutput | None = None,
    ) -> None:
        if not isinstance(output, ParallelTrajectoryOutput):
            raise TypeError("output must be ParallelTrajectoryOutput")
        if not isinstance(targets, ParallelTrajectoryTargets):
            raise TypeError("targets must be ParallelTrajectoryTargets")
        if output.mode_logits.shape[0] != targets.target_mask.shape[0]:
            raise ValueError("output and targets must share query order")
        resolved_loss = loss or route_mixture_nll(output, targets)
        self.route_nll_sum += float(
            resolved_loss.per_route_nll.detach().float().sum().item()
        )
        self.valid_route_count += resolved_loss.valid_route_count
        self.valid_horizon_count += resolved_loss.valid_horizon_count

        means = output.means.detach().float().cpu()
        scales = output.scales.detach().float().cpu()
        correlations = output.correlations.detach().float().cpu()
        probabilities = output.mode_probabilities.detach().float().cpu()
        targets_cpu = targets.target_displacements.detach().float().cpu()
        masks = targets.target_mask.detach().cpu()
        query_masks = (
            targets.query_mask.detach().cpu()
            & output.query_mask.detach().cpu()
            & masks.any(dim=-1)
        )
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (means, scales, correlations, probabilities)
        )
        self.all_outputs_finite = self.all_outputs_finite and finite
        if means.numel():
            self.maximum_absolute_normalized_coordinate = max(
                self.maximum_absolute_normalized_coordinate,
                float(means.abs().max().item()),
            )
            self.maximum_scale = max(self.maximum_scale, float(scales.max().item()))
            self.minimum_scale = min(self.minimum_scale, float(scales.min().item()))
            self.maximum_absolute_correlation = max(
                self.maximum_absolute_correlation,
                float(correlations.abs().max().item()),
            )

        meter_scale = torch.tensor(
            [
                self.profile.cell_width_world_units
                * self.profile.world_unit_scale.meters_per_world_unit,
                self.profile.cell_height_world_units
                * self.profile.world_unit_scale.meters_per_world_unit,
            ],
            dtype=torch.float32,
        )
        primary = output.primary_route_normalized.detach().float().cpu()
        marginal = output.marginal_mean_route_normalized.detach().float().cpu()
        pairs = tuple(
            (left, right)
            for left in range(NUM_ROUTE_MODES)
            for right in range(left + 1, NUM_ROUTE_MODES)
        )
        component_ll = _component_log_likelihood(output, targets).detach().cpu()
        posterior = torch.softmax(
            F.log_softmax(output.mode_logits.detach().float().cpu(), dim=-1)
            + component_ll,
            dim=-1,
        )

        for row in query_masks.nonzero(as_tuple=False).flatten().tolist():
            valid = masks[row]
            last = int(valid.nonzero(as_tuple=False)[-1].item())
            truth = targets_cpu[row]
            primary_distance = torch.linalg.vector_norm(
                (primary[row] - truth) * meter_scale, dim=-1
            )
            marginal_distance = torch.linalg.vector_norm(
                (marginal[row] - truth) * meter_scale, dim=-1
            )
            mode_distance = torch.linalg.vector_norm(
                (means[row] - truth.unsqueeze(0)) * meter_scale,
                dim=-1,
            )
            self.primary_ade_sum += float(primary_distance[valid].mean().item())
            self.primary_fde_sum += float(primary_distance[last].item())
            self.marginal_ade_sum += float(marginal_distance[valid].mean().item())
            self.marginal_fde_sum += float(marginal_distance[last].item())
            self.oracle_minade_sum += float(
                mode_distance[:, valid].mean(dim=-1).min().item()
            )
            self.oracle_minfde_sum += float(mode_distance[:, last].min().item())
            self.trajectory_query_count += 1

            for mode in range(NUM_ROUTE_MODES):
                self.mode_probability_sums[mode] += float(probabilities[row, mode])
                self.posterior_sums[mode] += float(posterior[row, mode])
            entropy = float(
                (-(probabilities[row] * torch.log(probabilities[row].clamp_min(1e-12))).sum()).item()
            )
            self.mode_entropy_sum += entropy
            self.effective_mode_count_sum += math.exp(entropy)
            self.argmax_mode_counts[int(probabilities[row].argmax().item())] += 1
            separations = [
                torch.linalg.vector_norm(
                    (means[row, left] - means[row, right]) * meter_scale,
                    dim=-1,
                )[valid].mean()
                for left, right in pairs
            ]
            self.pairwise_separation_sum += float(torch.stack(separations).mean().item())
            self.mode_diagnostic_query_count += 1

    @staticmethod
    def _mean(numerator: float, count: int) -> float | None:
        return numerator / count if count else None

    def metrics(self) -> dict[str, Any]:
        route_nll = self._mean(self.route_nll_sum, self.valid_route_count)
        nll_per_horizon = self._mean(
            self.route_nll_sum, self.valid_horizon_count
        )
        diagnostic_count = self.mode_diagnostic_query_count
        return {
            "optimization_objective": "route_mixture_nll_only",
            "route_nll": route_nll,
            "nll_per_valid_horizon": nll_per_horizon,
            "valid_route_count": self.valid_route_count,
            "valid_horizon_count": self.valid_horizon_count,
            "primary_mode": {
                "ade_meters": self._mean(
                    self.primary_ade_sum, self.trajectory_query_count
                ),
                "fde_meters": self._mean(
                    self.primary_fde_sum, self.trajectory_query_count
                ),
            },
            "marginal_mean": {
                "ade_meters": self._mean(
                    self.marginal_ade_sum, self.trajectory_query_count
                ),
                "fde_meters": self._mean(
                    self.marginal_fde_sum, self.trajectory_query_count
                ),
            },
            "oracle_diagnostic_only": {
                "minade_meters": self._mean(
                    self.oracle_minade_sum, self.trajectory_query_count
                ),
                "minfde_meters": self._mean(
                    self.oracle_minfde_sum, self.trajectory_query_count
                ),
            },
            "mode_utilization": {
                "mean_probabilities": [
                    value / diagnostic_count if diagnostic_count else None
                    for value in self.mode_probability_sums
                ],
                "mean_posterior_route_responsibilities": [
                    value / diagnostic_count if diagnostic_count else None
                    for value in self.posterior_sums
                ],
                "mean_entropy_nats": self._mean(
                    self.mode_entropy_sum, diagnostic_count
                ),
                "mean_effective_mode_count": self._mean(
                    self.effective_mode_count_sum, diagnostic_count
                ),
                "argmax_mode_counts": list(self.argmax_mode_counts),
                "mean_pairwise_route_separation_meters": self._mean(
                    self.pairwise_separation_sum, diagnostic_count
                ),
                "query_count": diagnostic_count,
            },
            "numerical_diagnostics": {
                "all_outputs_finite": self.all_outputs_finite,
                "minimum_scale": (
                    None if math.isinf(self.minimum_scale) else self.minimum_scale
                ),
                "maximum_scale": self.maximum_scale,
                "maximum_absolute_correlation": self.maximum_absolute_correlation,
                "maximum_absolute_normalized_coordinate": (
                    self.maximum_absolute_normalized_coordinate
                ),
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "profile_hash": self.profile.profile_hash,
            "route_nll_sum": self.route_nll_sum,
            "valid_route_count": self.valid_route_count,
            "valid_horizon_count": self.valid_horizon_count,
            "primary_ade_sum": self.primary_ade_sum,
            "primary_fde_sum": self.primary_fde_sum,
            "marginal_ade_sum": self.marginal_ade_sum,
            "marginal_fde_sum": self.marginal_fde_sum,
            "oracle_minade_sum": self.oracle_minade_sum,
            "oracle_minfde_sum": self.oracle_minfde_sum,
            "trajectory_query_count": self.trajectory_query_count,
            "mode_probability_sums": list(self.mode_probability_sums),
            "posterior_sums": list(self.posterior_sums),
            "mode_entropy_sum": self.mode_entropy_sum,
            "effective_mode_count_sum": self.effective_mode_count_sum,
            "argmax_mode_counts": list(self.argmax_mode_counts),
            "pairwise_separation_sum": self.pairwise_separation_sum,
            "mode_diagnostic_query_count": self.mode_diagnostic_query_count,
            "maximum_absolute_normalized_coordinate": self.maximum_absolute_normalized_coordinate,
            "maximum_scale": self.maximum_scale,
            "minimum_scale": self.minimum_scale,
            "maximum_absolute_correlation": self.maximum_absolute_correlation,
            "all_outputs_finite": self.all_outputs_finite,
        }

    def load_state_dict(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping) or raw.get("profile_hash") != self.profile.profile_hash:
            raise TrainingCompatibilityError("metric accumulator profile mismatch")
        expected = set(self.state_dict())
        if set(raw) != expected:
            raise TrainingCompatibilityError("metric accumulator state keys mismatch")
        try:
            for name in (
                "route_nll_sum",
                "primary_ade_sum",
                "primary_fde_sum",
                "marginal_ade_sum",
                "marginal_fde_sum",
                "oracle_minade_sum",
                "oracle_minfde_sum",
                "mode_entropy_sum",
                "effective_mode_count_sum",
                "pairwise_separation_sum",
                "maximum_absolute_normalized_coordinate",
                "maximum_scale",
                "minimum_scale",
                "maximum_absolute_correlation",
            ):
                setattr(self, name, float(raw[name]))
            for name in (
                "valid_route_count",
                "valid_horizon_count",
                "trajectory_query_count",
                "mode_diagnostic_query_count",
            ):
                setattr(self, name, int(raw[name]))
            self.mode_probability_sums = [float(value) for value in raw["mode_probability_sums"]]
            self.posterior_sums = [float(value) for value in raw["posterior_sums"]]
            self.argmax_mode_counts = [int(value) for value in raw["argmax_mode_counts"]]
            self.all_outputs_finite = bool(raw["all_outputs_finite"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingCompatibilityError(
                f"metric accumulator state is malformed: {exc}"
            ) from exc
        if not (
            len(self.mode_probability_sums)
            == len(self.posterior_sums)
            == len(self.argmax_mode_counts)
            == NUM_ROUTE_MODES
        ):
            raise TrainingCompatibilityError("metric accumulator mode width changed")


__all__ = ["ParallelTrajectoryMetricAccumulator"]
