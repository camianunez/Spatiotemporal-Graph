from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import NUM_HORIZONS, NUM_ROUTE_MODES
from .contracts import (
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
    RouteMixtureLossOutput,
)


def _connected_zero(*references: Tensor) -> Tensor:
    if not references:
        raise ValueError("at least one connected-zero reference is required")
    result = references[0].float().sum() * 0.0
    for reference in references[1:]:
        result = result + reference.float().sum() * 0.0
    return result


def route_mixture_nll(
    mode_logits: Tensor | ParallelTrajectoryOutput | None = None,
    means: Tensor | ParallelTrajectoryTargets | None = None,
    scales: Tensor | None = None,
    correlations: Tensor | None = None,
    target_displacements: Tensor | None = None,
    target_mask: Tensor | None = None,
    query_mask: Tensor | None = None,
    *,
    output: ParallelTrajectoryOutput | None = None,
    targets: ParallelTrajectoryTargets | None = None,
) -> RouteMixtureLossOutput:
    """Compute one route-level five-mode bivariate-Gaussian NLL in FP32.

    The public convenience forms are ``route_mixture_nll(output, targets)``
    and the fully explicit six-tensor form.  Horizon log densities are summed
    inside each route mode before the single mode ``logsumexp``.
    """

    if output is not None:
        if mode_logits is not None:
            raise ValueError("provide output or mode_logits, not both")
        mode_logits = output
    if isinstance(mode_logits, ParallelTrajectoryOutput):
        resolved_output = mode_logits
        if isinstance(means, ParallelTrajectoryTargets):
            if targets is not None:
                raise ValueError("targets were provided twice")
            targets = means
            means = None
        if any(value is not None for value in (means, scales, correlations)):
            raise ValueError("distribution tensors must be omitted with an output")
        mode_logits = resolved_output.mode_logits
        means = resolved_output.means
        scales = resolved_output.scales
        correlations = resolved_output.correlations
        if query_mask is None:
            query_mask = resolved_output.query_mask
    if targets is not None:
        if target_displacements is not None or target_mask is not None:
            raise ValueError("target tensors must be omitted with targets")
        target_displacements = targets.target_displacements
        target_mask = targets.target_mask
        if query_mask is None:
            query_mask = targets.query_mask

    tensors = {
        "mode_logits": mode_logits,
        "means": means,
        "scales": scales,
        "correlations": correlations,
        "target_displacements": target_displacements,
        "target_mask": target_mask,
    }
    if any(value is None for value in tensors.values()):
        raise TypeError("all distribution and target tensors are required")
    assert isinstance(mode_logits, Tensor)
    assert isinstance(means, Tensor)
    assert isinstance(scales, Tensor)
    assert isinstance(correlations, Tensor)
    assert isinstance(target_displacements, Tensor)
    assert isinstance(target_mask, Tensor)

    for name in (
        "mode_logits",
        "means",
        "scales",
        "correlations",
        "target_displacements",
    ):
        value = tensors[name]
        assert isinstance(value, Tensor)
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must have bool dtype")
    if mode_logits.ndim != 2 or mode_logits.shape[1] != NUM_ROUTE_MODES:
        raise ValueError("mode_logits must have shape [Q,5]")
    q = mode_logits.shape[0]
    expected = {
        "means": (q, NUM_ROUTE_MODES, NUM_HORIZONS, 2),
        "scales": (q, NUM_ROUTE_MODES, NUM_HORIZONS, 2),
        "correlations": (q, NUM_ROUTE_MODES, NUM_HORIZONS),
        "target_displacements": (q, NUM_HORIZONS, 2),
        "target_mask": (q, NUM_HORIZONS),
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:  # type: ignore[union-attr]
            raise ValueError(f"{name} must have shape {list(shape)}")
    if query_mask is None:
        query_mask = torch.ones(q, dtype=torch.bool, device=mode_logits.device)
    if not isinstance(query_mask, Tensor) or query_mask.dtype != torch.bool:
        raise TypeError("query_mask must have bool dtype")
    if tuple(query_mask.shape) != (q,):
        raise ValueError("query_mask must have shape [Q]")
    all_tensors = (
        mode_logits,
        means,
        scales,
        correlations,
        target_displacements,
        target_mask,
        query_mask,
    )
    if len({value.device for value in all_tensors}) != 1:
        raise ValueError("all route-mixture loss tensors must share a device")
    for name, value in (
        ("mode_logits", mode_logits),
        ("means", means),
        ("scales", scales),
        ("correlations", correlations),
    ):
        if value.numel() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    valid_targets = target_displacements.masked_select(target_mask.unsqueeze(-1))
    if valid_targets.numel() and not bool(torch.isfinite(valid_targets).all()):
        raise ValueError("valid target displacements must be finite")
    if scales.numel() and bool((scales <= 0).any()):
        raise ValueError("scales must be strictly positive")
    if correlations.numel() and bool((correlations.abs() >= 1).any()):
        raise ValueError("correlations must satisfy abs(rho) < 1")

    # Every likelihood-critical operation starts from an explicit FP32 view.
    logits32 = mode_logits.float()
    means32 = means.float()
    scales32 = scales.float()
    correlations32 = correlations.float()
    safe_targets = torch.where(
        target_mask.unsqueeze(-1),
        target_displacements,
        torch.zeros_like(target_displacements),
    ).float()

    residual = safe_targets[:, None, :, :] - means32
    standardized_x = residual[..., 0] / scales32[..., 0]
    standardized_y = residual[..., 1] / scales32[..., 1]
    one_minus_rho_squared = 1.0 - correlations32.square()
    quadratic = (
        standardized_x.square()
        - 2.0 * correlations32 * standardized_x * standardized_y
        + standardized_y.square()
    ) / one_minus_rho_squared
    component_log_density = (
        -math.log(2.0 * math.pi)
        - torch.log(scales32[..., 0])
        - torch.log(scales32[..., 1])
        - 0.5 * torch.log(one_minus_rho_squared)
        - 0.5 * quadratic
    )
    masked_log_density = torch.where(
        target_mask[:, None, :],
        component_log_density,
        torch.zeros_like(component_log_density),
    )
    component_log_likelihood = masked_log_density.sum(dim=-1)
    route_log_likelihood = torch.logsumexp(
        F.log_softmax(logits32, dim=-1) + component_log_likelihood,
        dim=-1,
    )
    raw_route_nll = -route_log_likelihood
    valid_routes = query_mask & target_mask.any(dim=-1)
    per_route_nll = torch.where(
        valid_routes, raw_route_nll, torch.zeros_like(raw_route_nll)
    )
    valid_route_count = int(valid_routes.sum().item())
    valid_horizon_count = int(
        (target_mask & query_mask.unsqueeze(-1)).sum().item()
    )
    if valid_route_count:
        loss = per_route_nll.sum() / valid_route_count
    else:
        loss = _connected_zero(mode_logits, means, scales, correlations)
    if valid_horizon_count:
        nll_per_valid_horizon = per_route_nll.sum() / valid_horizon_count
    else:
        nll_per_valid_horizon = _connected_zero(
            mode_logits, means, scales, correlations
        )
    return RouteMixtureLossOutput(
        loss=loss,
        valid_route_count=valid_route_count,
        valid_horizon_count=valid_horizon_count,
        per_route_nll=per_route_nll,
        nll_per_valid_horizon=nll_per_valid_horizon,
    )


__all__ = ["route_mixture_nll"]
