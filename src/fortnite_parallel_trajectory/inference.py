from __future__ import annotations

import torch
from torch import Tensor

from .config import NUM_HORIZONS, NUM_ROUTE_MODES
from .contracts import ParallelTrajectoryOutput


def _validate_modes(mode_probabilities: Tensor, means: Tensor) -> int:
    if not isinstance(mode_probabilities, Tensor) or not mode_probabilities.is_floating_point():
        raise TypeError("mode_probabilities must be a floating-point tensor")
    if not isinstance(means, Tensor) or not means.is_floating_point():
        raise TypeError("means must be a floating-point tensor")
    if mode_probabilities.ndim != 2 or mode_probabilities.shape[1] != NUM_ROUTE_MODES:
        raise ValueError("mode_probabilities must have shape [Q,5]")
    q = mode_probabilities.shape[0]
    if tuple(means.shape) != (q, NUM_ROUTE_MODES, NUM_HORIZONS, 2):
        raise ValueError("means must have shape [Q,5,12,2]")
    if mode_probabilities.device != means.device:
        raise ValueError("mode probabilities and means must share a device")
    if mode_probabilities.numel():
        if not bool(torch.isfinite(mode_probabilities).all()):
            raise ValueError("mode probabilities must be finite")
        if bool((mode_probabilities < 0).any()):
            raise ValueError("mode probabilities must be nonnegative")
        sums = mode_probabilities.float().sum(dim=-1)
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)):
            raise ValueError("mode probabilities must sum to one")
    if means.numel() and not bool(torch.isfinite(means).all()):
        raise ValueError("means must be finite")
    return q


def primary_mode_route(mode_probabilities: Tensor, means: Tensor) -> Tensor:
    """Return the highest-probability coherent route.

    ``torch.argmax`` returns the first maximum, so exact ties deterministically
    resolve to the lower mode index without perturbing the probabilities.
    """

    q = _validate_modes(mode_probabilities, means)
    indices = torch.argmax(mode_probabilities, dim=-1)
    rows = torch.arange(q, device=means.device)
    return means[rows, indices]


def marginal_mean_route(mode_probabilities: Tensor, means: Tensor) -> Tensor:
    """Return the mixture marginal mean, which is not itself a route mode."""

    _validate_modes(mode_probabilities, means)
    return (mode_probabilities[:, :, None, None] * means).sum(dim=1)


def sample_route_modes(
    mode_probabilities: Tensor | ParallelTrajectoryOutput,
    means: Tensor | None = None,
    scales: Tensor | None = None,
    correlations: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample one mode per query and its complete twelve-horizon route.

    Returns ``(sampled_normalized_route, sampled_mode_index)``.  A single mode
    index is drawn for each query and reused at every horizon.
    """

    if isinstance(mode_probabilities, ParallelTrajectoryOutput):
        if any(value is not None for value in (means, scales, correlations)):
            raise ValueError("tensor arguments must be omitted when sampling an output")
        output = mode_probabilities
        mode_probabilities = output.mode_probabilities
        means = output.means
        scales = output.scales
        correlations = output.correlations
    if means is None or scales is None or correlations is None:
        raise TypeError("means, scales, and correlations are required")
    q = _validate_modes(mode_probabilities, means)
    expected_scales = (q, NUM_ROUTE_MODES, NUM_HORIZONS, 2)
    expected_correlations = (q, NUM_ROUTE_MODES, NUM_HORIZONS)
    if not isinstance(scales, Tensor) or not scales.is_floating_point():
        raise TypeError("scales must be a floating-point tensor")
    if not isinstance(correlations, Tensor) or not correlations.is_floating_point():
        raise TypeError("correlations must be a floating-point tensor")
    if tuple(scales.shape) != expected_scales:
        raise ValueError("scales must have shape [Q,5,12,2]")
    if tuple(correlations.shape) != expected_correlations:
        raise ValueError("correlations must have shape [Q,5,12]")
    if scales.device != means.device or correlations.device != means.device:
        raise ValueError("all distribution tensors must share a device")
    if scales.numel() and (
        not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any())
    ):
        raise ValueError("scales must be finite and strictly positive")
    if correlations.numel() and (
        not bool(torch.isfinite(correlations).all())
        or bool((correlations.abs() >= 1).any())
    ):
        raise ValueError("correlations must be finite with abs(rho) < 1")
    if generator is not None and seed is not None:
        raise ValueError("provide either generator or seed, not both")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        generator = torch.Generator(device=mode_probabilities.device)
        generator.manual_seed(seed)

    if q == 0:
        return (
            means.new_empty((0, NUM_HORIZONS, 2)),
            torch.empty((0,), dtype=torch.int64, device=means.device),
        )
    indices = torch.multinomial(
        mode_probabilities,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(-1)
    rows = torch.arange(q, device=means.device)
    selected_means = means[rows, indices]
    selected_scales = scales[rows, indices]
    selected_correlations = correlations[rows, indices]
    standard = torch.randn(
        (q, NUM_HORIZONS, 2),
        dtype=selected_means.dtype,
        device=selected_means.device,
        generator=generator,
    )
    x = selected_means[..., 0] + selected_scales[..., 0] * standard[..., 0]
    orthogonal = torch.sqrt(
        torch.clamp(1.0 - selected_correlations.square(), min=0.0)
    )
    y = selected_means[..., 1] + selected_scales[..., 1] * (
        selected_correlations * standard[..., 0] + orthogonal * standard[..., 1]
    )
    return torch.stack((x, y), dim=-1), indices


__all__ = [
    "marginal_mean_route",
    "primary_mode_route",
    "sample_route_modes",
]
