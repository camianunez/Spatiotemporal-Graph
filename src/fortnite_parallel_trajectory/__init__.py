"""Parallel, non-autoregressive multi-horizon trajectory decoder version 1."""

from .config import (
    HORIZONS_SECONDS,
    PARALLEL_TRAJECTORY_ARCHITECTURE_ID,
    PARALLEL_TRAJECTORY_TARGET_SCHEMA_ID,
    PARALLEL_TRAJECTORY_TARGET_SCHEMA_VERSION,
    ParallelTrajectoryConfig,
)
from .contracts import (
    ParallelTrajectoryBatch,
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
    RouteMixtureLossOutput,
    normalized_displacements_to_world,
)
from .decoder import ParallelTrajectoryDecoder
from .inference import (
    marginal_mean_route,
    primary_mode_route,
    sample_route_modes,
)
from .losses import route_mixture_nll
from .model import ParallelTrajectoryModel
from .targets import build_parallel_trajectory_targets


__version__ = "1.0.0"


__all__ = [
    "HORIZONS_SECONDS",
    "PARALLEL_TRAJECTORY_ARCHITECTURE_ID",
    "PARALLEL_TRAJECTORY_TARGET_SCHEMA_ID",
    "PARALLEL_TRAJECTORY_TARGET_SCHEMA_VERSION",
    "ParallelTrajectoryBatch",
    "ParallelTrajectoryConfig",
    "ParallelTrajectoryDecoder",
    "ParallelTrajectoryModel",
    "ParallelTrajectoryOutput",
    "ParallelTrajectoryTargets",
    "RouteMixtureLossOutput",
    "build_parallel_trajectory_targets",
    "marginal_mean_route",
    "normalized_displacements_to_world",
    "primary_mode_route",
    "route_mixture_nll",
    "sample_route_modes",
]
