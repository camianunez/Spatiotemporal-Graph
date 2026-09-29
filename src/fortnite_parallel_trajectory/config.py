from __future__ import annotations

from dataclasses import dataclass


PARALLEL_TRAJECTORY_ARCHITECTURE_ID = "parallel_trajectory_decoder_v1"
PARALLEL_TRAJECTORY_TARGET_SCHEMA_VERSION = "1.0"
PARALLEL_TRAJECTORY_TARGET_SCHEMA_ID = (
    f"parallel-trajectory-targets:{PARALLEL_TRAJECTORY_TARGET_SCHEMA_VERSION}"
)

HORIZONS_SECONDS = tuple(range(5, 61, 5))
NUM_HORIZONS = 12
NUM_ROUTE_MODES = 5
D_MODEL = 256
N_HEADS = 8
FFN_WIDTH = 1024
N_LAYERS = 4
DROPOUT = 0.1
ZONE_STATE_WIDTH = 15
CONGESTION_RESIDUAL_SCALE = 0.25


@dataclass(frozen=True, slots=True)
class ParallelTrajectoryConfig:
    """Immutable architecture contract for ``parallel_trajectory_decoder_v1``.

    Version 1 is intentionally fixed.  Fields are exposed for auditability,
    not as architecture search knobs; constructing a different shape requires
    a new architecture identifier.
    """

    architecture_id: str = PARALLEL_TRAJECTORY_ARCHITECTURE_ID
    horizons_seconds: tuple[int, ...] = HORIZONS_SECONDS
    num_route_modes: int = NUM_ROUTE_MODES
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    ffn_width: int = FFN_WIDTH
    n_layers: int = N_LAYERS
    dropout: float = DROPOUT
    zone_state_width: int = ZONE_STATE_WIDTH

    def __post_init__(self) -> None:
        expected = {
            "architecture_id": PARALLEL_TRAJECTORY_ARCHITECTURE_ID,
            "horizons_seconds": HORIZONS_SECONDS,
            "num_route_modes": NUM_ROUTE_MODES,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "ffn_width": FFN_WIDTH,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "zone_state_width": ZONE_STATE_WIDTH,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"{name} is fixed at {value!r} for "
                    f"{PARALLEL_TRAJECTORY_ARCHITECTURE_ID}"
                )

    @property
    def num_horizons(self) -> int:
        return NUM_HORIZONS

    @property
    def congestion_residual_scale(self) -> float:
        return CONGESTION_RESIDUAL_SCALE


__all__ = [
    "CONGESTION_RESIDUAL_SCALE",
    "D_MODEL",
    "DROPOUT",
    "FFN_WIDTH",
    "HORIZONS_SECONDS",
    "N_HEADS",
    "N_LAYERS",
    "NUM_HORIZONS",
    "NUM_ROUTE_MODES",
    "PARALLEL_TRAJECTORY_ARCHITECTURE_ID",
    "PARALLEL_TRAJECTORY_TARGET_SCHEMA_ID",
    "PARALLEL_TRAJECTORY_TARGET_SCHEMA_VERSION",
    "ParallelTrajectoryConfig",
    "ZONE_STATE_WIDTH",
]
