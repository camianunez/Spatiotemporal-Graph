"""Fresh FNCS encoder training orchestration.

This package deliberately lives outside :mod:`fortnite_encoder`.  The latter
is a protected, byte-pinned architecture and data-contract package.
"""

from .config import FreshEncoderConfig, load_config
from .optimization import PiecewiseLinearScheduler, ScheduleContract

__all__ = [
    "FreshEncoderConfig",
    "PiecewiseLinearScheduler",
    "ScheduleContract",
    "load_config",
]
