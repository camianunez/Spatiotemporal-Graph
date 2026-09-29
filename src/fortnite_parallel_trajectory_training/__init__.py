"""Isolated training integration for ``parallel_trajectory_decoder_v1``."""

from .config import (
    ARCHITECTURE_ID,
    SCHEDULER_TYPE,
    TARGET_CONTRACT_VERSION,
    TRAINING_CHECKPOINT_SCHEMA,
    TRAINING_CONFIG_SCHEMA,
    ParallelTrajectoryTrainingConfig,
    TrainingCompatibilityError,
    TrainingConfigurationError,
    TrainingNumericalError,
)
from .training import run_training

__all__ = [
    "ARCHITECTURE_ID",
    "ParallelTrajectoryTrainingConfig",
    "SCHEDULER_TYPE",
    "TARGET_CONTRACT_VERSION",
    "TRAINING_CHECKPOINT_SCHEMA",
    "TRAINING_CONFIG_SCHEMA",
    "TrainingCompatibilityError",
    "TrainingConfigurationError",
    "TrainingNumericalError",
    "run_training",
]
