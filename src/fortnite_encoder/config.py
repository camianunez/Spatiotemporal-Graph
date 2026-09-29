from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Architecture and fixed, match-independent feature scales."""

    d_model: int = 256
    n_heads: int = 8
    player_embedding_dim: int = 64
    edge_hidden_dim: int = 64
    ffn_dim: int = 1024
    spatial_layers: int = 2
    temporal_layers: int = 4
    dropout: float = 0.1
    feature_projection_dropout: float | None = None
    spatial_attention_dropout: float | None = None
    spatial_residual_dropout: float | None = None
    temporal_attention_dropout: float | None = None
    temporal_residual_dropout: float | None = None
    neighbor_mode: Literal["full", "knn"] = "full"
    knn_k: int = 5
    xy_scale_uu: float = 100_000.0
    z_scale_uu: float = 100_000.0
    time_scale_seconds: float = 1_800.0
    max_timesteps: int = 512
    max_zone_phase: int = 16
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        positive_ints = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "player_embedding_dim": self.player_embedding_dim,
            "edge_hidden_dim": self.edge_hidden_dim,
            "ffn_dim": self.ffn_dim,
            "spatial_layers": self.spatial_layers,
            "temporal_layers": self.temporal_layers,
            "knn_k": self.knn_k,
            "max_timesteps": self.max_timesteps,
            "max_zone_phase": self.max_zone_phase,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        for name, value in {
            "dropout": self.dropout,
            "feature_projection_dropout": self.feature_projection_dropout,
            "spatial_attention_dropout": self.spatial_attention_dropout,
            "spatial_residual_dropout": self.spatial_residual_dropout,
            "temporal_attention_dropout": self.temporal_attention_dropout,
            "temporal_residual_dropout": self.temporal_residual_dropout,
        }.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value < 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1)")
        if self.neighbor_mode not in {"full", "knn"}:
            raise ValueError("neighbor_mode must be 'full' or 'knn'")
        for name, value in {
            "xy_scale_uu": self.xy_scale_uu,
            "z_scale_uu": self.z_scale_uu,
            "time_scale_seconds": self.time_scale_seconds,
            "epsilon": self.epsilon,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def _dropout_or_fallback(self, value: float | None) -> float:
        return float(self.dropout if value is None else value)

    @property
    def effective_feature_projection_dropout(self) -> float:
        return self._dropout_or_fallback(self.feature_projection_dropout)

    @property
    def effective_spatial_attention_dropout(self) -> float:
        return self._dropout_or_fallback(self.spatial_attention_dropout)

    @property
    def effective_spatial_residual_dropout(self) -> float:
        return self._dropout_or_fallback(self.spatial_residual_dropout)

    @property
    def effective_temporal_attention_dropout(self) -> float:
        return self._dropout_or_fallback(self.temporal_attention_dropout)

    @property
    def effective_temporal_residual_dropout(self) -> float:
        return self._dropout_or_fallback(self.temporal_residual_dropout)


@dataclass(frozen=True, slots=True)
class RotationHeadConfig:
    """Configuration for the reusable rotation prediction heads.

    The five-second cadence is part of the dataset contract, so every future
    position horizon must land on an exact sampled tick.
    """

    grid_height: int = 32
    grid_width: int = 32
    horizons_seconds: tuple[int, ...] = (15, 30, 60)
    horizon_embedding_dim: int = 32
    hidden_dim: int = 256
    entry_angle_bins: int = 36
    dropout: float = 0.1
    enable_survival: bool = True
    enable_placement: bool = True

    def __post_init__(self) -> None:
        for name, value in {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "horizon_embedding_dim": self.horizon_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "entry_angle_bins": self.entry_angle_bins,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        horizons = self.horizons_seconds
        if (
            not isinstance(horizons, tuple)
            or not horizons
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value % 5 != 0
                for value in horizons
            )
        ):
            raise ValueError(
                "horizons_seconds must be a nonempty tuple of positive "
                "five-second multiples"
            )
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons_seconds must be unique and strictly increasing")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be in [0, 1)")
        for name, value in {
            "enable_survival": self.enable_survival,
            "enable_placement": self.enable_placement,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def num_horizons(self) -> int:
        return len(self.horizons_seconds)

    @property
    def num_position_cells(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def num_position_classes(self) -> int:
        return self.num_position_cells + 1

    @property
    def eliminated_position_index(self) -> int:
        return self.num_position_cells

    @property
    def already_inside_index(self) -> int:
        return self.entry_angle_bins

    @property
    def eliminated_before_entry_index(self) -> int:
        return self.entry_angle_bins + 1

    @property
    def no_valid_entry_index(self) -> int:
        return self.entry_angle_bins + 2

    @property
    def num_entry_classes(self) -> int:
        return self.entry_angle_bins + 3

    # Readable aliases for callers that describe grids as rows/columns.
    @property
    def grid_rows(self) -> int:
        return self.grid_height

    @property
    def grid_columns(self) -> int:
        return self.grid_width

    @property
    def horizons_s(self) -> tuple[int, ...]:
        return self.horizons_seconds


@dataclass(frozen=True, slots=True)
class RotationLossConfig:
    """Weights for the non-position rotation objectives."""

    entry_weight: float = 1.0
    survival_weight: float = 1.0
    placement_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in {
            "entry_weight": self.entry_weight,
            "survival_weight": self.survival_weight,
            "placement_weight": self.placement_weight,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
