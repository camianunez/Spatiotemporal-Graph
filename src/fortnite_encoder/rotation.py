from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .config import EncoderConfig, RotationHeadConfig
from .contracts import (
    EncoderBatch,
    EncoderOutput,
    PositionProbabilities,
    RotationLogits,
    RotationModelOutput,
)
from .model import SpatiotemporalEncoder


class _PredictionMlp(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class RotationHeads(nn.Module):
    """Shared future-position head and rotation auxiliary classifiers."""

    def __init__(
        self,
        d_model: int | EncoderConfig | RotationHeadConfig = 256,
        config: RotationHeadConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(d_model, RotationHeadConfig):
            if config is not None:
                raise TypeError("pass RotationHeadConfig only once")
            config = d_model
            d_model = 256
        elif isinstance(d_model, EncoderConfig):
            d_model = d_model.d_model
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")

        self.d_model = d_model
        self.config = config or RotationHeadConfig()
        self.horizon_embedding = nn.Embedding(
            self.config.num_horizons,
            self.config.horizon_embedding_dim,
        )
        self.future_position_head = _PredictionMlp(
            d_model + self.config.horizon_embedding_dim,
            self.config.hidden_dim,
            self.config.num_position_classes,
            self.config.dropout,
        )
        self.zone_entry_head = _PredictionMlp(
            d_model,
            self.config.hidden_dim,
            self.config.num_entry_classes,
            self.config.dropout,
        )
        self.survival_head = (
            _PredictionMlp(
                d_model,
                self.config.hidden_dim,
                1,
                self.config.dropout,
            )
            if self.config.enable_survival
            else None
        )
        self.placement_head = (
            _PredictionMlp(
                d_model,
                self.config.hidden_dim,
                5,
                self.config.dropout,
            )
            if self.config.enable_placement
            else None
        )

    def forward(self, encoder_output: EncoderOutput) -> RotationLogits:
        z = encoder_output.z
        if z.ndim != 4 or z.shape[-1] != self.d_model:
            raise ValueError(
                f"encoder z must have shape [B,T,N,{self.d_model}]"
            )
        if encoder_output.team_alive_mask.shape != z.shape[:-1]:
            raise ValueError("team_alive_mask must match the encoder query axes")
        if encoder_output.time_mask.shape != z.shape[:2]:
            raise ValueError("time_mask must have shape [B,T]")

        query_mask = (
            encoder_output.team_alive_mask
            & encoder_output.time_mask.unsqueeze(-1)
        )
        horizon_ids = torch.arange(
            self.config.num_horizons,
            device=z.device,
        )
        horizon_embedding = self.horizon_embedding(horizon_ids)
        expanded_z = z.unsqueeze(-2).expand(
            *z.shape[:-1],
            self.config.num_horizons,
            self.d_model,
        )
        expanded_embedding = horizon_embedding.view(
            *((1,) * (z.ndim - 1)),
            self.config.num_horizons,
            self.config.horizon_embedding_dim,
        ).expand(
            *z.shape[:-1],
            self.config.num_horizons,
            self.config.horizon_embedding_dim,
        )
        future_position = self.future_position_head(
            torch.cat((expanded_z, expanded_embedding), dim=-1)
        )
        zone_entry = self.zone_entry_head(z)
        survival = (
            self.survival_head(z).squeeze(-1)
            if self.survival_head is not None
            else None
        )
        placement = (
            self.placement_head(z) if self.placement_head is not None else None
        )

        future_position = torch.where(
            query_mask[..., None, None],
            future_position,
            torch.zeros_like(future_position),
        )
        zone_entry = torch.where(
            query_mask[..., None],
            zone_entry,
            torch.zeros_like(zone_entry),
        )
        if survival is not None:
            survival = torch.where(
                query_mask,
                survival,
                torch.zeros_like(survival),
            )
        if placement is not None:
            placement = torch.where(
                query_mask[..., None],
                placement,
                torch.zeros_like(placement),
            )
        return RotationLogits(
            future_position=future_position,
            zone_entry=zone_entry,
            survival=survival,
            placement=placement,
            query_mask=query_mask,
        )


class RotationModel(nn.Module):
    """Convenience composition of the encoder and all rotation heads."""

    def __init__(
        self,
        encoder_config: EncoderConfig | None = None,
        head_config: RotationHeadConfig | None = None,
        *,
        encoder: SpatiotemporalEncoder | None = None,
    ) -> None:
        super().__init__()
        if encoder is not None and encoder_config is not None:
            raise TypeError("pass encoder or encoder_config, not both")
        self.encoder = encoder or SpatiotemporalEncoder(encoder_config)
        self.heads = RotationHeads(
            d_model=self.encoder.config.d_model,
            config=head_config,
        )

    def forward(self, batch: EncoderBatch) -> RotationModelOutput:
        encoded = self.encoder(batch)
        return RotationModelOutput(
            encoder=encoded,
            logits=self.heads(encoded),
        )


def future_position_probabilities(
    logits: Tensor | RotationLogits,
    config: RotationHeadConfig | None = None,
    *,
    grid_height: int | None = None,
    grid_width: int | None = None,
) -> PositionProbabilities:
    """Softmax all classes, then split the map heatmap from elimination."""

    if isinstance(logits, RotationLogits):
        logits = logits.future_position
    resolved = config or RotationHeadConfig()
    height = resolved.grid_height if grid_height is None else grid_height
    width = resolved.grid_width if grid_width is None else grid_width
    for name, value in {"grid_height": height, "grid_width": width}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    expected_classes = height * width + 1
    if logits.ndim < 1 or logits.shape[-1] != expected_classes:
        raise ValueError(
            f"position logits must have {expected_classes} classes on the last axis"
        )
    probabilities = torch.softmax(logits, dim=-1)
    heatmap = probabilities[..., :-1].reshape(
        *probabilities.shape[:-1],
        height,
        width,
    )
    return PositionProbabilities(
        heatmap=cast(torch.FloatTensor, heatmap),
        eliminated=cast(torch.FloatTensor, probabilities[..., -1]),
    )


position_probabilities = future_position_probabilities
