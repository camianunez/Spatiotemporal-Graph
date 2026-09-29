from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# These constants are the repository's already-validated covariance contract.
# Version 1 deliberately reuses them instead of defining a new scale floor or
# correlation boundary.
from fortnite_early_zone.decoder import (
    MAX_CORRELATION_MAGNITUDE,
    MIN_COMPONENT_SCALE,
)

from .config import (
    CONGESTION_RESIDUAL_SCALE,
    D_MODEL,
    DROPOUT,
    FFN_WIDTH,
    N_HEADS,
    N_LAYERS,
    NUM_HORIZONS,
    NUM_ROUTE_MODES,
    ParallelTrajectoryConfig,
)
from .contracts import ParallelTrajectoryBatch, ParallelTrajectoryOutput
from .inference import marginal_mean_route, primary_mode_route


class ParallelTrajectoryDecoderLayer(nn.Module):
    """One independent, bidirectional pre-LayerNorm decoder layer."""

    congestion_residual_scale = CONGESTION_RESIDUAL_SCALE

    def __init__(self) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(D_MODEL)
        self.self_attention = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True
        )

        self.memory_attention_norm = nn.LayerNorm(D_MODEL)
        self.memory_attention = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True
        )

        self.congestion_attention_norm = nn.LayerNorm(D_MODEL)
        self.congestion_attention = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True
        )

        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_WIDTH, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(
        self,
        state: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        congestion_tokens: Tensor,
    ) -> Tensor:
        normalized = self.self_attention_norm(state)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        state = state + update

        normalized = self.memory_attention_norm(state)
        update, _ = self.memory_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        state = state + update

        normalized = self.congestion_attention_norm(state)
        update, _ = self.congestion_attention(
            normalized,
            congestion_tokens,
            congestion_tokens,
            need_weights=False,
        )
        state = state + self.congestion_residual_scale * update
        return state + self.ffn(self.ffn_norm(state))


class ParallelTrajectoryDecoder(nn.Module):
    """Jointly predict five coherent complete routes over twelve horizons."""

    architecture_id = "parallel_trajectory_decoder_v1"
    d_model = D_MODEL
    n_heads = N_HEADS
    ffn_width = FFN_WIDTH
    n_layers = N_LAYERS
    dropout = DROPOUT
    num_horizons = NUM_HORIZONS
    num_route_modes = NUM_ROUTE_MODES
    congestion_residual_scale = CONGESTION_RESIDUAL_SCALE
    min_component_scale = MIN_COMPONENT_SCALE
    max_correlation_magnitude = MAX_CORRELATION_MAGNITUDE

    def __init__(self, config: ParallelTrajectoryConfig | None = None) -> None:
        super().__init__()
        self.config = config or ParallelTrajectoryConfig()
        self.horizon_embedding = nn.Embedding(NUM_HORIZONS, D_MODEL)
        self.query_projection = nn.Linear(D_MODEL, D_MODEL)
        self.zone_projection = nn.Linear(self.config.zone_state_width, D_MODEL)
        self.layers = nn.ModuleList(
            ParallelTrajectoryDecoderLayer() for _ in range(N_LAYERS)
        )
        self.final_norm = nn.LayerNorm(D_MODEL)

        self.route_norm = nn.LayerNorm(D_MODEL)
        self.mode_head = nn.Linear(D_MODEL, NUM_ROUTE_MODES)
        self.mean_head = nn.Linear(D_MODEL, NUM_ROUTE_MODES * 2)
        self.scale_head = nn.Linear(D_MODEL, NUM_ROUTE_MODES * 2)
        self.correlation_head = nn.Linear(D_MODEL, NUM_ROUTE_MODES)

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.layers

    @property
    def horizon_embeddings(self) -> nn.Embedding:
        return self.horizon_embedding

    def forward(
        self,
        batch: ParallelTrajectoryBatch,
        *,
        horizon_mask: Tensor | None = None,
    ) -> ParallelTrajectoryOutput:
        if not isinstance(batch, ParallelTrajectoryBatch):
            raise TypeError("batch must be a ParallelTrajectoryBatch")
        q = batch.z_query.shape[0]
        if horizon_mask is not None:
            if not isinstance(horizon_mask, Tensor) or horizon_mask.dtype != torch.bool:
                raise TypeError("horizon_mask must have bool dtype")
            if tuple(horizon_mask.shape) != (q, NUM_HORIZONS):
                raise ValueError("horizon_mask must have shape [Q,12]")
            if horizon_mask.device != batch.z_query.device:
                raise ValueError("horizon_mask must share the decoder device")

        horizon_ids = torch.arange(NUM_HORIZONS, device=batch.z_query.device)
        state = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(q, -1, -1)
        state = (
            state
            + self.query_projection(batch.z_query).unsqueeze(1)
            + self.zone_projection(batch.zone_state.to(batch.z_query.dtype)).unsqueeze(1)
        )
        if q:
            for layer in self.layers:
                state = layer(
                    state,
                    batch.memory,
                    batch.memory_mask,
                    batch.congestion_tokens,
                )
        state = self.final_norm(state)

        route_state = self.route_norm(state.mean(dim=1))
        mode_logits = self.mode_head(route_state)
        mode_probabilities = torch.softmax(mode_logits, dim=-1)

        means = self.mean_head(state).reshape(
            q, NUM_HORIZONS, NUM_ROUTE_MODES, 2
        ).permute(0, 2, 1, 3).contiguous()
        raw_scales = self.scale_head(state).reshape(
            q, NUM_HORIZONS, NUM_ROUTE_MODES, 2
        ).permute(0, 2, 1, 3).contiguous()
        scales = F.softplus(raw_scales) + MIN_COMPONENT_SCALE
        correlations = (
            MAX_CORRELATION_MAGNITUDE
            * torch.tanh(
                self.correlation_head(state)
                .permute(0, 2, 1)
                .contiguous()
            )
        )
        primary_indices = torch.argmax(mode_probabilities, dim=-1)
        primary = primary_mode_route(mode_probabilities, means)
        marginal = marginal_mean_route(mode_probabilities, means)
        return ParallelTrajectoryOutput(
            mode_logits=mode_logits,
            mode_probabilities=mode_probabilities,
            means=means,
            scales=scales,
            correlations=correlations,
            primary_mode_index=primary_indices,
            primary_route_normalized=primary,
            marginal_mean_route_normalized=marginal,
            query_mask=batch.query_mask.clone(),
            horizon_mask=None if horizon_mask is None else horizon_mask.clone(),
            current_xy=batch.current_xy.clone(),
        )


__all__ = [
    "MAX_CORRELATION_MAGNITUDE",
    "MIN_COMPONENT_SCALE",
    "ParallelTrajectoryDecoder",
    "ParallelTrajectoryDecoderLayer",
]
