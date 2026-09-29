from __future__ import annotations

import math

import torch
from torch import BoolTensor, LongTensor, Tensor, nn

from .config import EncoderConfig
from .masking import apply_node_mask, masked_softmax


class TemporalAttentionBlock(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.relative_lag_bias = nn.Parameter(
            torch.zeros(config.n_heads, config.max_timesteps)
        )
        self.attention_dropout = nn.Dropout(
            config.effective_temporal_attention_dropout
        )
        self.residual_dropout = nn.Dropout(
            config.effective_temporal_residual_dropout
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.effective_temporal_residual_dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )

    def forward(
        self,
        state: Tensor,
        team_alive_mask: BoolTensor,
        time_mask: BoolTensor,
        absolute_tick_index: LongTensor,
    ) -> Tensor:
        # state is [B, N, T, D]; masks arrive in public [B, T, N] order.
        b, n, t, _ = state.shape
        node_mask = (team_alive_mask & time_mask[:, :, None]).transpose(1, 2)
        state = apply_node_mask(state, node_mask)
        normalized = self.attention_norm(state)

        def split_heads(values: Tensor) -> Tensor:
            return values.view(
                b, n, t, self.config.n_heads, self.config.head_dim
            ).permute(0, 1, 3, 2, 4)

        query = split_heads(self.query(normalized))
        key = split_heads(self.key(normalized))
        value = split_heads(self.value(normalized))
        scores = torch.einsum("bnhqd,bnhkd->bnhqk", query, key)
        scores = scores / math.sqrt(self.config.head_dim)

        lag = (
            absolute_tick_index[:, :, None] - absolute_tick_index[:, None, :]
        )
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=state.device))
        valid_lags = lag[time_mask[:, :, None] & time_mask[:, None, :] & causal]
        if valid_lags.numel() and (
            int(valid_lags.min()) < 0
            or int(valid_lags.max()) >= self.config.max_timesteps
        ):
            raise ValueError("relative temporal lag exceeds configured capacity")
        safe_lag = lag.clamp(0, self.config.max_timesteps - 1)
        lag_bias = torch.nn.functional.embedding(
            safe_lag, self.relative_lag_bias.transpose(0, 1)
        ).permute(0, 3, 1, 2)
        scores = scores + lag_bias[:, None, :, :, :]

        key_eligible = node_mask[:, :, None, None, :]
        query_eligible = node_mask[:, :, None, :, None]
        eligible = (
            key_eligible
            & query_eligible
            & causal[None, None, None, :, :]
        )
        probabilities = masked_softmax(scores, eligible, dim=-1)
        probabilities = self.attention_dropout(probabilities)
        message = torch.einsum("bnhqk,bnhkd->bnhqd", probabilities, value)
        message = message.permute(0, 1, 3, 2, 4).reshape(
            b, n, t, self.config.d_model
        )
        state = state + self.residual_dropout(self.output(message))
        state = apply_node_mask(state, node_mask)
        state = state + self.residual_dropout(self.ffn(self.ffn_norm(state)))
        return apply_node_mask(state, node_mask)


class TemporalEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TemporalAttentionBlock(config) for _ in range(config.temporal_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        spatial_state: Tensor,
        team_alive_mask: BoolTensor,
        time_mask: BoolTensor,
        absolute_tick_index: LongTensor,
    ) -> Tensor:
        state = spatial_state.transpose(1, 2)
        for layer in self.layers:
            state = layer(
                state, team_alive_mask, time_mask, absolute_tick_index
            )
        state = self.final_norm(state).transpose(1, 2)
        return apply_node_mask(
            state, team_alive_mask & time_mask[:, :, None]
        )
