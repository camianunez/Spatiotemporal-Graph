from __future__ import annotations

import math

import torch
from torch import BoolTensor, Tensor, nn

from .config import EncoderConfig
from .contracts import DenseEdges
from .edges import EDGE_FEATURE_WIDTH
from .masking import apply_node_mask, masked_softmax


class SpatialAttentionBlock(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.edge_bias = nn.Sequential(
            nn.Linear(EDGE_FEATURE_WIDTH, config.edge_hidden_dim),
            nn.GELU(),
            nn.Linear(config.edge_hidden_dim, config.n_heads),
        )
        self.edge_value = nn.Linear(EDGE_FEATURE_WIDTH, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(
            config.effective_spatial_attention_dropout
        )
        self.residual_dropout = nn.Dropout(
            config.effective_spatial_residual_dropout
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.effective_spatial_residual_dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )

    def forward(
        self, state: Tensor, edges: DenseEdges, node_mask: BoolTensor
    ) -> Tensor:
        b, t, n, _ = state.shape
        state = apply_node_mask(state, node_mask)
        safe_edge_features = torch.where(
            edges.adjacency.unsqueeze(-1),
            edges.features,
            torch.zeros_like(edges.features),
        )
        normalized = self.attention_norm(state)

        def split_heads(values: Tensor) -> Tensor:
            return values.view(
                b, t, n, self.config.n_heads, self.config.head_dim
            ).permute(0, 1, 3, 2, 4)

        query = split_heads(self.query(normalized))
        key = split_heads(self.key(normalized))
        value = split_heads(self.value(normalized))
        scores = torch.einsum("bthid,bthjd->bthij", query, key)
        scores = scores / math.sqrt(self.config.head_dim)
        scores = scores + self.edge_bias(safe_edge_features).permute(0, 1, 4, 2, 3)
        probabilities = masked_softmax(
            scores, edges.adjacency[:, :, None, :, :], dim=-1
        )
        probabilities = self.attention_dropout(probabilities)

        edge_value = self.edge_value(safe_edge_features).view(
            b, t, n, n, self.config.n_heads, self.config.head_dim
        ).permute(0, 1, 4, 2, 3, 5)
        message = torch.einsum("bthij,bthjd->bthid", probabilities, value)
        message = message + torch.einsum(
            "bthij,bthijd->bthid", probabilities, edge_value
        )
        message = message.permute(0, 1, 3, 2, 4).reshape(
            b, t, n, self.config.d_model
        )
        attention_update = self.output(message)
        has_neighbor = edges.adjacency.any(dim=-1) & node_mask
        attention_update = torch.where(
            has_neighbor.unsqueeze(-1),
            attention_update,
            torch.zeros_like(attention_update),
        )
        state = state + self.residual_dropout(attention_update)
        state = apply_node_mask(state, node_mask)
        state = state + self.residual_dropout(self.ffn(self.ffn_norm(state)))
        return apply_node_mask(state, node_mask)


class SpatialEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SpatialAttentionBlock(config) for _ in range(config.spatial_layers)
        )

    def forward(
        self, state: Tensor, edges: DenseEdges, node_mask: BoolTensor
    ) -> Tensor:
        for layer in self.layers:
            state = layer(state, edges, node_mask)
        return state
