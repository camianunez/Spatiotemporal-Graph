from __future__ import annotations

from torch import nn

from .config import EncoderConfig
from .contracts import EncoderBatch, EncoderOutput
from .edges import build_dense_edges
from .features import FeatureProjector
from .spatial import SpatialEncoder
from .temporal import TemporalEncoder


class SpatiotemporalEncoder(nn.Module):
    """Components 1–4: features, edges, spatial attention, and causal time."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        self.features = FeatureProjector(self.config)
        self.spatial = SpatialEncoder(self.config)
        self.temporal = TemporalEncoder(self.config)

    def forward(self, batch: EncoderBatch) -> EncoderOutput:
        nodes = self.features(batch)
        edges = build_dense_edges(nodes.geometry, self.config)
        spatial_state = self.spatial(
            nodes.initial_state, edges, nodes.geometry.team_alive_mask
        )
        z = self.temporal(
            spatial_state,
            nodes.geometry.team_alive_mask,
            batch.time_mask,
            batch.absolute_tick_index,
        )
        return EncoderOutput(
            z=z,
            team_alive_mask=nodes.geometry.team_alive_mask,
            time_mask=batch.time_mask,
        )

