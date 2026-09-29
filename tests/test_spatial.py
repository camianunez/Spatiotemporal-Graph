from __future__ import annotations

from dataclasses import replace

import torch

from fortnite_encoder.contracts import DenseEdges
from fortnite_encoder.features import FeatureProjector
from fortnite_encoder.spatial import SpatialAttentionBlock, SpatialEncoder


def test_spatial_shapes_empty_neighbors_and_dead_zero(small_config) -> None:
    block = SpatialAttentionBlock(small_config).eval()
    state = torch.randn(1, 2, 3, small_config.d_model)
    node_mask = torch.tensor([[[True, False, True], [True, True, False]]])
    edges = DenseEdges(
        features=torch.zeros(1, 2, 3, 3, 8),
        adjacency=torch.zeros(1, 2, 3, 3, dtype=torch.bool),
    )
    output = block(state, edges, node_mask)
    assert output.shape == state.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output[~node_mask], torch.zeros_like(output[~node_mask]))


def test_spatial_encoder_uses_independent_layers(small_config) -> None:
    config = replace(small_config, spatial_layers=2)
    encoder = SpatialEncoder(config)
    assert len(encoder.layers) == 2
    assert encoder.layers[0].query.weight is not encoder.layers[1].query.weight


def test_player_slot_permutation_leaves_projected_nodes_unchanged(
    encoder_batch, small_config
) -> None:
    projector = FeatureProjector(small_config).eval()
    original = projector(encoder_batch)
    permutation = torch.tensor([1, 0])
    swapped = replace(
        encoder_batch,
        player_xyz_uu=encoder_batch.player_xyz_uu[:, :, :, permutation],
        player_alive=encoder_batch.player_alive[:, :, :, permutation],
        player_coord_mask=encoder_batch.player_coord_mask[:, :, :, permutation],
        life_index=encoder_batch.life_index[:, :, :, permutation],
        player_slot_mask=encoder_batch.player_slot_mask[:, :, permutation],
        prior_player_xyz_uu=encoder_batch.prior_player_xyz_uu[:, :, permutation],
        prior_player_alive=encoder_batch.prior_player_alive[:, :, permutation],
        prior_player_coord_mask=encoder_batch.prior_player_coord_mask[:, :, permutation],
        prior_life_index=encoder_batch.prior_life_index[:, :, permutation],
    )
    permuted = projector(swapped)
    assert torch.allclose(
        original.initial_state, permuted.initial_state, atol=1e-6
    )

