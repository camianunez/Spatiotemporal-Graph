from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fortnite_encoder import EncoderConfig
from fortnite_encoder.contracts import TeamGeometry
from fortnite_encoder.edges import build_dense_edges
from fortnite_encoder.features import (
    build_centroid_features,
    build_match_features,
    build_zone_features,
    derive_team_geometry,
)


def test_centroid_motion_heading_separation_and_membership(
    encoder_batch, small_config
) -> None:
    geometry = derive_team_geometry(encoder_batch, small_config)
    assert geometry.centroid_uu[0, 0, 0].tolist() == pytest.approx(
        [0.0, 10_000.0, 0.0]
    )
    assert geometry.centroid_uu[0, 1, 0].tolist() == pytest.approx(
        [10_000.0, 10_000.0, 0.0]
    )
    assert geometry.velocity_xy[0, 1, 0].tolist() == pytest.approx([0.1, 0.0])
    assert geometry.speed[0, 1, 0].item() == pytest.approx(0.1)
    assert geometry.heading_sin[0, 1, 0].item() == pytest.approx(0.0)
    assert geometry.heading_cos[0, 1, 0].item() == pytest.approx(1.0)
    assert geometry.separation_xy[0, 1, 0].item() == pytest.approx(0.2)
    assert not geometry.velocity_valid[0, 2, 0]
    assert not geometry.separation_valid[0, 2, 0]
    assert geometry.team_alive_mask[0, 3, 0]
    assert not geometry.centroid_valid[0, 3, 0]
    assert not geometry.velocity_valid[0, 3, 0]


def test_window_first_velocity_matches_full_match(
    tensorized_match, small_config
) -> None:
    from fortnite_encoder import collate_encoder_inputs, slice_window

    full = collate_encoder_inputs([tensorized_match]).batch
    window = collate_encoder_inputs([slice_window(tensorized_match, 1, 2)]).batch
    full_geometry = derive_team_geometry(full, small_config)
    window_geometry = derive_team_geometry(window, small_config)
    assert torch.equal(
        full_geometry.velocity_valid[:, 1],
        window_geometry.velocity_valid[:, 0],
    )
    assert torch.allclose(
        full_geometry.velocity_xy[:, 1],
        window_geometry.velocity_xy[:, 0],
    )


def test_centroid_feature_contract(encoder_batch, small_config) -> None:
    geometry = derive_team_geometry(encoder_batch, small_config)
    features = build_centroid_features(geometry, small_config)
    assert features.shape == (1, 4, 2, 16)
    # centroid xyz, velocity xy, speed, sin, cos, living one-hot
    assert features[0, 1, 0, :11].tolist() == pytest.approx(
        [0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 1.0]
    )


def test_zone_offsets_signed_distance_timing_and_zero_radius(
    encoder_batch, small_config
) -> None:
    geometry = derive_team_geometry(encoder_batch, small_config)
    features = build_zone_features(encoder_batch, geometry, small_config)
    assert features.shape == (1, 4, 2, 16)
    current = features[0, 1, 0, :6]
    assert current.tolist() == pytest.approx(
        [0.1, 0.1, 2**0.5 / 10 - 1, 1.0, 1.0, 1.0]
    )
    target = features[0, 1, 0, 6:12]
    assert target.tolist() == pytest.approx(
        [-0.2, 0.2, (0.08**0.5) - 1, 1.0, 0.5, 1.0]
    )
    assert features[0, 1, 0, 12:].tolist() == pytest.approx(
        [5 / 1800, 15 / 1800, 0.0, 1.0]
    )
    assert features[0, 3, 1, 14].item() == pytest.approx(0.5)

    zero_target = encoder_batch.target_circle_uu.clone()
    zero_target[:, 1, 3] = 0.0
    changed = replace(encoder_batch, target_circle_uu=zero_target)
    zero_features = build_zone_features(changed, geometry, small_config)
    assert torch.equal(
        zero_features[:, 1, :, 6:12],
        torch.zeros_like(zero_features[:, 1, :, 6:12]),
    )


def test_match_count_normalization(encoder_batch, small_config) -> None:
    values = build_match_features(encoder_batch, small_config, team_count=2)
    assert values[0, 2, 0].tolist() == pytest.approx(
        [3 / 4, 1.0, 10 / 1800, 1 / 4]
    )


def test_edge_bearing_and_positive_closing_speed(
    encoder_batch, small_config
) -> None:
    geometry = derive_team_geometry(encoder_batch, small_config)
    edges = build_dense_edges(geometry, small_config)
    edge = edges.features[0, 1, 0, 1]
    assert edge.tolist() == pytest.approx(
        [0.8, 0.0, 0.8, 0.0, 1.0, -0.2, 0.0, 0.2]
    )
    assert edges.adjacency[0, 1].tolist() == [
        [False, True],
        [True, False],
    ]
    assert torch.equal(
        edges.features[0, 2, 0, 1, 5:],
        torch.zeros(3),
    )


def _geometry_at_x(x: list[float]) -> TeamGeometry:
    centroid = torch.tensor(x, dtype=torch.float32).view(1, 1, -1, 1)
    centroid = torch.cat(
        (centroid, torch.zeros_like(centroid), torch.zeros_like(centroid)), dim=-1
    )
    shape = centroid.shape[:-1]
    valid = torch.ones(shape, dtype=torch.bool)
    zeros = torch.zeros((*shape, 2), dtype=torch.float32)
    scalar_zeros = torch.zeros(shape, dtype=torch.float32)
    return TeamGeometry(
        centroid_uu=centroid,
        centroid_valid=valid,
        team_alive_mask=valid,
        living_count=torch.ones(shape, dtype=torch.int64),
        velocity_xy=zeros,
        velocity_valid=torch.zeros(shape, dtype=torch.bool),
        speed=scalar_zeros,
        heading_sin=scalar_zeros,
        heading_cos=scalar_zeros,
        heading_valid=torch.zeros(shape, dtype=torch.bool),
        separation_xy=scalar_zeros,
        separation_valid=torch.zeros(shape, dtype=torch.bool),
    )


def test_knn_is_tie_inclusive_and_permutation_safe() -> None:
    geometry = _geometry_at_x([0.0, -10_000.0, 10_000.0, 30_000.0])
    config = EncoderConfig(
        neighbor_mode="knn",
        knn_k=1,
        spatial_layers=1,
        temporal_layers=1,
        dropout=0.0,
    )
    edges = build_dense_edges(geometry, config)
    assert edges.adjacency[0, 0, 0].tolist() == [False, True, True, False]

    permutation = torch.tensor([0, 2, 1, 3])
    permuted = replace(
        geometry,
        centroid_uu=geometry.centroid_uu[:, :, permutation],
        centroid_valid=geometry.centroid_valid[:, :, permutation],
        team_alive_mask=geometry.team_alive_mask[:, :, permutation],
        living_count=geometry.living_count[:, :, permutation],
        velocity_xy=geometry.velocity_xy[:, :, permutation],
        velocity_valid=geometry.velocity_valid[:, :, permutation],
        speed=geometry.speed[:, :, permutation],
        heading_sin=geometry.heading_sin[:, :, permutation],
        heading_cos=geometry.heading_cos[:, :, permutation],
        heading_valid=geometry.heading_valid[:, :, permutation],
        separation_xy=geometry.separation_xy[:, :, permutation],
        separation_valid=geometry.separation_valid[:, :, permutation],
    )
    permuted_edges = build_dense_edges(permuted, config)
    expected = edges.adjacency[:, :, permutation][:, :, :, permutation]
    assert torch.equal(permuted_edges.adjacency, expected)


def test_colocated_edge_has_zero_bearing_and_closing() -> None:
    geometry = _geometry_at_x([0.0, 0.0])
    config = EncoderConfig(spatial_layers=1, temporal_layers=1, dropout=0.0)
    edge = build_dense_edges(geometry, config).features[0, 0, 0, 1]
    assert edge[2:].tolist() == pytest.approx([0.0] * 6)

