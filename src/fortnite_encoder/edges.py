from __future__ import annotations

import torch

from .config import EncoderConfig
from .contracts import DenseEdges, TeamGeometry


EDGE_FEATURE_WIDTH = 8


def build_dense_edges(
    geometry: TeamGeometry, config: EncoderConfig
) -> DenseEdges:
    centroid_xy = geometry.centroid_uu[..., :2] / config.xy_scale_uu
    relative = centroid_xy.unsqueeze(-3) - centroid_xy.unsqueeze(-2)
    # Broadcasting above is target j minus source i: [..., i, j, xy].
    distance = torch.linalg.vector_norm(relative, dim=-1)
    nonzero = distance > config.epsilon
    bearing_sin = torch.where(
        nonzero, relative[..., 1] / distance.clamp_min(config.epsilon), 0.0
    )
    bearing_cos = torch.where(
        nonzero, relative[..., 0] / distance.clamp_min(config.epsilon), 0.0
    )

    both_motion_valid = (
        geometry.velocity_valid.unsqueeze(-1)
        & geometry.velocity_valid.unsqueeze(-2)
    )
    delta_velocity = (
        geometry.velocity_xy.unsqueeze(-3) - geometry.velocity_xy.unsqueeze(-2)
    )
    delta_velocity = torch.where(
        both_motion_valid.unsqueeze(-1),
        delta_velocity,
        torch.zeros_like(delta_velocity),
    )
    closing = -(
        relative * delta_velocity
    ).sum(dim=-1) / distance.clamp_min(config.epsilon)
    closing = torch.where(
        both_motion_valid & nonzero, closing, torch.zeros_like(closing)
    )

    n = centroid_xy.shape[-2]
    not_self = ~torch.eye(n, dtype=torch.bool, device=centroid_xy.device)
    valid_pair = (
        geometry.team_alive_mask.unsqueeze(-1)
        & geometry.team_alive_mask.unsqueeze(-2)
        & geometry.centroid_valid.unsqueeze(-1)
        & geometry.centroid_valid.unsqueeze(-2)
        & not_self
    )
    if config.neighbor_mode == "full":
        adjacency = valid_pair
    else:
        masked_distance = torch.where(
            valid_pair, distance, torch.full_like(distance, float("inf"))
        )
        take = min(config.knn_k, n)
        threshold = torch.topk(
            masked_distance, k=take, dim=-1, largest=False
        ).values[..., -1:]
        adjacency = valid_pair & (distance <= threshold)

    features = torch.stack(
        (
            relative[..., 0],
            relative[..., 1],
            distance,
            bearing_sin,
            bearing_cos,
            delta_velocity[..., 0],
            delta_velocity[..., 1],
            closing,
        ),
        dim=-1,
    )
    features = torch.where(
        valid_pair.unsqueeze(-1), features, torch.zeros_like(features)
    )
    return DenseEdges(features=features, adjacency=adjacency)

