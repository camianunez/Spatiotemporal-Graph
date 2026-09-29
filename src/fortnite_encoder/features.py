from __future__ import annotations

import torch
from torch import BoolTensor, Tensor, nn
from torch.nn import functional as F

from .config import EncoderConfig
from .contracts import EncoderBatch, NodeFeatures, TeamGeometry, ensure_same_device
from .masking import apply_node_mask


PLAYER_FEATURE_WIDTH = 7
CENTROID_FEATURE_WIDTH = 16
ZONE_FEATURE_WIDTH = 16
MATCH_FEATURE_WIDTH = 4
NODE_FEATURE_WIDTH = 100
NON_PLAYER_FEATURE_WIDTH = (
    CENTROID_FEATURE_WIDTH + ZONE_FEATURE_WIDTH + MATCH_FEATURE_WIDTH
)


def _team_masks(batch: EncoderBatch) -> tuple[BoolTensor, BoolTensor]:
    time_team = batch.time_mask[:, :, None] & batch.team_slot_mask[:, None, :]
    roster = batch.player_slot_mask[:, None, :, :]
    alive = batch.player_alive & roster & time_team[:, :, :, None]
    return time_team, alive


def derive_team_geometry(batch: EncoderBatch, config: EncoderConfig) -> TeamGeometry:
    ensure_same_device(
        batch.player_xyz_uu,
        batch.player_alive,
        batch.player_coord_mask,
        batch.life_index,
    )
    _, alive = _team_masks(batch)
    coord_valid = batch.player_coord_mask & alive
    living_count = alive.sum(dim=-1)
    team_alive = living_count > 0

    every_living_has_coordinates = (
        coord_valid.sum(dim=-1) == living_count
    )
    centroid_valid = team_alive & every_living_has_coordinates
    divisor = living_count.clamp_min(1).unsqueeze(-1).to(batch.player_xyz_uu.dtype)
    centroid = torch.where(
        alive.unsqueeze(-1),
        batch.player_xyz_uu,
        torch.zeros_like(batch.player_xyz_uu),
    ).sum(dim=-2) / divisor
    centroid = torch.where(
        centroid_valid.unsqueeze(-1), centroid, torch.zeros_like(centroid)
    )

    b, t, n, _, _ = batch.player_xyz_uu.shape
    previous_centroid = torch.zeros_like(centroid)
    previous_centroid_valid = torch.zeros_like(centroid_valid)
    previous_alive = torch.zeros_like(alive)
    previous_life = torch.zeros_like(batch.life_index)

    if t > 1:
        previous_centroid[:, 1:] = centroid[:, :-1]
        previous_centroid_valid[:, 1:] = centroid_valid[:, :-1]
        previous_alive[:, 1:] = alive[:, :-1]
        previous_life[:, 1:] = batch.life_index[:, :-1]

    prior_roster = batch.player_slot_mask
    prior_alive = batch.prior_player_alive & prior_roster
    prior_count = prior_alive.sum(dim=-1)
    prior_coord_valid = batch.prior_player_coord_mask & prior_alive
    prior_centroid_valid = (
        (prior_count > 0)
        & (prior_coord_valid.sum(dim=-1) == prior_count)
        & batch.team_slot_mask
        & batch.prior_state_available[:, None]
    )
    prior_divisor = prior_count.clamp_min(1).unsqueeze(-1).to(
        batch.prior_player_xyz_uu.dtype
    )
    prior_centroid = torch.where(
        prior_alive.unsqueeze(-1),
        batch.prior_player_xyz_uu,
        torch.zeros_like(batch.prior_player_xyz_uu),
    ).sum(dim=-2) / prior_divisor
    prior_centroid = torch.where(
        prior_centroid_valid.unsqueeze(-1),
        prior_centroid,
        torch.zeros_like(prior_centroid),
    )
    previous_centroid[:, 0] = prior_centroid
    previous_centroid_valid[:, 0] = prior_centroid_valid
    previous_alive[:, 0] = prior_alive
    previous_life[:, 0] = batch.prior_life_index

    roster = batch.player_slot_mask[:, None, :, :]
    same_membership = ((alive == previous_alive) | ~roster).all(dim=-1)
    same_life = ((batch.life_index == previous_life) | ~roster).all(dim=-1)
    has_previous_step = torch.zeros((b, t, n), dtype=torch.bool, device=centroid.device)
    if t > 1:
        has_previous_step[:, 1:] = (
            batch.time_mask[:, 1:, None] & batch.time_mask[:, :-1, None]
        )
    has_previous_step[:, 0] = batch.prior_state_available[:, None]
    velocity_valid = (
        centroid_valid
        & previous_centroid_valid
        & same_membership
        & same_life
        & has_previous_step
    )
    velocity = (centroid[..., :2] - previous_centroid[..., :2]) / config.xy_scale_uu
    velocity = torch.where(
        velocity_valid.unsqueeze(-1), velocity, torch.zeros_like(velocity)
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    heading_valid = velocity_valid & (speed >= config.epsilon)
    heading_sin = torch.where(
        heading_valid, velocity[..., 1] / speed.clamp_min(config.epsilon), 0.0
    )
    heading_cos = torch.where(
        heading_valid, velocity[..., 0] / speed.clamp_min(config.epsilon), 0.0
    )

    exactly_two = living_count == 2
    separation_valid = exactly_two & (coord_valid.sum(dim=-1) == 2)
    separation = torch.linalg.vector_norm(
        batch.player_xyz_uu[..., 0, :2] - batch.player_xyz_uu[..., 1, :2],
        dim=-1,
    ) / config.xy_scale_uu
    separation = torch.where(separation_valid, separation, torch.zeros_like(separation))

    return TeamGeometry(
        centroid_uu=centroid,
        centroid_valid=centroid_valid,
        team_alive_mask=team_alive,
        living_count=living_count,
        velocity_xy=velocity,
        velocity_valid=velocity_valid,
        speed=speed,
        heading_sin=heading_sin,
        heading_cos=heading_cos,
        heading_valid=heading_valid,
        separation_xy=separation,
        separation_valid=separation_valid,
    )


def build_centroid_features(
    geometry: TeamGeometry, config: EncoderConfig
) -> Tensor:
    centroid_normalized = torch.cat(
        (
            geometry.centroid_uu[..., :2] / config.xy_scale_uu,
            geometry.centroid_uu[..., 2:] / config.z_scale_uu,
        ),
        dim=-1,
    )
    living_one_hot = F.one_hot(
        geometry.living_count.clamp(0, 2), num_classes=3
    ).to(centroid_normalized.dtype)
    return torch.cat(
        (
            centroid_normalized,
            geometry.velocity_xy,
            geometry.speed.unsqueeze(-1),
            geometry.heading_sin.unsqueeze(-1),
            geometry.heading_cos.unsqueeze(-1),
            living_one_hot,
            geometry.separation_xy.unsqueeze(-1),
            geometry.centroid_valid.unsqueeze(-1),
            geometry.velocity_valid.unsqueeze(-1),
            geometry.heading_valid.unsqueeze(-1),
            geometry.separation_valid.unsqueeze(-1),
        ),
        dim=-1,
    ).to(centroid_normalized.dtype)


def _circle_features(
    centroid_uu: Tensor,
    centroid_valid: BoolTensor,
    circle_uu: Tensor,
    zone_mask: BoolTensor,
    config: EncoderConfig,
) -> Tensor:
    circle = circle_uu[:, :, None, :]
    radius = circle[..., 3]
    metric_valid = (
        centroid_valid & zone_mask[:, :, None] & (radius > config.epsilon)
    )
    offsets = (centroid_uu[..., :2] - circle[..., :2]) / radius.clamp_min(
        config.epsilon
    ).unsqueeze(-1)
    offsets = torch.where(
        metric_valid.unsqueeze(-1), offsets, torch.zeros_like(offsets)
    )
    edge_distance = torch.linalg.vector_norm(offsets, dim=-1) - 1.0
    edge_distance = torch.where(
        metric_valid, edge_distance, torch.zeros_like(edge_distance)
    )
    inside = metric_valid & (edge_distance <= 0.0)
    normalized_radius = torch.where(
        zone_mask[:, :, None],
        radius / config.xy_scale_uu,
        torch.zeros_like(radius),
    ).expand_as(metric_valid)
    return torch.cat(
        (
            offsets,
            edge_distance.unsqueeze(-1),
            inside.unsqueeze(-1),
            normalized_radius.unsqueeze(-1),
            metric_valid.unsqueeze(-1),
        ),
        dim=-1,
    ).to(centroid_uu.dtype)


def build_zone_features(
    batch: EncoderBatch, geometry: TeamGeometry, config: EncoderConfig
) -> Tensor:
    current = _circle_features(
        geometry.centroid_uu,
        geometry.centroid_valid,
        batch.current_circle_uu,
        batch.zone_mask,
        config,
    )
    target = _circle_features(
        geometry.centroid_uu,
        geometry.centroid_valid,
        batch.target_circle_uu,
        batch.zone_mask,
        config,
    )
    elapsed = batch.match_elapsed_s
    shrink_start = batch.phase_times_s[..., 1]
    closure = batch.phase_times_s[..., 2]
    until_movement = torch.clamp(shrink_start - elapsed, min=0.0)
    until_closure = torch.clamp(closure - elapsed, min=0.0)
    duration = closure - shrink_start
    progress = torch.clamp(
        (elapsed - shrink_start) / duration.clamp_min(config.epsilon), 0.0, 1.0
    )
    active = batch.zone_mask
    timing = torch.stack(
        (
            torch.where(active, until_movement / config.time_scale_seconds, 0.0),
            torch.where(active, until_closure / config.time_scale_seconds, 0.0),
            torch.where(active & (duration > config.epsilon), progress, 0.0),
            active.to(elapsed.dtype),
        ),
        dim=-1,
    )
    team_count = geometry.centroid_uu.shape[2]
    return torch.cat(
        (current, target, timing[:, :, None, :].expand(-1, -1, team_count, -1)),
        dim=-1,
    )


def build_match_features(
    batch: EncoderBatch, config: EncoderConfig, team_count: int
) -> Tensor:
    roster_players = batch.player_slot_mask.sum(dim=(-1, -2)).clamp_min(1)
    roster_teams = batch.team_slot_mask.sum(dim=-1).clamp_min(1)
    values = torch.stack(
        (
            batch.players_remaining / roster_players[:, None],
            batch.teams_remaining / roster_teams[:, None],
            batch.match_elapsed_s / config.time_scale_seconds,
            batch.zone_phase / config.max_zone_phase,
        ),
        dim=-1,
    ).to(batch.player_xyz_uu.dtype)
    return values[:, :, None, :].expand(-1, -1, team_count, -1)


class FeatureProjector(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.player_mlp = nn.Sequential(
            nn.Linear(PLAYER_FEATURE_WIDTH, config.player_embedding_dim),
            nn.GELU(),
            nn.Linear(config.player_embedding_dim, config.player_embedding_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(
                config.player_embedding_dim + NON_PLAYER_FEATURE_WIDTH,
                config.d_model,
            ),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.time_embedding = nn.Embedding(config.max_timesteps, config.d_model)
        self.phase_embedding = nn.Embedding(
            config.max_zone_phase + 1, config.d_model, padding_idx=0
        )
        self.feature_dropout = nn.Dropout(
            config.effective_feature_projection_dropout
        )

    def _player_pool(
        self, batch: EncoderBatch, geometry: TeamGeometry
    ) -> Tensor:
        _, alive = _team_masks(batch)
        coord = batch.player_coord_mask & alive
        offset_valid = coord & geometry.centroid_valid.unsqueeze(-1)
        offset = (
            batch.player_xyz_uu - geometry.centroid_uu.unsqueeze(-2)
        ) / torch.tensor(
            [
                self.config.xy_scale_uu,
                self.config.xy_scale_uu,
                self.config.z_scale_uu,
            ],
            dtype=batch.player_xyz_uu.dtype,
            device=batch.player_xyz_uu.device,
        )
        offset = torch.where(
            offset_valid.unsqueeze(-1), offset, torch.zeros_like(offset)
        )
        player_features = torch.cat(
            (
                offset,
                coord.unsqueeze(-1),
                offset_valid.unsqueeze(-1),
                torch.log1p(batch.life_index.to(batch.player_xyz_uu.dtype)).unsqueeze(-1),
                (batch.life_index > 0).unsqueeze(-1),
            ),
            dim=-1,
        ).to(batch.player_xyz_uu.dtype)
        player_features = torch.where(
            alive.unsqueeze(-1),
            player_features,
            torch.zeros_like(player_features),
        )
        embedded = self.player_mlp(player_features)
        living_count = geometry.living_count.clamp_min(1).unsqueeze(-1)
        return torch.where(
            alive.unsqueeze(-1), embedded, torch.zeros_like(embedded)
        ).sum(dim=-2) / living_count.to(embedded.dtype)

    def forward(self, batch: EncoderBatch) -> NodeFeatures:
        valid_ticks = batch.absolute_tick_index[batch.time_mask]
        if valid_ticks.numel() and (
            int(valid_ticks.min()) < 0
            or int(valid_ticks.max()) >= self.config.max_timesteps
        ):
            raise ValueError("absolute_tick_index exceeds configured embedding capacity")
        valid_phases = batch.zone_phase[batch.time_mask]
        if valid_phases.numel() and (
            int(valid_phases.min()) < 0
            or int(valid_phases.max()) > self.config.max_zone_phase
        ):
            raise ValueError("zone_phase exceeds configured embedding capacity")

        geometry = derive_team_geometry(batch, self.config)
        player = self._player_pool(batch, geometry)
        centroid = build_centroid_features(geometry, self.config)
        zone = build_zone_features(batch, geometry, self.config)
        match = build_match_features(batch, self.config, geometry.centroid_uu.shape[2])
        complete = torch.cat((player, centroid, zone, match), dim=-1)
        expected_width = self.config.player_embedding_dim + NON_PLAYER_FEATURE_WIDTH
        if complete.shape[-1] != expected_width:
            raise RuntimeError("internal node feature width mismatch")
        safe_ticks = torch.where(
            batch.time_mask,
            batch.absolute_tick_index,
            torch.zeros_like(batch.absolute_tick_index),
        )
        safe_phases = torch.where(
            batch.time_mask,
            batch.zone_phase,
            torch.zeros_like(batch.zone_phase),
        )
        state = (
            self.node_mlp(complete)
            + self.time_embedding(safe_ticks)[:, :, None, :]
            + self.phase_embedding(safe_phases)[:, :, None, :]
        )
        state = self.feature_dropout(state)
        state = apply_node_mask(state, geometry.team_alive_mask)
        return NodeFeatures(complete, state, geometry)
