from __future__ import annotations

import torch
from torch import Tensor, nn

from fortnite_encoder.contracts import EncoderBatch, EncoderOutput
from fortnite_encoder.model import SpatiotemporalEncoder
from fortnite_encoder.planner import (
    CongestionPredictor,
    CongestionTokenizer,
    ZONE_VECTOR_WIDTH,
)
from fortnite_encoder.planner_contracts import PlannerBatch, PlannerObservation
from fortnite_encoder.planner_policy import apply_planner_input_policy
from fortnite_encoder.world_grid import WorldGridProfile

from .config import D_MODEL
from .contracts import (
    ParallelTrajectoryBatch,
    ParallelTrajectoryOutput,
    ParallelTrajectoryTargets,
)
from .decoder import ParallelTrajectoryDecoder


class ParallelTrajectoryModel(nn.Module):
    """Compose the unchanged causal pipeline with the parallel decoder.

    No constructor performs checkpoint I/O.  Callers may inject already-loaded
    encoder, congestion-predictor, and congestion-tokenizer instances.
    """

    def __init__(
        self,
        world_grid_profile: WorldGridProfile | SpatiotemporalEncoder,
        encoder: nn.Module | WorldGridProfile | None = None,
        *,
        congestion_predictor: nn.Module | None = None,
        congestion_tokenizer: nn.Module | None = None,
        decoder: ParallelTrajectoryDecoder | None = None,
    ) -> None:
        super().__init__()
        if isinstance(world_grid_profile, SpatiotemporalEncoder):
            world_grid_profile, encoder = encoder, world_grid_profile
        if not isinstance(world_grid_profile, WorldGridProfile):
            raise TypeError("world_grid_profile must be a WorldGridProfile")
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an injected torch module")
        encoder_config = getattr(encoder, "config", None)
        if encoder_config is not None and getattr(encoder_config, "d_model", D_MODEL) != D_MODEL:
            raise ValueError("the parallel decoder requires encoder d_model=256")
        self.world_grid_profile = world_grid_profile
        self.encoder = encoder
        self.congestion_predictor = (
            CongestionPredictor()
            if congestion_predictor is None
            else congestion_predictor
        )
        self.congestion_tokenizer = (
            CongestionTokenizer()
            if congestion_tokenizer is None
            else congestion_tokenizer
        )
        self.decoder = ParallelTrajectoryDecoder() if decoder is None else decoder

    def sanitize(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
    ) -> PlannerBatch:
        return apply_planner_input_policy(batch, observation)

    def _normalize_circle(self, circle: Tensor, active: Tensor) -> Tensor:
        profile = self.world_grid_profile
        x_span = float(profile.world_x_max - profile.world_x_min)
        y_span = float(profile.world_y_max - profile.world_y_min)
        world_scale = max(x_span, y_span)
        normalized = torch.stack(
            (
                (circle[:, 0] - profile.world_x_min) / x_span,
                (circle[:, 1] - profile.world_y_min) / y_span,
                circle[:, 2] / world_scale,
                circle[:, 3] / world_scale,
            ),
            dim=-1,
        )
        return torch.where(active.unsqueeze(-1), normalized, torch.zeros_like(normalized))

    def _zone_state(
        self,
        planner_batch: PlannerBatch,
        query_indices: Tensor,
    ) -> Tensor:
        batch = planner_batch.encoder_batch
        if query_indices.shape[0] == 0:
            return batch.player_xyz_uu.new_empty((0, ZONE_VECTOR_WIDTH))
        batch_index, time_index = query_indices[:, 0], query_indices[:, 1]
        current_active = batch.zone_mask[batch_index, time_index]
        target_revealed = planner_batch.revealed_target_zone_mask[
            batch_index, time_index
        ]
        current = self._normalize_circle(
            batch.current_circle_uu[batch_index, time_index], current_active
        )
        target = self._normalize_circle(
            batch.target_circle_uu[batch_index, time_index], target_revealed
        )
        dtype = current.dtype
        timing = batch.phase_times_s[batch_index, time_index].to(dtype) / 1_800.0
        elapsed = (
            batch.match_elapsed_s[batch_index, time_index].to(dtype).unsqueeze(-1)
            / 1_800.0
        )
        phase = (
            batch.zone_phase[batch_index, time_index].to(dtype).unsqueeze(-1)
            / 16.0
        )
        return torch.cat(
            (
                current,
                target,
                timing,
                elapsed,
                phase,
                current_active.to(dtype).unsqueeze(-1),
                target_revealed.to(dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _current_centroids(
        self,
        planner_batch: PlannerBatch,
        query_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = planner_batch.encoder_batch
        q = query_indices.shape[0]
        output = batch.player_xyz_uu.new_zeros((q, 2))
        eligible = torch.zeros(q, dtype=torch.bool, device=output.device)
        profile = self.world_grid_profile
        for row, query in enumerate(query_indices):
            batch_index, time_index, team_index = (int(value) for value in query)
            roster = batch.player_slot_mask[batch_index, team_index]
            valid = (
                batch.player_alive[batch_index, time_index, team_index]
                & batch.player_coord_mask[batch_index, time_index, team_index]
                & roster
            )
            phase = int(batch.zone_phase[batch_index, time_index].item())
            if not bool(valid.any()) or not 1 <= phase <= 8:
                continue
            coordinates = batch.player_xyz_uu[
                batch_index, time_index, team_index, valid, :2
            ]
            if not bool(torch.isfinite(coordinates).all()):
                continue
            centroid = coordinates.double().mean(dim=0)
            x, y = float(centroid[0]), float(centroid[1])
            output[row] = centroid.to(output.dtype)
            eligible[row] = (
                profile.world_x_min <= x <= profile.world_x_max
                and profile.world_y_min <= y <= profile.world_y_max
            )
        return output, eligible

    def _causal_memory(
        self,
        planner_batch: PlannerBatch,
        encoder_output: EncoderOutput,
        query_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z = encoder_output.z
        valid_nodes = (
            encoder_output.team_alive_mask
            & encoder_output.time_mask.unsqueeze(-1)
        )
        q = query_indices.shape[0]
        if q == 0:
            return (
                z.new_empty((0, 1, D_MODEL)),
                torch.zeros((0, 1), dtype=torch.bool, device=z.device),
            )
        memories: list[Tensor] = []
        for query in query_indices:
            batch_index = int(query[0])
            time_index = int(query[1])
            eligible = valid_nodes[batch_index].clone()
            eligible[time_index + 1 :] = False
            values = z[batch_index][eligible]
            if values.shape[0] == 0:
                raise ValueError("each query requires causal encoder memory")
            memories.append(values)
        width = max(values.shape[0] for values in memories)
        memory = z.new_zeros((q, width, D_MODEL))
        memory_mask = torch.zeros((q, width), dtype=torch.bool, device=z.device)
        for row, values in enumerate(memories):
            memory[row, : values.shape[0]] = values
            memory_mask[row, : values.shape[0]] = True
        return memory, memory_mask

    def prepare(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
    ) -> ParallelTrajectoryBatch:
        planner_batch = self.sanitize(batch, observation)
        encoder_output = self.encoder(planner_batch.encoder_batch)
        if not isinstance(encoder_output, EncoderOutput):
            raise TypeError("encoder must return EncoderOutput")
        z = encoder_output.z
        if z.ndim != 4 or z.shape[-1] != D_MODEL:
            raise ValueError("encoder z must have shape [B,T,N,256]")
        if tuple(planner_batch.query_mask.shape) != tuple(z.shape[:-1]):
            raise ValueError("planner query mask must align with encoder output")
        if tuple(encoder_output.team_alive_mask.shape) != tuple(z.shape[:-1]):
            raise ValueError("encoder team_alive_mask must align with z")
        if tuple(encoder_output.time_mask.shape) != tuple(z.shape[:2]):
            raise ValueError("encoder time_mask must have shape [B,T]")
        if not torch.equal(
            encoder_output.time_mask, planner_batch.encoder_batch.time_mask
        ):
            raise ValueError("encoder output must come from the sanitized batch")

        query_indices = planner_batch.query_mask.nonzero(as_tuple=False)
        query_state = z[planner_batch.query_mask]
        current_xy, coordinate_eligible = self._current_centroids(
            planner_batch, query_indices
        )
        memory, memory_mask = self._causal_memory(
            planner_batch, encoder_output, query_indices
        )
        zone_state = self._zone_state(planner_batch, query_indices).to(query_state.dtype)
        density_result = self.congestion_predictor(
            query_state,
            zone_state,
            memory,
            ~memory_mask,
        )
        if (
            not isinstance(density_result, tuple)
            or len(density_result) != 2
            or not all(isinstance(value, Tensor) for value in density_result)
        ):
            raise TypeError("congestion predictor must return (logits, density)")
        _, density = density_result
        congestion_tokens = self.congestion_tokenizer(density)
        selected_node_valid = (
            encoder_output.team_alive_mask[planner_batch.query_mask]
            & encoder_output.time_mask[
                query_indices[:, 0], query_indices[:, 1]
            ]
        )
        query_mask = coordinate_eligible & selected_node_valid
        return ParallelTrajectoryBatch(
            z_query=query_state,
            memory=memory,
            memory_mask=memory_mask,
            current_xy=current_xy,
            zone_state=zone_state,
            congestion_tokens=congestion_tokens,
            query_mask=query_mask,
        )

    prepare_decoder_batch = prepare

    def forward_prepared(
        self,
        batch: ParallelTrajectoryBatch,
        targets: ParallelTrajectoryTargets | None = None,
    ) -> ParallelTrajectoryOutput:
        if targets is not None and targets.target_mask.shape[0] != batch.z_query.shape[0]:
            raise ValueError("targets must align with prepared query order")
        horizon_mask = None if targets is None else targets.target_mask.to(batch.z_query.device)
        return self.decoder(batch, horizon_mask=horizon_mask)

    def forward(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
        targets: ParallelTrajectoryTargets | None = None,
    ) -> ParallelTrajectoryOutput:
        return self.forward_prepared(self.prepare(batch, observation), targets)


__all__ = ["ParallelTrajectoryModel"]
