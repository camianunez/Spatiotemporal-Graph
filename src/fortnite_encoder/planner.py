from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import BoolTensor, Tensor, nn
from torch.nn import functional as F

from .contracts import EncoderBatch, EncoderOutput
from .model import SpatiotemporalEncoder
from .planner_contracts import (
    BeamRoutes,
    GeneratedRoute,
    PlannerBatch,
    PlannerLogits,
    PlannerObservation,
    PreviousWaypoints,
)
from .planner_policy import apply_planner_input_policy
from .world_grid import WorldGridProfile


D_MODEL = 256
N_HEADS = 8
FFN_DIM = 1024
ROUTE_STEPS = 12
PREVIOUS_ROUTE_STEPS = ROUTE_STEPS - 1
GRID_ROWS = 32
GRID_COLUMNS = 32
NUM_REGULAR_CELLS = GRID_ROWS * GRID_COLUMNS
NUM_CELL_CLASSES = NUM_REGULAR_CELLS
ZONE_VECTOR_WIDTH = 15
DROPOUT = 0.1
CONGESTION_ATTENTION_SCALE = 0.25


@dataclass(frozen=True, slots=True)
class _PlannerContext:
    query_state: Tensor
    zone_vector: Tensor
    memory: Tensor
    memory_padding_mask: BoolTensor
    initial_cells: Tensor
    initial_offsets: Tensor


class _FeedForward(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(D_MODEL, FFN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FFN_DIM, D_MODEL),
        )


class CongestionPredictor(nn.Module):
    """One causal-memory cross-attention block over twelve learned horizons."""

    def __init__(self) -> None:
        super().__init__()
        self.horizon_embedding = nn.Embedding(ROUTE_STEPS, D_MODEL)
        self.zone_projection = nn.Linear(ZONE_VECTOR_WIDTH, D_MODEL)
        self.memory_norm = nn.LayerNorm(D_MODEL)
        self.memory_attention = nn.MultiheadAttention(
            D_MODEL,
            N_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.memory_residual_dropout = nn.Dropout(DROPOUT)
        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = _FeedForward()
        self.ffn_residual_dropout = nn.Dropout(DROPOUT)
        self.density_head = nn.Linear(D_MODEL, NUM_REGULAR_CELLS)

    def forward(
        self,
        query_state: Tensor,
        zone_vector: Tensor,
        memory: Tensor,
        memory_padding_mask: BoolTensor,
    ) -> tuple[Tensor, Tensor]:
        q = query_state.shape[0]
        if q == 0:
            logits = query_state.new_empty((0, ROUTE_STEPS, GRID_ROWS, GRID_COLUMNS))
            return logits, F.softplus(logits)
        horizon_ids = torch.arange(ROUTE_STEPS, device=query_state.device)
        state = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(q, -1, -1)
        state = (
            state
            + query_state.unsqueeze(1)
            + self.zone_projection(zone_vector).unsqueeze(1)
        )
        normalized = self.memory_norm(state)
        update, _ = self.memory_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        state = state + self.memory_residual_dropout(update)
        state = state + self.ffn_residual_dropout(self.ffn(self.ffn_norm(state)))
        density_logits = self.density_head(state).reshape(
            q, ROUTE_STEPS, GRID_ROWS, GRID_COLUMNS
        )
        return density_logits, F.softplus(density_logits)


class CongestionTokenizer(nn.Module):
    """Turn every scalar in all predicted grids into an attention token."""

    def __init__(self) -> None:
        super().__init__()
        self.density_projection = nn.Linear(1, D_MODEL)
        self.row_embedding = nn.Embedding(GRID_ROWS, D_MODEL)
        self.column_embedding = nn.Embedding(GRID_COLUMNS, D_MODEL)
        self.horizon_embedding = nn.Embedding(ROUTE_STEPS, D_MODEL)

    def forward(self, density: Tensor) -> Tensor:
        if density.shape != (
            density.shape[0],
            ROUTE_STEPS,
            GRID_ROWS,
            GRID_COLUMNS,
        ):
            raise ValueError("density must have shape [Q,12,32,32]")
        q = density.shape[0]
        rows = self.row_embedding(
            torch.arange(GRID_ROWS, device=density.device)
        ).view(1, 1, GRID_ROWS, 1, D_MODEL)
        columns = self.column_embedding(
            torch.arange(GRID_COLUMNS, device=density.device)
        ).view(1, 1, 1, GRID_COLUMNS, D_MODEL)
        horizons = self.horizon_embedding(
            torch.arange(ROUTE_STEPS, device=density.device)
        ).view(1, ROUTE_STEPS, 1, 1, D_MODEL)
        tokens = self.density_projection(density.unsqueeze(-1))
        tokens = tokens + rows + columns + horizons
        return tokens.reshape(q, ROUTE_STEPS * NUM_REGULAR_CELLS, D_MODEL)


class RouteDecoderLayer(nn.Module):
    """Fixed-order pre-LayerNorm route decoder layer."""

    congestion_attention_scale = CONGESTION_ATTENTION_SCALE

    def __init__(self) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(D_MODEL)
        self.self_attention = nn.MultiheadAttention(
            D_MODEL,
            N_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.self_attention_dropout = nn.Dropout(DROPOUT)

        self.memory_attention_norm = nn.LayerNorm(D_MODEL)
        self.memory_attention = nn.MultiheadAttention(
            D_MODEL,
            N_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.memory_attention_dropout = nn.Dropout(DROPOUT)

        self.congestion_attention_norm = nn.LayerNorm(D_MODEL)
        self.congestion_attention = nn.MultiheadAttention(
            D_MODEL,
            N_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.congestion_attention_dropout = nn.Dropout(DROPOUT)

        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = _FeedForward()
        self.ffn_residual_dropout = nn.Dropout(DROPOUT)

    def forward(
        self,
        state: Tensor,
        memory: Tensor,
        memory_padding_mask: BoolTensor,
        congestion_tokens: Tensor,
        causal_mask: BoolTensor,
    ) -> Tensor:
        normalized = self.self_attention_norm(state)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        state = state + self.self_attention_dropout(update)

        normalized = self.memory_attention_norm(state)
        update, _ = self.memory_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        state = state + self.memory_attention_dropout(update)

        normalized = self.congestion_attention_norm(state)
        update, _ = self.congestion_attention(
            normalized,
            congestion_tokens,
            congestion_tokens,
            need_weights=False,
        )
        state = state + self.congestion_attention_scale * (
            self.congestion_attention_dropout(update)
        )
        state = state + self.ffn_residual_dropout(self.ffn(self.ffn_norm(state)))
        return state


class RouteDecoder(nn.Module):
    """Four-layer shifted autoregressive route decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.cell_embedding = nn.Embedding(NUM_CELL_CLASSES, D_MODEL)
        self.offset_projection = nn.Linear(2, D_MODEL)
        self.step_embedding = nn.Embedding(ROUTE_STEPS, D_MODEL)
        self.zone_projection = nn.Linear(ZONE_VECTOR_WIDTH, D_MODEL)
        self.input_dropout = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList(RouteDecoderLayer() for _ in range(4))
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.cell_head = nn.Linear(D_MODEL, NUM_CELL_CLASSES)
        self.offset_head = nn.Linear(D_MODEL, 2)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(ROUTE_STEPS, ROUTE_STEPS, dtype=torch.bool), 1),
            persistent=False,
        )

    def forward(
        self,
        query_state: Tensor,
        zone_vector: Tensor,
        memory: Tensor,
        memory_padding_mask: BoolTensor,
        congestion_tokens: Tensor,
        initial_cells: Tensor,
        initial_offsets: Tensor,
        previous_waypoints: PreviousWaypoints,
    ) -> tuple[Tensor, Tensor]:
        q = query_state.shape[0]
        if q == 0:
            return (
                query_state.new_empty((0, ROUTE_STEPS, NUM_CELL_CLASSES)),
                query_state.new_empty((0, ROUTE_STEPS, 2)),
            )
        initial = self.cell_embedding(initial_cells).unsqueeze(1)
        initial = initial + self.offset_projection(
            initial_offsets.to(query_state.dtype)
        ).unsqueeze(1)
        previous = self.cell_embedding(previous_waypoints.cells)
        previous = previous + self.offset_projection(
            previous_waypoints.offsets.to(query_state.dtype)
        )
        shifted = torch.cat((initial, previous), dim=1)
        step_ids = torch.arange(ROUTE_STEPS, device=query_state.device)
        state = (
            shifted
            + self.step_embedding(step_ids).unsqueeze(0)
            + query_state.unsqueeze(1)
            + self.zone_projection(zone_vector).unsqueeze(1)
        )
        state = self.input_dropout(state)
        for layer in self.layers:
            state = layer(
                state,
                memory,
                memory_padding_mask,
                congestion_tokens,
                self.causal_mask,
            )
        state = self.final_norm(state)
        return self.cell_head(state), torch.tanh(self.offset_head(state))


class ExpertRotationPlanner(nn.Module):
    """Planner-only congestion predictor and autoregressive route decoder."""

    d_model = D_MODEL
    route_steps = ROUTE_STEPS
    num_cell_classes = NUM_CELL_CLASSES

    def __init__(self, world_grid_profile: WorldGridProfile) -> None:
        super().__init__()
        if not isinstance(world_grid_profile, WorldGridProfile):
            raise TypeError("world_grid_profile must be a WorldGridProfile")
        if (
            world_grid_profile.grid_rows != GRID_ROWS
            or world_grid_profile.grid_columns != GRID_COLUMNS
            or world_grid_profile.cell_ordering != "row_major"
        ):
            raise ValueError("planner requires the canonical row-major 32x32 grid")
        self.world_grid_profile = world_grid_profile
        self.congestion_predictor = CongestionPredictor()
        self.congestion_tokenizer = CongestionTokenizer()
        self.route_decoder = RouteDecoder()

    @staticmethod
    def _validate_previous(previous: PreviousWaypoints, q: int, device: torch.device) -> None:
        if previous.cells.shape[0] != q:
            raise ValueError("previous waypoints Q axis must match query_mask")
        if previous.cells.device != device or previous.offsets.device != device:
            raise ValueError("previous waypoints must share the encoder device")

    def _derive_initial_waypoints(
        self,
        planner_batch: PlannerBatch,
        query_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Construct ``y0`` only from the sanitized query-time observation."""

        batch = planner_batch.encoder_batch
        q = query_indices.shape[0]
        if q == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=batch.player_xyz_uu.device),
                batch.player_xyz_uu.new_empty((0, 2)),
            )
        batch_index, time_index, team_index = query_indices.unbind(dim=-1)
        coordinates = batch.player_xyz_uu[
            batch_index, time_index, team_index, :, :2
        ]
        valid = (
            batch.player_alive[batch_index, time_index, team_index]
            & batch.player_coord_mask[batch_index, time_index, team_index]
            & batch.player_slot_mask[batch_index, team_index]
        )
        counts = valid.sum(dim=-1)
        if bool((counts == 0).any()):
            raise ValueError(
                "every planner query must have a valid query-time focal coordinate"
            )
        if not bool(torch.isfinite(coordinates[valid]).all()):
            raise ValueError("planner query y0 coordinates must be finite")
        centroid = (
            torch.where(valid.unsqueeze(-1), coordinates, torch.zeros_like(coordinates))
            .double()
            .sum(dim=1)
            / counts.double().unsqueeze(-1)
        )
        profile = self.world_grid_profile
        x, y = centroid.unbind(dim=-1)
        in_bounds = (
            (x >= float(profile.world_x_min))
            & (x <= float(profile.world_x_max))
            & (y >= float(profile.world_y_min))
            & (y <= float(profile.world_y_max))
        )
        if not bool(in_bounds.all()):
            raise ValueError("every planner query y0 must be inside the world grid")
        column = torch.floor(
            (x - float(profile.world_x_min))
            / float(profile.cell_width_world_units)
        ).clamp(max=GRID_COLUMNS - 1)
        row = torch.floor(
            (y - float(profile.world_y_min))
            / float(profile.cell_height_world_units)
        ).clamp(max=GRID_ROWS - 1)
        cells = (row * GRID_COLUMNS + column).to(torch.int64)
        fraction_x = (
            x
            - (
                float(profile.world_x_min)
                + column * float(profile.cell_width_world_units)
            )
        ) / float(profile.cell_width_world_units)
        fraction_y = (
            y
            - (
                float(profile.world_y_min)
                + row * float(profile.cell_height_world_units)
            )
        ) / float(profile.cell_height_world_units)
        offsets = torch.stack(
            (2.0 * fraction_x - 1.0, 2.0 * fraction_y - 1.0), dim=-1
        ).to(coordinates.dtype)
        if not bool((offsets.abs() <= 1.0).all()):
            raise RuntimeError("query y0 mapping produced an invalid offset")
        return cells, offsets

    def initial_waypoints(
        self,
        planner_batch: PlannerBatch,
    ) -> tuple[Tensor, Tensor]:
        """Expose the internally derived ``y0`` for contract verification."""

        return self._derive_initial_waypoints(
            planner_batch,
            planner_batch.query_mask.nonzero(as_tuple=False),
        )

    def _normalize_circle(self, circle: Tensor, active: BoolTensor) -> Tensor:
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
        return torch.where(
            active.unsqueeze(-1), normalized, torch.zeros_like(normalized)
        )

    def _gather_zone_vector(
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

    def _build_context(
        self,
        planner_batch: PlannerBatch,
        encoder_output: EncoderOutput,
    ) -> _PlannerContext:
        batch = planner_batch.encoder_batch
        z = encoder_output.z
        if z.ndim != 4 or z.shape[-1] != D_MODEL:
            raise ValueError("encoder z must have shape [B,T,N,256]")
        if tuple(encoder_output.team_alive_mask.shape) != tuple(z.shape[:-1]):
            raise ValueError("team_alive_mask must match encoder query axes")
        if tuple(encoder_output.time_mask.shape) != tuple(z.shape[:2]):
            raise ValueError("time_mask must have shape [B,T]")
        if tuple(planner_batch.query_mask.shape) != tuple(z.shape[:-1]):
            raise ValueError("query_mask must match encoder query axes")
        if not torch.equal(encoder_output.time_mask, batch.time_mask):
            raise ValueError("encoder output must come from the sanitized planner batch")
        if z.device != planner_batch.query_mask.device:
            raise ValueError("planner batch and encoder output must share a device")
        valid_nodes = (
            encoder_output.team_alive_mask
            & encoder_output.time_mask.unsqueeze(-1)
        )
        if torch.any(planner_batch.query_mask & ~valid_nodes):
            raise ValueError("every planner query must be a valid encoded node")

        query_indices = planner_batch.query_mask.nonzero(as_tuple=False)
        query_state = z[planner_batch.query_mask]
        initial_cells, initial_offsets = self._derive_initial_waypoints(
            planner_batch, query_indices
        )
        zone_vector = self._gather_zone_vector(planner_batch, query_indices).to(
            query_state.dtype
        )
        q = query_indices.shape[0]
        if q == 0:
            return _PlannerContext(
                query_state=query_state,
                zone_vector=zone_vector,
                memory=z.new_empty((0, 1, D_MODEL)),
                memory_padding_mask=torch.ones(
                    (0, 1), dtype=torch.bool, device=z.device
                ),
                initial_cells=initial_cells,
                initial_offsets=initial_offsets,
            )

        memories: list[Tensor] = []
        for query in query_indices:
            batch_index = int(query[0])
            time_index = int(query[1])
            eligible = valid_nodes[batch_index].clone()
            eligible[time_index + 1 :] = False
            memories.append(z[batch_index][eligible])
        max_memory = max(memory.shape[0] for memory in memories)
        memory = z.new_zeros((q, max_memory, D_MODEL))
        padding = torch.ones((q, max_memory), dtype=torch.bool, device=z.device)
        for index, values in enumerate(memories):
            memory[index, : values.shape[0]] = values
            padding[index, : values.shape[0]] = False
        return _PlannerContext(
            query_state,
            zone_vector,
            memory,
            padding,
            initial_cells,
            initial_offsets,
        )

    def _predict_context(
        self,
        context: _PlannerContext,
        previous_waypoints: PreviousWaypoints,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        density_logits, density = self.congestion_predictor(
            context.query_state,
            context.zone_vector,
            context.memory,
            context.memory_padding_mask,
        )
        # Teacher forcing uses density as an interpretation, not as a second
        # route-loss path into the predictor.  Congestion supervision still
        # trains both the predictor and encoder through ``density_logits``;
        # route losses train the tokenizer/decoder and their direct encoder
        # context while the predicted density crossing this boundary is fixed.
        congestion_tokens = self.congestion_tokenizer(density.detach())
        cell_logits, offsets = self.route_decoder(
            context.query_state,
            context.zone_vector,
            context.memory,
            context.memory_padding_mask,
            congestion_tokens,
            context.initial_cells,
            context.initial_offsets,
            previous_waypoints,
        )
        return density_logits, density, congestion_tokens, cell_logits, offsets

    def forward(
        self,
        planner_batch: PlannerBatch,
        encoder_output: EncoderOutput,
        previous_waypoints: PreviousWaypoints,
    ) -> PlannerLogits:
        context = self._build_context(planner_batch, encoder_output)
        self._validate_previous(
            previous_waypoints,
            context.query_state.shape[0],
            context.query_state.device,
        )
        density_logits, density, _, cell_logits, offsets = self._predict_context(
            context, previous_waypoints
        )
        return PlannerLogits(
            density_logits=density_logits,
            density=density,
            cell_logits=cell_logits,
            offsets=offsets,
            query_mask=planner_batch.query_mask.clone(),
        )

    def waypoints_from_cells(
        self,
        cells: Tensor,
        offsets: Tensor,
    ) -> tuple[Tensor, BoolTensor]:
        """Map regular row-major cells and bounded offsets to canonical world XY."""

        if cells.dtype != torch.int64:
            raise TypeError("cells must have int64 dtype")
        if offsets.shape != (*cells.shape, 2):
            raise ValueError("offsets must have shape cells.shape + (2,)")
        if cells.device != offsets.device:
            raise ValueError("cells and offsets must share a device")
        if offsets.numel() and (
            not bool(torch.isfinite(offsets).all())
            or bool((offsets.abs() > 1.0).any())
        ):
            raise ValueError("offsets must be finite and bounded by [-1,1]")
        if cells.numel() and (
            int(cells.min()) < 0 or int(cells.max()) >= NUM_CELL_CLASSES
        ):
            raise ValueError("cells must contain map cells in [0,1023]")

        valid = torch.ones_like(cells, dtype=torch.bool)
        row = torch.div(cells, GRID_COLUMNS, rounding_mode="floor")
        column = torch.remainder(cells, GRID_COLUMNS)
        fraction = (offsets + 1.0) / 2.0
        profile = self.world_grid_profile
        x = (
            float(profile.world_x_min)
            + (column.to(offsets.dtype) + fraction[..., 0])
            * float(profile.cell_width_world_units)
        )
        y = (
            float(profile.world_y_min)
            + (row.to(offsets.dtype) + fraction[..., 1])
            * float(profile.cell_height_world_units)
        )
        coordinates = torch.stack((x, y), dim=-1)
        return coordinates, valid

    def _decode_generation_step(
        self,
        context: _PlannerContext,
        previous_waypoints: PreviousWaypoints,
        congestion_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Overridable generation seam returning all twelve decoder positions."""

        return self.route_decoder(
            context.query_state,
            context.zone_vector,
            context.memory,
            context.memory_padding_mask,
            congestion_tokens,
            context.initial_cells,
            context.initial_offsets,
            previous_waypoints,
        )

    def _greedy_future_prefix(
        self,
        context: _PlannerContext,
        congestion_tokens: Tensor,
    ) -> PreviousWaypoints:
        """Generate detached ``y1..y11`` data for rollout supervision."""

        q = context.query_state.shape[0]
        device = context.query_state.device
        dtype = context.query_state.dtype
        cells = torch.zeros(
            (q, PREVIOUS_ROUTE_STEPS), dtype=torch.int64, device=device
        )
        offsets = torch.zeros(
            (q, PREVIOUS_ROUTE_STEPS, 2), dtype=dtype, device=device
        )
        with torch.no_grad():
            for step in range(PREVIOUS_ROUTE_STEPS):
                prefix = PreviousWaypoints(cells, offsets)
                logits, predicted_offsets = self._decode_generation_step(
                    context, prefix, congestion_tokens
                )
                cells[:, step] = torch.argmax(logits[:, step], dim=-1)
                offsets[:, step] = predicted_offsets[:, step]
        return PreviousWaypoints(cells.detach(), offsets.detach())

    def teacher_and_rollout(
        self,
        planner_batch: PlannerBatch,
        encoder_output: EncoderOutput,
        teacher_previous: PreviousWaypoints,
        *,
        include_rollout: bool,
    ) -> tuple[PlannerLogits, PlannerLogits | None, PreviousWaypoints | None]:
        """Run teacher and optional detached-prefix rollout from one context."""

        context = self._build_context(planner_batch, encoder_output)
        self._validate_previous(
            teacher_previous,
            context.query_state.shape[0],
            context.query_state.device,
        )
        density_logits, density = self.congestion_predictor(
            context.query_state,
            context.zone_vector,
            context.memory,
            context.memory_padding_mask,
        )
        congestion_tokens = self.congestion_tokenizer(density.detach())
        teacher_cells, teacher_offsets = self._decode_generation_step(
            context, teacher_previous, congestion_tokens
        )
        teacher = PlannerLogits(
            density_logits=density_logits,
            density=density,
            cell_logits=teacher_cells,
            offsets=teacher_offsets,
            query_mask=planner_batch.query_mask.clone(),
        )
        if not include_rollout:
            return teacher, None, None
        generated = self._greedy_future_prefix(context, congestion_tokens)
        rollout_cells, rollout_offsets = self._decode_generation_step(
            context, generated, congestion_tokens
        )
        rollout = PlannerLogits(
            density_logits=density_logits,
            density=density,
            cell_logits=rollout_cells,
            offsets=rollout_offsets,
            query_mask=planner_batch.query_mask.clone(),
        )
        return teacher, rollout, generated

    @staticmethod
    def _repeat_context(context: _PlannerContext, count: int) -> _PlannerContext:
        return _PlannerContext(
            query_state=context.query_state.expand(count, -1),
            zone_vector=context.zone_vector.expand(count, -1),
            memory=context.memory.expand(count, -1, -1),
            memory_padding_mask=context.memory_padding_mask.expand(count, -1),
            initial_cells=context.initial_cells.expand(count),
            initial_offsets=context.initial_offsets.expand(count, -1),
        )

    @torch.no_grad()
    def greedy(
        self,
        planner_batch: PlannerBatch,
        encoder_output: EncoderOutput,
    ) -> GeneratedRoute:
        context = self._build_context(planner_batch, encoder_output)
        q = context.query_state.shape[0]
        _, density = self.congestion_predictor(
            context.query_state,
            context.zone_vector,
            context.memory,
            context.memory_padding_mask,
        )
        congestion_tokens = self.congestion_tokenizer(density)
        device = context.query_state.device
        dtype = context.query_state.dtype
        cells = torch.zeros((q, ROUTE_STEPS), dtype=torch.int64, device=device)
        offsets = torch.zeros((q, ROUTE_STEPS, 2), dtype=dtype, device=device)
        step_log_probabilities = torch.zeros(
            (q, ROUTE_STEPS), dtype=dtype, device=device
        )
        for step in range(ROUTE_STEPS):
            previous = PreviousWaypoints(
                cells[:, :PREVIOUS_ROUTE_STEPS],
                offsets[:, :PREVIOUS_ROUTE_STEPS],
            )
            cell_logits, predicted_offsets = self._decode_generation_step(
                context, previous, congestion_tokens
            )
            log_probabilities = F.log_softmax(cell_logits[:, step], dim=-1)
            # torch.argmax resolves exact ties by the first (lowest) cell.
            selected = torch.argmax(log_probabilities, dim=-1)
            selected_log_probability = log_probabilities.gather(
                -1, selected.unsqueeze(-1)
            ).squeeze(-1)
            cells[:, step] = selected
            offsets[:, step] = predicted_offsets[:, step]
            step_log_probabilities[:, step] = selected_log_probability

        coordinates, waypoint_mask = self.waypoints_from_cells(cells, offsets)
        return GeneratedRoute(
            cells=cells,
            offsets=offsets,
            waypoints_xy_uu=coordinates,
            waypoint_mask=waypoint_mask,
            step_log_probabilities=step_log_probabilities,
            scores=step_log_probabilities.sum(dim=-1),
            query_mask=planner_batch.query_mask.clone(),
        )

    generate_greedy = greedy
    greedy_decode = greedy

    @torch.no_grad()
    def beam_search(
        self,
        planner_batch: PlannerBatch,
        encoder_output: EncoderOutput,
        beam_width: int,
    ) -> BeamRoutes:
        if (
            not isinstance(beam_width, int)
            or isinstance(beam_width, bool)
            or beam_width <= 0
        ):
            raise ValueError("beam_width must be a positive integer")
        context = self._build_context(planner_batch, encoder_output)
        q = context.query_state.shape[0]
        _, density = self.congestion_predictor(
            context.query_state,
            context.zone_vector,
            context.memory,
            context.memory_padding_mask,
        )
        all_congestion_tokens = self.congestion_tokenizer(density)
        device = context.query_state.device
        dtype = context.query_state.dtype
        if q == 0:
            return BeamRoutes(
                cells=torch.empty(
                    (0, beam_width, ROUTE_STEPS), dtype=torch.int64, device=device
                ),
                offsets=torch.empty(
                    (0, beam_width, ROUTE_STEPS, 2), dtype=dtype, device=device
                ),
                waypoints_xy_uu=torch.empty(
                    (0, beam_width, ROUTE_STEPS, 2), dtype=dtype, device=device
                ),
                waypoint_mask=torch.empty(
                    (0, beam_width, ROUTE_STEPS), dtype=torch.bool, device=device
                ),
                step_log_probabilities=torch.empty(
                    (0, beam_width, ROUTE_STEPS), dtype=dtype, device=device
                ),
                scores=torch.empty((0, beam_width), dtype=dtype, device=device),
                query_mask=planner_batch.query_mask.clone(),
            )

        query_cells: list[Tensor] = []
        query_offsets: list[Tensor] = []
        query_step_log_probabilities: list[Tensor] = []
        query_scores: list[Tensor] = []
        for query_index in range(q):
            single_context = _PlannerContext(
                query_state=context.query_state[query_index : query_index + 1],
                zone_vector=context.zone_vector[query_index : query_index + 1],
                memory=context.memory[query_index : query_index + 1],
                memory_padding_mask=context.memory_padding_mask[
                    query_index : query_index + 1
                ],
                initial_cells=context.initial_cells[query_index : query_index + 1],
                initial_offsets=context.initial_offsets[
                    query_index : query_index + 1
                ],
            )
            single_tokens = all_congestion_tokens[query_index : query_index + 1]
            cells = torch.zeros(
                (1, ROUTE_STEPS), dtype=torch.int64, device=device
            )
            offsets = torch.zeros((1, ROUTE_STEPS, 2), dtype=dtype, device=device)
            step_logs = torch.zeros((1, ROUTE_STEPS), dtype=dtype, device=device)
            scores = torch.zeros((1,), dtype=dtype, device=device)

            for step in range(ROUTE_STEPS):
                beam_count = cells.shape[0]
                repeated_context = self._repeat_context(single_context, beam_count)
                repeated_tokens = single_tokens.expand(beam_count, -1, -1)
                previous = PreviousWaypoints(
                    cells[:, :PREVIOUS_ROUTE_STEPS],
                    offsets[:, :PREVIOUS_ROUTE_STEPS],
                )
                cell_logits, predicted_offsets = self._decode_generation_step(
                    repeated_context, previous, repeated_tokens
                )
                log_probabilities = F.log_softmax(cell_logits[:, step], dim=-1)
                candidate_scores = scores.unsqueeze(-1) + log_probabilities
                candidate_values = candidate_scores.reshape(-1)
                # Flattening is parent-major then class-major. Stable sorting
                # therefore breaks ties by parent and then lower map-cell ID.
                ranking = torch.argsort(
                    candidate_values, descending=True, stable=True
                )[:beam_width]
                parent = torch.div(
                    ranking, NUM_CELL_CLASSES, rounding_mode="floor"
                )
                selected = torch.remainder(ranking, NUM_CELL_CLASSES)

                cells = cells[parent].clone()
                offsets = offsets[parent].clone()
                step_logs = step_logs[parent].clone()
                cells[:, step] = selected
                selected_step_log = log_probabilities[parent, selected]
                step_logs[:, step] = selected_step_log
                offsets[:, step] = predicted_offsets[parent, step]
                scores = candidate_values[ranking]

            query_cells.append(cells)
            query_offsets.append(offsets)
            query_step_log_probabilities.append(step_logs)
            query_scores.append(scores)

        cells = torch.stack(query_cells)
        offsets = torch.stack(query_offsets)
        step_logs = torch.stack(query_step_log_probabilities)
        scores = torch.stack(query_scores)
        coordinates, waypoint_mask = self.waypoints_from_cells(cells, offsets)
        return BeamRoutes(
            cells=cells,
            offsets=offsets,
            waypoints_xy_uu=coordinates,
            waypoint_mask=waypoint_mask,
            step_log_probabilities=step_logs,
            scores=scores,
            query_mask=planner_batch.query_mask.clone(),
        )

    generate_beam = beam_search


class ExpertRotationPlannerModel(nn.Module):
    """Compose the planner policy with an injected, otherwise unchanged encoder."""

    def __init__(
        self,
        world_grid_profile: WorldGridProfile | SpatiotemporalEncoder,
        encoder: SpatiotemporalEncoder | WorldGridProfile | None = None,
    ) -> None:
        super().__init__()
        # Accept both ``(profile, encoder)`` and ``(encoder, profile)`` while
        # keeping the explicit keyword form unambiguous.
        if isinstance(world_grid_profile, SpatiotemporalEncoder):
            world_grid_profile, encoder = encoder, world_grid_profile
        if not isinstance(world_grid_profile, WorldGridProfile):
            raise TypeError("world_grid_profile must be a WorldGridProfile")
        if not isinstance(encoder, SpatiotemporalEncoder):
            raise TypeError("encoder must be an injected SpatiotemporalEncoder")
        if encoder.config.d_model != D_MODEL:
            raise ValueError("the planner requires an encoder with d_model=256")
        self.encoder = encoder
        self.planner = ExpertRotationPlanner(world_grid_profile)

    def prepare(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
    ) -> tuple[PlannerBatch, EncoderOutput]:
        planner_batch = apply_planner_input_policy(batch, observation)
        return planner_batch, self.encoder(planner_batch.encoder_batch)

    def forward(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
        previous_waypoints: PreviousWaypoints,
    ) -> PlannerLogits:
        planner_batch, encoder_output = self.prepare(batch, observation)
        return self.planner(planner_batch, encoder_output, previous_waypoints)

    def teacher_and_rollout(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
        teacher_previous: PreviousWaypoints,
        *,
        include_rollout: bool,
    ) -> tuple[PlannerLogits, PlannerLogits | None, PreviousWaypoints | None]:
        planner_batch, encoder_output = self.prepare(batch, observation)
        return self.planner.teacher_and_rollout(
            planner_batch,
            encoder_output,
            teacher_previous,
            include_rollout=include_rollout,
        )

    def greedy(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
    ) -> GeneratedRoute:
        planner_batch, encoder_output = self.prepare(batch, observation)
        return self.planner.greedy(planner_batch, encoder_output)

    generate_greedy = greedy
    greedy_decode = greedy

    def beam_search(
        self,
        batch: EncoderBatch,
        observation: PlannerObservation,
        beam_width: int,
    ) -> BeamRoutes:
        planner_batch, encoder_output = self.prepare(batch, observation)
        return self.planner.beam_search(
            planner_batch, encoder_output, beam_width
        )

    generate_beam = beam_search


__all__ = [
    "BeamRoutes",
    "CongestionPredictor",
    "ExpertRotationPlanner",
    "ExpertRotationPlannerModel",
    "GeneratedRoute",
    "PlannerLogits",
    "PreviousWaypoints",
    "RouteDecoder",
    "RouteDecoderLayer",
]
