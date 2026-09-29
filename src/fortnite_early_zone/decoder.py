from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fortnite_encoder.planner import (
    CONGESTION_ATTENTION_SCALE,
    D_MODEL,
    DROPOUT,
    FFN_DIM,
    N_HEADS,
    NUM_REGULAR_CELLS,
    RouteDecoderLayer,
    ZONE_VECTOR_WIDTH,
)

from .contracts import (
    MIXTURE_COMPONENTS,
    PREVIOUS_ROUTE_STEPS,
    ROUTE_STEPS,
    EarlyZoneResidualDecoderOutput,
    MovementState,
    PreviousResidualRoute,
    PreviousSoftResidualRoute,
)


CONGESTION_TOKENS = ROUTE_STEPS * NUM_REGULAR_CELLS
MIN_COMPONENT_SCALE = 1e-3
MAX_CORRELATION_MAGNITUDE = 0.999


def interpolate_movement_embeddings(
    embedding: nn.Embedding,
    movement_targets: Tensor,
) -> Tensor:
    """Interpolate the unchanged HOLD/MOVE table for float targets in [0,1]."""

    if not isinstance(embedding, nn.Embedding) or embedding.num_embeddings != 2:
        raise TypeError("embedding must be a two-entry nn.Embedding")
    if not isinstance(movement_targets, Tensor) or not movement_targets.is_floating_point():
        raise TypeError("movement_targets must be a floating-point tensor")
    if movement_targets.numel() and (
        not bool(torch.isfinite(movement_targets).all())
        or bool(((movement_targets < 0.0) | (movement_targets > 1.0)).any())
    ):
        raise ValueError("movement_targets must be finite values in [0,1]")
    target = movement_targets.to(dtype=embedding.weight.dtype).unsqueeze(-1)
    hold, move = embedding.weight[0], embedding.weight[1]
    return (1.0 - target) * hold + target * move


movement_target_embeddings = interpolate_movement_embeddings


class _CoordinateMLP(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(2, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )


def expected_normalized_displacements(
    movement_logits: Tensor,
    mixture_logits: Tensor,
    component_means: Tensor,
) -> Tensor:
    """Return E[delta] for the joint HOLD/five-component distribution."""

    if movement_logits.ndim != 2:
        raise ValueError("movement_logits must have shape [Q,12]")
    if tuple(mixture_logits.shape) != (*movement_logits.shape, MIXTURE_COMPONENTS):
        raise ValueError("mixture_logits must have shape [Q,12,5]")
    if tuple(component_means.shape) != (
        *movement_logits.shape,
        MIXTURE_COMPONENTS,
        2,
    ):
        raise ValueError("component_means must have shape [Q,12,5,2]")
    move_probability = torch.sigmoid(movement_logits)
    mixture_probability = torch.softmax(mixture_logits, dim=-1)
    conditional_mean = (
        mixture_probability.unsqueeze(-1) * component_means
    ).sum(dim=-2)
    return move_probability.unsqueeze(-1) * conditional_mean


class EarlyZoneResidualDecoder(nn.Module):
    """Four-block autoregressive residual decoder for twelve route steps.

    ``causal_memory_mask`` deliberately uses public validity semantics:
    ``True`` denotes an observable memory token.  It is inverted only at the
    PyTorch attention boundary, where ``key_padding_mask=True`` means ignore.
    """

    d_model = D_MODEL
    n_heads = N_HEADS
    ffn_dim = FFN_DIM
    route_steps = ROUTE_STEPS
    mixture_components = MIXTURE_COMPONENTS
    dropout = DROPOUT
    congestion_attention_scale = CONGESTION_ATTENTION_SCALE

    def __init__(self) -> None:
        super().__init__()
        self.query_projection = nn.Linear(D_MODEL, D_MODEL)
        self.zone_projection = nn.Linear(ZONE_VECTOR_WIDTH, D_MODEL)
        self.step_embedding = nn.Embedding(ROUTE_STEPS, D_MODEL)
        self.position_mlp = _CoordinateMLP()
        self.displacement_mlp = _CoordinateMLP()
        self.movement_embedding = nn.Embedding(2, D_MODEL)
        self.layers = nn.ModuleList(RouteDecoderLayer() for _ in range(4))
        self.final_norm = nn.LayerNorm(D_MODEL)

        self.movement_head = nn.Linear(D_MODEL, 1)
        self.mixture_head = nn.Linear(D_MODEL, MIXTURE_COMPONENTS)
        self.mean_head = nn.Linear(D_MODEL, MIXTURE_COMPONENTS * 2)
        self.scale_head = nn.Linear(D_MODEL, MIXTURE_COMPONENTS * 2)
        self.correlation_head = nn.Linear(D_MODEL, MIXTURE_COMPONENTS)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(ROUTE_STEPS, ROUTE_STEPS, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    @staticmethod
    def _resolve_previous(
        query_state: Tensor,
        previous_displacements: (
            Tensor | PreviousResidualRoute | PreviousSoftResidualRoute | None
        ),
        previous_movement_states: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        q = query_state.shape[0]
        if isinstance(previous_displacements, PreviousResidualRoute):
            if previous_movement_states is not None:
                raise ValueError(
                    "previous_movement_states must be omitted with PreviousResidualRoute"
                )
            previous_movement_states = previous_displacements.movement_states
            previous_displacements = previous_displacements.normalized_displacements
        elif isinstance(previous_displacements, PreviousSoftResidualRoute):
            if previous_movement_states is not None:
                raise ValueError(
                    "previous_movement_states must be omitted with PreviousSoftResidualRoute"
                )
            previous_movement_states = previous_displacements.movement_targets
            previous_displacements = previous_displacements.normalized_displacements
        if previous_displacements is None and previous_movement_states is None:
            return (
                query_state.new_zeros((q, PREVIOUS_ROUTE_STEPS, 2)),
                torch.zeros(
                    (q, PREVIOUS_ROUTE_STEPS),
                    dtype=torch.int64,
                    device=query_state.device,
                ),
            )
        if previous_displacements is None or previous_movement_states is None:
            raise ValueError(
                "previous_displacements and previous_movement_states must be supplied together"
            )
        if not isinstance(previous_displacements, Tensor) or not isinstance(
            previous_movement_states, Tensor
        ):
            raise TypeError("teacher-forcing inputs must be tensors")
        if not previous_displacements.is_floating_point():
            raise TypeError("previous_displacements must be floating point")
        if previous_movement_states.dtype != torch.int64 and not (
            previous_movement_states.dtype == torch.float32
            or previous_movement_states.dtype == torch.float64
            or previous_movement_states.dtype == torch.bfloat16
            or previous_movement_states.dtype == torch.float16
        ):
            raise TypeError(
                "previous movement values must have int64 or floating-point dtype"
            )
        if tuple(previous_displacements.shape) != (q, PREVIOUS_ROUTE_STEPS, 2):
            raise ValueError("previous_displacements must have shape [Q,11,2]")
        if tuple(previous_movement_states.shape) != (q, PREVIOUS_ROUTE_STEPS):
            raise ValueError("previous_movement_states must have shape [Q,11]")
        if (
            previous_displacements.device != query_state.device
            or previous_movement_states.device != query_state.device
        ):
            raise ValueError("teacher-forcing inputs must share the query device")
        if previous_displacements.numel() and not bool(
            torch.isfinite(previous_displacements).all()
        ):
            raise ValueError("previous_displacements must be finite")
        if previous_movement_states.numel():
            if previous_movement_states.is_floating_point():
                if not bool(torch.isfinite(previous_movement_states).all()) or bool(
                    (
                        (previous_movement_states < 0.0)
                        | (previous_movement_states > 1.0)
                    ).any()
                ):
                    raise ValueError(
                        "previous movement targets must be finite values in [0,1]"
                    )
            elif bool(
                (
                    (previous_movement_states != int(MovementState.HOLD))
                    & (previous_movement_states != int(MovementState.MOVE))
                ).any()
            ):
                raise ValueError("previous_movement_states must contain HOLD or MOVE")
        return (
            previous_displacements.to(query_state.dtype),
            previous_movement_states,
        )

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.layers

    @property
    def position_projection(self) -> nn.Module:
        return self.position_mlp

    @property
    def displacement_projection(self) -> nn.Module:
        return self.displacement_mlp

    def shifted_teacher_forcing(
        self,
        query_state: Tensor,
        previous_displacements: (
            Tensor | PreviousResidualRoute | PreviousSoftResidualRoute | None
        ) = None,
        previous_movement_states: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Expose shifted displacement, cumulative-position, and state inputs."""

        previous_displacements, previous_movement_states = self._resolve_previous(
            query_state,
            previous_displacements,
            previous_movement_states,
        )
        q = query_state.shape[0]
        initial_displacement = query_state.new_zeros((q, 1, 2))
        initial_state = torch.zeros(
            (q, 1), dtype=previous_movement_states.dtype, device=query_state.device
        )
        shifted_displacements = torch.cat(
            (initial_displacement, previous_displacements), dim=1
        )
        shifted_states = torch.cat(
            (initial_state, previous_movement_states), dim=1
        )
        cumulative_positions = shifted_displacements.cumsum(dim=1)
        return shifted_displacements, cumulative_positions, shifted_states

    @staticmethod
    def _validate_context(
        query_state: Tensor,
        zone_vector: Tensor,
        causal_memory: Tensor,
        causal_memory_mask: Tensor,
        congestion_tokens: Tensor,
    ) -> None:
        if not isinstance(query_state, Tensor) or not query_state.is_floating_point():
            raise TypeError("query_state must be a floating-point tensor")
        if tuple(query_state.shape[1:]) != (D_MODEL,):
            raise ValueError("query_state must have shape [Q,256]")
        q = query_state.shape[0]
        expected = {
            "zone_vector": (q, ZONE_VECTOR_WIDTH),
            "causal_memory": (q, causal_memory.shape[1], D_MODEL)
            if causal_memory.ndim == 3
            else (),
            "causal_memory_mask": (q, causal_memory.shape[1])
            if causal_memory.ndim == 3
            else (),
            "congestion_tokens": (q, CONGESTION_TOKENS, D_MODEL),
        }
        for name, value in (
            ("zone_vector", zone_vector),
            ("causal_memory", causal_memory),
            ("congestion_tokens", congestion_tokens),
        ):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
            if tuple(value.shape) != expected[name]:
                if name == "causal_memory":
                    raise ValueError("causal_memory must have shape [Q,M,256]")
                raise ValueError(f"{name} must have shape {list(expected[name])}")
        if not isinstance(causal_memory_mask, Tensor):
            raise TypeError("causal_memory_mask must be a tensor")
        if causal_memory_mask.dtype != torch.bool:
            raise TypeError("causal_memory_mask must have bool dtype")
        if tuple(causal_memory_mask.shape) != expected["causal_memory_mask"]:
            raise ValueError("causal_memory_mask must have shape [Q,M]")
        if q and causal_memory.shape[1] == 0:
            raise ValueError("causal_memory must contain at least one token")
        if q and bool((causal_memory_mask.sum(dim=-1) == 0).any()):
            raise ValueError("each query must have at least one valid causal memory token")
        devices = {
            query_state.device,
            zone_vector.device,
            causal_memory.device,
            causal_memory_mask.device,
            congestion_tokens.device,
        }
        if len(devices) != 1:
            raise ValueError("all decoder inputs must share a device")
        for name, value in (
            ("query_state", query_state),
            ("zone_vector", zone_vector),
            ("causal_memory", causal_memory),
            ("congestion_tokens", congestion_tokens),
        ):
            if value.numel() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")

    def forward(
        self,
        query_state: Tensor,
        zone_vector: Tensor,
        causal_memory: Tensor,
        causal_memory_mask: Tensor,
        congestion_tokens: Tensor,
        previous_displacements: (
            Tensor | PreviousResidualRoute | PreviousSoftResidualRoute | None
        ) = None,
        previous_movement_states: Tensor | None = None,
    ) -> EarlyZoneResidualDecoderOutput:
        self._validate_context(
            query_state,
            zone_vector,
            causal_memory,
            causal_memory_mask,
            congestion_tokens,
        )
        shifted, positions, movement_states = self.shifted_teacher_forcing(
            query_state,
            previous_displacements,
            previous_movement_states,
        )
        q = query_state.shape[0]
        step_ids = torch.arange(ROUTE_STEPS, device=query_state.device)
        state = (
            self.query_projection(query_state).unsqueeze(1)
            + self.zone_projection(zone_vector.to(query_state.dtype)).unsqueeze(1)
            + self.step_embedding(step_ids).unsqueeze(0)
            + self.position_mlp(positions)
            + self.displacement_mlp(shifted)
            + (
                interpolate_movement_embeddings(
                    self.movement_embedding, movement_states
                ).to(query_state.dtype)
                if movement_states.is_floating_point()
                else self.movement_embedding(movement_states)
            )
        )
        if q:
            for layer in self.layers:
                state = layer(
                    state,
                    causal_memory,
                    ~causal_memory_mask,
                    congestion_tokens,
                    self.causal_mask,
                )
        state = self.final_norm(state)
        movement_logits = self.movement_head(state).squeeze(-1)
        mixture_logits = self.mixture_head(state)
        component_means = self.mean_head(state).reshape(
            q, ROUTE_STEPS, MIXTURE_COMPONENTS, 2
        )
        component_scales = F.softplus(
            self.scale_head(state).reshape(
                q, ROUTE_STEPS, MIXTURE_COMPONENTS, 2
            )
        ) + MIN_COMPONENT_SCALE
        correlations = MAX_CORRELATION_MAGNITUDE * torch.tanh(
            self.correlation_head(state)
        )
        movement_probabilities = torch.sigmoid(movement_logits)
        mixture_probabilities = torch.softmax(mixture_logits, dim=-1)
        expected = expected_normalized_displacements(
            movement_logits,
            mixture_logits,
            component_means,
        )
        return EarlyZoneResidualDecoderOutput(
            movement_logits=movement_logits,
            movement_probabilities=movement_probabilities,
            mixture_logits=mixture_logits,
            mixture_probabilities=mixture_probabilities,
            component_means=component_means,
            component_scales=component_scales,
            component_correlations=correlations,
            expected_displacements=expected,
            route_valid_mask=torch.ones(
                (q, ROUTE_STEPS), dtype=torch.bool, device=query_state.device
            ),
        )


__all__ = [
    "CONGESTION_TOKENS",
    "EarlyZoneResidualDecoder",
    "MAX_CORRELATION_MAGNITUDE",
    "MIN_COMPONENT_SCALE",
    "expected_normalized_displacements",
    "interpolate_movement_embeddings",
    "movement_target_embeddings",
]
