from __future__ import annotations

from dataclasses import dataclass, fields

from torch import BoolTensor, FloatTensor, LongTensor, Tensor


@dataclass(frozen=True, slots=True)
class BatchMetadata:
    """CPU-only identifiers used to map anonymous model slots back to matches."""

    session_ids: tuple[str, ...]
    team_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class TensorizedMatch:
    session_id: str
    team_ids: tuple[int, ...]
    player_xyz_uu: FloatTensor
    player_alive: BoolTensor
    player_coord_mask: BoolTensor
    life_index: LongTensor
    player_slot_mask: BoolTensor
    current_circle_uu: FloatTensor
    target_circle_uu: FloatTensor
    zone_mask: BoolTensor
    zone_phase: LongTensor
    phase_times_s: FloatTensor
    match_elapsed_s: FloatTensor
    players_remaining: LongTensor
    teams_remaining: LongTensor
    absolute_tick_index: LongTensor
    time_mask: BoolTensor
    team_slot_mask: BoolTensor
    prior_player_xyz_uu: FloatTensor
    prior_player_alive: BoolTensor
    prior_player_coord_mask: BoolTensor
    prior_life_index: LongTensor
    prior_state_available: BoolTensor

    @property
    def num_timesteps(self) -> int:
        return int(self.player_xyz_uu.shape[0])

    @property
    def num_teams(self) -> int:
        return int(self.player_xyz_uu.shape[1])


@dataclass(frozen=True, slots=True)
class EncoderBatch:
    """Identifier-free tensors accepted by :class:`SpatiotemporalEncoder`."""

    player_xyz_uu: FloatTensor
    player_alive: BoolTensor
    player_coord_mask: BoolTensor
    life_index: LongTensor
    player_slot_mask: BoolTensor
    current_circle_uu: FloatTensor
    target_circle_uu: FloatTensor
    zone_mask: BoolTensor
    zone_phase: LongTensor
    phase_times_s: FloatTensor
    match_elapsed_s: FloatTensor
    players_remaining: LongTensor
    teams_remaining: LongTensor
    absolute_tick_index: LongTensor
    time_mask: BoolTensor
    team_slot_mask: BoolTensor
    prior_player_xyz_uu: FloatTensor
    prior_player_alive: BoolTensor
    prior_player_coord_mask: BoolTensor
    prior_life_index: LongTensor
    prior_state_available: BoolTensor

    def to(self, *args: object, **kwargs: object) -> EncoderBatch:
        values = {
            field.name: getattr(self, field.name).to(*args, **kwargs)
            for field in fields(self)
        }
        return EncoderBatch(**values)


@dataclass(frozen=True, slots=True)
class CollatedEncoderInput:
    batch: EncoderBatch
    metadata: BatchMetadata

    def to(self, *args: object, **kwargs: object) -> CollatedEncoderInput:
        return CollatedEncoderInput(self.batch.to(*args, **kwargs), self.metadata)


@dataclass(frozen=True, slots=True)
class TeamGeometry:
    centroid_uu: FloatTensor
    centroid_valid: BoolTensor
    team_alive_mask: BoolTensor
    living_count: LongTensor
    velocity_xy: FloatTensor
    velocity_valid: BoolTensor
    speed: FloatTensor
    heading_sin: FloatTensor
    heading_cos: FloatTensor
    heading_valid: BoolTensor
    separation_xy: FloatTensor
    separation_valid: BoolTensor


@dataclass(frozen=True, slots=True)
class NodeFeatures:
    features: FloatTensor
    initial_state: FloatTensor
    geometry: TeamGeometry


@dataclass(frozen=True, slots=True)
class DenseEdges:
    features: FloatTensor
    adjacency: BoolTensor


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    z: FloatTensor
    team_alive_mask: BoolTensor
    time_mask: BoolTensor


@dataclass(frozen=True, slots=True)
class RotationLogits:
    """Identifier-free logits produced for every encoder query node."""

    future_position: FloatTensor
    zone_entry: FloatTensor
    survival: FloatTensor | None
    placement: FloatTensor | None
    query_mask: BoolTensor

    @property
    def future_position_logits(self) -> FloatTensor:
        return self.future_position

    @property
    def zone_entry_logits(self) -> FloatTensor:
        return self.zone_entry

    @property
    def survival_logits(self) -> FloatTensor | None:
        return self.survival

    @property
    def placement_logits(self) -> FloatTensor | None:
        return self.placement


@dataclass(frozen=True, slots=True)
class RotationModelOutput:
    encoder: EncoderOutput
    logits: RotationLogits

    @property
    def encoder_output(self) -> EncoderOutput:
        return self.encoder

    @property
    def rotation_logits(self) -> RotationLogits:
        return self.logits

    @property
    def future_position(self) -> FloatTensor:
        return self.logits.future_position

    @property
    def zone_entry(self) -> FloatTensor:
        return self.logits.zone_entry

    @property
    def survival(self) -> FloatTensor | None:
        return self.logits.survival

    @property
    def placement(self) -> FloatTensor | None:
        return self.logits.placement


@dataclass(frozen=True, slots=True)
class PositionProbabilities:
    heatmap: FloatTensor
    eliminated: FloatTensor

    @property
    def eliminated_probability(self) -> FloatTensor:
        return self.eliminated


@dataclass(frozen=True, slots=True)
class RotationTargets:
    """Target tensors only; no session, team, or transform identifiers."""

    future_position: LongTensor
    future_position_mask: BoolTensor
    zone_entry: LongTensor
    zone_entry_mask: BoolTensor
    survival: FloatTensor | None
    survival_mask: BoolTensor | None
    placement: LongTensor | None
    placement_mask: BoolTensor | None

    def to(self, *args: object, **kwargs: object) -> RotationTargets:
        values = {
            field.name: (
                value.to(*args, **kwargs) if value is not None else None
            )
            for field in fields(self)
            for value in (getattr(self, field.name),)
        }
        return RotationTargets(**values)  # type: ignore[arg-type]

    @property
    def future_position_labels(self) -> LongTensor:
        return self.future_position

    @property
    def zone_entry_labels(self) -> LongTensor:
        return self.zone_entry

    @property
    def survival_labels(self) -> FloatTensor | None:
        return self.survival

    @property
    def placement_labels(self) -> LongTensor | None:
        return self.placement


@dataclass(frozen=True, slots=True)
class RotationTargetMetadata:
    session_id: str
    team_ids: tuple[int, ...]
    world_grid_profile_id: str
    world_grid_profile_hash: str


@dataclass(frozen=True, slots=True)
class RotationBatchMetadata:
    session_ids: tuple[str, ...]
    team_ids: tuple[tuple[int, ...], ...]
    world_grid_profile_ids: tuple[str, ...]
    world_grid_profile_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RotationSupervision:
    """Full-match or sliced targets plus CPU-only alignment metadata."""

    targets: RotationTargets
    absolute_tick_index: LongTensor
    metadata: RotationTargetMetadata

    @property
    def num_timesteps(self) -> int:
        return int(self.absolute_tick_index.shape[0])

    @property
    def num_teams(self) -> int:
        return int(self.targets.zone_entry.shape[1])

    def to(self, *args: object, **kwargs: object) -> RotationSupervision:
        return RotationSupervision(
            targets=self.targets.to(*args, **kwargs),
            absolute_tick_index=self.absolute_tick_index.to(*args, **kwargs),
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class CollatedRotationSupervision:
    targets: RotationTargets
    metadata: RotationBatchMetadata

    def to(self, *args: object, **kwargs: object) -> CollatedRotationSupervision:
        return CollatedRotationSupervision(
            targets=self.targets.to(*args, **kwargs),
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class RotationLossOutput:
    total: FloatTensor
    future_position: FloatTensor
    future_position_by_horizon: tuple[FloatTensor, ...]
    zone_entry: FloatTensor
    survival: FloatTensor
    placement: FloatTensor
    future_position_valid_counts: tuple[int, ...]
    zone_entry_valid_count: int
    survival_valid_count: int
    placement_valid_count: int

    @property
    def total_loss(self) -> FloatTensor:
        return self.total

    @property
    def future_position_loss(self) -> FloatTensor:
        return self.future_position

    @property
    def zone_entry_loss(self) -> FloatTensor:
        return self.zone_entry

    @property
    def survival_loss(self) -> FloatTensor:
        return self.survival

    @property
    def placement_loss(self) -> FloatTensor:
        return self.placement

    @property
    def position_valid_counts(self) -> tuple[int, ...]:
        return self.future_position_valid_counts

    @property
    def valid_label_counts(self) -> dict[str, int | tuple[int, ...]]:
        return {
            "future_position": self.future_position_valid_counts,
            "zone_entry": self.zone_entry_valid_count,
            "survival": self.survival_valid_count,
            "placement": self.placement_valid_count,
        }


def ensure_same_device(*tensors: Tensor) -> None:
    if tensors and any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise ValueError("all encoder tensors must be on the same device")
