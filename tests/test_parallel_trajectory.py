from __future__ import annotations

import ast
import hashlib
import inspect
import math
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import fortnite_parallel_trajectory as parallel
from fortnite_encoder import (
    EncoderOutput,
    PlannerObservation,
    PlannerQuery,
    PlannerTargetSource,
    WorldGridProfile,
)
from fortnite_parallel_trajectory.decoder import (
    MAX_CORRELATION_MAGNITUDE,
    MIN_COMPONENT_SCALE,
    ParallelTrajectoryDecoderLayer,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "ml" / "src" / "fortnite_parallel_trajectory"
ENCODER_DIGEST = "a0eb81ae908cfe5dd7c48da0834e129312687f59f80c8c074a983d612944f56e"
DECODER_DIGEST = "be612a30a94781a8702af1940e3f87da85542a277797419e429d9b0a54fcf29d"


def _profile() -> WorldGridProfile:
    return WorldGridProfile.load(
        ROOT / "configs" / "world-grid" / "model-world-grid-v1.json"
    )


def _batch(q: int = 2, *, seed: int = 101) -> parallel.ParallelTrajectoryBatch:
    generator = torch.Generator().manual_seed(seed)
    memory_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]],
        dtype=torch.bool,
    )[:q]
    return parallel.ParallelTrajectoryBatch(
        z_query=torch.randn(q, 256, generator=generator),
        memory=torch.randn(q, 4, 256, generator=generator),
        memory_mask=memory_mask,
        current_xy=torch.tensor([[0.0, 0.0], [10.0, -20.0]])[:q],
        zone_state=torch.randn(q, 15, generator=generator),
        congestion_tokens=torch.randn(q, 3, 256, generator=generator),
        query_mask=torch.ones(q, dtype=torch.bool),
    )


@pytest.fixture(scope="module")
def decoder() -> parallel.ParallelTrajectoryDecoder:
    torch.manual_seed(103)
    return parallel.ParallelTrajectoryDecoder().eval()


def _assert_prediction_equal(
    first: parallel.ParallelTrajectoryOutput,
    second: parallel.ParallelTrajectoryOutput,
    *,
    atol: float = 0.0,
) -> None:
    for name in (
        "mode_logits",
        "mode_probabilities",
        "means",
        "scales",
        "correlations",
        "primary_route_normalized",
        "marginal_mean_route_normalized",
    ):
        assert torch.allclose(
            getattr(first, name), getattr(second, name), atol=atol, rtol=0.0
        ), name
    assert torch.equal(first.primary_mode_index, second.primary_mode_index)


def test_public_api_and_exact_architecture_contract(
    decoder: parallel.ParallelTrajectoryDecoder,
) -> None:
    required = {
        "ParallelTrajectoryConfig",
        "ParallelTrajectoryBatch",
        "ParallelTrajectoryOutput",
        "ParallelTrajectoryTargets",
        "ParallelTrajectoryDecoder",
        "ParallelTrajectoryModel",
        "RouteMixtureLossOutput",
        "build_parallel_trajectory_targets",
        "route_mixture_nll",
        "primary_mode_route",
        "marginal_mean_route",
        "sample_route_modes",
    }
    assert required <= set(parallel.__all__)
    config = parallel.ParallelTrajectoryConfig()
    assert config.architecture_id == "parallel_trajectory_decoder_v1"
    assert config.horizons_seconds == tuple(range(5, 61, 5))
    assert config.num_horizons == 12
    assert config.num_route_modes == 5
    assert config.d_model == 256
    assert config.n_heads == 8
    assert config.ffn_width == 1024
    assert config.n_layers == 4
    assert config.dropout == 0.1
    assert config.congestion_residual_scale == 0.25
    with pytest.raises(ValueError):
        parallel.ParallelTrajectoryConfig(n_layers=3)

    assert len(decoder.layers) == 4
    assert len({id(layer) for layer in decoder.layers}) == 4
    assert not hasattr(decoder, "causal_mask")
    for layer in decoder.layers:
        assert isinstance(layer, ParallelTrajectoryDecoderLayer)
        assert layer.congestion_residual_scale == 0.25
        for attention in (
            layer.self_attention,
            layer.memory_attention,
            layer.congestion_attention,
        ):
            assert attention.embed_dim == 256
            assert attention.num_heads == 8
            assert attention.dropout == 0.1
        assert isinstance(layer.ffn[0], nn.Linear)
        assert (layer.ffn[0].in_features, layer.ffn[0].out_features) == (256, 1024)
        assert isinstance(layer.ffn[1], nn.GELU)
        assert isinstance(layer.ffn[2], nn.Dropout) and layer.ffn[2].p == 0.1
        assert isinstance(layer.ffn[3], nn.Linear)
        assert (layer.ffn[3].in_features, layer.ffn[3].out_features) == (1024, 256)
        assert isinstance(layer.ffn[4], nn.Dropout) and layer.ffn[4].p == 0.1
    assert decoder.horizon_embedding.num_embeddings == 12
    assert decoder.horizon_embedding.embedding_dim == 256
    assert decoder.min_component_scale == MIN_COMPONENT_SCALE
    assert decoder.max_correlation_magnitude == MAX_CORRELATION_MAGNITUDE
    assert sum(value.numel() for value in decoder.parameters()) == 5_350_174


def test_self_attention_is_bidirectional_and_receives_no_causal_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = ParallelTrajectoryDecoderLayer().eval()
    observed: list[object] = []
    original = layer.self_attention.forward

    def recording(*args, **kwargs):
        observed.append(kwargs.get("attn_mask"))
        return original(*args, **kwargs)

    monkeypatch.setattr(layer.self_attention, "forward", recording)
    layer(
        torch.randn(1, 12, 256),
        torch.randn(1, 2, 256),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 3, 256),
    )
    assert observed == [None]


def test_output_shapes_probabilities_covariances_and_stationary_representation(
    decoder: parallel.ParallelTrajectoryDecoder,
) -> None:
    output = decoder(_batch())
    assert output.mode_logits.shape == (2, 5)
    assert output.mode_probabilities.shape == (2, 5)
    assert output.means.shape == (2, 5, 12, 2)
    assert output.scales.shape == (2, 5, 12, 2)
    assert output.correlations.shape == (2, 5, 12)
    assert output.primary_mode_index.shape == (2,)
    assert output.primary_route_normalized.shape == (2, 12, 2)
    assert output.marginal_mean_route_normalized.shape == (2, 12, 2)
    assert output.query_mask.shape == (2,)
    assert output.horizon_mask is None
    assert torch.allclose(output.mode_probabilities.sum(-1), torch.ones(2))
    assert torch.isfinite(output.scales).all() and (output.scales > 0).all()
    assert torch.isfinite(output.correlations).all()
    assert (output.correlations.abs() < 1).all()
    determinant = (
        output.scales[..., 0].square()
        * output.scales[..., 1].square()
        * (1.0 - output.correlations.square())
    )
    assert torch.isfinite(determinant).all() and (determinant > 0).all()

    # Near-zero means are a direct, valid representation of a stationary route.
    zero_means = torch.zeros_like(output.means)
    stationary = parallel.primary_mode_route(output.mode_probabilities, zero_means)
    assert torch.count_nonzero(stationary) == 0


def test_curved_and_moving_routes_are_direct_current_relative_means() -> None:
    time = torch.arange(1, 13, dtype=torch.float32)
    straight = torch.stack((0.2 * time, -0.1 * time), dim=-1)
    curved = torch.stack((torch.sin(time / 4), 0.03 * time.square()), dim=-1)
    means = torch.zeros(2, 5, 12, 2)
    means[0, 3] = straight
    means[1, 2] = curved
    probabilities = torch.zeros(2, 5)
    probabilities[0, 3] = 1
    probabilities[1, 2] = 1
    selected = parallel.primary_mode_route(probabilities, means)
    assert torch.equal(selected[0], straight)
    assert torch.equal(selected[1], curved)
    assert not torch.equal(selected[:, 1:] - selected[:, :-1], selected[:, :-1])


def test_memory_mask_blocks_invalid_tokens(
    decoder: parallel.ParallelTrajectoryDecoder,
) -> None:
    batch = _batch()
    baseline = decoder(batch)
    mutated_memory = batch.memory.clone()
    mutated_memory[~batch.memory_mask] = 1_000_000
    mutated = decoder(replace(batch, memory=mutated_memory))
    _assert_prediction_equal(baseline, mutated)


def test_query_order_equivariance(
    decoder: parallel.ParallelTrajectoryDecoder,
) -> None:
    batch = _batch()
    baseline = decoder(batch)
    permutation = torch.tensor([1, 0])
    permuted_batch = parallel.ParallelTrajectoryBatch(
        **{
            field.name: getattr(batch, field.name)[permutation]
            for field in fields(batch)
        }
    )
    permuted = decoder(permuted_batch)
    for name in (
        "mode_logits",
        "mode_probabilities",
        "means",
        "scales",
        "correlations",
    ):
        assert torch.equal(getattr(permuted, name), getattr(baseline, name)[permutation])


def _manual_log_density(target: Tensor, mean: Tensor, scale: Tensor, rho: Tensor) -> Tensor:
    residual = target - mean
    zx = residual[..., 0] / scale[..., 0]
    zy = residual[..., 1] / scale[..., 1]
    one_minus = 1 - rho.square()
    return (
        -math.log(2 * math.pi)
        - torch.log(scale[..., 0])
        - torch.log(scale[..., 1])
        - 0.5 * torch.log(one_minus)
        - 0.5 * (zx.square() - 2 * rho * zx * zy + zy.square()) / one_minus
    )


def test_route_level_likelihood_sums_horizons_before_mode_logsumexp() -> None:
    logits = torch.tensor([[0.3, -0.2, -2.0, -3.0, -4.0]], requires_grad=True)
    means = torch.zeros(1, 5, 12, 2, requires_grad=True)
    with torch.no_grad():
        means[0, 0, 0] = torch.tensor([0.0, 0.0])
        means[0, 0, 1] = torch.tensor([8.0, 0.0])
        means[0, 1, 0] = torch.tensor([8.0, 0.0])
        means[0, 1, 1] = torch.tensor([0.0, 0.0])
        means[0, 2:] = 20.0
    scales = torch.full((1, 5, 12, 2), 0.5, requires_grad=True)
    correlations = torch.zeros(1, 5, 12, requires_grad=True)
    targets = torch.zeros(1, 12, 2)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[:, :2] = True
    result = parallel.route_mixture_nll(
        logits, means, scales, correlations, targets, mask
    )

    component = _manual_log_density(
        targets[:, None], means, scales, correlations
    )
    expected = -torch.logsumexp(
        F.log_softmax(logits, dim=-1) + component[:, :, :2].sum(dim=-1),
        dim=-1,
    )
    wrong_switching = -(
        torch.logsumexp(
            F.log_softmax(logits, dim=-1) + component[:, :, 0], dim=-1
        )
        + torch.logsumexp(
            F.log_softmax(logits, dim=-1) + component[:, :, 1], dim=-1
        )
    )
    assert torch.allclose(result.per_route_nll, expected)
    assert not torch.allclose(result.per_route_nll, wrong_switching)
    assert result.valid_route_count == 1
    assert result.valid_horizon_count == 2
    assert torch.allclose(result.nll_per_valid_horizon, result.loss / 2)


def test_mode_permutation_invariance_and_masked_horizons_zero() -> None:
    generator = torch.Generator().manual_seed(107)
    logits = torch.randn(3, 5, generator=generator)
    means = torch.randn(3, 5, 12, 2, generator=generator)
    scales = F.softplus(torch.randn(3, 5, 12, 2, generator=generator)) + 1e-3
    correlations = 0.5 * torch.tanh(torch.randn(3, 5, 12, generator=generator))
    targets = torch.randn(3, 12, 2, generator=generator)
    mask = torch.tensor(
        [
            [True, False] * 6,
            [True] * 7 + [False] * 5,
            [False] * 12,
        ]
    )
    baseline = parallel.route_mixture_nll(
        logits, means, scales, correlations, targets, mask
    )
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = parallel.route_mixture_nll(
        logits[:, order],
        means[:, order],
        scales[:, order],
        correlations[:, order],
        targets,
        mask,
    )
    assert torch.allclose(baseline.loss, permuted.loss, atol=2e-6)
    assert torch.allclose(baseline.per_route_nll, permuted.per_route_nll, atol=2e-6)

    changed_targets = targets.clone()
    changed_targets[~mask] = 1_000_000
    changed = parallel.route_mixture_nll(
        logits, means, scales, correlations, changed_targets, mask
    )
    assert torch.equal(baseline.per_route_nll, changed.per_route_nll)
    assert baseline.per_route_nll[2].item() == 0.0


def test_empty_targets_return_differentiable_zero() -> None:
    logits = torch.randn(2, 5, requires_grad=True)
    means = torch.randn(2, 5, 12, 2, requires_grad=True)
    raw_scales = torch.randn(2, 5, 12, 2, requires_grad=True)
    scales = F.softplus(raw_scales) + 1e-3
    raw_correlations = torch.randn(2, 5, 12, requires_grad=True)
    correlations = 0.9 * torch.tanh(raw_correlations)
    result = parallel.route_mixture_nll(
        logits,
        means,
        scales,
        correlations,
        torch.zeros(2, 12, 2),
        torch.zeros(2, 12, dtype=torch.bool),
    )
    assert result.loss.item() == 0
    assert result.valid_route_count == 0
    assert result.valid_horizon_count == 0
    result.loss.backward()
    for value in (logits, means, raw_scales, raw_correlations):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) == 0


def test_equal_route_averaging_not_horizon_averaging() -> None:
    logits = torch.zeros(2, 5)
    means = torch.zeros(2, 5, 12, 2)
    scales = torch.ones(2, 5, 12, 2)
    correlations = torch.zeros(2, 5, 12)
    targets = torch.zeros(2, 12, 2)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, :4] = True
    result = parallel.route_mixture_nll(
        logits, means, scales, correlations, targets, mask
    )
    one = math.log(2 * math.pi)
    assert result.per_route_nll.tolist() == pytest.approx([one, 4 * one])
    assert result.loss.item() == pytest.approx(2.5 * one)
    assert result.nll_per_valid_horizon.item() == pytest.approx(one)


def test_primary_ties_marginal_mean_and_seeded_route_sampling() -> None:
    probabilities = torch.tensor(
        [[0.4, 0.4, 0.1, 0.1, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
    )
    means = torch.zeros(2, 5, 12, 2)
    for mode in range(5):
        means[:, mode] = mode
    primary = parallel.primary_mode_route(probabilities, means)
    assert torch.equal(primary[0], means[0, 0])
    assert torch.equal(primary[1], means[1, 3])
    marginal = parallel.marginal_mean_route(probabilities, means)
    assert torch.allclose(marginal[0], torch.full((12, 2), 0.9))
    assert torch.equal(marginal[1], means[1, 3])

    scales = torch.full_like(means, 0.02)
    correlations = torch.zeros(2, 5, 12)
    first_route, first_index = parallel.sample_route_modes(
        probabilities, means, scales, correlations, seed=109
    )
    second_route, second_index = parallel.sample_route_modes(
        probabilities, means, scales, correlations, seed=109
    )
    assert torch.equal(first_index, second_index)
    assert torch.equal(first_route, second_route)
    assert first_index[1].item() == 3
    selected = means[torch.arange(2), first_index]
    assert (first_route - selected).abs().max() < 0.1
    # One index per query proves there is no per-horizon component resampling.
    assert first_index.shape == (2,)


def test_world_coordinate_reconstruction_is_exact_unclamped_and_flagged() -> None:
    profile = _profile()
    current = torch.tensor(
        [[float(profile.world_x_min), float(profile.world_y_min)]]
    )
    normalized = torch.zeros(1, 12, 2)
    normalized[0, 0] = torch.tensor([1.0, 2.0])
    normalized[0, 1] = torch.tensor([-1.0, 0.0])
    world, valid = parallel.normalized_displacements_to_world(
        normalized, current, profile
    )
    assert world[0, 0, 0].item() == pytest.approx(
        profile.world_x_min + profile.cell_width_world_units
    )
    assert world[0, 0, 1].item() == pytest.approx(
        profile.world_y_min + 2 * profile.cell_height_world_units
    )
    assert world[0, 1, 0].item() == pytest.approx(
        profile.world_x_min - profile.cell_width_world_units
    )
    assert not valid[0, 1]
    assert world[0, 1, 0] < profile.world_x_min


def _target_source(match, profile: WorldGridProfile) -> PlannerTargetSource:
    centroids = torch.zeros(
        (match.num_timesteps, match.num_teams, 2), dtype=torch.float64
    )
    statuses: list[list[str]] = []
    for tick in range(match.num_timesteps):
        row: list[str] = []
        for team in range(match.num_teams):
            valid = (
                match.player_alive[tick, team]
                & match.player_coord_mask[tick, team]
                & match.player_slot_mask[team]
            )
            if bool(valid.any()):
                centroids[tick, team] = match.player_xyz_uu[
                    tick, team, valid, :2
                ].double().mean(0)
                row.append("valid")
            else:
                row.append("missing_centroid")
        statuses.append(row)
    return PlannerTargetSource(
        match=match,
        profile=profile,
        centroids_xy_uu=centroids,
        centroid_status=tuple(tuple(row) for row in statuses),
        lifecycle_deltas={team_id: () for team_id in match.team_ids},
        lifecycle_event_times=(),
        inferred_match_end_seconds=float(match.match_elapsed_s[-1]),
        lifecycle_complete=True,
    )


def _team_20_query(match) -> PlannerQuery:
    return PlannerQuery(match.session_id, 20, int(match.absolute_tick_index[0]))


def _phase_one_from_start(match):
    phase = match.zone_phase.clone()
    phase[0] = 1
    return replace(match, zone_phase=phase)


def test_target_builder_current_relative_conversion_and_transient_hole(
    tensorized_match,
) -> None:
    profile = _profile()
    match = _phase_one_from_start(tensorized_match)
    source = _target_source(match, profile)
    query = _team_20_query(match)
    statuses = [list(row) for row in source.centroid_status]
    statuses[1][1] = "missing_centroid"
    source = replace(source, centroid_status=tuple(tuple(row) for row in statuses))
    targets = parallel.build_parallel_trajectory_targets(source, [query])
    assert targets.query_mask.tolist() == [True]
    assert targets.target_mask[0, :3].tolist() == [False, True, True]
    assert not targets.target_mask[0, 3:].any()
    assert torch.count_nonzero(targets.target_displacements[0, 0]) == 0
    expected = (
        source.centroids_xy_uu[2, 1] - targets.current_xy[0].double()
    ) / torch.tensor(
        [profile.cell_width_world_units, profile.cell_height_world_units]
    )
    assert targets.target_displacements[0, 1].tolist() == pytest.approx(
        expected.tolist()
    )
    reconstructed, valid = parallel.normalized_displacements_to_world(
        targets.target_displacements, targets.current_xy, profile
    )
    assert torch.allclose(
        reconstructed[targets.target_mask],
        targets.future_xy[targets.target_mask],
        atol=2e-3,
        rtol=0.0,
    )
    assert valid[targets.target_mask].all()


def test_target_builder_elimination_and_phase_nine_are_suffix_masks(
    tensorized_match,
) -> None:
    profile = _profile()
    base_match = _phase_one_from_start(tensorized_match)
    query = _team_20_query(base_match)

    alive = base_match.player_alive.clone()
    coordinates = base_match.player_coord_mask.clone()
    alive[2:, 1] = False
    coordinates[2:, 1] = False
    eliminated_match = replace(
        base_match,
        player_alive=alive,
        player_coord_mask=coordinates,
    )
    eliminated = parallel.build_parallel_trajectory_targets(
        _target_source(eliminated_match, profile), [query]
    )
    assert eliminated.target_mask[0].tolist() == [True] + [False] * 11

    phase = base_match.zone_phase.clone()
    phase[2:] = 9
    phase_match = replace(base_match, zone_phase=phase)
    phase_nine = parallel.build_parallel_trajectory_targets(
        _target_source(phase_match, profile), [query]
    )
    assert phase_nine.target_mask[0].tolist() == [True] + [False] * 11


def test_target_query_eligibility_includes_phase_and_observation_policy(
    tensorized_match,
) -> None:
    profile = _profile()
    base_match = _phase_one_from_start(tensorized_match)
    source = _target_source(base_match, profile)
    query = _team_20_query(base_match)
    denied = parallel.build_parallel_trajectory_targets(
        source, [query], observation_query_mask=[False]
    )
    assert denied.query_mask.tolist() == [False]
    assert not denied.target_mask.any()

    phase = base_match.zone_phase.clone()
    phase[0] = 0
    ineligible = parallel.build_parallel_trajectory_targets(
        _target_source(replace(base_match, zone_phase=phase), profile),
        [query],
    )
    assert ineligible.query_mask.tolist() == [False]
    assert not ineligible.target_mask.any()


class _ProbeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=256)

    def forward(self, batch) -> EncoderOutput:
        roster = batch.player_slot_mask[:, None]
        alive = batch.player_alive & roster
        team_alive = alive.any(dim=-1)
        coordinates = torch.where(
            (alive & batch.player_coord_mask).unsqueeze(-1),
            batch.player_xyz_uu,
            torch.zeros_like(batch.player_xyz_uu),
        )
        local = coordinates.sum(dim=(-2, -1)) / 100_000
        lifecycle = batch.life_index.float().sum(dim=-1)
        zone = (
            batch.current_circle_uu.sum(dim=-1)
            + batch.target_circle_uu.sum(dim=-1)
            + batch.phase_times_s.sum(dim=-1)
            + batch.zone_phase.float()
        ) / 100_000
        scalar = local + lifecycle + zone[:, :, None]
        basis = torch.linspace(0.5, 1.5, 256, device=scalar.device)
        z = scalar.unsqueeze(-1) * basis
        return EncoderOutput(z=z, team_alive_mask=team_alive, time_mask=batch.time_mask)


class _TinyCongestionPredictor(nn.Module):
    def forward(self, query_state, zone_state, memory, memory_padding_mask):
        valid = (~memory_padding_mask).to(memory.dtype).unsqueeze(-1)
        pooled = (memory * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
        base = query_state.mean(-1) + zone_state.mean(-1) + pooled
        logits = base[:, None, None, None].expand(-1, 12, 1, 1)
        return logits, F.softplus(logits)


class _TinyCongestionTokenizer(nn.Module):
    def forward(self, density):
        basis = torch.linspace(0.25, 1.25, 256, device=density.device)
        return density.flatten(2).mean(-1, keepdim=True) * basis.view(1, 1, -1)


def _observation(batch) -> PlannerObservation:
    b, t, n, _ = batch.player_alive.shape
    causal = torch.zeros((b, t), dtype=torch.bool)
    causal[:, :2] = True
    own = torch.zeros((b, n), dtype=torch.bool)
    own[:, 0] = True
    query = torch.zeros((b, t, n), dtype=torch.bool)
    query[:, 1, 0] = True
    return PlannerObservation(
        causal_time_mask=causal,
        own_team_mask=own,
        observable_opponent_mask=torch.zeros((b, t, n), dtype=torch.bool),
        observable_prior_opponent_mask=torch.zeros((b, n), dtype=torch.bool),
        revealed_target_zone_mask=torch.zeros((b, t), dtype=torch.bool),
        query_mask=query,
    )


def _model() -> parallel.ParallelTrajectoryModel:
    torch.manual_seed(113)
    return parallel.ParallelTrajectoryModel(
        _profile(),
        _ProbeEncoder(),
        congestion_predictor=_TinyCongestionPredictor(),
        congestion_tokenizer=_TinyCongestionTokenizer(),
    ).eval()


def _replace_batch(batch, **updates):
    return replace(batch, **updates)


def test_composition_wrapper_sanitizes_every_future_leakage_boundary(
    encoder_batch,
) -> None:
    model = _model()
    observation = _observation(encoder_batch)
    baseline = model(encoder_batch, observation)

    future_xyz = encoder_batch.player_xyz_uu.clone()
    future_xyz[:, 2:] = 9_000_000
    _assert_prediction_equal(
        baseline,
        model(_replace_batch(encoder_batch, player_xyz_uu=future_xyz), observation),
    )

    hidden_xyz = encoder_batch.player_xyz_uu.clone()
    hidden_xyz[:, :2, 1] = -8_000_000
    _assert_prediction_equal(
        baseline,
        model(_replace_batch(encoder_batch, player_xyz_uu=hidden_xyz), observation),
    )

    target_zone = encoder_batch.target_circle_uu.clone()
    target_zone[:, :2] = 7_000_000
    _assert_prediction_equal(
        baseline,
        model(_replace_batch(encoder_batch, target_circle_uu=target_zone), observation),
    )

    future_alive = encoder_batch.player_alive.clone()
    future_life = encoder_batch.life_index.clone()
    future_alive[:, 2:] = ~future_alive[:, 2:]
    future_life[:, 2:] += 99
    _assert_prediction_equal(
        baseline,
        model(
            _replace_batch(
                encoder_batch,
                player_alive=future_alive,
                life_index=future_life,
            ),
            observation,
        ),
    )

    first_targets = parallel.ParallelTrajectoryTargets(
        torch.zeros(1, 12, 2), torch.ones(1, 12, dtype=torch.bool)
    )
    second_targets = parallel.ParallelTrajectoryTargets(
        torch.full((1, 12, 2), 123.0), torch.ones(1, 12, dtype=torch.bool)
    )
    first = model(encoder_batch, observation, first_targets)
    second = model(encoder_batch, observation, second_targets)
    _assert_prediction_equal(first, second)
    assert torch.equal(first.horizon_mask, second.horizon_mask)

    allowed = encoder_batch.player_xyz_uu.clone()
    allowed[:, 1, 0, :, 0] += 5_000
    changed = model(_replace_batch(encoder_batch, player_xyz_uu=allowed), observation)
    assert not torch.equal(baseline.mode_logits, changed.mode_logits)


def _permute_teams(batch, permutation: Tensor):
    return replace(
        batch,
        player_xyz_uu=batch.player_xyz_uu[:, :, permutation],
        player_alive=batch.player_alive[:, :, permutation],
        player_coord_mask=batch.player_coord_mask[:, :, permutation],
        life_index=batch.life_index[:, :, permutation],
        player_slot_mask=batch.player_slot_mask[:, permutation],
        team_slot_mask=batch.team_slot_mask[:, permutation],
        prior_player_xyz_uu=batch.prior_player_xyz_uu[:, permutation],
        prior_player_alive=batch.prior_player_alive[:, permutation],
        prior_player_coord_mask=batch.prior_player_coord_mask[:, permutation],
        prior_life_index=batch.prior_life_index[:, permutation],
    )


def test_player_order_invariance_and_team_permutation_equivariance(
    encoder_batch,
) -> None:
    model = _model()
    observation = _observation(encoder_batch)
    baseline = model(encoder_batch, observation)

    player_order = torch.tensor([1, 0])
    player_permuted = replace(
        encoder_batch,
        player_xyz_uu=encoder_batch.player_xyz_uu[:, :, :, player_order],
        player_alive=encoder_batch.player_alive[:, :, :, player_order],
        player_coord_mask=encoder_batch.player_coord_mask[:, :, :, player_order],
        life_index=encoder_batch.life_index[:, :, :, player_order],
        player_slot_mask=encoder_batch.player_slot_mask[:, :, player_order],
        prior_player_xyz_uu=encoder_batch.prior_player_xyz_uu[:, :, player_order],
        prior_player_alive=encoder_batch.prior_player_alive[:, :, player_order],
        prior_player_coord_mask=encoder_batch.prior_player_coord_mask[:, :, player_order],
        prior_life_index=encoder_batch.prior_life_index[:, :, player_order],
    )
    _assert_prediction_equal(baseline, model(player_permuted, observation))

    team_order = torch.tensor([1, 0])
    team_permuted = _permute_teams(encoder_batch, team_order)
    team_observation = replace(
        observation,
        own_team_mask=observation.own_team_mask[:, team_order],
        observable_opponent_mask=observation.observable_opponent_mask[:, :, team_order],
        observable_prior_opponent_mask=observation.observable_prior_opponent_mask[:, team_order],
        query_mask=observation.query_mask[:, :, team_order],
    )
    _assert_prediction_equal(
        baseline, model(team_permuted, team_observation), atol=2e-6
    )


def test_backpropagation_reaches_every_intended_decoder_group() -> None:
    torch.manual_seed(127)
    decoder = parallel.ParallelTrajectoryDecoder().train()
    batch = _batch(seed=131)
    output = decoder(batch)
    target_mask = torch.tensor(
        [[True] * 12, [True] * 8 + [False] * 4]
    )
    target_displacements = torch.randn(2, 12, 2)
    target_displacements[~target_mask] = 0
    targets = parallel.ParallelTrajectoryTargets(
        target_displacements=target_displacements,
        target_mask=target_mask,
    )
    loss = parallel.route_mixture_nll(output, targets).loss
    loss.backward()
    groups = {
        "horizon_embedding": decoder.horizon_embedding,
        "query_projection": decoder.query_projection,
        "zone_projection": decoder.zone_projection,
        "final_norm": decoder.final_norm,
        "route_norm": decoder.route_norm,
        "mode_head": decoder.mode_head,
        "mean_head": decoder.mean_head,
        "scale_head": decoder.scale_head,
        "correlation_head": decoder.correlation_head,
    }
    for index, layer in enumerate(decoder.layers):
        groups.update(
            {
                f"layer_{index}_self_attention": layer.self_attention,
                f"layer_{index}_memory_attention": layer.memory_attention,
                f"layer_{index}_congestion_attention": layer.congestion_attention,
                f"layer_{index}_ffn": layer.ffn,
                f"layer_{index}_norms": nn.ModuleList(
                    [
                        layer.self_attention_norm,
                        layer.memory_attention_norm,
                        layer.congestion_attention_norm,
                        layer.ffn_norm,
                    ]
                ),
            }
        )
    for name, module in groups.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert gradients and all(value is not None for value in gradients), name
        assert all(torch.isfinite(value).all() for value in gradients if value is not None), name
        norm = sum(float(value.abs().sum()) for value in gradients if value is not None)
        assert norm > 0, name


def _source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def test_protected_source_digests_are_unchanged() -> None:
    assert _source_tree_digest(ROOT / "ml" / "src" / "fortnite_encoder") == ENCODER_DIGEST
    assert _source_tree_digest(PACKAGE_ROOT) == DECODER_DIGEST


def test_package_contains_no_rejected_mechanism_or_training_entrypoint() -> None:
    banned_identifiers = (
        "teacher_forcing",
        "self_condition",
        "movement_gate",
        "movement_head",
        "movement_bce",
        "beam_search",
        "previous_waypoint",
        "recurrent",
        "recurrence",
    )
    python_files = tuple(PACKAGE_ROOT.glob("*.py"))
    assert python_files
    assert not any(
        path.name in {"train.py", "training.py", "checkpoint.py"}
        for path in python_files
    )
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name.lower())
            if isinstance(node, ast.arg):
                names.append(node.arg.lower())
            if isinstance(node, ast.Name):
                names.append(node.id.lower())
            for name in names:
                assert not any(value in name for value in banned_identifiers), (
                    path.name,
                    name,
                )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"save", "save_checkpoint"}
            for node in ast.walk(tree)
        )
