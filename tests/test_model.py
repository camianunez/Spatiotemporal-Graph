from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from fortnite_encoder import (
    EncoderConfig,
    SpatiotemporalEncoder,
    collate_encoder_inputs,
    slice_window,
)


def _permute_teams(batch, permutation: torch.Tensor):
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


def test_end_to_end_mixed_batch_has_default_contract(tensorized_match) -> None:
    short = slice_window(tensorized_match, 1, 2)
    collated = collate_encoder_inputs([tensorized_match, short])
    model = SpatiotemporalEncoder(EncoderConfig(dropout=0.0)).eval()
    output = model(collated.batch)
    assert output.z.shape == (2, 4, 2, 256)
    assert output.team_alive_mask.shape == (2, 4, 2)
    assert output.time_mask.shape == (2, 4)
    assert torch.isfinite(output.z).all()
    invalid = ~(output.team_alive_mask & output.time_mask[:, :, None])
    assert torch.equal(output.z[invalid], torch.zeros_like(output.z[invalid]))


def test_team_permutation_equivariance(encoder_batch, small_config) -> None:
    model = SpatiotemporalEncoder(small_config).eval()
    original = model(encoder_batch)
    permutation = torch.tensor([1, 0])
    permuted = model(_permute_teams(encoder_batch, permutation))
    assert torch.allclose(
        permuted.z, original.z[:, :, permutation], atol=2e-6
    )
    assert torch.equal(
        permuted.team_alive_mask,
        original.team_alive_mask[:, :, permutation],
    )


def test_player_permutation_invariance(encoder_batch, small_config) -> None:
    model = SpatiotemporalEncoder(small_config).eval()
    original = model(encoder_batch)
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
    permuted = model(swapped)
    assert torch.allclose(original.z, permuted.z, atol=2e-6)


def test_arbitrary_padded_values_do_not_affect_valid_outputs(
    tensorized_match, small_config
) -> None:
    short = slice_window(tensorized_match, 1, 2)
    batch = collate_encoder_inputs([tensorized_match, short]).batch
    model = SpatiotemporalEncoder(small_config).eval()
    original = model(batch)

    xyz = batch.player_xyz_uu.clone()
    xyz[1, 2:] = float("nan")
    alive = batch.player_alive.clone()
    alive[1, 2:] = True
    coords = batch.player_coord_mask.clone()
    coords[1, 2:] = True
    life = batch.life_index.clone()
    life[1, 2:] = -1
    current = batch.current_circle_uu.clone()
    target = batch.target_circle_uu.clone()
    current[1, 2:] = float("nan")
    target[1, 2:] = float("nan")
    ticks = batch.absolute_tick_index.clone()
    phase = batch.zone_phase.clone()
    ticks[1, 2:] = 10_000
    phase[1, 2:] = 10_000
    changed_batch = replace(
        batch,
        player_xyz_uu=xyz,
        player_alive=alive,
        player_coord_mask=coords,
        life_index=life,
        current_circle_uu=current,
        target_circle_uu=target,
        absolute_tick_index=ticks,
        zone_phase=phase,
    )
    changed = model(changed_batch)
    valid = original.time_mask[:, :, None] & original.team_alive_mask
    assert torch.allclose(original.z[valid], changed.z[valid], atol=1e-6)
    assert torch.isfinite(changed.z).all()


def test_end_to_end_is_causal(encoder_batch, small_config) -> None:
    model = SpatiotemporalEncoder(small_config).eval()
    original = model(encoder_batch)
    xyz = encoder_batch.player_xyz_uu.clone()
    xyz[:, 3] += 1_000_000.0
    current = encoder_batch.current_circle_uu.clone()
    current[:, 3, :2] -= 500_000.0
    changed = model(
        replace(
            encoder_batch,
            player_xyz_uu=xyz,
            current_circle_uu=current,
        )
    )
    assert torch.allclose(original.z[:, :3], changed.z[:, :3], atol=1e-6)


def test_capacity_indices_are_rejected_not_clamped(
    encoder_batch, small_config
) -> None:
    model = SpatiotemporalEncoder(small_config)
    ticks = encoder_batch.absolute_tick_index.clone()
    ticks[0, 0] = small_config.max_timesteps
    with pytest.raises(ValueError, match="tick"):
        model(replace(encoder_batch, absolute_tick_index=ticks))
    phases = encoder_batch.zone_phase.clone()
    phases[0, 0] = small_config.max_zone_phase + 1
    with pytest.raises(ValueError, match="phase"):
        model(replace(encoder_batch, zone_phase=phases))


def test_all_masked_batch_remains_finite_and_zero(
    encoder_batch, small_config
) -> None:
    time_mask = torch.zeros_like(encoder_batch.time_mask)
    output = SpatiotemporalEncoder(small_config)(
        replace(encoder_batch, time_mask=time_mask)
    )
    assert torch.isfinite(output.z).all()
    assert torch.equal(output.z, torch.zeros_like(output.z))


def test_configuration_rejects_invalid_dimensions_and_scales() -> None:
    with pytest.raises(ValueError, match="divisible"):
        EncoderConfig(d_model=30, n_heads=8)
    with pytest.raises(ValueError, match="dropout"):
        EncoderConfig(dropout=1.0)
    with pytest.raises(ValueError, match="xy_scale"):
        EncoderConfig(xy_scale_uu=math.nan)
    with pytest.raises(ValueError, match="neighbor_mode"):
        EncoderConfig(neighbor_mode="ranked")  # type: ignore[arg-type]
