from __future__ import annotations

import torch

from fortnite_encoder import EncoderConfig
from fortnite_encoder.temporal import TemporalAttentionBlock, TemporalEncoder


def _temporal_config() -> EncoderConfig:
    return EncoderConfig(
        d_model=4,
        n_heads=1,
        player_embedding_dim=4,
        edge_hidden_dim=4,
        ffn_dim=8,
        spatial_layers=1,
        temporal_layers=1,
        dropout=0.0,
        max_timesteps=32,
        max_zone_phase=4,
    )


def test_future_perturbation_cannot_change_earlier_outputs() -> None:
    config = _temporal_config()
    encoder = TemporalEncoder(config).eval()
    state = torch.randn(1, 4, 1, 4)
    alive = torch.ones(1, 4, 1, dtype=torch.bool)
    time_mask = torch.ones(1, 4, dtype=torch.bool)
    ticks = torch.tensor([[3, 4, 5, 6]])
    original = encoder(state, alive, time_mask, ticks)
    changed_state = state.clone()
    changed_state[:, 3] = 10_000.0
    changed = encoder(changed_state, alive, time_mask, ticks)
    assert torch.allclose(original[:, :3], changed[:, :3], atol=1e-6)


def test_relative_bias_uses_absolute_tick_lags_and_excludes_dead_gap() -> None:
    config = _temporal_config()
    block = TemporalAttentionBlock(config).eval()
    with torch.no_grad():
        block.query.weight.zero_()
        block.query.bias.zero_()
        block.key.weight.zero_()
        block.key.bias.zero_()
        block.value.weight.copy_(torch.eye(4))
        block.value.bias.zero_()
        block.output.weight.copy_(torch.eye(4))
        block.output.bias.zero_()
        for parameter in block.ffn.parameters():
            parameter.zero_()
        block.relative_lag_bias.zero_()
        block.relative_lag_bias[0, 0] = 0.0
        block.relative_lag_bias[0, 3] = 1.0
        block.relative_lag_bias[0, 5] = 2.0

    state = torch.tensor(
        [[[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]]]
    )
    alive = torch.tensor([[[True], [False], [True]]])
    times = torch.ones(1, 3, dtype=torch.bool)
    ticks = torch.tensor([[10, 12, 15]])
    result = block(state, alive, times, ticks)

    normalized = block.attention_norm(state)
    weights = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
    expected_message = (
        weights[0] * normalized[0, 0, 0]
        + weights[1] * normalized[0, 0, 2]
    )
    assert torch.allclose(
        result[0, 0, 2], state[0, 0, 2] + expected_message, atol=1e-6
    )
    assert torch.equal(result[0, 0, 1], torch.zeros(4))


def test_padding_is_zero_and_window_history_is_truncated() -> None:
    config = _temporal_config()
    encoder = TemporalEncoder(config).eval()
    state = torch.randn(1, 3, 1, 4)
    alive = torch.ones(1, 3, 1, dtype=torch.bool)
    ticks = torch.tensor([[4, 5, 6]])
    full = encoder(state, alive, torch.ones(1, 3, dtype=torch.bool), ticks)
    window = encoder(
        state[:, 1:],
        alive[:, 1:],
        torch.ones(1, 2, dtype=torch.bool),
        ticks[:, 1:],
    )
    assert not torch.allclose(full[:, 2], window[:, 1])

    padded_state = torch.cat((state, torch.full((1, 1, 1, 4), float("nan"))), dim=1)
    padded_alive = torch.cat(
        (alive, torch.ones(1, 1, 1, dtype=torch.bool)), dim=1
    )
    padded_ticks = torch.tensor([[4, 5, 6, 999]])
    padded = encoder(
        padded_state,
        padded_alive,
        torch.tensor([[True, True, True, False]]),
        padded_ticks,
    )
    assert torch.allclose(full, padded[:, :3], atol=1e-6)
    assert torch.equal(padded[:, 3], torch.zeros_like(padded[:, 3]))
