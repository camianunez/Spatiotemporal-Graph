from __future__ import annotations

import torch

from fortnite_parallel_trajectory import (
    ParallelTrajectoryBatch,
    ParallelTrajectoryDecoder,
    ParallelTrajectoryTargets,
    route_mixture_nll,
)


def _synthetic_routes() -> tuple[ParallelTrajectoryBatch, ParallelTrajectoryTargets]:
    q = 10
    context_ids = (0, 1, 2, 3, 4, 5, 6, 6, 7, 7)
    z_query = torch.zeros(q, 256)
    for row, context in enumerate(context_ids):
        z_query[row, context] = 3.0
        z_query[row, 16 + context] = -1.5
    zone_state = torch.zeros(q, 15)
    for row, context in enumerate(context_ids):
        zone_state[row, context % 15] = 0.25 * (context + 1)
    memory = torch.stack(
        (z_query * 0.5, z_query * -0.25), dim=1
    )
    congestion_tokens = torch.stack(
        (z_query * 0.1, z_query * 0.2), dim=1
    )
    batch = ParallelTrajectoryBatch(
        z_query=z_query,
        memory=memory,
        memory_mask=torch.ones(q, 2, dtype=torch.bool),
        current_xy=torch.zeros(q, 2),
        zone_state=zone_state,
        congestion_tokens=congestion_tokens,
        query_mask=torch.ones(q, dtype=torch.bool),
    )

    u = torch.arange(1, 13, dtype=torch.float32) / 12.0
    stationary = torch.zeros(12, 2)
    straight_right = torch.stack((1.20 * u, 0.20 * u), dim=-1)
    straight_diagonal = torch.stack((-0.75 * u, 0.95 * u), dim=-1)
    curve_left = torch.stack(
        (-0.65 * torch.sin(torch.pi * u), 1.10 * u), dim=-1
    )
    curve_right = torch.stack(
        (0.65 * torch.sin(torch.pi * u), 1.10 * u), dim=-1
    )
    branch_up = torch.stack((0.85 * u, 1.40 * u), dim=-1)
    branch_down = torch.stack((0.85 * u, -1.40 * u), dim=-1)
    hook_left = torch.stack(
        (-0.90 * u, 0.45 * torch.sin(torch.pi * u)), dim=-1
    )
    hook_right = torch.stack(
        (0.90 * u, 0.45 * torch.sin(torch.pi * u)), dim=-1
    )
    target = torch.stack(
        (
            stationary,
            stationary,
            straight_right,
            straight_diagonal,
            curve_left,
            curve_right,
            branch_up,
            branch_down,
            hook_left,
            hook_right,
        )
    )
    mask = torch.tensor(
        [
            [True] * 12,
            [True] * 7 + [False] * 5,
            [True] * 12,
            [True, True, False, True, True, True, False, True, True, True, True, True],
            [True] * 9 + [False] * 3,
            [True, False] * 6,
            [True] * 12,
            [True] * 12,
            [True] * 12,
            [True] * 12,
        ],
        dtype=torch.bool,
    )
    target = torch.where(mask.unsqueeze(-1), target, torch.zeros_like(target))
    return batch, ParallelTrajectoryTargets(target, mask)


def _coherent_mode_errors(
    means: torch.Tensor,
    targets: ParallelTrajectoryTargets,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(
        means - targets.target_displacements[:, None], dim=-1
    )
    weighted = torch.where(
        targets.target_mask[:, None], distances, torch.zeros_like(distances)
    )
    return weighted.sum(-1) / targets.target_mask.sum(-1, keepdim=True)


def test_deterministic_synthetic_route_mixture_overfit() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(211)
        batch, targets = _synthetic_routes()
        decoder = ParallelTrajectoryDecoder().eval()
        with torch.no_grad():
            decoder.mode_head.weight.zero_()
            decoder.mode_head.bias.zero_()
            decoder.mean_head.bias.copy_(
                torch.tensor(
                    [0.0, 0.0, 0.0, 0.9, 0.0, -0.9, 0.9, 0.0, -0.9, 0.0]
                )
            )
            decoder.scale_head.weight.zero_()
            decoder.scale_head.bias.fill_(-1.5)
            decoder.correlation_head.weight.zero_()
            decoder.correlation_head.bias.zero_()
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        for module in (decoder.mode_head, decoder.mean_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(
            (*decoder.mode_head.parameters(), *decoder.mean_head.parameters()),
            lr=2e-2,
        )
        with torch.no_grad():
            initial_output = decoder.eval()(batch)
            initial_nll = float(route_mixture_nll(initial_output, targets).loss)
        for _ in range(400):
            optimizer.zero_grad(set_to_none=True)
            output = decoder(batch)
            loss = route_mixture_nll(output, targets).loss
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()

        decoder.eval()
        with torch.no_grad():
            final_output = decoder(batch)
            final_loss = route_mixture_nll(final_output, targets)
            errors = _coherent_mode_errors(final_output.means, targets)
            best_error, best_mode = errors.min(dim=-1)

        print(
            {
                "initial_route_nll": initial_nll,
                "final_route_nll": float(final_loss.loss),
                "best_coherent_route_errors": best_error.tolist(),
                "multimodal_selected_modes": best_mode[6:].tolist(),
                "multimodal_mode_probabilities": final_output.mode_probabilities[
                    [6, 8]
                ].tolist(),
            }
        )

        assert final_loss.loss.item() < initial_nll - 8.0
        assert torch.isfinite(final_output.mode_logits).all()
        assert torch.isfinite(final_output.means).all()
        assert torch.isfinite(final_output.scales).all()
        assert torch.isfinite(final_output.correlations).all()
        assert best_error[:2].max().item() < 0.12
        assert best_error[2:6].max().item() < 0.20

        # Rows 6/7 and 8/9 are identical inputs with genuinely different
        # complete routes.  Each target must be explained by one aggregate-best
        # mode, and the paired targets must use distinguishable modes.
        assert best_error[6:].max().item() < 0.22
        assert best_mode[6].item() != best_mode[7].item()
        assert best_mode[8].item() != best_mode[9].item()
        for left, right in ((6, 7), (8, 9)):
            assert torch.equal(final_output.mode_logits[left], final_output.mode_logits[right])
            assert torch.equal(final_output.means[left], final_output.means[right])
            selected = final_output.mode_probabilities[left, best_mode[[left, right]]]
            assert (selected > 0.08).all()
            route_separation = torch.linalg.vector_norm(
                final_output.means[left, best_mode[left]]
                - final_output.means[left, best_mode[right]],
                dim=-1,
            ).mean()
            assert route_separation.item() > 0.35

        # This is explicitly a coherent-route check: errors were minimized once
        # over K after aggregating all valid horizons, never independently by t.
        selected_routes = final_output.means[
            torch.arange(final_output.means.shape[0]), best_mode
        ]
        selected_distance = torch.linalg.vector_norm(
            selected_routes - targets.target_displacements, dim=-1
        )
        coherent_error = torch.where(
            targets.target_mask, selected_distance, torch.zeros_like(selected_distance)
        ).sum(-1) / targets.target_mask.sum(-1)
        assert torch.allclose(coherent_error, best_error)

    finally:
        torch.set_num_threads(previous_threads)
