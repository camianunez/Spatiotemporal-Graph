# Parallel trajectory decoder v1

Architecture identifier: `parallel_trajectory_decoder_v1`  
Target schema: `parallel-trajectory-targets:1.0`  
Package: `ml/src/fortnite_parallel_trajectory`

## Decision and scope

The predecessor decoded waypoints autoregressively. Its own coordinate decisions
were fed into later steps, so a deterministic early error changed the state from
which every later waypoint was predicted. Teacher-forced prefixes could hide that
drift during fitting, while self-conditioned prefixes reproduced it at inference.
That formulation, its HOLD/MOVE branch, per-horizon component choice, sequence
beam search, R4 adapter, and R5 local correction are not part of this package.

Version 1 predicts the complete 60-second route in one parallel operation. It is
an architecture, target, loss, and inference package only. It has no training
runner, checkpoint reader/writer, real-corpus evaluation path, validation-data
path, or test-data path.

## Causal input boundary

`ParallelTrajectoryModel` first calls the unchanged public
`apply_planner_input_policy` sanitizer, then the injected unchanged
`SpatiotemporalEncoder`. It gathers only selected own-team query states and
causal sanitized memory. The injected congestion predictor and tokenizer create
the congestion tokens consumed by the new decoder. Constructors do no checkpoint
I/O.

The target-free decoder batch contains:

| Tensor | Shape | Meaning |
| --- | ---: | --- |
| `z_query` | `[Q,256]` | selected causal own-team query state |
| `memory` | `[Q,M,256]` | padded sanitized causal encoder memory |
| `memory_mask` | `[Q,M]` | `True` for readable memory tokens |
| `current_xy` | `[Q,2]` | valid query-time team centroid in world units |
| `zone_state` | `[Q,15]` | existing revealed-zone vector |
| `congestion_tokens` | `[Q,C,256]` | all tokens emitted by the congestion tokenizer |
| `query_mask` | `[Q]` | current-query eligibility |

Future encoder states, hidden opponent state, unrevealed target zones,
post-query lifecycle state, identity, placement, rank, survival labels, and
supervision tensors are outside this decoder input. The low-level decoder never
receives a future route prefix. The composition wrapper attaches a supplied
target horizon mask to the returned metadata only after causal features have
been prepared; it does not use that mask to compute distribution parameters.

## Parallel horizon-query architecture

The fixed horizons are 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, and 60
seconds. For every query and horizon (t), the initial prediction slot is

\[
s_t^{(0)} = e_t + W_q z_q + W_z z_{zone}.
\]

There are 12 learned 256-dimensional horizon embeddings. They contain no
observed future values. All horizon slots attend bidirectionally to one another;
this is joint prediction, not temporal leakage.

Exactly four independent pre-LayerNorm layers apply, in order:

1. eight-head bidirectional self-attention over all 12 horizon slots;
2. eight-head cross-attention to sanitized causal encoder memory, with invalid
   padded keys masked;
3. eight-head cross-attention to all predicted congestion tokens, whose residual
   has the fixed multiplier 0.25; and
4. `Linear(256,1024) -> GELU -> Dropout(0.1) -> Linear(1024,256) -> Dropout(0.1)`.

Each attention module also uses dropout 0.1. Residuals follow each sublayer and a
final LayerNorm follows layer four. There is no horizon causal mask, recurrent
state, or predicted-coordinate feedback.

## Route-level Gaussian mixture

The final horizon state has shared projections for five means, five pairs of
scales, and five correlations. Tensor layouts are mode-major:

| Output | Shape |
| --- | ---: |
| `mode_logits` | `[Q,5]` |
| `mode_probabilities` | `[Q,5]` |
| `means` | `[Q,5,12,2]` |
| `scales` | `[Q,5,12,2]` |
| `correlations` | `[Q,5,12]` |

The route mixture weights come from one pooled route representation:

\[
r = \operatorname{LayerNorm}\left(\frac{1}{12}\sum_t s_t\right),\qquad
\pi = \operatorname{softmax}(W_\pi r).
\]

Mode index (k) denotes one coherent route across all horizons. It is never
selected independently by horizon. Mode indices are exchangeable: a consistent
permutation has no probabilistic effect, and an index has no required semantic
meaning across separate runs. Scales use the existing validated repository
contract `softplus(raw_scale) + 1e-3`; correlations use
`0.999 * tanh(raw_correlation)`. Thus both scales are positive and finite,
`abs(rho) < 1`, and

\[
\Sigma_{k,t}=\begin{bmatrix}
\sigma_x^2 & \rho\sigma_x\sigma_y\\
\rho\sigma_x\sigma_y & \sigma_y^2
\end{bmatrix}
\]

has determinant
\(\sigma_x^2\sigma_y^2(1-\rho^2)>0\).

## Full-match targets and masking

Targets are constructed from a validated `PlannerTargetSource` while the full
match is still available, before causal query windows are sliced. A query is
eligible only when the focal team and time slots are valid, the team is alive,
the current living-player centroid is finite and inside the canonical grid, the
phase is 1 through 8, and the caller-supplied observation-policy mask passes.

For current centroid ((x_0,y_0)) and future centroid ((x_t,y_t)), the direct
current-relative target is

\[
\Delta x_t=(x_t-x_0)/w_{cell},\qquad
\Delta y_t=(y_t-y_0)/h_{cell}.
\]

These are not incremental displacements between adjacent predicted horizons.
Each valid horizon requires an exact recorded timestamp, a finite available
future centroid, an alive and lifecycle-consistent team, phase 1 through 8, and
a coordinate inside the reviewed grid envelope. Elimination, lifecycle
ambiguity, recording end, or phase 9 masks the current horizon and all later
horizons. A transient missing/nonfinite/out-of-envelope coordinate or an internal
timestamp gap masks only that horizon when later lifecycle continuity remains
valid. Invalid displacement slots are zero and have no class or terminal token.

World reconstruction uses the canonical `WorldGridProfile`:

\[
x_{k,t}=x_0+\mu^x_{k,t}w_{cell},\qquad
y_{k,t}=y_0+\mu^y_{k,t}h_{cell}.
\]

The conversion helpers return the unmodified coordinates plus a separate
in-envelope validity mask. They never clamp predictions.

## Primary objective

All likelihood-critical operations are explicitly converted to FP32. For every
mode, valid horizon log densities are summed first:

\[
c_{q,k}=\sum_t m_{q,t}\log\mathcal{N}_2
(\Delta Y_{q,t};\mu_{q,k,t},\Sigma_{q,k,t}).
\]

Only then is the route mixture marginalized:

\[
\operatorname{NLL}_q=-\operatorname{logsumexp}_k
(\log\operatorname{softmax}(a_q)_k+c_{q,k}).
\]

`route_mixture_nll` averages equally over eligible routes having at least one
valid horizon. It returns the raw mean loss, valid-route count, valid-horizon
count, per-route NLL, and total NLL divided by valid horizons as a diagnostic.
An empty mask returns a differentiable zero connected to every distribution
tensor. No movement, path, congestion, smoothness, ranking, ADE, FDE, or learned
task-weight objective is present.

## Inference semantics

`primary_mode_route` uses the single highest-probability route index for all 12
horizons. Exact probability ties resolve to the lower index through the standard
first-maximum rule, without modifying probabilities.

`marginal_mean_route` computes
\(\sum_k \pi_k\mu_{k,t}\). It is explicitly a marginal mean and is not described
as a coherent mixture component.

`sample_route_modes` draws one categorical route index per query, reuses that
index across all horizons, and then samples the 12 correlated bivariate
Gaussians belonging to that route. It accepts either an explicit generator or a
seed and never performs sequence search.

## Version 1 exclusions

Version 1 intentionally excludes full or corpus training, checkpoint format
changes, automatic checkpoint loading, real validation/test evaluation,
autoregressive recurrence, future waypoint prefixes, self-conditioning,
HOLD/MOVE prediction, hard movement decisions, movement-scaled coordinates,
per-horizon mixture switching, sequence beam search, terminal classes, auxiliary
losses, R4 adapters, and R5 corrections.
