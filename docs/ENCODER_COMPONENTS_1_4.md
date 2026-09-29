# Encoder components 1–4

This document defines the leakage-safe tensorization, feature, spatial, and
temporal contracts implemented by `ml/src/fortnite_encoder`. The encoder
consumes normalized Parquet dataset v2 sessions and returns

\[
Z\in\mathbb{R}^{B\times T\times N\times256}.
\]

It has no training loop, targets, losses, decoder, placement head, or map-cell
representation. Normalized-map cells remain entirely decoder-side.

## Input boundary

`load_match_session(path)` opens exactly four files:

- `match_samples.parquet`
- `player_samples.parquet`
- `zone_samples.parquet`
- `zone_phases.parquet`

It never opens `team_samples.parquet`, which contains `final_placement`;
`player_events.parquet`; or `centroid_neighbors.parquet`. Centroids and
neighbors are recomputed from player telemetry. Player IDs are used only to
form stable roster slots and are discarded. Team IDs are retained only in
CPU-side `BatchMetadata`; `EncoderBatch`, the model input, contains no IDs.

The loader validates the exact v2 PyArrow schemas, a complete five-second grid
starting at battle-bus time zero, unique and complete tick joins, duo capacity,
finite complete coordinate triples, lifecycle transitions and survival counts,
and active-zone/phase consistency.

`slice_window(match, start_tick, length)` addresses ticks by their absolute
zero-based match index. A window retains the immediately preceding player
state as a motion-only sidecar. It does not add that tick to temporal
attention. `collate_encoder_inputs(matches)` pads the time and team axes.

## Raw tensors and masks

The duo roster dimension is fixed at \(R=2\).

| Field | Shape | Meaning |
| --- | ---: | --- |
| `player_xyz_uu` | `[B,T,N,R,3]` | Raw Unreal XYZ; invalid triples are zero |
| `player_alive` | `[B,T,N,R]` | Lifecycle-authoritative alive state |
| `player_coord_mask` | `[B,T,N,R]` | Complete finite XYZ is available |
| `life_index` | `[B,T,N,R]` | Zero-based pawn life |
| `player_slot_mask` | `[B,N,R]` | Real roster slot |
| `current_circle_uu` | `[B,T,4]` | Current `(x,y,z,r)` |
| `target_circle_uu` | `[B,T,4]` | Current phase target `(x,y,z,r)` |
| `zone_mask` | `[B,T]` | An active phase/circle is available |
| `zone_phase` | `[B,T]` | Phase number; zero before phase one |
| `phase_times_s` | `[B,T,3]` | Activation, shrink start, closure |
| `match_elapsed_s` | `[B,T]` | Seconds from battle-bus start |
| `players_remaining` | `[B,T]` | Lifecycle-derived count |
| `teams_remaining` | `[B,T]` | Lifecycle-derived count |
| `absolute_tick_index` | `[B,T]` | Original match tick, including in windows |
| `time_mask` | `[B,T]` | Real, non-padded timestep |
| `team_slot_mask` | `[B,N]` | Real, non-padded team |

The preceding-state sidecar has player shapes `[B,N,R,3]` and `[B,N,R]`,
plus `prior_state_available[B]`. All masks use `True` for valid or eligible.
Attention rows with no eligible keys have exactly zero probabilities.

## Fixed scales and team geometry

All matches use fixed scales:

\[
L_{xy}=L_z=100{,}000\ {\rm UU},\qquad L_t=1{,}800\ {\rm s}.
\]

No per-match, dataset-fitted, or map-calibrated statistic is used.
For living-player set \(A_{i,t}\),

\[
c_{i,t}=\frac{1}{|A_{i,t}|}\sum_{r\in A_{i,t}}p_{i,r,t}.
\]

`centroid_valid` requires a living team and valid coordinates for every living
player. A survivor supplies a valid centroid; a living teammate with missing
telemetry invalidates it. Motion is

\[
v_{i,t}=\frac{c^{xy}_{i,t}-c^{xy}_{i,t-1}}{L_{xy}}.
\]

It is valid only when both centroids are valid and living membership and all
roster life indices are unchanged. Consequently, death and reboot transitions
cannot appear as physical velocity. Speed is \(\lVert v\rVert_2\), and

\[
\sin\theta=v_y/\lVert v\rVert_2,\qquad
\cos\theta=v_x/\lVert v\rVert_2.
\]

Heading is invalid and zero below `epsilon`. Duo separation is XY distance
divided by \(L_{xy}\), valid only for two living players with coordinates.

## Node features

Every living player is transformed by the same MLP. No ID or roster-slot
embedding is present:

\[
u_{i,r,t}=[
\Delta x/L_{xy},\Delta y/L_{xy},\Delta z/L_z,
m^{coord},m^{offset},\log(1+\mathrm{life}),\mathbf1[\mathrm{life}>0]].
\]

Offsets are relative to a valid team centroid. The shared
\(7\rightarrow64\rightarrow64\) player MLP is mean-pooled over all living
players, including living players with missing coordinates. This makes player
slot order irrelevant.

The 16 centroid features are normalized XYZ; normalized XY motion; speed;
heading sine and cosine; one-hot living counts zero, one, and two; normalized
duo separation; and centroid, velocity, heading, and separation validity.

For current and target circles \(b\),

\[
u_x^b=(c_x-c_x^b)/r_b,\quad
u_y^b=(c_y-c_y^b)/r_b,\quad
d_{\rm edge}^b=\sqrt{(u_x^b)^2+(u_y^b)^2}-1.
\]

Each circle contributes
`[ux, uy, d_edge, inside, r/L_xy, metric_valid]`. Ratios are zero when
the centroid/circle is unavailable or radius is zero, while radius and
`zone_active` remain represented. Four more zone values are nonnegative time
to movement and closure divided by \(L_t\), clipped shrink progress, and
`zone_active`, for 16 zone features total.

Four match features broadcast across teams: remaining players divided by
roster players, remaining teams divided by roster teams, elapsed time divided
by \(L_t\), and phase divided by `max_zone_phase`.

At default dimensions the concatenated width is

\[
64_{\rm player}+16_{\rm centroid}+16_{\rm zone}+4_{\rm match}=100.
\]

The node MLP is `Linear(100,256) → GELU → Linear(256,256)`. Learned absolute
tick and phase embeddings are added. Phase zero is a zero padding embedding.
Valid indices must be below configured capacities; indices are rejected, not
clamped. Dead and padded node states are zeroed.

## Dense edges and adjacency

Edge offsets and distances use normalized XY coordinates. For valid distinct
centroids,

\[
r_{ij}=(c_j^{xy}-c_i^{xy})/L_{xy},\quad d_{ij}=\lVert r_{ij}\rVert_2,
\]
\[
\Delta v_{ij}=v_j-v_i,\qquad
c_{ij}=-\frac{r_{ij}^{\mathsf T}\Delta v_{ij}}
{\max(d_{ij},\epsilon)}.
\]

Positive \(c_{ij}\) means the teams approach. If either motion is invalid,
relative velocity and closing speed are zero. At zero distance, bearing and
closing speed are zero. The exact edge vector is

\[
e_{ij}=[
\Delta x,\Delta y,d,\sin\theta_{ij},\cos\theta_{ij},
\Delta v_x,\Delta v_y,c_{ij}].
\]

Dense features have shape `[B,T,N,N,8]`; adjacency is `[B,T,N,N]`. Both nodes
must be alive with valid centroids, and self edges are excluded. `full` connects
all valid pairs. `knn` selects the nearest configured \(k\) and includes every
candidate tied at the kth distance; it uses no ID-derived tiebreaker.

## Spatial attention

Each of two default independent layers has eight heads of width 32. With
pre-normalized node states,

\[
s^a_{ij}=\frac{(q_i^a)^\mathsf T k_j^a}{\sqrt{32}}
+\mathrm{MLP}_{edge}(e_{ij})_a,
\]
\[
g_i^a=\sum_j\alpha^a_{ij}
\left(v_j^a+(W_Ee_{ij})^a\right).
\]

Masked softmax runs over eligible neighbors. Concatenated head messages pass
through \(W_O\), a residual connection, and a pre-normalized
`256 → 1024 → 256` GELU FFN residual. A living node with no valid centroid has
zero attention update but retains its local residual/FFN path.

## Causal temporal attention

Spatial output is rearranged to `[B,N,T,256]`. Each of four default blocks has
eight heads and a learned per-head relative-lag table:

\[
a^a_{t,\tau}=
\frac{(q_t^a)^\mathsf T k_\tau^a}{\sqrt{32}}
+b^a_{k_t-k_\tau}.
\]

A key is eligible only when \(\tau\le t\), the key timestep is valid, and the
team is alive at the key. The query must likewise be valid and alive. Absolute
ticks select lag biases, so windows preserve match-time position. Temporal
attention never reaches before a window; only the first motion feature can use
the sidecar. Dead-gap keys are excluded, while a rebooted living query may
attend to earlier living states.

After four residual attention/FFN blocks, final LayerNorm is applied and invalid
outputs are exactly zero.

## Invariance contract

- Permuting player slots leaves every valid output unchanged.
- Permuting teams applies the same permutation to outputs.
- Tie-inclusive kNN preserves team permutation equivariance.
- Changing dead or padded values cannot affect valid outputs.
- Changing future inputs cannot affect earlier outputs.
- Missing numeric values are zero-filled before neural use and represented by
  explicit validity features.
- The model is deliberately not translation-, rotation-, or
  reflection-invariant because absolute world axes are meaningful node inputs.
  Edge and zone-relative subsets retain their stated relative geometry.

Permutation checks must run in evaluation mode so dropout does not obscure the
deterministic architectural invariant.

`EncoderConfig.dropout` remains the compatibility fallback for every encoder
dropout location. Training can override feature-projection, spatial-attention,
spatial-residual/FFN, temporal-attention, and temporal-residual/FFN
probabilities independently. Feature dropout is applied only after projected
node features and time/phase embeddings are combined; masks, geometry,
targets, and final logits are never dropout inputs.
