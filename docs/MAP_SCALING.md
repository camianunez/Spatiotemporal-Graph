# Engineered model-world scaling

Dataset v2 preserves raw Unreal `{x,y,z}` values. Rotation supervision maps
raw XY directly into the published `model-world-grid-v1` partition. It does
not use an image, pixels, control points, observed extrema, or an affine fit.

## What is verified and what is engineered

The following are verified coordinate facts for the supported corpus:

- the exact replay profiles are 41.10/55434016 and 41.20/55798846;
- all 63 supported replays identify
  `/Hera_Map/Maps/Hera_V2_Terrain`;
- replicated `FortPoiManager` spacing is 512 Unreal units on both axes;
- Unreal X maps to columns, Unreal Y maps to rows, and both increase with
  their indices;
- one Unreal world unit is 0.01 m; and
- all reported metadata extents are contained by
  `[-131072,131072]` on X and Y.

The envelope itself is an engineered model partition. It is deliberately
defined as:

```text
profile_kind = engineered_model_world_partition
authoritative_terrain_bounds = false
world_x = [-131072, 131072] inclusive
world_y = [-131072, 131072] inclusive
grid = 32 rows x 32 columns, row-major
cell size = 8192 x 8192 Unreal units = 81.92 x 81.92 m
lattice origin = (0, 0)
lattice spacing = 512 Unreal units
```

It must not be described as a coastline, terrain, playable-area, or replicated
metadata boundary. Accepted coordinates are used only to measure coverage;
they never fit, define, clamp, or expand the envelope.

## Retained metadata diagnostics

The schema-2.0 publication audit retains every raw metadata observation across
51 build-41.10 and 12 build-41.20 replays. The two reported tuples conflict at
floating-point precision. A diagnostic zero-origin, 512-unit, round-to-even
canonicalization converges to indices `[-238,220,-237,235]` and attempted
bounds `(-121856,112640,-121344,120320)`, but it still fails:

- residuals are approximately `14.231445..252.435388` world units against the
  `1e-5` absolute tolerance; and
- reconstructed counts are `458 x 472`, while reported counts are
  `457 x 471`.

`lattice_canonicalization.passed` therefore remains false. This is diagnostic
evidence about the reported tuples, not a failed derivation of the engineered
model envelope.

## Coordinate coverage and masking

The reviewed 41-session coordinate audit scans every absolute XY stream in
accepted dataset v2:

| Stream | Finite pairs | Outside |
| --- | ---: | ---: |
| Player samples | 600,512 | 60 |
| Player events | 64,980 | 0 |
| Eligible-player team centroids | 318,383 | 46 |
| Zone-phase source centers | 489 | 0 |
| Zone-phase target centers | 489 | 0 |
| Sampled current zone centers | 10,344 | 0 |
| Sampled target zone centers | 10,344 | 0 |

Null pairs remain unavailable. Offsets, radii, Z values, and
centroid-neighbor deltas are not envelope coordinates and are not scanned as
such. The 106 finite misses remain unchanged and unclamped; future-position
targets using those coordinates are masked from spatial-class supervision.
Raw pre-sampling movement and rejected-session coordinates remain separate
replay diagnostics.

## Two-pass approval

`world-grid-audit` requires both the replay corpus and `--dataset-root`.
Without approval it writes an audit with `publication.status=approval_required`
and a deterministic `coordinate_audit_hash`, but no profile. After review,
rerun the complete audit with:

```text
--approved-coordinate-audit-sha256 <exact reviewed hash>
```

Publication succeeds only if the recomputed hash matches exactly and every
replay/build/world, metadata-containment, lattice/scale, geometry, and
dataset-session binding check passes. Reordering replay records, ingestion
items, sessions, or Parquet rows does not change the hash. Any substantive
corpus change does and requires explicit republication.

The current reviewed coordinate hash is
`fe617bbba2bc92583dac58068b36d3c0ed4a3f36147c7ab098ebcec1f103c7aa`.
The published profile hash is
`398e24034b574fd2306db3f975e0d218480d2485cb8c2b19e6ca80340a931bee`.

## Normalization and cells

```text
u = (x + 131072) / 262144
v = (y + 131072) / 262144
column = min(31, floor((x + 131072) / 8192))
row = min(31, floor((y + 131072) / 8192))
class_index = row * 32 + column
```

Only values inside the inclusive envelope are spatially available. Exact
maximum bounds belong to the final row or column. Spatial classes remain
`0..1023`; eliminated-before-horizon remains class `1024`. Encoder scaling,
target formulas, and class count are unchanged.

## Image calibration boundary

An image-specific transform may be designed later for visualization. It would
need an image hash, dimensions, orientation, and independently verified
control points. It remains separate from `WorldGridProfile` and is not imported
or configured by target construction or training.
