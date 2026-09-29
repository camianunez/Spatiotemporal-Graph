# Normalized Parquet dataset v2

Dataset version `2.0.0` is a breaking replacement for the nested JSON and
duo-wide Parquet v1 outputs. An included replay creates one directory named from
its session ID under either `attested/` or `unattested/` and exactly seven
Parquet tables. Coordinates are raw Unreal world coordinates and match time is
seconds since battle-bus start.

`ingestion-report.json` records each replay hash and exact build profile.
Fresh reports also expose the decoded `FortPoiManager` grid metadata and replay
level identity. Rotation training joins the selected session and replay hash
to `audit/world-grid-41.10-41.20.json`; the seven Parquet schemas remain raw and
unchanged.

## Tables

| File | Columns |
| --- | --- |
| `match_samples.parquet` | `session_id`, `match_time_seconds`, `players_remaining`, `teams_remaining` |
| `player_samples.parquet` | `session_id`, `player_id`, `team_id`, `match_time_seconds`, `alive`, `life_index`, `x`, `y`, `z`, `target_zone_offset_x`, `target_zone_offset_y`, `target_zone_inside`, `target_zone_edge_distance_normalized`, `current_boundary_offset_x`, `current_boundary_offset_y`, `current_boundary_inside`, `current_boundary_edge_distance_normalized` |
| `team_samples.parquet` | `session_id`, `team_id`, `match_time_seconds`, `final_placement`, `x`, `y`, `z` |
| `player_events.parquet` | `session_id`, `player_id`, `team_id`, `match_time_seconds`, `event_type`, `life_index`, `x`, `y`, `z`, `zone_phase`, `zone_basis`, `location_quality` |
| `zone_phases.parquet` | `session_id`, `zone_phase`, `activation_time_seconds`, `shrink_start_time_seconds`, `closure_time_seconds`, `source_x`, `source_y`, `source_z`, `source_radius`, `target_x`, `target_y`, `target_z`, `target_radius` |
| `zone_samples.parquet` | `session_id`, `match_time_seconds`, `zone_phase`, `time_until_closure_seconds`, `current_boundary_x`, `current_boundary_y`, `current_boundary_z`, `current_boundary_radius`, `target_x`, `target_y`, `target_z`, `target_radius` |
| `centroid_neighbors.parquet` | `session_id`, `match_time_seconds`, `source_team_id`, `neighbor_team_id`, `neighbor_rank`, `distance_xy`, `delta_x`, `delta_y` |

The grid is fixed at `t=0,5,10,...` through the greatest complete five-second tick no later than match end. `player_events` and `centroid_neighbors` are still written with their complete schemas when they contain no rows.

## Lifecycle and positions

`player_id` is an HMAC-based identifier derived only from the stable account ID and the configured salt. A missing stable account ID rejects the replay. `team_id` is the decoded numeric team ID.

DBNO does not change `alive`. Death changes it immediately and emits one `death` event with the reliable death coordinate. A completed reboot increments `life_index` and emits a coordinate-free `reboot` event. Coordinates and zone-relative values remain null after a reboot until the first movement in that life.

`players_remaining` and `teams_remaining` are derived exclusively from these
lifecycle transitions. Replicated `PlayersLeft` and `TeamsLeft` observations are
optional diagnostics; disagreement produces warnings and never replaces
lifecycle state.

Lifecycle `event_type` values are `dbno`, `death`, and `reboot`. Zone event types are `zone_initially_inside`, `zone_entry`, and `zone_reentry`.

Team centroids average X, Y, and Z across alive teammates. Dead players are excluded, so a survivor supplies a duo's centroid. A team with no alive players, or with any alive player missing telemetry, has a null centroid.

## Zones and events

Phase 1 activates at safe-zone start; each later phase activates at the previous phase's closure. Its target is fixed. The current boundary holds its source circle until shrink start, interpolates to the target during shrink, and remains at the target after closure. The final phase remains active with a zero countdown after its closure.

For either zone basis, offsets are `player - circle_center` in XY and:

```text
edge_distance_normalized = (sqrt(offset_x^2 + offset_y^2) - radius) / radius
```

Negative values are inside, zero is the boundary, and positive values are outside. Inclusion ignores Z. The normalized value is null at radius zero.

Zone events use `zone_basis` values `target` and `current_boundary`. A player inside at activation emits `zone_initially_inside`; outside-to-inside crossings emit `zone_entry` and subsequent crossings emit `zone_reentry`. State resets for every phase, life, and basis. Safe actor-continuous segments of at most two seconds are solved against the fixed or moving circle. Discontinuous transitions use the first observed inside coordinate with `location_quality=observed_fallback`. Zone exits are not emitted.

## Centroid neighbors

Neighbor rows are directed. For every non-null source centroid, candidates with non-null centroids are ordered by XY Euclidean distance and then numeric neighbor team ID. Rank is one-based, `delta_x` and `delta_y` are `neighbor - source`, self is excluded, and at most `--nearest-centroids <k>` rows are retained.

## Publication

All seven files are serialized in a hidden staging directory. They are reopened
to verify exact schemas, row formulas, primary keys, complete finite coordinate
triples, lifecycle-derived counts, and zone invariants before a directory rename
makes the session visible. A failed serialization, validation, or commit leaves
no partial visible session. A later run with the same identity secret and
complete valid provenance publishes under `attested/` and removes the verified
`unattested/` copy.
