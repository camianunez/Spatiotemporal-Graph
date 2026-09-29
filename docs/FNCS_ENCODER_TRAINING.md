# FNCS fresh encoder training

This run trains a new `SpatiotemporalEncoder` from random initialization on the
1,150-session FNCS pilot allowlist. It uses the verified `RotationModel` heads
only as training-time supervision. No trajectory decoder, planner, congestion
loss, beam search, or prior model weights participate in the run.

## Immutable authorities

- Dataset: `FNDATA/decoded-v2-fncs-pilot-20260817/ingestion-report.json`.
  The preflight requires exactly 1,150 accepted canonical sessions, 1,357
  rejected replay payloads, Dataset schema `2.0.0`, and seven byte/hash/row/schema
  verified Parquet tables per accepted session.
- Encoder and auxiliary-head authority: `runs/rotation-placeholder`, whose
  completed `best.pt` is inspected for configuration and tensor contracts only.
  Its weights are never loaded into the new model.
- Protected encoder source digest:
  `a0eb81ae908cfe5dd7c48da0834e129312687f59f80c8c074a983d612944f56e`.
- World grid: the immutable `model-world-grid-v1` profile with profile hash
  `398e24034b574fd2306db3f975e0d218480d2485cb8c2b19e6ca80340a931bee`.
  A new compatibility audit scans every accepted session and records every
  out-of-bounds coordinate; out-of-bounds targets are masked, never clamped.

## Fixed training contract

The fixed seed is `20260727`. The canonical-session split is 920 training, 115
validation, and 115 sealed test sessions. Because all accepted sessions are
unattested and have incomplete event provenance, region/division/event/week
stratification is unavailable; the split is an ascending deterministic SHA-256
ranking of the seed and canonical `game_session_id`.

The encoder remains `[B,T,N,256]`: width 256, eight attention heads, two spatial
layers, four causal temporal layers, feed-forward width 1,024, full spatial
neighbors, and dropout 0.1. The auxiliary heads predict future position at
15/30/60 seconds, zone entry, survival, and placement. The objective is:

```text
L_total = L_position + lambda_entry * L_entry + lambda_survival * L_survival + lambda_placement * L_placement + L_regularization
```

All three lambdas are 1.0 and `L_regularization` is zero. AdamW supplies only
optimizer-level weight decay (`0.01`), with bias, normalization/one-dimensional,
and relative-lag-bias parameters excluded. Other fixed values are batch size 4,
accumulation 1, 64-tick context, 8-tick stride, 20 epochs, peak learning rate
`3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, gradient clip 1.0, CUDA BF16, and
validation every epoch. `future_position` validation loss is fixed as the
checkpoint-selection metric before training.

There are 230 optimizer updates per epoch and 4,600 over the 20-epoch contract.
The recomputed piecewise schedule has 230 linear-warmup updates, 230 constant-LR
updates, and 4,140 linear-decay updates, ending at 10% of the peak LR.

## Reproduction and artifacts

The canonical production configuration is
`configs/encoder-fncs-pilot-20260817-fresh-run-2.json`. With the CUDA environment
active and `ml/src` on `PYTHONPATH`, the two entry points are:

```powershell
python -m fncs_encoder_training.preflight --config configs\encoder-fncs-pilot-20260817-fresh-run-2.json
python -m fncs_encoder_training.runtime --config configs\encoder-fncs-pilot-20260817-fresh-run-2.json
```

Preflight evidence is written once to
`audit/encoder-fncs-pilot-20260817-fresh-run-2-preflight`. The immutable training
run is `runs/encoder-fncs-pilot-20260817-fresh-run-2`. Its resolved configuration
binds the ingestion report, revalidated inventory, split, protected source,
training-orchestrator source, world-grid profile/publication audit/new dataset
audit, initial parameter hash, optimizer groups, scheduler boundaries,
environment, and zero-access sealed-test policy. `last.pt` contains the partial
epoch metrics plus model, optimizer, scheduler, sampler, RNG, epoch, and step
state needed for exact resumption; `best.pt` is selected only on the locked
validation split. Encoder-only best/last exports contain no auxiliary-head
weights and are the intended inputs to a later, separately authorized decoder
task.
