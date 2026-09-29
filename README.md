# Spatiotemporal Trajectory Forecasting

Research code for probabilistic 5-60 second team movement forecasts from replay telemetry. It focuses on data contracts, coordinate calibration, a causal spatial-temporal encoder, a parallel five-route decoder, training, evaluation, and focused tests. Fortnite duo server replays are the only training and validation case study; performance on other games is unmeasured.

## Model and result

The encoder builds 256-dimensional team states with two spatial and four causal temporal attention layers. Auxiliary future-position, zone-entry, survival, and placement heads train the encoder. The final run freezes it and trains a decoder that jointly predicts twelve future horizons with five route-level Gaussian modes. See [encoder design](docs/ENCODER_COMPONENTS_1_4.md) and [decoder design](docs/parallel-trajectory-decoder-v1.md).

The experiment used 1,150 accepted sessions: 920 train, 115 validation, and 115 reserved test. On **validation**, the selected decoder reached 75.128 m ADE versus 80.124 m for static position, and 127.725 m 60-second FDE versus 143.711 m for static position. See the [result summary](docs/FNCS_DECODER_RESULT.md) and [evaluation report](evidence/validation/final_report.md). The test split was not evaluated. All accepted sessions have data warnings and incomplete tournament provenance. These results do not establish globally optimal movement, cross-game generalization, or production readiness.

## Contents

- `ml/src/fortnite_encoder/` and `fncs_encoder_training/`: tensorization, masking, encoder and its final training path.
- `ml/src/fortnite_parallel_trajectory/` and `fortnite_parallel_trajectory_fncs_training/`: decoder, targets, likelihood, and final frozen-encoder training path.
- `ml/src/fortnite_parallel_trajectory_training/`: shared training components.
- `ml/src/fortnite_parallel_trajectory_posttraining/`: validation metrics and reporting.
- `ml/src/fortnite_early_zone/`: two shared decoder primitive modules retained for published weight compatibility.
- `ml/tests/`: synthetic data, calibration, masking, model, training, and metric tests.

The `fortnite_*` module names preserve imports and checkpoint compatibility. The full replay parser, collector, and earlier experimental branches remain in the separate full snapshot.

## Install and verify

Use Python 3.11 or newer:

```powershell
python -m pip install -e ".\ml[test]"
python -m pytest ml/tests -q
```

The tests use synthetic sessions and need no private replay corpus. Published tensor-only weights are on [Hugging Face](https://huggingface.co/BiLSTM/SpatioTemporalDecoder/tree/main). After downloading both files:

```powershell
python examples/load_public_model.py .\encoder_state.pt .\decoder_state.pt
```

See [model weights](docs/MODEL_WEIGHTS.md) for the strict loader and checkpoint differences.

## Data, calibration, and training

Training requires compatible [seven-table Parquet sessions](docs/DATASET_V2.md), plus an inventory, split manifest, ingestion and validation reports, and a world-grid audit. The [world-grid profile](configs/world-grid/model-world-grid-v1.json) is an engineered 32 x 32 coordinate partition, not terrain geometry. See [calibration](docs/MAP_SCALING.md).

The archived two-stage training commands are:

```powershell
python -m fncs_encoder_training.preflight --config configs/encoder-fncs-pilot-20260817-fresh-run-3.json
python -m fncs_encoder_training.runtime --config configs/encoder-fncs-pilot-20260817-fresh-run-3.json
python -m fortnite_parallel_trajectory_fncs_training.preflight --config configs/parallel-trajectory-training-fncs-frozen-encoder-v1.json
python -m fortnite_parallel_trajectory_fncs_training.train --config configs/parallel-trajectory-training-fncs-frozen-encoder-v1.json
```

These configs pin private corpus paths, hashes, checkpoints, and run lineage. They **cannot complete on this public code-only snapshot**. To use another compatible dataset, regenerate the bound manifests and hashes and adapt the training configuration. The public weights can be loaded without that corpus; independent retraining and validation require data.

## License

Original code is MIT licensed; see [LICENSE](LICENSE).
