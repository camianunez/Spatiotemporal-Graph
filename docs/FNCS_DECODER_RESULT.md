# FNCS decoder validation result

This page summarizes the completed frozen-encoder parallel trajectory decoder run. It reports development validation on the locked 115-session decoder-validation split. The 115-session decoder-test split was not evaluated.

## Run and checkpoint

- Corpus: 1,150 accepted FNCS pilot sessions, split 920 train / 115 validation / 115 test.
- Model: 256-wide spatiotemporal encoder, frozen for decoder training, followed by a five-mode parallel 12-horizon trajectory decoder.
- Selection rule: lowest validation route-mixture negative log-likelihood (NLL), fixed before training.
- Selected point: epoch 20 of 20, optimizer step 9,200 of 9,200.
- Decoder artifact: `runs/parallel-trajectory-decoder-v1-fncs-frozen-encoder-run-1-resume-1/best.pt`.
- Decoder SHA-256: `7e210adfb5453511dd4cda35c1db968245921f8e08848fa51dfb035eaa07adc6`.
- Required frozen encoder: `runs/encoder-fncs-pilot-20260817-fresh-run-3/encoder-only-best.pt`, SHA-256 `71c48713546c60aaf7573eacd199f8972c2fd012c9375c2296d89cff2d78cfb5`.

## Validation measurements

The point metrics below use the mean route of the model's highest-probability mixture component. They cover 3,680 valid queries and 42,265 valid future positions.

| Method | ADE | FDE at 60 s | Last-valid-horizon FDE |
| --- | ---: | ---: | ---: |
| Static position | 80.124 m | 143.711 m | 136.233 m |
| Constant velocity | 114.990 m | 226.330 m | 215.652 m |
| Decoder | **75.128 m** | **127.725 m** | **121.330 m** |

Route-mixture NLL was **7.004335993**. The paired session-bootstrap comparison supported lower decoder error than both baselines for all three point metrics. The decoder's session-bootstrap 95% intervals were 72.305–78.129 m for ADE, 123.052–132.729 m for 60-second FDE, and 116.856–126.066 m for last-valid FDE.

Validation route NLL at epochs 18, 19, and 20 was **7.5073, 7.3553, and 7.0043**. Epoch 20 was the lowest of all 20 validation checkpoints and improved on epoch 19 by 4.77%. This supports trying a separately evaluated longer run; it does not establish that additional epochs would improve held-out performance.

## Evidence and limits

The completed local evaluation is under `runs/parallel-trajectory-decoder-v1-fncs-frozen-encoder-run-1-post-training-evaluation-20260902/`. It reproduced the selected checkpoint's NLL exactly and recorded no decoder-test data access. The decoder and encoder checkpoint hashes above identify the evaluated weights.

All 1,150 accepted sessions were accepted with data warnings and lack complete tournament provenance. The earlier 107-session decoder run used a different encoder policy, data vintage, and training schedule, so these runs do not isolate the effect of dataset size. These measurements support a validation-stage movement forecasting result, not a production or optimal-rotation claim.
