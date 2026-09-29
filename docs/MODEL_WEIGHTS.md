# Model weights

The [Hugging Face model repository](https://huggingface.co/BiLSTM/SpatioTemporalDecoder/tree/main) holds the frozen encoder and trained decoder for the completed FNCS pilot run. This GitHub source snapshot contains no checkpoint binaries.

| File | Contents | SHA-256 |
| --- | --- | --- |
| `encoder_state.pt` | 124 frozen encoder tensors, safe for restricted PyTorch loading | `b4c66614d08dbf1c4e4ad6b2cafb78a939d87e17a270dc9801bcac78329d80ee` |
| `decoder_state.pt` | 135 decoder and congestion tensors, safe for restricted PyTorch loading | `7c89d654d6c356ffcb5696452b8ede3afd20f9c7d327002fe36e929930eb4126` |
| `best.pt` | Original epoch-20 decoder training checkpoint, including optimizer state | `7e210adfb5453511dd4cda35c1db968245921f8e08848fa51dfb035eaa07adc6` |

The tensor-only files are derived from the original local checkpoints. Their tensor-state digests match the hashes embedded in those checkpoints. See [weight metadata](../metadata/model_weights.json) and [checkpoint selection evidence](../evidence/checkpoints/decoder_best.json).

Download and inspect the tensor-only state dictionaries:

```python
import torch
from huggingface_hub import hf_hub_download

repo = "BiLSTM/SpatioTemporalDecoder"
encoder_state = torch.load(
    hf_hub_download(repo, "encoder_state.pt"),
    map_location="cpu",
    weights_only=True,
)
decoder_state = torch.load(
    hf_hub_download(repo, "decoder_state.pt"),
    map_location="cpu",
    weights_only=True,
)
assert len(encoder_state) == 124
assert len(decoder_state) == 135
```

The encoder tensors load into `SpatiotemporalEncoder` with `load_state_dict(..., strict=True)`. The decoder tensors load through `fortnite_parallel_trajectory_fncs_training.binding.load_downstream_state` after constructing the matching `ParallelTrajectoryModel`. Use the [archived configuration](../configs/parallel-trajectory-training-fncs-frozen-encoder-v1.json) and [world-grid profile](../configs/world-grid/model-world-grid-v1.json). The full private replay corpus is required to reproduce training or validation.

`best.pt` is a Python pickle based training checkpoint and is intended only for trusted provenance checks. It is not a Transformers `from_pretrained` package. The tensor-only exports are the recommended files for inspection and inference.


## Strict model load

After downloading both tensor-only files, run this from the repository root:

```powershell
python examples/load_public_model.py .\encoder_state.pt .\decoder_state.pt
```

The [loader example](../examples/load_public_model.py) checks both file hashes and embedded tensor-state digests, loads the encoder and decoder with strict tensor names, validates the world-grid profile, and returns a CPU model in evaluation mode. This has been checked against the published exports: 124 encoder tensors, 135 decoder tensors, and 11,424,238 total model parameters. No private replay files are needed to load the model; forecasting still requires inputs matching the documented dataset and tensorization contracts.
