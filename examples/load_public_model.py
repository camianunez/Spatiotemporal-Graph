"""Load verified public encoder and decoder tensor files into the model.

Download encoder_state.pt and decoder_state.pt from Hugging Face, then run:
    python examples/load_public_model.py encoder_state.pt decoder_state.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml/src"))

from fncs_encoder_training.common import tensor_state_sha256  # noqa: E402
from fortnite_encoder.model import SpatiotemporalEncoder  # noqa: E402
from fortnite_encoder.world_grid import WorldGridProfile  # noqa: E402
from fortnite_parallel_trajectory.model import ParallelTrajectoryModel  # noqa: E402
from fortnite_parallel_trajectory_fncs_training.binding import (  # noqa: E402
    downstream_state,
    load_downstream_state,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path, name: str, metadata: dict) -> dict[str, torch.Tensor]:
    expected = metadata[name]
    if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"{name} does not match the published size and SHA-256")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != expected["tensor_count"]:
        raise ValueError(f"{name} has an unexpected tensor dictionary")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{name} contains a non-tensor value")
    if tensor_state_sha256(state) != expected["tensor_state_sha256"]:
        raise ValueError(f"{name} does not match the source checkpoint tensor digest")
    return state


def load_public_model(encoder_path: Path, decoder_path: Path) -> ParallelTrajectoryModel:
    """Return the matching CPU model in evaluation mode after strict checks."""
    metadata = json.loads((ROOT / "metadata/model_weights.json").read_text(encoding="utf-8"))["files"]
    encoder_state = load_state(encoder_path, "encoder_state.pt", metadata)
    decoder_state = load_state(decoder_path, "decoder_state.pt", metadata)
    training = json.loads(
        (ROOT / "configs/parallel-trajectory-training-fncs-frozen-encoder-v1.json").read_text(
            encoding="utf-8"
        )
    )
    profile = WorldGridProfile.load(
        ROOT / "configs/world-grid/model-world-grid-v1.json",
        expected_hash=training["expected_world_grid_profile_hash"],
    )
    encoder = SpatiotemporalEncoder()
    encoder.load_state_dict(encoder_state, strict=True)
    model = ParallelTrajectoryModel(profile, encoder)
    load_downstream_state(model, decoder_state)
    if tensor_state_sha256(model.encoder.state_dict()) != metadata["encoder_state.pt"]["tensor_state_sha256"]:
        raise ValueError("loaded encoder state differs")
    if tensor_state_sha256(downstream_state(model)) != metadata["decoder_state.pt"]["tensor_state_sha256"]:
        raise ValueError("loaded decoder state differs")
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder_state", type=Path)
    parser.add_argument("decoder_state", type=Path)
    args = parser.parse_args()
    model = load_public_model(args.encoder_state, args.decoder_state)
    print(
        f"Loaded {model.world_grid_profile.profile_id}: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters, "
        "CPU evaluation mode"
    )


if __name__ == "__main__":
    main()
