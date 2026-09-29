from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the frozen parallel_trajectory_decoder_v1 integration."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Use 'last' or an exact checkpoint path; omit for a new run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = run_training(arguments.config, resume=arguments.resume)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
