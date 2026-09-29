from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train parallel_trajectory_decoder_v1 against the immutable FNCS encoder."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=("last",), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_training(arguments.config, resume=arguments.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
