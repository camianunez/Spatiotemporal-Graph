from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .training import run_rotation_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the deterministic Fortnite rotation model.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Strict JSON training configuration.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from 'last' or an exact checkpoint path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_rotation_training(
        arguments.config,
        resume=arguments.resume,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
