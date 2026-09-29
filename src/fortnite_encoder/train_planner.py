from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .planner_training import run_planner_training, verify_planner_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the deterministic causal Fortnite route planner.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Strict JSON planner-training configuration.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from 'last' or an exact planner checkpoint path.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Validate hashes, split coverage/leakage, schedule, and CUDA "
            "without opening Parquet or creating the run directory."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.verify_only and arguments.resume is not None:
        _parser().error("--verify-only cannot be combined with --resume")
    summary = (
        verify_planner_training(arguments.config)
        if arguments.verify_only
        else run_planner_training(
            arguments.config,
            resume=arguments.resume,
        )
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
