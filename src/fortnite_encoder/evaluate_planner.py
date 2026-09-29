from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .planner_training import evaluate_planner_test


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a locked planner best.pt once on its pinned test split."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Strict self-conditioned planner-training configuration.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = evaluate_planner_test(arguments.config)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
