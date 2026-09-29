from __future__ import annotations

from typing import Any

from fortnite_parallel_trajectory_fncs_recovery.recovery import _build_setups
from fortnite_parallel_trajectory_fncs_training.binding import VerifiedSetup

from .common import CONFIG_PATH, PARENT_RUN, read_json


def load_lineage_setups() -> tuple[VerifiedSetup, VerifiedSetup, dict[str, Any], dict[str, Any]]:
    """Return (current-source, parent-checkpoint, old-manifest, current-manifest)."""

    parent_resolved = read_json(PARENT_RUN / "resolved_config.json", "parent resolved config")
    return _build_setups(CONFIG_PATH, parent_resolved)


__all__ = ["load_lineage_setups"]
