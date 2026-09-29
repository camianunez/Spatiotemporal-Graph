from __future__ import annotations

from copy import deepcopy

import pytest

from fortnite_parallel_trajectory_fncs_training.config import (
    TrainingCompatibilityError,
)
from fortnite_parallel_trajectory_fncs_training.training import (
    validate_resolved_resume_contract,
)


OLD_SOURCE = "1" * 64
NEW_SOURCE = "2" * 64


def _contract(source_digest: str = OLD_SOURCE) -> dict[str, object]:
    return {
        "schema_version": "fncs-frozen-decoder-resolved-training:1.0",
        "created_utc": "2026-08-25T21:53:31Z",
        "command_line": ["python", "train.py", "--config", "locked.json"],
        "training_configuration": {
            "run_directory": "C:/parent",
            "seed": 20260727,
            "max_epochs": 20,
            "max_optimizer_steps": 9200,
            "batch_size": 2,
            "queries_per_session": 16,
            "precision": "bf16",
        },
        "checkpoint_bindings": {
            "architecture_id": "parallel_trajectory_decoder_v1",
            "training_source_digest": source_digest,
            "encoder_state_sha256": "3" * 64,
            "split_manifest_sha256": "4" * 64,
        },
        "training_source_manifest": {
            "hash_root": "C:/workspace/ml/src/package",
            "entries": [{"path": "training.py", "sha256": source_digest}],
            "source_tree_sha256": source_digest,
        },
        "optimizer": {
            "optimizer_type": "AdamW",
            "downstream_learning_rate": 0.0003,
            "betas": [0.9, 0.999],
        },
        "scheduler": {"scheduler_type": "locked", "total_steps": 9200},
        "dataset_bindings": {"inventory_raw_sha256": "5" * 64},
        "environment": {
            "created_utc": "2026-08-25T21:53:30Z",
            "process_id": 100,
            "parent_process_id": 99,
            "complete_command_line": ["python", "train.py"],
            "launcher": {"kind": "powershell"},
            "git": {"dirty_tree_digest": "6" * 64},
            "python": "3.13",
            "torch": "2.12.1+cu130",
            "cuda_runtime": "13.0",
            "deterministic_algorithms": True,
        },
    }


def test_resume_contract_accepts_only_volatile_launch_changes_and_approved_source() -> None:
    persisted = _contract()
    current = _contract(NEW_SOURCE)
    current["created_utc"] = "2026-08-31T00:40:00Z"
    current["command_line"] = [
        "python",
        "recover.py",
        "--resume",
        "C:/audit/last-step-00001840.pt",
    ]
    current["physical_run_directory"] = "C:/child"
    current["resume_control"] = {
        "checkpoint": "C:/audit/last-step-00001840.pt"
    }
    environment = current["environment"]
    assert isinstance(environment, dict)
    environment.update(
        {
            "created_utc": "2026-08-31T00:39:59Z",
            "process_id": 200,
            "parent_process_id": 150,
            "complete_command_line": ["python", "recover.py", "--resume"],
            "launch_timestamp": "2026-08-31T00:39:59Z",
            "launcher": {"kind": "background-powershell", "pid": 150},
            "git": {"dirty_tree_digest": "7" * 64},
        }
    )
    validate_resolved_resume_contract(
        persisted,
        current,
        approved_source_transition=(OLD_SOURCE, NEW_SOURCE),
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("training_configuration", "seed", 20260728),
        ("training_configuration", "max_optimizer_steps", 9199),
        ("training_configuration", "batch_size", 4),
        ("training_configuration", "queries_per_session", 8),
        ("training_configuration", "precision", "fp32"),
        ("checkpoint_bindings", "architecture_id", "different_decoder"),
        ("checkpoint_bindings", "encoder_state_sha256", "8" * 64),
        ("checkpoint_bindings", "split_manifest_sha256", "9" * 64),
        ("optimizer", "downstream_learning_rate", 0.0004),
        ("scheduler", "total_steps", 9201),
        ("dataset_bindings", "inventory_raw_sha256", "a" * 64),
    ],
)
def test_resume_contract_rejects_every_semantic_training_change(
    section: str, field: str, replacement: object
) -> None:
    persisted = _contract()
    current = deepcopy(persisted)
    target = current[section]
    assert isinstance(target, dict)
    target[field] = replacement
    with pytest.raises(TrainingCompatibilityError, match="resolved training contract changed"):
        validate_resolved_resume_contract(persisted, current)


def test_resume_contract_rejects_unapproved_source_change() -> None:
    with pytest.raises(TrainingCompatibilityError, match="resolved training contract changed"):
        validate_resolved_resume_contract(_contract(), _contract(NEW_SOURCE))
