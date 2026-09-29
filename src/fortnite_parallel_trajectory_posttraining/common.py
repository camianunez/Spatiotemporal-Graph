from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping
import uuid

from fncs_encoder_training.common import file_sha256, utc_now


class PostTrainingError(RuntimeError):
    """Raised when a locked post-training invariant is violated."""


PACKAGE_ROOT = Path(__file__).resolve().parent
ML_ROOT = PACKAGE_ROOT.parents[1]
REPOSITORY_ROOT = ML_ROOT.parent

LINEAGE_NAME = "parallel-trajectory-decoder-v1-fncs-frozen-encoder-run-1"
PARENT_RUN = REPOSITORY_ROOT / "runs" / LINEAGE_NAME
CHILD_RUN = REPOSITORY_ROOT / "runs" / f"{LINEAGE_NAME}-resume-1"
RECOVERY_AUDIT = (
    REPOSITORY_ROOT
    / "audit"
    / f"{LINEAGE_NAME}-recovery-20260831T003527Z"
)
STEP1840_CHECKPOINT = RECOVERY_AUDIT / "checkpoint-copy" / "last-step-00001840.pt"
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "parallel-trajectory-training-fncs-frozen-encoder-v1.json"
)
FINALIZATION_DIRECTORY = (
    REPOSITORY_ROOT / "audit" / f"{LINEAGE_NAME}-finalization-20260902"
)
ANALYSIS_DIRECTORY = (
    REPOSITORY_ROOT / "runs" / f"{LINEAGE_NAME}-post-training-analysis-20260902"
)
SCALE_RUN_ROOT = (
    REPOSITORY_ROOT / "runs" / f"{LINEAGE_NAME}-fixed-compute-scale-20260902"
)

EXPECTED_ENCODER_STATE_SHA256 = (
    "a70edc067a0b4e284b93b4a60ae472d0f2d9b4bc226143feecd841a505a9489f"
)
EXPECTED_INITIAL_DOWNSTREAM_SHA256 = (
    "e5b11a2af8fe1793f371ab309d1ed7e466f9160bde7e0101de8825f72fbcb974"
)
EXPECTED_FINAL_STEP = 9200
EXPECTED_FIXED_COMPUTE_STEP = 1840
EXPECTED_FINAL_ROUTE_NLL = 7.004335993010065
SEED = 20260727
HORIZONS_SECONDS = tuple(range(5, 61, 5))
SCALE_SESSION_COUNTS = (41, 115, 230, 460, 920)


def current_utc() -> str:
    return utc_now()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostTrainingError(message)


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        encoded = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    else:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return (encoded + "\n").encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_seal(value: Mapping[str, Any], field: str = "payload_sha256") -> dict[str, Any]:
    require(field not in value, f"unsealed payload unexpectedly contains {field}")
    result = dict(value)
    result[field] = payload_sha256(result)
    return result


def verify_self_seal(value: Mapping[str, Any], field: str = "payload_sha256") -> None:
    actual = value.get(field)
    require(isinstance(actual, str) and len(actual) == 64, f"invalid {field}")
    unsealed = dict(value)
    del unsealed[field]
    require(payload_sha256(unsealed) == actual, f"{field} mismatch")


def read_json(path: str | Path, label: str | None = None) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostTrainingError(f"cannot read {label or source}: {exc}") from exc
    require(isinstance(value, dict), f"{label or source} must be a JSON object")
    return value


def read_jsonl(path: str | Path, label: str | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PostTrainingError(f"cannot read {label or source}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PostTrainingError(
                f"{label or source} line {index} is invalid JSON: {exc}"
            ) from exc
        require(isinstance(value, dict), f"{label or source} line {index} is not an object")
        rows.append(value)
    return rows


def atomic_write_new(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"refusing to overwrite sealed artifact: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_new_json(path: str | Path, value: Any) -> None:
    atomic_write_new(path, canonical_json_bytes(value, pretty=True))


def atomic_replace(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def replace_json(path: str | Path, value: Any) -> None:
    atomic_replace(path, canonical_json_bytes(value, pretty=True))


def file_record(path: str | Path, *, base: Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    require(source.is_file(), f"required artifact is missing: {source}")
    label = (
        source.relative_to(base.resolve()).as_posix()
        if base is not None and source.is_relative_to(base.resolve())
        else str(source)
    )
    return {
        "path": label,
        "size_bytes": source.stat().st_size,
        "sha256": file_sha256(source),
        "read_only": is_read_only(source),
    }


def directory_inventory(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path).resolve()
    require(root.is_dir(), f"inventory directory is missing: {root}")
    return [
        file_record(item, base=root)
        for item in sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
    ]


def inventory_sha256(rows: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    return payload_sha256(normalized)


def is_read_only(path: str | Path) -> bool:
    source = Path(path)
    attributes = getattr(source.stat(), "st_file_attributes", 0)
    windows_flag = getattr(stat, "FILE_ATTRIBUTE_READONLY", 1)
    if attributes:
        return bool(attributes & windows_flag)
    return not bool(source.stat().st_mode & stat.S_IWUSR)


def make_file_read_only(path: str | Path) -> None:
    source = Path(path)
    require(source.is_file(), f"cannot seal missing file: {source}")
    source.chmod(stat.S_IREAD)
    require(is_read_only(source), f"failed to make file read-only: {source}")


def make_tree_files_read_only(path: str | Path) -> int:
    root = Path(path)
    require(root.is_dir(), f"cannot seal missing directory: {root}")
    files = [candidate for candidate in root.rglob("*") if candidate.is_file()]
    for candidate in files:
        make_file_read_only(candidate)
    return len(files)


__all__ = [
    "ANALYSIS_DIRECTORY",
    "CHILD_RUN",
    "CONFIG_PATH",
    "EXPECTED_ENCODER_STATE_SHA256",
    "EXPECTED_FINAL_ROUTE_NLL",
    "EXPECTED_FINAL_STEP",
    "EXPECTED_FIXED_COMPUTE_STEP",
    "EXPECTED_INITIAL_DOWNSTREAM_SHA256",
    "FINALIZATION_DIRECTORY",
    "HORIZONS_SECONDS",
    "LINEAGE_NAME",
    "PACKAGE_ROOT",
    "PARENT_RUN",
    "PostTrainingError",
    "RECOVERY_AUDIT",
    "REPOSITORY_ROOT",
    "SCALE_RUN_ROOT",
    "SCALE_SESSION_COUNTS",
    "SEED",
    "STEP1840_CHECKPOINT",
    "atomic_replace",
    "atomic_write_new",
    "canonical_json_bytes",
    "current_utc",
    "directory_inventory",
    "file_record",
    "file_sha256",
    "inventory_sha256",
    "is_read_only",
    "make_file_read_only",
    "make_tree_files_read_only",
    "payload_sha256",
    "read_json",
    "read_jsonl",
    "replace_json",
    "require",
    "self_seal",
    "verify_self_seal",
    "write_new_json",
]
