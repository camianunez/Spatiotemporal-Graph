from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def rendered_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, rendered_json(value))


def atomic_write_text(path: str | Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = (canonical_json(value) + "\n").encode("utf-8")
    # Metrics are append-only.  Flush each complete JSON line so a durable
    # checkpoint can embed the exact prefix without quadratically rewriting
    # the entire history on every optimizer update.
    with destination.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_torch_save(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: Tensor) -> str:
    return sha256_bytes(tensor_bytes(value))


def tensor_manifest(state: Mapping[str, Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "raw_tensor_sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(state.items())
    ]


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    return sha256_bytes(canonical_json(tensor_manifest(state)).encode("utf-8"))


def source_tree_manifest(root: str | Path) -> dict[str, Any]:
    source_root = Path(root).resolve()
    files = sorted(source_root.rglob("*.py"), key=lambda item: item.as_posix())
    aggregate = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        raw_digest = hashlib.sha256(path.read_bytes()).digest()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(raw_digest)
        entries.append({"path": relative, "sha256": raw_digest.hex()})
    return {
        "schema_version": "source-tree-manifest:1.0",
        "hash_root": str(source_root),
        "hash_contract": "UTF8(relative POSIX path) || NUL || raw SHA-256 bytes",
        "extension_filter": "*.py",
        "file_count": len(entries),
        "entries": entries,
        "source_tree_sha256": aggregate.hexdigest(),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def _tree_digest(paths: Iterable[Path], root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return digest.hexdigest(), count


def git_environment(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "-z"],
            check=True,
            capture_output=True,
        ).stdout
        return {
            "git_available": True,
            "git_revision": revision,
            "dirty": bool(status),
            "dirty_tree_digest": sha256_bytes(status),
            "dirty_digest_contract": "SHA-256 of git status --porcelain=v1 -z bytes",
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        scopes = [
            root / "ml" / "src",
            root / "ml" / "tests",
            root / "configs",
            root / "docs",
            root / "src",
            root / "tests",
        ]
        candidates = (
            path
            for scope in scopes
            if scope.is_dir()
            for path in scope.rglob("*")
            if "__pycache__" not in path.parts and ".pytest_cache" not in path.parts
        )
        digest, count = _tree_digest(candidates, root)
        return {
            "git_available": False,
            "git_revision": None,
            "git_error": str(exc),
            "dirty": None,
            "dirty_tree_digest": digest,
            "dirty_tree_file_count": count,
            "dirty_digest_contract": (
                "fallback SHA-256 path/NUL/raw-file-digest stream over source, tests, "
                "configs, and docs because the workspace .git directory is not a "
                "recognized repository"
            ),
        }
