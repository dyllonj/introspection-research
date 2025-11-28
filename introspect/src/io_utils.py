"""Utility helpers for deterministic runs, logging, and JSONL output.

This module centralises helpers that are used across the evaluation scripts:

* ``seed_everything`` makes RNG behaviour reproducible across ``random``,
  ``numpy`` and ``torch`` while enabling deterministic kernels when possible.
* ``JsonlWriter`` wraps JSONL logging with optional schema validation and
  metadata injection.
* ``gather_runtime_metadata`` collects environment information that should be
  stored alongside experimental results.
* ``truncate_text`` keeps prompt/response previews short in logs without
  losing the ability to inspect their boundaries.
* ``setup_logging`` defines a consistent logging layout for CLI entry points.

The functions are intentionally lightweight so they can be used by both tests
and long‑running experiments without pulling in additional dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


__all__ = [
    "JsonlWriter",
    "SchemaValidationError",
    "gather_runtime_metadata",
    "seed_everything",
    "setup_logging",
    "truncate_text",
]


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class SchemaValidationError(ValueError):
    """Raised when a JSONL record violates the declared schema."""


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all major RNGs for deterministic behaviour.

    Parameters
    ----------
    seed:
        Seed value that will be forwarded to Python's ``random`` module,
        ``numpy`` and ``torch``.
    deterministic:
        When ``True`` the function requests deterministic algorithms from
        PyTorch where available. This may reduce performance but is essential
        for reproducible research results.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            logging.getLogger(__name__).info(
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA kernels."
            )

        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:  # pragma: no cover - raised on unsupported builds
            logging.getLogger(__name__).warning(
                "Deterministic algorithms are not fully supported on this build of PyTorch."
            )

        cudnn_backend = getattr(torch.backends, "cudnn", None)
        if cudnn_backend is not None:
            cudnn_backend.deterministic = True
            cudnn_backend.benchmark = False

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        else:
            if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
                cudnn_backend.allow_tf32 = False
            cuda_backend = getattr(torch.backends, "cuda", None)
            if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
                cuda_backend.matmul.allow_tf32 = False


@dataclass(slots=True)
class _SchemaField:
    """Internal representation of schema expectations for JSONL records."""

    name: str
    expected_type: type | tuple[type, ...] | None
    required: bool

    def validate(self, record: Mapping[str, Any]) -> None:
        if self.required and self.name not in record:
            raise SchemaValidationError(f"Missing required key '{self.name}'.")

        if self.name in record and self.expected_type is not None:
            value = record[self.name]
            if not isinstance(value, self.expected_type):
                expected = self.expected_type
                raise SchemaValidationError(
                    f"Field '{self.name}' has type {type(value)!r}, expected {expected!r}."
                )


def _normalise_schema(
    schema: Mapping[str, type | tuple[type, ...]] | Mapping[str, Any] | None,
) -> dict[str, _SchemaField]:
    if schema is None:
        return {}

    normalised: dict[str, _SchemaField] = {}
    for name, expected in schema.items():
        if isinstance(expected, Mapping):
            expected_type = expected.get("type")
            required = expected.get("required", True)
        else:
            expected_type = expected
            required = True

        if expected_type is not None and not isinstance(expected_type, tuple):
            expected_type = (expected_type,)

        normalised[name] = _SchemaField(name=name, expected_type=expected_type, required=required)
    return normalised


class JsonlWriter(AbstractContextManager["JsonlWriter"]):
    """Context manager for appending structured records to a JSONL file.

    Parameters
    ----------
    path:
        Destination JSONL file. The parent directories will be created
        automatically.
    schema:
        Optional mapping describing required keys and their expected types.
        A schema entry can either map to a type (or tuple of types) which will
        be enforced, or to a dictionary with ``type`` and ``required`` flags for
        more control.
    append:
        When ``True`` (default) records are appended to the existing file. When
        ``False`` the file is truncated before writing.
    metadata:
        Additional key-value pairs merged into each record before validation.
        Fields already present in the record take precedence.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema: Mapping[str, Any] | None = None,
        append: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = _normalise_schema(schema)
        self._append = append
        self._metadata = dict(metadata or {})
        self._fh: Any | None = None
        self._records_written = 0

    def __enter__(self) -> "JsonlWriter":
        mode = "a" if self._append else "w"
        self._fh = self.path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def records_written(self) -> int:
        return self._records_written

    def write(self, record: Mapping[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlWriter must be used as a context manager before writing.")

        combined: dict[str, Any] = {**self._metadata, **dict(record)}

        for field in self._schema.values():
            field.validate(combined)

        json_record = json.dumps(combined, ensure_ascii=False)
        self._fh.write(json_record + "\n")
        self._fh.flush()
        self._records_written += 1


def gather_runtime_metadata(
    *,
    extra: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return runtime metadata that should accompany evaluation results."""

    from importlib import metadata as importlib_metadata
    import platform

    metadata: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }

    for package in ("torch", "transformers", "accelerate", "numpy"):
        try:
            metadata[f"{package}_version"] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            metadata[f"{package}_version"] = None

    metadata.update(_collect_git_metadata(repo_root=repo_root))

    if extra:
        metadata.update(extra)

    return metadata


def _collect_git_metadata(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    def _run_git_command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None

        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    commit = _run_git_command("rev-parse", "HEAD")
    branch = _run_git_command("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git_command("status", "--porcelain")

    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_is_dirty": bool(status),
    }


def truncate_text(text: str, *, max_length: int = 200, ellipsis: str = "…") -> str:
    """Return a shortened preview of ``text`` preserving boundaries.

    The function keeps both the beginning and the end of the text so it remains
    possible to reason about prefixes and suffixes when reviewing logs. If the
    input is shorter than ``max_length`` characters it is returned unchanged.
    """

    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    if len(text) <= max_length:
        return text

    keep_each_side = (max_length - len(ellipsis)) // 2
    if keep_each_side <= 0:
        return text[:max_length]

    prefix = text[:keep_each_side]
    suffix = text[-keep_each_side:]
    return f"{prefix}{ellipsis}{suffix}"


def setup_logging(
    *,
    level: int | str = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
) -> None:
    """Configure the root logger with a consistent format.

    The configuration uses ``force=True`` so that repeated invocations (for
    instance, during tests) reset previous handlers instead of stacking them.
    """

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True)

    # Quiet overly verbose third-party libraries by default.
    logging.getLogger("transformers").setLevel(max(logging.WARNING, logging.getLogger().level))
    logging.getLogger("accelerate").setLevel(max(logging.WARNING, logging.getLogger().level))
