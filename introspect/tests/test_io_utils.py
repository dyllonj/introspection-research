from __future__ import annotations

import json
import logging
import random
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from introspect.src.io_utils import (
    JsonlWriter,
    SchemaValidationError,
    gather_runtime_metadata,
    seed_everything,
    setup_logging,
    truncate_text,
)


def test_seed_everything_controls_rngs() -> None:
    seed_everything(123)
    python_random = random.randint(0, 10**6)
    numpy_random = int(np.random.randint(0, 10**6))

    seed_everything(123)
    assert random.randint(0, 10**6) == python_random
    assert int(np.random.randint(0, 10**6)) == numpy_random


def test_jsonl_writer_writes_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    schema = {"id": int, "text": {"type": str, "required": False}}

    with JsonlWriter(path, schema=schema, append=False, metadata={"meta": "info"}) as writer:
        writer.write({"id": 1, "text": "hello"})

    data = [json.loads(line) for line in path.read_text().splitlines()]
    assert data == [{"meta": "info", "id": 1, "text": "hello"}]
    assert writer.records_written == 1


def test_jsonl_writer_schema_validation(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"

    with JsonlWriter(path, schema={"id": int}) as writer:
        with pytest.raises(SchemaValidationError):
            writer.write({"missing": 1})


def test_gather_runtime_metadata_includes_expected_keys() -> None:
    metadata = gather_runtime_metadata(extra={"experiment": "demo"})
    assert "timestamp" in metadata
    assert metadata["experiment"] == "demo"
    assert "git_is_dirty" in metadata


def test_truncate_text_round_trip() -> None:
    text = "a" * 100 + "b" * 100
    truncated = truncate_text(text, max_length=40)
    assert truncated.startswith("a" * 19)
    assert truncated.endswith("b" * 19)
    assert "…" in truncated

    with pytest.raises(ValueError):
        truncate_text("hello", max_length=0)


def test_setup_logging_configures_root_logger() -> None:
    setup_logging(level=logging.DEBUG)
    root_logger = logging.getLogger()

    assert root_logger.level == logging.DEBUG
    assert any(isinstance(handler, logging.Handler) for handler in root_logger.handlers)
