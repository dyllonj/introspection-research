"""Utilities for restructuring task JSONL files by model slug.

This command reorganises task outputs produced by the evaluation
scripts so that each ``task_*.jsonl`` lives under a directory derived
from its ``model_id``. The implementation operates purely on the
filesystem—no model loading or GPU context is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from introspect.src.plotting import model_id_to_slug

LOGGER = logging.getLogger(__name__)


def _read_first_record(path: Path) -> dict[str, object] | None:
    """Return the first JSON object found in ``path``.

    Empty lines are ignored. The function returns ``None`` when the file is
    empty or all lines fail to parse.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON line in %s", path)
                continue
    return None


def _normalise_file(path: Path, dest_root: Path, *, copy: bool, overwrite: bool) -> Path | None:
    record = _read_first_record(path)
    if not record:
        LOGGER.warning("Skipping %s because it contains no valid JSON records", path)
        return None

    model_id = record.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        LOGGER.warning("Skipping %s because the first record lacks a 'model_id' field", path)
        return None

    slug = model_id_to_slug(model_id)
    target_dir = dest_root / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / path.name

    if target_path.resolve() == path.resolve():
        LOGGER.info("File %s already located at %s", path, target_dir)
        return target_path

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination file {target_path} exists. Use --overwrite to replace it."
        )

    if copy:
        LOGGER.info("Copying %s → %s", path, target_path)
        shutil.copy2(path, target_path)
    else:
        LOGGER.info("Moving %s → %s", path, target_path)
        shutil.move(path, target_path)

    return target_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group task JSONL files by model slug. The command manipulates files only "
            "and does not require GPU access."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("introspect/results"),
        help="Directory containing task_*.jsonl files",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("introspect/results"),
        help="Destination directory for grouped results",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the destination",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    source = args.source
    dest = args.dest

    if not source.exists() or not source.is_dir():
        parser.error(f"Source directory {source} does not exist or is not a directory")

    dest.mkdir(parents=True, exist_ok=True)

    files = sorted(source.glob("task_*.jsonl"))
    if not files:
        LOGGER.info("No task_*.jsonl files found under %s", source)
        return 0

    processed = 0
    for path in files:
        try:
            if _normalise_file(path, dest, copy=args.copy, overwrite=args.overwrite):
                processed += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Failed to normalise %s: %s", path, exc)
            if args.verbose:
                raise

    LOGGER.info("Normalised %d file(s)", processed)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
