"""Utilities for inspecting grading quality in introspection results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

import pandas as pd

from introspect.src.plotting import load_task_a_dataframe, load_task_b_dataframe

TASK_LOADERS: Mapping[str, Callable[[Path], pd.DataFrame]] = {
    "task_A": load_task_a_dataframe,
    "task_B": load_task_b_dataframe,
}


def _iter_slug_directories(results_root: Path) -> Iterator[Path]:
    if not results_root.exists():
        return

    yielded = False
    for path in sorted(results_root.iterdir()):
        if path.is_dir():
            yielded = True
            yield path
    if not yielded:
        yield results_root


def _summarise_dataframe(slug: str, task: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "grading.notes" not in df.columns:
        return pd.DataFrame(
            columns=["slug", "task", "mode", "grading.notes", "parsed.label", "count"]
        )

    data = df.copy()
    mode_column = None
    for candidate in ("mode", "vector_kind", "condition"):
        if candidate in data.columns:
            mode_column = candidate
            break

    label_column = "parsed.label" if "parsed.label" in data.columns else None

    data["__mode"] = (
        data[mode_column].astype(str) if mode_column is not None else "<all>"
    )
    data["__notes"] = data["grading.notes"].fillna("<missing>").astype(str)
    data["__label"] = (
        data[label_column].fillna("<missing>").astype(str)
        if label_column is not None
        else "<missing>"
    )

    grouped = (
        data.groupby(["__mode", "__notes", "__label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grouped.insert(0, "task", task)
    grouped.insert(0, "slug", slug)
    grouped = grouped.rename(
        columns={"__mode": "mode", "__notes": "grading.notes", "__label": "parsed.label"}
    )
    return grouped[["slug", "task", "mode", "grading.notes", "parsed.label", "count"]]


def summarise_results(results_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for slug_dir in _iter_slug_directories(results_root):
        slug = slug_dir.name if slug_dir != results_root else results_root.name
        for task, loader in TASK_LOADERS.items():
            jsonl_path = slug_dir / f"{task}.jsonl"
            if not jsonl_path.exists():
                continue
            df = loader(jsonl_path)
            summary_frame = _summarise_dataframe(slug, task, df)
            if not summary_frame.empty:
                frames.append(summary_frame)
    if not frames:
        return pd.DataFrame(
            columns=["slug", "task", "mode", "grading.notes", "parsed.label", "count"]
        )
    summary = pd.concat(frames, ignore_index=True)
    return summary.sort_values(["slug", "task", "mode", "grading.notes", "parsed.label"]).reset_index(
        drop=True
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate grading.notes counts per task/mode for introspection results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory containing per-model slug folders with task JSONL files.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv"),
        default="table",
        help="Output format for the summary table.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = summarise_results(args.results_root)
    if summary.empty:
        print("No grading.notes entries found under", args.results_root, file=sys.stderr)
        return 1

    if args.format == "csv":
        print(summary.to_csv(index=False))
    else:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
