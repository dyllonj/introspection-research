"""Post-process introspection task outputs and regenerate plots.

Usage:
    python -m introspect.src.tools.postprocess --results-root results \
        --drop-notes invalid invalid_format

The command walks over each model slug directory under ``--results-root``,
loads the task JSONL files, filters out rows whose ``grading.notes`` column
matches any of the provided ``--drop-notes`` values, and writes cleaned tables
to ``postprocessed`` sub-directories (CSV and JSON formats). The filtered JSON
files are then used to recompute the aggregate tables and to invoke
``generate_all_plots`` so that visualisations reflect the filtered inputs.

All operations execute on the CPU; no model weights are loaded.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from introspect.src.plotting import (
    compute_task_a_metrics,
    compute_task_b_success,
    compute_task_c_delta,
    compute_task_d_curve,
    generate_all_plots,
    load_task_a_dataframe,
    load_task_b_dataframe,
    load_task_c_dataframe,
    load_task_d_dataframe,
    model_id_to_slug,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    filename: str


TASKS: tuple[TaskDefinition, ...] = (
    TaskDefinition("task_A", "task_A.jsonl"),
    TaskDefinition("task_B", "task_B.jsonl"),
    TaskDefinition("task_C", "task_C.jsonl"),
    TaskDefinition("task_D", "task_D.jsonl"),
)


def _iter_slug_directories(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []

    slug_dirs = [
        path
        for path in sorted(results_root.iterdir())
        if path.is_dir() and path.name != "postprocessed"
    ]

    if slug_dirs:
        return slug_dirs

    return [results_root]


def _load_dataframe(task: str, path: Path) -> pd.DataFrame:
    loader_map = {
        "task_A": load_task_a_dataframe,
        "task_B": load_task_b_dataframe,
        "task_C": load_task_c_dataframe,
        "task_D": load_task_d_dataframe,
    }
    loader = loader_map[task]
    return loader(path)


def _drop_by_notes(df: pd.DataFrame, notes_to_drop: Sequence[str]) -> pd.DataFrame:
    if df.empty or not notes_to_drop or "grading.notes" not in df.columns:
        return df

    mask = df["grading.notes"].astype(str).isin(set(notes_to_drop))
    if mask.any():
        LOGGER.debug("Dropping %d rows based on grading.notes", int(mask.sum()))
    return df.loc[~mask].reset_index(drop=True)


def _write_dataframe(
    df: pd.DataFrame,
    base_path: Path,
    *,
    index: bool = False,
    json_orient: str = "records",
) -> list[Path]:
    if df.empty:
        return []

    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")
    df.to_csv(csv_path, index=index)
    df.to_json(json_path, orient=json_orient)
    return [csv_path, json_path]


def _write_jsonl(df: pd.DataFrame, path: Path) -> Path | None:
    if df.empty:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def _extract_model_id(df: pd.DataFrame) -> str | None:
    if df.empty or "model_id" not in df.columns:
        return None

    series = df["model_id"].dropna()
    if series.empty:
        return None

    value = series.iloc[0]
    return str(value)


def _compute_task_outputs(
    task: str,
    df: pd.DataFrame,
    tables_dir: Path,
) -> list[Path]:
    tables: list[Path] = []

    if task == "task_A":
        metrics = compute_task_a_metrics(df)
        if metrics is None:
            return tables
        for name in ("tpr", "fpr", "net"):
            data = getattr(metrics, name)
            tables.extend(
                _write_dataframe(
                    data,
                    tables_dir / f"task_A_{name}",
                    index=True,
                    json_orient="split",
                )
            )
        return tables

    if task == "task_B":
        result = compute_task_b_success(df)
        tables.extend(
            _write_dataframe(
                result,
                tables_dir / "task_B_success",
                index=False,
                json_orient="records",
            )
        )
        return tables

    if task == "task_C":
        result = compute_task_c_delta(df)
        tables.extend(
            _write_dataframe(
                result,
                tables_dir / "task_C_delta",
                index=False,
                json_orient="records",
            )
        )
        return tables

    if task == "task_D":
        result = compute_task_d_curve(df)
        tables.extend(
            _write_dataframe(
                result,
                tables_dir / "task_D_curve",
                index=False,
                json_orient="records",
            )
        )
        return tables

    raise ValueError(f"Unsupported task: {task}")


def process_slug(
    slug_dir: Path,
    *,
    results_root: Path,
    notes_to_drop: Sequence[str],
) -> tuple[list[Path], list[Path]]:
    LOGGER.info("Processing %s", slug_dir)

    post_dir = slug_dir / "postprocessed"
    raw_tables_dir = post_dir / "tables"
    metrics_tables_dir = post_dir / "metrics"
    json_root = post_dir / "jsonl_root"
    model_dir = json_root / slug_dir.name

    produced_tables: list[Path] = []
    model_id: str | None = None

    for task_def in TASKS:
        task_path = slug_dir / task_def.filename
        if not task_path.exists():
            LOGGER.info(
                "Skipping %s for %s – %s not found", task_def.name, slug_dir.name, task_path.name
            )
            continue

        df = _load_dataframe(task_def.name, task_path)
        filtered = _drop_by_notes(df, notes_to_drop)
        if model_id is None:
            model_id = _extract_model_id(filtered) or _extract_model_id(df)

        produced_tables.extend(
            _write_dataframe(
                filtered,
                raw_tables_dir / f"{task_def.name}_filtered",
                index=False,
                json_orient="records",
            )
        )

        jsonl_path = _write_jsonl(filtered, model_dir / task_def.filename)
        if jsonl_path is not None:
            produced_tables.append(jsonl_path)

        produced_tables.extend(
            _compute_task_outputs(task_def.name, filtered, metrics_tables_dir)
        )

    if model_id is None:
        inferred = slug_dir.name.replace("__", "/")
        model_id = inferred if model_id_to_slug(inferred) == slug_dir.name else slug_dir.name

    if not model_dir.exists():
        LOGGER.warning("No filtered task files produced for %s; skipping plots", slug_dir)
        return produced_tables, []

    plots = generate_all_plots(model_id, results_root=json_root)
    if plots:
        LOGGER.info("Generated plots for %s: %s", slug_dir.name, ", ".join(map(str, plots)))
    else:
        LOGGER.info("No plots generated for %s", slug_dir.name)

    return produced_tables, plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter introspection task results by grading notes and regenerate plots.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory containing per-model slug sub-directories.",
    )
    parser.add_argument(
        "--drop-notes",
        nargs="*",
        default=(),
        help="List of grading.notes values to exclude (e.g. invalid invalid_format).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results_root = args.results_root
    if not results_root.exists():
        parser.error(f"Results root {results_root} does not exist")

    slug_dirs = _iter_slug_directories(results_root)
    if not slug_dirs:
        LOGGER.info("No directories to process under %s", results_root)
        return 0

    for slug_dir in slug_dirs:
        tables, plots = process_slug(
            slug_dir,
            results_root=results_root,
            notes_to_drop=args.drop_notes,
        )
        if tables:
            LOGGER.info(
                "Produced %d table artifact(s) for %s", len(tables), slug_dir.name
            )
        else:
            LOGGER.info("No table artifacts produced for %s", slug_dir.name)
        if plots:
            LOGGER.info(
                "Generated %d plot(s) for %s", len(plots), slug_dir.name
            )
        else:
            LOGGER.info("No plots generated for %s", slug_dir.name)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
