"""Visualisation utilities for introspection experiment results.

This module provides a small collection of helpers that transform the JSONL
artifacts produced by the evaluation scripts into publication friendly plots.
The implementation favours ``pandas``/``matplotlib`` primitives so that it can
run in headless environments (e.g. CI) without requiring an interactive
backend.

Functions are intentionally composable: data loading, aggregation and plotting
are split into separate helpers to simplify testing and to let callers re-use
the metrics for custom reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskAMetrics:
    """Aggregated Task A metrics (TPR/FPR/Net) organised as pivot tables."""

    tpr: pd.DataFrame
    fpr: pd.DataFrame
    net: pd.DataFrame


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        LOGGER.debug("JSONL file %s not found; skipping", path)
        return []

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON line in %s", path)
    return records


def _normalise_dataframe(records: Sequence[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep=".")


def model_id_to_slug(model_id: str) -> str:
    """Return a filesystem friendly identifier for ``model_id``."""

    return model_id.replace("/", "__")


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_task_a_dataframe(path: Path) -> pd.DataFrame:
    return _normalise_dataframe(_load_jsonl(path))


def compute_task_a_metrics(df: pd.DataFrame) -> TaskAMetrics | None:
    if df.empty:
        return None

    if "vector_kind" not in df.columns:
        LOGGER.warning("Task A dataframe missing 'vector_kind' column")
        return None

    for column in ("grading.tp", "grading.fp"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if "grading.matched" in df.columns:
        df["grading.matched"] = (
            df["grading.matched"].astype(float, errors="ignore")
            if df["grading.matched"].dtype != bool
            else df["grading.matched"].astype(float)
        )

    key_cols = ["layer", "alpha"]
    for key in key_cols:
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce")

    targets = df[df["vector_kind"] == "target"].copy()
    if targets.empty:
        LOGGER.debug("Task A dataframe does not contain target rows")
        return None

    if "grading.matched" in targets.columns:
        matched = pd.to_numeric(targets["grading.matched"], errors="coerce").fillna(0.0)
    else:
        matched = pd.Series(0.0, index=targets.index)
    targets["success"] = matched.astype(float)
    tpr = (
        targets.groupby(key_cols)["success"].mean().reset_index().pivot(
            index="layer", columns="alpha", values="success"
        )
    )

    non_targets = df[df["vector_kind"] != "target"].copy()
    if non_targets.empty:
        fpr = pd.DataFrame(0.0, index=tpr.index, columns=tpr.columns)
    else:
        if "grading.fp" in non_targets.columns:
            false_pos = (
                pd.to_numeric(non_targets["grading.fp"], errors="coerce").fillna(0.0)
            )
        else:
            false_pos = pd.Series(0.0, index=non_targets.index)
        non_targets["false_positive"] = false_pos.astype(float)
        fpr = (
            non_targets.groupby(key_cols)["false_positive"]
            .mean()
            .reset_index()
            .pivot(index="layer", columns="alpha", values="false_positive")
        )
        fpr = fpr.reindex_like(tpr).fillna(0.0)

    net = tpr - fpr
    return TaskAMetrics(tpr=tpr.sort_index(), fpr=fpr.sort_index(), net=net.sort_index())


def plot_task_a_heatmaps(metrics: TaskAMetrics, output_dir: Path, model_id: str) -> list[Path]:
    if metrics is None:
        return []

    output = _ensure_output_dir(output_dir)
    saved: list[Path] = []

    for attr, title in (
        ("tpr", "Task A True Positive Rate"),
        ("fpr", "Task A False Positive Rate"),
        ("net", "Task A Net Score (TPR - FPR)"),
    ):
        data = getattr(metrics, attr)
        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(data.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels([f"{col:.2f}" for col in data.columns], rotation=45)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels([str(int(idx)) for idx in data.index])
        ax.set_xlabel("α")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title}\n{model_id}")
        fig.colorbar(cax, ax=ax)

        filename = output / f"task_A_{attr}.png"
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        saved.append(filename)
        LOGGER.info("Saved Task A %s heatmap to %s", attr, filename)

    return saved


def load_task_b_dataframe(path: Path) -> pd.DataFrame:
    return _normalise_dataframe(_load_jsonl(path))


def compute_task_b_success(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["grading.matched"] = df.get("grading.matched", False).astype(float)
    grouped = (
        df.groupby(["mode", "condition"])["grading.matched"]
        .mean()
        .reset_index(name="success_rate")
    )
    return grouped


def plot_task_b_success_bars(df: pd.DataFrame, output_dir: Path, model_id: str) -> Path | None:
    if df.empty:
        return None

    pivot = df.pivot(index="mode", columns="condition", values="success_rate").fillna(0.0)
    modes = list(pivot.index)
    conditions = ["control", "injected"]
    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, condition in enumerate(conditions):
        values = pivot.get(condition)
        if values is None:
            values = np.zeros(len(modes))
        ax.bar(x + (idx - 0.5) * width, values, width, label=condition.title())

    ax.set_xticks(x)
    ax.set_xticklabels([mode.title() for mode in modes])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Strict success rate")
    ax.set_title(f"Task B Success Rates\n{model_id}")
    ax.legend()

    output = _ensure_output_dir(output_dir)
    path = output / "task_B_success_rates.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved Task B success bar chart to %s", path)
    return path


def load_task_c_dataframe(path: Path) -> pd.DataFrame:
    return _normalise_dataframe(_load_jsonl(path))


def compute_task_c_delta(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["intent_yes"] = df.get("parsed.label", "") == "intent_yes"
    grouped = (
        df.groupby(["layer", "condition"])["intent_yes"]
        .mean()
        .reset_index()
        .pivot(index="layer", columns="condition", values="intent_yes")
        .fillna(0.0)
    )
    grouped["delta"] = grouped.get("injected", 0.0) - grouped.get("control", 0.0)
    return grouped.reset_index()


def plot_task_c_delta_bars(df: pd.DataFrame, output_dir: Path, model_id: str) -> Path | None:
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["layer"].astype(int).astype(str), df["delta"], color="#4f8bc9")
    ax.set_ylabel("Δ intent (Injected − Control)")
    ax.set_xlabel("Layer")
    ax.set_title(f"Task C Intent Shift\n{model_id}")
    ax.axhline(0.0, color="black", linewidth=0.8)

    output = _ensure_output_dir(output_dir)
    path = output / "task_C_delta_intent.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved Task C delta-intent bar chart to %s", path)
    return path


def load_task_d_dataframe(path: Path) -> pd.DataFrame:
    return _normalise_dataframe(_load_jsonl(path))


def compute_task_d_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rows: list[dict[str, float]] = []
    for _, record in df.iterrows():
        layers = record.get("layers", [])
        deltas = record.get("delta_curve", [])
        if not isinstance(layers, Sequence) or not isinstance(deltas, Sequence):
            continue
        for layer, delta in zip(layers, deltas, strict=False):
            rows.append({"layer": int(layer), "delta": float(delta)})

    if not rows:
        return pd.DataFrame()

    curve = pd.DataFrame(rows)
    aggregated = (
        curve.groupby("layer")["delta"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values("layer")
    )
    aggregated["std"] = aggregated["std"].fillna(0.0)
    return aggregated


def plot_task_d_curve(df: pd.DataFrame, output_dir: Path, model_id: str) -> Path | None:
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["layer"], df["mean"], label="Δ activation", color="#d17a22")
    if "std" in df.columns:
        lower = df["mean"] - df["std"]
        upper = df["mean"] + df["std"]
        ax.fill_between(df["layer"], lower, upper, color="#d17a22", alpha=0.2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δ activation")
    ax.set_title(f"Task D Δ activation curves\n{model_id}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()

    output = _ensure_output_dir(output_dir)
    path = output / "task_D_delta_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved Task D curve plot to %s", path)
    return path


def generate_all_plots(
    model_id: str,
    *,
    results_root: Path | None = None,
    plots_subdir: str = "plots",
) -> list[Path]:
    results_root = Path(results_root or Path("results"))
    model_dir = results_root / model_id_to_slug(model_id)
    plot_dir = model_dir / plots_subdir

    saved: list[Path] = []

    task_a_path = model_dir / "task_A.jsonl"
    task_a_df = load_task_a_dataframe(task_a_path)
    metrics = compute_task_a_metrics(task_a_df)
    if metrics is not None:
        saved.extend(plot_task_a_heatmaps(metrics, plot_dir, model_id))

    task_b_path = model_dir / "task_B.jsonl"
    task_b_df = load_task_b_dataframe(task_b_path)
    task_b_success = compute_task_b_success(task_b_df)
    path = plot_task_b_success_bars(task_b_success, plot_dir, model_id)
    if path is not None:
        saved.append(path)

    task_c_path = model_dir / "task_C.jsonl"
    task_c_df = load_task_c_dataframe(task_c_path)
    delta_df = compute_task_c_delta(task_c_df)
    path = plot_task_c_delta_bars(delta_df, plot_dir, model_id)
    if path is not None:
        saved.append(path)

    task_d_path = model_dir / "task_D.jsonl"
    task_d_df = load_task_d_dataframe(task_d_path)
    curve_df = compute_task_d_curve(task_d_df)
    path = plot_task_d_curve(curve_df, plot_dir, model_id)
    if path is not None:
        saved.append(path)

    if not saved:
        LOGGER.warning("No plots were generated for model %s", model_id)

    return saved


__all__ = [
    "TaskAMetrics",
    "compute_task_a_metrics",
    "compute_task_b_success",
    "compute_task_c_delta",
    "compute_task_d_curve",
    "generate_all_plots",
    "load_task_a_dataframe",
    "load_task_b_dataframe",
    "load_task_c_dataframe",
    "load_task_d_dataframe",
    "model_id_to_slug",
    "plot_task_a_heatmaps",
    "plot_task_b_success_bars",
    "plot_task_c_delta_bars",
    "plot_task_d_curve",
]
