from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from introspect.src.analysis import data_quality


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(rec) for rec in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_summarise_results_counts(tmp_path: Path) -> None:
    slug_dir = tmp_path / "model__slug"
    slug_dir.mkdir()

    task_a_records = [
        {"vector_kind": "target", "grading": {"notes": "ok"}},
        {"vector_kind": "target", "grading": {"notes": "ok"}},
        {"vector_kind": "baseline", "grading": {"notes": "skip"}},
    ]
    _write_jsonl(slug_dir / "task_A.jsonl", task_a_records)

    task_b_records = [
        {
            "mode": "no_injection",
            "grading": {"notes": "pass"},
            "parsed": {"label": "YES"},
        },
        {
            "mode": "injection",
            "grading": {"notes": "fail"},
            "parsed": {"label": "NO"},
        },
        {
            "mode": "injection",
            "grading": {"notes": "fail"},
            "parsed": {"label": "NO"},
        },
    ]
    _write_jsonl(slug_dir / "task_B.jsonl", task_b_records)

    summary = data_quality.summarise_results(tmp_path)

    expected = pd.DataFrame(
        [
            {
                "slug": "model__slug",
                "task": "task_A",
                "mode": "target",
                "grading.notes": "ok",
                "parsed.label": "<missing>",
                "count": 2,
            },
            {
                "slug": "model__slug",
                "task": "task_A",
                "mode": "baseline",
                "grading.notes": "skip",
                "parsed.label": "<missing>",
                "count": 1,
            },
            {
                "slug": "model__slug",
                "task": "task_B",
                "mode": "injection",
                "grading.notes": "fail",
                "parsed.label": "NO",
                "count": 2,
            },
            {
                "slug": "model__slug",
                "task": "task_B",
                "mode": "no_injection",
                "grading.notes": "pass",
                "parsed.label": "YES",
                "count": 1,
            },
        ]
    ).sort_values(["slug", "task", "mode", "grading.notes", "parsed.label"]).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(summary.reset_index(drop=True), expected)


def test_summarise_results_handles_missing_files(tmp_path: Path) -> None:
    summary = data_quality.summarise_results(tmp_path)
    assert summary.empty
