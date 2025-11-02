from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from introspect.src.tools import postprocess


@pytest.fixture()
def results_root(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    slug_dir = root / "meta-llama__Test-Model"
    slug_dir.mkdir(parents=True)
    for task_def in postprocess.TASKS:
        (slug_dir / task_def.filename).write_text("{}\n", encoding="utf-8")
    return root


def test_process_slug_filters_invalid_notes_and_generates_plots(
    results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slug_dir = postprocess._iter_slug_directories(results_root)[0]

    model_id = "meta-llama/Test-Model"

    task_frames = {
        "task_A": pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "vector_kind": "target",
                    "grading.notes": "ok",
                    "grading.matched": 1,
                    "grading.fp": 0,
                    "layer": 1,
                    "alpha": 1.0,
                },
                {
                    "model_id": model_id,
                    "vector_kind": "target",
                    "grading.notes": "invalid",
                    "grading.matched": 0,
                    "grading.fp": 0,
                    "layer": 1,
                    "alpha": 2.0,
                },
                {
                    "model_id": model_id,
                    "vector_kind": "random",
                    "grading.notes": "ok",
                    "grading.fp": 1,
                    "layer": 1,
                    "alpha": 1.0,
                },
            ]
        ),
        "task_B": pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "mode": "baseline",
                    "condition": "control",
                    "grading.matched": 1,
                    "grading.notes": "ok",
                },
                {
                    "model_id": model_id,
                    "mode": "baseline",
                    "condition": "injected",
                    "grading.matched": 0,
                    "grading.notes": "invalid",
                },
            ]
        ),
        "task_C": pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "layer": 0,
                    "condition": "control",
                    "parsed.label": "intent_no",
                    "grading.notes": "ok",
                },
                {
                    "model_id": model_id,
                    "layer": 0,
                    "condition": "injected",
                    "parsed.label": "intent_yes",
                    "grading.notes": "invalid_format",
                },
            ]
        ),
        "task_D": pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "layers": [0, 1],
                    "delta_curve": [0.1, 0.2],
                    "grading.notes": "ok",
                },
                {
                    "model_id": model_id,
                    "layers": [0, 1],
                    "delta_curve": [0.3, 0.4],
                    "grading.notes": "invalid",
                },
            ]
        ),
    }

    def fake_loader(task: str, path: Path) -> pd.DataFrame:
        return task_frames[task].copy()

    monkeypatch.setattr(postprocess, "_load_dataframe", fake_loader)

    saved_figures: list[Path] = []

    from matplotlib.figure import Figure

    def fake_savefig(self, fname, *args, **kwargs):  # type: ignore[override]
        saved_figures.append(Path(fname))

    monkeypatch.setattr(Figure, "savefig", fake_savefig)

    tables, plots = postprocess.process_slug(
        slug_dir,
        results_root=results_root,
        notes_to_drop=["invalid", "invalid_format"],
    )

    assert tables, "Expected table artifacts to be produced"
    assert plots, "Expected plots to be generated"
    assert saved_figures, "Expected matplotlib savefig to be invoked"

    filtered_csv = slug_dir / "postprocessed" / "tables" / "task_B_filtered.csv"
    filtered_df = pd.read_csv(filtered_csv)
    assert list(filtered_df["grading.notes"]) == ["ok"]

    jsonl_path = slug_dir / "postprocessed" / "jsonl_root" / slug_dir.name / "task_B.jsonl"
    jsonl_content = jsonl_path.read_text(encoding="utf-8")
    assert "invalid" not in jsonl_content

    assert any("task_A_tpr.csv" in str(path) for path in tables)
    assert any(path.name.startswith("task_A") for path in plots)
