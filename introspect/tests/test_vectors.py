from __future__ import annotations

from pathlib import Path

import pytest
import torch

from introspect.src import vectors


@pytest.fixture()
def patched_activations(monkeypatch: pytest.MonkeyPatch) -> None:
    activations = {
        "target": torch.tensor([3.0, 0.0, 0.0], dtype=torch.float32),
        "baseline_one": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "baseline_two": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        "baseline_three": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    }

    def fake_activation(adapter, layer_idx: int, prompt: str) -> torch.Tensor:
        del adapter, layer_idx
        try:
            word = prompt.split("\"")[1]
        except IndexError as exc:  # pragma: no cover - format guard
            raise AssertionError(f"Unexpected prompt format: {prompt}") from exc
        key = "target" if word == "target" else f"baseline_{word}"
        return activations[key]

    monkeypatch.setattr(vectors, "_activation_for_prompt", fake_activation)


def test_build_concept_vector_normalizes(patched_activations: None) -> None:
    del patched_activations
    adapter = object()
    vector, sampled = vectors.build_concept_vector(
        adapter,
        0,
        target_word="target",
        baseline_words=["one", "two", "three"],
        return_sampled_baselines=True,
        prompt_template="Think about \"{word}\".",
    )

    norm = torch.linalg.vector_norm(vector)
    assert torch.isclose(norm, torch.tensor(1.0), atol=1e-6)
    assert sampled == ["one", "two", "three"]
    assert torch.isclose(vector.mean(), torch.tensor(0.0), atol=1e-6)


def test_vector_cache_roundtrip(tmp_path: Path) -> None:
    vector = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
    path = vectors.cache_path("toy-model", 3, "target", cache_dir=tmp_path)
    vectors.save_vector(vector, path, metadata={"layer": 3})

    loaded, metadata = vectors.load_vector(path)
    assert torch.allclose(loaded, vector)
    assert metadata is not None and metadata["layer"] == 3

    with pytest.raises(ValueError):
        vectors.load_vector(path.with_suffix(".json"))
