"""Unit tests for concept vector extraction utilities."""

from __future__ import annotations

import random
from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from introspect.src.vectors import (
    METADATA_VERSION,
    build_concept_vector,
    load_vector,
    save_vector,
)


class SimpleTokenizer:
    """Very small tokenizer mapping entire prompts to integer ids."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def __call__(self, text: str, /, **_: object) -> dict[str, torch.Tensor]:
        token_id = self._vocab.setdefault(text, len(self._vocab))
        return {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }


class DummyLayer(torch.nn.Module):
    """Minimal transformer block used for testing hooks."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.eye_(self.linear.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.tanh(self.linear(hidden_states) + hidden_states)


class DummyModel(torch.nn.Module):
    """Tiny model exposing transformer-like layers."""

    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int = 128) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            weight = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            weight = weight.view(vocab_size, hidden_size)
            self.embedding.weight.copy_(weight / (hidden_size * vocab_size))
        self.layers = torch.nn.ModuleList(
            DummyLayer(hidden_size) for _ in range(num_layers)
        )

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class DummyAdapter:
    """Adapter satisfying :class:`BaseModelAdapter` for tests."""

    name = "dummy"
    hidden_size = 8
    num_layers = 2

    def __init__(self, model: DummyModel, tokenizer: SimpleTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype,
        device_map: object,
        *,
        seed: int | None = None,
    ) -> "DummyAdapter":
        del model_id, dtype, device_map
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        model = DummyModel(hidden_size=cls.hidden_size, num_layers=cls.num_layers)
        return cls(model=model, tokenizer=SimpleTokenizer())

    def layer_module(self, layer_idx: int) -> torch.nn.Module:
        return self.model.layers[layer_idx]

    def register_residual_hook(self, layer_idx: int, hook_fn):  # type: ignore[override]
        return self.layer_module(layer_idx).register_forward_hook(hook_fn)

    def tokens_for_spans(self, text: str, span_slices):  # type: ignore[override]
        del text, span_slices
        return []

    def generate(self, prompt: str, /, **_: object) -> str:  # type: ignore[override]
        return prompt


@pytest.fixture()
def toy_adapter() -> DummyAdapter:
    return DummyAdapter.load("dummy", dtype=torch.float32, device_map="cpu", seed=0)


def test_build_concept_vector_normalized(toy_adapter: DummyAdapter) -> None:
    vector = build_concept_vector(
        toy_adapter,
        layer_idx=1,
        target_word="alpha",
        baseline_words=["beta", "gamma", "delta"],
        prompt_template="{word}",
    )
    norm = torch.linalg.vector_norm(vector).item()
    assert pytest.approx(1.0, rel=1e-5) == norm


def test_build_concept_vector_sampling(toy_adapter: DummyAdapter) -> None:
    vector, sampled = build_concept_vector(
        toy_adapter,
        layer_idx=0,
        target_word="alpha",
        baseline_words=["beta", "gamma", "delta"],
        prompt_template="{word}",
        baseline_sample_size=2,
        rng=random.Random(42),
        return_sampled_baselines=True,
    )
    assert len(sampled) == 2
    assert torch.isfinite(vector).all()


def test_cache_round_trip(tmp_path: Path, toy_adapter: DummyAdapter) -> None:
    vector = build_concept_vector(
        toy_adapter,
        layer_idx=1,
        target_word="alpha",
        baseline_words=["beta", "gamma", "delta"],
        prompt_template="{word}",
    )
    out_path = tmp_path / "alpha.npy"
    metadata = {"model_id": "dummy", "layer": 1, "word": "alpha"}
    save_vector(vector, out_path, metadata=metadata)

    loaded, meta = load_vector(out_path)
    assert torch.allclose(loaded, vector)
    assert meta is not None
    assert meta["model_id"] == "dummy"
    assert meta["shape"] == list(vector.shape)
    assert meta["version"] == METADATA_VERSION
