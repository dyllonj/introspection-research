from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from introspect.src.eval_A_injected_report import TaskAConfig
from introspect.src.eval_A_injected_report import run as run_task_a
from introspect.src.eval_B_thoughts_vs_text import TaskBConfig
from introspect.src.eval_B_thoughts_vs_text import run as run_task_b
from introspect.src.eval_C_prefill_intent import TaskCConfig
from introspect.src.eval_C_prefill_intent import run as run_task_c
from introspect.src.eval_common import LoadedAdapter
from introspect.src.inject import (
    DEFAULT_STOP_SEQUENCES,
    InjectionResult,
    describe_injection_spec,
)
from introspect.src.vectors import ConceptWordSet
from introspect.tests.utils import make_toy_adapter


def _fake_vector(hidden_size: int) -> torch.Tensor:
    vec = torch.arange(1, hidden_size + 1, dtype=torch.float32)
    return vec / torch.linalg.vector_norm(vec)


EXPECTED_GENERATION = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 64,
    "do_sample": False,
    "stop_sequences": list(DEFAULT_STOP_SEQUENCES),
    "allowed_formats": [],
}


def _render_stub_completion(enable_injection: bool, prompt: str) -> str:
    if "THOUGHT:" in prompt:
        word = "alpha" if enable_injection else "baseline"
        return f"Assistant: THOUGHT: {word}"
    if "REPEAT:" in prompt:
        lines = [line for line in prompt.splitlines() if line and not line.startswith("Human:")]
        sentence = lines[1].strip() if len(lines) > 1 else ""
        return f"Assistant: REPEAT: {sentence}"
    if "CHOICE:" in prompt:
        return "Assistant: CHOICE: 1"
    if "INTENT:" in prompt:
        return "Assistant: INTENT: YES" if enable_injection else "Assistant: INTENT: NO"
    return "INJECTION: alpha" if enable_injection else "NO_INJECTION"


def _patch_module_dependencies(
    monkeypatch,
    module_name: str,
    loaded: LoadedAdapter,
    word_set: ConceptWordSet,
) -> None:
    base = f"introspect.src.{module_name}"
    monkeypatch.setattr(f"{base}.load_adapter_from_registry", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(
        f"{base}.ensure_vector",
        lambda **kwargs: _fake_vector(loaded.adapter.hidden_size),
    )
    monkeypatch.setattr(f"{base}.load_words", lambda *_args, **_kwargs: word_set)
    monkeypatch.setattr(
        f"{base}.token_positions_for_substring",
        lambda adapter, text, substring, occurrence=0: [0],
    )

    def _inject_stub(adapter, prompt, spec, enable_injection=True, **_ignored):
        del adapter
        return InjectionResult(
            text=_render_stub_completion(enable_injection, prompt),
            generation=dict(EXPECTED_GENERATION),
            injection_spec=describe_injection_spec(spec),
        )

    monkeypatch.setattr(f"{base}.inject_once", _inject_stub)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _assert_schema(records: list[dict[str, object]], required: set[str]) -> None:
    assert records, "Expected at least one trial record"
    for record in records:
        assert required.issubset(record)


def _assert_metadata(
    records: list[dict[str, object]],
    *,
    expected_layer: int,
    expected_alpha: float,
    expected_positions: list[int],
    vector_dim: int,
    expected_seed: int,
) -> None:
    for record in records:
        assert record["seed"] == expected_seed
        assert record["generation"] == EXPECTED_GENERATION
        spec = record["injection_spec"]
        assert spec["layer_idx"] == expected_layer
        assert spec["alpha"] == expected_alpha
        assert spec["token_positions"] == expected_positions
        assert spec["apply_on_input"] is False
        assert spec["vector_dim"] == vector_dim
        assert math.isclose(spec["vector_norm"], 1.0, rel_tol=1e-6)


def test_task_runs_smoke(tmp_path: Path, monkeypatch) -> None:
    adapter = make_toy_adapter(hidden_size=8, num_layers=2)
    loaded = LoadedAdapter(
        adapter=adapter,
        adapter_name="ToyAdapter",
        dtype=torch.float32,
        device_map="cpu",
    )
    word_set = ConceptWordSet(targets=["alpha"], baselines=[f"base{i}" for i in range(1, 12)])

    for module in ("eval_A_injected_report", "eval_B_thoughts_vs_text", "eval_C_prefill_intent"):
        _patch_module_dependencies(monkeypatch, module, loaded, word_set)

    task_a_path = tmp_path / "task_a.jsonl"
    config_a = TaskAConfig(
        model="toy",
        adapter="ToyAdapter",
        dtype=None,
        device_map=None,
        layers=[0],
        alphas=[1.0],
        n_concepts=1,
        seed=0,
        words_file=Path("unused"),
        cache_dir=tmp_path,
        results_path=task_a_path,
        overwrite=True,
        baseline_sample=None,
        prompt_template="Think about the concept \"{word}\".",
        deterministic=True,
    )
    run_task_a(config_a)
    records_a = _read_jsonl(task_a_path)
    _assert_schema(
        records_a,
        {"task", "layer", "alpha", "word", "injected", "vector_kind", "generation", "injection_spec", "seed"},
    )
    _assert_metadata(
        records_a,
        expected_layer=0,
        expected_alpha=config_a.alphas[0],
        expected_positions=[0],
        vector_dim=adapter.hidden_size,
        expected_seed=config_a.seed,
    )

    task_b_path = tmp_path / "task_b.jsonl"
    config_b = TaskBConfig(
        model="toy",
        adapter="ToyAdapter",
        dtype=None,
        device_map=None,
        layers=[0],
        alpha=1.0,
        n_trials=1,
        seed=0,
        words_file=Path("unused"),
        cache_dir=tmp_path,
        results_path=task_b_path,
        overwrite=True,
        baseline_sample=None,
        prompt_template="Think about the concept \"{word}\".",
        deterministic=True,
    )
    run_task_b(config_b)
    records_b = _read_jsonl(task_b_path)
    _assert_schema(
        records_b,
        {"task", "layer", "word", "mode", "condition", "injected", "generation", "injection_spec", "seed"},
    )
    _assert_metadata(
        records_b,
        expected_layer=0,
        expected_alpha=config_b.alpha,
        expected_positions=[0],
        vector_dim=adapter.hidden_size,
        expected_seed=config_b.seed,
    )

    task_c_path = tmp_path / "task_c.jsonl"
    config_c = TaskCConfig(
        model="toy",
        adapter="ToyAdapter",
        dtype=None,
        device_map=None,
        layers=[0],
        alpha=1.0,
        n_trials=1,
        seed=0,
        words_file=Path("unused"),
        cache_dir=tmp_path,
        results_path=task_c_path,
        overwrite=True,
        baseline_sample=None,
        prompt_template="Think about the concept \"{word}\".",
        deterministic=True,
    )
    run_task_c(config_c)
    records_c = _read_jsonl(task_c_path)
    _assert_schema(
        records_c,
        {"task", "layer", "word", "condition", "injected", "generation", "injection_spec", "seed"},
    )
    _assert_metadata(
        records_c,
        expected_layer=0,
        expected_alpha=config_c.alpha,
        expected_positions=[0],
        vector_dim=adapter.hidden_size,
        expected_seed=config_c.seed,
    )
