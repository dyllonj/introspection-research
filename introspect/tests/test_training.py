"""Lightweight tests for optional training helpers."""

from __future__ import annotations

import json
from pathlib import Path

from introspect.src.inject import resolve_injection_positions
from introspect.src.training.data_generation import PreferenceDataConfig, generate_task_a_preference_pairs
from introspect.src.training.split_concepts import split_concepts
from introspect.tests.utils import make_toy_adapter
from introspect.src.prompts import task_a_paper_messages
from introspect.src.generation import build_chat_prompt
from introspect.src.vectors import DEFAULT_WORDS_PATH


def test_resolve_injection_positions_prefix_and_suffix():
    adapter = make_toy_adapter()
    prompt, _ = build_chat_prompt(adapter.tokenizer, task_a_paper_messages())

    positions, suffix_start = resolve_injection_positions(adapter, prompt, mode="prefix")
    assert positions, "prefix mode should return at least one token position"
    assert suffix_start is None or suffix_start >= 1

    suffix_positions, suffix_start_b = resolve_injection_positions(adapter, prompt, mode="suffix")
    assert suffix_positions == []
    assert suffix_start_b is None or suffix_start_b >= 1


def test_data_generation_respects_holdout(tmp_path: Path):
    adapter = make_toy_adapter()
    config = PreferenceDataConfig(
        n_concepts=2,
        target_words=["bread", "ocean"],
        holdout_concepts=["bread"],
        layers=[0],
        alphas=[1.0],
        samples_per_concept=1,
        baseline_sample_size=2,
        injection_mode="prefix",
        words_file=Path(DEFAULT_WORDS_PATH),
    )

    samples = list(generate_task_a_preference_pairs(adapter, config, vector_cache_dir=tmp_path))
    assert samples, "should generate samples"
    assert all(s.concept_word != "bread" for s in samples if s.concept_word)

    # With one concept, expect two records (injection + control)
    assert len(samples) == 2


def test_split_concepts_deterministic(tmp_path: Path):
    words_file = Path(DEFAULT_WORDS_PATH)
    result_a = split_concepts(words_file, n_holdout=5, seed=123)
    result_b = split_concepts(words_file, n_holdout=5, seed=123)
    assert result_a == result_b

    output_path = tmp_path / "split.json"
    output_path.write_text(json.dumps(result_a), encoding="utf-8")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(loaded["train"]).isdisjoint(set(loaded["holdout"]))
