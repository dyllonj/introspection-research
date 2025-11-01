"""Unit tests for :mod:`introspect.src.prompts`."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from introspect.src import prompts


def test_select_corpus_sentence_index():
    sentence = prompts.select_corpus_sentence(index=3)
    assert sentence == "His favorite ice cream flavor was mint chocolate chip."


def test_select_corpus_sentence_seed():
    sentence = prompts.select_corpus_sentence(seed=42)
    # Random selection should be deterministic for the same seed.
    assert sentence == prompts.select_corpus_sentence(seed=42)


def test_render_task_a_detection_prompt_contains_required_markers():
    prompt = prompts.render_task_a_detection_prompt()
    assert "NO_INJECTION" in prompt
    assert "INJECTION: <word>" in prompt
    # Guard against accidental concrete word leakage pattern "INJECTION:" followed by alpha characters.
    assert not re.search(r"INJECTION: [A-Za-z]+", prompt.replace("INJECTION: <word>", ""))


def test_task_b_open_ended_prompt_blocks_word_leakage():
    sentence = "The aquarium is quiet at night."
    with pytest.raises(ValueError):
        prompts.render_task_b_open_ended_prompt(
            sentence=sentence,
            target_word="aquarium",
        )

    prompt = prompts.render_task_b_open_ended_prompt(
        sentence=sentence,
        target_word="aquarium",
        allow_target=True,
    )
    assert "THOUGHT:" in prompt


def test_task_b_repetition_prompt_renders_sentence():
    sentence = "The book fell open to page 217."
    prompt = prompts.render_task_b_repetition_prompt(
        sentence=sentence,
        target_word="book",
        allow_target=True,
    )
    assert sentence in prompt
    assert "REPEAT:" in prompt


def test_task_b_multiple_choice_prompt_mapping():
    sentence = "The coffee shop was bustling with morning customers."
    mc_prompt = prompts.render_task_b_multiple_choice_prompt(
        sentence=sentence,
        options=[
            "rabbit",
            "dog",
            "cat",
            "horse",
            "piano",
            "window",
            "bottle",
            "garden",
            "lamp",
            "river",
        ],
        target_word="dog",
    )
    assert mc_prompt.option_map[mc_prompt.correct_option] == "dog"
    assert sentence in mc_prompt.prompt
    assert "CHOICE:" in mc_prompt.prompt


def test_task_c_prefill_dialog_enforces_single_word_occurrence():
    sentence = "The professor explained the theory with great enthusiasm."
    prompt = prompts.render_task_c_prefill_dialog(
        sentence=sentence,
        prefill_word="train",
        intent_query="Did that happen on purpose? Respond INTENT: YES or INTENT: NO only.",
    )
    assert prompt.count("train") == 1
    assert "INTENT:" in prompt

    with pytest.raises(ValueError):
        prompts.render_task_c_prefill_dialog(
            sentence=sentence,
            prefill_word="train",
            intent_query="Was that train intentional? Respond INTENT: YES or INTENT: NO only.",
        )

    with pytest.raises(ValueError):
        prompts.render_task_c_prefill_dialog(
            sentence=sentence,
            prefill_word="train",
            intent_query="   ",
        )


@pytest.mark.parametrize("variant", list(prompts.TaskDVariant))
def test_task_d_prompt_variants_have_exact_word_once(variant: prompts.TaskDVariant):
    sentence = "The museum exhibit featured ancient Egyptian artifacts."
    word = "artifact"
    prompt = prompts.render_task_d_prompt(
        sentence=sentence,
        word=word,
        variant=variant,
    )
    assert len(re.findall(rf"\b{word}\b", prompt)) == 1
    assert sentence in prompt


def test_task_d_prompt_invalid_variant():
    with pytest.raises(KeyError):
        prompts.render_task_d_prompt(
            sentence="Example",
            word="example",
            variant="missing",  # type: ignore[arg-type]
        )
